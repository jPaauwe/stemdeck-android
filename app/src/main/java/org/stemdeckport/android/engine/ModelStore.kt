package org.stemdeckport.android.engine

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException

/** One model file: where it comes from, pinned to a revision, and its SHA-256. */
data class ModelSpec(
    val fileName: String,
    val url: String,
    val sha256: String,
    val bytes: Long,
    /** For the stage line, e.g. "the beat model". */
    val label: String,
)

/** The models the app downloads on first use. None of them is in the APK. */
object Models {
    /** htdemucs_6s in demucs.cpp's ggml format (bit for bit Demucs v4.1.0's weights; CI checks this). */
    val DEMUCS = ModelSpec(
        fileName = "ggml-model-htdemucs-6s-f16.bin",
        url = "https://huggingface.co/datasets/Retrobear/demucs.cpp/resolve/" +
            "8f58ac0491bbea657275bcd3e38af1ab3a27bfc9/ggml-model-htdemucs-6s-f16.bin",
        sha256 = "09704f4ceae204e56e77d5eefd6ac71d7275be81fd507e6913371d59abcee856",
        bytes = 54_855_129,
        label = "the separation model",
    )

    /**
     * Beat This! final0 (CPJKU, MIT), exported to ONNX from the published
     * checkpoint and checked against PyTorch by scripts/export_models.py,
     * reproducibly (models/models.json). Published by CI as a release asset.
     */
    val BEAT_THIS = ModelSpec(
        fileName = "beat_this_final0.onnx",
        url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/beat_this_final0.onnx",
        sha256 = "80d04915143f9fcef27223dfe03cd4569bf5c7c9ff4cd7a34918e4b1424c682c",
        bytes = 82_001_399,
        label = "the beat model",
    )
}

/**
 * Downloads model files once, into [dir], and checks each against its pinned
 * SHA-256 before it is ever used. An interrupted download resumes where it
 * stopped (HTTP Range) the next time it is needed.
 */
class ModelStore(private val dir: File) {
    init {
        dir.mkdirs()
    }

    fun file(spec: ModelSpec) = File(dir, spec.fileName)

    fun isReady(spec: ModelSpec): Boolean = file(spec).let { it.isFile && it.length() == spec.bytes }

    /**
     * Returns the model file, downloading it first if needed. [onProgress]
     * gets 0..1 while downloading and returns false to cancel (the partial
     * file is kept so the next attempt resumes).
     */
    @Throws(IOException::class)
    fun ensure(spec: ModelSpec, onProgress: (Float) -> Boolean): File {
        val target = file(spec)
        if (isReady(spec)) return target
        val part = File(dir, "${spec.fileName}.part")
        if (part.length() > spec.bytes) part.delete()
        val digest = MessageDigest.getInstance("SHA-256")
        var have = if (part.isFile) part.length() else 0L
        if (have > 0) part.inputStream().use { input -> digestAll(input, digest) }
        if (have < spec.bytes) {
            val conn = open(spec.url, have)
            try {
                val code = conn.responseCode
                if (code == 200 && have > 0) {
                    // The server ignored the range: start again.
                    digest.reset()
                    have = 0
                } else if (code != 200 && code != 206) {
                    throw IOException("The model server answered $code")
                } else if (code == 206) {
                    val start = conn.getHeaderField("Content-Range")
                        ?.substringAfter("bytes ")?.substringBefore('-')?.toLongOrNull()
                    if (start != have) throw IOException("The model server resumed at the wrong place")
                }
                RandomAccessFile(part, "rw").use { out ->
                    out.setLength(have)
                    out.seek(have)
                    conn.inputStream.use { input ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            digest.update(buf, 0, n)
                            have += n
                            if (have > spec.bytes) throw IOException("The model file is larger than expected")
                            if (!onProgress(have.toFloat() / spec.bytes)) throw CancellationException("cancelled")
                        }
                    }
                }
            } finally {
                conn.disconnect()
            }
        }
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        if (have != spec.bytes || hex != spec.sha256) {
            part.delete()
            throw IOException("The downloaded ${spec.label.removePrefix("the ")} did not match its checksum. Try again on a different network.")
        }
        if (!part.renameTo(target)) throw IOException("Could not store ${spec.label}")
        return target
    }

    private fun digestAll(input: java.io.InputStream, digest: MessageDigest) {
        val buf = ByteArray(1 shl 16)
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            digest.update(buf, 0, n)
        }
    }

    private fun open(urlString: String, from: Long): HttpURLConnection {
        var url = URL(urlString)
        var redirects = 0
        while (true) {
            val conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", "StemDeck-Android")
                if (from > 0) setRequestProperty("Range", "bytes=$from-")
            }
            val code = conn.responseCode
            if (code in 300..399) {
                val loc = conn.getHeaderField("Location") ?: throw IOException("Bad redirect")
                conn.disconnect()
                url = URL(url, loc)
                if (++redirects > 5) throw IOException("Too many redirects")
                continue
            }
            if (code == 416) {
                // Range not satisfiable: the part file is already complete or stale.
                conn.disconnect()
                throw IOException("The model download could not be resumed")
            }
            return conn
        }
    }
}
