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
        sha256 = "94553b89d3d8eb5dbdbaea796583df6a6e6340b8b5488d48b04143a4c10d9729",
        bytes = 82_098_874,
        label = "the beat model",
    )

    /**
     * UVR-MDX-NET Karaoke 2 (Ultimate Vocal Remover, MIT with credit to UVR),
     * the lead/backing vocal split model desktop downloads through
     * audio-separator, from the same UVR model repository release.
     */
    val KARAOKE = ModelSpec(
        fileName = "UVR_MDXNET_KARA_2.onnx",
        url = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
        sha256 = "bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4",
        bytes = 52_786_726,
        label = "the vocal split model",
    )

    /**
     * All-In-One harmonix-all (Kim & Nam, MIT), the eight folds desktop
     * averages, each exported to ONNX from its checkpoint and checked against
     * PyTorch by scripts/export_models.py (models/models.json). 25 MB in all.
     */
    val SECTIONS: List<ModelSpec> = listOf(
        ModelSpec(
            fileName = "allinone_harmonix_fold0.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold0.onnx",
            sha256 = "7581c0c3cf74104342f6fe731fe3987870e6f3d6da7996177b0f5ed193646cde",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold1.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold1.onnx",
            sha256 = "7d45cd6421ccf8b6bd684e3bd626b708b4071ae9d1ea42d0737b87331e70d396",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold2.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold2.onnx",
            sha256 = "eb6de214aa957833bd6103a84d2c0f4bf72b2bdeab8af217f15ebbc8a0c24534",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold3.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold3.onnx",
            sha256 = "11dd0068462559280165296d87e5b0366edd12c38512511bc31aed650fbe51d3",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold4.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold4.onnx",
            sha256 = "df67e2ce1b9b1f5da2376940cefe104fb0e431daba3e17b38ebfb7daafc1c984",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold5.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold5.onnx",
            sha256 = "14158336792ebc5319dff0a129c85591e239ce16dde65321c46c7a548c413c25",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold6.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold6.onnx",
            sha256 = "43b2627438425db41c41c1e93a77b248a3ac32df5cfc47190b385752c0d3fb6a",
            bytes = 3_151_198,
            label = "the section model",
        ),
        ModelSpec(
            fileName = "allinone_harmonix_fold7.onnx",
            url = "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold7.onnx",
            sha256 = "c5709a1be3d2dcf2a1dd2b0acf279dc94b2649167b37d963bdf1f4af1af7c767",
            bytes = 3_151_198,
            label = "the section model",
        ),
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
    fun isReady(specs: List<ModelSpec>): Boolean = specs.all { isReady(it) }

    /** [ensure] for several files, with one progress over their total size. */
    fun ensureAll(specs: List<ModelSpec>, onProgress: (Float) -> Boolean): List<File> {
        val total = specs.sumOf { it.bytes }.toFloat()
        var before = 0L
        return specs.map { spec ->
            val f = ensure(spec) { p -> onProgress((before + p * spec.bytes) / total) }
            before += spec.bytes
            f
        }
    }

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
