package org.stemdeckport.android.engine

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException

/**
 * The htdemucs_6s weights of Demucs v4.1.0, exactly as the Demucs project
 * publishes them on the HuggingFace hub (adefossez/HTDemucs-6s, checkpoint
 * 5c90dfd2, safetensors). Downloaded once on first use and checked against
 * a pinned SHA-256 before it is ever loaded; the engine reads the file as is.
 */
class ModelManager(context: Context) {
    companion object {
        const val FILE_NAME = "htdemucs_6s-5c90dfd2-v4.1.0.safetensors"

        // Pinned to a revision of the repository, so the file cannot change under us.
        const val URL_STRING =
            "https://huggingface.co/adefossez/HTDemucs-6s/resolve/" +
                "3c5ee475be622df764938de97e4281a7b07ffa58/5c90dfd2.safetensors"
        const val SHA256 = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"

        // The demucs.cpp conversion of the same weights that builds up to 24
        // downloaded; removed once the v4.1.0 file is in place.
        private const val OLD_FILE_NAME = "ggml-model-htdemucs-6s-f16.bin"
    }

    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val modelFile = File(dir, FILE_NAME)

    fun isReady(): Boolean {
        val ready = modelFile.isFile && modelFile.length() > 1_000_000
        if (ready) File(dir, OLD_FILE_NAME).delete()
        return ready
    }

    /**
     * Returns the model file, downloading it first if needed. [onProgress]
     * gets 0..1 while downloading and returns false to cancel.
     */
    @Throws(IOException::class)
    fun ensure(onProgress: (Float) -> Boolean): File {
        if (isReady()) return modelFile
        val part = File(dir, "$FILE_NAME.part")
        part.delete()
        var url = URL(URL_STRING)
        var conn: HttpURLConnection
        var redirects = 0
        while (true) {
            conn = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", "StemDeck-Android")
            }
            val code = conn.responseCode
            if (code in 300..399) {
                val loc = conn.getHeaderField("Location") ?: throw IOException("Bad redirect")
                conn.disconnect()
                url = URL(url, loc)
                if (++redirects > 5) throw IOException("Too many redirects")
                continue
            }
            if (code != 200) {
                conn.disconnect()
                throw IOException("The model server answered $code")
            }
            break
        }
        val total = conn.contentLengthLong
        val digest = MessageDigest.getInstance("SHA-256")
        try {
            conn.inputStream.use { input ->
                part.outputStream().buffered(1 shl 16).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var done = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        digest.update(buf, 0, n)
                        done += n
                        if (total > 0 && !onProgress(done.toFloat() / total)) {
                            throw CancellationException("cancelled")
                        }
                    }
                }
            }
        } catch (e: Exception) {
            part.delete()
            throw e
        } finally {
            conn.disconnect()
        }
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        if (hex != SHA256) {
            part.delete()
            throw IOException("The downloaded model did not match its checksum. Try again on a different network.")
        }
        if (!part.renameTo(modelFile)) throw IOException("Could not store the model")
        return modelFile
    }
}
