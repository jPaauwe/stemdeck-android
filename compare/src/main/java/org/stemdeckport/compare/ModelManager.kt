package org.stemdeckport.compare

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException

/**
 * The htdemucs_6s weights in demucs.cpp's ggml format. Downloaded once on
 * first use (desktop StemDeck does the same on its first launch) and checked
 * against a pinned SHA-256 before it is ever loaded.
 */
class ModelManager(context: Context) {
    companion object {
        const val FILE_NAME = "ggml-model-htdemucs-6s-f16.bin"

        // Pinned to a revision of the dataset, so the file cannot change under us.
        const val URL_STRING =
            "https://huggingface.co/datasets/Retrobear/demucs.cpp/resolve/" +
                "8f58ac0491bbea657275bcd3e38af1ab3a27bfc9/$FILE_NAME"
        const val SHA256 = "09704f4ceae204e56e77d5eefd6ac71d7275be81fd507e6913371d59abcee856"
    }

    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val modelFile = File(dir, FILE_NAME)

    fun isReady(): Boolean = modelFile.isFile && modelFile.length() > 1_000_000

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
                setRequestProperty("User-Agent", "StemDeck-Compare")
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
