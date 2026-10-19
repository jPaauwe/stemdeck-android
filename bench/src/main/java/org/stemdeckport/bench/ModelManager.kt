package org.stemdeckport.bench

import android.content.Context
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException

/**
 * A model file, downloaded once on first use and checked against a pinned
 * SHA-256 before it is ever loaded. By default the htdemucs_6s weights in
 * demucs.cpp's ggml format; [beatThis] is the app's beat model.
 */
class ModelManager(
    context: Context,
    private val FILE_NAME: String = "ggml-model-htdemucs-6s-f16.bin",
    // Pinned to a revision of the dataset, so the file cannot change under us.
    private val URL_STRING: String = "https://huggingface.co/datasets/Retrobear/demucs.cpp/resolve/" +
        "8f58ac0491bbea657275bcd3e38af1ab3a27bfc9/ggml-model-htdemucs-6s-f16.bin",
    private val SHA256: String = "09704f4ceae204e56e77d5eefd6ac71d7275be81fd507e6913371d59abcee856",
) {
    companion object {
        /** The same file and checksum as the app (app/.../engine/ModelStore.kt). */
        fun beatThis(context: Context) = ModelManager(
            context, "beat_this_final0.onnx",
            "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/beat_this_final0.onnx",
            "94553b89d3d8eb5dbdbaea796583df6a6e6340b8b5488d48b04143a4c10d9729",
        )

        /** The section model's eight folds (the app's Models.SECTIONS). */
        fun sections(context: Context) = listOf(
            ModelManager(
                context, "allinone_harmonix_fold0.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold0.onnx",
                "7581c0c3cf74104342f6fe731fe3987870e6f3d6da7996177b0f5ed193646cde",
            ),
            ModelManager(
                context, "allinone_harmonix_fold1.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold1.onnx",
                "7d45cd6421ccf8b6bd684e3bd626b708b4071ae9d1ea42d0737b87331e70d396",
            ),
            ModelManager(
                context, "allinone_harmonix_fold2.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold2.onnx",
                "eb6de214aa957833bd6103a84d2c0f4bf72b2bdeab8af217f15ebbc8a0c24534",
            ),
            ModelManager(
                context, "allinone_harmonix_fold3.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold3.onnx",
                "11dd0068462559280165296d87e5b0366edd12c38512511bc31aed650fbe51d3",
            ),
            ModelManager(
                context, "allinone_harmonix_fold4.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold4.onnx",
                "df67e2ce1b9b1f5da2376940cefe104fb0e431daba3e17b38ebfb7daafc1c984",
            ),
            ModelManager(
                context, "allinone_harmonix_fold5.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold5.onnx",
                "14158336792ebc5319dff0a129c85591e239ce16dde65321c46c7a548c413c25",
            ),
            ModelManager(
                context, "allinone_harmonix_fold6.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold6.onnx",
                "43b2627438425db41c41c1e93a77b248a3ac32df5cfc47190b385752c0d3fb6a",
            ),
            ModelManager(
                context, "allinone_harmonix_fold7.onnx",
                "https://github.com/jPaauwe/stemdeck-android/releases/download/models-1/allinone_harmonix_fold7.onnx",
                "c5709a1be3d2dcf2a1dd2b0acf279dc94b2649167b37d963bdf1f4af1af7c767",
            ),
        )

        /** The lead/backing split model (the app's Models.KARAOKE). */
        fun karaoke(context: Context) = ModelManager(
            context, "UVR_MDXNET_KARA_2.onnx",
            "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
            "bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4",
        )
    }

    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val modelFile = File(dir, FILE_NAME)

    fun isReady(): Boolean = modelFile.isFile && modelFile.length() > 1_000_000 // every model here is larger

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
                setRequestProperty("User-Agent", "StemDeck-Bench")
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
