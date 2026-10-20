package org.stemdeckport.android.engine

import android.content.Context
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.CancellationException

/**
 * Link import (1.4): yt-dlp on Chaquopy's Python (src/main/python/stemdeck_dl.py),
 * with QuickJS-NG (libqjs.so) for YouTube's JavaScript challenges.
 */
object LinkDownloader {
    /** Called with fraction 0..1 and a stage line; -1 only asks whether to go on. Return false to cancel. */
    fun interface Listener {
        fun onProgress(fraction: Double, text: String): Boolean
    }

    class Result(
        val file: File, val title: String?, val duration: Double?, val thumbnail: String?,
        val artist: String?, val track: String?, val album: String?,
    )

    private fun updateFile(context: Context) = File(context.filesDir, "yt-dlp/yt-dlp.zip")
    private fun qjs(context: Context) = File(context.applicationInfo.nativeLibraryDir, "libqjs.so").path

    @Synchronized
    private fun python(context: Context): Python {
        if (!Python.isStarted()) {
            // An updated yt-dlp (Settings) is put first on sys.path by stemdeck_dl.
            android.system.Os.setenv("STEMDECK_YTDLP_UPDATE", updateFile(context).path, true)
            Python.start(AndroidPlatform(context.applicationContext))
        }
        return Python.getInstance()
    }

    /** Downloads the audio of an already validated link into [dir]. Throws CancellationException or IOException. */
    fun download(context: Context, url: String, dir: File, maxDurationSec: Double, listener: Listener): Result {
        dir.mkdirs()
        try {
            val out = python(context).getModule("stemdeck_dl")
                .callAttr("download", url, dir.path, qjs(context), maxDurationSec, listener).toString()
            val o = JSONObject(out)
            fun s(k: String) = if (o.isNull(k)) null else o.optString(k).takeIf { it.isNotBlank() }
            return Result(
                File(o.getString("file")), s("title"), if (o.isNull("duration")) null else o.optDouble("duration"),
                s("thumbnail"), s("artist"), s("track"), s("album"),
            )
        } catch (e: PyException) {
            val msg = e.message.orEmpty()
            if ("Cancelled" in msg) throw CancellationException("cancelled")
            throw IOException(clean(msg), e)
        }
    }

    /** yt-dlp's errors, readable: "ERROR: [youtube] x: Video unavailable" -> "Video unavailable". */
    private fun clean(msg: String): String {
        val line = msg.lineSequence().firstOrNull { "ERROR:" in it || "Error:" in it } ?: msg.lineSequence().first()
        return line.substringAfterLast("ERROR: ").substringAfter(": ", line).trim().take(300)
            .ifEmpty { "the download failed" }
    }

    /** {"yt_dlp": version, "js_runtime": "quickjs-ng 0.17.0" or null, "updated": bool}. */
    fun info(context: Context): JSONObject = try {
        JSONObject(python(context).getModule("stemdeck_dl").callAttr("runtime_info", qjs(context)).toString())
    } catch (e: Exception) {
        JSONObject().put("error", e.message ?: e.javaClass.simpleName)
    }

    /**
     * Downloads the latest yt-dlp release (its zipapp, importable as is) and
     * checks it against the release's SHA2-256SUMS; it is used from the next
     * start of the app. YouTube changes often, and the yt-dlp in the APK is
     * only as new as the app.
     */
    fun update(context: Context): String {
        val base = "https://github.com/yt-dlp/yt-dlp/releases/latest/download"
        val sums = get("$base/SHA2-256SUMS").toString(Charsets.UTF_8)
        val want = sums.lineSequence().map { it.trim().split(Regex("\\s+")) }
            .firstOrNull { it.size == 2 && it[1].trimStart('*') == "yt-dlp" }?.get(0)
            ?: throw IOException("the release lists no checksum for yt-dlp")
        val data = get("$base/yt-dlp")
        val got = MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }
        if (!got.equals(want, ignoreCase = true)) throw IOException("the download does not match its checksum")
        val f = updateFile(context)
        f.parentFile?.mkdirs()
        val tmp = File(f.parentFile, "yt-dlp.zip.part")
        tmp.writeBytes(data)
        if (!tmp.renameTo(f)) throw IOException("could not save the update")
        return want
    }

    private fun get(url: String): ByteArray {
        var u = URL(url)
        repeat(6) {
            val c = u.openConnection() as HttpURLConnection
            c.instanceFollowRedirects = false
            c.connectTimeout = 15_000
            c.readTimeout = 30_000
            c.setRequestProperty("User-Agent", "StemDeck-Android (https://github.com/jPaauwe/stemdeck-android)")
            try {
                when (c.responseCode) {
                    in 300..399 -> { u = URL(u, c.getHeaderField("Location")); return@repeat }
                    200 -> return c.inputStream.use { it.readBytes() }
                    else -> throw IOException("GitHub answered ${c.responseCode}")
                }
            } finally {
                c.disconnect()
            }
        }
        throw IOException("too many redirects")
    }
}
