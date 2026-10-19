package org.stemdeckport.android.server

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import org.stemdeckport.android.data.Job
import org.stemdeckport.android.data.JobStore
import org.stemdeckport.android.data.MixExport
import org.stemdeckport.android.data.Settings
import org.stemdeckport.android.data.StemsZip
import org.stemdeckport.android.engine.NativeEngine
import java.io.File
import java.io.FileInputStream
import java.io.FilterInputStream
import java.io.InputStream
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.net.URLDecoder
import java.security.SecureRandom
import kotlin.concurrent.thread

/**
 * The part of desktop StemDeck's HTTP API that the bundled mobile UI calls,
 * reimplemented on the phone. Listens on 127.0.0.1 only, and every request
 * must carry the per-launch cookie the WebView receives from /auth, so other
 * apps on the phone cannot use it.
 */
class LocalServer(
    private val context: Context,
    private val store: JobStore,
    private val settings: Settings,
    private val onJobQueued: () -> Unit,
) : NanoHTTPD("127.0.0.1", 0) {

    val token: String = ByteArray(24).also { SecureRandom().nextBytes(it) }
        .joinToString("") { "%02x".format(it) }

    private val assets: AssetManager = context.assets

    fun startServer() {
        start(SOCKET_READ_TIMEOUT, true)
        Log.i(TAG, "listening on 127.0.0.1:$listeningPort")
    }

    /** The URL the WebView opens first; it trades the token for a cookie. */
    fun entryUrl(tab: String? = null): String =
        "http://127.0.0.1:$listeningPort/auth?t=$token" + (tab?.let { "&tab=$it" } ?: "")

    // An SSE stream must reach the page frame by frame; gzip would buffer it.
    override fun useGzipWhenAccepted(r: Response): Boolean = false

    override fun serve(session: IHTTPSession): Response {
        return try {
            route(session)
        } catch (e: Exception) {
            Log.e(TAG, "request failed: ${session.method} ${session.uri}", e)
            json(500, JSONObject().put("detail", "Internal error: ${e.message}"))
        }
    }

    private fun route(s: IHTTPSession): Response {
        val uri = s.uri
        if (uri == "/auth") {
            if (s.parameters["t"]?.firstOrNull() != token) return text(403, "Forbidden")
            val tab = s.parameters["tab"]?.firstOrNull()?.takeIf { it.matches(Regex("[a-z]+")) }
            return newFixedLengthResponse(status(302, "Found"), MIME_PLAINTEXT, "").apply {
                addHeader("Location", "/mobile/" + (tab?.let { "#$it" } ?: ""))
                addHeader("Set-Cookie", "$COOKIE=$token; Path=/; HttpOnly; SameSite=Strict")
                addHeader("Cache-Control", "no-store")
            }
        }
        if (s.cookies.read(COOKIE) != token) return text(403, "Forbidden")

        if (uri.startsWith("/api/")) return api(s)
        if (uri == "/" || uri == "/mobile") return redirect("/mobile/")
        return asset(uri)
    }

    // ─── static files from assets/web ───
    private fun asset(uri: String): Response {
        var path = URLDecoder.decode(uri, "UTF-8").trimStart('/')
        if (path.split('/').any { it == ".." }) return text(404, "Not found")
        if (path.endsWith("/") || path.isEmpty()) path += "index.html"
        val stream = try {
            assets.open("web/$path")
        } catch (e: Exception) {
            return text(404, "Not found")
        }
        return newChunkedResponse(status(200, "OK"), mimeFor(path), stream).apply {
            addHeader("Cache-Control", "no-cache")
        }
    }

    // ─── API ───
    private fun api(s: IHTTPSession): Response {
        val parts = s.uri.removePrefix("/api/").trimEnd('/').split('/')
        val method = s.method
        if (parts[0] == "config") {
            // Android: which models still have to be downloaded, so the Extract
            // screen can say so before the first song.
            val models = org.stemdeckport.android.App.instance.models
            return json(200, JSONObject().put("platform", "android").put("models", JSONObject()
                .put("separation", models.isReady(org.stemdeckport.android.engine.Models.DEMUCS))
                .put("beat", models.isReady(org.stemdeckport.android.engine.Models.BEAT_THIS))))
        }
        // Android: app settings (the stem file format).
        if (parts[0] == "settings") {
            if (method == Method.PUT || method == Method.POST) {
                settings.apply(JSONObject(readBody(s).ifBlank { "{}" }))
            }
            return json(200, settings.toJson())
        }
        if (parts[0] == "diagnostics") {
            return text(200, org.stemdeckport.android.Diagnostics.lastReport() ?: "No crash recorded.")
        }
        if (parts[0] != "jobs") return json(404, detail("not found"))

        if (parts.size == 1) {
            return when (method) {
                Method.GET -> json(200, JSONArray().apply { store.list().forEach { put(it.toState()) } })
                Method.POST -> createJob(s)
                else -> json(405, detail("method not allowed"))
            }
        }
        val id = parts[1]
        if (!ID_RE.matches(id)) return json(404, detail("job not found"))
        val job = store.get(id) ?: return json(404, detail("job not found"))

        if (parts.size == 2) {
            return if (method == Method.GET) json(200, job.toState()) else json(405, detail("method not allowed"))
        }
        return when {
            parts.size == 3 && parts[2] == "events" -> events(job)
            parts.size == 3 && parts[2] == "favorite" && method == Method.PUT -> {
                val body = readBody(s)
                val fav = JSONObject(body.ifBlank { "{}" }).optBoolean("favorite", false)
                store.setFavorite(id, fav)
                json(200, JSONObject().put("favorite", fav))
            }
            parts.size == 3 && parts[2] == "trash" && method == Method.POST -> {
                store.trash(id)
                json(200, JSONObject().put("ok", true))
            }
            parts.size == 3 && parts[2] == "cancel" && method == Method.POST -> {
                store.requestStop(id, "cancelled")
                json(200, JSONObject().put("ok", true))
            }
            parts.size == 3 && parts[2] == "vocal-split" ->
                json(501, detail("The lead/backing vocal split is not available on Android yet."))
            parts.size == 3 && parts[2] == "mixdown.wav" -> mixdown(s, job)
            // Android 1.1: the click track's beat grid, and every stem in one ZIP.
            parts.size == 4 && parts[2] == "stems" && parts[3] == "beats.json" -> beatGrid(job)
            parts.size == 4 && parts[2] == "stems" && parts[3] == "all.zip" -> stemsZip(s, job)
            parts.size == 4 && parts[2] == "stems" && (parts[3].endsWith(".wav") || parts[3].endsWith(".mp3")) ->
                stem(s, job, parts[3].substringBeforeLast('.'))
            else -> json(404, detail("not found"))
        }
    }

    private fun createJob(s: IHTTPSession): Response {
        val files = HashMap<String, String>()
        s.parseBody(files)
        val upload = files["file"]
        if (upload == null) {
            // JSON body: a link import. Not supported on the phone yet.
            return json(400, detail("Links are not supported on Android yet. Download the audio and upload the file."))
        }
        val stemsParam = s.parameters["stems"]?.firstOrNull()
        val requested: List<String>? = try {
            stemsParam?.let { p -> JSONArray(p).let { a -> List(a.length()) { a.getString(it) } } }
        } catch (e: Exception) {
            null
        }
        val selected = requested
            ?.filter { it in NativeEngine.SOURCES }
            ?.takeIf { it.isNotEmpty() }
            ?: NativeEngine.SOURCES
        val fileName = s.parameters["file"]?.firstOrNull()?.takeIf { it.isNotBlank() } ?: "audio"
        // parseBody's temp file is deleted when this request ends, so take a copy now.
        val staged = File(context.cacheDir, "upload-${System.nanoTime()}.${fileName.substringAfterLast('.', "bin").take(8)}")
        File(upload).copyTo(staged, overwrite = true)
        val format = s.parameters["format"]?.firstOrNull()?.takeIf { it in Settings.FORMATS } ?: settings.stemFormat
        val job = store.create(titleFrom(fileName), "local:${fileName.substringBeforeLast('.')}", selected, staged, format)
        onJobQueued()
        return json(200, JSONObject().put("job_id", job.id).put("status", job.status))
    }

    private fun events(job: Job): Response {
        val r = newChunkedResponse(status(200, "OK"), "text/event-stream", SseStream(store, job.id))
        r.addHeader("Cache-Control", "no-cache")
        r.addHeader("X-Accel-Buffering", "no")
        return r
    }

    private fun stem(s: IHTTPSession, job: Job, name: String): Response {
        if (name !in job.stems) return json(404, detail("stem not found"))
        val f = store.stemFile(job.id, name)
        if (!f.isFile) return json(404, detail("stem not found"))
        return fileWithRange(s, f, MixExport.mimeType(f.extension))
    }

    /** desktop's GET /jobs/{id}/stems/beats.json: 404 when the song has no grid. */
    private fun beatGrid(job: Job): Response {
        if (job.status != "done") return json(404, detail("job not ready"))
        val f = File(store.stemsDir(job.id), "beats.json")
        if (!f.isFile) return json(404, detail("beat grid not found"))
        return newFixedLengthResponse(status(200, "OK"), "application/json", FileInputStream(f), f.length()).apply {
            addHeader("Cache-Control", "no-cache")
        }
    }

    /** desktop's GET /jobs/{id}/stems/all.zip?stems=a,b, streamed as it is made. */
    private fun stemsZip(s: IHTTPSession, job: Job): Response {
        if (job.status != "done") return json(404, detail("job not ready"))
        val fmt = s.parameters["format"]?.firstOrNull()
        if (fmt != null && fmt != job.format) {
            return json(422, detail("this song's stems are stored as ${job.format.uppercase()}"))
        }
        val names = StemsZip.resolve(job, s.parameters["stems"]?.firstOrNull())
            .getOrElse { return json(422, detail(it.message ?: "bad request")) }
        val files = names.map { it to store.stemFile(job.id, it) }.filter { it.second.isFile }
        if (files.isEmpty()) return json(404, detail("no stems found"))
        val pipeIn = object : PipedInputStream(1 shl 16) {
            @Volatile var closed = false
            override fun close() { closed = true; super.close() }
        }
        val pipeOut = PipedOutputStream(pipeIn)
        thread(name = "zip") {
            try {
                pipeOut.use { StemsZip.write(files, StemsZip.titleSlug(job.title), it) { !pipeIn.closed } }
            } catch (e: Exception) {
                Log.w(TAG, "zip stopped", e)
            }
        }
        return newChunkedResponse(status(200, "OK"), "application/zip", pipeIn).apply {
            addHeader("Content-Disposition", "attachment; filename=\"${StemsZip.fileName(job)}\"")
        }
    }

    private fun mixdown(s: IHTTPSession, job: Job): Response {
        val names = s.parameters["stems"]?.firstOrNull()?.split(',')?.filter { it in job.stems } ?: job.stems
        val gains = s.parameters["gains"]?.firstOrNull()?.split(',')?.map { it.toFloatOrNull() ?: 1f }
        if (names.isEmpty()) return json(400, detail("nothing to mix"))
        val g = names.indices.map { gains?.getOrNull(it)?.coerceIn(0f, 2f) ?: 1f }
        val format = settings.stemFormat
        val click = MixExport.clickFrom(store, job) { k -> s.parameters[k]?.firstOrNull() }
        // Streamed as it is made. The work stops when the reader goes away
        // (the WebView hands downloads to the app and drops this request).
        val pipeIn = object : PipedInputStream(1 shl 16) {
            @Volatile var closed = false
            override fun close() { closed = true; super.close() }
        }
        val pipeOut = PipedOutputStream(pipeIn)
        thread(name = "mixdown") {
            try {
                pipeOut.use { MixExport.write(store, job, names, g, format, context.cacheDir, it, click) { !pipeIn.closed } }
            } catch (e: Exception) {
                Log.w(TAG, "mixdown stopped", e)
            }
        }
        return newChunkedResponse(status(200, "OK"), MixExport.mimeType(format), pipeIn).apply {
            addHeader("Content-Disposition", "attachment; filename=\"${safeFileName(job.title)} (mix).$format\"")
        }
    }

    private fun fileWithRange(s: IHTTPSession, f: File, mime: String): Response {
        val total = f.length()
        val range = s.headers["range"]
        if (range != null && range.startsWith("bytes=")) {
            val spec = range.removePrefix("bytes=").split(',')[0].trim()
            val dash = spec.indexOf('-')
            var start: Long
            var end: Long
            if (dash == 0) { // suffix range: last N bytes
                val n = spec.substring(1).toLongOrNull() ?: 0
                start = (total - n).coerceAtLeast(0); end = total - 1
            } else {
                start = spec.substring(0, dash).toLongOrNull() ?: 0
                end = spec.substring(dash + 1).toLongOrNull() ?: (total - 1)
            }
            end = end.coerceAtMost(total - 1)
            if (start >= total || start > end) {
                return newFixedLengthResponse(status(416, "Range Not Satisfiable"), MIME_PLAINTEXT, "").apply {
                    addHeader("Content-Range", "bytes */$total")
                }
            }
            val len = end - start + 1
            val input = FileInputStream(f).apply { channel.position(start) }
            return newFixedLengthResponse(status(206, "Partial Content"), mime, Bounded(input, len), len).apply {
                addHeader("Content-Range", "bytes $start-$end/$total")
                addHeader("Accept-Ranges", "bytes")
                addHeader("Cache-Control", "no-cache")
            }
        }
        return newFixedLengthResponse(status(200, "OK"), mime, FileInputStream(f), total).apply {
            addHeader("Accept-Ranges", "bytes")
        }
    }

    private fun readBody(s: IHTTPSession): String {
        val files = HashMap<String, String>()
        s.parseBody(files)
        files["postData"]?.let { return it }
        files["content"]?.let { p -> return File(p).readText() }
        return ""
    }

    private class Bounded(input: InputStream, private var left: Long) : FilterInputStream(input) {
        override fun read(): Int = if (left <= 0) -1 else super.read().also { if (it >= 0) left-- }
        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (left <= 0) return -1
            val n = super.read(b, off, minOf(len.toLong(), left).toInt())
            if (n > 0) left -= n
            return n
        }
    }

    companion object {
        private const val TAG = "LocalServer"
        private const val COOKIE = "sdtok"
        private val ID_RE = Regex("^[0-9a-f]{12}$")

        fun status(code: Int, text: String) = object : NanoHTTPD.Response.IStatus {
            override fun getDescription() = "$code $text"
            override fun getRequestStatus() = code
        }

        private fun reason(code: Int) = when (code) {
            200 -> "OK"; 400 -> "Bad Request"; 403 -> "Forbidden"; 404 -> "Not Found"
            405 -> "Method Not Allowed"; 422 -> "Unprocessable Entity"; 500 -> "Internal Server Error"; 501 -> "Not Implemented"
            else -> "Status"
        }

        fun json(code: Int, body: Any): NanoHTTPD.Response =
            NanoHTTPD.newFixedLengthResponse(status(code, reason(code)), "application/json", body.toString()).apply {
                addHeader("Cache-Control", "no-store")
            }

        fun text(code: Int, body: String): NanoHTTPD.Response =
            NanoHTTPD.newFixedLengthResponse(status(code, reason(code)), NanoHTTPD.MIME_PLAINTEXT, body)

        fun redirect(to: String): NanoHTTPD.Response =
            NanoHTTPD.newFixedLengthResponse(status(302, "Found"), NanoHTTPD.MIME_PLAINTEXT, "").apply { addHeader("Location", to) }

        fun detail(msg: String) = JSONObject().put("detail", msg)

        fun titleFrom(fileName: String): String =
            fileName.substringBeforeLast('.').replace('_', ' ').trim().ifEmpty { "Untitled" }

        fun safeFileName(s: String) = s.replace(Regex("[\\\\/:*?\"<>|]"), "_").take(120)

        fun mimeFor(path: String): String = when (path.substringAfterLast('.').lowercase()) {
            "html" -> "text/html; charset=utf-8"
            "js", "mjs" -> "text/javascript; charset=utf-8"
            "css" -> "text/css; charset=utf-8"
            "json" -> "application/json"
            "svg" -> "image/svg+xml"
            "png" -> "image/png"
            "wasm" -> "application/wasm"
            "txt" -> "text/plain; charset=utf-8"
            else -> "application/octet-stream"
        }
    }
}

/** Server-Sent Events for one job, in the format desktop's /events sends. */
private class SseStream(private val store: JobStore, private val id: String) : InputStream() {
    private var buf = ByteArray(0)
    private var pos = 0
    private var lastVersion = -1L
    private var finished = false
    @Volatile private var closed = false

    private fun fill(): Boolean {
        if (finished || closed) return false
        val job = store.awaitChange(id, lastVersion, 15_000) ?: run { finished = true; return false }
        val text = if (job.version != lastVersion) {
            lastVersion = job.version
            if (job.finished) finished = true
            "data: ${job.toState()}\n\n"
        } else {
            ": keepalive\n\n"
        }
        buf = text.toByteArray()
        pos = 0
        return true
    }

    override fun read(): Int {
        if (pos >= buf.size && !fill()) return -1
        return buf[pos++].toInt() and 0xff
    }

    override fun read(b: ByteArray, off: Int, len: Int): Int {
        if (len == 0) return 0
        if (pos >= buf.size && !fill()) return -1
        val n = minOf(len, buf.size - pos)
        System.arraycopy(buf, pos, b, off, n)
        pos += n
        return n
    }

    override fun available(): Int = buf.size - pos

    override fun close() { closed = true }
}
