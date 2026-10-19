package org.stemdeckport.compare

import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ContentValues
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CancellationException
import kotlin.concurrent.thread

/**
 * StemDeck Compare: runs one song through several engine builds bundled in
 * this APK and compares them: speed, memory, and how much their stems
 * differ.
 */
class CompareActivity : Activity() {
    private companion object {
        const val PICK = 1
        val STEMS = listOf("drums", "bass", "other", "vocals", "guitar", "piano")
    }

    private lateinit var log: TextView
    private lateinit var scroll: ScrollView
    private lateinit var songLabel: TextView
    private lateinit var refSpin: Spinner
    private lateinit var threadSpin: Spinner
    private lateinit var cooldown: CheckBox
    private lateinit var saveStems: CheckBox
    private lateinit var runBtn: Button
    private lateinit var cancelBtn: Button
    private lateinit var listenBtn: Button
    private val engineBoxes = mutableListOf<Pair<EngineInfo, CheckBox>>()
    private val report = StringBuilder()
    private var songUri: Uri? = null
    private var songName = ""
    @Volatile private var running = false
    @Volatile private var cancelled = false

    private val threadChoices = listOf(1, 2, 3, 4, 6)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val fg = Color.parseColor("#eaeaec")
        val dim = Color.parseColor("#8a8a92")
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0b0b0d"))
            setPadding(32, 48, 32, 32)
            fitsSystemWindows = true
        }
        fun label(t: String, size: Float = 14f, color: Int = fg) =
            TextView(this).apply { text = t; setTextColor(color); textSize = size }

        root.addView(TextView(this).apply {
            text = "StemDeck Compare"
            setTextColor(Color.parseColor("#4a9bf5")); textSize = 22f; typeface = Typeface.DEFAULT_BOLD
        })
        root.addView(label("Runs your song through the engine builds you pick and compares speed, memory and the stems they produce.", 13f, dim))

        val songRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        songRow.addView(Button(this).apply { text = "Choose MP3"; setOnClickListener { pick() } })
        songLabel = label("  no song chosen", 13f, dim)
        songRow.addView(songLabel)
        root.addView(songRow)

        root.addView(label("Engine builds:"))
        val engines = Host.engines
        for ((i, e) in engines.withIndex()) {
            val cb = CheckBox(this).apply {
                text = "${e.label}  —  ${e.description}"
                setTextColor(fg); textSize = 12f
                isChecked = i == 0 || i == engines.lastIndex
            }
            engineBoxes += e to cb
            root.addView(cb)
        }

        val opts = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        opts.addView(label("Reference: "))
        refSpin = Spinner(this).apply {
            adapter = ArrayAdapter(this@CompareActivity, android.R.layout.simple_spinner_dropdown_item, engines.map { it.label })
        }
        opts.addView(refSpin)
        opts.addView(label("   Threads: "))
        threadSpin = Spinner(this).apply {
            adapter = ArrayAdapter(this@CompareActivity, android.R.layout.simple_spinner_dropdown_item, threadChoices.map { "$it" })
            setSelection(1)
        }
        opts.addView(threadSpin)
        root.addView(opts)
        cooldown = CheckBox(this).apply {
            text = "Let the phone cool down between builds (fairer timing)"; setTextColor(fg); textSize = 12f; isChecked = true
        }
        root.addView(cooldown)
        saveStems = CheckBox(this).apply {
            text = "Save every build's stems to Downloads/StemDeck Compare"; setTextColor(fg); textSize = 12f
        }
        root.addView(saveStems)

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        runBtn = Button(this).apply { text = "Run"; setOnClickListener { start() } }
        cancelBtn = Button(this).apply {
            text = "Cancel"; isEnabled = false
            setOnClickListener { cancelled = true; Host.nCancel(); out("Cancelling…") }
        }
        listenBtn = Button(this).apply {
            text = "Listen"
            isEnabled = ListenActivity.hasResults(this@CompareActivity)
            setOnClickListener { startActivity(Intent(this@CompareActivity, ListenActivity::class.java)) }
        }
        buttons.addView(runBtn)
        buttons.addView(cancelBtn)
        buttons.addView(listenBtn)
        buttons.addView(Button(this).apply { text = "Copy"; setOnClickListener { copy() } })
        buttons.addView(Button(this).apply { text = "Share"; setOnClickListener { share() } })
        root.addView(buttons)

        log = TextView(this).apply { setTextColor(fg); textSize = 11f; typeface = Typeface.MONOSPACE; setTextIsSelectable(true) }
        scroll = ScrollView(this).apply { addView(log) }
        root.addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)

        if (engines.isEmpty()) out("This APK has no engine builds bundled.")
        intent?.let { handleShare(it) }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleShare(intent)
    }

    private fun handleShare(i: Intent) {
        val uri = if (i.action == Intent.ACTION_SEND) {
            if (Build.VERSION.SDK_INT >= 33) i.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            else @Suppress("DEPRECATION") i.getParcelableExtra(Intent.EXTRA_STREAM)
        } else if (i.action == Intent.ACTION_VIEW) i.data else null
        uri?.let { setSong(it) }
    }

    private fun pick() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
        }, PICK)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK && resultCode == RESULT_OK) data?.data?.let { setSong(it) }
    }

    private fun setSong(uri: Uri) {
        songUri = uri
        songName = try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
                if (it.moveToFirst()) it.getString(0) else null
            }
        } catch (_: Exception) { null } ?: (uri.lastPathSegment ?: "song")
        songLabel.text = "  $songName"
    }

    private fun out(line: String) {
        report.append(line).append('\n')
        runOnUiThread {
            log.text = report
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun copy() {
        (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager)
            .setPrimaryClip(ClipData.newPlainText("StemDeck Compare", report.toString()))
        Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
    }

    private fun share() {
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"; putExtra(Intent.EXTRA_TEXT, report.toString())
        }, "Share results"))
    }

    private fun thermal(): String = when ((getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "none"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
        else -> "CRITICAL"
    }

    private fun thermalLevel(): Int = (getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus

    // ─── memory sampling ───
    private class PeakRss {
        @Volatile var peakKb = 0L
        @Volatile private var stop = false
        private val t = thread(name = "rss") {
            while (!stop) {
                val kb = rssKb()
                if (kb > peakKb) peakKb = kb
                Thread.sleep(100)
            }
        }
        fun finish(): Long { stop = true; t.join(); return peakKb }
        companion object {
            fun rssKb(): Long = try {
                File("/proc/self/status").readLines().first { it.startsWith("VmRSS:") }.split(Regex("\\s+"))[1].toLong()
            } catch (_: Exception) { 0L }
        }
    }

    private data class Result(
        val engine: EngineInfo, val dir: File,
        val seconds: Double, val loadSeconds: Double, val peakGb: Double,
        val thermalStart: String, val thermalEnd: String,
    )

    // ─── run ───
    private fun start() {
        if (running) return
        val uri = songUri ?: run { Toast.makeText(this, "Choose an MP3 first", Toast.LENGTH_SHORT).show(); return }
        val chosen = engineBoxes.filter { it.second.isChecked }.map { it.first }
        val ref = Host.engines.getOrNull(refSpin.selectedItemPosition)
        if (chosen.isEmpty() || ref == null) { Toast.makeText(this, "Pick at least one build", Toast.LENGTH_SHORT).show(); return }
        // The reference always runs, first.
        val order = listOf(ref) + chosen.filter { it.id != ref.id }
        val threads = threadChoices[threadSpin.selectedItemPosition]
        val cool = cooldown.isChecked
        val save = saveStems.isChecked
        running = true; cancelled = false
        runBtn.isEnabled = false; cancelBtn.isEnabled = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        report.clear()
        listenBtn.isEnabled = false
        thread(name = "compare") {
            try {
                run(uri, order, ref, threads, cool, save)
            } catch (e: CancellationException) {
                out("\nCancelled.")
            } catch (e: Throwable) {
                out("\nFAILED: ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                Host.nClose()
                running = false
                runOnUiThread {
                    runBtn.isEnabled = true; cancelBtn.isEnabled = false
                    listenBtn.isEnabled = ListenActivity.hasResults(this)
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            }
        }
    }

    private fun checkCancel() { if (cancelled) throw CancellationException() }

    private fun progressLine(prefix: String): Progress {
        var last = -1
        return Progress { f ->
            val pct = (f * 100).toInt()
            if (pct / 10 != last / 10) { last = pct; out("   $prefix ${pct / 10 * 10}%") }
            !cancelled
        }
    }

    private fun run(uri: Uri, order: List<EngineInfo>, ref: EngineInfo, threads: Int, cool: Boolean, save: Boolean) {
        out(header())
        val models = ModelManager(this)
        if (!models.isReady()) out("Downloading the model (55 MB, first run only)…")
        val model = models.ensure { !cancelled }.absolutePath
        checkCancel()

        // Decode the song once, prepare it with the reference build.
        val work = ListenActivity.workDir(this).apply { deleteRecursively(); mkdirs() }
        val input = File(work, "input.${songName.substringAfterLast('.', "mp3")}")
        contentResolver.openInputStream(uri)!!.use { s -> input.outputStream().use { s.copyTo(it, 1 shl 16) } }
        out("Decoding $songName …")
        val raw = File(work, "raw.f32")
        val dec = AudioDecoder.decode(input, raw) { !cancelled }
        val prepared = File(work, "prepared.f32")
        Host.nOpen(ref.id).takeIf { it.isNotEmpty() }?.let { throw IllegalStateException(it) }
        val pr = Host.nPrepare(raw.path, dec.sampleRate, dec.channels, prepared.path, progressLine("preparing"))
        if (pr < 0) throw CancellationException()
        if (pr == 0) throw IllegalStateException("preparing the audio failed")
        raw.delete(); input.delete()
        val songSeconds = prepared.length() / 8.0 / 44100.0
        out("Song: $songName, ${fmtTime(songSeconds)}, decoded from ${dec.sampleRate} Hz ${dec.channels} ch")
        out("Threads: $threads   Reference: ${ref.label}\n")

        val results = mutableListOf<Result>()
        for ((i, e) in order.withIndex()) {
            checkCancel()
            if (cool && i > 0) coolDown()
            out("▶ ${e.label}: ${e.description}")
            val open = Host.nOpen(e.id)
            if (open.isNotEmpty()) { out("   could not open: $open"); continue }
            val t0 = SystemClock.elapsedRealtime()
            val err = Host.nLoadModel(model)
            if (err.isNotEmpty()) { out("   model: $err"); continue }
            val loadS = (SystemClock.elapsedRealtime() - t0) / 1000.0
            val dir = File(work, e.id).apply { mkdirs() }
            val th0 = thermal()
            val rss = PeakRss()
            val t1 = SystemClock.elapsedRealtime()
            val js = JSONObject(Host.nSeparate(prepared.path, dir.path, threads, 24, progressLine("separating")))
            val secs = (SystemClock.elapsedRealtime() - t1) / 1000.0
            val peak = rss.finish()
            Host.nClose()
            if (js.has("cancelled")) throw CancellationException()
            if (js.has("error")) { out("   failed: ${js.getString("error")}"); continue }
            out("   done in %.1f s (%.2fx song length)".format(secs, secs / songSeconds))
            results += Result(e, dir, secs, loadS, peak / 1e6, th0, thermal())
        }
        if (results.isEmpty()) return
        val refRes = results.firstOrNull { it.engine.id == ref.id }

        // ─── speed ───
        out("\nSPEED (${fmtTime(songSeconds)} song, $threads thread${if (threads == 1) "" else "s"})")
        out("  %-14s %9s %8s %9s %9s  %s".format("build", "time", "x song", "speed-up", "peak mem", "thermal"))
        for (r in results) {
            val up = refRes?.let { it.seconds / r.seconds }
            out("  %-14s %8.1fs %7.2fx %9s %7.2f GB  %s→%s".format(
                r.engine.label.take(14), r.seconds, r.seconds / songSeconds,
                up?.let { "%.2fx".format(it) } ?: "-", r.peakGb, r.thermalStart, r.thermalEnd))
        }

        // What the Listen screen needs: the builds, and where each differs.
        val diffs = JSONObject()

        // ─── differences vs reference ───
        if (refRes != null && results.size > 1) {
            out("\nDIFFERENCE FROM ${ref.label.uppercase()} (SNR in dB; >60 identical, 40–60 inaudible, 20–40 small, <20 clearly different)")
            out("  %-14s ".format("build") + STEMS.joinToString(" ") { "%7s".format(it) } + "     all")
            val worst = mutableListOf<String>()
            for (r in results) {
                if (r === refRes) continue
                val d = JSONObject(Host.nDiff(refRes.dir.path, r.dir.path))
                diffs.put(r.engine.id, d)
                val stems = d.getJSONArray("stems")
                val cells = (0 until stems.length()).map { stems.getJSONObject(it) }
                out("  %-14s ".format(r.engine.label.take(14)) +
                    cells.joinToString(" ") { "%7.1f".format(it.getDouble("snr_db")) } +
                    "  %6.1f".format(d.getDouble("all_snr_db")))
                cells.filter { it.getLong("worst_at") >= 0 }.minByOrNull { it.getDouble("worst_db") }?.let {
                    worst += "  ${r.engine.label}: ${it.getString("name")} differs most at ${fmtTime(it.getLong("worst_at").toDouble())} (%.1f dB in that second)".format(it.getDouble("worst_db"))
                }
                cells.filter { it.getLong("frames") != it.getLong("frames_ref") }.forEach {
                    worst += "  ${r.engine.label}: ${it.getString("name")} has ${it.getLong("frames")} frames, reference ${it.getLong("frames_ref")}"
                }
            }
            if (worst.isNotEmpty()) { out("\n  Where to listen:"); worst.forEach { out(it) } }

            if (results.size > 2) {
                out("\nDIFFERENCE FROM THE PREVIOUS BUILD (all stems, SNR dB)")
                for (k in 1 until results.size) {
                    val d = JSONObject(Host.nDiff(results[k - 1].dir.path, results[k].dir.path))
                    out("  %s → %s: %.1f dB".format(results[k - 1].engine.label, results[k].engine.label, d.getDouble("all_snr_db")))
                }
            }
        }

        // ─── per-build quality proxies ───
        out("\nSTEMS ADD BACK UP TO THE SONG (SNR dB, higher is better)")
        for (r in results) {
            val j = JSONObject(Host.nReconstruction(r.dir.path, prepared.path))
            out("  %-14s %s".format(r.engine.label.take(14), if (j.has("snr_db")) "%.1f".format(j.getDouble("snr_db")) else j.optString("error")))
        }
        out("\nSTEM LEVELS (RMS dBFS)")
        out("  %-14s ".format("build") + STEMS.joinToString(" ") { "%7s".format(it) })
        for (r in results) {
            val d = JSONObject(Host.nDiff(r.dir.path, r.dir.path))
            val stems = d.getJSONArray("stems")
            out("  %-14s ".format(r.engine.label.take(14)) +
                (0 until stems.length()).joinToString(" ") { "%7.1f".format(stems.getJSONObject(it).getDouble("rms_db")) })
        }

        writeRunInfo(work, results, refRes?.engine ?: ref, songSeconds, diffs)

        if (save) {
            out("\nSaving stems to Downloads/StemDeck Compare …")
            for (r in results) saveToDownloads(r)
            out("  saved")
        }
        out("\nDone. Tap Listen to hear the builds side by side, or Copy to share these results.")
    }

    private fun writeRunInfo(work: File, results: List<Result>, ref: EngineInfo, songSeconds: Double, diffs: JSONObject) {
        val o = JSONObject()
            .put("song", songName)
            .put("seconds", songSeconds)
            .put("ref", ref.id)
            .put("engines", org.json.JSONArray().apply {
                for (r in results) put(JSONObject()
                    .put("id", r.engine.id).put("label", r.engine.label)
                    .put("description", r.engine.description).put("seconds", r.seconds))
            })
            .put("diffs", diffs)
        File(work, ListenActivity.RUN_INFO).writeText(o.toString())
    }

    private fun coolDown() {
        // Wait at least 20 s, and until the phone reports no thermal
        // pressure (at most 3 minutes), so later builds are not timed on a
        // hot, throttled CPU.
        val end = SystemClock.elapsedRealtime() + 180_000
        val minEnd = SystemClock.elapsedRealtime() + 20_000
        out("   cooling down (thermal ${thermal()})…")
        while (SystemClock.elapsedRealtime() < end) {
            checkCancel()
            if (SystemClock.elapsedRealtime() > minEnd && thermalLevel() <= PowerManager.THERMAL_STATUS_NONE) break
            Thread.sleep(1000)
        }
    }

    private fun saveToDownloads(r: Result) {
        val base = songName.substringBeforeLast('.').replace(Regex("[\\\\/:*?\"<>|]"), "_")
        for (s in STEMS) {
            val f = File(r.dir, "$s.wav")
            if (!f.isFile) continue
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, "$s.wav")
                put(MediaStore.MediaColumns.MIME_TYPE, "audio/wav")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/StemDeck Compare/$base/${r.engine.label}")
            }
            val target = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values) ?: continue
            contentResolver.openOutputStream(target)?.use { o -> f.inputStream().use { it.copyTo(o, 1 shl 16) } }
        }
    }

    private fun fmtTime(s: Double): String {
        val t = s.toInt()
        return "%d:%02d".format(t / 60, t % 60)
    }

    private fun header(): String {
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
        val ver = try {
            val pi = packageManager.getPackageInfo(packageName, 0)
            "${pi.versionName} (build ${if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode else 0})"
        } catch (_: Exception) { "?" }
        return "StemDeck Compare $ver, ${SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())}\n" +
            "${Build.MANUFACTURER} ${Build.MODEL}, $soc, Android ${Build.VERSION.RELEASE}, " +
            "${Runtime.getRuntime().availableProcessors()} cores, %.1f GB RAM".format(mi.totalMem / 1e9)
    }
}
