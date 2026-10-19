package org.stemdeckport.bench

import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.os.SystemClock
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
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread
import kotlin.math.roundToInt

/**
 * StemDeck Bench: measures the separation engine on this phone.
 *
 *  1. One 7.8 s segment with each engine (the original demucs.cpp and
 *     builds 4–10 on one thread; the app's engine on one core, on every
 *     core, and on every core plus the GPU): time, where it goes, and
 *     whether each output matches the original's.
 *  2. A whole test song through the real separation path, the app's engine
 *     with several segments at once, CPU only and with the GPU: time,
 *     real-time factor and peak memory.
 */
class BenchActivity : Activity() {
    private lateinit var log: TextView
    private lateinit var scroll: ScrollView
    private lateinit var runBtn: Button
    private lateinit var lengthSpin: Spinner
    private lateinit var baselineFull: CheckBox
    private val threadBoxes = mutableMapOf<Int, CheckBox>()
    private val report = StringBuilder()
    @Volatile private var running = false

    private val songLengths = listOf(30, 60, 180)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val bg = Color.parseColor("#0b0b0d")
        val fg = Color.parseColor("#eaeaec")
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(bg)
            setPadding(32, 48, 32, 32)
            fitsSystemWindows = true
        }
        fun label(t: String) = TextView(this).apply { text = t; setTextColor(fg); textSize = 14f }

        root.addView(TextView(this).apply {
            text = "StemDeck Bench"
            setTextColor(Color.parseColor("#f5b417")); textSize = 22f; typeface = Typeface.DEFAULT_BOLD
        })
        root.addView(label("Times the separation engine on this phone. Keep the screen on and the phone at room temperature; a full run takes several minutes."))

        val row1 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row1.addView(label("Test song length: "))
        lengthSpin = Spinner(this).apply {
            adapter = ArrayAdapter(this@BenchActivity, android.R.layout.simple_spinner_dropdown_item, songLengths.map { "$it s" })
            setSelection(0)
        }
        row1.addView(lengthSpin)
        root.addView(row1)

        val row2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row2.addView(label("Segments at once: "))
        for (t in listOf(1, 2, 3, 4)) {
            val cb = CheckBox(this).apply { text = "$t"; setTextColor(fg); isChecked = t in setOf(1, 2, 3) }
            threadBoxes[t] = cb
            row2.addView(cb)
        }
        root.addView(row2)
        baselineFull = CheckBox(this).apply {
            text = "Also run the full song with the original demucs.cpp (1 thread) and builds 4–10 (2 threads)"
            setTextColor(fg); isChecked = true
        }
        root.addView(baselineFull)

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        runBtn = Button(this).apply { text = "Run benchmark"; setOnClickListener { start() } }
        buttons.addView(runBtn)
        buttons.addView(Button(this).apply { text = "Copy"; setOnClickListener { copy() } })
        buttons.addView(Button(this).apply { text = "Share"; setOnClickListener { share() } })
        root.addView(buttons)

        log = TextView(this).apply {
            setTextColor(fg); textSize = 11f; typeface = Typeface.MONOSPACE; setTextIsSelectable(true)
        }
        scroll = ScrollView(this).apply { addView(log) }
        root.addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)

        out(deviceInfo())
    }

    private fun out(line: String) {
        report.append(line).append('\n')
        runOnUiThread {
            log.text = report
            scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun copy() {
        val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("StemDeck Bench", report.toString()))
        Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
    }

    private fun share() {
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, report.toString())
        }, "Share results"))
    }

    // ─── device ───
    private fun deviceInfo(): String {
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val cores = Runtime.getRuntime().availableProcessors()
        val freqs = (0 until cores).map { c ->
            readText("/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq")?.trim()?.toLongOrNull()?.div(1000)
        }
        val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
        return buildString {
            append("StemDeck Bench ${BuildConfigInfo.version(this@BenchActivity)}\n")
            append("${SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())}\n")
            append("Device: ${Build.MANUFACTURER} ${Build.MODEL}, SoC: $soc\n")
            append("Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}), ABI ${Build.SUPPORTED_ABIS.firstOrNull()}\n")
            append("CPU: $cores cores, max MHz per core: ${freqs.joinToString(" ") { it?.toString() ?: "?" }}\n")
            append("RAM: %.1f GB".format(mi.totalMem / 1e9))
        }
    }

    private fun thermal(): String {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        return when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "MODERATE (throttling likely)"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE (throttling)"
            else -> "CRITICAL (throttling)"
        }
    }

    private fun readText(path: String): String? = try { File(path).readText() } catch (_: Exception) { null }

    // ─── memory sampling ───
    private class PeakRss {
        @Volatile var peakKb = 0L
        @Volatile private var stop = false
        private val t = thread(name = "rss") {
            while (!stop) {
                val kb = currentRssKb()
                if (kb > peakKb) peakKb = kb
                Thread.sleep(100)
            }
        }
        fun finish(): Long { stop = true; t.join(); return peakKb }

        companion object {
            fun currentRssKb(): Long = try {
                File("/proc/self/status").readLines().first { it.startsWith("VmRSS:") }
                    .split(Regex("\\s+"))[1].toLong()
            } catch (_: Exception) { 0L }
        }
    }

    // ─── run ───
    private fun start() {
        if (running) return
        val threads = threadBoxes.filterValues { it.isChecked }.keys.sorted()
        val seconds = songLengths[lengthSpin.selectedItemPosition]
        val withBaseline = baselineFull.isChecked
        running = true
        runBtn.isEnabled = false
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        report.clear()
        out(deviceInfo())
        thread(name = "bench") {
            try {
                run(threads, seconds, withBaseline)
            } catch (e: Throwable) {
                out("\nFAILED: ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                running = false
                runOnUiThread {
                    runBtn.isEnabled = true
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            }
        }
    }

    private fun run(threads: List<Int>, songSeconds: Int, withBaseline: Boolean) {
        out("Thermal at start: ${thermal()}")
        val models = ModelManager(this)
        if (!models.isReady()) out("\nDownloading the model (55 MB, first run only)…")
        var lastPct = -1
        val model = models.ensure { f ->
            val pct = (f * 10).toInt() * 10
            if (pct != lastPct) { lastPct = pct; out("  $pct%") }
            true
        }.absolutePath
        val work = File(filesDir, "bench").apply { deleteRecursively(); mkdirs() }

        // How the app's engine is set up for each run: threads doing maths
        // (0 = all cores) and the GPU. The other two engines have no such
        // settings and run on one thread.
        data class Variant(val key: String, val label: String, val eng: EngineNative, val threads: Int, val gpu: Boolean)
        fun EngineNative.plain(label: String = this.label) = Variant(key, label, this, 1, false)
        fun setup(v: Variant): String = if (v.eng === OptNative) OptNative.nSetCompute(v.threads, v.gpu) else "1 core"

        val gpuInfo = OptNative.nSetCompute(0, true)
        val hasGpu = gpuInfo.contains("GPU: ")
        out("\nApp engine: $gpuInfo")
        val app1 = Variant("app1", "app engine, 1 core", OptNative, 1, false)
        val appAll = Variant("appall", "app engine, all cores", OptNative, 0, false)
        val appGpu = Variant("appgpu", "app engine, cores + GPU", OptNative, 0, true)
        val segVariants = listOf(OrigNative.plain(), BaseNative.plain(), app1, appAll) + if (hasGpu) listOf(appGpu) else emptyList()

        // 1. One segment, every engine and setup.
        out("\n1. One 7.8 s segment (2 rounds each, best kept)")
        val results = mutableMapOf<String, JSONObject>()
        for (round in 1..2) {
            for (v in segVariants) {
                setup(v)
                val t0 = SystemClock.elapsedRealtime()
                val err = v.eng.load(model)
                if (err.isNotEmpty()) throw IllegalStateException("${v.label}: $err")
                val loadMs = SystemClock.elapsedRealtime() - t0
                val js = JSONObject(v.eng.profile(File(work, "seg-${v.key}.f32").path))
                v.eng.free()
                if (js.has("error")) throw IllegalStateException("${v.label}: ${js.getString("error")}")
                val ms = js.getDouble("ms")
                out("  round $round %-26s %6.2f s   (model load %.1f s, thermal %s)".format(v.label, ms / 1000, loadMs / 1000.0, thermal()))
                val prev = results[v.key]
                if (prev == null || ms < prev.getDouble("ms")) results[v.key] = js
            }
        }
        val origMs = results.getValue(OrigNative.key).getDouble("ms")
        out("  best:")
        for (v in segVariants) {
            val ms = results.getValue(v.key).getDouble("ms")
            out("   %-26s %6.2f s   %.2fx the original's speed".format(v.label, ms / 1000, origMs / ms))
        }
        val names = listOf("drums", "bass", "other", "vocals", "guitar", "piano")
        out("  output vs the original, SNR dB (higher = closer; >60 is identical to the ear):")
        for (v in segVariants.drop(1)) {
            val cmp = JSONObject(OptNative.compare(File(work, "seg-original.f32").path, File(work, "seg-${v.key}.f32").path))
            if (cmp.has("snr_db")) {
                val snr = cmp.getJSONArray("snr_db")
                out("   %-26s ".format(v.label) + names.indices.joinToString("  ") { "${names[it]} %.0f".format(snr.getDouble(it)) })
            } else out("   ${v.label}: $cmp")
        }

        val top = if (hasGpu) appGpu else appAll
        val best = results.getValue(top.key)
        val topMs = best.getDouble("ms")
        out("\n  Where the time goes (${top.label}, per stage):")
        table(best.getJSONArray("stages"), topMs, nameIdx = 0, msIdx = 1, callsIdx = -1)
        out("\n  Per operation (${top.label}):")
        table(best.getJSONArray("ops"), topMs, nameIdx = 0, msIdx = 2, callsIdx = 1)

        // 2. Full song through the real separation path.
        out("\n2. Full ${songSeconds} s test song, as the app separates it")
        val song = File(work, "song.f32").path
        if (!OptNative.writeSong(song, songSeconds.toDouble())) throw IllegalStateException("could not write the test song")
        val runs = mutableListOf<Pair<Variant, Int>>()
        if (withBaseline) { runs += OrigNative.plain() to 1; runs += BaseNative.plain() to 2 }
        threads.forEach { w ->
            runs += appAll to w
            if (hasGpu) runs += appGpu to w
        }
        var stems: File? = null  // one app-engine separation, kept for section 3
        for ((v, w) in runs) {
            setup(v)
            val outDir = File(work, "full-${v.key}-$w").apply { mkdirs() }
            val rss = PeakRss()
            val js = JSONObject(v.eng.separate(model, song, outDir.path, w, 24))
            val peak = rss.finish()
            if (stems == null && v.eng === OptNative && !js.has("error")) stems = outDir else outDir.deleteRecursively()
            if (js.has("error")) { out("  ${v.label}, $w at once: ${js.getString("error")}"); continue }
            val sec = js.getDouble("seconds")
            out("  %-26s %d at once: %6.1f s  = %.2fx song length   peak memory %.2f GB   thermal %s".format(
                v.label, w, sec, sec / songSeconds, peak / 1e6, thermal()))
        }
        OptNative.nSetCompute(0, true)

        // 3. The analysis the app runs after separation, on those stems.
        analysis(work, song, stems, songSeconds)
        out("\nDone. Tap Copy and paste the results to Claude.")
    }

    /**
     * Section 3: the beat grid (Beat This! on ONNX Runtime) on the drums stem
     * with ONNX Runtime's CPU provider at a few thread counts, XNNPACK and
     * NNAPI (whether they are faster, and whether they find the same beats),
     * and the key from the harmony stems.
     */
    private fun analysis(work: File, song: String, kept: File?, songSeconds: Int) {
        out("\n3. Analysis after separation, on the ${songSeconds} s test song's stems")
        out("  ONNX Runtime ${AnalysisNative.nOrtVersion()}")
        val stems = kept ?: File(work, "analysis").apply {
            mkdirs()
            OptNative.nSetCompute(0, true)
            val js = JSONObject(OptNative.separate(ModelManager(this@BenchActivity).modelFile.path, song, path, 2, 24))
            if (js.has("error")) { out("  could not separate: ${js.getString("error")}"); return }
        }
        val beatModels = ModelManager.beatThis(this)
        if (!beatModels.isReady()) out("  Downloading the beat model (82 MB, first run only)…")
        val beatModel = beatModels.ensure { true }.path
        val drums = File(stems, "drums.wav").path
        var reference: String? = null
        val variants = listOf(
            "CPU, default threads" to (0 to ""), "CPU, 2 threads" to (2 to ""), "CPU, 4 threads" to (4 to ""),
            "CPU, 6 threads" to (6 to ""), "CPU, 8 threads" to (8 to ""),
            "XNNPACK, 4 threads" to (4 to "XNNPACK"), "NNAPI" to (0 to "NNAPI"),
        )
        for ((label, cfg) in variants) {
            val rss = PeakRss()
            val js = JSONObject(AnalysisNative.nBeatGrid(drums, beatModel, cfg.first, cfg.second))
            val peak = rss.finish()
            if (js.has("error")) { out("  beat grid, %-22s failed: %s".format(label, js.getString("error"))); continue }
            val beats = js.getJSONArray("beats").toString()
            if (reference == null) reference = beats
            val same = if (beats == reference) "same beats" else "DIFFERENT beats"
            out("  beat grid, %-22s %6.2f s = %.3fx song length (model %.2f s, log-mel %.2f s, decode %.2f s, clean-up %.2f s)  %s  peak memory %.2f GB  thermal %s".format(
                label, js.getDouble("seconds"), js.getDouble("seconds") / songSeconds, js.getDouble("model"), js.getDouble("mel"),
                js.getDouble("load"), js.getDouble("grid"), if (js.getBoolean("ok")) "%d beats, %.1f BPM, %s".format(js.getJSONArray("beats").length(), js.getDouble("bpm"), same) else "no grid (${js.getString("why")})",
                peak / 1e6, thermal()))
        }
        val key = JSONObject(AnalysisNative.nKey(stems.path))
        if (key.has("error")) out("  key: failed: ${key.getString("error")}")
        else out("  key: %.2f s = %.3fx song length (%s)".format(key.getDouble("seconds"), key.getDouble("seconds") / songSeconds, key.getString("key")))

        // Song sections (1.2): All-In-One, eight folds, on bass/drums/other+guitar+piano/vocals.
        val folds = ModelManager.sections(this)
        if (folds.any { !it.isReady() }) out("  Downloading the section model (25 MB, first run only)…")
        val foldPaths = folds.map { it.ensure { true }.path }.toTypedArray()
        for ((label, threads) in listOf("default threads" to 0, "4 threads" to 4, "8 threads" to 8)) {
            val rss = PeakRss()
            val js = JSONObject(AnalysisNative.nSections(stems.path, foldPaths, threads))
            val peak = rss.finish()
            if (js.has("error")) { out("  sections, %-16s failed: %s".format(label, js.getString("error"))); continue }
            out("  sections, %-16s %6.2f s = %.3fx song length (eight folds %.2f s, spectrogram %.2f s)  %d sections  peak memory %.2f GB  thermal %s".format(
                label, js.getDouble("seconds"), js.getDouble("seconds") / songSeconds, js.getDouble("model"), js.getDouble("spec"),
                js.getInt("count"), peak / 1e6, thermal()))
        }

        // Lead/backing vocal split (1.2): UVR-MDX-NET Karaoke 2 on the vocals stem.
        val kara = ModelManager.karaoke(this)
        if (!kara.isReady()) out("  Downloading the vocal split model (53 MB, first run only)…")
        val karaPath = kara.ensure { true }.path
        val splitOut = File(work, "split").apply { mkdirs() }
        run {
            val rss = PeakRss()
            val js = JSONObject(AnalysisNative.nVocalSplit(File(stems, "vocals.wav").path, splitOut.path, karaPath, 0))
            val peak = rss.finish()
            if (js.has("error")) out("  vocal split failed: ${js.getString("error")}")
            else out("  vocal split: %.2f s = %.3fx song length (model %.2f s)%s  peak memory %.2f GB  thermal %s".format(
                js.getDouble("seconds"), js.getDouble("seconds") / songSeconds, js.getDouble("model"),
                if (js.getBoolean("ok")) "" else " (no split: ${js.getString("why")})", peak / 1e6, thermal()))
        }
        splitOut.deleteRecursively()
        stems.deleteRecursively()
    }

    private fun table(rows: JSONArray, totalMs: Double, nameIdx: Int, msIdx: Int, callsIdx: Int) {
        val items = (0 until rows.length()).map { rows.getJSONArray(it) }
        // Merge stages with the same name (e.g. repeated progress messages).
        val merged = LinkedHashMap<String, Pair<Double, Long>>()
        for (r in items) {
            val n = r.getString(nameIdx)
            val ms = r.getDouble(msIdx)
            val calls = if (callsIdx >= 0) r.getLong(callsIdx) else 1L
            val p = merged[n]
            merged[n] = if (p == null) ms to calls else (p.first + ms) to (p.second + calls)
        }
        var sum = 0.0
        for ((n, p) in merged.entries.sortedByDescending { it.value.first }) {
            sum += p.first
            val calls = if (callsIdx >= 0) "  %5d×".format(p.second) else ""
            out("   %5.1f%%  %8.0f ms%s  %s".format(100 * p.first / totalMs, p.first, calls, n.take(48)))
        }
        if (callsIdx >= 0) out("   %5.1f%%  %8.0f ms          (not in the list above)".format(100 * (totalMs - sum) / totalMs, totalMs - sum))
    }
}

private object BuildConfigInfo {
    fun version(a: Activity): String = try {
        val pi = a.packageManager.getPackageInfo(a.packageName, 0)
        "${pi.versionName} (build ${if (Build.VERSION.SDK_INT >= 28) pi.longVersionCode else 0})"
    } catch (_: Exception) { "?" }
}
