package org.stemdeckport.compare

import android.app.Activity
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import kotlin.math.max

/**
 * Listen to the stems the engine builds produced: pick a stem, play it, and
 * tap between builds while it plays. The switch keeps the position, so you
 * hear the same moment from each build. You can also play only the
 * difference from the reference build, jump to where two builds differ most,
 * loop a few seconds, and hide the build names for a blind test.
 */
class ListenActivity : Activity() {
    companion object {
        const val RUN_INFO = "run.json"
        val STEMS = listOf("drums", "bass", "other", "vocals", "guitar", "piano")
        private const val ALL = "all"
        private const val ORIGINAL = "original"

        fun workDir(c: Context) = File(c.getExternalFilesDir(null) ?: c.filesDir, "compare")

        fun hasResults(c: Context) = File(workDir(c), RUN_INFO).isFile
    }

    private data class Build(val id: String, val label: String, val description: String, val dir: File)

    private val fg = Color.parseColor("#eaeaec")
    private val dim = Color.parseColor("#8a8a92")
    private val accent = Color.parseColor("#4a9bf5")

    private lateinit var work: File
    private lateinit var builds: List<Build>
    private lateinit var refId: String
    private var diffs = JSONObject()
    private lateinit var player: StemPlayer

    // What is playing.
    private var stem = "vocals"
    private var buildIdx = 0
    private var diffMode = false
    private var boost = false
    private var blind = false
    private var revealed = false
    private var blindOrder: List<Int> = emptyList()

    private val sources = HashMap<String, StemPlayer.Source>()
    private val closeables = mutableListOf<java.io.Closeable>()

    private lateinit var controls: LinearLayout
    private lateinit var seek: SeekBar
    private lateinit var timeLabel: TextView
    private lateinit var playBtn: Button
    private lateinit var loopBox: CheckBox
    private var dragging = false
    private var shownPlaying = false
    private val ui = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        work = workDir(this)
        val info = try {
            JSONObject(File(work, RUN_INFO).readText())
        } catch (e: Exception) {
            Toast.makeText(this, "Run a comparison first.", Toast.LENGTH_LONG).show()
            finish(); return
        }
        val engines = info.getJSONArray("engines")
        builds = (0 until engines.length()).map { i ->
            val e = engines.getJSONObject(i)
            Build(e.getString("id"), e.getString("label"), e.optString("description"), File(work, e.getString("id")))
        }.filter { File(it.dir, "vocals.wav").isFile }
        if (builds.isEmpty()) {
            Toast.makeText(this, "The stems of the last run are gone. Run it again.", Toast.LENGTH_LONG).show()
            finish(); return
        }
        refId = info.optString("ref", builds[0].id)
        diffs = info.optJSONObject("diffs") ?: JSONObject()
        blindOrder = builds.indices.toList()

        val prepared = File(work, "prepared.f32")
        val frames = if (prepared.isFile) prepared.length() / 8
        else builds.maxOf { b -> STEMS.maxOf { max(0L, (File(b.dir, "$it.wav").length() - 44) / 4) } }
        player = StemPlayer(frames)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0b0b0d"))
            setPadding(32, 48, 32, 24)
            fitsSystemWindows = true
        }
        root.addView(TextView(this).apply {
            text = "Listen"
            setTextColor(accent); textSize = 22f; typeface = Typeface.DEFAULT_BOLD
        })
        root.addView(label(info.optString("song", ""), 13f, dim))

        controls = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(ScrollView(this).apply { addView(controls) },
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        // Transport, always visible at the bottom.
        timeLabel = label("0:00 / ${fmt(frames)}", 13f, fg)
        root.addView(timeLabel)
        seek = SeekBar(this).apply {
            max = 1000
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                    if (fromUser) timeLabel.text = "${fmt(frames * p / 1000)} / ${fmt(frames)}"
                }
                override fun onStartTrackingTouch(sb: SeekBar) { dragging = true }
                override fun onStopTrackingTouch(sb: SeekBar) {
                    dragging = false
                    player.loop = null; showLoop(false)
                    player.seek(frames * sb.progress / 1000)
                }
            })
        }
        root.addView(seek)
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row.addView(Button(this).apply { text = "−5 s"; setOnClickListener { nudge(-5) } })
        playBtn = Button(this).apply { text = "Play"; setOnClickListener { togglePlay() } }
        row.addView(playBtn)
        row.addView(Button(this).apply { text = "+5 s"; setOnClickListener { nudge(5) } })
        loopBox = CheckBox(this).apply {
            text = "Loop 5 s"; setTextColor(fg)
            setOnCheckedChangeListener { _, on -> onLoopToggled(on) }
        }
        row.addView(loopBox)
        root.addView(row)
        setContentView(root)

        render()
        applySource()
        tick()
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacksAndMessages(null)
        if (::player.isInitialized) player.release()
        closeables.forEach { runCatching { it.close() } }
    }

    // ─── what plays ───

    private fun build(i: Int) = builds[i]
    private val ref: Build get() = builds.firstOrNull { it.id == refId } ?: builds[0]

    private fun wav(b: Build, s: String): StemPlayer.Source = sources.getOrPut("${b.id}|$s") {
        StemPlayer.WavSource(File(b.dir, "$s.wav")).also { closeables += it }
    }

    private fun stemSource(b: Build, s: String): StemPlayer.Source = when (s) {
        ALL -> sources.getOrPut("${b.id}|all") { StemPlayer.SumSource(STEMS.map { wav(b, it) }) }
        else -> wav(b, s)
    }

    private fun currentSource(): StemPlayer.Source {
        if (stem == ORIGINAL) return sources.getOrPut("original") {
            StemPlayer.F32Source(File(work, "prepared.f32")).also { closeables += it }
        }
        val b = build(buildIdx)
        if (!diffMode) return stemSource(b, stem)
        val gain = if (boost) 4f else 1f
        return sources.getOrPut("diff|${b.id}|$stem|$gain") {
            StemPlayer.DiffSource(stemSource(b, stem), stemSource(ref, stem), gain)
        }
    }

    private fun applySource() {
        player.source = currentSource()
    }

    private fun togglePlay() {
        if (player.playing) player.pause() else player.play()
        updatePlayState()
    }

    private fun onLoopToggled(on: Boolean) {
        player.loop = if (on) {
            val p = player.position
            p until minOf(player.duration, p + 5L * StemPlayer.RATE)
        } else null
    }

    /** Ticks the Loop box without starting a new 5-second loop. */
    private fun showLoop(on: Boolean) {
        loopBox.setOnCheckedChangeListener(null)
        loopBox.isChecked = on
        loopBox.setOnCheckedChangeListener { _, v -> onLoopToggled(v) }
    }

    private fun updatePlayState() {
        shownPlaying = player.playing
        playBtn.text = if (player.playing) "Pause" else "Play"
        if (player.playing) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun nudge(seconds: Int) {
        player.loop = null; showLoop(false)
        player.seek(player.position + seconds.toLong() * StemPlayer.RATE)
    }

    private fun tick() {
        val pos = player.position
        if (!dragging) {
            seek.progress = if (player.duration > 0) (pos * 1000 / player.duration).toInt() else 0
            timeLabel.text = "${fmt(pos)} / ${fmt(player.duration)}"
        }
        if (shownPlaying != player.playing) updatePlayState()
        ui.postDelayed(::tick, 100)
    }

    /** Jumps to where [b] differs most from the reference in the current stem, and loops it. */
    private fun jumpTo(second: Long, b: Int?, spotStem: String) {
        if (b != null && !blind) buildIdx = b
        if (stem == ALL) stem = spotStem
        val start = max(0L, second - 1) * StemPlayer.RATE
        val end = minOf(player.duration, start + 4L * StemPlayer.RATE)
        player.seek(start)
        player.loop = start until end
        showLoop(true)
        applySource()
        render()
        if (!player.playing) togglePlay()
    }

    // ─── the controls ───

    private fun render() {
        controls.removeAllViews()

        controls.addView(heading("Stem"))
        val stemNames = STEMS + listOf(ALL) + if (File(work, "prepared.f32").isFile) listOf(ORIGINAL) else emptyList()
        stemNames.chunked(4).forEach { chunk ->
            val r = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            chunk.forEach { s ->
                val name = when (s) { ALL -> "All stems"; ORIGINAL -> "Original"; else -> s.replaceFirstChar(Char::uppercase) }
                r.addView(pill(name, stem == s) { stem = s; applySource(); render() }, weighted())
            }
            controls.addView(r)
        }

        if (stem == ORIGINAL) {
            controls.addView(label("The song as it went into every build. Pick a stem to compare the builds.", 12f, dim).apply { setPadding(0, 16, 0, 0) })
            return
        }

        val showNames = !blind || revealed
        controls.addView(heading(if (blind) "Build (blind test)" else "Build"))
        val order = if (blind) blindOrder else builds.indices.toList()
        order.forEachIndexed { pos, i ->
            val b = build(i)
            val title = if (showNames) b.label + if (b.id == refId) "  · reference" else "" else "Build ${'A' + pos}"
            val sub = when {
                !showNames -> ""
                b.id == refId -> b.description
                else -> {
                    val snr = stemSnr(b.id, stem)
                    (if (snr != null) "%.1f dB from the reference · ".format(snr) else "") + b.description
                }
            }
            controls.addView(pill(twoLines(title, sub), buildIdx == i) { buildIdx = i; applySource(); render() },
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = 10 })
        }

        controls.addView(heading("Hear"))
        val hear = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        hear.addView(pill("The stem", !diffMode) { diffMode = false; applySource(); render() }, weighted())
        hear.addView(pill(if (showNames) "Only the difference from ${ref.label}" else "Only the difference from the reference", diffMode) {
            diffMode = true; applySource(); render()
        }, weighted(2f))
        controls.addView(hear)
        if (diffMode) {
            controls.addView(CheckBox(this).apply {
                text = "Boost the difference ×4 (+12 dB)"; setTextColor(fg); isChecked = boost
                setOnCheckedChangeListener { _, on -> boost = on; applySource() }
            })
            controls.addView(label(
                "You hear this build minus the reference. Silence means they are the same here." +
                    if (build(buildIdx).id == refId) " (This is the reference itself, so it is silent.)" else "",
                12f, dim))
        }

        val spots = differSpots()
        if (spots.isNotEmpty()) {
            controls.addView(heading("Where they differ most (${if (stem == ALL) "any stem" else stem})"))
            for ((b, s, second, db) in spots) {
                val who = if (showNames) build(b).label else "A build"
                val what = if (stem == ALL) " in $s" else ""
                controls.addView(pill(twoLines("$who$what at ${fmtSec(second)}", "%.1f dB in that second · tap to loop it".format(db)), false) {
                    jumpTo(second, b, s)
                }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = 10 })
            }
        }

        controls.addView(heading("Blind test"))
        val br = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        br.addView(CheckBox(this).apply {
            text = "Hide the build names"; setTextColor(fg); isChecked = blind
            setOnCheckedChangeListener { _, on ->
                blind = on; revealed = false
                if (on) { blindOrder = builds.indices.shuffled(); buildIdx = blindOrder[0] }
                applySource(); render()
            }
        })
        if (blind && !revealed) br.addView(Button(this).apply { text = "Reveal"; setOnClickListener { revealed = true; render() } })
        controls.addView(br)
        if (blind) controls.addView(label("The builds are shuffled and named A, B, C. Pick the one you prefer, then tap Reveal.", 12f, dim))
    }

    private data class Spot(val build: Int, val stem: String, val second: Long, val db: Double)

    /** Where each build differs most from the reference, for the current stem (or any stem). */
    private fun differSpots(): List<Spot> {
        val out = mutableListOf<Spot>()
        for ((i, b) in builds.withIndex()) {
            if (b.id == refId) continue
            val d = diffs.optJSONObject(b.id) ?: continue
            val arr = d.optJSONArray("stems") ?: continue
            var best: Spot? = null
            for (k in 0 until arr.length()) {
                val o = arr.getJSONObject(k)
                if (stem != ALL && o.optString("name") != stem) continue
                val at = o.optLong("worst_at", -1)
                if (at < 0) continue
                val spot = Spot(i, o.optString("name"), at, o.optDouble("worst_db"))
                if (best == null || spot.db < best.db) best = spot
            }
            best?.let { out += it }
        }
        return out.sortedBy { it.db }
    }

    private fun stemSnr(id: String, s: String): Double? {
        val d = diffs.optJSONObject(id) ?: return null
        if (s == ALL) return d.optDouble("all_snr_db").takeUnless { it.isNaN() }
        val arr = d.optJSONArray("stems") ?: return null
        for (k in 0 until arr.length()) {
            val o = arr.getJSONObject(k)
            if (o.optString("name") == s) return o.optDouble("snr_db").takeUnless { it.isNaN() }
        }
        return null
    }

    // ─── small view helpers ───

    private fun label(t: String, size: Float, color: Int) =
        TextView(this).apply { text = t; setTextColor(color); textSize = size }

    private fun heading(t: String) = label(t.uppercase(), 11f, dim).apply {
        letterSpacing = 0.08f
        setPadding(0, 36, 0, 10)
    }

    private fun weighted(w: Float = 1f) =
        LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, w).apply { marginEnd = 10; topMargin = 10 }

    private fun twoLines(title: String, sub: String): CharSequence {
        val sb = SpannableStringBuilder(title)
        sb.setSpan(StyleSpan(Typeface.BOLD), 0, title.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        if (sub.isNotEmpty()) {
            val start = sb.length
            sb.append("\n").append(sub)
            sb.setSpan(RelativeSizeSpan(0.82f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            sb.setSpan(ForegroundColorSpan(dim), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        return sb
    }

    private fun pill(text: CharSequence, on: Boolean, onClick: () -> Unit): View = TextView(this).apply {
        this.text = text
        setTextColor(fg)
        textSize = 14f
        gravity = Gravity.CENTER_VERTICAL
        setPadding(28, 22, 28, 22)
        background = GradientDrawable().apply {
            cornerRadius = 28f
            if (on) {
                setColor(Color.parseColor("#1d3550"))
                setStroke(3, accent)
            } else {
                setColor(Color.TRANSPARENT)
                setStroke(2, Color.parseColor("#33ffffff"))
            }
        }
        isClickable = true
        setOnClickListener { onClick() }
    }

    private fun fmt(frames: Long) = fmtSec(frames / StemPlayer.RATE)

    private fun fmtSec(s: Long) = "%d:%02d".format(s / 60, s % 60)
}
