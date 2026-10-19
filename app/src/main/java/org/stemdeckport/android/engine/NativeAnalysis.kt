package org.stemdeckport.android.engine

import org.json.JSONObject
import java.io.File

/**
 * The analysis stages after separation, in C++ (/analysis): desktop
 * StemDeck's beat grid and key detection, ported and checked against it
 * (scripts/analysis_parity.py), and the click renderer for exports. The
 * models run on ONNX Runtime (the onnxruntime-android library, loaded by
 * the native code).
 */
object NativeAnalysis {
    init {
        System.loadLibrary("sdengine")
    }

    @JvmStatic private external fun nativeBeatGrid(
        source: String, label: String, model: String, threads: Int, listener: ProgressListener?,
    ): String
    @JvmStatic private external fun nativeKey(harmony: Array<String>, weights: DoubleArray, bass: String): String
    @JvmStatic private external fun nativeRenderClick(
        beats: DoubleArray, barBeats: IntArray, barLengths: IntArray, duration: Double, sampleRate: Int,
        multiplier: Double, accent: Int, out: String,
    ): Boolean
    @JvmStatic private external fun nativeOrtVersion(): String

    class BeatGrid(val json: JSONObject, val tempoStability: Int?) {
        val bpm: Double get() = json.optDouble("bpm")
    }

    /**
     * The beat grid of [source] (a 16-bit WAV stem), or null with [why] set
     * when the song has none. Throws CancellationException when cancelled,
     * IllegalStateException when the model cannot run.
     */
    fun beatGrid(source: File, label: String, model: File, threads: Int, listener: ProgressListener, why: StringBuilder): BeatGrid? {
        val r = JSONObject(nativeBeatGrid(source.path, label, model.path, threads, listener))
        if (!r.optBoolean("ok")) {
            why.append(r.optString("why", "no beat grid"))
            return null
        }
        val ts = r.optInt("tempo_stability", -1)
        return BeatGrid(r.getJSONObject("grid"), ts.takeIf { it >= 0 })
    }

    data class Key(val key: String, val scale: String, val confidence: Int)

    /** Key from the harmony stems (with weights) and the bass stem, or null for silence. */
    fun key(harmony: List<Pair<File, Double>>, bass: File?): Key? {
        val s = nativeKey(
            harmony.map { it.first.path }.toTypedArray(),
            harmony.map { it.second }.toDoubleArray(),
            bass?.path ?: "",
        )
        if (s.isEmpty()) return null
        val o = JSONObject(s)
        return Key(o.getString("key"), o.getString("scale"), o.getInt("confidence"))
    }

    /**
     * Writes the click track of a beat grid (beats.json) as a stereo 16-bit
     * WAV at 44.1 kHz, as desktop's click_render.py renders it. Returns false
     * when there is nothing to render.
     */
    fun renderClick(grid: JSONObject, multiplier: Double, accent: Int, out: File): Boolean {
        val beats = grid.optJSONArray("beats") ?: return false
        val bars = grid.optJSONArray("bars")
        val n = bars?.length() ?: 0
        val at = IntArray(n) { bars!!.getJSONObject(it).optInt("beat") }
        val len = IntArray(n) { bars!!.getJSONObject(it).optInt("beats_per_bar") }
        return nativeRenderClick(
            DoubleArray(beats.length()) { beats.getDouble(it) }, at, len, grid.optDouble("duration", 0.0),
            44100, multiplier, accent, out.path,
        )
    }

    /** ONNX Runtime's version, or why it cannot be loaded. */
    fun runtimeVersion(): String = nativeOrtVersion()
}
