package org.stemdeckport.android.data

import org.json.JSONObject
import org.stemdeckport.android.engine.Mp3Codec
import org.stemdeckport.android.engine.NativeAnalysis
import java.io.File
import java.io.OutputStream
import kotlin.math.sqrt

/** "Export mix": the chosen stems at the chosen gains, as MP3 (320 kbps) or WAV. */
object MixExport {
    fun mimeType(format: String) = if (format == "mp3") "audio/mpeg" else "audio/wav"

    /**
     * The click track mixed into an export, as desktop's ?click=1 does it:
     * rendered from the song's beat grid at the rate and accent the user was
     * monitoring with, at [gain]. Desktop's ffmpeg graph spreads the mono
     * click over both channels at -3 dB; so does this.
     */
    class Click(val grid: JSONObject, val multiplier: Double, val accent: Int, val gain: Float)

    /**
     * Writes the mix to [out]. MP3 stems are decoded first (sample-aligned),
     * so a mix of MP3 stems lines up exactly. [keepGoing] is polled during
     * the slow steps; when it returns false the export stops with a
     * CancellationException.
     */
    fun write(
        store: JobStore, job: Job, names: List<String>, gains: List<Float>, format: String,
        tmpRoot: File, out: OutputStream, click: Click? = null, keepGoing: () -> Boolean = { true },
    ) {
        val tmp = File(tmpRoot, "mix-${System.nanoTime()}").apply { mkdirs() }
        try {
            val wavs = names.map { name ->
                val f = store.stemFile(job.id, name)
                if (!f.isFile) throw IllegalStateException("The $name stem is missing.")
                if (f.extension == "mp3") {
                    File(tmp, "$name.wav").also { Mp3Codec.decode(f, it) { keepGoing() } }
                } else f
            }.toMutableList()
            val allGains = gains.toMutableList()
            if (click != null) {
                val wav = File(tmp, "click.wav")
                if (NativeAnalysis.renderClick(click.grid, click.multiplier, click.accent, wav)) {
                    wavs += wav
                    allGains += click.gain * sqrt(0.5f)
                }
            }
            if (format == "mp3") {
                val wav = File(tmp, "mix.wav")
                wav.outputStream().buffered(1 shl 16).use { Mixdown.write(wavs, allGains, it) }
                val mp3 = File(tmp, "mix.mp3")
                Mp3Codec.encode(wav, mp3) { keepGoing() }
                mp3.inputStream().use { it.copyTo(out, 1 shl 16) }
            } else {
                Mixdown.write(wavs, allGains, out)
            }
            out.flush()
        } finally {
            tmp.deleteRecursively()
        }
    }

    /** The click options a mixdown request asks for (desktop's query parameters), or null. */
    fun clickFrom(store: JobStore, job: Job, param: (String) -> String?): Click? {
        if (param("click") !in setOf("1", "true")) return null
        val f = File(store.stemsDir(job.id), "beats.json")
        if (!f.isFile) return null
        val grid = try { JSONObject(f.readText()) } catch (e: Exception) { return null }
        val mult = param("click_mult")?.toDoubleOrNull()?.takeIf { it == 0.5 || it == 1.0 || it == 2.0 } ?: 1.0
        val accent = param("click_accent")?.toIntOrNull()?.coerceIn(-1, 32) ?: -1
        val gain = param("click_gain")?.toFloatOrNull()?.coerceIn(0f, 4f) ?: 0.6f
        return Click(grid, mult, accent, gain)
    }
}
