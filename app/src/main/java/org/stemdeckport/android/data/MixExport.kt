package org.stemdeckport.android.data

import org.stemdeckport.android.engine.Mp3Codec
import java.io.File
import java.io.OutputStream

/** "Export mix": the chosen stems at the chosen gains, as MP3 (320 kbps) or WAV. */
object MixExport {
    fun mimeType(format: String) = if (format == "mp3") "audio/mpeg" else "audio/wav"

    /**
     * Writes the mix to [out]. MP3 stems are decoded first (sample-aligned),
     * so a mix of MP3 stems lines up exactly. [keepGoing] is polled during
     * the slow steps; when it returns false the export stops with a
     * CancellationException.
     */
    fun write(
        store: JobStore, job: Job, names: List<String>, gains: List<Float>, format: String,
        tmpRoot: File, out: OutputStream, keepGoing: () -> Boolean = { true },
    ) {
        val tmp = File(tmpRoot, "mix-${System.nanoTime()}").apply { mkdirs() }
        try {
            val wavs = names.map { name ->
                val f = store.stemFile(job.id, name)
                if (!f.isFile) throw IllegalStateException("The $name stem is missing.")
                if (f.extension == "mp3") {
                    File(tmp, "$name.wav").also { Mp3Codec.decode(f, it) { keepGoing() } }
                } else f
            }
            if (format == "mp3") {
                val wav = File(tmp, "mix.wav")
                wav.outputStream().buffered(1 shl 16).use { Mixdown.write(wavs, gains, it) }
                val mp3 = File(tmp, "mix.mp3")
                Mp3Codec.encode(wav, mp3) { keepGoing() }
                mp3.inputStream().use { it.copyTo(out, 1 shl 16) }
            } else {
                Mixdown.write(wavs, gains, out)
            }
            out.flush()
        } finally {
            tmp.deleteRecursively()
        }
    }
}
