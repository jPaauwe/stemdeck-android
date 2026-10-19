package org.stemdeckport.android.data

import java.io.File
import java.io.OutputStream
import java.util.zip.CRC32
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * "Export stems": the chosen stems in one ZIP, named after the song, as
 * desktop's GET /api/jobs/{id}/stems/all.zip builds it (app/api/stems.py).
 * The stems go in as they are stored (320 kbps MP3 or WAV), STORED rather
 * than deflated: audio barely compresses, and storing keeps it fast.
 */
object StemsZip {
    /** Desktop's STEM_NAMES then EXTRA_STEM_NAMES: the order entries are written in. */
    val ORDER = listOf("vocals", "drums", "bass", "guitar", "piano", "other", "lead_vocals", "backing_vocals")
    private val VOCAL_PARTS = setOf("lead_vocals", "backing_vocals")

    /** Stems to export, or an error message for desktop's 422 cases. */
    fun resolve(job: Job, requested: String?): Result<List<String>> {
        if (requested.isNullOrBlank()) return Result.success(ORDER.filter { it in job.stems })
        val asked = requested.split(',').filter { it.isNotEmpty() }.toSet()
        if (!ORDER.containsAll(asked)) return Result.failure(IllegalArgumentException("unknown stem requested"))
        if ("vocals" in asked && asked.any { it in VOCAL_PARTS }) {
            return Result.failure(IllegalArgumentException("vocals cannot be combined with lead_vocals/backing_vocals (same signal)"))
        }
        return Result.success(ORDER.filter { it in asked && it in job.stems })
    }

    /** _title_slug: [A-Za-z0-9_] only, so it is also a safe ZIP member name. */
    fun titleSlug(title: String?): String {
        val safe = Regex("[^a-zA-Z0-9]+").replace(title ?: "", "_")
        return Regex("_{2,}").replace(safe, "_").trim('_').take(80).trim('_')
    }

    fun fileName(job: Job) = "${titleSlug(job.title).ifEmpty { "stems" }}_stems.zip"

    fun entryName(prefix: String, name: String, ext: String) = if (prefix.isNotEmpty()) "${prefix}_$name.$ext" else "$name.$ext"

    /** Writes the ZIP to [out]. [keepGoing] is polled between blocks. */
    fun write(files: List<Pair<String, File>>, prefix: String, out: OutputStream, keepGoing: () -> Boolean = { true }) {
        val zip = ZipOutputStream(out)
        zip.setMethod(ZipOutputStream.STORED)
        val buf = ByteArray(1 shl 16)
        for ((name, f) in files) {
            val crc = CRC32()
            f.inputStream().use { input ->
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    crc.update(buf, 0, n)
                }
            }
            val entry = ZipEntry(entryName(prefix, name, f.extension)).apply {
                method = ZipEntry.STORED
                size = f.length()
                compressedSize = f.length()
                this.crc = crc.value
                time = f.lastModified()
            }
            zip.putNextEntry(entry)
            f.inputStream().use { input ->
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    zip.write(buf, 0, n)
                    if (!keepGoing()) throw java.util.concurrent.CancellationException("stopped")
                }
            }
            zip.closeEntry()
        }
        zip.finish()
        out.flush()
    }
}
