package org.stemdeckport.android

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.stemdeckport.android.data.Job
import org.stemdeckport.android.data.StemsZip
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import kotlin.random.Random

class StemsZipTest {
    private fun job(stems: List<String>, title: String = "Plug In Baby (Live)") =
        Job("a1b2c3d4e5f6", title, "local:x", stems, 0.0).apply { this.stems = stems; format = "mp3"; status = "done" }

    @Test
    fun entries_are_stored_byte_for_byte_and_named_after_the_song() {
        val dir = kotlin.io.path.createTempDirectory("sdtest").toFile()
        val rng = Random(7)
        val files = listOf("vocals", "drums", "bass").map { n ->
            n to File(dir, "$n.mp3").apply { writeBytes(rng.nextBytes(100_000 + rng.nextInt(5000))) }
        }
        val out = ByteArrayOutputStream()
        StemsZip.write(files, StemsZip.titleSlug("Plug In Baby (Live)"), out)
        val seen = mutableListOf<String>()
        ZipInputStream(out.toByteArray().inputStream()).use { zin ->
            while (true) {
                val e = zin.nextEntry ?: break
                assertEquals(ZipEntry.STORED, e.method)
                val bytes = zin.readBytes()
                val name = e.name.removePrefix("Plug_In_Baby_Live_").removeSuffix(".mp3")
                // Byte-identical, so the LAME gapless header survives and the
                // stems still decode sample-aligned.
                assertArrayEquals(File(dir, "$name.mp3").readBytes(), bytes)
                seen += e.name
            }
        }
        assertEquals(listOf("Plug_In_Baby_Live_vocals.mp3", "Plug_In_Baby_Live_drums.mp3", "Plug_In_Baby_Live_bass.mp3"), seen)
        dir.deleteRecursively()
    }

    @Test
    fun title_slug_matches_desktop() {
        assertEquals("Plug_In_Baby_Live", StemsZip.titleSlug("Plug In Baby (Live)"))
        assertEquals("", StemsZip.titleSlug("!!!"))
        assertEquals("stems_stems.zip", StemsZip.fileName(job(listOf("drums"), "!!!")))
        assertEquals("a".repeat(80), StemsZip.titleSlug("a".repeat(120)))
        assertEquals("Bj_rk_J_ga", StemsZip.titleSlug("Björk — Jóga"))
    }

    @Test
    fun subset_order_and_the_vocals_rule() {
        val j = job(listOf("drums", "bass", "other", "vocals"))
        assertEquals(listOf("vocals", "drums", "bass", "other"), StemsZip.resolve(j, null).getOrThrow())
        assertEquals(listOf("drums", "other"), StemsZip.resolve(j, "other,drums").getOrThrow())
        assertTrue(StemsZip.resolve(j, "vocals,lead_vocals").isFailure)
        assertTrue(StemsZip.resolve(j, "kazoo").isFailure)
    }
}
