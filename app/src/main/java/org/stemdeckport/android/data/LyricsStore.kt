package org.stemdeckport.android.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * A song's lyrics on disk, as desktop keeps them: lyrics.json in the song's
 * folder (the version kept, with "timing", "others" and the user's
 * "offset_sec"), or lyrics_candidates.json when LRCLIB had only versions to
 * offer. GET /api/jobs/{id}/lyrics answers from these.
 */
class LyricsStore(private val store: JobStore) {
    fun lyricsFile(id: String) = File(store.jobDir(id), "lyrics.json")
    private fun candidatesFile(id: String) = File(store.jobDir(id), "lyrics_candidates.json")

    /** What the lookup sends: the user's own entry, else the file's tags, else "Artist - Title" from the file name. */
    fun query(job: Job): LrcLib.Query? {
        val edit = job.lyricsQuery
        val tags = job.audioTags
        val (artist, track, album) = when {
            edit != null -> Triple(edit.optString("artist").trim(), edit.optString("title").trim(), edit.optString("album").trim())
            !tags["title"].isNullOrBlank() -> {
                val a = tags["artist"]?.trim().orEmpty()
                Triple(a, LrcLib.songFromTitle(tags["title"]!!, a).ifEmpty { tags["title"]!!.trim() }, tags["album"]?.trim().orEmpty())
            }
            else -> {
                val parts = job.title.split(Regex("\\s+\\p{Pd}\\s+"), limit = 2)
                if (parts.size == 2) Triple(parts[0].trim(), LrcLib.songFromTitle(parts[1], parts[0]), "")
                else Triple("", LrcLib.songFromTitle(job.title), "")
            }
        }
        if (track.isBlank()) return null
        return LrcLib.Query(artist, track, album, job.duration ?: 0.0)
    }

    /** Keeps a lookup's answer; true when the song now has lyrics. */
    fun keep(id: String, answer: LrcLib.Answer): Boolean {
        val entry = answer.entry
        if (entry != null) {
            writeAtomic(lyricsFile(id), entry)
            candidatesFile(id).delete()
            store.update(id) { it.hasLyrics = true }
            return true
        }
        writeAtomic(candidatesFile(id), JSONObject().put("others", JSONArray(answer.others)).put("asked_at", System.currentTimeMillis() / 1000.0))
        return false
    }

    /** lyrics.json with its others (or the candidates as others), or null. */
    fun read(id: String): JSONObject? = try {
        lyricsFile(id).takeIf { it.isFile }?.let { JSONObject(it.readText()) }?.takeIf { it.optInt("v") == 1 }
    } catch (e: Exception) {
        null
    }

    fun candidates(id: String): JSONArray = try {
        candidatesFile(id).takeIf { it.isFile }?.let { JSONObject(it.readText()).optJSONArray("others") } ?: JSONArray()
    } catch (e: Exception) {
        JSONArray()
    }

    fun setOffset(id: String, offset: Double): Double? {
        val entry = read(id) ?: return null
        val clean = LrcLib.cleanOffset(offset) ?: return null
        if (clean == 0.0) entry.remove("offset_sec") else entry.put("offset_sec", clean)
        writeAtomic(lyricsFile(id), entry)
        return clean
    }

    /** Makes another version (from "others") the kept one; the old one joins the others. */
    fun choose(id: String, lrclibId: Long): JSONObject? {
        val entry = read(id)
        val pool = ArrayList<JSONObject>()
        val others = entry?.optJSONArray("others") ?: candidates(id)
        for (i in 0 until others.length()) others.optJSONObject(i)?.let { pool.add(it) }
        val pick = pool.firstOrNull { it.optLong("lrclib_id") == lrclibId } ?: return null
        val rest = pool.filter { it.optLong("lrclib_id") != lrclibId }.toMutableList()
        if (entry != null) rest.add(0, JSONObject(entry.toString()).apply { remove("others"); remove("timing"); remove("offset_sec") })
        val next = JSONObject(pick.toString())
            .put("timing", "unverified".takeIf { pick.optString("synced").isNotEmpty() && entry?.optDouble("duration") != pick.optDouble("duration") } ?: "exact")
            .put("others", JSONArray(rest.take(LrcLib.OTHERS_MAX)))
        writeAtomic(lyricsFile(id), next)
        candidatesFile(id).delete()
        store.update(id) { it.hasLyrics = true }
        return next
    }

    private fun writeAtomic(f: File, o: JSONObject) {
        f.parentFile?.mkdirs()
        val tmp = File(f.parentFile, ".${f.name}.tmp")
        tmp.writeText(o.toString())
        if (!tmp.renameTo(f)) {
            f.delete()
            if (!tmp.renameTo(f)) throw java.io.IOException("could not save ${f.name}")
        }
    }
}
