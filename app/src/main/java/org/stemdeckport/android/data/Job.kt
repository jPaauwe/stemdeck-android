package org.stemdeckport.android.data

import org.json.JSONArray
import org.json.JSONObject

/**
 * One song in the library. Field names and JSON shape follow desktop
 * StemDeck's Job.to_state() (app/core/models.py), so the bundled mobile UI
 * reads it unchanged.
 */
class Job(
    val id: String,
    var title: String,
    var sourceUrl: String,
    var selectedStems: List<String>,
    var createdAt: Double,
) {
    var status: String = "queued"
    var progress: Double = 0.0
    var stage: String = "Queued"
    var duration: Double? = null
    var lufs: Double? = null
    var peakDb: Double? = null
    var dynamicRange: Double? = null
    var stemPresence: Map<String, Int>? = null
    var stems: List<String> = emptyList()
    var error: String? = null
    var favorite: Boolean = false
    var trashedAt: Double? = null
    var computeSeconds: Double? = null
    /** File type of the stems: "mp3" or "wav". Songs from before MP3 support are "wav". */
    var format: String = "wav"
    /** What it was separated on, e.g. "10 cores, GPU: …". */
    var computeDevice: String = "phone"
    // Android 1.1: analysis after separation (beat grid, key), as desktop's
    // fields. All null for songs separated before it existed.
    var bpm: Int? = null
    var key: String? = null
    var scale: String? = null
    var keyConfidence: Int? = null
    var tempoStability: Int? = null
    /** Why an analysis stage produced nothing, by stage ("beats", "key"). Android only. */
    var analysisNotes: Map<String, String> = emptyMap()
    /** Android 1.2: automatic song sections, desktop's {id, name, kind, start, end, color} list. */
    var sections: JSONArray? = null
    /** "automatic" (or desktop's "manual", once editing exists). */
    var sectionsSource: String? = null
    /** Whether the sections stage runs for this song, chosen at import (desktop's auto_sections). */
    var autoSections: Boolean = false
    /** Android 1.3: lyrics kept beside the song (lyrics.json), as desktop's has_lyrics. */
    var hasLyrics: Boolean = false
    /** The file's own tags (title, artist, album), as desktop's audio_tags. */
    var audioTags: Map<String, String> = emptyMap()
    /** Artist/title/album the user typed in the Lyrics tab, used for lookups instead of the tags. */
    var lyricsQuery: JSONObject? = null
    /** Desktop's on-demand lead/backing split: "none", "running", "done" or "error". */
    var vocalSplit: String = "none"
    /** Seconds per stage, as desktop records them (#293). */
    var stageTimings: Map<String, Double> = emptyMap()

    /** Bumped on every change; the SSE stream sends a frame when it moves. */
    @Volatile var version: Long = 0

    val finished: Boolean get() = status in FINISHED

    fun stemUrl(name: String) = "/api/jobs/$id/stems/$name.$format"

    fun toState(): JSONObject = JSONObject().apply {
        put("job_id", id)
        put("status", status)
        put("progress", progress)
        put("stage", stage)
        put("title", title)
        put("duration", duration ?: JSONObject.NULL)
        put("thumbnail", JSONObject.NULL)
        put("bpm", bpm ?: JSONObject.NULL)
        put("key", key ?: JSONObject.NULL)
        put("scale", scale ?: JSONObject.NULL)
        put("key_confidence", keyConfidence ?: JSONObject.NULL)
        put("lufs", lufs ?: JSONObject.NULL)
        put("peak_db", peakDb ?: JSONObject.NULL)
        put("dynamic_range", dynamicRange ?: JSONObject.NULL)
        put("tempo_stability", tempoStability ?: JSONObject.NULL)
        put("stem_presence", stemPresence?.let { JSONObject(it as Map<*, *>) } ?: JSONObject.NULL)
        put("stems", JSONArray().apply {
            for (s in stems) put(JSONObject().put("name", s).put("url", stemUrl(s)))
        })
        put("selected_stems", JSONArray(selectedStems))
        put("source_url", sourceUrl)
        put("source_format", JSONObject.NULL)
        put("has_video", false)
        put("has_lyrics", hasLyrics)
        put("audio_tags", JSONObject(audioTags as Map<*, *>))
        put("lyrics_query", lyricsQuery ?: JSONObject.NULL)
        put("error", error ?: JSONObject.NULL)
        put("compute_device", computeDevice)
        put("trashed_at", trashedAt ?: JSONObject.NULL)
        put("favorite", favorite)
        put("created_at", createdAt)
        put("stem_format", format)
        put("vocal_split", vocalSplit)
        put("sections", sections ?: JSONObject.NULL)
        put("sections_source", sectionsSource ?: JSONObject.NULL)
        put("analysis_notes", JSONObject(analysisNotes as Map<*, *>))
        put("stage_timings", JSONObject(stageTimings as Map<*, *>))
    }

    fun toRecord(): JSONObject = toState().apply {
        put("compute_seconds", computeSeconds ?: JSONObject.NULL)
        put("auto_sections", autoSections)
    }

    companion object {
        val FINISHED = setOf("done", "error", "cancelled", "stopped")

        fun fromRecord(o: JSONObject): Job {
            fun optD(k: String): Double? = if (o.isNull(k)) null else o.optDouble(k).takeUnless { it.isNaN() }
            val job = Job(
                id = o.getString("job_id"),
                title = o.optString("title", "Untitled"),
                sourceUrl = o.optString("source_url", "local:"),
                selectedStems = o.optJSONArray("selected_stems")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList(),
                createdAt = o.optDouble("created_at", 0.0),
            )
            job.status = o.optString("status", "error")
            job.progress = o.optDouble("progress", 0.0)
            job.stage = o.optString("stage", "")
            job.duration = optD("duration")
            job.lufs = optD("lufs")
            job.peakDb = optD("peak_db")
            job.dynamicRange = optD("dynamic_range")
            job.stemPresence = o.optJSONObject("stem_presence")?.let { p ->
                p.keys().asSequence().associateWith { p.getInt(it) }
            }
            job.stems = o.optJSONArray("stems")?.let { a -> List(a.length()) { a.getJSONObject(it).getString("name") } } ?: emptyList()
            job.error = if (o.isNull("error")) null else o.optString("error")
            job.favorite = o.optBoolean("favorite", false)
            job.trashedAt = optD("trashed_at")
            job.computeSeconds = optD("compute_seconds")
            job.format = o.optString("stem_format", "wav").takeIf { it == "mp3" } ?: "wav"
            job.computeDevice = o.optString("compute_device", "phone")
            fun optI(k: String): Int? = if (o.isNull(k) || !o.has(k)) null else o.optInt(k)
            fun optS(k: String): String? = if (o.isNull(k) || !o.has(k)) null else o.optString(k)
            job.bpm = optI("bpm")
            job.key = optS("key")
            job.scale = optS("scale")
            job.keyConfidence = optI("key_confidence")
            job.tempoStability = optI("tempo_stability")
            job.analysisNotes = o.optJSONObject("analysis_notes")?.let { n ->
                n.keys().asSequence().associateWith { n.getString(it) }
            } ?: emptyMap()
            job.hasLyrics = o.optBoolean("has_lyrics", false)
            job.audioTags = o.optJSONObject("audio_tags")?.let { t ->
                t.keys().asSequence().associateWith { t.optString(it) }
            } ?: emptyMap()
            job.lyricsQuery = o.optJSONObject("lyrics_query")
            // A split cut short by the app being closed is not still running.
            job.vocalSplit = o.optString("vocal_split", "none").let { if (it == "running") "none" else it }
            job.sections = o.optJSONArray("sections")
            job.sectionsSource = optS("sections_source")
            job.autoSections = o.optBoolean("auto_sections", false)
            job.stageTimings = o.optJSONObject("stage_timings")?.let { n ->
                n.keys().asSequence().associateWith { n.getDouble(it) }
            } ?: emptyMap()
            return job
        }
    }
}
