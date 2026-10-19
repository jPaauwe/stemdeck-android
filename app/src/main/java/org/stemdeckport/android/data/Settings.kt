package org.stemdeckport.android.data

import android.content.Context
import org.json.JSONObject

/** App settings, shown in the web UI through /api/settings. */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    /** "mp3" (320 kbps, the default) or "wav" (16-bit). Applies to new songs and mix exports. */
    var stemFormat: String
        get() = prefs.getString(KEY_FORMAT, null)?.takeIf { it in FORMATS } ?: "mp3"
        set(v) { if (v in FORMATS) prefs.edit().putString(KEY_FORMAT, v).apply() }

    /** Use the GPU for separation (Vulkan). Default on; the engine falls back to the CPU on its own. */
    var gpu: Boolean
        get() = prefs.getBoolean(KEY_GPU, true)
        set(v) { prefs.edit().putBoolean(KEY_GPU, v).apply() }

    /**
     * The click track's settings in the Mixer (on/off, volume, accent, count-in),
     * as desktop keeps them ("stemdeck:metronome"). Kept here rather than in
     * the page's localStorage, whose origin (the server's port) changes on
     * every launch.
     */
    var click: JSONObject
        get() = try {
            JSONObject(prefs.getString(KEY_CLICK, null) ?: "{}")
        } catch (e: Exception) {
            JSONObject()
        }
        set(v) {
            val clean = JSONObject()
            if (v.has("on")) clean.put("on", v.optBoolean("on", false))
            if (v.has("volume")) clean.put("volume", v.optDouble("volume", 0.6).coerceIn(0.0, 1.0))
            if (v.has("accent")) clean.put("accent", v.optInt("accent", -1).coerceIn(-1, 32))
            if (v.has("count_in")) clean.put("count_in", v.optInt("count_in", 0).coerceIn(0, 4))
            prefs.edit().putString(KEY_CLICK, clean.toString()).apply()
        }

    /**
     * Network (1.3): look up lyrics on LRCLIB, sending only artist, title,
     * album and length. Off until the user turns it on.
     */
    var lyricsLookup: Boolean
        get() = prefs.getBoolean(KEY_LYRICS, false)
        set(v) { prefs.edit().putBoolean(KEY_LYRICS, v).apply() }

    /**
     * Find song sections for the next import (desktop's auto_sections). Not
     * saved: like desktop it starts off every session, because the stage
     * costs an extra model pass per song and should be asked for each time.
     */
    @Volatile var autoSections: Boolean = false

    fun toJson(): JSONObject = JSONObject().put("stem_format", stemFormat).put("gpu", gpu).put("click", click)
        .put("auto_sections", autoSections).put("lyrics_lookup", lyricsLookup)

    fun apply(o: JSONObject) {
        o.optString("stem_format").takeIf { it.isNotEmpty() }?.let { stemFormat = it }
        if (o.has("gpu")) gpu = o.optBoolean("gpu", true)
        o.optJSONObject("click")?.let { click = it }
        if (o.has("auto_sections")) autoSections = o.optBoolean("auto_sections", false)
        if (o.has("lyrics_lookup")) lyricsLookup = o.optBoolean("lyrics_lookup", false)
    }

    companion object {
        private const val KEY_FORMAT = "stem_format"
        private const val KEY_GPU = "gpu"
        private const val KEY_CLICK = "click"
        private const val KEY_LYRICS = "lyrics_lookup"
        val FORMATS = setOf("mp3", "wav")
    }
}
