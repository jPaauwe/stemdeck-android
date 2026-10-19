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

    fun toJson(): JSONObject = JSONObject().put("stem_format", stemFormat)

    fun apply(o: JSONObject) {
        o.optString("stem_format").takeIf { it.isNotEmpty() }?.let { stemFormat = it }
    }

    companion object {
        private const val KEY_FORMAT = "stem_format"
        val FORMATS = setOf("mp3", "wav")
    }
}
