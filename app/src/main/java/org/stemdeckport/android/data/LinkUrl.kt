package org.stemdeckport.android.data

import java.net.URI
import java.net.URLDecoder

/**
 * Desktop StemDeck's link validation (app/pipeline/download.py,
 * validate_youtube_url / normalize_youtube_url): only http(s) links on
 * YouTube's and SoundCloud's own hosts are handed to yt-dlp, YouTube links
 * reduced to the single-video watch form, and on.soundcloud.com (a shortener
 * that redirects anywhere) refused.
 */
object LinkUrl {
    class Invalid(message: String) : IllegalArgumentException(message)

    private val VIDEO_ID = Regex("^[A-Za-z0-9_-]{11}$")
    private val YOUTUBE_HOSTS = setOf(
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
        "youtube-nocookie.com", "www.youtube-nocookie.com",
    )
    private val SOUNDCLOUD_HOSTS = setOf("soundcloud.com", "www.soundcloud.com")
    val ALLOWED_HOSTS = YOUTUBE_HOSTS + SOUNDCLOUD_HOSTS

    /** The link to download, or throws [Invalid] with desktop's reasons. */
    fun validate(url: String?): String {
        val u = url?.trim().orEmpty()
        if (u.isEmpty()) throw Invalid("URL is required")
        val parsed = try {
            URI(u)
        } catch (e: Exception) {
            throw Invalid("URL must use http or https")
        }
        if (parsed.scheme?.lowercase() !in setOf("http", "https")) throw Invalid("URL must use http or https")
        val host = parsed.host?.lowercase().orEmpty()
        if (host !in ALLOWED_HOSTS) throw Invalid("unsupported host: ${host.ifEmpty { "(empty)" }}")
        if (host in SOUNDCLOUD_HOSTS) return u
        val normalized = normalize(u)
        if (!normalized.startsWith("https://www.youtube.com/watch?v=")) throw Invalid("could not extract a video ID from URL")
        return normalized
    }

    private fun query(q: String?): Map<String, String> {
        if (q.isNullOrEmpty()) return emptyMap()
        val out = LinkedHashMap<String, String>()
        for (part in q.split('&')) {
            if (part.isEmpty()) continue
            val k = URLDecoder.decode(part.substringBefore('='), "UTF-8")
            val v = URLDecoder.decode(part.substringAfter('=', ""), "UTF-8")
            if (k !in out && v.isNotEmpty()) out[k] = v  // parse_qs keeps the first, drops blanks
        }
        return out
    }

    /** normalize_youtube_url: a YouTube link in single-video form; anything else unchanged. */
    fun normalize(url: String): String {
        val parsed = try { URI(url) } catch (e: Exception) { return url }
        var host = parsed.host?.lowercase().orEmpty()
        for (prefix in listOf("www.", "m.", "music.")) if (host.startsWith(prefix)) { host = host.removePrefix(prefix); break }
        if (host !in setOf("youtube.com", "youtu.be", "youtube-nocookie.com")) return url
        val qs = query(parsed.rawQuery)
        qs["v"]?.let { if (VIDEO_ID.matches(it)) return "https://www.youtube.com/watch?v=$it" }
        qs["list"]?.let { if (it.startsWith("RD") && it.length >= 13 && VIDEO_ID.matches(it.substring(2, 13))) return "https://www.youtube.com/watch?v=${it.substring(2, 13)}" }
        val path = parsed.rawPath.orEmpty()
        if (host == "youtu.be") {
            val vid = path.trimStart('/')
            if (VIDEO_ID.matches(vid)) return "https://www.youtube.com/watch?v=$vid"
        }
        if (host == "youtube.com" || host == "youtube-nocookie.com") {
            for (prefix in listOf("/shorts/", "/live/", "/embed/")) {
                if (path.startsWith(prefix)) {
                    val vid = path.removePrefix(prefix).trimStart('/').substringBefore('/')
                    if (VIDEO_ID.matches(vid)) return "https://www.youtube.com/watch?v=$vid"
                    break
                }
            }
        }
        return url
    }
}
