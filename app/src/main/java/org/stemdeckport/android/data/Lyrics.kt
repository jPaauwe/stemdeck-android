package org.stemdeckport.android.data

import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.Normalizer

/**
 * Lyrics from LRCLIB (lrclib.net), the useful core of desktop StemDeck's
 * app/pipeline/lyrics_lookup.py: the exact match (/api/get), the artist's
 * versions of the song (/api/search by artist and track) and the song's name
 * alone, kept only when a version is this song by this artist (belongs_to),
 * ranked by closeness of length, synced before plain. Only the artist, title,
 * album and length are ever sent.
 *
 * Not ported (desktop's identification, name aliases, retiming against the
 * vocals, accent repair, transcription): see the README.
 */
object LrcLib {
    const val API = "https://lrclib.net/api"
    /** Each request is abandoned after this long (desktop's TIMEOUT_LYRICS_LOOKUP). */
    const val TIMEOUT_MS = 10_000
    /** No request is started this far into a lookup (LYRICS_LOOKUP_BUDGET_SEC). */
    const val BUDGET_SEC = 30.0
    /** 429/503 are asked again this many times, waiting 1 s then 2 s (LYRICS_LOOKUP_RETRIES). */
    const val RETRIES = 2
    const val RETRY_SEC = 1.0
    /** The same recording lands within this many seconds (LYRICS_SAME_RECORDING_SEC). */
    const val SAME_RECORDING_SEC = 3.0
    const val OTHERS_MAX = 11
    const val OFFSET_MAX_SEC = 600.0
    private const val MAX_BYTES = 8 * 1024 * 1024
    private const val NAME_MAX = 512
    private const val TEXT_MAX = 200_000

    data class Query(val artist: String, val track: String, val album: String, val duration: Double)

    /** An HTTP status other than 200 and 404. */
    class Status(val code: Int, val retryAfterSec: Double?) : IOException("LRCLIB answered $code")

    /** One request: the parsed JSON, or null for a 404 ("no such track"). */
    fun interface Fetch {
        fun get(endpoint: String, params: Map<String, String>): Any?
    }

    /** What was found: the version kept (with "timing" and "others"), or only versions to offer. */
    class Answer(val entry: JSONObject?, val others: List<JSONObject>)

    /** The real thing, with the User-Agent LRCLIB asks clients to send. */
    fun http(userAgent: String): Fetch = Fetch { endpoint, params ->
        val q = params.entries.joinToString("&") { (k, v) -> "$k=" + URLEncoder.encode(v, "UTF-8") }
        val conn = URL("$API/$endpoint?$q").openConnection() as HttpURLConnection
        try {
            conn.connectTimeout = TIMEOUT_MS
            conn.readTimeout = TIMEOUT_MS
            conn.setRequestProperty("User-Agent", userAgent)
            conn.setRequestProperty("Accept", "application/json")
            val code = conn.responseCode
            if (code == 404) return@Fetch null
            if (code != 200) throw Status(code, conn.getHeaderField("Retry-After")?.trim()?.toDoubleOrNull())
            val body = conn.inputStream.use { input ->
                val out = java.io.ByteArrayOutputStream()
                val buf = ByteArray(1 shl 16)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    if (out.size() > MAX_BYTES) throw IOException("LRCLIB answer too large")
                }
                out.toString("UTF-8")
            }
            val t = body.trimStart()
            if (t.startsWith("[")) JSONArray(t) else JSONObject(t)
        } finally {
            conn.disconnect()
        }
    }

    // ── one LRCLIB row as a version (normalise) ──

    private fun number(v: Any?): Double = when (v) {
        is Boolean -> 0.0
        is Number -> v.toDouble().takeIf { it.isFinite() } ?: 0.0
        is String -> v.trim().ifEmpty { "0" }.toDoubleOrNull()?.takeIf { it.isFinite() } ?: 0.0
        else -> 0.0
    }

    private fun text(v: Any?, limit: Int) = (v as? String)?.take(limit) ?: ""

    private val OTHER_BREAKS = Regex("\r\n?|[\u000b\u000c\u001c-\u001e\u0085\u2028\u2029]")
    private fun lyricText(v: Any?) = (v as? String)?.replace("\ufeff", "")?.replace(OTHER_BREAKS, "\n")?.take(TEXT_MAX) ?: ""

    fun normalise(row: Any?): JSONObject {
        val r = row as? JSONObject ?: JSONObject()
        val id = number(r.opt("id")).toLong()
        return JSONObject()
            .put("v", 1)
            .put("source", "lrclib")
            .put("track", text(r.opt("trackName"), NAME_MAX))
            .put("artist", text(r.opt("artistName"), NAME_MAX))
            .put("album", text(r.opt("albumName"), NAME_MAX))
            .put("duration", number(r.opt("duration")))
            .put("synced", lyricText(r.opt("syncedLyrics")))
            .put("plain", lyricText(r.opt("plainLyrics")))
            .put("instrumental", r.opt("instrumental") == true)
            .put("lrclib_id", if (id > 0) id else JSONObject.NULL)
    }

    private fun JSONObject.id(): Long = optLong("lrclib_id", 0)
    private fun JSONObject.synced() = optString("synced").isNotEmpty()

    /** rank_matches: best first for a track [duration] seconds long (0 when unknown). */
    fun rankMatches(rows: Any?, duration: Double): List<JSONObject> {
        val list = (rows as? JSONArray)?.let { a -> List(a.length()) { a.opt(it) } } ?: emptyList()
        return rank(
            list.map(::normalise).filter {
                it.id() > 0 && (it.synced() || it.optString("plain").isNotEmpty() || it.optBoolean("instrumental"))
            },
            duration,
        )
    }

    private fun rank(matches: List<JSONObject>, duration: Double): List<JSONObject> {
        fun off(m: JSONObject): Double {
            val d = m.optDouble("duration", 0.0)
            return if (duration > 0 && d > 0) Math.abs(d - duration) else 0.0
        }
        return matches.withIndex().sortedWith { a, b ->
            val da = off(a.value)
            val db = off(b.value)
            when {
                Math.abs(da - db) > SAME_RECORDING_SEC -> da.compareTo(db)
                a.value.synced() != b.value.synced() -> if (a.value.synced()) -1 else 1
                da != db -> da.compareTo(db)
                else -> a.index.compareTo(b.index)
            }
        }.map { it.value }
    }

    fun sameRecording(m: JSONObject, duration: Double): Boolean {
        val d = m.optDouble("duration", 0.0)
        return duration > 0 && d > 0 && Math.abs(d - duration) <= SAME_RECORDING_SEC
    }

    // ── whose song a version is (belongs_to, for names in any script but
    //    without desktop's traditional/simplified Chinese folding) ──

    /** Compared folded: compatibility forms, case and accents ("Beyoncé" is "beyonce"). */
    private fun fold(s: Any?): String =
        Normalizer.normalize(Normalizer.normalize(s?.toString() ?: "", Normalizer.Form.NFKC).lowercase(), Normalizer.Form.NFD)
            .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }

    private fun isWordChar(c: Char) = Character.isLetterOrDigit(c) || Character.getType(c) == Character.NON_SPACING_MARK.toInt() ||
        Character.getType(c) == Character.COMBINING_SPACING_MARK.toInt()

    private fun key(s: String) = s.filter { isWordChar(it) }
    private val LEADING_THE = Regex("^\\s*the\\s+", RegexOption.IGNORE_CASE)
    private fun nameKey(s: Any?) = key(LEADING_THE.replace(fold(s), ""))
    private fun nameWords(s: Any?): List<String> = fold(s).split(Regex("[^\\p{L}\\p{M}\\p{N}]+")).map(::key).filter { it.isNotEmpty() }
    private val NAME_LIST = Regex(
        "\\s*(?:[&,+/;]|\\band\\b|\\bwith\\b|\\bfeat\\.?|\\bft\\.?|\\bfeaturing\\b|\\bvs\\.?)\\s*", RegexOption.IGNORE_CASE,
    )
    private fun nameParts(s: Any?) = NAME_LIST.split(fold(s)).map { nameKey(it) }.filter { it.isNotEmpty() }
    private fun weight(s: String): Int = s.fold(0) { acc, c -> acc + if (Character.UnicodeScript.of(c.code) in CJK) 2 else 1 }
    private val CJK = setOf(Character.UnicodeScript.HAN, Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA, Character.UnicodeScript.HANGUL)
    private fun containsRun(words: List<String>, run: List<String>): Boolean {
        if (run.size < 2 || run.size > words.size) return false
        return (0..words.size - run.size).any { i -> words.subList(i, i + run.size) == run }
    }

    fun sameArtist(found: Any?, names: List<String>): Boolean {
        val theirs = nameKey(found)
        if (theirs.isEmpty()) return false
        val theirParts = nameParts(found)
        val theirWords = nameWords(found)
        for (name in names) {
            if (name.isBlank()) continue
            if (theirs == nameKey(name)) return true
            if (nameParts(name).any { weight(it) >= 4 && it in theirParts }) return true
            val mine = nameWords(name)
            if (containsRun(mine, theirWords) || containsRun(theirWords, mine)) return true
        }
        return false
    }

    private val SONG_FEATURING = Regex("\\s+(?:ft\\.?|feat\\.?|featuring)\\s.*$", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))
    private val SONG_BRACKETS = Regex("[(\\[{【][^()\\[\\]{}【】]*[)\\]}】]")
    private val SONG_TAIL = Regex("\\s+\\p{Pd}+\\s+.*$", RegexOption.DOT_MATCHES_ALL)

    private fun songKeys(s: Any?): Pair<String, String> {
        val t = fold(s)
        var full = SONG_FEATURING.replace(SONG_BRACKETS.replace(t, " "), "")
        if (key(full).isEmpty()) full = t
        return key(full) to key(SONG_TAIL.replace(full, ""))
    }

    fun sameSong(found: Any?, song: Any?): Boolean {
        val (aFull, aHead) = songKeys(found)
        val (bFull, bHead) = songKeys(song)
        if (aFull.isEmpty() || bFull.isEmpty()) return false
        return aFull == bFull || aFull == bHead || aHead == bFull
    }

    fun belongsTo(m: JSONObject, song: String, names: List<String>) =
        sameSong(m.optString("track"), song) && sameArtist(m.optString("artist"), names)

    // ── the song's name from a title (song_from_title) ──

    private val TITLE_NOISE = Regex(
        "[(\\[][^)\\]]*(?<!\\p{L})(official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|remaster(?:ed)?|m/?v|explicit|clean|full album)(?!\\p{L})[^)\\]]*[)\\]]",
        RegexOption.IGNORE_CASE,
    )
    private val FEATURING = Regex("\\s+(?:ft\\.?|feat\\.?|featuring)\\s[^\\p{Pd}(\\[]*", RegexOption.IGNORE_CASE)
    private const val DASH = "[\\p{Pd}:|]"

    fun songFromTitle(title: String, artist: String = ""): String {
        var song = TITLE_NOISE.replace(title, " ")
        song = FEATURING.replace(song, " ")
        val name = artist.trim()
        if (name.isNotEmpty()) {
            val lit = Regex.escape(name)
            song = Regex("^\\s*$lit\\s*$DASH\\s*", RegexOption.IGNORE_CASE).replaceFirst(song, "")
            song = Regex("\\s*$DASH\\s*$lit\\s*$", RegexOption.IGNORE_CASE).replaceFirst(song, "")
        }
        song = song.filter { c -> c != '"' && Character.getType(c) != Character.INITIAL_QUOTE_PUNCTUATION.toInt() &&
            Character.getType(c) != Character.FINAL_QUOTE_PUNCTUATION.toInt() }
        song = song.replace(Regex("\\s+"), " ")
        fun edge(c: Char) = c.isWhitespace() || Character.getType(c) == Character.DASH_PUNCTUATION.toInt() || c == ':' || c == '|'
        return song.trim { edge(it) }.trim()
    }

    // ── the lookup ──

    private class Stop : RuntimeException()

    /**
     * What LRCLIB has for [q]: null when the lookup was cancelled or ran out of
     * time before anything was kept. Throws when LRCLIB cannot be reached.
     * [clock] is seconds (monotonic); [sleep] waits that many seconds.
     */
    fun lookup(
        q: Query, fetch: Fetch, cancelled: () -> Boolean = { false }, budgetSec: Double = BUDGET_SEC,
        clock: () -> Double = { System.nanoTime() / 1e9 }, sleep: (Double) -> Unit = { Thread.sleep((it * 1000).toLong()) },
    ): Answer? {
        if (q.track.isBlank()) return null
        val deadline = clock() + budgetSec
        val duration = q.duration
        val names = listOf(q.artist).filter { it.isNotBlank() }
        val found = ArrayList<JSONObject>()
        val song = ArrayList<JSONObject>()
        var chosen: JSONObject? = null

        fun check() {
            if (cancelled() || clock() > deadline) throw Stop()
        }
        fun ask(endpoint: String, params: Map<String, String>): Any? {
            var attempt = 0
            while (true) {
                check()
                try {
                    return fetch.get(endpoint, params)
                } catch (e: Status) {
                    val busy = e.code == 429 || e.code == 503
                    // LRCLIB asks clients to honour Retry-After on a 429.
                    val wait = maxOf(RETRY_SEC * (1 shl attempt), e.retryAfterSec ?: 0.0)
                    if (!busy || attempt == RETRIES || clock() + wait > deadline) throw e
                    sleep(wait)
                    attempt++
                }
            }
        }
        fun mine(rows: List<JSONObject>) = rows.filter { belongsTo(it, q.track, names) }
        fun held(rows: List<JSONObject>) = if (duration > 0) rows.filter { sameRecording(it, duration) } else rows
        fun search(params: Map<String, String>): List<JSONObject> {
            val rows = mine(rankMatches(ask("search", params), duration))
            found.addAll(rows)
            return rows
        }

        try {
            if (q.artist.isNotBlank()) {
                // (a) The exact match. A failure here is not the end: the search asks another index.
                val params = LinkedHashMap<String, String>()
                params["artist_name"] = q.artist
                params["track_name"] = q.track
                if (q.album.isNotBlank()) params["album_name"] = q.album
                if (duration > 0) params["duration"] = Math.round(duration).toString()
                var exact = try {
                    mine(rankMatches(JSONArray().apply { ask("get", params)?.let { put(it) } }, duration))
                } catch (e: Status) {
                    emptyList()
                }
                song.addAll(exact)
                exact = held(exact)
                chosen = exact.firstOrNull()
                // (b) The artist's versions of the song.
                val rows = search(mapOf("artist_name" to q.artist, "track_name" to q.track))
                song.addAll(rows)
                chosen = chosen ?: held(rows).firstOrNull()
            }
            if (chosen == null && duration > 0 && names.isNotEmpty()) {
                // (c) The name alone; only a version by this artist is kept.
                val rows = search(mapOf("q" to q.track))
                song.addAll(rows)
                chosen = held(rows).firstOrNull()
            }
        } catch (e: Stop) {
            if (cancelled() || (chosen == null && song.isEmpty())) return null
        }

        var timing = "exact"
        if (chosen == null && song.isNotEmpty()) {
            // No version the track's length, but the song: a synced one when there is one.
            val ranked = rank(song, duration)
            chosen = ranked.firstOrNull { it.synced() } ?: ranked.first()
            timing = if (chosen.synced()) "unverified" else "exact"
        }
        val seen = HashSet<Long>()
        chosen?.let { seen.add(it.id()) }
        val others = ArrayList<JSONObject>()
        for (m in rank(found, duration)) if (seen.add(m.id())) others.add(m)
        val kept = others.take(OTHERS_MAX)
        if (chosen == null) return Answer(null, kept)
        return Answer(JSONObject(chosen.toString()).put("timing", timing).put("others", JSONArray(kept)), emptyList())
    }

    /** An offset within ±600 s to the hundredth (clean_offset), or null. */
    fun cleanOffset(v: Double): Double? {
        if (!v.isFinite() || Math.abs(v) > OFFSET_MAX_SEC) return null
        return Math.round(v * 100) / 100.0 + 0.0
    }
}
