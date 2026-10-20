package org.stemdeckport.android

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.stemdeckport.android.data.LrcLib

/**
 * The lookup against recorded LRCLIB answers, no network: the cases of
 * desktop's tests/test_lyrics_lookup.py for the parts ported (ranking, the
 * exact match, the artist's search, the name alone, belongs_to,
 * song_from_title, retries).
 */
class LrcLibTest {
    private fun row(
        id: Int, duration: Double, kind: String = "synced",
        artist: String = "Dream Theater", track: String = "Metropolis", album: String = "",
    ) = JSONObject()
        .put("id", id).put("trackName", track).put("artistName", artist).put("albumName", album)
        .put("duration", duration).put("instrumental", kind == "instrumental")
        .put("syncedLyrics", if (kind == "synced") "[00:01.00]line $id" else JSONObject.NULL)
        .put("plainLyrics", if (kind == "synced" || kind == "plain") "line $id" else JSONObject.NULL)

    /** Answers /get with [exact] and a search by its artist (or "q:<name>") from [searches]. */
    private class Fake(val exact: Any? = null, val searches: Map<String, List<JSONObject>> = emptyMap()) : LrcLib.Fetch {
        val asked = ArrayList<Pair<String, Map<String, String>>>()
        val searched get() = asked.filter { it.first == "search" }.map { it.second["artist_name"] ?: "q:${it.second["q"]}" }
        override fun get(endpoint: String, params: Map<String, String>): Any? {
            asked.add(endpoint to params)
            if (endpoint == "get") {
                if (exact is Exception) throw exact
                return exact
            }
            return JSONArray(searches[params["artist_name"] ?: "q:${params["q"]}"] ?: emptyList<JSONObject>())
        }
    }

    private fun query(artist: String = "Dream Theater", track: String = "Metropolis", album: String = "", duration: Double = 572.0) =
        LrcLib.Query(artist, track, album, duration)

    private fun kept(q: LrcLib.Query, f: LrcLib.Fetch) = LrcLib.lookup(q, f)?.entry
    private fun ids(a: JSONArray?) = (0 until (a?.length() ?: 0)).map { a!!.getJSONObject(it).getLong("lrclib_id").toInt() }
    private fun ids(l: List<JSONObject>) = l.map { it.getLong("lrclib_id").toInt() }

    @Test fun closest_length_first_synced_preferred_within_a_few_seconds_empty_rows_dropped() {
        val rows = JSONArray(listOf(row(1, 769.0), row(2, 571.0, "plain"), row(3, 573.0), row(4, 572.0, "none"), row(5, 590.0)))
        assertEquals(listOf(3, 2, 5, 1), ids(LrcLib.rankMatches(rows, 572.0)))
    }

    @Test fun an_instrumental_row_is_kept() {
        assertEquals(1, LrcLib.rankMatches(JSONArray(listOf(row(9, 100.0, "instrumental"))), 100.0).size)
    }

    @Test fun with_no_length_known_lrclibs_order_synced_first() {
        assertEquals(listOf(2, 1), ids(LrcLib.rankMatches(JSONArray(listOf(row(1, 10.0, "plain"), row(2, 500.0))), 0.0)))
    }

    @Test fun not_a_list_gives_nothing() {
        assertTrue(LrcLib.rankMatches(null, 0.0).isEmpty())
        assertTrue(LrcLib.rankMatches(JSONObject().put("error", 1), 0.0).isEmpty())
    }

    @Test fun a_row_becomes_a_version_of_lyrics_json() {
        val v = LrcLib.rankMatches(JSONArray(listOf(row(7, 200.5, album = "Images and Words"))), 0.0).single()
        assertEquals("lrclib", v.getString("source"))
        assertEquals(1, v.getInt("v"))
        assertEquals("Images and Words", v.getString("album"))
        assertEquals(200.5, v.getDouble("duration"), 0.0)
        assertEquals("[00:01.00]line 7", v.getString("synced"))
        assertEquals(7L, v.getLong("lrclib_id"))
    }

    @Test fun the_song_from_a_title() {
        val en = "–"
        assertEquals("Pull Me Under", LrcLib.songFromTitle("Dream Theater - Pull Me Under (Official Video)", "Dream Theater"))
        assertEquals("Bohemian Rhapsody", LrcLib.songFromTitle("Bohemian Rhapsody $en Queen [HD]", "Queen"))
        assertEquals("Metropolis - Part I: The Miracle and the Sleeper",
            LrcLib.songFromTitle("Metropolis - Part I: \"The Miracle and the Sleeper\"", "Dream Theater"))
        assertEquals("Empire State of Mind", LrcLib.songFromTitle("Jay-Z ft. Alicia Keys - Empire State of Mind", "Jay-Z"))
        assertEquals("Shadow Of The Day (Shadow Remix)", LrcLib.songFromTitle("Shadow Of The Day (Shadow Remix)"))
        assertEquals("Hello", LrcLib.songFromTitle("Hello (Official Music Video) (Remastered 2015)"))
    }

    @Test fun an_exact_match_is_kept_and_the_artists_versions_offered_beside_it() {
        val f = Fake(row(10, 572.0), mapOf("Dream Theater" to listOf(row(11, 769.0), row(10, 572.0), row(12, 571.0))))
        val found = kept(query(album = "Images and Words"), f)!!
        assertEquals(10L, found.getLong("lrclib_id"))
        assertEquals(listOf(12, 11), ids(found.getJSONArray("others")))
        assertEquals("get" to mapOf("artist_name" to "Dream Theater", "track_name" to "Metropolis",
            "album_name" to "Images and Words", "duration" to "572"), f.asked[0])
        assertEquals(listOf("Dream Theater"), f.searched)
    }

    @Test fun no_exact_match_keeps_the_artists_version_closest_in_length() {
        val f = Fake(searches = mapOf("Dream Theater" to listOf(row(1, 769.0), row(3, 573.0), row(2, 571.0, "plain"))))
        val found = kept(query(), f)!!
        assertEquals(3L, found.getLong("lrclib_id"))
        assertEquals(listOf(2, 1), ids(found.getJSONArray("others")))
        assertEquals(listOf("Dream Theater"), f.searched)
    }

    @Test fun an_exact_match_that_errors_still_searches() {
        val f = Fake(LrcLib.Status(500, null), mapOf("Dream Theater" to listOf(row(3, 573.0))))
        assertEquals(3L, kept(query(), f)!!.getLong("lrclib_id"))
    }

    @Test fun others_are_capped() {
        val f = Fake(searches = mapOf("Dream Theater" to (1 until 30).map { row(it, 572.0) }))
        assertEquals(11, kept(query(), f)!!.getJSONArray("others").length())
    }

    @Test fun the_name_alone_keeps_the_artists_song_timed_by_its_length() {
        var found = kept(query(), Fake(searches = mapOf("q:Metropolis" to listOf(row(40, 575.5), row(41, 600.0)))))!!
        assertEquals(40L to "unverified", found.getLong("lrclib_id") to found.getString("timing"))
        found = kept(query(), Fake(searches = mapOf("q:Metropolis" to listOf(row(40, 574.9), row(41, 600.0)))))!!
        assertEquals(40L to "exact", found.getLong("lrclib_id") to found.getString("timing"))
    }

    @Test fun the_name_alone_is_never_asked_without_the_tracks_length() {
        val f = Fake(searches = mapOf("q:Metropolis" to listOf(row(40, 572.0))))
        assertNull(kept(query(duration = 0.0), f))
        assertFalse("q:Metropolis" in f.searched)
    }

    @Test fun a_track_known_by_no_name_gets_no_lyrics() {
        val f = Fake(searches = mapOf("q:Metropolis" to listOf(row(40, 572.0))))
        val a = LrcLib.lookup(query(artist = ""), f)!!
        assertNull(a.entry)
        assertTrue(a.others.isEmpty())
        assertTrue(f.asked.isEmpty())
    }

    @Test fun another_artists_song_of_that_name_is_neither_kept_nor_offered() {
        val f = Fake(
            row(10, 572.0, artist = "Metropolis Tribute Orchestra"),
            mapOf(
                "Dream Theater" to listOf(row(11, 572.0, track = "Metropolis Part 2"), row(12, 572.0, artist = "Kesha")),
                "q:Metropolis" to listOf(row(13, 572.0, artist = "Someone Else")),
            ),
        )
        val a = LrcLib.lookup(query(), f)!!
        assertNull(a.entry)
        assertTrue(a.others.isEmpty())
    }

    @Test fun a_credited_artist_or_a_song_tail_still_belongs() {
        val q = query("Keala Settle & The Greatest Showman Ensemble", "This Is Me", duration = 235.0)
        val tail = row(50, 234.9, artist = "Keala Settle", track = "This Is Me - From The Greatest Showman")
        assertEquals(50L, kept(q, Fake(searches = mapOf(q.artist to listOf(tail))))!!.getLong("lrclib_id"))
    }

    @Test fun same_artist() {
        assertTrue(LrcLib.sameArtist("The Beatles", listOf("Beatles")))
        assertTrue(LrcLib.sameArtist("Beyonce", listOf("Beyoncé")))
        assertTrue(LrcLib.sameArtist("Keala Settle", listOf("Keala Settle & The Greatest Showman Ensemble")))
        assertTrue(LrcLib.sameArtist("The Greatest Showman Cast", listOf("The Greatest Showman")))
        assertFalse(LrcLib.sameArtist("Pink", listOf("Pink Floyd")))
        assertFalse(LrcLib.sameArtist("DJ", listOf("DJ & Someone Else")))
        assertFalse(LrcLib.sameArtist("Kesha", listOf("Keala Settle")))
        assertFalse(LrcLib.sameArtist("", listOf("Keala Settle")))
        assertFalse(LrcLib.sameArtist("Keala Settle", listOf("")))
    }

    @Test fun same_song() {
        assertTrue(LrcLib.sameSong("This Is Me", "this is me"))
        assertTrue(LrcLib.sameSong("This Is Me (feat. Someone) [Live]", "This Is Me"))
        assertTrue(LrcLib.sameSong("This Is Me - From The Greatest Showman", "This Is Me"))
        assertFalse(LrcLib.sameSong("Part I - Dawn", "Part I - Dusk"))
        assertFalse(LrcLib.sameSong("This Is Not Me", "This Is Me"))
        assertFalse(LrcLib.sameSong("", "This Is Me"))
    }

    @Test fun a_version_the_tracks_length_is_kept_with_exact_timing() {
        assertEquals("exact", kept(query(), Fake(searches = mapOf("Dream Theater" to listOf(row(1, 769.0), row(3, 573.0)))))!!.getString("timing"))
    }

    @Test fun the_song_in_another_length_is_kept_with_timing_unverified() {
        val f = Fake(searches = mapOf("Dream Theater" to listOf(row(1, 769.0), row(2, 590.0))))
        val found = kept(query(), f)!!
        assertEquals(2L to "unverified", found.getLong("lrclib_id") to found.getString("timing"))
        assertEquals(listOf(1), ids(found.getJSONArray("others")))
        assertEquals(listOf("Dream Theater", "q:Metropolis"), f.searched)
    }

    @Test fun a_version_the_tracks_length_under_another_name_wins_over_the_song_in_another() {
        val found = kept(query(), Fake(searches = mapOf("Dream Theater" to listOf(row(2, 590.0)), "q:Metropolis" to listOf(row(40, 572.0)))))!!
        assertEquals(40L to "exact", found.getLong("lrclib_id") to found.getString("timing"))
    }

    @Test fun a_synced_version_is_preferred_when_none_is_the_tracks_length() {
        var found = kept(query(), Fake(searches = mapOf("Dream Theater" to listOf(row(1, 590.0, "plain"), row(2, 700.0)))))!!
        assertEquals(2L to "unverified", found.getLong("lrclib_id") to found.getString("timing"))
        found = kept(query(), Fake(searches = mapOf("Dream Theater" to listOf(row(1, 590.0, "plain")))))!!
        assertEquals(1L to "exact", found.getLong("lrclib_id") to found.getString("timing"))
    }

    @Test fun an_exact_match_of_another_length_is_not_preferred() {
        assertEquals(3L, kept(query(), Fake(row(10, 600.0), mapOf("Dream Theater" to listOf(row(3, 571.0)))))!!.getLong("lrclib_id"))
    }

    @Test fun with_the_length_unknown_the_artists_best_is_kept() {
        assertEquals(1L, kept(query(duration = 0.0), Fake(searches = mapOf("Dream Theater" to listOf(row(1, 769.0), row(2, 590.0)))))!!.getLong("lrclib_id"))
    }

    @Test fun nothing_on_lrclib_is_an_answer_with_nothing_to_offer() {
        val a = LrcLib.lookup(query(), Fake())!!
        assertNull(a.entry)
        assertTrue(a.others.isEmpty())
    }

    @Test fun out_of_time_with_nothing_kept_is_no_answer() {
        var now = 0.0
        val slow = LrcLib.Fetch { _, _ -> now += 31.0; JSONArray() }
        assertNull(LrcLib.lookup(query(), slow, clock = { now }))
    }

    @Test fun a_cancel_between_requests_stops_the_next() {
        var cancel = false
        val f = object : LrcLib.Fetch {
            var n = 0
            override fun get(endpoint: String, params: Map<String, String>): Any? { n++; cancel = true; return null }
        }
        assertNull(LrcLib.lookup(query(), f, cancelled = { cancel }))
        assertEquals(1, f.n)
    }

    @Test fun busy_answers_are_asked_again_with_backoff_and_retry_after() {
        val waits = ArrayList<Double>()
        var calls = 0
        val f = LrcLib.Fetch { endpoint, _ ->
            calls++
            if (endpoint == "get") null
            else if (calls < 4) throw LrcLib.Status(if (calls == 2) 429 else 503, if (calls == 2) 1.5 else null)
            else JSONArray(listOf(row(3, 573.0)))
        }
        val found = LrcLib.lookup(query(), f, sleep = { waits.add(it) })!!.entry!!
        assertEquals(3L, found.getLong("lrclib_id"))
        assertEquals(listOf(1.5, 2.0), waits)  // Retry-After honoured, then 1 s x 2
    }

    @Test fun offsets_are_cleaned() {
        assertEquals(1.24, LrcLib.cleanOffset(1.2351)!!, 0.0)
        assertEquals(0.0, LrcLib.cleanOffset(-0.001)!!, 0.0)
        assertNull(LrcLib.cleanOffset(601.0))
        assertNull(LrcLib.cleanOffset(Double.NaN))
    }
}
