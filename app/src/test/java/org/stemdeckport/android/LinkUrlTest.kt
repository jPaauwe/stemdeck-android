package org.stemdeckport.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.stemdeckport.android.data.LinkUrl

/** Desktop StemDeck's tests/test_url_validation.py, case for case. */
class LinkUrlTest {
    private val want = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    @Test fun accepts_youtube_urls() {
        for (u in listOf(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PLfoo",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
            "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://m.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLfoo&index=3",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&si=aBcDeFgHiJkLmNoP",
            "https://youtu.be/dQw4w9WgXcQ?si=aBcDeFgHiJkLmNoP&t=42",
            "https://www.youtube.com/watch?app=desktop&v=dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
            "https://www.youtube-nocookie.com/watch?v=dQw4w9WgXcQ",
        )) assertEquals(u, want, LinkUrl.validate(u))
    }

    @Test fun rejects_bad_urls() {
        for ((u, reason) in listOf(
            "" to "required", "   " to "required", "not a url" to "http",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ" to "http",
            "https://example.com/foo" to "unsupported host",
            "https://www.youtube.com/playlist?list=PLfoo" to "video ID",
            "https://evil.com/watch?v=dQw4w9WgXcQ" to "unsupported host",
            "https://on.soundcloud.com/abc123" to "unsupported host",
        )) {
            try {
                LinkUrl.validate(u)
                fail("accepted $u")
            } catch (e: LinkUrl.Invalid) {
                assertTrue("$u: ${e.message}", e.message!!.contains(reason))
            }
        }
    }

    @Test fun accepts_soundcloud_urls() {
        for (u in listOf("https://soundcloud.com/artist/track", "https://www.soundcloud.com/artist/track", "  https://soundcloud.com/artist/track  "))
            assertEquals(u.trim(), LinkUrl.validate(u))
    }
}
