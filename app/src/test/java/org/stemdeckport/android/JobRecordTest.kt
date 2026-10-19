package org.stemdeckport.android

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.stemdeckport.android.data.Job

class JobRecordTest {
    @Test
    fun a_record_from_1_0_still_loads() {
        val old = JSONObject(
            """{"job_id":"a1b2c3d4e5f6","status":"done","progress":1.0,"stage":"Done","title":"Old","duration":200.0,
            "thumbnail":null,"bpm":null,"key":null,"scale":null,"lufs":-9.1,"tempo_stability":null,"stems":[{"name":"drums","url":"x"}],
            "selected_stems":["drums"],"source_url":"local:Old","has_video":false,"has_lyrics":false,"error":null,
            "compute_device":"phone","trashed_at":null,"favorite":false,"created_at":1.0,"stem_format":"mp3"}""",
        )
        val j = Job.fromRecord(old)
        assertNull(j.bpm)
        assertNull(j.key)
        assertTrue(j.analysisNotes.isEmpty())
        val s = j.toState()
        assertTrue(s.isNull("bpm"))
        assertTrue(s.isNull("key_confidence"))
    }

    @Test
    fun analysis_fields_round_trip() {
        val j = Job("a1b2c3d4e5f6", "New", "local:New", listOf("drums"), 2.0)
        j.bpm = 128; j.key = "G# maj"; j.scale = "Major"; j.keyConfidence = 81; j.tempoStability = 97
        j.analysisNotes = mapOf("beats" to "No beat grid: test.")
        j.stageTimings = mapOf("beatgrid" to 4.2)
        val back = Job.fromRecord(JSONObject(j.toRecord().toString()))
        assertEquals(128, back.bpm)
        assertEquals("G# maj", back.key)
        assertEquals("Major", back.scale)
        assertEquals(81, back.keyConfidence)
        assertEquals(97, back.tempoStability)
        assertEquals("No beat grid: test.", back.analysisNotes["beats"])
        assertEquals(4.2, back.stageTimings["beatgrid"]!!, 0.0)
    }
}
