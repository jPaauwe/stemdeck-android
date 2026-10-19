package org.stemdeckport.android.engine

import java.io.File
import java.util.concurrent.CancellationException

/** MP3 stems with LAME (see app/src/main/cpp/mp3_codec.hpp). */
object Mp3Codec {
    fun interface Progress {
        /** Return false to stop. */
        fun onProgress(fraction: Float): Boolean
    }

    init {
        System.loadLibrary("sdengine")
    }

    const val KBPS = 320

    @JvmStatic private external fun nativeEncode(wav: String, mp3: String, kbps: Int, p: Progress?): String
    @JvmStatic private external fun nativeDecode(mp3: String, wav: String, p: Progress?): String

    /** 16-bit WAV → constant-bitrate MP3 (320 kbps by default), gapless. */
    fun encode(wav: File, mp3: File, kbps: Int = KBPS, progress: Progress? = null) =
        check(nativeEncode(wav.path, mp3.path, kbps, progress))

    /** An MP3 written by [encode] → 16-bit WAV, sample-aligned with the original. */
    fun decode(mp3: File, wav: File, progress: Progress? = null) =
        check(nativeDecode(mp3.path, wav.path, progress))

    private fun check(err: String) {
        if (err == "cancelled") throw CancellationException()
        if (err.isNotEmpty()) throw IllegalStateException(err)
    }
}
