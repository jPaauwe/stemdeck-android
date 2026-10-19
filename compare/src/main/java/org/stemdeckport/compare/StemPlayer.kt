package org.stemdeckport.compare

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * Plays 44.1 kHz stereo audio from one of several sources and switches
 * between them on the fly, at the same position, with a short crossfade. That
 * is what makes A/B listening between engine builds work: the moment you tap
 * another build you hear the same note from it.
 */
class StemPlayer(private val totalFrames: Long) {
    /** Something to play: 44.1 kHz stereo, read at any frame. */
    interface Source {
        /** Fills out[0 until 2n] with interleaved stereo from [frame]. Zeros past the end. */
        fun read(frame: Long, n: Int, out: FloatArray)
    }

    companion object {
        const val RATE = 44100
        private const val CHUNK = 2048 // frames per write, ~46 ms
        private const val FADE = 512 // crossfade on a switch, ~12 ms
    }

    @Volatile var source: Source? = null
    @Volatile var playing = false
        private set
    /** Loop between these frames while set. */
    @Volatile var loop: LongRange? = null

    // Every change of position (play, pause, seek) bumps gen; a chunk being
    // written for an older gen is dropped, so no stale audio is heard.
    private var gen = 0L
    private val markers = ArrayDeque<LongArray>() // (frames written so far, source frame)
    private val lock = Object()
    @Volatile private var alive = true
    private var readPos = 0L
    private var written = 0L
    private val track: AudioTrack
    private val thread: Thread

    init {
        val minBuf = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_FLOAT)
        track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO).build())
            .setBufferSizeInBytes(max(minBuf, CHUNK * 8 * 3))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        thread = Thread(::pump, "stem-player").apply { start() }
    }

    val duration: Long get() = totalFrames

    /** The frame being heard now. */
    val position: Long
        get() = synchronized(lock) {
            if (!playing) return readPos
            val head = track.playbackHeadPosition.toLong() and 0xffffffffL
            var best: LongArray? = null
            for (m in markers) if (m[0] <= head) best = m
            val m = best ?: return readPos
            min(totalFrames, m[1] + (head - m[0]))
        }

    fun play() {
        synchronized(lock) {
            if (playing) return
            if (readPos >= totalFrames) readPos = 0
            gen++
            track.flush()
            markers.clear(); written = 0
            playing = true
            track.play()
            lock.notifyAll()
        }
    }

    fun pause() {
        synchronized(lock) {
            if (!playing) return
            readPos = position
            playing = false
            gen++
            track.pause()
            track.flush()
            markers.clear(); written = 0
        }
    }

    fun seek(frame: Long) {
        val f = frame.coerceIn(0, max(0, totalFrames - 1))
        synchronized(lock) {
            gen++
            readPos = f
            if (playing) {
                track.pause(); track.flush()
                markers.clear(); written = 0
                track.play()
            }
        }
    }

    fun release() {
        alive = false
        synchronized(lock) { playing = false; gen++; lock.notifyAll() }
        thread.join(1000)
        track.release()
    }

    private fun pump() {
        val cur = FloatArray(CHUNK * 2)
        val next = FloatArray(CHUNK * 2)
        var lastSource: Source? = null
        while (alive) {
            var g = 0L
            var pos = 0L
            var n = CHUNK
            synchronized(lock) {
                while (alive && !playing) lock.wait()
                g = gen
                val lp = loop
                if (lp != null) {
                    if (readPos < lp.first || readPos >= lp.last) readPos = lp.first
                    n = min(n.toLong(), lp.last - readPos).toInt().coerceAtLeast(1)
                } else if (readPos >= totalFrames) {
                    // The end: let what is buffered play out, then stop there.
                    track.stop()
                    playing = false
                    gen++
                    markers.clear(); written = 0
                    readPos = totalFrames
                    n = 0
                } else {
                    n = min(n.toLong(), totalFrames - readPos).toInt()
                }
                pos = readPos
            }
            if (!alive) break
            if (n <= 0) continue
            val src = source
            if (src == null) next.fill(0f, 0, n * 2) else src.read(pos, n, next)
            val prev = lastSource
            if (prev != null && src != null && prev !== src) {
                // Crossfade from the build we were on to the new one.
                prev.read(pos, n, cur)
                val fade = min(FADE, n)
                for (i in 0 until fade) {
                    val t = i.toFloat() / fade
                    next[2 * i] = cur[2 * i] * (1 - t) + next[2 * i] * t
                    next[2 * i + 1] = cur[2 * i + 1] * (1 - t) + next[2 * i + 1] * t
                }
            }
            lastSource = src
            for (i in 0 until n * 2) next[i] = next[i].coerceIn(-1f, 1f)

            var off = 0
            var dropped = false
            while (off < n * 2) {
                synchronized(lock) {
                    if (g != gen || !playing) {
                        dropped = true
                    } else {
                        if (off == 0) {
                            markers.addLast(longArrayOf(written, pos))
                            while (markers.size > 64) markers.removeFirst()
                        }
                        val w = track.write(next, off, n * 2 - off, AudioTrack.WRITE_NON_BLOCKING)
                        if (w < 0) dropped = true else off += w
                    }
                }
                if (dropped) break
                if (off < n * 2) Thread.sleep(5)
            }
            if (!dropped) synchronized(lock) {
                if (g == gen) {
                    written += n
                    readPos = pos + n
                }
            }
        }
    }

    // ─── sources ───

    /** A 16-bit stereo WAV as the engines write it (44-byte header). */
    class WavSource(file: File) : Source, java.io.Closeable {
        private val raf = RandomAccessFile(file, "r")
        private val frames = max(0L, (raf.length() - 44) / 4)
        private var buf = ByteArray(0)
        override fun close() = raf.close()
        override fun read(frame: Long, n: Int, out: FloatArray) {
            out.fill(0f, 0, n * 2)
            if (frame >= frames) return
            val count = min(n.toLong(), frames - frame).toInt()
            if (buf.size < count * 4) buf = ByteArray(count * 4)
            synchronized(raf) {
                raf.seek(44 + frame * 4)
                raf.readFully(buf, 0, count * 4)
            }
            val bb = ByteBuffer.wrap(buf, 0, count * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count * 2) out[i] = bb.short / 32768f
        }
    }

    /** Raw interleaved stereo float32 (the decoded original song). */
    class F32Source(file: File) : Source, java.io.Closeable {
        private val raf = RandomAccessFile(file, "r")
        private val frames = raf.length() / 8
        private var buf = ByteArray(0)
        override fun close() = raf.close()
        override fun read(frame: Long, n: Int, out: FloatArray) {
            out.fill(0f, 0, n * 2)
            if (frame >= frames) return
            val count = min(n.toLong(), frames - frame).toInt()
            if (buf.size < count * 8) buf = ByteArray(count * 8)
            synchronized(raf) {
                raf.seek(frame * 8)
                raf.readFully(buf, 0, count * 8)
            }
            val bb = ByteBuffer.wrap(buf, 0, count * 8).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until count * 2) out[i] = bb.float
        }
    }

    /** Several sources added together (all stems = the full mix). */
    class SumSource(private val parts: List<Source>) : Source {
        private var tmp = FloatArray(0)
        override fun read(frame: Long, n: Int, out: FloatArray) {
            out.fill(0f, 0, n * 2)
            if (tmp.size < n * 2) tmp = FloatArray(n * 2)
            for (p in parts) {
                p.read(frame, n, tmp)
                for (i in 0 until n * 2) out[i] += tmp[i]
            }
        }
    }

    /** a − b, times gain: what one build changed compared with another. */
    class DiffSource(private val a: Source, private val b: Source, private val gain: Float) : Source {
        private var tmp = FloatArray(0)
        override fun read(frame: Long, n: Int, out: FloatArray) {
            if (tmp.size < n * 2) tmp = FloatArray(n * 2)
            a.read(frame, n, out)
            b.read(frame, n, tmp)
            for (i in 0 until n * 2) out[i] = (out[i] - tmp[i]) * gain
        }
    }
}
