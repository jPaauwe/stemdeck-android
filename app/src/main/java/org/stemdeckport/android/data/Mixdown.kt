package org.stemdeckport.android.data

import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/** Mixes 16-bit stereo 44.1 kHz stems (as the engine writes them) into one WAV. */
object Mixdown {
    private const val HEADER = 44

    fun outputBytes(stems: List<File>): Long {
        val frames = stems.maxOf { (it.length() - HEADER).coerceAtLeast(0) / 4 }
        return HEADER + frames * 4
    }

    @Throws(IOException::class)
    fun write(stems: List<File>, gains: List<Float>, out: OutputStream) {
        require(stems.isNotEmpty() && stems.size == gains.size)
        val frames = stems.maxOf { (it.length() - HEADER).coerceAtLeast(0) / 4 }
        out.write(header(frames))
        val files = stems.map { RandomAccessFile(it, "r").apply { seek(HEADER.toLong()) } }
        try {
            val block = 16384
            val inBuf = ByteArray(block * 4)
            val acc = FloatArray(block * 2)
            val outBuf = ByteBuffer.allocate(block * 4).order(ByteOrder.LITTLE_ENDIAN)
            var done = 0L
            while (done < frames) {
                val n = minOf(block.toLong(), frames - done).toInt()
                acc.fill(0f, 0, n * 2)
                for ((i, f) in files.withIndex()) {
                    val g = gains[i]
                    val got = readFully(f, inBuf, n * 4)
                    val bb = ByteBuffer.wrap(inBuf, 0, got).order(ByteOrder.LITTLE_ENDIAN)
                    var j = 0
                    while (bb.remaining() >= 2) {
                        acc[j++] += bb.short * g
                    }
                }
                outBuf.clear()
                for (j in 0 until n * 2) {
                    outBuf.putShort(acc[j].roundToInt().coerceIn(-32768, 32767).toShort())
                }
                out.write(outBuf.array(), 0, n * 4)
                done += n
            }
        } finally {
            files.forEach { it.close() }
        }
        out.flush()
    }

    private fun readFully(f: RandomAccessFile, buf: ByteArray, len: Int): Int {
        var got = 0
        while (got < len) {
            val r = f.read(buf, got, len - got)
            if (r < 0) break
            got += r
        }
        return got
    }

    private fun header(frames: Long): ByteArray {
        val data = (frames * 4).toInt()
        return ByteBuffer.allocate(HEADER).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + data); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(2)
            putInt(44100); putInt(44100 * 4); putShort(4); putShort(16)
            put("data".toByteArray()); putInt(data)
        }.array()
    }
}
