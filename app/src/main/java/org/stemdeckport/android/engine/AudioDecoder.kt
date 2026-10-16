package org.stemdeckport.android.engine

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

/**
 * Decodes the audio track of any file Android can play (MP3, WAV, FLAC, OGG,
 * Opus, M4A/AAC, MP4) to raw interleaved float32 with the platform decoders.
 */
object AudioDecoder {
    data class Decoded(val sampleRate: Int, val channels: Int, val frames: Long)

    /** Two hours at most; anything longer is almost certainly a mistake. */
    private const val MAX_SECONDS = 2 * 60 * 60

    @Throws(IOException::class)
    fun decode(input: File, rawOut: File, onProgress: (Float) -> Boolean): Decoded {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(input.absolutePath)
        } catch (e: Exception) {
            extractor.release()
            throw IOException("This file could not be opened as audio.")
        }
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                trackIndex = i
                format = f
                break
            }
        }
        if (trackIndex < 0 || format == null) {
            extractor.release()
            throw IOException("This file has no audio track.")
        }
        extractor.selectTrack(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
        if (durationUs > MAX_SECONDS * 1_000_000L) {
            extractor.release()
            throw IOException("This file is longer than two hours.")
        }
        format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_FLOAT)

        val codec = try {
            MediaCodec.createDecoderByType(mime).apply { configure(format, null, null, 0) }
        } catch (e: Exception) {
            extractor.release()
            throw IOException("This phone cannot decode $mime audio.")
        }

        var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var encoding = AudioFormat.ENCODING_PCM_16BIT
        var frames = 0L
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var lastReport = 0L
        val scratch = ByteBuffer.allocate(1 shl 18).order(ByteOrder.LITTLE_ENDIAN)

        codec.start()
        try {
            rawOut.outputStream().channel.use { ch ->
                while (!outputDone) {
                    if (!inputDone) {
                        val inIdx = codec.dequeueInputBuffer(10_000)
                        if (inIdx >= 0) {
                            val buf = codec.getInputBuffer(inIdx)!!
                            val n = extractor.readSampleData(buf, 0)
                            if (n < 0) {
                                codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                    val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                    when {
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val of = codec.outputFormat
                            sampleRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            encoding = if (of.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                of.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            } else AudioFormat.ENCODING_PCM_16BIT
                        }
                        outIdx >= 0 -> {
                            val out = codec.getOutputBuffer(outIdx)!!
                            if (info.size > 0) {
                                out.position(info.offset)
                                out.limit(info.offset + info.size)
                                frames += writeAsFloat(out.order(ByteOrder.LITTLE_ENDIAN), encoding, scratch, ch) / channels
                            }
                            codec.releaseOutputBuffer(outIdx, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                            val now = System.currentTimeMillis()
                            if (now - lastReport > 200) {
                                lastReport = now
                                val frac = if (durationUs > 0) (info.presentationTimeUs.toFloat() / durationUs).coerceIn(0f, 1f) else 0f
                                if (!onProgress(frac)) throw CancellationException("cancelled")
                            }
                        }
                    }
                }
            }
        } finally {
            try { codec.stop() } catch (_: Exception) {}
            codec.release()
            extractor.release()
        }
        if (frames < 1024) throw IOException("The file decoded to no audio.")
        return Decoded(sampleRate, channels, frames)
    }

    /** Converts one decoder buffer to float32 and writes it. Returns samples written. */
    private fun writeAsFloat(
        src: ByteBuffer, encoding: Int, scratch: ByteBuffer, ch: java.nio.channels.FileChannel,
    ): Long {
        var samples = 0L
        when (encoding) {
            AudioFormat.ENCODING_PCM_FLOAT -> {
                samples = (src.remaining() / 4).toLong()
                while (src.hasRemaining()) ch.write(src)
            }
            AudioFormat.ENCODING_PCM_16BIT, AudioFormat.ENCODING_PCM_8BIT,
            AudioFormat.ENCODING_PCM_24BIT_PACKED, AudioFormat.ENCODING_PCM_32BIT -> {
                val bytesPer = when (encoding) {
                    AudioFormat.ENCODING_PCM_8BIT -> 1
                    AudioFormat.ENCODING_PCM_24BIT_PACKED -> 3
                    AudioFormat.ENCODING_PCM_32BIT -> 4
                    else -> 2
                }
                while (src.remaining() >= bytesPer) {
                    scratch.clear()
                    while (scratch.remaining() >= 4 && src.remaining() >= bytesPer) {
                        val v: Float = when (bytesPer) {
                            1 -> ((src.get().toInt() and 0xff) - 128) / 128f
                            3 -> {
                                val b0 = src.get().toInt() and 0xff
                                val b1 = src.get().toInt() and 0xff
                                val b2 = src.get().toInt()
                                ((b2 shl 16) or (b1 shl 8) or b0) / 8388608f
                            }
                            4 -> src.getInt() / 2147483648f
                            else -> src.getShort() / 32768f
                        }
                        scratch.putFloat(v)
                        samples++
                    }
                    scratch.flip()
                    while (scratch.hasRemaining()) ch.write(scratch)
                }
            }
            else -> throw IOException("Unsupported decoder output (encoding $encoding).")
        }
        return samples
    }
}
