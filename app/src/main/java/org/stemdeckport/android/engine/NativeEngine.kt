package org.stemdeckport.android.engine

/** Progress from the native engine. Return false to ask it to stop. */
fun interface ProgressListener {
    fun onProgress(fraction: Float, stage: String): Boolean
}

/**
 * The on-device separation engine: demucs.cpp running htdemucs_6s, the same
 * model desktop StemDeck uses. See app/src/main/cpp/engine.hpp.
 */
object NativeEngine {
    init {
        System.loadLibrary("sdengine")
    }

    /** The six sources, in the order the model produces them. */
    val SOURCES: List<String> = listOf("drums", "bass", "other", "vocals", "guitar", "piano")

    @JvmStatic private external fun nativeLoadModel(path: String): String
    @JvmStatic private external fun nativePrepareInput(
        raw: String, sampleRate: Int, channels: Int, out: String, listener: ProgressListener?,
    ): String
    @JvmStatic private external fun nativeAnalyze(path: String): DoubleArray
    @JvmStatic private external fun nativeSeparate(
        prepared: String, outDir: String, write: BooleanArray, threads: Int,
        chunkSeconds: Int, computeThreads: Int, gpu: Boolean, listener: ProgressListener?,
    ): DoubleArray
    @JvmStatic private external fun nativeComputeInfo(computeThreads: Int, gpu: Boolean): String
    @JvmStatic private external fun nativeCancel()

    /**
     * What separation runs on, e.g. "10 cores, GPU: Samsung Xclipse 940
     * (Vulkan 1.3)", or why the GPU is not used. Starts the GPU (and its
     * self-test) on first call.
     */
    fun computeInfo(gpu: Boolean, computeThreads: Int = 0): String = nativeComputeInfo(computeThreads, gpu)

    /** Throws IllegalStateException with a readable message on failure. */
    @Synchronized
    fun loadModel(path: String) {
        val err = nativeLoadModel(path)
        if (err.isNotEmpty()) throw IllegalStateException(err)
    }

    /**
     * Raw interleaved float32 at any rate/channels → stereo float32 at 44.1 kHz.
     * Throws CancellationException when cancelled.
     */
    fun prepareInput(raw: String, sampleRate: Int, channels: Int, out: String, listener: ProgressListener) {
        val err = nativePrepareInput(raw, sampleRate, channels, out, listener)
        if (err.isNotEmpty()) throw IllegalStateException(err)
    }

    data class InputStats(val durationSec: Double, val peakDb: Double?, val lufs: Double?)

    fun analyze(prepared: String): InputStats {
        val v = nativeAnalyze(prepared)
        return InputStats(
            durationSec = v[0],
            peakDb = v[1].takeUnless { it.isNaN() },
            lufs = v[2].takeUnless { it.isNaN() },
        )
    }

    /**
     * Writes <outDir>/<source>.wav for every selected source and returns the
     * RMS of each (0 for sources not written), in [SOURCES] order. [threads]
     * segments run at once; their maths uses every core, and the GPU when
     * [gpu] is on. Throws CancellationException when cancelled,
     * RuntimeException on failure.
     */
    fun separate(
        prepared: String, outDir: String, selected: Set<String>, threads: Int,
        chunkSeconds: Int, gpu: Boolean, listener: ProgressListener,
    ): DoubleArray {
        val write = BooleanArray(SOURCES.size) { SOURCES[it] in selected }
        return nativeSeparate(prepared, outDir, write, threads, chunkSeconds, 0, gpu, listener)
    }

    fun cancel() = nativeCancel()
}
