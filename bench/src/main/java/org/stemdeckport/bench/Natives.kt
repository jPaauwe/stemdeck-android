package org.stemdeckport.bench

/** The same native API, implemented by each engine (see bench_jni.cpp). */
interface EngineNative {
    /** Short id, used in file names. */
    val key: String
    val label: String
    fun load(path: String): String
    fun free()
    fun profile(out: String): String
    fun writeSong(path: String, seconds: Double): Boolean
    fun compare(a: String, b: String): String
    fun separate(model: String, song: String, outDir: String, threads: Int, chunkSeconds: Int): String
}

/** The app's engine (build 13 + every core + GPU, from build 24), with per-operation profiling. */
object OptNative : EngineNative {
    init { System.loadLibrary("sdbench_opt") }
    override val key = "app"
    override val label = "app engine"
    @JvmStatic external fun nLoad(path: String): String
    @JvmStatic external fun nFree()
    @JvmStatic external fun nProfile(out: String): String
    @JvmStatic external fun nWriteSong(path: String, seconds: Double): Boolean
    @JvmStatic external fun nCompare(a: String, b: String): String
    @JvmStatic external fun nSeparate(model: String, song: String, outDir: String, threads: Int, chunk: Int): String
    /** Threads doing maths (0 = all cores) and the GPU from now on; returns what that is on this phone. */
    @JvmStatic external fun nSetCompute(threads: Int, gpu: Boolean): String
    override fun load(path: String) = nLoad(path)
    override fun free() = nFree()
    override fun profile(out: String) = nProfile(out)
    override fun writeSong(path: String, seconds: Double) = nWriteSong(path, seconds)
    override fun compare(a: String, b: String) = nCompare(a, b)
    override fun separate(model: String, song: String, outDir: String, threads: Int, chunkSeconds: Int) =
        nSeparate(model, song, outDir, threads, chunkSeconds)
}

/** The first released engine, builds 4–10 (engines/b4). */
object BaseNative : EngineNative {
    init { System.loadLibrary("sdbench_base") }
    override val key = "b4"
    override val label = "builds 4–10"
    @JvmStatic external fun nLoad(path: String): String
    @JvmStatic external fun nFree()
    @JvmStatic external fun nProfile(out: String): String
    @JvmStatic external fun nWriteSong(path: String, seconds: Double): Boolean
    @JvmStatic external fun nCompare(a: String, b: String): String
    @JvmStatic external fun nSeparate(model: String, song: String, outDir: String, threads: Int, chunk: Int): String
    override fun load(path: String) = nLoad(path)
    override fun free() = nFree()
    override fun profile(out: String) = nProfile(out)
    override fun writeSong(path: String, seconds: Double) = nWriteSong(path, seconds)
    override fun compare(a: String, b: String) = nCompare(a, b)
    override fun separate(model: String, song: String, outDir: String, threads: Int, chunkSeconds: Int) =
        nSeparate(model, song, outDir, threads, chunkSeconds)
}

/** Upstream demucs.cpp, unmodified (engines/original). Separates on one thread. */
object OrigNative : EngineNative {
    init { System.loadLibrary("sdbench_orig") }
    override val key = "original"
    override val label = "original demucs.cpp"
    @JvmStatic external fun nLoad(path: String): String
    @JvmStatic external fun nFree()
    @JvmStatic external fun nProfile(out: String): String
    @JvmStatic external fun nWriteSong(path: String, seconds: Double): Boolean
    @JvmStatic external fun nCompare(a: String, b: String): String
    @JvmStatic external fun nSeparate(model: String, song: String, outDir: String, threads: Int, chunk: Int): String
    override fun load(path: String) = nLoad(path)
    override fun free() = nFree()
    override fun profile(out: String) = nProfile(out)
    override fun writeSong(path: String, seconds: Double) = nWriteSong(path, seconds)
    override fun compare(a: String, b: String) = nCompare(a, b)
    override fun separate(model: String, song: String, outDir: String, threads: Int, chunkSeconds: Int) =
        nSeparate(model, song, outDir, threads, chunkSeconds)
}

/**
 * The analysis stages the app runs after separation (/analysis): the beat
 * grid (Beat This! on ONNX Runtime) and the key, timed. Same code as the app.
 */
object AnalysisNative {
    init { System.loadLibrary("sdbench_analysis") }
    /** JSON: seconds per step, beat count and BPM, or an error. */
    @JvmStatic external fun nBeatGrid(wav: String, model: String, threads: Int, provider: String): String
    /** JSON: seconds and the key, from <dir>/{guitar,piano,other,vocals,bass}.wav. */
    @JvmStatic external fun nKey(dir: String): String
    @JvmStatic external fun nOrtVersion(): String
    /** JSON: seconds (spectrogram, eight folds, post-processing) and the section count. */
    @JvmStatic external fun nSections(dir: String, folds: Array<String>, threads: Int): String
    /** JSON: seconds (and in the model) for the lead/backing split of a vocals WAV. */
    @JvmStatic external fun nVocalSplit(vocals: String, outDir: String, model: String, threads: Int): String
}
