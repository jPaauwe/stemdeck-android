package org.stemdeckport.bench

/** The same native API, implemented by two engines (see bench_jni.cpp). */
interface EngineNative {
    val label: String
    fun load(path: String): String
    fun free()
    fun profile(out: String): String
    fun writeSong(path: String, seconds: Double): Boolean
    fun compare(a: String, b: String): String
    fun separate(model: String, song: String, outDir: String, threads: Int, chunkSeconds: Int): String
}

/** The engine being optimized, with per-operation profiling. */
object OptNative : EngineNative {
    init { System.loadLibrary("sdbench_opt") }
    override val label = "optimized"
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

/** The engine as shipped in build-10, frozen. */
object BaseNative : EngineNative {
    init { System.loadLibrary("sdbench_base") }
    override val label = "baseline"
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
