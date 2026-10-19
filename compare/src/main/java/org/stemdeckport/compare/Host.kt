package org.stemdeckport.compare

fun interface Progress {
    fun onProgress(fraction: Float): Boolean
}

/** One engine version bundled in this APK. */
data class EngineInfo(val id: String, val label: String, val description: String)

/** See compare_jni.cpp. One engine plugin is open at a time. */
object Host {
    init { System.loadLibrary("sdcompare") }

    @JvmStatic private external fun nEngines(): String
    @JvmStatic external fun nOpen(id: String): String
    @JvmStatic external fun nClose()
    @JvmStatic external fun nLoadModel(path: String): String
    @JvmStatic external fun nPrepare(raw: String, rate: Int, channels: Int, out: String, p: Progress?): Int
    @JvmStatic external fun nSeparate(prepared: String, outDir: String, threads: Int, chunk: Int, p: Progress?): String
    @JvmStatic external fun nCancel()
    @JvmStatic external fun nDiff(refDir: String, dir: String): String
    @JvmStatic external fun nReconstruction(dir: String, prepared: String): String

    val engines: List<EngineInfo> by lazy {
        nEngines().lines().filter { it.isNotBlank() }.map {
            val f = it.split('|')
            EngineInfo(f[0], f.getOrElse(1) { f[0] }, f.getOrElse(2) { "" })
        }
    }
}
