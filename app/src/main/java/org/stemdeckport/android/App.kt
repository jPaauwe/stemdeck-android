package org.stemdeckport.android

import android.app.ActivityManager
import android.app.Application
import org.stemdeckport.android.data.JobStore
import org.stemdeckport.android.data.Settings
import org.stemdeckport.android.engine.ModelStore
import org.stemdeckport.android.engine.NativeEngine
import org.stemdeckport.android.server.LocalServer
import org.stemdeckport.android.work.SeparationService
import java.io.File

class App : Application() {
    lateinit var store: JobStore
        private set
    lateinit var models: ModelStore
        private set
    lateinit var settings: Settings
        private set
    lateinit var lyrics: org.stemdeckport.android.data.LyricsStore
        private set

    /** What LRCLIB is told the app is (it asks clients for name, version and homepage). */
    val userAgent: String by lazy {
        val v = try { packageManager.getPackageInfo(packageName, 0).versionName } catch (e: Exception) { null } ?: "dev"
        "StemDeck-Android/$v (https://github.com/jPaauwe/stemdeck-android)"
    }

    /** Started on first use, so a process woken only for the service skips it. */
    val server: LocalServer by lazy {
        LocalServer(this, store, settings) { SeparationService.start(this) }.also { it.startServer() }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Diagnostics.install(this)
        // App-specific external storage: stems stay reachable over USB
        // (Android/data/org.stemdeckport.android/files/jobs) and are removed
        // with the app.
        val base = getExternalFilesDir(null) ?: filesDir
        store = JobStore(File(base, "jobs"))
        models = ModelStore(File(filesDir, "models"))
        settings = Settings(this)
        lyrics = org.stemdeckport.android.data.LyricsStore(store)
        store.onStopRequested = { NativeEngine.cancel() }
    }

    /**
     * How many segments to separate at once. Each holds its own inference
     * buffers, so this is bounded by memory; the maths inside them is
     * spread over every core (and the GPU) regardless.
     */
    fun separationThreads(): Int {
        val mi = ActivityManager.MemoryInfo()
        (getSystemService(ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        val gb = mi.totalMem / 1e9
        // ~0.8 GB per worker for 24 s chunks, plus ~0.3 GB for the model.
        val byMemory = when {
            gb >= 11 -> 3
            gb >= 7 -> 2
            else -> 1
        }
        return minOf(byMemory, maxOf(1, Runtime.getRuntime().availableProcessors()))
    }

    companion object {
        lateinit var instance: App
            private set
    }
}
