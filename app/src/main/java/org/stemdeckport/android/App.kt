package org.stemdeckport.android

import android.app.ActivityManager
import android.app.Application
import org.stemdeckport.android.data.JobStore
import org.stemdeckport.android.engine.ModelManager
import org.stemdeckport.android.engine.NativeEngine
import org.stemdeckport.android.server.LocalServer
import org.stemdeckport.android.work.SeparationService
import java.io.File

class App : Application() {
    lateinit var store: JobStore
        private set
    lateinit var models: ModelManager
        private set

    /** Started on first use, so a process woken only for the service skips it. */
    val server: LocalServer by lazy {
        LocalServer(this, store) { SeparationService.start(this) }.also { it.startServer() }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        // App-specific external storage: stems stay reachable over USB
        // (Android/data/org.stemdeckport.android/files/jobs) and are removed
        // with the app.
        val base = getExternalFilesDir(null) ?: filesDir
        store = JobStore(File(base, "jobs"))
        models = ModelManager(this)
        store.onStopRequested = { NativeEngine.cancel() }
    }

    /**
     * How many chunks to separate at once. Each worker holds its own
     * inference buffers, so this is bounded by memory more than by cores.
     */
    fun separationThreads(): Int {
        val mi = ActivityManager.MemoryInfo()
        (getSystemService(ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        val gb = mi.totalMem / 1e9
        val byMemory = when {
            gb >= 11 -> 4
            gb >= 7 -> 3
            gb >= 5 -> 2
            else -> 1
        }
        return minOf(byMemory, maxOf(1, Runtime.getRuntime().availableProcessors() / 2))
    }

    companion object {
        lateinit var instance: App
            private set
    }
}
