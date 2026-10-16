package org.stemdeckport.android.work

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import org.stemdeckport.android.App
import org.stemdeckport.android.R
import org.stemdeckport.android.data.Job
import org.stemdeckport.android.engine.AudioDecoder
import org.stemdeckport.android.engine.NativeEngine
import org.stemdeckport.android.ui.MainActivity
import java.io.File
import java.util.concurrent.CancellationException
import kotlin.concurrent.thread
import kotlin.math.roundToInt

/**
 * Runs queued separations one after another, in the foreground with a
 * progress notification, so a separation keeps going with the screen off.
 */
class SeparationService : Service() {
    companion object {
        private const val TAG = "SeparationService"
        private const val CHANNEL = "separation"
        private const val NOTIFICATION_ID = 1
        const val ACTION_CANCEL = "org.stemdeckport.android.CANCEL"
        private const val CHUNK_SECONDS = 24

        fun start(context: Context) {
            try {
                ContextCompat.startForegroundService(context, Intent(context, SeparationService::class.java))
            } catch (e: Exception) {
                // Starting from the background is refused on Android 12+; the
                // queue is picked up the next time the app is opened.
                Log.w(TAG, "could not start the separation service", e)
            }
        }
    }

    private val app get() = application as App
    private val lock = Any()
    private var running = false
    private var lastNotify = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            app.store.runningJob()?.let { app.store.requestStop(it.id, "cancelled") }
            return START_NOT_STICKY
        }
        createChannel()
        startInForeground(buildNotification("StemDeck", "Getting ready…", null))
        synchronized(lock) {
            if (!running) {
                running = true
                thread(name = "separation", priority = Thread.NORM_PRIORITY) { runQueue() }
            }
        }
        return START_NOT_STICKY
    }

    private fun startInForeground(n: Notification) {
        val type = if (Build.VERSION.SDK_INT >= 35) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        } else {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, n, type)
    }

    private fun runQueue() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "StemDeck:separation")
        wake.acquire(6 * 60 * 60 * 1000L)
        try {
            while (true) {
                val job = synchronized(lock) {
                    app.store.nextQueued() ?: run { running = false; null }
                } ?: break
                process(job)
            }
        } finally {
            if (wake.isHeld) wake.release()
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun process(job: Job) {
        val store = app.store
        val id = job.id
        val work = store.workDir(id).apply { mkdirs() }
        val raw = File(work, "raw.f32")
        val prepared = File(work, "prepared.f32")
        val t0 = SystemClock.elapsedRealtime()
        val stopped = { store.stopRequested(id) != null }

        fun set(status: String, stage: String, progress: Double, persist: Boolean = false) {
            store.update(id, persist) {
                it.status = status
                it.stage = stage
                it.progress = progress
            }
            notifyProgress(job.title, stage, progress)
        }

        try {
            // 1. The model, downloaded on first use.
            if (!app.models.isReady()) {
                set("downloading", "Downloading the separation model (first time only)", 0.0, persist = true)
            }
            val modelFile = app.models.ensure { f ->
                set("downloading", "Downloading the separation model (first time only)", 0.04 * f)
                !stopped()
            }
            set("analyzing", "Loading the model", 0.04, persist = true)
            NativeEngine.loadModel(modelFile.absolutePath)

            // 2. Decode and bring to 44.1 kHz stereo.
            val input = store.inputFile(id) ?: throw IllegalStateException("The imported file is missing.")
            set("analyzing", "Decoding audio", 0.05)
            val dec = AudioDecoder.decode(input, raw) { f ->
                set("analyzing", "Decoding audio", 0.05 + 0.03 * f); !stopped()
            }
            NativeEngine.prepareInput(raw.path, dec.sampleRate, dec.channels, prepared.path) { f, stage ->
                set("analyzing", stage, 0.08 + 0.02 * f); !stopped()
            }
            raw.delete()

            // 3. Loudness, peak and length of the whole song.
            val stats = NativeEngine.analyze(prepared.path)
            store.update(id) {
                it.duration = stats.durationSec
                it.lufs = stats.lufs?.let { v -> Math.round(v * 10) / 10.0 }
                it.peakDb = stats.peakDb?.let { v -> Math.round(v * 10) / 10.0 }
                it.dynamicRange = if (stats.lufs != null && stats.peakDb != null) {
                    Math.round((stats.peakDb - stats.lufs) * 10) / 10.0
                } else null
            }

            // 4. Separation.
            val selected = job.selectedStems.filter { it in NativeEngine.SOURCES }.ifEmpty { NativeEngine.SOURCES }
            val stemsDir = store.stemsDir(id).apply { mkdirs() }
            set("separating", "Separating stems", 0.10, persist = true)
            val rms = NativeEngine.separate(
                prepared.path, stemsDir.path, selected.toSet(), app.separationThreads(), CHUNK_SECONDS,
            ) { f, _ ->
                set("separating", "Separating stems", 0.10 + 0.89 * f); !stopped()
            }

            // 5. Done. Presence = each stem's RMS as a share of the loudest,
            // as desktop computes it.
            val written = NativeEngine.SOURCES.filter { it in selected }
            val loudest = written.maxOf { rms[NativeEngine.SOURCES.indexOf(it)] }.takeIf { it > 0 } ?: 1.0
            val presence = written.associateWith { (rms[NativeEngine.SOURCES.indexOf(it)] / loudest * 100).roundToInt() }
            store.update(id) {
                it.status = "done"
                it.stage = "Done"
                it.progress = 1.0
                it.stems = written
                it.stemPresence = presence
                it.computeSeconds = (SystemClock.elapsedRealtime() - t0) / 1000.0
            }
            work.deleteRecursively()
            Log.i(TAG, "job $id done in ${(SystemClock.elapsedRealtime() - t0) / 1000}s")
        } catch (e: CancellationException) {
            val terminal = store.stopRequested(id) ?: "cancelled"
            store.update(id) {
                it.status = terminal
                it.stage = terminal.replaceFirstChar(Char::uppercase)
            }
            store.stemsDir(id).listFiles()?.forEach { it.delete() }
            work.deleteRecursively()
        } catch (e: Throwable) {
            Log.e(TAG, "job $id failed", e)
            val msg = when (e) {
                is OutOfMemoryError -> "The phone ran out of memory."
                else -> e.message ?: e.javaClass.simpleName
            }
            store.update(id) {
                it.status = "error"
                it.stage = "Failed"
                it.error = msg
            }
            store.stemsDir(id).listFiles()?.forEach { it.delete() }
            work.deleteRecursively()
        } finally {
            store.clearStop(id)
            raw.delete()
            prepared.delete()
        }
    }

    private fun notifyProgress(title: String, stage: String, progress: Double) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastNotify < 1000 && progress < 1.0) return
        lastNotify = now
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, buildNotification(title, "$stage · ${(progress * 100).roundToInt()}%", progress))
    }

    private fun buildNotification(title: String, text: String?, progress: Double?): Notification {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).putExtra(MainActivity.EXTRA_TAB, "extract"),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val cancel = PendingIntent.getService(
            this, 1,
            Intent(this, SeparationService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_stemdeck)
            .setContentTitle(title)
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setSilent(true)
            .setContentIntent(open)
            .setProgress(1000, ((progress ?: 0.0) * 1000).roundToInt(), progress == null)
            .addAction(0, getString(R.string.cancel), cancel)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    private fun createChannel() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, getString(R.string.channel_separation), NotificationManager.IMPORTANCE_LOW),
            )
        }
    }
}
