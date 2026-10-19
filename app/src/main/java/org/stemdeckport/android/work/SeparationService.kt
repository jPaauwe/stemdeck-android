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
import org.stemdeckport.android.engine.Models
import org.stemdeckport.android.engine.Mp3Codec
import org.stemdeckport.android.engine.NativeAnalysis
import org.stemdeckport.android.engine.NativeEngine
import org.stemdeckport.android.ui.MainActivity
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
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
        if (!startInForeground(buildNotification("StemDeck", "Getting ready…", null))) {
            // Android refused to run the service. Fail the queued songs with a
            // message instead of crashing, so the app still opens.
            while (true) {
                val job = app.store.nextQueued() ?: break
                app.store.update(job.id) {
                    it.status = "error"
                    it.stage = "Failed"
                    it.error = "Android did not allow StemDeck to separate in the background."
                }
            }
            stopSelf()
            return START_NOT_STICKY
        }
        synchronized(lock) {
            if (!running) {
                running = true
                thread(name = "separation", priority = Thread.NORM_PRIORITY) { runQueue() }
            }
        }
        return START_NOT_STICKY
    }

    /**
     * Calls the framework directly: androidx ServiceCompat masks out service
     * types newer than the ones it knows, which turned mediaProcessing into
     * "none" and made Android 15+ kill the app. Falls back to dataSync if
     * mediaProcessing is refused. Returns false if Android refuses both.
     */
    private fun startInForeground(n: Notification): Boolean {
        val types = if (Build.VERSION.SDK_INT >= 35) {
            listOf(ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            listOf(ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        }
        for (type in types) {
            try {
                startForeground(NOTIFICATION_ID, n, type)
                return true
            } catch (e: Exception) {
                Log.w(TAG, "startForeground with type $type refused", e)
            }
        }
        return false
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
            if (!app.models.isReady(Models.DEMUCS)) {
                set("downloading", "Downloading the separation model (first time only, 55 MB)", 0.0, persist = true)
            }
            val modelFile = app.models.ensure(Models.DEMUCS) { f ->
                set("downloading", "Downloading the separation model (first time only, 55 MB)", 0.04 * f)
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

            // 4. Separation. Every source is written, the ones the user did not
            // pick only until the analysis below has used them (the beat grid
            // needs the drums, the key the harmony and the bass).
            val selected = job.selectedStems.filter { it in NativeEngine.SOURCES }.ifEmpty { NativeEngine.SOURCES }
            val stemsDir = store.stemsDir(id).apply { mkdirs() }
            val mp3 = job.format == "mp3"
            val sepEnd = 0.80
            // Every core, plus the GPU when it is on and passes its self-test.
            val useGpu = app.settings.gpu
            val compute = NativeEngine.computeInfo(useGpu)
            Log.i(TAG, "job $id: $compute")
            val cores = compute.substringBefore(',').trim()
            val onGpu = compute.contains("GPU: ")
            val sepStage = "Separating stems ($cores${if (onGpu) " + GPU" else ""})"
            store.update(id) { it.computeDevice = compute }
            set("separating", sepStage, 0.10, persist = true)
            var mark = SystemClock.elapsedRealtime()
            val timings = LinkedHashMap<String, Double>()
            fun lap(name: String) {
                val now = SystemClock.elapsedRealtime()
                timings[name] = Math.round((now - mark) / 100.0) / 10.0
                mark = now
            }
            val rms = NativeEngine.separate(
                prepared.path, stemsDir.path, NativeEngine.SOURCES.toSet(), app.separationThreads(), CHUNK_SECONDS, useGpu,
            ) { f, _ ->
                set("separating", sepStage, 0.10 + (sepEnd - 0.10) * f); !stopped()
            }
            lap("separate")
            val written = NativeEngine.SOURCES.filter { it in selected }

            // 5. Analysis, each stage best-effort: a failure leaves that
            // feature unavailable (and says why) but never fails the song.
            val notes = LinkedHashMap<String, String>()
            analyzeBeats(id, stemsDir, notes, ::set, stopped, 0.80, 0.90)
            lap("beatgrid")
            analyzeKey(id, stemsDir, notes, ::set, 0.90)
            lap("key")
            store.update(id) { it.analysisNotes = notes }
            for (s in NativeEngine.SOURCES) if (s !in written) File(stemsDir, "$s.wav").delete()

            // 6. MP3 stems (320 kbps), unless the user chose WAV.
            val mp3Start = 0.93
            if (mp3) {
                set("separating", "Saving MP3 stems", mp3Start, persist = true)
                encodeMp3(written.map { File(stemsDir, "$it.wav") }) { f ->
                    set("separating", "Saving MP3 stems", mp3Start + (0.99 - mp3Start) * f); !stopped()
                }
                lap("mp3")
            }

            // 7. Done. Presence = each stem's RMS as a share of the loudest,
            // as desktop computes it.
            val loudest = written.maxOf { rms[NativeEngine.SOURCES.indexOf(it)] }.takeIf { it > 0 } ?: 1.0
            val presence = written.associateWith { (rms[NativeEngine.SOURCES.indexOf(it)] / loudest * 100).roundToInt() }
            store.update(id) {
                it.status = "done"
                it.stage = "Done"
                it.progress = 1.0
                it.stems = written
                it.stemPresence = presence
                it.computeSeconds = (SystemClock.elapsedRealtime() - t0) / 1000.0
                it.stageTimings = timings
            }
            work.deleteRecursively()
            Log.i(TAG, "job $id done in ${(SystemClock.elapsedRealtime() - t0) / 1000}s: $timings")
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

    /**
     * The click track's beat grid, from the drums stem (desktop's
     * compute_beat_grid). Downloads the beat model on first use. Writes
     * stems/beats.json and the job's BPM and tempo stability.
     */
    private fun analyzeBeats(
        id: String, stemsDir: File, notes: MutableMap<String, String>,
        set: (String, String, Double, Boolean) -> Unit, stopped: () -> Boolean, from: Double, to: Double,
    ) {
        val store = app.store
        try {
            val dlStage = "Downloading the beat model (first time only, 82 MB)"
            if (!app.models.isReady(Models.BEAT_THIS)) set("analyzing", dlStage, from, true)
            val model = app.models.ensure(Models.BEAT_THIS) { f ->
                set("analyzing", dlStage, from + 0.3 * (to - from) * f, false); !stopped()
            }
            val start = from + 0.3 * (to - from)
            set("analyzing", "Finding the beat", start, true)
            val why = StringBuilder()
            val grid = NativeAnalysis.beatGrid(
                File(stemsDir, "drums.wav"), "drums", model, 0,
                { f, _ -> set("analyzing", "Finding the beat", start + (to - start) * f, false); !stopped() }, why,
            )
            if (grid == null) {
                notes["beats"] = "No beat grid: $why."
                return
            }
            val tmp = File(stemsDir, "beats.json.tmp")
            tmp.writeText(grid.json.toString())
            if (!tmp.renameTo(File(stemsDir, "beats.json"))) throw IllegalStateException("could not save beats.json")
            store.update(id) {
                it.bpm = grid.bpm.takeIf { b -> b > 0 }?.roundToInt()
                it.tempoStability = grid.tempoStability
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            Log.w(TAG, "beat grid failed for $id", e)
            notes["beats"] = "No beat grid: " + when (e) {
                is java.io.IOException -> "the beat model could not be downloaded (${e.message})."
                is OutOfMemoryError -> "the phone ran out of memory."
                else -> (e.message ?: e.javaClass.simpleName) + "."
            }
        }
    }

    /**
     * The key from the separated harmony stems and the bass, as desktop's
     * refine_key_from_stems does it (vocals count for half).
     */
    private fun analyzeKey(
        id: String, stemsDir: File, notes: MutableMap<String, String>,
        set: (String, String, Double, Boolean) -> Unit, at: Double,
    ) {
        try {
            set("analyzing", "Finding the key", at, true)
            fun stem(n: String) = File(stemsDir, "$n.wav").takeIf { it.isFile }
            val harmony = listOf("guitar" to 1.0, "piano" to 1.0, "other" to 1.0, "vocals" to 0.5)
                .mapNotNull { (n, w) -> stem(n)?.let { it to w } }
            val k = NativeAnalysis.key(harmony, stem("bass"))
            if (k == null) {
                notes["key"] = "No key: the harmony stems are silent."
                return
            }
            app.store.update(id) {
                it.key = k.key
                it.scale = k.scale
                it.keyConfidence = k.confidence
            }
        } catch (e: Throwable) {
            Log.w(TAG, "key detection failed for $id", e)
            notes["key"] = "No key: " + (e.message ?: e.javaClass.simpleName) + "."
        }
    }

    /**
     * Encodes each WAV to a 320 kbps MP3 next to it, a few at a time, and
     * deletes the WAV. [progress] is called on this thread; returning false
     * stops the encoding with a CancellationException.
     */
    private fun encodeMp3(wavs: List<File>, progress: (Double) -> Boolean) {
        val done = DoubleArray(wavs.size)
        val stop = AtomicBoolean(false)
        val errors = ConcurrentLinkedQueue<Throwable>()
        val next = AtomicInteger(0)
        val workers = minOf(wavs.size, maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        val threads = List(workers) { w ->
            thread(name = "mp3-$w") {
                while (!stop.get()) {
                    val i = next.getAndIncrement()
                    if (i >= wavs.size) break
                    val wav = wavs[i]
                    val mp3 = File(wav.parentFile, wav.nameWithoutExtension + ".mp3")
                    try {
                        Mp3Codec.encode(wav, mp3) { f -> done[i] = f.toDouble(); !stop.get() }
                        wav.delete()
                    } catch (e: Throwable) {
                        mp3.delete()
                        errors += e
                        stop.set(true)
                    }
                }
            }
        }
        while (threads.any { it.isAlive }) {
            if (!progress(done.average())) stop.set(true)
            threads.firstOrNull { it.isAlive }?.join(250)
        }
        errors.firstOrNull { it is CancellationException }?.let { throw it }
        errors.firstOrNull()?.let { throw it }
        if (stop.get()) throw CancellationException()
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
