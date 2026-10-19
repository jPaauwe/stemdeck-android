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
import org.json.JSONObject
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
                // Songs first, then lead/backing splits asked for on done songs.
                val next: Any = synchronized(lock) {
                    app.store.nextQueued() ?: app.store.nextSplit() ?: run { running = false; null }
                } ?: break
                if (next is Job) process(next) else splitVocals(next as String)
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
            // With sections the analysis gets a bigger share of the bar.
            val withSections = job.autoSections
            val sepEnd = if (withSections) 0.70 else 0.80
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
            val beatEnd = if (withSections) 0.76 else 0.90
            analyzeBeats(id, stemsDir, notes, ::set, stopped, sepEnd, beatEnd)
            lap("beatgrid")
            analyzeKey(id, stemsDir, notes, ::set, beatEnd)
            lap("key")
            if (withSections) {
                analyzeSections(id, stemsDir, notes, ::set, stopped, 0.77, 0.92)
                lap("sections")
            }
            // Lyrics (1.3): only when the user turned the lookup on (Settings, Network).
            if (app.settings.lyricsLookup) {
                lookUpLyrics(id, notes, ::set, stopped, 0.92)
                lap("lyrics")
            }
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
     * Lyrics from LRCLIB for the song's artist/title/album/length (desktop's
     * lookup at import, its useful core). Best-effort, within 30 s.
     */
    private fun lookUpLyrics(
        id: String, notes: MutableMap<String, String>,
        set: (String, String, Double, Boolean) -> Unit, stopped: () -> Boolean, at: Double,
    ) {
        try {
            val job = app.store.get(id) ?: return
            val query = app.lyrics.query(job)
            if (query == null) {
                notes["lyrics"] = "No lyrics: the file names no song to look up (add the title in the Lyrics tab)."
                return
            }
            set("analyzing", "Looking up the lyrics", at, true)
            val answer = org.stemdeckport.android.data.LrcLib.lookup(query, org.stemdeckport.android.data.LrcLib.http(app.userAgent), stopped)
            if (stopped()) throw CancellationException()
            if (answer == null || !app.lyrics.keep(id, answer)) {
                notes["lyrics"] = "No lyrics: LRCLIB has none for “${query.artist.ifBlank { "?" }} – ${query.track}”."
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            Log.w(TAG, "lyrics lookup failed for $id", e)
            notes["lyrics"] = "No lyrics: LRCLIB could not be reached (${e.message ?: e.javaClass.simpleName})."
        }
    }

    /**
     * Desktop's on-demand lead/backing vocal split (POST /jobs/{id}/vocal-split,
     * app/pipeline/vocal_split.py): UVR-MDX-NET Karaoke 2 on the vocals stem.
     * The song stays done throughout; only vocal_split and the stage line
     * move. Adds lead_vocals and backing_vocals in the song's format.
     */
    private fun splitVocals(id: String) {
        val store = app.store
        val job = store.get(id) ?: return
        val stemsDir = store.stemsDir(id)
        val tmp = File(store.workDir(id), "split").apply { deleteRecursively(); mkdirs() }
        val stopped = { store.stopRequested(id) != null }
        fun stage(text: String, p: Double, persist: Boolean = false) {
            store.update(id, persist) { it.stage = text; it.progress = p }
            notifyProgress(job.title, text, p)
        }
        try {
            val dl = "Downloading the vocal split model (first time only, 53 MB)"
            if (!app.models.isReady(Models.KARAOKE)) stage(dl, 0.0, true)
            val model = app.models.ensure(Models.KARAOKE) { f -> stage(dl, 0.1 * f); !stopped() }
            val vocals = if (job.format == "mp3") {
                stage("Splitting lead/backing vocals...", 0.1, true)
                File(tmp, "vocals.wav").also { Mp3Codec.decode(store.stemFile(id, "vocals"), it) { !stopped() } }
            } else store.stemFile(id, "vocals")
            if (!vocals.isFile) throw IllegalStateException("This song has no vocals stem to split.")
            val lead = File(tmp, "lead_vocals.wav")
            val backing = File(tmp, "backing_vocals.wav")
            val why = StringBuilder()
            val r = NativeAnalysis.vocalSplit(
                vocals, lead, backing, model, app.separationThreads(),
                { f, _ -> stage("Splitting lead/backing vocals...", 0.12 + 0.8 * f); !stopped() }, why,
            ) ?: throw IllegalStateException("vocal split failed: $why")
            val names = listOf("lead_vocals" to lead, "backing_vocals" to backing)
            if (job.format == "mp3") {
                stage("Saving MP3 stems", 0.93, true)
                for ((name, wav) in names) Mp3Codec.encode(wav, File(stemsDir, "$name.mp3")) { !stopped() }
            } else {
                for ((name, wav) in names) {
                    val dest = File(stemsDir, "$name.wav")
                    if (!wav.renameTo(dest)) { wav.copyTo(dest, overwrite = true); wav.delete() }
                }
            }
            // desktop's presence_for_split: the loudest stem's RMS is recovered from vocals.
            val vocalsPct = job.stemPresence?.get("vocals")
            val extra = if (vocalsPct != null && vocalsPct > 0 && r.vocalsRms > 0) {
                val loudest = r.vocalsRms / (vocalsPct / 100.0)
                if (loudest < 1e-9) emptyMap() else mapOf(
                    "lead_vocals" to (r.leadRms / loudest * 100).roundToInt().coerceIn(0, 100),
                    "backing_vocals" to (r.backingRms / loudest * 100).roundToInt().coerceIn(0, 100),
                )
            } else emptyMap()
            store.update(id) {
                it.stems = it.stems.filter { s -> s !in setOf("lead_vocals", "backing_vocals") } + listOf("lead_vocals", "backing_vocals")
                if (extra.isNotEmpty()) it.stemPresence = (it.stemPresence ?: emptyMap()) + extra
                it.vocalSplit = "done"
                it.stage = "Done"
                it.progress = 1.0
            }
        } catch (e: CancellationException) {
            store.update(id) { it.vocalSplit = "none"; it.stage = "Done"; it.progress = 1.0 }
        } catch (e: Throwable) {
            Log.w(TAG, "vocal split failed for $id", e)
            try {
                File(stemsDir, "vocal_split_error.txt").writeText((e.message ?: e.javaClass.simpleName) + "\n")
            } catch (_: Exception) {
            }
            store.update(id) { it.vocalSplit = "error"; it.stage = "Done"; it.progress = 1.0 }
        } finally {
            store.clearStop(id)
            tmp.deleteRecursively()
        }
    }

    /**
     * Song sections (desktop's detect_sections): All-In-One on bass, drums,
     * vocals and other + guitar + piano, refined against the beat grid.
     * Downloads the section model on first use.
     */
    private fun analyzeSections(
        id: String, stemsDir: File, notes: MutableMap<String, String>,
        set: (String, String, Double, Boolean) -> Unit, stopped: () -> Boolean, from: Double, to: Double,
    ) {
        val store = app.store
        try {
            val duration = store.get(id)?.duration
            if (duration == null || duration <= 0) {
                notes["sections"] = "No sections: the song's length is unknown."
                return
            }
            val dlStage = "Downloading the section model (first time only, 25 MB)"
            if (!app.models.isReady(Models.SECTIONS)) set("analyzing", dlStage, from, true)
            val folds = app.models.ensureAll(Models.SECTIONS) { f ->
                set("analyzing", dlStage, from + 0.1 * (to - from) * f, false); !stopped()
            }
            val start = from + 0.1 * (to - from)
            set("analyzing", "Finding the song's sections", start, true)
            val grid = File(stemsDir, "beats.json").takeIf { it.isFile }?.let { JSONObject(it.readText()) }
            val why = StringBuilder()
            val r = NativeAnalysis.sections(
                stemsDir, folds, grid, duration, 0,
                { f, _ -> set("analyzing", "Finding the song's sections", start + (to - start) * f, false); !stopped() }, why,
            )
            if (r == null) {
                notes["sections"] = "No sections: $why."
                return
            }
            store.update(id) {
                it.sections = r.sections
                it.sectionsSource = "automatic"
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            Log.w(TAG, "sections failed for $id", e)
            notes["sections"] = "No sections: " + when (e) {
                is java.io.IOException -> "the section model could not be downloaded (${e.message})."
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
