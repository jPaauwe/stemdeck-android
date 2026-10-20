package org.stemdeckport.android.data

import android.util.Log
import org.json.JSONObject
import java.io.File
import java.security.SecureRandom

/**
 * The library: every job's state lives in <root>/<id>/job.json, its stems in
 * <root>/<id>/stems/<name>.mp3 (or .wav). Thread-safe; changes wake anyone waiting in
 * [awaitChange] (the SSE stream).
 */
class JobStore(val root: File) {
    private val lock = Object()
    private val jobs = LinkedHashMap<String, Job>()
    private val stopRequests = HashMap<String, String>()
    private val rng = SecureRandom()

    /** Called when a running job is asked to stop, so the engine can be interrupted. */
    @Volatile var onStopRequested: ((String) -> Unit)? = null

    init {
        root.mkdirs()
        load()
    }

    fun jobDir(id: String) = File(root, id)
    fun stemsDir(id: String) = File(jobDir(id), "stems")
    fun workDir(id: String) = File(jobDir(id), "work")
    fun stemFile(id: String, name: String): File {
        val ext = synchronized(lock) { jobs[id]?.format } ?: "wav"
        return File(stemsDir(id), "$name.$ext")
    }

    private fun load() {
        val now = System.currentTimeMillis() / 1000.0
        root.listFiles()?.forEach { dir ->
            val f = File(dir, "job.json")
            if (!f.isFile) return@forEach
            try {
                val job = Job.fromRecord(JSONObject(f.readText()))
                // Trash older than 30 days is emptied.
                val trashed = job.trashedAt
                if (trashed != null && now - trashed > 30 * 86400) {
                    dir.deleteRecursively()
                    return@forEach
                }
                // A job that never started stays queued. One that was being
                // separated when the process died is not retried: if the
                // separation itself killed the app (a crash, or Android
                // reclaiming its memory), retrying it on every launch would
                // stop the app from opening at all.
                if (!job.finished && job.status != "queued") {
                    job.status = "error"
                    job.stage = "Failed"
                    job.error = "StemDeck stopped while separating this song (it may have run out of memory). Import it again to retry."
                    workDir(job.id).deleteRecursively()
                    stemsDir(job.id).listFiles()?.forEach { it.delete() }
                    persist(job)
                } else if (job.status == "queued" && inputFile(job.id) == null) {
                    job.status = "error"; job.stage = "Failed"
                    job.error = "The imported file is missing. Import it again."
                    persist(job)
                }
                jobs[job.id] = job
            } catch (e: Exception) {
                Log.w("JobStore", "skipping unreadable job in $dir", e)
            }
        }
    }

    fun list(): List<Job> = synchronized(lock) {
        jobs.values.filter { it.trashedAt == null }.sortedByDescending { it.createdAt }
    }

    fun get(id: String): Job? = synchronized(lock) { jobs[id] }

    fun hasQueued(): Boolean = synchronized(lock) {
        jobs.values.any { it.status == "queued" && it.trashedAt == null }
    }

    fun nextQueued(): Job? = synchronized(lock) {
        jobs.values.filter { it.status == "queued" && it.trashedAt == null }.minByOrNull { it.createdAt }
    }

    private fun newId(): String {
        val b = ByteArray(6)
        rng.nextBytes(b)
        return b.joinToString("") { "%02x".format(it) }
    }

    /** Creates a queued job; [input] is moved into the job's folder. */
    fun create(
        title: String, sourceUrl: String, selected: List<String>, input: File, format: String, autoSections: Boolean = false,
    ): Job {
        val id = newId()
        val work = workDir(id).apply { mkdirs() }
        stemsDir(id).mkdirs()
        val ext = input.name.substringAfterLast('.', "").take(8).lowercase()
        val dest = File(work, if (ext.isNotEmpty()) "input.$ext" else "input")
        if (!input.renameTo(dest)) {
            input.copyTo(dest, overwrite = true)
            input.delete()
        }
        val job = Job(id, title, sourceUrl, selected, System.currentTimeMillis() / 1000.0)
        job.format = if (format == "mp3") "mp3" else "wav"
        job.autoSections = autoSections
        synchronized(lock) {
            jobs[id] = job
            persist(job)
            lock.notifyAll()
        }
        return job
    }

    /** A queued job for a link (1.4): no input file yet, the service downloads it first. */
    fun createFromUrl(title: String, url: String, selected: List<String>, format: String, autoSections: Boolean): Job {
        val id = newId()
        workDir(id).mkdirs()
        stemsDir(id).mkdirs()
        val job = Job(id, title, url, selected, System.currentTimeMillis() / 1000.0)
        job.format = if (format == "mp3") "mp3" else "wav"
        job.autoSections = autoSections
        synchronized(lock) {
            jobs[id] = job
            persist(job)
            lock.notifyAll()
        }
        return job
    }

    fun inputFile(id: String): File? =
        workDir(id).listFiles()?.firstOrNull { it.name.startsWith("input") }

    /** Applies [block] and wakes waiters. Progress-only updates skip the disk. */
    fun update(id: String, persist: Boolean = true, block: (Job) -> Unit) {
        synchronized(lock) {
            val job = jobs[id] ?: return
            block(job)
            job.version++
            if (persist) persist(job)
            lock.notifyAll()
        }
    }

    /** Waits until [id] changes past [since] or [timeoutMs] passes. */
    fun awaitChange(id: String, since: Long, timeoutMs: Long): Job? {
        val deadline = System.currentTimeMillis() + timeoutMs
        synchronized(lock) {
            while (true) {
                val job = jobs[id] ?: return null
                if (job.version != since) return job
                val left = deadline - System.currentTimeMillis()
                if (left <= 0) return job
                lock.wait(left)
            }
        }
    }

    fun setFavorite(id: String, favorite: Boolean): Boolean {
        if (get(id) == null) return false
        update(id) { it.favorite = favorite }
        return true
    }

    /**
     * Moves a job to the Trash. A job still separating is stopped first and
     * keeps whatever it has, like desktop's "stopped" (#748).
     */
    fun trash(id: String): Boolean {
        val job = get(id) ?: return false
        if (!job.finished) requestStop(id, "stopped")
        update(id) { it.trashedAt = System.currentTimeMillis() / 1000.0 }
        return true
    }

    fun requestStop(id: String, terminal: String) {
        val job = get(id) ?: return
        if (job.finished && job.vocalSplit != "running") return
        if (job.status == "queued") {
            update(id) { it.status = terminal; it.stage = terminal.replaceFirstChar(Char::uppercase) }
            return
        }
        synchronized(lock) { stopRequests[id] = terminal }
        onStopRequested?.invoke(id)
    }

    // Android 1.2: lead/backing vocal splits waiting for the service, in order.
    private val splitQueue = ArrayDeque<String>()

    /** Queues desktop's on-demand vocal split for a done song; false if it is already running. */
    fun queueSplit(id: String): Boolean {
        synchronized(lock) {
            val job = jobs[id] ?: return false
            if (job.vocalSplit == "running") return false
            splitQueue.addLast(id)
        }
        update(id) {
            it.vocalSplit = "running"
            it.stage = "Splitting lead/backing vocals..."
            it.progress = 0.0
        }
        return true
    }

    fun nextSplit(): String? = synchronized(lock) { splitQueue.removeFirstOrNull() }

    fun hasSplits(): Boolean = synchronized(lock) { splitQueue.isNotEmpty() }

    /** Waits until [id]'s vocal split is no longer running. */
    fun awaitSplit(id: String): Job? {
        synchronized(lock) {
            while (true) {
                val job = jobs[id] ?: return null
                if (job.vocalSplit != "running") return job
                lock.wait(1000)
            }
        }
    }

    fun stopRequested(id: String): String? = synchronized(lock) { stopRequests[id] }
    fun clearStop(id: String) = synchronized(lock) { stopRequests.remove(id) }

    fun runningJob(): Job? = synchronized(lock) {
        jobs.values.firstOrNull { !it.finished && it.status != "queued" }
            ?: jobs.values.firstOrNull { it.vocalSplit == "running" }
    }

    private fun persist(job: Job) {
        try {
            val dir = jobDir(job.id).apply { mkdirs() }
            val tmp = File(dir, "job.json.tmp")
            tmp.writeText(job.toRecord().toString())
            if (!tmp.renameTo(File(dir, "job.json"))) Log.w("JobStore", "could not save ${job.id}")
        } catch (e: Exception) {
            Log.w("JobStore", "could not save ${job.id}", e)
        }
    }
}
