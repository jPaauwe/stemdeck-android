package org.stemdeckport.android

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Keeps a record of why the app last stopped, so a crash on someone's phone
 * can be diagnosed afterwards: Kotlin exceptions are written as they happen,
 * and native crashes and out-of-memory kills are read back from Android's
 * exit records on the next launch. Everything is also logged under the
 * "StemDeckDiag" tag for `adb logcat`.
 */
object Diagnostics {
    private const val TAG = "StemDeckDiag"
    private lateinit var dir: File

    val reportFile: File get() = File(dir, "last-exit.txt")

    fun install(context: Context) {
        dir = File(context.filesDir, "diagnostics").apply { mkdirs() }

        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            try {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                write("Uncaught exception in thread ${thread.name} at ${now()}\n\n$sw")
            } catch (_: Throwable) {
            }
            previous?.uncaughtException(thread, e)
        }

        if (Build.VERSION.SDK_INT >= 30) recordLastExit(context)
    }

    /** The saved report, or null when the last run ended normally. */
    fun lastReport(): String? = reportFile.takeIf { it.isFile }?.readText()

    fun clear() {
        reportFile.delete()
    }

    private fun recordLastExit(context: Context) {
        try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val info = am.getHistoricalProcessExitReasons(context.packageName, 0, 1).firstOrNull() ?: return
            val seen = File(dir, "seen-exit")
            if (seen.isFile && seen.readText() == info.timestamp.toString()) return
            seen.writeText(info.timestamp.toString())

            val reason = when (info.reason) {
                ApplicationExitInfo.REASON_CRASH -> "crash (Kotlin/Java)"
                ApplicationExitInfo.REASON_CRASH_NATIVE -> "native crash"
                ApplicationExitInfo.REASON_LOW_MEMORY -> "killed by Android for low memory"
                ApplicationExitInfo.REASON_ANR -> "not responding (ANR)"
                ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "excessive resource usage"
                ApplicationExitInfo.REASON_SIGNALED -> "killed by signal ${info.status}"
                ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "initialization failure"
                else -> return // normal exits, user swipes, updates: nothing to report
            }
            val sb = StringBuilder()
            sb.append("Last run ended: $reason\n")
            sb.append("When: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(info.timestamp))}\n")
            sb.append("Process: ${info.processName}, importance ${info.importance}\n")
            sb.append("Memory at exit: PSS ${info.pss / 1024} MB, RSS ${info.rss / 1024} MB\n")
            sb.append("Description: ${info.description}\n")
            if (info.reason == ApplicationExitInfo.REASON_CRASH_NATIVE || info.reason == ApplicationExitInfo.REASON_ANR) {
                try {
                    info.traceInputStream?.use { stream ->
                        val bytes = stream.readBytes()
                        sb.append("\n--- trace (${bytes.size} bytes, first 64 KB) ---\n")
                        sb.append(String(bytes, 0, minOf(bytes.size, 65536), Charsets.ISO_8859_1))
                    }
                } catch (_: Throwable) {
                }
            }
            val existing = lastReport()
            // A Kotlin crash already wrote its stack trace; keep it below the summary.
            val text = if (existing != null && info.reason == ApplicationExitInfo.REASON_CRASH) "$sb\n$existing" else sb.toString()
            reportFile.writeText(text)
            Log.e(TAG, text.take(4000))
        } catch (e: Throwable) {
            Log.w(TAG, "could not read the exit record", e)
        }
    }

    private fun write(text: String) {
        reportFile.writeText(text)
        Log.e(TAG, text)
    }

    private fun now() = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
}
