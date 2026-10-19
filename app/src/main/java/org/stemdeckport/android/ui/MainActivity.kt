package org.stemdeckport.android.ui

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.view.ViewGroup
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.stemdeckport.android.App
import org.stemdeckport.android.BuildConfig
import org.stemdeckport.android.Diagnostics
import org.stemdeckport.android.data.MixExport
import org.stemdeckport.android.engine.NativeEngine
import org.stemdeckport.android.server.LocalServer
import org.stemdeckport.android.work.SeparationService
import java.io.File
import kotlin.concurrent.thread

/**
 * Hosts StemDeck's own mobile UI (bundled in assets/web) in a WebView, talking
 * to the on-device server in [LocalServer].
 */
class MainActivity : ComponentActivity() {
    companion object {
        const val EXTRA_TAB = "tab"
    }

    private val app get() = application as App
    private lateinit var web: WebView
    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private var pageLoaded = false

    private val pickFile = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
        val uris = WebChromeClient.FileChooserParams.parseResult(res.resultCode, res.data)
        fileCallback?.onReceiveValue(uris)
        fileCallback = null
    }

    private val askNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)

        val root = FrameLayout(this).apply { setBackgroundColor(Color.parseColor("#0b0b0d")) }
        web = WebView(this).apply {
            setBackgroundColor(Color.parseColor("#0b0b0d"))
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.mediaPlaybackRequiresUserGesture = false
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                    val url = request.url
                    if (url.host == "127.0.0.1") return false
                    // Anything else (links in lyrics, credits…) opens in the browser.
                    try { startActivity(Intent(Intent.ACTION_VIEW, url)) } catch (_: Exception) {}
                    return true
                }

                override fun onPageFinished(view: WebView, url: String) {
                    pageLoaded = true
                }
            }
            webChromeClient = object : WebChromeClient() {
                override fun onShowFileChooser(
                    view: WebView, callback: ValueCallback<Array<Uri>>, params: FileChooserParams,
                ): Boolean {
                    fileCallback?.onReceiveValue(null)
                    fileCallback = callback
                    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "*/*"
                        putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/mp4", "application/ogg"))
                    }
                    return try {
                        pickFile.launch(intent); true
                    } catch (e: Exception) {
                        fileCallback = null; false
                    }
                }
            }
            setDownloadListener { url, _, _, _, _ -> onDownload(Uri.parse(url)) }
        }
        root.addView(web, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        setContentView(root)

        val tab = intent.getStringExtra(EXTRA_TAB)
        val shared = sharedUri(intent)
        if (savedInstanceState != null) {
            web.restoreState(savedInstanceState)
        }
        web.loadUrl(app.server.entryUrl(if (shared != null) "library" else tab))
        shared?.let { importShared(it) }

        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        // Anything queued before the app was last closed, unless the last run
        // crashed: then wait for the next import, so a crash in the service
        // itself cannot keep the app from opening.
        if (app.store.hasQueued() && Diagnostics.lastReport() == null) SeparationService.start(this)

        if (savedInstanceState == null) showLastExitReport()
    }

    /** If the app stopped unexpectedly last time, say why and offer the details. */
    private fun showLastExitReport() {
        val report = Diagnostics.lastReport() ?: return
        val summary = report.lineSequence().take(4).joinToString("\n")
        android.app.AlertDialog.Builder(this)
            .setTitle("StemDeck stopped unexpectedly last time")
            .setMessage("$summary\n\nCopy the details and send them to whoever is fixing this.")
            .setPositiveButton("Copy details") { _, _ ->
                val cm = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
                cm.setPrimaryClip(android.content.ClipData.newPlainText("StemDeck crash report", report))
                Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show()
                Diagnostics.clear()
            }
            .setNegativeButton("Dismiss") { _, _ -> Diagnostics.clear() }
            .show()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        sharedUri(intent)?.let { importShared(it) } ?: intent.getStringExtra(EXTRA_TAB)?.let { showTab(it) }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        web.saveState(outState)
    }

    override fun onDestroy() {
        web.destroy()
        super.onDestroy()
    }

    private fun showTab(tab: String) {
        if (!pageLoaded) return
        web.evaluateJavascript(
            "location.hash = ${org.json.JSONObject.quote(tab)}; window.dispatchEvent(new HashChangeEvent('hashchange'));",
            null,
        )
    }

    // ─── Share → StemDeck / Open with ───
    private fun sharedUri(intent: Intent?): Uri? = when (intent?.action) {
        Intent.ACTION_SEND -> if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            @Suppress("DEPRECATION") intent.getParcelableExtra(Intent.EXTRA_STREAM)
        }
        Intent.ACTION_VIEW -> intent.data
        else -> null
    }

    private fun importShared(uri: Uri) {
        // Consume the intent so a rotation does not import it twice.
        intent = Intent(this, MainActivity::class.java)
        Toast.makeText(this, "Importing…", Toast.LENGTH_SHORT).show()
        thread(name = "import") {
            try {
                val name = displayName(uri) ?: "Shared audio"
                val ext = name.substringAfterLast('.', "bin").take(8)
                val staged = File(cacheDir, "share-${System.nanoTime()}.$ext")
                contentResolver.openInputStream(uri)?.use { input ->
                    staged.outputStream().use { input.copyTo(it, 1 shl 16) }
                } ?: throw IllegalStateException("The shared file could not be read.")
                val title = LocalServer.titleFrom(name)
                app.store.create(title, "local:${name.substringBeforeLast('.')}", NativeEngine.SOURCES, staged, app.settings.stemFormat)
                SeparationService.start(this)
                runOnUiThread {
                    Toast.makeText(this, "Separating “$title”", Toast.LENGTH_SHORT).show()
                    showTab("library")
                }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, e.message ?: "Import failed", Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun displayName(uri: Uri): String? = try {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: uri.lastPathSegment
    } catch (_: Exception) {
        uri.lastPathSegment
    }

    // ─── "Export mix" → Downloads/StemDeck ───
    private fun onDownload(url: Uri) {
        val seg = url.pathSegments
        // /api/jobs/<id>/mixdown.wav
        if (seg.size != 4 || seg[0] != "api" || seg[1] != "jobs" || seg[3] != "mixdown.wav") return
        val job = app.store.get(seg[2]) ?: return
        val names = url.getQueryParameter("stems")?.split(',')?.filter { it in job.stems } ?: job.stems
        val gains = url.getQueryParameter("gains")?.split(',')?.map { it.toFloatOrNull() ?: 1f }
        if (names.isEmpty()) return
        val g = names.indices.map { gains?.getOrNull(it)?.coerceIn(0f, 2f) ?: 1f }
        val format = app.settings.stemFormat
        val fileName = "${LocalServer.safeFileName(job.title)} (mix).$format"
        Toast.makeText(this, "Exporting the mix…", Toast.LENGTH_SHORT).show()
        thread(name = "export") {
            try {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, MixExport.mimeType(format))
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/StemDeck")
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                val target = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("Could not create the file in Downloads.")
                try {
                    contentResolver.openOutputStream(target)!!.buffered(1 shl 16).use {
                        MixExport.write(app.store, job, names, g, format, cacheDir, it)
                    }
                    contentResolver.update(target, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
                } catch (e: Exception) {
                    contentResolver.delete(target, null, null)
                    throw e
                }
                runOnUiThread { Toast.makeText(this, "Saved to Downloads/StemDeck/$fileName", Toast.LENGTH_LONG).show() }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "Export failed: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }
}
