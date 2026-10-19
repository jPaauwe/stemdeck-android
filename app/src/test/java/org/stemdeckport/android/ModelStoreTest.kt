package org.stemdeckport.android

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.stemdeckport.android.engine.ModelSpec
import org.stemdeckport.android.engine.ModelStore
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

class ModelStoreTest {
    private val data = Random(3).nextBytes(3_000_000)
    private val sha = MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }
    private val ranged = AtomicInteger(0)

    /** A one-file HTTP server that honours "Range: bytes=N-" (the test classpath is android.jar, so no com.sun.net.httpserver). */
    private class Server(private val data: ByteArray, private val ranged: AtomicInteger) {
        val socket = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))
        val port get() = socket.localPort
        private val thread = Thread {
            while (!socket.isClosed) {
                val c = try { socket.accept() } catch (e: IOException) { break }
                Thread {
                    c.use { conn ->
                        val reader = conn.getInputStream().bufferedReader(Charsets.ISO_8859_1)
                        var range: String? = null
                        while (true) {
                            val line = reader.readLine() ?: break
                            if (line.isEmpty()) break
                            if (line.startsWith("Range:", ignoreCase = true)) range = line.substringAfter(':').trim()
                        }
                        val from = range?.removePrefix("bytes=")?.substringBefore('-')?.toInt() ?: 0
                        val head = if (range != null) {
                            ranged.incrementAndGet()
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes $from-${data.size - 1}/${data.size}\r\n"
                        } else "HTTP/1.1 200 OK\r\n"
                        val out = conn.getOutputStream()
                        try {
                            out.write((head + "Content-Length: ${data.size - from}\r\nConnection: close\r\n\r\n").toByteArray(Charsets.ISO_8859_1))
                            out.write(data, from, data.size - from)
                            out.flush()
                        } catch (e: IOException) {
                            // the client hung up (the cancelled download)
                        }
                    }
                }.apply { isDaemon = true }.start()
            }
        }.apply { isDaemon = true; start() }

        fun stop() = socket.close()
    }

    private fun serve() = Server(data, ranged)

    @Test
    fun an_interrupted_download_resumes_and_is_checked() {
        val server = serve()
        val dir = kotlin.io.path.createTempDirectory("sdtest").toFile()
        val spec = ModelSpec("model.bin", "http://127.0.0.1:${server.port}/model.bin", sha, data.size.toLong(), "the test model")
        val store = ModelStore(dir)
        try {
            store.ensure(spec) { it < 0.4f }
            fail("expected cancellation")
        } catch (e: CancellationException) {
        }
        assertTrue(!store.isReady(spec))
        val f = store.ensure(spec) { true }
        assertEquals(1, ranged.get())
        assertArrayEquals(data, f.readBytes())
        assertTrue(store.isReady(spec))
        server.stop()
        dir.deleteRecursively()
    }

    @Test
    fun a_wrong_checksum_is_refused_and_not_kept() {
        val server = serve()
        val dir = kotlin.io.path.createTempDirectory("sdtest").toFile()
        val spec = ModelSpec("model.bin", "http://127.0.0.1:${server.port}/model.bin", "0".repeat(64), data.size.toLong(), "the test model")
        val store = ModelStore(dir)
        try {
            store.ensure(spec) { true }
            fail("expected a checksum failure")
        } catch (e: IOException) {
            assertTrue(e.message!!.contains("checksum"))
        }
        assertTrue(!store.isReady(spec))
        assertTrue(dir.listFiles()!!.isEmpty())
        server.stop()
        dir.deleteRecursively()
    }
}
