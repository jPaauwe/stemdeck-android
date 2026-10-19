package org.stemdeckport.android

import com.sun.net.httpserver.HttpServer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.stemdeckport.android.engine.ModelSpec
import org.stemdeckport.android.engine.ModelStore
import java.io.IOException
import java.net.InetSocketAddress
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

class ModelStoreTest {
    private val data = Random(3).nextBytes(3_000_000)
    private val sha = MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }
    private val ranged = AtomicInteger(0)

    private fun serve(): HttpServer = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
        createContext("/model.bin") { ex ->
            val range = ex.requestHeaders.getFirst("Range")
            val from = range?.removePrefix("bytes=")?.substringBefore('-')?.toInt() ?: 0
            if (range != null) {
                ranged.incrementAndGet()
                ex.responseHeaders.add("Content-Range", "bytes $from-${data.size - 1}/${data.size}")
                ex.sendResponseHeaders(206, (data.size - from).toLong())
            } else {
                ex.sendResponseHeaders(200, data.size.toLong())
            }
            ex.responseBody.use { it.write(data, from, data.size - from) }
        }
        start()
    }

    @Test
    fun an_interrupted_download_resumes_and_is_checked() {
        val server = serve()
        val dir = kotlin.io.path.createTempDirectory("sdtest").toFile()
        val spec = ModelSpec("model.bin", "http://127.0.0.1:${server.address.port}/model.bin", sha, data.size.toLong(), "the test model")
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
        server.stop(0)
        dir.deleteRecursively()
    }

    @Test
    fun a_wrong_checksum_is_refused_and_not_kept() {
        val server = serve()
        val dir = kotlin.io.path.createTempDirectory("sdtest").toFile()
        val spec = ModelSpec("model.bin", "http://127.0.0.1:${server.address.port}/model.bin", "0".repeat(64), data.size.toLong(), "the test model")
        val store = ModelStore(dir)
        try {
            store.ensure(spec) { true }
            fail("expected a checksum failure")
        } catch (e: IOException) {
            assertTrue(e.message!!.contains("checksum"))
        }
        assertTrue(!store.isReady(spec))
        assertTrue(dir.listFiles()!!.isEmpty())
        server.stop(0)
        dir.deleteRecursively()
    }
}
