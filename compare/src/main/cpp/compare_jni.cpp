// Host library for StemDeck Compare: loads engine plugins one at a time,
// runs them, and diffs their outputs.
#include <dlfcn.h>
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engines_list.h" // generated: const char *kEnginesList

namespace
{
typedef int (*progress_fn)(void *ctx, float f);
struct Plugin
{
    void *handle = nullptr;
    int (*load_model)(const char *, char *, int) = nullptr;
    void (*unload_model)() = nullptr;
    int (*prepare)(const char *, int, int, const char *, progress_fn, void *, char *, int) = nullptr;
    int (*separate)(const char *, const char *, int, int, progress_fn, void *, double *, char *, int) = nullptr;
    void (*cancel)() = nullptr;
};

Plugin g_plugin;

std::string str(JNIEnv *env, jstring s)
{
    if (!s)
        return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string o(c ? c : "");
    if (c)
        env->ReleaseStringUTFChars(s, c);
    return o;
}

std::string esc(const std::string &s)
{
    std::string o;
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            o += '\\';
        o += (c == '\n') ? ' ' : c;
    }
    return o;
}

struct JniProgress
{
    JNIEnv *env;
    jobject listener;
    jmethodID mid;
};

int progress_tramp(void *ctx, float f)
{
    auto *p = static_cast<JniProgress *>(ctx);
    if (!p || !p->listener)
        return 1;
    jboolean keep = p->env->CallBooleanMethod(p->listener, p->mid, f);
    if (p->env->ExceptionCheck())
    {
        p->env->ExceptionClear();
        return 0;
    }
    return keep == JNI_TRUE ? 1 : 0;
}

JniProgress make_progress(JNIEnv *env, jobject listener)
{
    JniProgress p{env, listener, nullptr};
    if (listener)
        p.mid = env->GetMethodID(env->GetObjectClass(listener), "onProgress", "(F)Z");
    return p;
}

void close_plugin()
{
    if (g_plugin.handle)
    {
        if (g_plugin.unload_model)
            g_plugin.unload_model();
        dlclose(g_plugin.handle);
    }
    g_plugin = Plugin{};
}

// ─── WAV comparison ───
struct Wav
{
    std::FILE *f = nullptr;
    long frames = 0;
    bool open(const std::string &path)
    {
        f = std::fopen(path.c_str(), "rb");
        if (!f)
            return false;
        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 44, SEEK_SET); // the engine writes a plain 44-byte header
        frames = (bytes - 44) / 4;
        return frames > 0;
    }
    size_t read(std::vector<int16_t> &buf, size_t frames_)
    {
        buf.resize(frames_ * 2);
        return std::fread(buf.data(), 4, frames_, f);
    }
    ~Wav()
    {
        if (f)
            std::fclose(f);
    }
};

const char *kStems[6] = {"drums", "bass", "other", "vocals", "guitar", "piano"};

double db(double num, double den)
{
    if (den <= 0)
        return 200.0;
    if (num <= 0)
        return -200.0;
    return 10.0 * std::log10(num / den);
}
} // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nEngines(JNIEnv *env, jclass)
{
    return env->NewStringUTF(kEnginesList);
}

// Loads libsdengine_<id>.so (unloading any previous one). Returns "" or an error.
JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nOpen(JNIEnv *env, jclass, jstring id)
{
    close_plugin();
    const std::string lib = "libsdengine_" + str(env, id) + ".so";
    void *h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h)
        return env->NewStringUTF((std::string("dlopen failed: ") + dlerror()).c_str());
    Plugin p;
    p.handle = h;
    p.load_model = reinterpret_cast<decltype(p.load_model)>(dlsym(h, "sdp_load_model"));
    p.unload_model = reinterpret_cast<decltype(p.unload_model)>(dlsym(h, "sdp_unload_model"));
    p.prepare = reinterpret_cast<decltype(p.prepare)>(dlsym(h, "sdp_prepare"));
    p.separate = reinterpret_cast<decltype(p.separate)>(dlsym(h, "sdp_separate"));
    p.cancel = reinterpret_cast<decltype(p.cancel)>(dlsym(h, "sdp_cancel"));
    if (!p.load_model || !p.unload_model || !p.prepare || !p.separate || !p.cancel)
    {
        dlclose(h);
        return env->NewStringUTF("the engine library is missing functions");
    }
    g_plugin = p;
    return env->NewStringUTF("");
}

JNIEXPORT void JNICALL Java_org_stemdeckport_compare_Host_nClose(JNIEnv *, jclass)
{
    close_plugin();
}

JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nLoadModel(JNIEnv *env, jclass, jstring path)
{
    if (!g_plugin.handle)
        return env->NewStringUTF("no engine open");
    char err[512] = {0};
    g_plugin.load_model(str(env, path).c_str(), err, sizeof(err));
    return env->NewStringUTF(err);
}

JNIEXPORT jint JNICALL
Java_org_stemdeckport_compare_Host_nPrepare(JNIEnv *env, jclass, jstring raw, jint rate, jint ch,
                                           jstring out, jobject listener)
{
    if (!g_plugin.handle)
        return 0;
    JniProgress p = make_progress(env, listener);
    char err[512] = {0};
    return g_plugin.prepare(str(env, raw).c_str(), rate, ch, str(env, out).c_str(),
                            progress_tramp, &p, err, sizeof(err));
}

// Returns JSON {"ok":1} / {"error":...} / {"cancelled":1}.
JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nSeparate(JNIEnv *env, jclass, jstring prepared, jstring outDir,
                                            jint threads, jint chunk, jobject listener)
{
    if (!g_plugin.handle)
        return env->NewStringUTF("{\"error\": \"no engine open\"}");
    JniProgress p = make_progress(env, listener);
    char err[512] = {0};
    double rms[6] = {0};
    const int r = g_plugin.separate(str(env, prepared).c_str(), str(env, outDir).c_str(), threads,
                                    chunk, progress_tramp, &p, rms, err, sizeof(err));
    if (r < 0)
        return env->NewStringUTF("{\"cancelled\": 1}");
    if (r == 0)
        return env->NewStringUTF(("{\"error\": \"" + esc(err) + "\"}").c_str());
    return env->NewStringUTF("{\"ok\": 1}");
}

JNIEXPORT void JNICALL Java_org_stemdeckport_compare_Host_nCancel(JNIEnv *, jclass)
{
    if (g_plugin.cancel)
        g_plugin.cancel();
}

// Compares the six stems in `dir` against the same stems in `refDir`.
// JSON: {"stems":[{"name", "snr_db", "max_abs", "ref_rms_db", "rms_db",
//        "worst_db", "worst_at"}...], "all_snr_db", "frames"}
JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nDiff(JNIEnv *env, jclass, jstring refDir, jstring dir)
{
    const std::string a = str(env, refDir), b = str(env, dir);
    std::string out = "{\"stems\": [";
    double allSig = 0, allErr = 0;
    long framesOut = 0;
    char buf[512];
    for (int s = 0; s < 6; ++s)
    {
        Wav wa, wb;
        if (!wa.open(a + "/" + kStems[s] + ".wav") || !wb.open(b + "/" + kStems[s] + ".wav"))
            continue;
        const long n = std::min(wa.frames, wb.frames);
        framesOut = std::max(framesOut, n);
        double sig = 0, err = 0, sb = 0, maxAbs = 0;
        // per-second windows: find where the two differ most (relative to
        // the reference), ignoring near-silent seconds
        const long win = 44100;
        double wSig = 0, wErr = 0;
        double worst = 1e9;
        long worstAt = -1, inWin = 0, winIdx = 0;
        std::vector<int16_t> ba, bb;
        const size_t block = 1 << 16;
        long done = 0;
        while (done < n)
        {
            const size_t want = static_cast<size_t>(std::min<long>(block, n - done));
            const size_t ga = wa.read(ba, want), gb = wb.read(bb, want);
            const size_t got = std::min(ga, gb);
            if (got == 0)
                break;
            for (size_t i = 0; i < got * 2; ++i)
            {
                const double x = ba[i] / 32767.0, y = bb[i] / 32767.0, d = y - x;
                sig += x * x;
                sb += y * y;
                err += d * d;
                wSig += x * x;
                wErr += d * d;
                maxAbs = std::max(maxAbs, std::fabs(d));
                if ((i & 1) == 1 && ++inWin == win)
                {
                    if (wSig / (2.0 * win) > 1e-6) // above -60 dBFS
                    {
                        const double w = db(wSig, wErr);
                        if (w < worst)
                        {
                            worst = w;
                            worstAt = winIdx;
                        }
                    }
                    wSig = wErr = 0;
                    inWin = 0;
                    ++winIdx;
                }
            }
            done += static_cast<long>(got);
        }
        allSig += sig;
        allErr += err;
        const double cnt = 2.0 * std::max<long>(1, n);
        std::snprintf(buf, sizeof(buf),
                      "%s{\"name\": \"%s\", \"snr_db\": %.2f, \"max_abs\": %.6f, \"ref_rms_db\": %.2f, "
                      "\"rms_db\": %.2f, \"worst_db\": %.2f, \"worst_at\": %ld, \"frames_ref\": %ld, \"frames\": %ld}",
                      out.back() == '[' ? "" : ", ", kStems[s], db(sig, err), maxAbs,
                      db(sig / cnt, 1.0), db(sb / cnt, 1.0), worstAt < 0 ? 200.0 : worst, worstAt,
                      wa.frames, wb.frames);
        out += buf;
    }
    std::snprintf(buf, sizeof(buf), "], \"all_snr_db\": %.2f, \"frames\": %ld}", db(allSig, allErr), framesOut);
    out += buf;
    return env->NewStringUTF(out.c_str());
}

// How well the six stems in `dir` add back up to the input (prepared
// stereo float32). JSON {"snr_db": x}.
JNIEXPORT jstring JNICALL
Java_org_stemdeckport_compare_Host_nReconstruction(JNIEnv *env, jclass, jstring dir, jstring prepared)
{
    const std::string d = str(env, dir);
    std::FILE *mf = std::fopen(str(env, prepared).c_str(), "rb");
    if (!mf)
        return env->NewStringUTF("{\"error\": \"input missing\"}");
    Wav w[6];
    long n = -1;
    for (int s = 0; s < 6; ++s)
    {
        if (!w[s].open(d + "/" + kStems[s] + ".wav"))
        {
            std::fclose(mf);
            return env->NewStringUTF("{\"error\": \"stems missing\"}");
        }
        n = n < 0 ? w[s].frames : std::min(n, w[s].frames);
    }
    const size_t block = 1 << 15;
    std::vector<float> mix(block * 2);
    std::vector<int16_t> st[6];
    double sig = 0, err = 0;
    long done = 0;
    while (done < n)
    {
        const size_t want = static_cast<size_t>(std::min<long>(block, n - done));
        const size_t gm = std::fread(mix.data(), 8, want, mf);
        size_t got = gm;
        for (int s = 0; s < 6; ++s)
            got = std::min(got, w[s].read(st[s], want));
        if (got == 0)
            break;
        for (size_t i = 0; i < got * 2; ++i)
        {
            double sum = 0;
            for (int s = 0; s < 6; ++s)
                sum += st[s][i] / 32767.0;
            const double m = mix[i], e = sum - m;
            sig += m * m;
            err += e * e;
        }
        done += static_cast<long>(got);
    }
    std::fclose(mf);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "{\"snr_db\": %.2f}", db(sig, err));
    return env->NewStringUTF(buf);
}

} // extern "C"
