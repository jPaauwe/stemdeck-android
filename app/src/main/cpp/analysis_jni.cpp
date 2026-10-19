// JNI bridge for org.stemdeckport.android.engine.NativeAnalysis: the
// analysis stages in /analysis (beat grid, key, click render).
#include <android/log.h>
#include <jni.h>

#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "click.hpp"
#include "key.hpp"
#include "ort.hpp"
#include "pipeline.hpp"
#include "wav.hpp"

#define LOG_TAG "StemDeckAnalysis"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

std::string str(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

void throwNew(JNIEnv* env, const char* cls, const std::string& msg) {
    jclass c = env->FindClass(cls);
    if (c) env->ThrowNew(c, msg.c_str());
}

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if ((unsigned char)c < 0x20) continue;
        o += c;
    }
    return o;
}

// Kotlin's ProgressListener (fraction, stage) -> keep going; ~10 calls a second.
sda::Progress progress(JNIEnv* env, jobject listener, const char* stage) {
    if (!listener) return nullptr;
    jclass cls = env->GetObjectClass(listener);
    jmethodID mid = env->GetMethodID(cls, "onProgress", "(FLjava/lang/String;)Z");
    auto last = std::make_shared<long long>(0);
    std::string st = stage;
    return [env, listener, mid, last, st](double f) -> bool {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        long long now = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
        if (now - *last < 100 && f < 1.0) return true;
        *last = now;
        jstring js = env->NewStringUTF(st.c_str());
        jboolean keep = env->CallBooleanMethod(listener, mid, (jfloat)f, js);
        env->DeleteLocalRef(js);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            return false;
        }
        return keep == JNI_TRUE;
    };
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeAnalysis_nativeBeatGrid(JNIEnv* env, jclass, jstring source, jstring label,
                                                                    jstring model, jint threads, jobject listener) {
    try {
        sda::BeatGrid g = sda::run_beat_grid(str(env, source), str(env, label), str(env, model), threads,
                                             progress(env, listener, "Beat grid"));
        std::string out = g.ok ? "{\"ok\": true, \"tempo_stability\": " + std::to_string(g.tempo_stability) +
                                     ", \"grid\": " + g.to_json() + "}"
                               : "{\"ok\": false, \"why\": \"" + esc(g.why) + "\"}";
        return env->NewStringUTF(out.c_str());
    } catch (const sda::Cancelled&) {
        throwNew(env, "java/util/concurrent/CancellationException", "cancelled");
    } catch (const std::exception& e) {
        throwNew(env, "java/lang/IllegalStateException", e.what());
    }
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeAnalysis_nativeKey(JNIEnv* env, jclass, jobjectArray harmony,
                                                               jdoubleArray weights, jstring bass) {
    try {
        std::vector<std::pair<std::string, double>> h;
        jsize n = env->GetArrayLength(harmony);
        std::vector<double> w(n);
        env->GetDoubleArrayRegion(weights, 0, n, w.data());
        for (jsize i = 0; i < n; i++) {
            auto js = (jstring)env->GetObjectArrayElement(harmony, i);
            h.push_back({str(env, js), w[i]});
            env->DeleteLocalRef(js);
        }
        sda::KeyResult k = sda::key_from_stems(h, str(env, bass));
        if (!k.ok) return env->NewStringUTF("");
        std::string out = "{\"key\": \"" + k.key + "\", \"scale\": \"" + k.scale + "\", \"confidence\": " +
                          std::to_string(k.confidence) + "}";
        return env->NewStringUTF(out.c_str());
    } catch (const std::exception& e) {
        throwNew(env, "java/lang/IllegalStateException", e.what());
    }
    return nullptr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_stemdeckport_android_engine_NativeAnalysis_nativeRenderClick(JNIEnv* env, jclass, jdoubleArray beats,
                                                                       jintArray barBeats, jintArray barLengths,
                                                                       jdouble duration, jint sampleRate,
                                                                       jdouble multiplier, jint accent, jstring out) {
    try {
        std::vector<double> b(env->GetArrayLength(beats));
        env->GetDoubleArrayRegion(beats, 0, (jsize)b.size(), b.data());
        jsize nb = env->GetArrayLength(barBeats);
        std::vector<jint> at(nb), len(nb);
        env->GetIntArrayRegion(barBeats, 0, nb, at.data());
        env->GetIntArrayRegion(barLengths, 0, nb, len.data());
        std::vector<sda::BeatBar> bars;
        for (jsize i = 0; i < nb; i++) bars.push_back({at[i], len[i]});
        auto pcm = sda::render_click(b, bars, duration, sampleRate, multiplier, accent, {});
        if (pcm.empty()) return JNI_FALSE;
        // Stereo, both channels the same, like the stems the mix is made of.
        std::vector<float> st(pcm.size() * 2);
        for (size_t i = 0; i < pcm.size(); i++) st[2 * i] = st[2 * i + 1] = pcm[i] / 32767.0f;
        sda::write_wav16(str(env, out), st.data(), pcm.size(), 2, sampleRate);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        throwNew(env, "java/lang/IllegalStateException", e.what());
    }
    return JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeAnalysis_nativeOrtVersion(JNIEnv* env, jclass) {
    try {
        return env->NewStringUTF(sda::OrtModel::version().c_str());
    } catch (const std::exception& e) {
        return env->NewStringUTF((std::string("unavailable: ") + e.what()).c_str());
    }
}
