// JNI bridge for org.stemdeckport.android.engine.NativeEngine.
#include "engine.hpp"

#include <android/log.h>
#include <jni.h>

#include <ctime>
#include <limits>
#include <memory>
#include <string>

#define LOG_TAG "StemDeckEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace
{

std::string toString(JNIEnv *env, jstring s)
{
    if (!s)
        return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c)
        env->ReleaseStringUTFChars(s, c);
    return out;
}

void throwNew(JNIEnv *env, const char *cls, const std::string &msg)
{
    jclass c = env->FindClass(cls);
    if (c)
        env->ThrowNew(c, msg.c_str());
}

// Wraps the Kotlin ProgressListener. Calls are throttled to ~10 per second so
// the UI thread is not flooded.
sdengine::ProgressFn makeProgress(JNIEnv *env, jobject listener)
{
    if (!listener)
        return [](float, const char *) { return true; };
    jclass cls = env->GetObjectClass(listener);
    jmethodID mid = env->GetMethodID(cls, "onProgress", "(FLjava/lang/String;)Z");
    auto last = std::make_shared<long long>(0);
    return [env, listener, mid, last](float frac, const char *stage) -> bool {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        const long long now = ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
        if (now - *last < 100 && frac < 1.0f)
            return true;
        *last = now;
        jstring jstage = env->NewStringUTF(stage);
        jboolean keep = env->CallBooleanMethod(listener, mid, frac, jstage);
        env->DeleteLocalRef(jstage);
        if (env->ExceptionCheck())
        {
            env->ExceptionClear();
            return false;
        }
        return keep == JNI_TRUE;
    };
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeLoadModel(JNIEnv *env, jclass,
                                                                  jstring path)
{
    std::string err = sdengine::load_model(toString(env, path));
    if (err.empty())
        LOGI("model loaded");
    return env->NewStringUTF(err.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativePrepareInput(
    JNIEnv *env, jclass, jstring raw, jint rate, jint channels, jstring out,
    jobject listener)
{
    std::string err = sdengine::prepare_input(toString(env, raw), rate, channels,
                                              toString(env, out),
                                              makeProgress(env, listener));
    if (err == "cancelled")
    {
        throwNew(env, "java/util/concurrent/CancellationException", "cancelled");
        return nullptr;
    }
    return env->NewStringUTF(err.c_str());
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeAnalyze(JNIEnv *env, jclass,
                                                                jstring path)
{
    sdengine::InputStats st = sdengine::analyze_input(toString(env, path));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    jdouble vals[3] = {st.durationSec, st.hasPeak ? st.peakDb : nan,
                       st.hasLufs ? st.lufs : nan};
    jdoubleArray arr = env->NewDoubleArray(3);
    env->SetDoubleArrayRegion(arr, 0, 3, vals);
    return arr;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeSeparate(
    JNIEnv *env, jclass, jstring prepared, jstring outDir, jbooleanArray write,
    jint threads, jint chunkSeconds, jint computeThreads, jboolean gpu, jobject listener)
{
    sdengine::SeparateOptions opts;
    opts.threads = threads;
    opts.chunkSeconds = chunkSeconds;
    opts.computeThreads = computeThreads;
    opts.gpu = gpu == JNI_TRUE;
    if (write && env->GetArrayLength(write) == sdengine::kNumSources)
    {
        jboolean w[sdengine::kNumSources];
        env->GetBooleanArrayRegion(write, 0, sdengine::kNumSources, w);
        for (int i = 0; i < sdengine::kNumSources; ++i)
            opts.write[i] = w[i] == JNI_TRUE;
    }
    sdengine::SeparateResult r = sdengine::separate(
        toString(env, prepared), toString(env, outDir), opts,
        makeProgress(env, listener));
    if (r.cancelled)
    {
        throwNew(env, "java/util/concurrent/CancellationException", "cancelled");
        return nullptr;
    }
    if (!r.ok)
    {
        throwNew(env, "java/lang/RuntimeException",
                 r.error.empty() ? "Separation failed." : r.error);
        return nullptr;
    }
    jdoubleArray arr = env->NewDoubleArray(sdengine::kNumSources);
    env->SetDoubleArrayRegion(arr, 0, sdengine::kNumSources, r.rms.data());
    return arr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeComputeInfo(JNIEnv *env, jclass,
                                                                    jint computeThreads,
                                                                    jboolean gpu)
{
    sdengine::SeparateOptions opts;
    opts.computeThreads = computeThreads;
    opts.gpu = gpu == JNI_TRUE;
    return env->NewStringUTF(sdengine::compute_info(opts).c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeCancel(JNIEnv *, jclass)
{
    sdengine::request_cancel();
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_NativeEngine_nativeSourceName(JNIEnv *env, jclass,
                                                                   jint index)
{
    if (index < 0 || index >= sdengine::kNumSources)
        return nullptr;
    return env->NewStringUTF(sdengine::kSourceNames[index]);
}
