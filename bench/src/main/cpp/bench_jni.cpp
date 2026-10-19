// JNI for the bench app. Compiled twice: into libsdbench_opt.so (the engine
// being optimized, with per-operation profiling) and libsdbench_base.so (the
// frozen baseline). Each registers its natives on its own Kotlin object
// (SDB_CLASS), and every other symbol is hidden, so the two engines never
// see each other's code.
#include "bench_core.hpp"
#include "engine.hpp"

#include <jni.h>

#include <chrono>
#include <cstdio>
#include <string>

namespace
{
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

jstring nLoad(JNIEnv *env, jclass, jstring path)
{
    return env->NewStringUTF(sdbench::load_model(str(env, path)).c_str());
}

void nFree(JNIEnv *, jclass) { sdbench::free_model(); }

jstring nProfile(JNIEnv *env, jclass, jstring out)
{
    return env->NewStringUTF(sdbench::profile_segment(str(env, out)).c_str());
}

jboolean nWriteSong(JNIEnv *env, jclass, jstring path, jdouble seconds)
{
    return sdbench::write_test_song(str(env, path), seconds) ? JNI_TRUE : JNI_FALSE;
}

jstring nCompare(JNIEnv *env, jclass, jstring a, jstring b)
{
    return env->NewStringUTF(sdbench::compare_outputs(str(env, a), str(env, b)).c_str());
}

// Full separation with the engine's own chunked, multi-threaded path, as
// the app runs it.
jstring nSeparate(JNIEnv *env, jclass, jstring model, jstring song, jstring outDir,
                  jint threads, jint chunkSeconds)
{
    std::string err = sdengine::load_model(str(env, model));
    if (!err.empty())
        return env->NewStringUTF(("{\"error\": \"" + err + "\"}").c_str());
    sdengine::SeparateOptions o;
    o.threads = threads;
    o.chunkSeconds = chunkSeconds;
    const auto t0 = std::chrono::steady_clock::now();
    auto r = sdengine::separate(str(env, song), str(env, outDir), o,
                                [](float, const char *) { return true; });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char buf[256];
    if (!r.ok)
        std::snprintf(buf, sizeof(buf), "{\"error\": \"%s\"}", r.error.c_str());
    else
        std::snprintf(buf, sizeof(buf), "{\"seconds\": %.2f}", s);
    return env->NewStringUTF(buf);
}

const JNINativeMethod kMethods[] = {
    {"nLoad", "(Ljava/lang/String;)Ljava/lang/String;", (void *)nLoad},
    {"nFree", "()V", (void *)nFree},
    {"nProfile", "(Ljava/lang/String;)Ljava/lang/String;", (void *)nProfile},
    {"nWriteSong", "(Ljava/lang/String;D)Z", (void *)nWriteSong},
    {"nCompare", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;", (void *)nCompare},
    {"nSeparate",
     "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;II)Ljava/lang/String;",
     (void *)nSeparate},
};
} // namespace

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *)
{
    JNIEnv *env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK)
        return JNI_ERR;
    jclass cls = env->FindClass(SDB_CLASS);
    if (!cls)
        return JNI_ERR;
    if (env->RegisterNatives(cls, kMethods, sizeof(kMethods) / sizeof(kMethods[0])) != 0)
        return JNI_ERR;
    return JNI_VERSION_1_6;
}
