// JNI bridge for org.stemdeckport.android.engine.Mp3Codec.
#include "mp3_codec.hpp"

#include <jni.h>

#include <string>

namespace
{
std::string str(JNIEnv *env, jstring s)
{
    if (!s)
        return {};
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c)
        env->ReleaseStringUTFChars(s, c);
    return out;
}

sdmp3::Progress progress(JNIEnv *env, jobject listener)
{
    if (!listener)
        return nullptr;
    jclass cls = env->GetObjectClass(listener);
    jmethodID mid = env->GetMethodID(cls, "onProgress", "(F)Z");
    return [env, listener, mid](float f) {
        jboolean keep = env->CallBooleanMethod(listener, mid, f);
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
Java_org_stemdeckport_android_engine_Mp3Codec_nativeEncode(JNIEnv *env, jclass, jstring wav,
                                                           jstring mp3, jint kbps, jobject listener)
{
    return env->NewStringUTF(sdmp3::encode_wav(str(env, wav), str(env, mp3), kbps, progress(env, listener)).c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_android_engine_Mp3Codec_nativeDecode(JNIEnv *env, jclass, jstring mp3,
                                                           jstring wav, jobject listener)
{
    return env->NewStringUTF(sdmp3::decode_to_wav(str(env, mp3), str(env, wav), progress(env, listener)).c_str());
}
