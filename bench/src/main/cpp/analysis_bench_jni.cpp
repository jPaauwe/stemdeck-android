// Bench bindings for the analysis stages (/analysis), timed.
#include <jni.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "key.hpp"
#include "ort.hpp"
#include "wav.hpp"
#include "pipeline.hpp"
#include "sections.hpp"
#include "vocal_split.hpp"

namespace {
std::string str(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if ((unsigned char)c >= 0x20) o += c; }
    return o;
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_bench_AnalysisNative_nBeatGrid(JNIEnv* env, jclass, jstring wav, jstring model, jint threads,
                                                     jstring provider) {
    auto t0 = std::chrono::steady_clock::now();
    try {
        sda::BeatGrid g = sda::run_beat_grid(str(env, wav), "drums", str(env, model), threads, nullptr, str(env, provider));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::string beats;
        for (size_t i = 0; i < g.beats.size(); i++) { char b[32]; std::snprintf(b, sizeof b, "%s%.6f", i ? "," : "", g.beats[i]); beats += b; }
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "{\"seconds\": %.3f, \"load\": %.3f, \"mel\": %.3f, \"model\": %.3f, \"grid\": %.3f, \"ok\": %s, "
                      "\"bpm\": %.3f, \"why\": \"%s\", \"beats\": [",
                      s, g.t_load, g.t_mel, g.t_model, g.t_grid, g.ok ? "true" : "false", g.bpm, esc(g.why).c_str());
        return env->NewStringUTF((std::string(buf) + beats + "]}").c_str());
    } catch (const std::exception& e) {
        return env->NewStringUTF(("{\"error\": \"" + esc(e.what()) + "\"}").c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_bench_AnalysisNative_nKey(JNIEnv* env, jclass, jstring dir) {
    std::string d = str(env, dir);
    auto t0 = std::chrono::steady_clock::now();
    try {
        auto p = [&](const char* n) { return d + "/" + n + ".wav"; };
        sda::KeyResult k = sda::key_from_stems({{p("guitar"), 1.0}, {p("piano"), 1.0}, {p("other"), 1.0}, {p("vocals"), 0.5}}, p("bass"));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[256];
        std::snprintf(buf, sizeof buf, "{\"seconds\": %.3f, \"key\": \"%s %s\"}", s, k.key.c_str(), k.scale.c_str());
        return env->NewStringUTF(buf);
    } catch (const std::exception& e) {
        return env->NewStringUTF(("{\"error\": \"" + esc(e.what()) + "\"}").c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_bench_AnalysisNative_nOrtVersion(JNIEnv* env, jclass) {
    try {
        return env->NewStringUTF(sda::OrtModel::version().c_str());
    } catch (const std::exception& e) {
        return env->NewStringUTF((std::string("unavailable: ") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_bench_AnalysisNative_nSections(JNIEnv* env, jclass, jstring dir, jobjectArray folds, jint threads) {
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::vector<std::string> f;
        for (jsize i = 0; i < env->GetArrayLength(folds); i++) {
            auto js = (jstring)env->GetObjectArrayElement(folds, i);
            f.push_back(str(env, js));
            env->DeleteLocalRef(js);
        }
        std::string d = str(env, dir);
        double dur = 0;
        { sda::WavReader r(d + "/bass.wav"); dur = r.frames() / (double)r.sample_rate(); }
        sda::SectionsResult r = sda::run_sections(d, f, {}, -1, dur, threads, nullptr);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[256];
        std::snprintf(buf, sizeof buf, "{\"seconds\": %.3f, \"spec\": %.3f, \"model\": %.3f, \"post\": %.3f, \"count\": %d}",
                      s, r.t_spec, r.t_model, r.t_post, r.count);
        return env->NewStringUTF(buf);
    } catch (const std::exception& e) {
        return env->NewStringUTF(("{\"error\": \"" + esc(e.what()) + "\"}").c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_stemdeckport_bench_AnalysisNative_nVocalSplit(JNIEnv* env, jclass, jstring vocals, jstring outDir, jstring model, jint threads) {
    auto t0 = std::chrono::steady_clock::now();
    try {
        std::string o = str(env, outDir);
        sda::VocalSplitResult r = sda::split_vocals(str(env, vocals), o + "/lead_vocals.wav", o + "/backing_vocals.wav",
                                                    str(env, model), threads, nullptr);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char buf[256];
        std::snprintf(buf, sizeof buf, "{\"seconds\": %.3f, \"model\": %.3f, \"ok\": %s, \"why\": \"%s\"}", s, r.t_model,
                      r.ok ? "true" : "false", esc(r.why).c_str());
        return env->NewStringUTF(buf);
    } catch (const std::exception& e) {
        return env->NewStringUTF(("{\"error\": \"" + esc(e.what()) + "\"}").c_str());
    }
}
