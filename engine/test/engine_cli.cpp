// Desktop driver for the Android separation engine, used by CI to test the
// exact engine code (chunking, blending, resampling, loudness) against the
// real model before an APK is built.
//
//   sdengine_cli prepare  <raw.f32> <rate> <channels> <out.f32>
//   sdengine_cli analyze  <prepared.f32>
//   sdengine_cli separate <model.bin> <prepared.f32> <outdir> [threads] [chunkSeconds]
//     (env SD_NO_GPU=1 for CPU only, SD_THREADS=n for the maths threads)
#include "engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool printProgress(float f, const char *stage)
{
    static int lastPct = -1;
    int pct = static_cast<int>(f * 100.0f);
    if (pct / 10 != lastPct / 10)
    {
        std::fprintf(stderr, "  %3d%%  %s\n", pct, stage);
        lastPct = pct;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: see source\n");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "prepare" && argc == 6)
    {
        std::string err = sdengine::prepare_input(argv[2], std::atoi(argv[3]),
                                                  std::atoi(argv[4]), argv[5],
                                                  printProgress);
        if (!err.empty())
        {
            std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    if (cmd == "analyze" && argc == 3)
    {
        sdengine::InputStats st = sdengine::analyze_input(argv[2]);
        std::printf("{\"duration\": %.3f, \"peak_db\": %s, \"lufs\": %s}\n",
                    st.durationSec,
                    st.hasPeak ? std::to_string(st.peakDb).c_str() : "null",
                    st.hasLufs ? std::to_string(st.lufs).c_str() : "null");
        return 0;
    }
    if (cmd == "separate" && argc >= 5)
    {
        std::string err = sdengine::load_model(argv[2]);
        if (!err.empty())
        {
            std::fprintf(stderr, "load failed: %s\n", err.c_str());
            return 1;
        }
        sdengine::SeparateOptions opts;
        if (argc > 5)
            opts.threads = std::atoi(argv[5]);
        if (argc > 6)
            opts.chunkSeconds = std::atoi(argv[6]);
#ifdef SDENGINE_HAS_COMPUTE
        // SD_NO_GPU=1: CPU only. SD_THREADS=n: threads doing maths.
        opts.gpu = std::getenv("SD_NO_GPU") == nullptr;
        if (const char *t = std::getenv("SD_THREADS"))
            opts.computeThreads = std::atoi(t);
        std::fprintf(stderr, "%s\n", sdengine::compute_info(opts).c_str());
#endif
        auto t0 = std::chrono::steady_clock::now();
        sdengine::SeparateResult r =
            sdengine::separate(argv[3], argv[4], opts, printProgress);
        double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
        if (!r.ok)
        {
            std::fprintf(stderr, "separate failed: %s\n", r.error.c_str());
            return 1;
        }
        std::printf("{\"seconds\": %.1f, \"rms\": {", secs);
        for (int s = 0; s < sdengine::kNumSources; ++s)
            std::printf("%s\"%s\": %.6f", s ? ", " : "", sdengine::kSourceNames[s],
                        r.rms[s]);
        std::printf("}}\n");
        return 0;
    }
    std::fprintf(stderr, "bad arguments\n");
    return 2;
}
