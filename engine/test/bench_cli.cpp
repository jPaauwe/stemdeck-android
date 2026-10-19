// Desktop driver for the benchmark core (same code as the bench app).
//
//   sdbench_cli segment <model.bin> [out.f32]   one 7.8 s segment, profiled
//   sdbench_cli song <seconds> <out.f32>        write the test song
//   sdbench_cli compare <a.f32> <b.f32>         SNR of b against a
//   sdbench_cli separate <model.bin> <song.f32> <outdir> <threads> <chunk>
#include "bench_core.hpp"
#include "engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
        return 2;
    const std::string cmd = argv[1];
    if (cmd == "segment" && argc >= 3)
    {
        std::string err = sdbench::load_model(argv[2]);
        if (!err.empty())
        {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        std::printf("%s\n", sdbench::profile_segment(argc > 3 ? argv[3] : "").c_str());
        return 0;
    }
    if (cmd == "song" && argc == 4)
        return sdbench::write_test_song(argv[3], std::atof(argv[2])) ? 0 : 1;
    if (cmd == "compare" && argc == 4)
    {
        std::printf("%s\n", sdbench::compare_outputs(argv[2], argv[3]).c_str());
        return 0;
    }
    if (cmd == "separate" && argc == 7)
    {
        std::string err = sdengine::load_model(argv[2]);
        if (!err.empty())
        {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        sdengine::SeparateOptions o;
        o.threads = std::atoi(argv[5]);
        o.chunkSeconds = std::atoi(argv[6]);
        auto t0 = std::chrono::steady_clock::now();
        auto r = sdengine::separate(argv[3], argv[4], o, [](float, const char *) { return true; });
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!r.ok)
        {
            std::fprintf(stderr, "%s\n", r.error.c_str());
            return 1;
        }
        std::printf("{\"seconds\": %.1f}\n", s);
        return 0;
    }
    return 2;
}
