// Host tool for scripts/analysis_parity.py: runs each analysis stage, or a
// piece of one, on files, so its output can be compared with desktop
// StemDeck's Python.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "beat_this.hpp"
#include "beatgrid.hpp"
#include "click.hpp"
#include "cqt.hpp"
#include "key.hpp"
#include "librosa.hpp"
#include "pipeline.hpp"
#include "wav.hpp"

using namespace sda;

static void write_f32(const std::string& path, const float* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(p, 4, n, f);
    fclose(f);
}
static std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    size_t n = (size_t)in.tellg() / 4;
    in.seekg(0);
    std::vector<float> v(n);
    in.read((char*)v.data(), n * 4);
    return v;
}
static std::vector<double> read_list(const std::string& path) {
    std::ifstream in(path);
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    return v;
}
static void write_text(const std::string& path, const std::string& s) {
    std::ofstream(path) << s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sdanalysis_cli <beatgrid|mono22k|onset|logmel|melfb|grid> ...\n");
        return 2;
    }
    std::string cmd = argv[1];
    try {
        if (cmd == "beatgrid" && argc >= 5) {
            int threads = argc > 5 ? std::atoi(argv[5]) : 0;
            auto t0 = std::chrono::steady_clock::now();
            BeatGrid g = run_beat_grid(argv[2], "drums", argv[3], threads, nullptr);
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!g.ok) { std::fprintf(stderr, "no grid: %s\n", g.why.c_str()); write_text(argv[4], "null"); return 0; }
            write_text(argv[4], g.to_json());
            std::fprintf(stderr, "%zu beats, %.3f BPM, stability %ld, %.2f s\n", g.beats.size(), g.bpm, g.tempo_stability, secs);
        } else if (cmd == "mono22k" && argc >= 4) {
            auto y = load_mono_22k(argv[2]);
            write_f32(argv[3], y.data(), y.size());
        } else if (cmd == "onset" && argc >= 7) {
            auto y = read_floats(argv[2]);
            auto e = onset_strength_median(y, 22050.0, std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]));
            write_f32(argv[6], e.data(), e.size());
        } else if (cmd == "logmel" && argc >= 4) {
            auto y = read_floats(argv[2]);
            Spec s = beat_this_logmel(y);
            write_f32(argv[3], s.v.data(), s.v.size());
        } else if (cmd == "melfb" && argc >= 3) {
            auto fb = beat_this_mel_fb();
            write_f32(argv[2], fb.data(), fb.size());
        } else if (cmd == "postproc" && argc >= 6) {
            // beat logits, downbeat logits (f32) -> beats.txt, downbeats.txt
            auto b = read_floats(argv[2]), d = read_floats(argv[3]);
            std::vector<double> beats, downbeats;
            beat_this_minimal(b, d, beats, downbeats);
            std::string s1, s2;
            char buf[64];
            for (double t : beats) { snprintf(buf, sizeof buf, "%.17g\n", t); s1 += buf; }
            for (double t : downbeats) { snprintf(buf, sizeof buf, "%.17g\n", t); s2 += buf; }
            write_text(argv[4], s1);
            write_text(argv[5], s2);
        } else if (cmd == "grid" && argc >= 6) {
            auto y = read_floats(argv[2]);
            BeatGrid g = compute_beat_grid(y, 22050.0, read_list(argv[3]), read_list(argv[4]), "beat_this", "drums");
            write_text(argv[5], g.ok ? g.to_json() : "null");
        } else if (cmd == "tuning" && argc >= 4) {
            auto y = read_floats(argv[2]);
            std::printf("%.17g\n", estimate_tuning(y, 22050.0, std::atoi(argv[3])));
        } else if (cmd == "chroma" && argc >= 5) {
            auto y = read_floats(argv[2]);
            Spec c = chroma_cqt(y, 22050.0, 2048, kC1, std::atoi(argv[3]));
            write_f32(argv[4], c.v.data(), c.v.size());
        } else if (cmd == "halve" && argc >= 4) {
            auto y = read_floats(argv[2]);
            auto h = halve_rate(y);
            write_f32(argv[3], h.data(), h.size());
        } else if (cmd == "key" && argc >= 5) {
            // harmony.f32 bass.f32|- whole(0/1), mono 22050 Hz
            auto h = read_floats(argv[2]);
            std::vector<float> b;
            if (std::string(argv[3]) != "-") b = read_floats(argv[3]);
            KeyResult k = detect_key_from_audio(h, b, std::atoi(argv[4]) != 0);
            std::printf("{\"key\": \"%s\", \"scale\": \"%s\", \"confidence\": %ld}\n", k.key.c_str(), k.scale.c_str(), k.confidence);
        } else if (cmd == "keystems" && argc >= 3) {
            // stems dir with guitar/piano/other/vocals/bass .wav
            std::string d = argv[2];
            auto ex = [&](const std::string& n) { std::string p = d + "/" + n + ".wav"; FILE* f = fopen(p.c_str(), "rb"); if (!f) return std::string(); fclose(f); return p; };
            KeyResult k = key_from_stems({{ex("guitar"), 1.0}, {ex("piano"), 1.0}, {ex("other"), 1.0}, {ex("vocals"), 0.5}}, ex("bass"));
            std::printf("{\"key\": \"%s\", \"scale\": \"%s\", \"confidence\": %ld}\n", k.key.c_str(), k.scale.c_str(), k.confidence);
        } else if (cmd == "levels" && argc >= 5) {
            // accent groups(3+2+2|-) count [bars as beat:per,beat:per]
            int accent = std::atoi(argv[2]);
            std::vector<int> groups;
            if (std::string(argv[3]) != "-") for (char* t = std::strtok(argv[3], "+"); t; t = std::strtok(nullptr, "+")) groups.push_back(std::atoi(t));
            std::vector<BeatBar> bars;
            if (argc > 5) for (char* t = std::strtok(argv[5], ","); t; t = std::strtok(nullptr, ",")) {
                long b = 0, n = 0; std::sscanf(t, "%ld:%ld", &b, &n); bars.push_back({b, n});
            }
            int count = std::atoi(argv[4]);
            for (int i = 0; i < count; i++) std::printf("%d%s", beat_level(i, bars, accent, groups), i + 1 < count ? " " : "\n");
        } else if (cmd == "defgroup" && argc >= 3) {
            auto g = default_grouping(std::atoi(argv[2]));
            for (size_t i = 0; i < g.size(); i++) std::printf("%d%s", g[i], i + 1 < g.size() ? " " : "");
            std::printf("\n");
        } else if (cmd == "clickwav" && argc >= 9) {
            // beats.txt bars(beat:per,..|-) duration mult accent out.s16 sample_rate
            auto beats = read_list(argv[2]);
            std::vector<BeatBar> bars;
            if (std::string(argv[3]) != "-") for (char* t = std::strtok(argv[3], ","); t; t = std::strtok(nullptr, ",")) {
                long b = 0, n = 0; std::sscanf(t, "%ld:%ld", &b, &n); bars.push_back({b, n});
            }
            auto pcm = render_click(beats, bars, std::atof(argv[4]), std::atoi(argv[8]), std::atof(argv[5]), std::atoi(argv[6]), {});
            FILE* f = fopen(argv[7], "wb");
            fwrite(pcm.data(), 2, pcm.size(), f);
            fclose(f);
        } else {
            std::fprintf(stderr, "bad arguments\n");
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
