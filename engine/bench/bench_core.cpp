#include "bench_core.hpp"

#include "dsp.hpp"
#include "model.hpp"

#if __has_include("sdprof.hpp")
#include "sdprof.hpp"
#endif
#if __has_include("gpu.hpp")
#include "gpu.hpp"
#define SDB_CLEAR_GPU() sdgpu::clear_weights()
#else
#define SDB_CLEAR_GPU()
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace sdbench
{
namespace
{
std::unique_ptr<demucscpp::demucs_model> g_model;

double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string json_str(const std::string &s)
{
    std::string o = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            o += '\\';
        if (c == '\n' || c == '\t')
        {
            o += ' ';
            continue;
        }
        o += c;
    }
    return o + "\"";
}

// Deterministic "song": kick, hats, bass line, a vibrato lead and a pad.
void synth(float *l, float *r, long n, int sr)
{
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    const double beat = 0.5;
    const double notes[4] = {41.2, 41.2, 55.0, 49.0};
    double bassPhase = 0.0, leadPhase = 0.0;
    for (long i = 0; i < n; ++i)
    {
        const double t = double(i) / sr;
        const double inBeat = std::fmod(t, beat);
        double kick = 0.0, hat = 0.0;
        if (inBeat < 0.15)
        {
            const double env = std::exp(-inBeat / 0.03);
            kick = std::sin(2 * M_PI * (50 + 80 * env) * inBeat) * env * 0.6;
        }
        const double hb = inBeat - beat / 2;
        if (hb >= 0 && hb < 0.05)
            hat = noise(rng) * std::exp(-hb / 0.01) * 0.15;
        bassPhase += 2 * M_PI * notes[int(t / (2 * beat)) % 4] / sr;
        const double bass = 0.3 * std::sin(bassPhase);
        leadPhase += 2 * M_PI * 220 * (1 + 0.01 * std::sin(2 * M_PI * 5 * t)) / sr;
        double lead = 0.0;
        for (int h = 1; h < 8; ++h)
            lead += std::sin(h * leadPhase) / h;
        lead *= 0.15 * (t > 2.0 ? 1.0 : 0.0);
        const double pad = 0.05 * (std::sin(2 * M_PI * 261.6 * t) +
                                   std::sin(2 * M_PI * 329.6 * t) +
                                   std::sin(2 * M_PI * 392.0 * t));
        const double m = kick + hat + bass + lead;
        l[i] = float(0.5 * (m + pad));
        r[i] = float(0.5 * (m + 0.5 * pad));
    }
}
} // namespace

std::string load_model(const std::string &path)
{
    auto m = std::make_unique<demucscpp::demucs_model>();
    if (!demucscpp::load_demucs_model(path, m.get()))
        return "could not load the model";
    if (m->is_4sources)
        return "not the 6-source model";
    SDB_CLEAR_GPU(); // weights of an earlier model may sit at the same address
    g_model = std::move(m);
    return "";
}

void free_model()
{
    SDB_CLEAR_GPU();
    g_model.reset();
}

std::string profile_segment(const std::string &outPath)
{
    if (!g_model)
        return "{\"error\": \"model not loaded\"}";
    const int sr = demucscpp::SUPPORTED_SAMPLE_RATE;
    const int seg = int(demucscpp::SEGMENT_LEN_SECS * sr);

    demucscpp::demucs_segment_buffers buffers(2, seg, 6);
    demucscpp::stft_buffers stft_buf(buffers.padded_segment_samples);

    std::vector<float> l(seg), r(seg);
    synth(l.data(), r.data(), seg, sr);
    // The engine normalizes by mean/std before the network; do the same.
    double sum = 0, sumsq = 0;
    for (int i = 0; i < seg; ++i)
    {
        const double m = 0.5 * (l[i] + r[i]);
        sum += m;
        sumsq += m * m;
    }
    const double mean = sum / seg;
    const double sd = std::sqrt((sumsq - seg * mean * mean) / (seg - 1));
    for (int i = 0; i < seg; ++i)
    {
        buffers.mix(0, i) = float((l[i] - mean) / sd);
        buffers.mix(1, i) = float((r[i] - mean) / sd);
    }

#ifdef SD_PROFILE
    sdprof::reset();
#endif
    std::vector<std::pair<std::string, double>> stages;
    double last = now_ms();
    const double t0 = last;
    demucscpp::ProgressCallback cb = [&](float, const std::string &msg) {
        const double t = now_ms();
        stages.emplace_back(msg, t - last);
        last = t;
    };
    demucscpp::model_inference(*g_model, buffers, stft_buf, cb, 0.0f, 1.0f);
    const double total = now_ms() - t0;

    if (!outPath.empty())
    {
        std::FILE *f = std::fopen(outPath.c_str(), "wb");
        if (f)
        {
            std::fwrite(buffers.targets_out.data(), sizeof(float),
                        buffers.targets_out.size(), f);
            std::fclose(f);
        }
    }

    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << "{\"ms\": " << total << ", \"segment_seconds\": "
      << demucscpp::SEGMENT_LEN_SECS << ", \"stages\": [";
    bool first = true;
    for (auto &s : stages)
    {
        // Skip the shape/debug messages that do no work of their own.
        if (s.second < 0.5)
            continue;
        o << (first ? "" : ", ") << "[" << json_str(s.first) << ", " << s.second
          << "]";
        first = false;
    }
    o << "], \"ops\": [";
#ifdef SD_PROFILE
    std::istringstream rep(sdprof::report());
    std::string line;
    first = true;
    while (std::getline(rep, line))
    {
        const auto a = line.find('\t'), b = line.rfind('\t');
        if (a == std::string::npos || b == a)
            continue;
        o << (first ? "" : ", ") << "[" << json_str(line.substr(0, a)) << ", "
          << line.substr(a + 1, b - a - 1) << ", " << line.substr(b + 1) << "]";
        first = false;
    }
#endif
    o << "]}";
    return o.str();
}

bool write_test_song(const std::string &path, double seconds)
{
    const int sr = demucscpp::SUPPORTED_SAMPLE_RATE;
    const long n = long(seconds * sr);
    std::vector<float> l(n), r(n), inter(2 * n);
    synth(l.data(), r.data(), n, sr);
    for (long i = 0; i < n; ++i)
    {
        inter[2 * i] = l[i];
        inter[2 * i + 1] = r[i];
    }
    std::FILE *f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const bool ok = std::fwrite(inter.data(), sizeof(float), inter.size(), f) == inter.size();
    return (std::fclose(f) == 0) && ok;
}

std::string compare_outputs(const std::string &a, const std::string &b)
{
    auto read = [](const std::string &p) {
        std::vector<float> v;
        std::FILE *f = std::fopen(p.c_str(), "rb");
        if (!f)
            return v;
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f) / long(sizeof(float));
        std::fseek(f, 0, SEEK_SET);
        v.resize(n);
        if (std::fread(v.data(), sizeof(float), n, f) != size_t(n))
            v.clear();
        std::fclose(f);
        return v;
    };
    const auto va = read(a), vb = read(b);
    if (va.empty() || va.size() != vb.size() || va.size() % 6 != 0)
        return "{\"error\": \"outputs missing or of different size\"}";
    const size_t per = va.size() / 6; // samples x channels per source
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(1);
    o << "{\"snr_db\": [";
    double maxd = 0;
    // targets_out is a column-major Tensor3dXf(6, 2, N): source is the
    // fastest-varying index.
    for (int s = 0; s < 6; ++s)
    {
        double sig = 0, err = 0;
        for (size_t k = 0; k < per; ++k)
        {
            const size_t i = size_t(s) + 6 * k;
            const double d = double(vb[i]) - va[i];
            sig += double(va[i]) * va[i];
            err += d * d;
            maxd = std::max(maxd, std::fabs(d));
        }
        const double snr = err > 0 ? 10 * std::log10(sig / err) : 200.0;
        o << (s ? ", " : "") << snr;
    }
    o.precision(6);
    o << "], \"max_abs_diff\": " << maxd << "}";
    return o.str();
}

} // namespace sdbench
