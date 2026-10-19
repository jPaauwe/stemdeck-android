#include "beat_this.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "fft.hpp"

namespace sda {
namespace {
// torch.linspace(start, end, steps) in float32: each half counted from its
// own end, with a fused multiply-add (one rounding), as torch's kernel does.
std::vector<float> torch_linspace(float start, float end, int steps) {
    std::vector<float> out(steps);
    if (steps == 1) { out[0] = start; return out; }
    const float step = (end - start) / (float)(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; i++)
        out[i] = i < halfway ? std::fma(step, (float)i, start) : std::fma(-step, (float)(steps - i - 1), end);
    return out;
}
}  // namespace

// The 130 mel band edges in Hz exactly as torchaudio 2.6 computes them
// (torch's vectorised float32 exp differs from libm's in the last bit in a
// few places, so they are tabulated rather than recomputed).
static const float kMelEdgesHz[130] = {
    0x1.dffffe0000000p+4f, 0x1.bc5a080000000p+5f, 0x1.445a0a0000000p+6f, 0x1.aa87100000000p+6f, 0x1.085a0a0000000p+7f, 0x1.3b708e0000000p+7f,
    0x1.6e87100000000p+7f, 0x1.a19d900000000p+7f, 0x1.d4b4140000000p+7f, 0x1.03e54c0000000p+8f, 0x1.1d708c0000000p+8f, 0x1.36fbce0000000p+8f,
    0x1.5087100000000p+8f, 0x1.6a12500000000p+8f, 0x1.839d920000000p+8f, 0x1.9d28d40000000p+8f, 0x1.b6b4140000000p+8f, 0x1.d03f560000000p+8f,
    0x1.e9ca980000000p+8f, 0x1.01aaec0000000p+9f, 0x1.0e708c0000000p+9f, 0x1.1b362e0000000p+9f, 0x1.27fbce0000000p+9f, 0x1.34c1700000000p+9f,
    0x1.41870e0000000p+9f, 0x1.4e4cb00000000p+9f, 0x1.5b12500000000p+9f, 0x1.67d7f20000000p+9f, 0x1.749d920000000p+9f, 0x1.8163340000000p+9f,
    0x1.8e28d20000000p+9f, 0x1.9aee740000000p+9f, 0x1.a7b4140000000p+9f, 0x1.b479b60000000p+9f, 0x1.c13f560000000p+9f, 0x1.ce04f80000000p+9f,
    0x1.daca980000000p+9f, 0x1.e790360000000p+9f, 0x1.f458900000000p+9f, 0x1.00d9d00000000p+10f, 0x1.07b4f80000000p+10f, 0x1.0ebefc0000000p+10f,
    0x1.15f91a0000000p+10f, 0x1.1d649a0000000p+10f, 0x1.2502ce0000000p+10f, 0x1.2cd5100000000p+10f, 0x1.34dcc40000000p+10f, 0x1.3d1b580000000p+10f,
    0x1.4592420000000p+10f, 0x1.4e43040000000p+10f, 0x1.572f2a0000000p+10f, 0x1.6058460000000p+10f, 0x1.69bffc0000000p+10f, 0x1.7367f20000000p+10f,
    0x1.7d51ea0000000p+10f, 0x1.877fa00000000p+10f, 0x1.91f2e40000000p+10f, 0x1.9cad900000000p+10f, 0x1.a7b18e0000000p+10f, 0x1.b300d40000000p+10f,
    0x1.be9d5e0000000p+10f, 0x1.ca89420000000p+10f, 0x1.d6c6a00000000p+10f, 0x1.e3579e0000000p+10f, 0x1.f03e7e0000000p+10f, 0x1.fd7d860000000p+10f,
    0x1.058b8a0000000p+11f, 0x1.0c86c60000000p+11f, 0x1.13b1b80000000p+11f, 0x1.1b0da60000000p+11f, 0x1.229bdc0000000p+11f, 0x1.2a5db40000000p+11f,
    0x1.32548c0000000p+11f, 0x1.3a81d40000000p+11f, 0x1.42e6fa0000000p+11f, 0x1.4b857e0000000p+11f, 0x1.545eea0000000p+11f, 0x1.5d74cc0000000p+11f,
    0x1.66c8c80000000p+11f, 0x1.705c7e0000000p+11f, 0x1.7a31a40000000p+11f, 0x1.8449fc0000000p+11f, 0x1.8ea7540000000p+11f, 0x1.994b7a0000000p+11f,
    0x1.a438580000000p+11f, 0x1.af6fe40000000p+11f, 0x1.baf4100000000p+11f, 0x1.c6c6f00000000p+11f, 0x1.d2ea9e0000000p+11f, 0x1.df613c0000000p+11f,
    0x1.ec2d060000000p+11f, 0x1.f950360000000p+11f, 0x1.03669e0000000p+12f, 0x1.0a53340000000p+12f, 0x1.116f1c0000000p+12f, 0x1.18bb960000000p+12f,
    0x1.2039f40000000p+12f, 0x1.27eb820000000p+12f, 0x1.2fd1a60000000p+12f, 0x1.37edc40000000p+12f, 0x1.40414c0000000p+12f, 0x1.48cdb80000000p+12f,
    0x1.5194920000000p+12f, 0x1.5a97620000000p+12f, 0x1.63d7c80000000p+12f, 0x1.6d57680000000p+12f, 0x1.7717ec0000000p+12f, 0x1.811b140000000p+12f,
    0x1.8b62a80000000p+12f, 0x1.95f07c0000000p+12f, 0x1.a0c66e0000000p+12f, 0x1.abe66a0000000p+12f, 0x1.b7526e0000000p+12f, 0x1.c30c7e0000000p+12f,
    0x1.cf16ae0000000p+12f, 0x1.db73240000000p+12f, 0x1.e824120000000p+12f, 0x1.f52bb20000000p+12f, 0x1.0146360000000p+13f, 0x1.0824440000000p+13f,
    0x1.0f31400000000p+13f, 0x1.166e680000000p+13f, 0x1.1ddd0a0000000p+13f, 0x1.257e740000000p+13f, 0x1.2d54080000000p+13f, 0x1.355f1e0000000p+13f,
    0x1.3da12c0000000p+13f, 0x1.461ba80000000p+13f, 0x1.4ed0140000000p+13f, 0x1.57bffc0000000p+13f
};

std::vector<float> beat_this_mel_fb() {
    const int n_freqs = 513, n_mels = 128;
    auto all_freqs = torch_linspace(0.0f, (float)(kBeatSr / 2), n_freqs);
    const float* f_pts = kMelEdgesHz;
    std::vector<float> fb((size_t)n_freqs * n_mels);
    for (int k = 0; k < n_freqs; k++)
        for (int m = 0; m < n_mels; m++) {
            float down = (-1.0f * (f_pts[m] - all_freqs[k])) / (f_pts[m + 1] - f_pts[m]);
            float up = (f_pts[m + 2] - all_freqs[k]) / (f_pts[m + 2] - f_pts[m + 1]);
            fb[(size_t)k * n_mels + m] = std::max(0.0f, std::min(down, up));
        }
    return fb;
}

Spec beat_this_logmel(const std::vector<float>& y) {
    const int n_fft = 1024, hop = 441, pad = n_fft / 2, n_mels = 128, nb = n_fft / 2 + 1;
    const long len = (long)y.size();
    Spec out;
    if (len <= pad) return out;  // reflect padding needs more samples than the pad
    out.frames = (size_t)(1 + len / hop);
    out.bins = n_mels;
    out.v.assign(out.frames * n_mels, 0.0f);
    static const std::vector<float> fb = beat_this_mel_fb();
    std::vector<int> first(n_mels, nb), last(n_mels, -1);
    for (int k = 0; k < nb; k++)
        for (int m = 0; m < n_mels; m++)
            if (fb[(size_t)k * n_mels + m] != 0.0f) { first[m] = std::min(first[m], k); last[m] = std::max(last[m], k); }
    std::vector<double> win(n_fft);
    for (int k = 0; k < n_fft; k++) win[k] = (double)(float)(0.5 - 0.5 * std::cos(2.0 * M_PI * k / n_fft));
    auto at = [&](long j) -> float {  // reflect, without repeating the edge
        if (j < 0) j = -j;
        if (j >= len) j = 2 * (len - 1) - j;
        return y[(size_t)j];
    };
    std::vector<double> frame(n_fft);
    std::vector<std::complex<double>> spec;
    std::vector<float> mag(nb);
    const double norm = 1.0 / std::sqrt((double)n_fft);
    for (size_t t = 0; t < out.frames; t++) {
        long start = (long)t * hop - pad;
        for (int i = 0; i < n_fft; i++) frame[i] = win[i] * (double)at(start + i);
        RealFFT::forward(frame, spec);
        for (int k = 0; k < nb; k++) mag[k] = (float)(std::abs(spec[k]) * norm);
        float* row = out.row(t);
        for (int m = 0; m < n_mels; m++) {
            double acc = 0;
            for (int k = first[m]; k <= last[m]; k++) acc += (double)fb[(size_t)k * n_mels + m] * mag[k];
            float v = (float)acc;
            row[m] = std::log1p(1000.0f * v);
        }
    }
    return out;
}

std::vector<long> beat_this_chunk_starts(long frames) {
    const long chunk = kBeatChunk, border = kBeatBorder, step = chunk - 2 * border;
    std::vector<long> starts;
    for (long s = -border; s < frames - border; s += step) starts.push_back(s);
    if (starts.empty()) starts.push_back(-border);
    if (frames > step) starts.back() = frames - (chunk - border);
    return starts;
}

bool beat_this_logits(const Spec& mel, const BeatModelFn& model, std::vector<float>& beat, std::vector<float>& downbeat,
                      const std::function<bool(double)>& progress) {
    const long T = (long)mel.frames, chunk = kBeatChunk, border = kBeatBorder;
    beat.assign(T, -1000.0f);
    downbeat.assign(T, -1000.0f);
    auto starts = beat_this_chunk_starts(T);
    std::vector<float> in, b, d;
    // keep_first: later chunks are written first, so earlier ones overwrite them.
    for (size_t c = 0; c < starts.size(); c++) {
        long s = starts[starts.size() - 1 - c];
        long lo = std::max(s, 0L), hi = std::min(s + chunk, T);
        long left = std::max(0L, -s), right = std::max(0L, std::min(border, s + chunk - T));
        long n = left + (hi - lo) + right;
        in.assign((size_t)n * 128, 0.0f);
        std::copy(mel.v.begin() + lo * 128, mel.v.begin() + hi * 128, in.begin() + left * 128);
        b.assign(n, 0.0f);
        d.assign(n, 0.0f);
        if (!model(in.data(), n, b.data(), d.data())) return false;
        // pred[border:-border] -> piece[s + border : s + chunk - border] (clamped)
        for (long i = border; i < n - border; i++) {
            long at = s + i;
            if (at < s + border || at >= s + chunk - border || at < 0 || at >= T) continue;
            beat[at] = b[i];
            downbeat[at] = d[i];
        }
        if (progress && !progress((double)(c + 1) / starts.size())) return false;
    }
    return true;
}

namespace {
std::vector<double> peaks_minimal(const std::vector<float>& x) {
    // logit == max_pool1d(logit, 7, 1, 3) and logit > 0
    const long T = (long)x.size();
    std::vector<long> frames;
    for (long t = 0; t < T; t++) {
        float m = -INFINITY;
        for (long k = std::max(0L, t - 3); k <= std::min(T - 1, t + 3); k++) m = std::max(m, x[k]);
        float v = x[t] != m ? -1000.0f : x[t];
        if (v > 0) frames.push_back(t);
    }
    // deduplicate_peaks(width=1): runs of adjacent frames become their running mean.
    std::vector<double> out;
    if (frames.empty()) return out;
    double p = (double)frames[0];
    long c = 1;
    // the comparison uses the running mean p, as the Python does
    for (size_t i = 1; i < frames.size(); i++) {
        long p2 = frames[i];
        if ((double)p2 - p <= 1) {
            c += 1;
            p += ((double)p2 - p) / c;
        } else {
            out.push_back(p);
            p = (double)p2;
            c = 1;
        }
    }
    out.push_back(p);
    return out;
}
}  // namespace

void beat_this_minimal(const std::vector<float>& beat, const std::vector<float>& downbeat,
                       std::vector<double>& beats, std::vector<double>& downbeats) {
    auto bf = peaks_minimal(beat);
    auto df = peaks_minimal(downbeat);
    beats.resize(bf.size());
    for (size_t i = 0; i < bf.size(); i++) beats[i] = bf[i] / kBeatFps;
    std::vector<double> db(df.size());
    for (size_t i = 0; i < df.size(); i++) db[i] = df[i] / kBeatFps;
    if (!beats.empty()) {
        for (auto& d : db) {
            size_t best = 0;
            double bd = std::fabs(beats[0] - d);
            for (size_t j = 1; j < beats.size(); j++) {
                double e = std::fabs(beats[j] - d);
                if (e < bd) { bd = e; best = j; }
            }
            d = beats[best];
        }
    }
    std::sort(db.begin(), db.end());
    db.erase(std::unique(db.begin(), db.end()), db.end());
    downbeats = db;
}

}  // namespace sda
