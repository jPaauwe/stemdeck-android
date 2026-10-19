#include "vocal_split.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "fft.hpp"
#include "ort.hpp"
#include "pipeline.hpp"
#include "wav.hpp"

namespace sda {
namespace {
constexpr int kNfft = 5120, kHop = 1024, kDimF = 2048, kDimT = 256;
constexpr int kBins = kNfft / 2 + 1;                  // 2561
constexpr int kTrim = kNfft / 2;                      // 2560
constexpr long kChunk = (long)kHop * (kDimT - 1);     // 261120
constexpr long kGen = kChunk - 2 * kTrim;             // 256000
constexpr double kOverlap = 0.25;
constexpr float kCompensate = 1.065f;  // a Python float meeting float32 arrays: float32 arithmetic
constexpr float kPeak = 0.9f;          // normalization_threshold

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// write_audio_pydub: normalize, then (x * 32767).astype(int16) (truncation).
// Returns false for a near-silent stem, which pydub's path does not write.
bool write_stem(const std::string& path, std::vector<float>& left, std::vector<float>& right, double* rms) {
    // normalize() sees both channels together (one peak for the stem)
    float maxv = 0;
    for (float v : left) maxv = std::max(maxv, std::fabs(v));
    for (float v : right) maxv = std::max(maxv, std::fabs(v));
    if (maxv > kPeak) {
        const float s = kPeak / maxv;
        for (auto& v : left) v = v * s;
        for (auto& v : right) v = v * s;
        maxv = maxv * s;
    }
    if (maxv < 1e-6f) return false;
    const size_t n = left.size();
    std::vector<int16_t> pcm(2 * n);
    double ss = 0;
    for (size_t i = 0; i < n; i++) {
        pcm[2 * i] = (int16_t)(int)(left[i] * 32767.0f);
        pcm[2 * i + 1] = (int16_t)(int)(right[i] * 32767.0f);
        ss += (double)pcm[2 * i] * pcm[2 * i] + (double)pcm[2 * i + 1] * pcm[2 * i + 1];
    }
    *rms = n ? std::sqrt(ss / (2.0 * n)) / 32768.0 : 0.0;
    write_wav16_pcm(path, pcm.data(), n, 2, 44100);
    return true;
}

// torch.stft(center=True, reflect) of one channel of a chunk, first dim_f bins,
// as the model's [re; im] planes [dim_f][dim_t].
void stft_chunk(const float* x, const std::vector<double>& win, float* re, float* im) {
    std::vector<double> frame(kNfft);
    std::vector<std::complex<double>> spec;
    for (int t = 0; t < kDimT; t++) {
        const long start = (long)t * kHop - kNfft / 2;
        for (int i = 0; i < kNfft; i++) {
            long j = start + i;
            if (j < 0) j = -j;                                    // reflect
            else if (j >= kChunk) j = 2 * (kChunk - 1) - j;
            frame[i] = (double)x[j] * win[i];
        }
        RealFFT::forward(frame, spec);
        for (int f = 0; f < kDimF; f++) {
            re[(size_t)f * kDimT + t] = (float)spec[f].real();
            im[(size_t)f * kDimT + t] = (float)spec[f].imag();
        }
    }
}

// torch.istft(center=True) of [re; im] planes with bins >= dim_f zero: kChunk samples.
void istft_chunk(const float* re, const float* im, const std::vector<double>& win, float* out) {
    const long total = (long)kHop * (kDimT - 1) + kNfft;
    std::vector<double> ola(total, 0.0), env(total, 0.0);
    std::vector<std::complex<double>> half(kBins);
    std::vector<double> y;
    for (int t = 0; t < kDimT; t++) {
        for (int f = 0; f < kBins; f++)
            half[f] = f < kDimF ? std::complex<double>(re[(size_t)f * kDimT + t], im[(size_t)f * kDimT + t]) : 0.0;
        RealFFT::inverse_real(half, y, kNfft);
        const long at = (long)t * kHop;
        for (int i = 0; i < kNfft; i++) {
            ola[at + i] += (double)(float)y[i] * win[i];
            env[at + i] += win[i] * win[i];
        }
    }
    for (long i = 0; i < kChunk; i++) {
        double e = env[kTrim + i];
        out[i] = (float)(e > 1e-11 ? ola[kTrim + i] / e : 0.0);
    }
}
}  // namespace

VocalSplitResult split_vocals(const std::string& vocals_wav, const std::string& lead_wav, const std::string& backing_wav,
                              const std::string& model_path, int threads, const std::function<bool(double)>& progress,
                              const std::string& provider) {
    VocalSplitResult res;
    // prepare_mix: librosa.load(mono=False, sr=44100) -> (2, N) float32; mono is doubled.
    std::vector<float> mix[2];
    {
        WavReader r(vocals_wav);
        if (r.sample_rate() != 44100) throw std::runtime_error("the vocals stem must be 44.1 kHz");
        const int c = r.channels();
        mix[0].reserve(r.frames());
        mix[1].reserve(r.frames());
        std::vector<float> buf;
        while (size_t n = r.read(buf, 1 << 16))
            for (size_t i = 0; i < n; i++) {
                mix[0].push_back(buf[i * c]);
                mix[1].push_back(buf[i * c + (c > 1 ? 1 : 0)]);
            }
    }
    const long N = (long)mix[0].size();
    res.seconds = N / 44100.0;
    float peak = 0;
    for (int c = 0; c < 2; c++)
        for (float v : mix[c]) peak = std::max(peak, std::fabs(v));
    {
        double ss = 0;
        for (int c = 0; c < 2; c++)
            for (float v : mix[c]) ss += (double)v * v;
        res.vocals_rms = N ? std::sqrt(ss / (2.0 * N)) : 0.0;
    }
    if (N == 0 || peak == 0) {
        res.why = "the vocals stem is silent";
        return res;
    }
    // normalize(mix) in place, one peak across both channels
    if (peak > kPeak) {
        const float s = kPeak / peak;
        for (int c = 0; c < 2; c++)
            for (auto& v : mix[c]) v = v * s;
    }

    // demix(): chunks of kChunk every `step`, Hann-weighted overlap-add.
    const long pad = kGen + kTrim - (N % kGen);
    const long M = kTrim + N + pad;
    const long step = (long)((1 - kOverlap) * kChunk);
    std::vector<double> swin(kNfft);  // torch.hann_window(5120, periodic=True), float32
    for (int i = 0; i < kNfft; i++) swin[i] = (double)(float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / kNfft));
    auto mixture = [&](int c, long x) -> float {
        long j = x - kTrim;
        return (j >= 0 && j < N) ? mix[c][j] : 0.0f;
    };
    std::vector<float> primary[2] = {std::vector<float>(N), std::vector<float>(N)};
    // Rolling accumulators for positions [base, base + kChunk).
    std::vector<float> acc[2] = {std::vector<float>(kChunk, 0.0f), std::vector<float>(kChunk, 0.0f)};
    std::vector<float> div(kChunk, 0.0f);
    long base = 0;
    auto emit_until = [&](long upto) {  // positions [base, upto) are final
        for (long x = base; x < upto; x++) {
            long k = x - base, j = x - kTrim;
            if (j >= 0 && j < N)
                for (int c = 0; c < 2; c++) primary[c][j] = (acc[c][k] / div[k]) * peak;
        }
        long shift = upto - base;
        for (int c = 0; c < 2; c++) {
            std::move(acc[c].begin() + shift, acc[c].end(), acc[c].begin());
            std::fill(acc[c].end() - shift, acc[c].end(), 0.0f);
        }
        std::move(div.begin() + shift, div.end(), div.begin());
        std::fill(div.end() - shift, div.end(), 0.0f);
        base = upto;
    };

    OrtModel model(model_path, threads, provider);
    const long chunks = (M + step - 1) / step;
    std::vector<float> part(kChunk), spek((size_t)4 * kDimF * kDimT), wave(kChunk);
    double t_model = 0;
    long done = 0;
    for (long i = 0; i < M; i += step) {
        if (i > base) emit_until(i);
        const long end = std::min(i + kChunk, M);
        const long actual = end - i;
        std::vector<double> w(actual);  // np.hanning(actual)
        for (long k = 0; k < actual; k++) {
            double n = (double)(1 - actual + 2 * k);
            w[k] = actual > 1 ? 0.5 + 0.5 * std::cos(M_PI * n / (double)(actual - 1)) : 1.0;
        }
        for (int c = 0; c < 2; c++) {
            for (long k = 0; k < kChunk; k++) part[k] = k < actual ? mixture(c, i + k) : 0.0f;
            stft_chunk(part.data(), swin, spek.data() + (size_t)(2 * c) * kDimF * kDimT,
                       spek.data() + (size_t)(2 * c + 1) * kDimF * kDimT);
        }
        for (int p = 0; p < 4; p++)  // spek[:, :, :3, :] *= 0
            std::fill(spek.begin() + (size_t)p * kDimF * kDimT, spek.begin() + (size_t)p * kDimF * kDimT + 3 * kDimT, 0.0f);
        double t0 = now();
        auto outs = model.run(spek.data(), {1, 4, kDimF, kDimT});
        t_model += now() - t0;
        if (outs.empty() || outs[0].data.size() != spek.size()) throw std::runtime_error("unexpected model output");
        const float* o = outs[0].data.data();
        for (int c = 0; c < 2; c++) {
            istft_chunk(o + (size_t)(2 * c) * kDimF * kDimT, o + (size_t)(2 * c + 1) * kDimF * kDimT, swin, wave.data());
            for (long k = 0; k < actual; k++) acc[c][i - base + k] += (float)((double)wave[k] * w[k]);
        }
        for (long k = 0; k < actual; k++) div[i - base + k] = (float)((double)div[i - base + k] + w[k]);
        done++;
        if (progress && !progress((double)done / chunks)) throw Cancelled{};
    }
    emit_until(std::min(M, base + kChunk));
    res.t_model = t_model;

    // secondary = (-primary * compensate) + mix: the model's "Vocals".
    for (int c = 0; c < 2; c++)
        for (long j = 0; j < N; j++) mix[c][j] = (-primary[c][j] * kCompensate) + mix[c][j];
    // Karaoke 2's primary stem is "Instrumental": the backing vocals.
    bool wrote_backing = write_stem(backing_wav, primary[0], primary[1], &res.backing_rms);
    std::vector<float>().swap(primary[0]);
    std::vector<float>().swap(primary[1]);
    bool wrote_lead = write_stem(lead_wav, mix[0], mix[1], &res.lead_rms);
    if (!wrote_backing || !wrote_lead) {
        res.why = !wrote_lead ? "no lead vocal was found" : "no backing vocals were found";
        return res;
    }
    res.ok = true;
    return res;
}

}  // namespace sda
