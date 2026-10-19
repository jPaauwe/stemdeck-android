#include "librosa.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "fft.hpp"

namespace sda {

std::vector<double> hann_periodic(int n) {
    std::vector<double> w(n);
    for (int k = 0; k < n; k++) w[k] = 0.5 - 0.5 * std::cos(2.0 * M_PI * k / n);
    return w;
}

Spec stft_power(const std::vector<float>& y, int n_fft, int hop, double power) {
    const int pad = n_fft / 2;
    const long len = (long)y.size();
    if (len == 0) return {};
    const long padded = len + 2 * pad;
    if (padded < n_fft) throw std::runtime_error("signal too short for the STFT");
    Spec S;
    S.frames = (size_t)(1 + (padded - n_fft) / hop);
    S.bins = (size_t)(n_fft / 2 + 1);
    S.v.resize(S.frames * S.bins);
    const auto win = hann_periodic(n_fft);
    std::vector<double> frame(n_fft);
    std::vector<std::complex<double>> spec;
    for (size_t t = 0; t < S.frames; t++) {
        long start = (long)t * hop - pad;
        for (int i = 0; i < n_fft; i++) {
            long j = start + i;
            frame[i] = (j >= 0 && j < len) ? win[i] * (double)y[j] : 0.0;
        }
        RealFFT::forward(frame, spec);
        float* row = S.row(t);
        for (size_t k = 0; k < S.bins; k++) {
            // complex64 then np.abs (float32), then ** power in float32
            std::complex<float> c((float)spec[k].real(), (float)spec[k].imag());
            float mag = std::abs(c);
            row[k] = power == 1.0 ? mag : power == 2.0 ? mag * mag : (float)std::pow(mag, power);
        }
    }
    return S;
}

namespace {
double hz_to_mel(double f) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + std::log(f / min_log_hz) / logstep : f / f_sp;
}
double mel_to_hz(double m) {
    const double f_sp = 200.0 / 3, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
}
}  // namespace

std::vector<float> mel_filters(double sr, int n_fft, int n_mels, double fmin, double fmax) {
    const int nb = n_fft / 2 + 1;
    std::vector<double> fft_f(nb);
    for (int k = 0; k < nb; k++) fft_f[k] = k * sr / n_fft;  // np.fft.rfftfreq(n_fft, 1/sr)
    // np.linspace(min_mel, max_mel, n_mels + 2)
    const int n = n_mels + 2;
    const double lo = hz_to_mel(fmin), hi = hz_to_mel(fmax);
    std::vector<double> mel_f(n);
    const double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n; i++) mel_f[i] = mel_to_hz(lo + i * step);
    mel_f[n - 1] = mel_to_hz(hi);
    std::vector<float> w((size_t)n_mels * nb);
    for (int i = 0; i < n_mels; i++) {
        double fd0 = mel_f[i + 1] - mel_f[i], fd1 = mel_f[i + 2] - mel_f[i + 1];
        double enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (int k = 0; k < nb; k++) {
            double lower = -(mel_f[i] - fft_f[k]) / fd0;
            double upper = (mel_f[i + 2] - fft_f[k]) / fd1;
            float v = (float)std::max(0.0, std::min(lower, upper));
            w[(size_t)i * nb + k] = (float)((double)v * enorm);
        }
    }
    return w;
}

std::vector<float> onset_strength_median(const std::vector<float>& y, double sr, int n_fft, int hop, int n_mels) {
    Spec S = stft_power(y, n_fft, hop, 2.0);
    if (S.frames == 0) return {};
    const auto fb = mel_filters(sr, n_fft, n_mels, 0.0, 0.5 * sr);
    const size_t nb = S.bins;
    // Each filter's nonzero range, so the product skips the zeros (it is exact).
    std::vector<int> first(n_mels, 0), last(n_mels, -1);
    for (int m = 0; m < n_mels; m++)
        for (size_t k = 0; k < nb; k++)
            if (fb[(size_t)m * nb + k] != 0.0f) {
                if (last[m] < 0) first[m] = (int)k;
                last[m] = (int)k;
            }
    // mel power -> dB (float32), then the top_db floor over the whole spectrogram.
    std::vector<float> db(S.frames * n_mels);
    float peak = -INFINITY;
    for (size_t t = 0; t < S.frames; t++) {
        const float* row = S.row(t);
        for (int m = 0; m < n_mels; m++) {
            double acc = 0;
            const float* f = fb.data() + (size_t)m * nb;
            for (int k = first[m]; k <= last[m]; k++) acc += (double)f[k] * row[k];
            float p = (float)acc;
            float v = (float)(10.0 * std::log10((double)std::max(1e-10f, p)));
            db[t * n_mels + m] = v;
            peak = std::max(peak, v);
        }
    }
    const float floor_db = peak - 80.0f;
    for (auto& v : db) v = std::max(v, floor_db);
    // Spectral flux (lag 1), half-wave rectified, median over the mel bands.
    const int pad = 1 + n_fft / (2 * hop);
    std::vector<float> env(S.frames, 0.0f);
    std::vector<float> diff(n_mels);
    for (size_t t = 1; t < S.frames; t++) {
        for (int m = 0; m < n_mels; m++) diff[m] = std::max(0.0f, db[t * n_mels + m] - db[(t - 1) * n_mels + m]);
        float med = median(diff);
        size_t at = (t - 1) + pad;
        if (at < S.frames) env[at] = med;
    }
    return env;
}

std::vector<int> peak_pick(const std::vector<double>& x, int pre_max, int post_max, int pre_avg, int post_avg,
                           double delta, int wait) {
    std::vector<int> out;
    const int n_x = (int)x.size();
    if (n_x == 0) return out;
    auto mx = [&](int a, int b) { double m = x[a]; for (int i = a + 1; i < b; i++) m = std::max(m, x[i]); return m; };
    auto mean = [&](int a, int b) {
        // np.mean on float64: pairwise summation; for these short windows a
        // straight sum in double agrees to the last bit or two.
        double s = 0; for (int i = a; i < b; i++) s += x[i]; return s / (b - a);
    };
    bool p0 = x[0] >= mx(0, std::min(post_max, n_x));
    p0 = p0 && (x[0] >= mean(0, std::min(post_avg, n_x)) + delta);
    if (p0) out.push_back(0);
    int n = p0 ? wait + 1 : 1;
    while (n < n_x) {
        double maxn = mx(std::max(0, n - pre_max), std::min(n + post_max, n_x));
        if (!(x[n] == maxn)) { n += 1; continue; }
        double avgn = mean(std::max(0, n - pre_avg), std::min(n + post_avg, n_x));
        if (!(x[n] >= avgn + delta)) { n += 1; continue; }
        out.push_back(n);
        n += wait + 1;
    }
    return out;
}

std::vector<double> onset_detect_times(const std::vector<float>& env_in, double sr, int hop) {
    std::vector<float> env = env_in;
    if (env.empty()) return {};
    // normalize: (x - min) / (max + tiny), in float32
    float mn = *std::min_element(env.begin(), env.end());
    for (auto& v : env) v = v - mn;
    float mxv = *std::max_element(env.begin(), env.end());
    const float tiny = 1.17549435e-38f;  // np.finfo(float32).tiny
    for (auto& v : env) v = v / (mxv + tiny);
    bool any = false;
    for (float v : env) { if (v != 0.0f) any = true; if (!std::isfinite(v)) return {}; }
    if (!any) return {};
    auto fl = [&](double secs) { return std::floor(secs * sr / hop); };  // python's float // int
    int pre_max = (int)std::ceil(fl(0.03)), post_max = (int)std::ceil(0.00 * sr / hop >= 0 ? std::floor(0.0 * sr / hop) + 1 : 1);
    int pre_avg = (int)std::ceil(fl(0.10)), post_avg = (int)std::ceil(fl(0.10) + 1), wait = (int)std::ceil(fl(0.03));
    std::vector<double> xd(env.begin(), env.end());
    // peak_pick runs on the float32 array; comparisons are exact either way,
    // and the means are tiny windows.
    auto peaks = peak_pick(xd, pre_max, post_max, pre_avg, post_avg, 0.07, wait);
    std::vector<double> t(peaks.size());
    for (size_t i = 0; i < peaks.size(); i++) t[i] = (double)peaks[i] * hop / sr;
    return t;
}

std::vector<float> rms(const std::vector<float>& y, int frame_length, int hop) {
    const long len = (long)y.size(), pad = frame_length / 2;
    const long padded = len + 2 * pad;
    if (padded < frame_length) return {};
    const size_t frames = (size_t)(1 + (padded - frame_length) / hop);
    std::vector<float> out(frames);
    for (size_t t = 0; t < frames; t++) {
        long start = (long)t * hop - pad;
        double s = 0;
        for (int i = 0; i < frame_length; i++) {
            long j = start + i;
            if (j >= 0 && j < len) { float v = y[j]; s += (double)(v * v); }
        }
        out[t] = (float)std::sqrt((float)(s / frame_length));
    }
    return out;
}

template <typename T>
double percentile(const std::vector<T>& x, double q) {
    if (x.empty()) return NAN;
    std::vector<T> s(x);
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    double vi = (q / 100.0) * (double)(n - 1);
    double prev = std::floor(vi);
    size_t i0 = (size_t)prev, i1 = std::min(i0 + 1, n - 1);
    double g = vi - prev;
    T a = s[i0], b = s[i1];
    T diff = (T)(b - a);
    double r = (double)a + (double)diff * g;
    if (g >= 0.5) r = (double)b - (double)diff * (1.0 - g);
    return r;
}
template double percentile<float>(const std::vector<float>&, double);
template double percentile<double>(const std::vector<double>&, double);

template <typename T>
T median(std::vector<T> x) {
    const size_t n = x.size();
    if (n == 0) return (T)NAN;
    size_t h = n / 2;
    std::nth_element(x.begin(), x.begin() + h, x.end());
    T hi = x[h];
    if (n % 2) return hi;
    T lo = *std::max_element(x.begin(), x.begin() + h);
    return (T)(lo + hi) / (T)2;
}
template float median<float>(std::vector<float>);
template double median<double>(std::vector<double>);

}  // namespace sda
