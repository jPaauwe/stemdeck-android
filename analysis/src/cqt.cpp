#include "cqt.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "fft.hpp"
#include "numpy.hpp"

namespace sda {

namespace {
using cf = std::complex<float>;
using cd = std::complex<double>;

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 500; k++) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

// Linear-phase lowpass for 2:1 decimation: Kaiser window, passband to 0.913
// and stopband from 1.0 of the output Nyquist, 120 dB.
const std::vector<double>& halfband_fir() {
    static const std::vector<double> h = [] {
        const double atten = 120.0, beta = 0.1102 * (atten - 8.7);
        const double fp = 0.913 * 0.25, fs = 0.25;  // cycles per input sample
        const double df = fs - fp;
        int n = (int)std::ceil((atten - 7.95) / (14.36 * df)) + 1;
        if (n % 2 == 0) n++;
        const int m = n / 2;
        const double fc = 0.5 * (fp + fs);
        std::vector<double> t(n);
        double sum = 0;
        const double i0b = bessel_i0(beta);
        for (int i = 0; i < n; i++) {
            double x = i - m;
            double sinc = x == 0 ? 2 * fc : std::sin(2 * M_PI * fc * x) / (M_PI * x);
            double r = (double)(i - m) / m;
            double w = bessel_i0(beta * std::sqrt(std::max(0.0, 1 - r * r))) / i0b;
            t[i] = sinc * w;
            sum += t[i];
        }
        for (auto& v : t) v /= sum;
        return t;
    }();
    return h;
}
}  // namespace

std::vector<float> halve_rate(const std::vector<float>& y) {
    const auto& h = halfband_fir();
    const long n = (long)y.size(), taps = (long)h.size(), m = taps / 2;
    const long out_n = (n + 1) / 2;  // ceil(n * 0.5)
    std::vector<float> out(out_n);
    const double scale = std::sqrt(2.0);  // scale=True: / sqrt(ratio)
    for (long k = 0; k < out_n; k++) {
        long c = 2 * k;
        long lo = std::max(0L, c - m), hi = std::min(n - 1, c + m);
        double acc = 0;
        for (long j = lo; j <= hi; j++) acc += h[j - c + m] * (double)y[j];
        out[k] = (float)(acc * scale);
    }
    return out;
}

double estimate_tuning(const std::vector<float>& y, double sr, int bins_per_octave) {
    // piptrack(y, sr, n_fft=2048, hop=512, fmin=150, fmax=4000, threshold=0.1)
    const int n_fft = 2048, hop = 512;
    Spec S = stft_power(y, n_fft, hop, 1.0);
    const size_t nb = S.bins;
    std::vector<float> pitches, mags;
    std::vector<float> avg(nb), shift(nb), col(nb);
    const double fmin = 150.0, fmax = std::min(4000.0, sr / 2);
    for (size_t t = 0; t < S.frames; t++) {
        const float* x = S.row(t);
        // np.gradient along frequency
        avg[0] = x[1] - x[0];
        avg[nb - 1] = x[nb - 1] - x[nb - 2];
        for (size_t k = 1; k + 1 < nb; k++) avg[k] = (x[k + 1] - x[k - 1]) / 2.0f;
        // _parabolic_interpolation
        shift[0] = shift[nb - 1] = 0;
        for (size_t k = 1; k + 1 < nb; k++) {
            float a = x[k + 1] + x[k - 1] - 2 * x[k];
            double b = (double)(x[k + 1] - x[k - 1]) / 2;
            shift[k] = std::fabs(b) >= std::fabs((double)a) ? 0.0f : (float)(-b / a);
        }
        float ref = 0;
        for (size_t k = 0; k < nb; k++) ref = std::max(ref, x[k]);
        float ref_value = (float)(0.1 * ref);
        for (size_t k = 0; k < nb; k++) col[k] = x[k] * (float)(x[k] > ref_value);
        for (size_t k = 0; k < nb; k++) {
            double f = k * sr / n_fft;
            if (!(fmin <= f && f < fmax)) continue;
            bool lmax;
            if (k == 0) lmax = false;
            else if (k == nb - 1) lmax = col[k] > col[k - 1];
            else lmax = col[k] > col[k - 1] && col[k] >= col[k + 1];
            if (!lmax) continue;
            float dskew = 0.5f * avg[k] * shift[k];
            pitches.push_back((float)(((double)k + (double)shift[k]) * sr / n_fft));
            mags.push_back(x[k] + dskew);
        }
    }
    // threshold = median of the magnitudes where a pitch was found
    std::vector<float> pm;
    for (size_t i = 0; i < pitches.size(); i++) if (pitches[i] > 0) pm.push_back(mags[i]);
    float threshold = pm.empty() ? 0.0f : median(pm);
    std::vector<float> freqs;
    for (size_t i = 0; i < pitches.size(); i++)
        if (pitches[i] > 0 && mags[i] >= threshold) freqs.push_back(pitches[i]);
    if (freqs.empty()) return 0.0;
    // pitch_tuning(resolution=0.01)
    const int n_bins = (int)std::ceil(1.0 / 0.01);  // 100 bins, 101 edges
    std::vector<double> edges(n_bins + 1);
    {
        const double step = 1.0 / n_bins;
        for (int i = 0; i <= n_bins; i++) edges[i] = -0.5 + i * step;
        edges[n_bins] = 0.5;
    }
    std::vector<long> counts(n_bins, 0);
    for (float f : freqs) {
        float octs = std::log2(f / (float)(440.0 / 16));
        float v = (float)bins_per_octave * octs;
        float r = std::fmod(v, 1.0f);
        if (r < 0) r += 1.0f;  // np.mod: sign of the divisor
        if (r >= 0.5f) r -= 1.0f;
        double a = r;
        if (a < edges[0] || a > edges[n_bins]) continue;
        long idx = (long)((a - edges[0]) * (n_bins / (edges[n_bins] - edges[0])));
        if (idx == n_bins) idx--;
        if (a < edges[idx]) idx--;
        if (idx < n_bins - 1 && a >= edges[idx + 1]) idx++;
        counts[idx]++;
    }
    long best = 0;
    for (int i = 1; i < n_bins; i++) if (counts[i] > counts[best]) best = i;
    return edges[best];
}

Spec chroma_cqt(const std::vector<float>& y_in, double sr_in, int hop_in, double fmin_c, int n_octaves) {
    const int bpo = 36, n_bins = n_octaves * bpo, n_filters = bpo;
    const double tuning = estimate_tuning(y_in, sr_in, bpo);
    const double fmin = fmin_c * std::pow(2.0, tuning / bpo);
    // interval_frequencies("equal")
    std::vector<double> freqs(n_bins);
    for (int o = 0; o < n_octaves; o++)
        for (int k = 0; k < bpo; k++)
            freqs[o * bpo + k] = (std::pow(2.0, (double)o) * std::pow(2.0, (double)k / bpo)) * fmin;
    std::sort(freqs.begin(), freqs.end());
    // _relative_bandwidth
    std::vector<double> alpha(n_bins);
    {
        std::vector<double> logf(n_bins), bp(n_bins);
        for (int i = 0; i < n_bins; i++) logf[i] = std::log2(freqs[i]);
        bp[0] = 1 / (logf[1] - logf[0]);
        bp[n_bins - 1] = 1 / (logf[n_bins - 1] - logf[n_bins - 2]);
        for (int i = 1; i < n_bins - 1; i++) bp[i] = 2 / (logf[i + 1] - logf[i - 1]);
        for (int i = 0; i < n_bins; i++) {
            double p = std::pow(2.0, 2 / bp[i]);
            alpha[i] = (p - 1) / (p + 1);
        }
    }
    const double hann_bw = 1.50018310546875;
    auto lengths_for = [&](double sr, int from, int count, std::vector<double>& out) {
        out.resize(count);
        for (int i = 0; i < count; i++) {
            double Q = 1.0 / alpha[from + i];
            out[i] = Q * sr / freqs[from + i];
        }
    };
    double cutoff = 0;
    for (int i = 0; i < n_bins; i++) cutoff = std::max(cutoff, freqs[i] * (1 + 0.5 * hann_bw * alpha[i]));
    const double nyquist = sr_in / 2.0;
    if (cutoff > nyquist) throw std::runtime_error("CQT bins above Nyquist");
    // __early_downsample
    std::vector<float> y = y_in;
    double sr = sr_in;
    int hop = hop_in;
    {
        int c1 = std::max(0, (int)std::ceil(std::log2(nyquist / cutoff)) - 1 - 1);
        int twos = 0;
        for (int x = hop; x > 0 && x % 2 == 0; x /= 2) twos++;
        int c2 = std::max(0, twos - n_octaves + 1);
        int count = std::min(c1, c2);
        if (count > 0) {
            int factor = 1 << count;
            hop /= factor;
            if ((long)y.size() < factor) throw std::runtime_error("signal too short for the CQT");
            // resample(y, orig_sr=factor, target_sr=1, scale=True): here as `count` halvings
            for (int i = 0; i < count; i++) y = halve_rate(y);
            sr = sr / factor;
        }
    }
    // octave by octave, from the top
    std::vector<std::vector<std::vector<cf>>> resp;  // [octave][frame][filter]
    std::vector<int> resp_bins;
    std::vector<float> my_y = y;
    double my_sr = sr;
    int my_hop = hop;
    size_t min_frames = SIZE_MAX;
    for (int i = 0; i < n_octaves; i++) {
        int hi = n_bins - n_filters * i, lo = hi - n_filters;
        // __vqt_filter_fft: wavelet basis, padded to a power of two, FFT, sparsified
        std::vector<double> lens;
        lengths_for(my_sr, lo, n_filters, lens);
        double max_len = *std::max_element(lens.begin(), lens.end());
        int n_fft = (int)std::pow(2.0, std::ceil(std::log2(max_len)));
        const int nb = n_fft / 2 + 1;
        std::vector<std::vector<std::pair<int, cf>>> basis(n_filters);
        std::vector<cd> buf(n_fft), spec;
        for (int f = 0; f < n_filters; f++) {
            double ilen = lens[f], freq = freqs[lo + f];
            double start = std::floor(-ilen / 2), stop = std::floor(ilen / 2);
            int n = (int)std::ceil(stop - start);
            std::vector<cd> sig(n);
            auto win = hann_periodic(n);
            std::vector<double> absv(n);
            for (int j = 0; j < n; j++) {
                double ph = (start + j) * 2 * M_PI * freq / my_sr;
                sig[j] = cd(std::cos(ph), std::sin(ph)) * win[j];
                absv[j] = std::abs(sig[j]);
            }
            const double l1 = np_sum(absv);  // util.normalize(norm=1)
            int lpad = (n_fft - n) / 2;
            std::fill(buf.begin(), buf.end(), cd(0, 0));
            for (int j = 0; j < n; j++) {
                cd v = sig[j] / l1;
                cf c((float)v.real(), (float)v.imag());               // complex64 basis
                cd scaled = cd(c.real(), c.imag()) * (ilen / n_fft);  // basis *= lengths / n_fft
                buf[lpad + j] = cd((float)scaled.real(), (float)scaled.imag());
            }
            RealFFT::forward_c(buf, spec);
            // sparsify_rows(quantile=0.01), on the positive frequencies
            std::vector<float> mags(nb);
            std::vector<cf> row(nb);
            for (int k = 0; k < nb; k++) {
                row[k] = cf((float)spec[k].real(), (float)spec[k].imag());
                mags[k] = std::abs(row[k]);
            }
            std::vector<float> sorted = mags;
            std::sort(sorted.begin(), sorted.end());
            float norm = np_sum(mags);
            float cum = 0;
            int j = 0;
            for (; j < nb; j++) {
                cum += sorted[j] / norm;
                if (!(cum < 0.01f)) break;
            }
            if (j >= nb) j = 0;  // argmin of all-True is 0
            float thresh = sorted[j];
            const float rescale = (float)std::sqrt(sr / my_sr);
            for (int k = 0; k < nb; k++)
                if (mags[k] >= thresh) basis[f].push_back({k, row[k] * rescale});
        }
        // __cqt_response: STFT with a rectangular window, then basis . D
        const long len = (long)my_y.size(), pad = n_fft / 2;
        const long frames = 1 + (len + 2 * pad - n_fft) / my_hop;
        std::vector<std::vector<cf>> out((size_t)frames, std::vector<cf>(n_filters));
        std::vector<double> fr(n_fft);
        std::vector<cd> D;
        for (long t = 0; t < frames; t++) {
            long s = t * my_hop - pad;
            for (int k = 0; k < n_fft; k++) {
                long jdx = s + k;
                fr[k] = (jdx >= 0 && jdx < len) ? (double)my_y[jdx] : 0.0;
            }
            RealFFT::forward(fr, D);
            for (int f = 0; f < n_filters; f++) {
                cd acc(0, 0);
                for (auto& [k, v] : basis[f]) {
                    cf d((float)D[k].real(), (float)D[k].imag());
                    acc += cd(v.real(), v.imag()) * cd(d.real(), d.imag());
                }
                out[t][f] = cf((float)acc.real(), (float)acc.imag());
            }
        }
        min_frames = std::min(min_frames, (size_t)frames);
        resp.push_back(std::move(out));
        resp_bins.push_back(lo);
        if (my_hop % 2 == 0) {
            my_hop /= 2;
            my_sr /= 2.0;
            my_y = halve_rate(my_y);
        }
    }
    // |V| / sqrt(lengths), with lengths at the (early-downsampled) rate
    std::vector<double> all_len;
    lengths_for(sr, 0, n_bins, all_len);
    Spec C;
    C.frames = min_frames;
    C.bins = 12;
    C.v.assign(C.frames * 12, 0.0f);
    std::vector<float> mag(n_bins);
    for (size_t t = 0; t < C.frames; t++) {
        for (size_t o = 0; o < resp.size(); o++) {
            int lo = resp_bins[o];
            for (int f = 0; f < n_filters; f++) {
                cf v = resp[o][t][f];
                v = cf((float)(v.real() / std::sqrt(all_len[lo + f])), (float)(v.imag() / std::sqrt(all_len[lo + f])));
                mag[lo + f] = std::abs(v);
            }
        }
        // cq_to_chroma(fmin=C1): bin j of an octave counts towards ((j + 1) % 36) / 3
        float* row = C.row(t);
        double acc[12] = {0};
        for (int b = 0; b < n_bins; b++) acc[((b % bpo) + 1) % bpo / 3] += mag[b];
        float mx = 0;
        for (int c = 0; c < 12; c++) {
            row[c] = std::max(0.0f, (float)acc[c]);
            mx = std::max(mx, std::fabs(row[c]));
        }
        double length = mx < 1.17549435e-38f ? 1.0 : (double)mx;
        for (int c = 0; c < 12; c++) row[c] = (float)((double)row[c] / length);
    }
    return C;
}

}  // namespace sda
