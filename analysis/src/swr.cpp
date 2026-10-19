#include "swr.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sda {

std::vector<float> downmix_like_ffmpeg(const float* x, size_t frames, int channels) {
    std::vector<float> out(frames);
    if (channels == 1) {
        out.assign(x, x + frames);
    } else if (channels == 2) {
        const float c = (float)M_SQRT1_2;
        for (size_t i = 0; i < frames; i++) out[i] = c * x[2 * i] + c * x[2 * i + 1];
    } else {
        // Not needed by the app (stems are stereo); a plain average.
        for (size_t i = 0; i < frames; i++) {
            float s = 0;
            for (int c = 0; c < channels; c++) s += x[i * channels + c];
            out[i] = s / channels;
        }
    }
    return out;
}

namespace {
// av_bessel_i0 (libavutil/mathematics.c) is a polynomial approximation; the
// series below agrees with it to double precision for the arguments used here.
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 500; k++) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}
}  // namespace

std::vector<float> resample_like_ffmpeg(const std::vector<float>& x, int in_rate, int out_rate) {
    if (in_rate == out_rate) return x;
    const int filter_size = 32, phase_shift = 10;
    const double cutoff = 0.97, beta = 9.0;
    double factor = std::min(out_rate * cutoff / in_rate, 1.0);
    int filter_length = std::max((int)std::ceil(filter_size / factor), 1);
    if (filter_length > 1) filter_length = (filter_length + 1) & ~1;
    // exact_rational
    int g = std::gcd(out_rate, in_rate);
    int num = out_rate / g, den = in_rate / g;
    int phase_count = 1 << phase_shift;
    if (num > phase_count) throw std::invalid_argument("resample ratio not supported");
    phase_count = num;
    // build_filter (float bank, scale 1)
    const int tap_count = filter_length;
    const int center = (tap_count - 1) / 2;
    std::vector<std::vector<float>> bank(phase_count, std::vector<float>(tap_count));
    {
        int ph_nb = phase_count % 2 ? phase_count : phase_count / 2 + 1;
        double f = factor > 1.0 ? 1.0 : factor;
        std::vector<double> tab(tap_count);
        double norm = 0;
        for (int ph = 0; ph < ph_nb; ph++) {
            for (int i = 0; i < tap_count; i++) {
                double xx = M_PI * ((double)(i - center) - (double)ph / phase_count) * f;
                double y = xx == 0 ? 1.0 : std::sin(xx) / xx;
                double w = 2.0 * xx / (f * tap_count * M_PI);
                y *= bessel_i0(beta * std::sqrt(std::max(1 - w * w, 0.0)));
                tab[i] = y;
                if (!ph) norm += y;
            }
            for (int i = 0; i < tap_count; i++) bank[ph][i] = (float)(tab[i] / norm);
            if (phase_count % 2 == 0 && ph > 0)
                for (int i = 0; i < tap_count; i++) bank[phase_count - ph][tap_count - 1 - i] = bank[ph][i];
        }
    }
    const int64_t n = (int64_t)x.size();
    if (n == 0) return {};
    // Mirrored start (invert_initial_buffer): x[-k] = x[k].
    // Mirrored end (resample_flush): `reflection` samples, x[n + j] = x[n - 1 - j].
    const int64_t reflection = (std::min<int64_t>(n, filter_length) + 1) / 2;
    auto at = [&](int64_t i) -> float {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 1 - i;
        if (i < 0 || i >= n) return 0.0f;
        return x[(size_t)i];
    };
    // Position of output k, in input samples: k * den / num. The first output's
    // window starts `center` samples before input 0 (index = -phase_count*center).
    const int64_t total_in = n + reflection;
    std::vector<float> out;
    out.reserve((size_t)(n * num / den + 2));
    int64_t index = 0;  // in units of 1/phase_count input samples, relative to input 0
    while (true) {
        int64_t sample = index / phase_count;
        int ph = (int)(index % phase_count);
        int64_t start = sample - center;
        if (start + filter_length > total_in) break;
        const float* filt = bank[ph].data();
        double acc = 0;
        for (int i = 0; i < filter_length; i++) acc += (double)at(start + i) * filt[i];
        out.push_back((float)acc);
        index += (int64_t)den;  // dst_incr / src_incr = den / num, in phase units: den
    }
    return out;
}

}  // namespace sda
