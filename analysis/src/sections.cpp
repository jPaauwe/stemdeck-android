#include "sections.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include "fft.hpp"
#include "numpy.hpp"
#include "ort.hpp"
#include "pipeline.hpp"
#include "wav.hpp"

namespace sda {

const char* const kHarmonixLabels[10] = {"start", "end", "intro", "outro", "break",
                                         "bridge", "inst", "solo", "verse", "chorus"};

namespace {
constexpr int kFrame = 2048;
constexpr double kHop = 441.0;  // 44100 / fps 100, a float in madmom
constexpr int kBlock = 1 << 16;

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

// ── madmom front end ─────────────────────────────────────────────────

MadmomSignal madmom_load_stem(const std::string& path) {
    WavReader r(path);
    if (r.sample_rate() != 44100) throw std::runtime_error("section stems must be 44.1 kHz");
    MadmomSignal s;
    s.int16 = !r.is_float();
    s.x.reserve(r.frames());
    std::vector<float> buf;
    const int c = r.channels();
    while (size_t n = r.read(buf, kBlock)) {
        for (size_t i = 0; i < n; i++) {
            if (c == 1) {
                s.x.push_back(s.int16 ? std::nearbyint(buf[i] * 32768.0f) : buf[i]);
                continue;
            }
            if (s.int16) {
                // np.mean over int16 channels accumulates in float64; astype(int16) truncates.
                double acc = 0;
                for (int k = 0; k < c; k++) acc += (double)std::nearbyint(buf[i * c + k] * 32768.0f);
                s.x.push_back((float)(int16_t)(acc / c));
            } else {
                float acc = 0;
                for (int k = 0; k < c; k++) acc += buf[i * c + k];
                s.x.push_back(acc / (float)c);
            }
        }
    }
    return s;
}

MadmomSignal madmom_load_mix(const std::vector<std::string>& paths) {
    std::vector<std::unique_ptr<WavReader>> rs;
    size_t longest = 0;
    int c = 0;
    for (auto& p : paths) {
        rs.emplace_back(new WavReader(p));
        if (rs.back()->sample_rate() != 44100) throw std::runtime_error("section stems must be 44.1 kHz");
        if (c && rs.back()->channels() != c) throw std::runtime_error("section stems differ in channel count");
        c = rs.back()->channels();
        longest = std::max(longest, rs.back()->frames());
    }
    MadmomSignal s;
    s.int16 = false;
    s.x.reserve(longest);
    std::vector<float> acc, buf;
    for (size_t done = 0; done < longest;) {
        size_t n = std::min((size_t)kBlock, longest - done);
        acc.assign(n * c, 0.0f);
        for (auto& r : rs) {  // amix: out += in * 1.0f, input by input (duration=longest: silence after the end)
            size_t got = r->read(buf, n);
            for (size_t i = 0; i < got * c; i++) acc[i] += buf[i];
        }
        for (size_t i = 0; i < n; i++) {
            float m = 0;
            for (int k = 0; k < c; k++) m += acc[i * c + k];
            s.x.push_back(m / (float)c);
        }
        done += n;
    }
    return s;
}

std::vector<float> madmom_filterbank(int* bands_out) {
    const int nbins = kFrame / 2;
    // np.fft.fftfreq(2048, 1 / 44100)[:1024]
    const double d = 1.0 / 44100.0;
    const double val = 1.0 / (kFrame * d);
    std::vector<double> bf(nbins);
    for (int k = 0; k < nbins; k++) bf[k] = (double)k * val;
    // log_frequencies(12, 30, 17000, 440)
    const double fref = 440.0, fmin = 30.0, fmax = 17000.0;
    const double left = std::floor(std::log2(fmin / fref) * 12.0);
    const double right = std::ceil(std::log2(fmax / fref) * 12.0);
    std::vector<double> freqs;
    for (double v = left; v < right; v += 1.0) freqs.push_back(fref * std::pow(2.0, v / 12.0));
    freqs.erase(freqs.begin(), std::lower_bound(freqs.begin(), freqs.end(), fmin));
    freqs.erase(std::upper_bound(freqs.begin(), freqs.end(), fmax), freqs.end());
    // frequencies2bins(..., unique_bins=True)
    std::vector<long> bins;
    for (double f : freqs) {
        long i = std::lower_bound(bf.begin(), bf.end(), f) - bf.begin();
        i = std::max(1L, std::min((long)nbins - 1, i));
        if (f - bf[i - 1] < bf[i] - f) i -= 1;
        bins.push_back(i);
    }
    std::sort(bins.begin(), bins.end());
    bins.erase(std::unique(bins.begin(), bins.end()), bins.end());
    if (bins.size() < 3) throw std::runtime_error("not enough bins for the filterbank");
    const int bands = (int)bins.size() - 2;
    std::vector<float> fb((size_t)nbins * bands, 0.0f);
    for (int b = 0; b < bands; b++) {
        long start = bins[b], center = bins[b + 1], stop = bins[b + 2];
        if (stop - start < 2) {
            center = start;
            stop = start + 1;
        }
        const long rc = center - start, rs = stop - start;
        std::vector<float> data(rs);
        // np.linspace(0, 1, rc, endpoint=False) and np.linspace(1, 0, rs - rc, endpoint=False)
        for (long i = 0; i < rc; i++) data[i] = (float)((double)i * (1.0 / (double)rc) + 0.0);
        const long nf = rs - rc;
        for (long i = 0; i < nf; i++) data[rc + i] = (float)((double)i * (-1.0 / (double)nf) + 1.0);
        const float sum = np_sum(data.data(), data.size());
        for (auto& v : data) v = v / sum;
        for (long i = 0; i < rs; i++) {
            long bin = start + i;
            if (bin < 0 || bin >= nbins) continue;
            float& t = fb[(size_t)bin * bands + b];
            t = std::max(t, data[i]);
        }
    }
    if (bands_out) *bands_out = bands;
    return fb;
}

std::vector<float> madmom_logspec(const MadmomSignal& s, bool int16_window, size_t* frames_out) {
    int bands = 0;
    static const std::vector<float> fb = madmom_filterbank(&bands);
    bands = (int)(fb.size() / (kFrame / 2));
    // Sparse columns of the filterbank.
    struct Tap { int bin; float w; };
    std::vector<std::vector<Tap>> taps(bands);
    for (int k = 0; k < kFrame / 2; k++)
        for (int b = 0; b < bands; b++)
            if (fb[(size_t)k * bands + b] != 0.0f) taps[b].push_back({k, fb[(size_t)k * bands + b]});

    // np.hanning(2048), divided by np.iinfo(int16).max for integer signals.
    std::vector<double> win(kFrame);
    for (int i = 0; i < kFrame; i++) {
        double n = (double)(1 - kFrame + 2 * i);
        win[i] = 0.5 + 0.5 * std::cos(M_PI * n / (double)(kFrame - 1));
        if (int16_window) win[i] = win[i] / 32767.0;
    }
    const size_t n = s.x.size();
    const size_t frames = (size_t)std::ceil((double)n / kHop);
    std::vector<float> out(frames * bands);
    std::vector<double> buf(kFrame);
    std::vector<std::complex<double>> spec;
    std::vector<float> mag(kFrame / 2);
    for (size_t f = 0; f < frames; f++) {
        const long ref = (long)((double)f * kHop);
        const long start = ref - kFrame / 2;
        for (int i = 0; i < kFrame; i++) {
            long j = start + i;
            double v = (j >= 0 && (size_t)j < n) ? (double)s.x[j] : 0.0;
            buf[i] = v * win[i];
        }
        RealFFT::forward(buf, spec);
        for (int k = 0; k < kFrame / 2; k++) {
            // complex64 storage, then a float64 magnitude rounded to float32
            double re = (double)(float)spec[k].real(), im = (double)(float)spec[k].imag();
            mag[k] = (float)std::sqrt(re * re + im * im);
        }
        float* o = out.data() + f * bands;
        for (int b = 0; b < bands; b++) {
            double acc = 0;
            for (const Tap& t : taps[b]) acc += (double)mag[t.bin] * (double)t.w;
            float v = (float)acc + 1.0f;
            o[b] = (float)std::log10((double)v);
        }
    }
    if (frames_out) *frames_out = frames;
    return out;
}

// ── functional post-processing (all-in-one-infer) ────────────────────

namespace {
float sigmoidf(float x) { return (float)(1.0 / (1.0 + std::exp(-(double)x))); }

// Sliding maximum over x padded with `pad` on both sides: out[i] = max(P[i : i + w]).
std::vector<float> sliding_max(const std::vector<float>& x, int lpad, int rpad, float pad) {
    const size_t T = x.size();
    const size_t w = (size_t)lpad + rpad + 1;
    auto at = [&](size_t j) -> float {  // padded index
        if (j < (size_t)lpad || j >= (size_t)lpad + T) return pad;
        return x[j - lpad];
    };
    std::vector<float> out(T);
    std::deque<size_t> dq;
    size_t next = 0;
    for (size_t i = 0; i < T; i++) {
        while (next < i + w) {
            float v = at(next);
            while (!dq.empty() && at(dq.back()) <= v) dq.pop_back();
            dq.push_back(next++);
        }
        while (dq.front() < i) dq.pop_front();
        out[i] = at(dq.front());
    }
    return out;
}
}  // namespace

std::vector<RawSegment> functional_segments(const SectionActivations& a) {
    const size_t T = a.frames;
    if (T == 0) return {};
    std::vector<float> raw(T);
    for (size_t t = 0; t < T; t++) raw[t] = sigmoidf(a.section_logits[t]);
    // softmax over the 10 labels
    std::vector<float> prob(10 * T);
    for (size_t t = 0; t < T; t++) {
        double mx = -1e300;
        for (int l = 0; l < 10; l++) mx = std::max(mx, (double)a.function_logits[l * T + t]);
        double e[10], sum = 0;
        for (int l = 0; l < 10; l++) sum += (e[l] = std::exp((double)a.function_logits[l * T + t] - mx));
        for (int l = 0; l < 10; l++) prob[l * T + t] = (float)(e[l] / sum);
    }
    // local_maxima(filter_size = 4 * min_hops_per_beat + 1 = 97)
    const int half = 48;
    std::vector<float> mx = sliding_max(raw, half, half, -std::numeric_limits<float>::infinity());
    std::vector<float> ps(T, 0.0f);
    for (size_t t = 0; t < T; t++)
        if (raw[t] == mx[t]) ps[t] = raw[t];
    // peak_picking(window_past = window_future = 12 * fps)
    const int wp = 12 * kSecFps, wf = 12 * kSecFps;
    std::vector<float> pmax = sliding_max(ps, wp, wf, 0.0f);
    std::vector<float> padded((size_t)wp + T + wf, 0.0f);
    std::copy(ps.begin(), ps.end(), padded.begin() + wp);
    std::vector<char> boundary(T, 0);
    for (size_t t = 0; t < T; t++) {
        if (!(ps[t] == pmax[t] && ps[t] > 0)) continue;
        float past = np_sum(padded.data() + t, (size_t)wp) / (float)wp;
        float future = np_sum(padded.data() + t + wp + 1, (size_t)wf) / (float)wf;
        float strength = ps[t] - ((past + future) / 2.0f);
        boundary[t] = strength > 0.0f;
    }
    const double duration = (double)(T * 441) / 44100.0;
    std::vector<double> times;
    std::vector<size_t> idx;
    for (size_t t = 0; t < T; t++)
        if (boundary[t]) {
            times.push_back((double)(t * 441) / 44100.0);
            if (t > 0) idx.push_back(t);
        }
    if (times.empty()) return {};  // upstream indexes times[0] and fails: no sections
    if (times[0] != 0) times.insert(times.begin(), 0.0);
    if (times.back() != duration) times.push_back(duration);
    // labels: argmax of the mean probability over each segment's frames
    std::vector<int> labels;
    size_t lo = 0;
    std::vector<size_t> cuts = idx;
    cuts.push_back(T);
    for (size_t c : cuts) {
        size_t hi = c;
        int best = 0;
        float bestv = -1;
        if (hi > lo) {
            for (int l = 0; l < 10; l++) {
                float m = np_sum(prob.data() + l * T + lo, hi - lo) / (float)(hi - lo);
                if (m > bestv) { bestv = m; best = l; }
            }
        } else {
            best = 0;  // mean of an empty slice is nan; argmax of all-nan is 0
        }
        labels.push_back(best);
        lo = hi;
    }
    std::vector<RawSegment> segs;
    for (size_t i = 0; i + 1 < times.size() && i < labels.size(); i++)
        segs.push_back({times[i], times[i + 1], kHarmonixLabels[labels[i]]});
    return segs;
}

// ── section_refine.py ────────────────────────────────────────────────

namespace {
constexpr double kGridMinConfidence = 70;
constexpr double kBeatSnap = 0.12;
constexpr double kMinActivation = 0.05;
constexpr double kMinNovelty = 0.35;
constexpr double kNoveltyWindow = 8.0;
constexpr double kMinSegment = 6.0;
constexpr double kMinLabelMargin = 0.08;
constexpr double kRecurrenceSimilarity = 0.85;
constexpr double kRecurrenceLabelMargin = 0.25;
constexpr double kTolerance = 0.25;

bool is_sentinel(const std::string& l) { return l == "start" || l == "end"; }

double np_median(std::vector<double> v) {
    const size_t n = v.size();
    std::sort(v.begin(), v.end());
    if (n % 2) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

struct Features {
    size_t frames = 0, dim = 0;
    std::vector<double> v;  // frames x dim
    const double* row(size_t t) const { return v.data() + t * dim; }
};

// Column mean of rows [lo, hi) (numpy's axis-0 reduction adds row after row).
std::vector<double> row_mean(const Features& f, size_t lo, size_t hi) {
    std::vector<double> m(f.dim, 0.0);
    for (size_t t = lo; t < hi; t++)
        for (size_t d = 0; d < f.dim; d++) m[d] += f.row(t)[d];
    for (auto& x : m) x /= (double)(hi - lo);
    return m;
}
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}
double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

double nearest_beat(double value, const std::vector<double>& beats, bool* found) {
    *found = false;
    if (beats.empty()) return 0;
    size_t i = std::lower_bound(beats.begin(), beats.end(), value) - beats.begin();
    size_t a = i > 0 ? i - 1 : 0, b = std::min(beats.size(), i + 1);
    if (a >= b) return 0;
    double best = beats[a];
    for (size_t k = a + 1; k < b; k++)
        if (std::fabs(beats[k] - value) < std::fabs(best - value)) best = beats[k];
    if (std::fabs(best - value) <= kBeatSnap) {
        *found = true;
        return best;
    }
    return 0;
}

double novelty(const Features& f, size_t frame, double fps) {
    long window = std::max(1L, py_round_int(kNoveltyWindow * fps));
    if ((long)frame - window < 0 || (long)frame + window > (long)f.frames) return 0.0;
    auto left = row_mean(f, frame - window, frame);
    auto right = row_mean(f, frame, frame + window);
    double ln = norm(left), rn = norm(right);
    if (ln <= 1e-9 || rn <= 1e-9) return 0.0;
    double sim = dot(left, right) / (ln * rn);
    return 1.0 - std::max(-1.0, std::min(1.0, sim));
}

size_t span_index(const std::vector<RawSegment>& spans, double mid) {
    for (size_t i = 0; i < spans.size(); i++)
        if (spans[i].start <= mid && mid < spans[i].end) return i;
    return spans.size() - 1;
}
bool is_bracket(const std::vector<RawSegment>& spans, size_t i) {
    return is_sentinel(spans[i].label) && (i == 0 || i == spans.size() - 1);
}

struct Record {
    double start, end;
    std::string label;
    double margin;
};
}  // namespace

std::vector<RawSegment> refine_segments(const std::vector<RawSegment>& raw_in, const SectionActivations& a,
                                        const std::vector<double>& beats_in, double confidence) {
    const std::vector<RawSegment>& fallback = raw_in;
    // _parse_segments
    if (raw_in.empty()) return fallback;
    std::vector<RawSegment> spans;
    for (auto s : raw_in) {
        if (!std::isfinite(s.start) || !std::isfinite(s.end) || s.end <= s.start) return fallback;
        std::string l;
        size_t b = s.label.find_first_not_of(" \t\n\r\f\v"), e = s.label.find_last_not_of(" \t\n\r\f\v");
        l = b == std::string::npos ? "" : s.label.substr(b, e - b + 1);
        for (auto& ch : l) ch = (char)std::tolower((unsigned char)ch);
        s.label = l;
        spans.push_back(s);
    }
    std::stable_sort(spans.begin(), spans.end(), [](const RawSegment& x, const RawSegment& y) {
        return x.start != y.start ? x.start < y.start : x.end < y.end;
    });
    for (size_t i = 0; i + 1 < spans.size(); i++)
        if (std::fabs(spans[i + 1].start - spans[i].end) > kTolerance) return fallback;
    const double fps = kSecFps;
    // _evidence_arrays
    const size_t T = a.frames;
    if (T < 2) return fallback;
    std::vector<double> boundary(T);
    for (size_t t = 0; t < T; t++) boundary[t] = (double)sigmoidf(a.section_logits[t]);
    std::vector<double> probs(10 * T);
    for (size_t t = 0; t < T; t++) {
        double mx = -1e300;
        for (int l = 0; l < 10; l++) mx = std::max(mx, (double)a.function_logits[l * T + t]);
        double e[10], sum = 0;
        for (int l = 0; l < 10; l++) sum += (e[l] = std::exp((double)a.function_logits[l * T + t] - mx));
        for (int l = 0; l < 10; l++) probs[l * T + t] = (double)(float)(e[l] / sum);
    }
    Features feat;
    feat.frames = T;
    feat.dim = a.embedding.size() / T;
    feat.v = a.embedding;
    for (double v : feat.v)
        if (!std::isfinite(v)) return fallback;
    for (size_t d = 0; d < feat.dim; d++) {
        std::vector<double> col(T);
        for (size_t t = 0; t < T; t++) col[t] = feat.v[t * feat.dim + d];
        double center = np_median(col);
        for (size_t t = 0; t < T; t++) col[t] = std::fabs(col[t] - center);
        double scale = std::max(np_median(col), 1e-6);
        for (size_t t = 0; t < T; t++) feat.v[t * feat.dim + d] = (feat.v[t * feat.dim + d] - center) / scale;
    }
    const double duration = (double)T / fps;
    // _trusted_beats
    std::vector<double> beats;
    if (confidence >= kGridMinConfidence) {
        for (double b : beats_in) {
            if (!std::isfinite(b) || (!beats.empty() && b <= beats.back())) {
                beats.clear();
                break;
            }
            beats.push_back(b);
        }
    }
    std::vector<double> bounds{spans[0].start};
    for (size_t i = 0; i + 1 < spans.size(); i++) bounds.push_back((spans[i].end + spans[i + 1].start) / 2);
    bounds.push_back(spans.back().end);
    if (!beats.empty())
        for (size_t i = 1; i + 1 < bounds.size(); i++) {
            bool ok;
            double b = nearest_beat(bounds[i], beats, &ok);
            if (ok && b != 0.0) bounds[i] = b;  // `snapped or value`: a beat at 0.0 is falsy
        }
    std::vector<size_t> peaks;
    for (size_t i = 1; i + 1 < T; i++)
        if (boundary[i] > boundary[i - 1] && boundary[i] >= boundary[i + 1] && boundary[i] >= kMinActivation)
            peaks.push_back(i);
    std::stable_sort(peaks.begin(), peaks.end(), [&](size_t x, size_t y) { return boundary[x] > boundary[y]; });
    for (size_t frame : peaks) {
        double candidate = (double)frame / fps;
        bool ok = false;
        double snapped = beats.empty() ? 0 : nearest_beat(candidate, beats, &ok);
        double position = ok ? snapped : candidate;
        if (is_bracket(spans, span_index(spans, candidate))) continue;
        double closest = 1e300;
        for (double b : bounds) closest = std::min(closest, std::fabs(position - b));
        if (closest < kMinSegment) continue;
        if (position < kMinSegment || duration - position < kMinSegment) continue;
        if (novelty(feat, frame, fps) < kMinNovelty) continue;
        bounds.push_back(position);
    }
    std::set<double> uniq(bounds.begin(), bounds.end());
    bounds.assign(uniq.begin(), uniq.end());

    auto clamp_lo_hi = [&](double start, double end, size_t n, size_t* lo, size_t* hi) {
        long l = std::max(0L, std::min((long)n - 1, py_round_int(start * fps)));
        long h = std::max(l + 1, std::min((long)n, py_round_int(end * fps)));
        *lo = (size_t)l;
        *hi = (size_t)h;
    };
    std::vector<Record> records;
    for (size_t i = 0; i + 1 < bounds.size(); i++) {
        double start = bounds[i], end = bounds[i + 1], mid = (start + end) / 2;
        size_t si = span_index(spans, mid);
        if (is_bracket(spans, si)) {
            records.push_back({start, end, spans[si].label, 1.0});
            continue;
        }
        size_t lo, hi;
        clamp_lo_hi(start, end, T, &lo, &hi);
        double mean[10];
        for (int l = 0; l < 10; l++) mean[l] = np_sum(probs.data() + l * T + lo, hi - lo) / (double)(hi - lo);
        std::vector<int> ranked{2, 3, 4, 5, 6, 7, 8, 9};
        std::stable_sort(ranked.begin(), ranked.end(), [&](int x, int y) { return mean[x] > mean[y]; });
        double margin = mean[ranked[0]] - mean[ranked[1]];
        std::string label = margin >= kMinLabelMargin ? kHarmonixLabels[ranked[0]] : "part";
        records.push_back({start, end, label, margin});
    }
    // _regularize_neutral_labels
    std::vector<std::vector<double>> vectors;
    for (auto& r : records) {
        size_t lo, hi;
        clamp_lo_hi(r.start, r.end, T, &lo, &hi);
        auto v = row_mean(feat, lo, hi);
        double n = norm(v);
        if (n > 1e-9)
            for (auto& x : v) x /= n;
        else
            std::fill(v.begin(), v.end(), 0.0);
        vectors.push_back(v);
    }
    for (size_t i = 0; i < records.size(); i++) {
        if (records[i].label != "part") continue;
        std::vector<std::pair<double, size_t>> matches;
        double dur = records[i].end - records[i].start;
        for (size_t j = 0; j < records.size(); j++) {
            const Record& o = records[j];
            if ((i > j ? i - j : j - i) <= 1 || is_sentinel(o.label) || o.label == "part") continue;
            if (o.margin < kRecurrenceLabelMargin) continue;
            double od = o.end - o.start;
            if (od < dur / 2 || od > dur * 2) continue;
            double sim = dot(vectors[i], vectors[j]);
            if (sim >= kRecurrenceSimilarity) matches.push_back({sim, j});
        }
        std::sort(matches.begin(), matches.end(), [](auto& x, auto& y) {
            return x.first != y.first ? x.first > y.first : x.second > y.second;
        });
        if (matches.empty()) continue;
        if (matches.size() > 1 && matches[0].first - matches[1].first < 0.05) continue;
        records[i].label = records[matches[0].second].label;
    }
    std::vector<RawSegment> out;
    for (auto& r : records) out.push_back({r.start, r.end, r.label});
    return out;
}

// ── sections.normalize_sections ──────────────────────────────────────

std::string normalize_sections(const std::vector<RawSegment>& raw, double duration, int* count) {
    static const std::map<std::string, std::pair<const char*, const char*>> kinds = {
        {"intro", {"Intro", "#4a7fff"}},   {"verse", {"Verse", "#00c8a0"}},
        {"chorus", {"Chorus", "#9a4aff"}}, {"bridge", {"Bridge", "#ff8a20"}},
        {"break", {"Break", "#2ab8e8"}},   {"inst", {"Instrumental", "#e8c840"}},
        {"solo", {"Solo", "#ff4a90"}},     {"outro", {"Outro", "#00d4d4"}},
        {"part", {"Part", "#8391a5"}}};
    const double kMin = 0.5;
    if (count) *count = 0;
    if (!std::isfinite(duration) || duration < 2 * kMin || raw.empty()) return "[]";
    struct P { double start, end; std::string kind; };
    std::vector<P> parsed;
    for (auto& s : raw) {
        std::string k;
        size_t b = s.label.find_first_not_of(" \t\n\r\f\v"), e = s.label.find_last_not_of(" \t\n\r\f\v");
        k = b == std::string::npos ? "" : s.label.substr(b, e - b + 1);
        for (auto& ch : k) ch = (char)std::tolower((unsigned char)ch);
        if (!kinds.count(k) && !is_sentinel(k)) return "[]";
        if (!std::isfinite(s.start) || !std::isfinite(s.end) || s.end <= s.start) return "[]";
        double start = std::max(0.0, std::min(duration, s.start));
        double end = std::max(0.0, std::min(duration, s.end));
        if (end <= start) continue;
        parsed.push_back({start, end, k});
    }
    std::stable_sort(parsed.begin(), parsed.end(), [](const P& x, const P& y) {
        return x.start != y.start ? x.start < y.start : x.end < y.end;
    });
    size_t lo = 0, hi = parsed.size();
    while (lo < hi && is_sentinel(parsed[lo].kind)) lo++;
    while (hi > lo && is_sentinel(parsed[hi - 1].kind)) hi--;
    for (size_t i = lo; i < hi; i++)
        if (is_sentinel(parsed[i].kind)) parsed[i].kind = "part";
    for (size_t i = 0; i + 1 < parsed.size(); i++) {
        double delta = parsed[i + 1].start - parsed[i].end;
        if (std::fabs(delta) > 0.25) return "[]";
        double boundary = (parsed[i].end + parsed[i + 1].start) / 2;
        parsed[i].end = boundary;
        parsed[i + 1].start = boundary;
    }
    std::vector<P> m;
    for (size_t i = lo; i < hi; i++)
        if (kinds.count(parsed[i].kind)) m.push_back(parsed[i]);
    if (m.size() < 2) return "[]";
    m.front().start = 0.0;
    m.back().end = duration;
    // _merge_short
    while (true) {
        size_t si = m.size();
        for (size_t i = 0; i < m.size(); i++)
            if (m[i].end - m[i].start < kMin) { si = i; break; }
        if (si == m.size()) break;
        if (m.size() <= 2) { m.clear(); break; }
        size_t i = si;
        if (i > 0 && i < m.size() - 1 && m[i - 1].kind == m[i + 1].kind) {
            m[i - 1].end = m[i + 1].end;
            m.erase(m.begin() + i, m.begin() + i + 2);
        } else if (i > 0) {
            m[i - 1].end = m[i].end;
            m.erase(m.begin() + i);
        } else {
            m[i + 1].start = m[i].start;
            m.erase(m.begin() + i);
        }
    }
    if (m.size() < 2) return "[]";
    std::string json = "[";
    for (size_t i = 0; i < m.size(); i++) {
        double start = py_round(m[i].start, 3), end = py_round(m[i].end, 3);
        if (end - start < kMin) return "[]";
        const auto& kc = kinds.at(m[i].kind);
        char id[32];
        std::snprintf(id, sizeof id, "auto-%03zu", i + 1);
        if (i) json += ", ";
        json += std::string("{\"id\": \"") + id + "\", \"name\": \"" + kc.first + "\", \"kind\": \"" + m[i].kind +
                "\", \"start\": " + json_number(start) + ", \"end\": " + json_number(end) + ", \"color\": \"" +
                kc.second + "\"}";
    }
    json += "]";
    if (count) *count = (int)m.size();
    return json;
}

// ── the stage ────────────────────────────────────────────────────────

std::vector<float> section_features(const std::string& dir, size_t* frames_out,
                                    const std::function<bool(double)>& progress) {
    const char* order[4] = {"bass", "drums", "other", "vocals"};
    std::vector<std::vector<float>> specs(4);
    size_t T = 0;
    bool int16_window = true;
    for (int k = 0; k < 4; k++) {
        MadmomSignal s = k == 2 ? madmom_load_mix({dir + "/other.wav", dir + "/guitar.wav", dir + "/piano.wav"})
                                : madmom_load_stem(dir + "/" + order[k] + ".wav");
        if (k == 0) int16_window = s.int16;  // the processor's cached window comes from bass
        size_t frames = 0;
        specs[k] = madmom_logspec(s, int16_window, &frames);
        if (k == 0) T = frames;
        else if (frames != T) throw std::runtime_error("the section stems differ in length");
        if (progress) progress((k + 1) / 4.0);
    }
    std::vector<float> features(4 * T * kSecBins);
    for (int k = 0; k < 4; k++) {
        std::copy(specs[k].begin(), specs[k].end(), features.begin() + (size_t)k * T * kSecBins);
        std::vector<float>().swap(specs[k]);
    }
    *frames_out = T;
    return features;
}

SectionsResult run_sections(const std::string& dir, const std::vector<std::string>& fold_models,
                            const std::vector<double>& beats, double confidence, double duration, int threads,
                            const std::function<bool(double)>& progress, const std::string& provider) {
    SectionsResult res;
    auto step = [&](double p) {
        if (progress && !progress(p)) throw Cancelled{};
    };
    if (fold_models.size() != kSecFolds) throw std::runtime_error("the section model needs its eight folds");
    double t0 = now();
    size_t T = 0;
    std::vector<float> features = section_features(dir, &T, [&](double p) { step(0.16 * p); return true; });
    if (T < 2) {
        res.why = "the song is too short for sections";
        return res;
    }
    res.t_spec = now() - t0;
    t0 = now();
    SectionActivations act;
    act.frames = T;
    std::vector<double> sec(T, 0.0), fun(10 * T, 0.0);
    act.embedding.assign(T * 96, 0.0);
    for (int f = 0; f < kSecFolds; f++) {
        OrtModel m(fold_models[f], threads, provider);
        auto outs = m.run(features.data(), {1, 4, (int64_t)T, kSecBins});
        const auto& names = m.output_names();
        for (size_t i = 0; i < outs.size(); i++) {
            const auto& o = outs[i];
            if (names[i] == "section") {
                if (o.data.size() != T) throw std::runtime_error("section output has the wrong size");
                for (size_t t = 0; t < T; t++) sec[t] += o.data[t];
            } else if (names[i] == "function") {
                if (o.data.size() != 10 * T) throw std::runtime_error("function output has the wrong size");
                for (size_t j = 0; j < 10 * T; j++) fun[j] += o.data[j];
            } else if (names[i] == "embedding") {
                if (o.data.size() != 4 * T * 24) throw std::runtime_error("embedding output has the wrong size");
                // [1, 4, T, 24] -> [T][stem * 24 + d]
                for (size_t s = 0; s < 4; s++)
                    for (size_t t = 0; t < T; t++)
                        for (size_t d = 0; d < 24; d++)
                            act.embedding[t * 96 + s * 24 + d] += o.data[(s * T + t) * 24 + d];
            }
        }
        step(0.16 + 0.8 * (f + 1) / kSecFolds);
    }
    std::vector<float>().swap(features);
    act.section_logits.resize(T);
    act.function_logits.resize(10 * T);
    for (size_t t = 0; t < T; t++) act.section_logits[t] = (float)(sec[t] / kSecFolds);
    for (size_t j = 0; j < 10 * T; j++) act.function_logits[j] = (float)(fun[j] / kSecFolds);
    for (auto& v : act.embedding) v = v / kSecFolds;  // refine averages the stacked float32 folds in float64
    res.t_model = now() - t0;
    t0 = now();
    res.raw = functional_segments(act);
    res.refined = refine_segments(res.raw, act, beats, confidence);
    res.sections_json = normalize_sections(res.refined, duration, &res.count);
    res.t_post = now() - t0;
    res.ok = res.count >= 2;
    if (!res.ok) res.why = "no clear structure was found";
    step(1.0);
    return res;
}

}  // namespace sda
