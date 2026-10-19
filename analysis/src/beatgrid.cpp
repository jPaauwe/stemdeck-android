#include "beatgrid.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "librosa.hpp"
#include "numpy.hpp"

namespace sda {
namespace {
// app/core/config.py
constexpr int BEATGRID_HOP = 512;
constexpr double BEATGRID_GAP_FILL_MAX_RESIDUAL = 0.15;
constexpr int BEATGRID_GAP_LOCAL_WINDOW = 8;
constexpr double BEATGRID_MAX_IRREGULARITY = 0.20;
constexpr double BEATGRID_PHASE_FLIP_RATIO = 2.0;
constexpr int BEATGRID_REFINE_HOP = 32;
constexpr int BEATGRID_REFINE_NFFT = 512;
constexpr int BEATGRID_REFINE_MELS = 64;
constexpr double BEATGRID_REFINE_WINDOW = 0.030;
constexpr double BEATGRID_REFINE_WINDOW_OF_BEAT = 0.15;
constexpr double BEATGRID_REFINE_WINDOW_MAX = 0.100;
constexpr double BEATGRID_REFINE_MOVE_OF_BEAT = 0.10;
constexpr double BEATGRID_REFINE_MOVE_MIN = 0.029;
constexpr double BEATGRID_REFINE_MOVE_MAX = 0.070;
constexpr double BEATGRID_ONSET_PERCENTILE = 90;
constexpr double BEATGRID_ONSET_DELTA_MULT = 1.5;
constexpr size_t BEATGRID_MAX_ONSETS = 6000;
constexpr double BEATGRID_REFINE_MAX_INTERP_FRAC = 0.25;
constexpr double BEATGRID_REFINE_PERCENTILE = 70;
constexpr double BEATGRID_ONSET_TOL_FRAC = 0.15;
constexpr double BEATGRID_ONSET_TOL_MAX = 0.07;
// beatgrid.py
constexpr size_t MIN_BEATS = 8;
constexpr double MIN_BPM = 30.0, MAX_BPM = 300.0;
constexpr double GRID_OUTLIER_FRAC = 0.10;
constexpr size_t EXTRAPOLATE_EDGE_BEATS = 9;
constexpr long MIN_BAR = 2, MAX_BAR = 16;
constexpr int BAR_CHANGE_PERSISTENCE = 2;

using Vec = std::vector<double>;

Vec diff(const Vec& a) {
    Vec d;
    for (size_t i = 1; i < a.size(); i++) d.push_back(a[i] - a[i - 1]);
    return d;
}

Vec sanitize(const Vec& t, double duration) {
    Vec a;
    for (double v : t)
        if (std::isfinite(v) && v >= 0.0 && v <= duration) a.push_back(v);
    std::sort(a.begin(), a.end());
    Vec out;
    for (size_t i = 0; i < a.size(); i++)
        if (i == 0 || a[i] - a[i - 1] > 1e-3) out.push_back(py_round(a[i], 6));
    return out;
}

std::vector<double> fine_onsets(const std::vector<float>& env_f, double sr, size_t limit) {
    std::vector<double> arr(env_f.begin(), env_f.end());
    if (arr.size() < 3) return {};
    const double frame_dur = (double)BEATGRID_REFINE_HOP / sr;
    auto peaks = peak_pick(arr, (int)py_round_int(0.015 / frame_dur), (int)py_round_int(0.015 / frame_dur),
                           (int)py_round_int(0.045 / frame_dur), (int)py_round_int(0.060 / frame_dur),
                           percentile(arr, BEATGRID_ONSET_PERCENTILE) * BEATGRID_ONSET_DELTA_MULT,
                           (int)py_round_int(0.030 / frame_dur));
    if (peaks.empty()) return {};
    if (peaks.size() > limit) {
        std::vector<size_t> order(peaks.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return arr[peaks[a]] < arr[peaks[b]]; });
        std::vector<int> keep;
        for (size_t i = order.size() - limit; i < order.size(); i++) keep.push_back(peaks[order[i]]);
        std::sort(keep.begin(), keep.end());
        peaks = keep;
    }
    std::vector<double> times;
    for (int p : peaks) {
        double delta = 0.0;
        if (0 < p && p < (long)arr.size() - 1) {
            double a = arr[p - 1], b = arr[p], c = arr[p + 1];
            double denom = a - 2.0 * b + c;
            if (denom != 0.0) delta = std::max(-0.5, std::min(0.5, 0.5 * (a - c) / denom));
        }
        times.push_back(py_round(((double)p + delta) * frame_dur, 4));
    }
    return times;
}

// numpy.interp
double interp(double x, const Vec& xp, const Vec& fp) {
    const size_t n = xp.size();
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    if (x == xp[n - 1]) return fp[n - 1];
    size_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin() - 1;
    double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) r = slope * (x - xp[j + 1]) + fp[j + 1];
    return r;
}

std::pair<Vec, long> refine_beats(const Vec& coarse, double beat_interval, const std::vector<float>& env, double sr) {
    if (coarse.empty()) return {coarse, 0};
    if (env.size() < 3) return {coarse, 0};
    const double frame_dur = (double)BEATGRID_REFINE_HOP / sr;
    double window_sec = std::min(BEATGRID_REFINE_WINDOW_MAX,
                                 std::max(BEATGRID_REFINE_WINDOW, beat_interval * BEATGRID_REFINE_WINDOW_OF_BEAT));
    long window_frames = std::max(1L, py_round_int(window_sec / frame_dur));
    double threshold = percentile(env, BEATGRID_REFINE_PERCENTILE);
    double max_move = std::min(BEATGRID_REFINE_MOVE_MAX,
                               std::max(BEATGRID_REFINE_MOVE_MIN, beat_interval * BEATGRID_REFINE_MOVE_OF_BEAT));
    Vec ridx, rtime;
    const long n = (long)env.size();
    for (size_t i = 0; i < coarse.size(); i++) {
        double beat = coarse[i];
        long center = py_round_int(beat / frame_dur);
        long lo = std::max(0L, center - window_frames);
        long hi = std::min(n - 1, center + window_frames);
        if (hi <= lo) continue;
        long peak = lo;
        for (long k = lo + 1; k <= hi; k++)
            if (env[k] > env[peak]) peak = k;
        if ((double)env[peak] <= threshold) continue;
        double delta = 0.0;
        if (0 < peak && peak < n - 1) {
            double a = env[peak - 1], b = env[peak], c = env[peak + 1];
            double denom = a - 2.0 * b + c;
            if (denom != 0.0) {
                delta = 0.5 * (a - c) / denom;
                delta = std::max(-0.5, std::min(0.5, delta));
            }
        }
        double snapped = ((double)peak + delta) * frame_dur;
        if (std::fabs(snapped - beat) > max_move) continue;
        ridx.push_back((double)i);
        rtime.push_back(snapped);
    }
    if (ridx.size() < 2) return {coarse, (long)ridx.size()};
    Vec out(coarse.size());
    for (size_t i = 0; i < coarse.size(); i++) out[i] = interp((double)i, ridx, rtime);
    double head_slope = (rtime[1] - rtime[0]) / std::max(1e-9, ridx[1] - ridx[0]);
    size_t m = ridx.size();
    double tail_slope = (rtime[m - 1] - rtime[m - 2]) / std::max(1e-9, ridx[m - 1] - ridx[m - 2]);
    for (size_t i = 0; i < coarse.size(); i++) {
        double x = (double)i;
        if (x < ridx[0]) out[i] = rtime[0] + (x - ridx[0]) * head_slope;
        if (x > ridx[m - 1]) out[i] = rtime[m - 1] + (x - ridx[m - 1]) * tail_slope;
    }
    double limit = beat_interval * BEATGRID_REFINE_MAX_INTERP_FRAC;
    for (size_t i = 0; i < coarse.size(); i++) out[i] = std::min(std::max(out[i], coarse[i] - limit), coarse[i] + limit);
    return {out, (long)ridx.size()};
}

std::pair<Vec, long> enforce_grid_consistency(const Vec& beats) {
    if (beats.size() < 3) return {beats, 0};
    const Vec& arr = beats;
    double med = median(diff(arr));
    if (med <= 0) return {beats, 0};
    double tol = med * GRID_OUTLIER_FRAC;
    Vec out = arr;
    long corrected = 0;
    for (size_t i = 1; i + 1 < arr.size(); i++) {
        double pred = (arr[i - 1] + arr[i + 1]) / 2.0;
        if (std::fabs(arr[i] - pred) > tol) { out[i] = pred; corrected++; }
    }
    double first_pred = out[1] - (out[2] - out[1]);
    if (std::fabs(out[0] - first_pred) > tol) { out[0] = first_pred; corrected++; }
    size_t L = out.size();
    double last_pred = out[L - 2] + (out[L - 2] - out[L - 3]);
    if (std::fabs(out[L - 1] - last_pred) > tol) { out[L - 1] = last_pred; corrected++; }
    return {out, corrected};
}

double interval_spread(const Vec& beats) {
    if (beats.size() < 5) return 0.0;
    Vec iv = diff(beats);
    double med = median(iv);
    if (med <= 0) return INFINITY;
    return (percentile(iv, 75) - percentile(iv, 25)) / med;
}

double onset_support(const std::vector<float>& env_f, double sr, const Vec& t) {
    if (env_f.size() < 3 || t.empty()) return 0.0;
    const double frame_dur = (double)BEATGRID_REFINE_HOP / sr;
    const long n = (long)env_f.size();
    long span = std::max(1L, py_round_int(0.010 / frame_dur));
    Vec maxes;
    for (double v : t) {
        long c = (long)std::nearbyint(v / frame_dur);
        long lo = std::min(std::max(c - span, 0L), n - 1), hi = std::min(std::max(c + span, 0L), n - 1);
        double m = -INFINITY;
        for (long k = lo; k <= hi; k++) m = std::max(m, (double)env_f[k]);
        if (lo > hi) m = NAN;  // empty slice: numpy would raise; cannot happen after clipping
        maxes.push_back(m);
    }
    return np_mean(maxes);
}

std::pair<Vec, bool> correct_grid_phase(const Vec& beats, const std::vector<float>& env, double sr, double duration) {
    if (beats.size() < 8) return {beats, false};
    double half = median(diff(beats)) / 2.0;
    if (half <= 0) return {beats, false};
    double here = onset_support(env, sr, beats);
    Vec shifted;
    for (double b : beats) if (b + half <= duration) shifted.push_back(b + half);
    if (shifted.size() < 8) return {beats, false};
    double there = onset_support(env, sr, shifted);
    if (there > here * BEATGRID_PHASE_FLIP_RATIO) return {shifted, true};
    return {beats, false};
}

std::pair<Vec, long> fill_interior_gaps(const Vec& beats) {
    if (beats.size() < 4) return {beats, 0};
    Vec iv = diff(beats);
    const long win = BEATGRID_GAP_LOCAL_WINDOW, n = (long)iv.size();
    Vec out{beats[0]};
    long inserted = 0;
    double global_med = median(iv);
    for (long i = 0; i < n; i++) {
        double d = iv[i];
        long lo = std::max(0L, i - win), hi = std::min(n, i + win + 1);
        Vec nb;
        for (long k = lo; k < i; k++) nb.push_back(iv[k]);
        for (long k = i + 1; k < hi; k++) nb.push_back(iv[k]);
        double period = nb.empty() ? global_med : median(nb);
        if (period > 0) {
            long k = py_round_int(d / period);
            if (k >= 2 && std::fabs(d / k - period) < BEATGRID_GAP_FILL_MAX_RESIDUAL * period) {
                for (long j = 1; j < k; j++) {
                    out.push_back(beats[i] + d * j / k);
                    inserted++;
                }
            }
        }
        out.push_back(beats[i + 1]);
    }
    return {out, inserted};
}

std::vector<BeatBar> downbeats_to_bars(const Vec& beats, const Vec& downbeats) {
    if (downbeats.empty() || beats.size() < 2) return {};
    std::vector<long> idx;
    for (double t : downbeats) {
        size_t best = 0;
        double bd = std::fabs(beats[0] - t);
        for (size_t j = 1; j < beats.size(); j++) {
            double e = std::fabs(beats[j] - t);
            if (e < bd) { bd = e; best = j; }
        }
        idx.push_back((long)best);
    }
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    if (idx.size() < 3) return {};
    std::vector<std::pair<long, long>> spans;
    for (size_t i = 0; i + 1 < idx.size(); i++) {
        long len = idx[i + 1] - idx[i];
        if (len >= MIN_BAR && len <= MAX_BAR) spans.push_back({idx[i], len});
    }
    if (spans.empty()) return {};
    std::map<long, long> counts;
    for (auto& s : spans) counts[s.second]++;
    long modal = 0, best = -1;
    for (auto& [len, c] : counts)
        if (c > best) { best = c; modal = len; }  // ascending keys: ties go to the smaller length, like bincount.argmax
    std::vector<BeatBar> bars{{spans[0].first, modal}};
    long current = modal, run_at = 0, run_len = 0, run_count = 0;
    for (auto& [at, len] : spans) {
        if (len == current) { run_count = 0; continue; }
        if (run_count && len == run_len) run_count += 1;
        else { run_at = at; run_len = len; run_count = 1; }
        if (run_count >= BAR_CHANGE_PERSISTENCE) {
            bars.push_back({run_at, len});
            current = len;
            run_count = 0;
        }
    }
    return bars;
}

struct Extended { Vec beats; long head = 0, tail = 0; };
Extended extend_to_track_edges(const Vec& beats, double duration, double irregularity) {
    if (irregularity > BEATGRID_MAX_IRREGULARITY || beats.size() < 3) return {beats, 0, 0};
    size_t edge = std::min(EXTRAPOLATE_EDGE_BEATS, beats.size());
    double head_iv = median(diff(Vec(beats.begin(), beats.begin() + edge)));
    double tail_iv = median(diff(Vec(beats.end() - edge, beats.end())));
    if (head_iv <= 0 || tail_iv <= 0) return {beats, 0, 0};
    Vec head;
    for (double t = beats[0] - head_iv; t >= 0.0; t -= head_iv) head.push_back(t);
    std::reverse(head.begin(), head.end());
    Vec tail;
    for (double t = beats.back() + tail_iv; t <= duration; t += tail_iv) tail.push_back(t);
    Extended e;
    e.beats = head;
    e.beats.insert(e.beats.end(), beats.begin(), beats.end());
    e.beats.insert(e.beats.end(), tail.begin(), tail.end());
    e.head = (long)head.size();
    e.tail = (long)tail.size();
    return e;
}

long grid_confidence(const Vec& beats, Vec onsets, double tol) {
    Vec on;
    for (double v : onsets) if (std::isfinite(v)) on.push_back(v);
    if (on.empty() || beats.empty()) return 0;
    std::sort(on.begin(), on.end());
    long hits = 0;
    const long m = (long)on.size();
    for (double b : beats) {
        long idx = std::lower_bound(on.begin(), on.end(), b) - on.begin();
        long left = std::min(std::max(idx - 1, 0L), m - 1), right = std::min(std::max(idx, 0L), m - 1);
        double nearest = std::min(std::fabs(b - on[left]), std::fabs(b - on[right]));
        if (nearest <= tol) hits++;
    }
    double mean = (double)hits / (double)beats.size();
    return py_round_int(mean * 100);
}
}  // namespace

BeatGrid compute_beat_grid(const std::vector<float>& y, double sr, const std::vector<double>& beat_times,
                           const std::vector<double>& downbeat_times, const std::string& detector,
                           const std::string& source) {
    BeatGrid g;
    g.source = source;
    g.detector = detector;
    const double duration = (double)y.size() / sr;
    if (duration <= 0) { g.why = "no audio"; return g; }
    auto onset_env = onset_strength_median(y, sr, 2048, BEATGRID_HOP, 128);
    bool any = false;
    for (float v : onset_env) if (v != 0.0f) { any = true; break; }
    if (!any) { g.why = "flat onset envelope (silent stem?)"; return g; }
    Vec beats = sanitize(beat_times, duration);
    if (beats.size() < MIN_BEATS) { g.why = "only " + std::to_string(beats.size()) + " beats detected"; return g; }
    double coarse_interval = median(diff(beats));
    if (coarse_interval <= 0) { g.why = "no beat interval"; return g; }
    auto fine_env = onset_strength_median(y, sr, BEATGRID_REFINE_NFFT, BEATGRID_REFINE_HOP, BEATGRID_REFINE_MELS);
    auto [phased, shifted] = correct_grid_phase(beats, fine_env, sr, duration);
    beats = phased;
    auto [refined, refined_count] = refine_beats(beats, coarse_interval, fine_env, sr);
    beats = refined;
    long detected_count = (long)beats.size();
    double irregularity = interval_spread(beats);
    bool regular = irregularity <= BEATGRID_MAX_IRREGULARITY;
    long gap_filled = 0, corrected_count = 0;
    if (regular) {
        auto f = fill_interior_gaps(beats);
        beats = sanitize(f.first, duration);
        gap_filled = f.second;
        auto c = enforce_grid_consistency(beats);
        beats = sanitize(c.first, duration);
        corrected_count = c.second;
    }
    if (beats.size() < MIN_BEATS) { g.why = "post-processing left only " + std::to_string(beats.size()) + " beats"; return g; }
    Vec intervals = diff(beats);
    double median_interval = median(intervals);
    if (median_interval <= 0) { g.why = "no beat interval"; return g; }
    double bpm = 60.0 / median_interval;
    if (!(MIN_BPM <= bpm && bpm <= MAX_BPM)) { g.why = "implausible tempo"; return g; }
    double tol = std::min(BEATGRID_ONSET_TOL_MAX, median_interval * BEATGRID_ONSET_TOL_FRAC);
    auto onsets = onset_detect_times(onset_env, sr, BEATGRID_HOP);
    long confidence = grid_confidence(beats, onsets, tol);
    double cv = np_std(intervals) / median_interval;
    // Android: the job's tempo stability, from this grid rather than from
    // librosa's tracker on the mix (analyze.py's formula: std over mean).
    double mean_iv = np_mean(intervals);
    if (intervals.size() >= 2 && mean_iv > 0)
        g.tempo_stability = std::max(0L, std::min(100L, py_round_int((1 - std::min(np_std(intervals) / mean_iv, 1.0)) * 100)));
    auto ext = extend_to_track_edges(beats, duration, irregularity);
    beats = sanitize(ext.beats, duration);
    if (beats.size() < MIN_BEATS) { g.why = "too few beats"; return g; }
    double final_median = median(diff(beats));
    bpm = final_median > 0 ? 60.0 / final_median : bpm;
    g.bars = downbeats_to_bars(beats, downbeat_times);
    g.ok = true;
    g.bpm = py_round(bpm, 3);
    g.duration = py_round(duration, 3);
    g.confidence = confidence;
    g.interval_cv = py_round(cv, 4);
    g.irregularity = py_round(irregularity, 4);
    g.refined = refined_count;
    g.corrected = corrected_count;
    g.onsets = fine_onsets(fine_env, sr, BEATGRID_MAX_ONSETS);
    g.detected = detected_count;
    g.extrapolated_head = ext.head;
    g.extrapolated_tail = ext.tail;
    g.gap_filled = gap_filled;
    g.phase_shifted = shifted;
    for (double b : beats) g.beats.push_back(py_round(b, 6));
    return g;
}

std::string BeatGrid::to_json() const {
    auto list = [](const std::vector<double>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); i++) { if (i) s += ", "; s += json_number(v[i]); }
        return s + "]";
    };
    auto esc = [](const std::string& s) {
        std::string o = "\"";
        for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
        return o + "\"";
    };
    std::string b = "[";
    for (size_t i = 0; i < bars.size(); i++) {
        if (i) b += ", ";
        b += "{\"beat\": " + std::to_string(bars[i].beat) + ", \"beats_per_bar\": " + std::to_string(bars[i].beats_per_bar) + "}";
    }
    b += "]";
    std::string s = "{";
    s += "\"version\": 1, \"source\": " + esc(source) + ", \"detector\": " + esc(detector);
    s += ", \"bars\": " + b;
    s += ", \"bpm\": " + json_number(bpm) + ", \"duration\": " + json_number(duration);
    s += ", \"confidence\": " + std::to_string(confidence);
    s += ", \"interval_cv\": " + json_number(interval_cv) + ", \"irregularity\": " + json_number(irregularity);
    s += ", \"refined\": " + std::to_string(refined) + ", \"corrected\": " + std::to_string(corrected);
    s += ", \"onsets\": " + list(onsets);
    s += ", \"detected\": " + std::to_string(detected);
    s += ", \"extrapolated_head\": " + std::to_string(extrapolated_head);
    s += ", \"extrapolated_tail\": " + std::to_string(extrapolated_tail);
    s += ", \"gap_filled\": " + std::to_string(gap_filled);
    s += ", \"phase_shifted\": " + std::string(phase_shifted ? "true" : "false");
    s += ", \"beats\": " + list(beats) + "}";
    return s;
}

}  // namespace sda
