#include "key.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

#include "cqt.hpp"
#include "librosa.hpp"
#include "numpy.hpp"
#include "pipeline.hpp"

namespace sda {
namespace {
const double MAJOR[12] = {5.47, 0.14, 2.55, 0.14, 3.15, 2.16, 0.37, 4.92, 0.21, 1.84, 0.18, 1.86};
const double MINOR[12] = {5.06, 0.14, 2.42, 2.42, 0.35, 1.96, 0.35, 4.16, 2.53, 0.28, 2.67, 0.62};
const char* PITCHES[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
constexpr double MINOR_TIE_BREAK_FRAC = 0.05, EDGE_WEIGHT = 0.4, FAMILY_WINDOW = 0.1, RELATIVE_MARGIN = 0.15;
constexpr double HARMONIC_RATIO = 1.3, HARMONIC_FLOOR = 0.1, CONFIDENCE_FULL_GAP = 0.25, EDGE_CONFIDENT_MARGIN = 0.3;
constexpr double EDGE_SECONDS = 4.0;
constexpr double STEM_KEY_MAX_SECONDS = 600.0;

struct Cand { double corr; int root; bool major; };

double correlate(const double* profile, const std::vector<double>& chroma, int shift) {
    const int n = 12;
    double rot[12];
    for (int i = 0; i < n; i++) rot[i] = chroma[(i + shift) % n];
    double sp = 0, sc = 0;
    for (int i = 0; i < n; i++) sp += profile[i];
    for (int i = 0; i < n; i++) sc += rot[i];
    double mp = sp / n, mc = sc / n;
    double num = 0, dp = 0, dc = 0;
    for (int i = 0; i < n; i++) num += (profile[i] - mp) * (rot[i] - mc);
    for (int i = 0; i < n; i++) dp += (profile[i] - mp) * (profile[i] - mp);
    for (int i = 0; i < n; i++) dc += (rot[i] - mc) * (rot[i] - mc);
    dp = std::pow(dp, 0.5);
    dc = std::pow(dc, 0.5);
    if (dp == 0 || dc == 0) return 0.0;
    return num / (dp * dc);
}

std::vector<Cand> key_candidates(const std::vector<double>& chroma) {
    std::vector<Cand> out;
    for (int s = 0; s < 12; s++) {
        out.push_back({correlate(MAJOR, chroma, s), s, true});
        out.push_back({correlate(MINOR, chroma, s), s, false});
    }
    std::stable_sort(out.begin(), out.end(), [](const Cand& a, const Cand& b) { return a.corr > b.corr; });
    return out;
}

std::pair<int, bool> relative(int root, bool major) {
    return major ? std::make_pair((root + 9) % 12, false) : std::make_pair((root + 3) % 12, true);
}

double tonic_fit(int root, bool major, const std::vector<Edge>& edges) {
    std::set<int> triad{root, (root + (major ? 4 : 3)) % 12, (root + 7) % 12};
    double score = 0.0;
    for (auto& [harmony, bass] : edges) {
        double inside = 0;
        for (int i : triad) inside += harmony[i];
        inside /= 3;
        double outside = 0;
        for (int i = 0; i < 12; i++) if (!triad.count(i)) outside += harmony[i];
        outside /= 9;
        score += inside - outside + bass[root];
    }
    return score / edges.size();
}
}  // namespace

KeyResult detect_key(const std::vector<double>& chroma_mean, const std::vector<Edge>& edges) {
    auto cands = key_candidates(chroma_mean);
    const Cand best = cands[0];
    const auto rel_key = relative(best.root, best.major);
    const int rr = rel_key.first;
    const bool rm = rel_key.second;
    Cand rel = *std::find_if(cands.begin(), cands.end(), [&](const Cand& c) { return c.root == rr && c.major == rm; });
    Cand winner = best;
    double clarity = 1.0;
    if (!edges.empty()) {
        auto find = [&](int r, bool m) {
            return *std::find_if(cands.begin(), cands.end(), [&](const Cand& c) { return c.root == r && c.major == m; });
        };
        std::set<std::pair<int, bool>> pool;
        for (auto& c : cands)
            if (best.corr - c.corr <= FAMILY_WINDOW) {
                pool.insert({c.root, c.major});
                pool.insert(relative(c.root, c.major));
            }
        std::vector<std::pair<double, Cand>> scored;
        for (auto& k : pool) {
            Cand c = find(k.first, k.second);
            scored.push_back({c.corr + EDGE_WEIGHT * tonic_fit(k.first, k.second, edges), c});
        }
        std::stable_sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
        winner = scored[0].second;
        double runner = scored.size() > 1 ? scored[1].first : scored[0].first - EDGE_CONFIDENT_MARGIN;
        clarity = std::min(1.0, (scored[0].first - runner) / EDGE_CONFIDENT_MARGIN);
    } else if (best.corr - rel.corr < RELATIVE_MARGIN) {
        double threshold = std::max(std::fabs(best.corr), std::fabs(rel.corr)) * MINOR_TIE_BREAK_FRAC;
        if (best.corr - rel.corr <= threshold && !rel.major) winner = rel;
        clarity = 0.5;
    }
    KeyResult r;
    r.ok = true;
    r.key = std::string(PITCHES[winner.root]) + (winner.major ? " maj" : " min");
    r.scale = "Major";
    if (!winner.major) {
        double raised = chroma_mean[(winner.root + 11) % 12];
        double flat = chroma_mean[(winner.root + 10) % 12];
        double peak = *std::max_element(chroma_mean.begin(), chroma_mean.end());
        if (peak == 0) peak = 1.0;
        bool harmonic = raised > HARMONIC_RATIO * flat && raised > HARMONIC_FLOOR * peak;
        r.scale = harmonic ? "Harmonic Minor" : "Natural Minor";
    }
    if (!edges.empty()) {
        r.confidence = py_round_int(clarity * 100);
        return r;
    }
    const Cand* other = nullptr;
    for (auto& c : cands) {
        bool is_best = c.root == best.root && c.major == best.major;
        bool is_rel = c.root == rel.root && c.major == rel.major;
        if (!is_best && !is_rel) { other = &c; break; }
    }
    double family_gap = best.corr - other->corr;
    double confidence = std::min(1.0, std::max(0.0, family_gap / CONFIDENCE_FULL_GAP)) * clarity;
    r.confidence = py_round_int(confidence * 100);
    return r;
}

KeyResult detect_key_from_audio(const std::vector<float>& harmony, const std::vector<float>& bass, bool whole_song) {
    const double sr = 22050;
    const int hop = 2048;
    Spec chroma = chroma_cqt(harmony, sr, hop, kC1, 7);
    Spec bass_chroma = chroma_cqt(bass.empty() ? harmony : bass, sr, hop, kC1, 3);
    std::vector<float> lvl_src = harmony;
    if (!bass.empty())
        for (size_t i = 0; i < lvl_src.size() && i < bass.size(); i++) lvl_src[i] = harmony[i] + bass[i];
    auto level = rms(lvl_src, 2048, hop);
    const size_t frames = std::min({chroma.frames, bass_chroma.frames, level.size()});
    if (frames == 0) return {};
    level.resize(frames);
    float lmax = *std::max_element(level.begin(), level.end());
    float gate = 0.1f * (lmax != 0.0f ? lmax : 1.0f);
    std::vector<size_t> active;
    for (size_t t = 0; t < frames; t++) if (level[t] > gate) active.push_back(t);
    const size_t span = std::max<size_t>(1, (size_t)(EDGE_SECONDS * sr / hop));
    auto col_mean = [&](const Spec& s, size_t a, size_t b, int c) {
        b = std::min(b, frames);
        std::vector<float> v;
        for (size_t t = a; t < b; t++) v.push_back(s.row(t)[c]);
        return v.empty() ? NAN : np_sum(v) / (float)v.size();
    };
    auto region = [&](size_t a, size_t b) {
        std::vector<float> h(12), lo(12);
        for (int c = 0; c < 12; c++) { h[c] = col_mean(chroma, a, b, c); lo[c] = col_mean(bass_chroma, a, b, c); }
        float hm = *std::max_element(h.begin(), h.end()), lm = *std::max_element(lo.begin(), lo.end());
        if (hm == 0) hm = 1.0f;
        if (lm == 0) lm = 1.0f;
        Edge e;
        for (int c = 0; c < 12; c++) { e.first.push_back((double)(h[c] / hm)); e.second.push_back((double)(lo[c] / lm)); }
        return e;
    };
    std::vector<Edge> edges;
    if (!active.empty()) {
        size_t start = active.front();
        edges.push_back(region(start, start + span));
        if (whole_song) {
            size_t end = active.back() + 1;
            edges.push_back(region(end > span ? end - span : 0, end));
        }
    }
    std::vector<double> chroma_mean(12);
    bool any = false;
    for (int c = 0; c < 12; c++) {
        chroma_mean[c] = (double)col_mean(chroma, 0, frames, c);
        if (chroma_mean[c] != 0) any = true;
    }
    if (!any) return {};
    return detect_key(chroma_mean, edges);
}

KeyResult key_from_stems(const std::vector<std::pair<std::string, double>>& harmony_paths, const std::string& bass_path) {
    const double sr = 22050;
    const size_t cap = (size_t)(STEM_KEY_MAX_SECONDS * 44100);  // ffmpeg -t 600 on the 44.1 kHz input
    auto load = [&](const std::string& p) { return load_mono_22k(p, cap); };
    std::vector<float> harmony;
    size_t longest = 0;
    for (auto& [path, weight] : harmony_paths) {
        if (path.empty()) continue;
        std::vector<float> y = load(path);
        for (auto& v : y) v = v * (float)weight;
        longest = std::max(longest, y.size());
        if (harmony.empty()) harmony = std::move(y);
        else {
            if (y.size() > harmony.size()) harmony.resize(y.size(), 0.0f);
            for (size_t i = 0; i < y.size(); i++) harmony[i] = harmony[i] + y[i];
        }
    }
    if (harmony.empty()) return {};
    std::vector<float> bass;
    if (!bass_path.empty()) {
        bass = load(bass_path);
        size_t n = std::max(bass.size(), harmony.size());
        harmony.resize(n, 0.0f);
        bass.resize(n, 0.0f);
    }
    bool whole = (double)longest < (STEM_KEY_MAX_SECONDS - 1) * sr;
    return detect_key_from_audio(harmony, bass, whole);
}

}  // namespace sda
