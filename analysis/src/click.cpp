#include "click.hpp"

#include <algorithm>
#include <cmath>

#include "numpy.hpp"

namespace sda {
namespace {
constexpr double CLICK_FREQ = 1000.0, GROUP_FREQ = 1225.0, ACCENT_FREQ = 1500.0;
constexpr double CLICK_DECAY = 0.035, CLICK_ATTACK = 0.001;
constexpr double CLICK_PEAK = 0.7, GROUP_PEAK = 0.85, ACCENT_PEAK = 1.0;
constexpr double RAMP_FLOOR = 0.0001;

std::vector<double> voice(double peak, double freq, int sr) {
    const long n = py_round_int(CLICK_DECAY * sr);
    std::vector<double> out(n);
    const double span = CLICK_DECAY - CLICK_ATTACK;
    for (long i = 0; i < n; i++) {
        double t = (double)i / sr;
        double env = t <= CLICK_ATTACK ? RAMP_FLOOR * std::pow(peak / RAMP_FLOOR, t / CLICK_ATTACK)
                                       : peak * std::pow(RAMP_FLOOR / peak, (t - CLICK_ATTACK) / span);
        out[i] = std::sin(2.0 * M_PI * freq * t) * env;
    }
    return out;
}

bool bar_position(long index, const std::vector<BeatBar>& bars, int accent_mode, long& offset, long& per_bar) {
    if (accent_mode == ACCENT_OFF) return false;
    if (accent_mode > 0) {
        offset = index % accent_mode;
        per_bar = accent_mode;
        return true;
    }
    const BeatBar* mark = nullptr;
    for (auto& b : bars) {
        if (b.beat <= index) mark = &b;
        else break;
    }
    if (!mark || mark->beats_per_bar < 1) return false;
    long d = index - mark->beat;
    offset = ((d % mark->beats_per_bar) + mark->beats_per_bar) % mark->beats_per_bar;  // Python's %
    per_bar = mark->beats_per_bar;
    return true;
}
}  // namespace

std::vector<int> default_grouping(int n) {
    if (n < 1) return {};
    if (n == 5 || n == 7) {
        std::vector<int> g{3};
        for (int i = 0; i < (n - 3) / 2; i++) g.push_back(2);
        return g;
    }
    if (n >= 6 && n % 3 == 0) return std::vector<int>(n / 3, 3);
    return {n};
}

std::vector<int> normalise_grouping(const std::vector<int>& groups, int n) {
    if (n < 1) return {};
    if (groups.empty()) return default_grouping(n);
    long sum = 0;
    for (int g : groups) {
        if (g < 1) return default_grouping(n);
        sum += g;
    }
    if (sum != n) return default_grouping(n);
    return groups;
}

std::set<int> group_offsets(const std::vector<int>& groups) {
    std::set<int> out;
    int at = 0;
    for (size_t i = 0; i + 1 < groups.size(); i++) {
        at += groups[i];
        out.insert(at);
    }
    return out;
}

int beat_level(long index, const std::vector<BeatBar>& bars, int accent_mode, const std::vector<int>& groups) {
    if (index < 0) return LEVEL_WEAK;
    long offset = 0, per_bar = 0;
    if (!bar_position(index, bars, accent_mode, offset, per_bar)) return LEVEL_WEAK;
    if (offset == 0) return LEVEL_DOWNBEAT;
    auto g = accent_mode > 0 ? normalise_grouping(groups, (int)per_bar) : default_grouping((int)per_bar);
    return group_offsets(g).count((int)offset) ? LEVEL_GROUP : LEVEL_WEAK;
}

std::vector<int16_t> render_click(const std::vector<double>& beats_in, const std::vector<BeatBar>& bars,
                                  double duration, int sr, double multiplier, int accent_mode,
                                  const std::vector<int>& groups) {
    if (multiplier != 0.5 && multiplier != 1.0 && multiplier != 2.0) multiplier = 1.0;
    // rescale_beats
    std::vector<double> grid;
    if (multiplier == 2.0) {
        for (size_t i = 0; i + 1 < beats_in.size(); i++) {
            grid.push_back(beats_in[i]);
            grid.push_back((beats_in[i] + beats_in[i + 1]) / 2.0);
        }
        if (!beats_in.empty()) grid.push_back(beats_in.back());
    } else if (multiplier == 0.5) {
        for (size_t i = 0; i < beats_in.size(); i += 2) grid.push_back(beats_in[i]);
    } else {
        grid = beats_in;
    }
    const long total = py_round_int(duration * sr);
    if (grid.empty() || total <= 0) return {};
    std::vector<float> buf((size_t)total, 0.0f);
    const std::vector<double> voices[3] = {voice(CLICK_PEAK, CLICK_FREQ, sr), voice(GROUP_PEAK, GROUP_FREQ, sr),
                                           voice(ACCENT_PEAK, ACCENT_FREQ, sr)};
    for (size_t i = 0; i < grid.size(); i++) {
        // source_index
        long src = multiplier == 2.0 ? (i % 2 == 0 ? (long)(i / 2) : -1) : multiplier == 0.5 ? (long)(i * 2) : (long)i;
        int level = beat_level(src, bars, accent_mode, groups);
        long start = py_round_int(grid[i] * sr);
        if (start >= total || start < 0) continue;
        const auto& v = voices[level];
        long n = std::min((long)v.size(), total - start);
        for (long k = 0; k < n; k++) buf[start + k] = (float)((double)buf[start + k] + v[k]);
    }
    std::vector<int16_t> pcm((size_t)total);
    for (long i = 0; i < total; i++) {
        float x = std::min(1.0f, std::max(-1.0f, buf[i]));
        pcm[i] = (int16_t)(x * 32767.0f);  // numpy astype: truncation
    }
    return pcm;
}

}  // namespace sda
