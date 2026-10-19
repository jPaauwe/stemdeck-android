// The click track as an exportable WAV: desktop StemDeck's
// app/pipeline/click_render.py (Apache-2.0), which reproduces the browser's
// Web Audio click (static/js/metronome.js) sample for sample. Both are
// pinned to analysis/test/fixtures/click_levels.json, upstream's parity table.
#pragma once
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "beatgrid.hpp"

namespace sda {

constexpr int LEVEL_WEAK = 0, LEVEL_GROUP = 1, LEVEL_DOWNBEAT = 2;
constexpr int ACCENT_AUTO = -1, ACCENT_OFF = 0;

std::vector<int> default_grouping(int beats_per_bar);
std::vector<int> normalise_grouping(const std::vector<int>& groups, int beats_per_bar);
std::set<int> group_offsets(const std::vector<int>& groups);
// beat_level(index, bars, accent_mode, groups); index < 0 means None.
int beat_level(long index, const std::vector<BeatBar>& bars, int accent_mode, const std::vector<int>& groups);

// render_click_wav's samples: mono, int16, round(duration * sample_rate) long,
// or empty when there is nothing to render.
std::vector<int16_t> render_click(const std::vector<double>& beats, const std::vector<BeatBar>& bars, double duration,
                                  int sample_rate, double multiplier, int accent_mode, const std::vector<int>& groups);

}  // namespace sda
