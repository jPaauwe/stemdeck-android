// Desktop StemDeck's beat grid clean-up (app/pipeline/beatgrid.py, Apache-2.0),
// ported line by line: it takes the detector's beats and downbeats and the
// audio they came from, and produces stems/beats.json.
#pragma once
#include <string>
#include <vector>

namespace sda {

struct BeatBar { long beat; long beats_per_bar; };

struct BeatGrid {
    bool ok = false;
    std::string why;  // when !ok
    std::string source, detector;
    std::vector<BeatBar> bars;
    double bpm = 0, duration = 0, interval_cv = 0, irregularity = 0;
    long confidence = 0, refined = 0, corrected = 0, detected = 0;
    long extrapolated_head = 0, extrapolated_tail = 0, gap_filled = 0;
    bool phase_shifted = false;
    std::vector<double> onsets, beats;
    // For the job record: tempo stability as analyze.py computes it
    // (1 - std/mean of the beat intervals, before the grid is extended).
    long tempo_stability = -1;
    // Seconds per step, for the bench: decode + resample, log-mel, model, clean-up.
    double t_load = 0, t_mel = 0, t_model = 0, t_grid = 0;

    std::string to_json() const;
};

// compute_beat_grid() from the detection on: `y` is the mono 22050 Hz audio
// the beats were detected on.
BeatGrid compute_beat_grid(const std::vector<float>& y, double sr, const std::vector<double>& beat_times,
                           const std::vector<double>& downbeat_times, const std::string& detector,
                           const std::string& source);

}  // namespace sda
