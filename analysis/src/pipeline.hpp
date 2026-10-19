// The analysis stages as the app runs them: each takes the files on disk
// and returns what goes into the job (and beside the stems).
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "beatgrid.hpp"

namespace sda {

// Progress 0..1; return false to cancel (the stage then throws Cancelled).
using Progress = std::function<bool(double)>;
struct Cancelled {};

// Mono 22050 Hz float, decoded the way desktop's ffmpeg call does it, from a
// 16-bit/float WAV or the engine's prepared stereo float32 input (.f32).
// `max_input_frames` caps the input first, like ffmpeg's -t (0: whole file).
std::vector<float> load_mono_22k(const std::string& path, size_t max_input_frames = 0);

// The click track's beat grid from one source (the drums stem): Beat This!
// through `model_path`, then desktop's clean-up. Never throws for a song it
// cannot grid (BeatGrid::ok is false and why says why); throws Cancelled, or
// std::runtime_error when the model cannot run at all.
BeatGrid run_beat_grid(const std::string& source_path, const std::string& source_label, const std::string& model_path,
                       int threads, const Progress& progress, const std::string& provider = "");

}  // namespace sda
