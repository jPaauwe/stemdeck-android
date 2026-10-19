// Beat This! (github.com/CPJKU/beat_this, MIT) around its ONNX export:
// the log-mel front end (torchaudio's MelSpectrogram as LogMelSpect builds
// it), the 1500-frame chunking with 6-frame borders, and the "minimal"
// post-processing that turns logits into beat and downbeat times.
#pragma once
#include <functional>
#include <vector>

#include "librosa.hpp"

namespace sda {

constexpr int kBeatSr = 22050;
constexpr int kBeatFps = 50;
constexpr int kBeatChunk = 1500;
constexpr int kBeatBorder = 6;

// LogMelSpect(): frames x 128, from mono audio at 22050 Hz.
Spec beat_this_logmel(const std::vector<float>& y);
// torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, None, "slaney"), 513 x 128.
std::vector<float> beat_this_mel_fb();

// split_piece(..., avoid_short_end=True): chunk start frames (may be negative).
std::vector<long> beat_this_chunk_starts(long frames);

// Runs the model over a whole piece the way split_predict_aggregate does
// (keep_first). `model(input, frames, beat_out, downbeat_out)` runs one chunk
// of `frames` x 128 values and fills two `frames` long outputs; return false
// to stop (cancellation). Returns false if stopped.
using BeatModelFn = std::function<bool(const float* input, long frames, float* beat, float* downbeat)>;
bool beat_this_logits(const Spec& mel, const BeatModelFn& model, std::vector<float>& beat, std::vector<float>& downbeat,
                      const std::function<bool(double)>& progress = nullptr);

// Postprocessor(type="minimal"): beat and downbeat times in seconds.
void beat_this_minimal(const std::vector<float>& beat, const std::vector<float>& downbeat,
                       std::vector<double>& beats, std::vector<double>& downbeats);

}  // namespace sda
