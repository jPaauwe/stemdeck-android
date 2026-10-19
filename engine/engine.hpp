// StemDeck for Android: the on-device separation engine.
//
// Wraps demucs.cpp (MIT, github.com/sevagh/demucs.cpp), a C++ port of the
// same htdemucs_6s model desktop StemDeck runs through PyTorch. Nothing in
// here knows about JNI, so the CLI in test/ can drive the exact same code on a
// desktop machine in CI.
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace sdengine
{

constexpr int kSampleRate = 44100;
constexpr int kNumSources = 6;

// Output order of the 6-source model.
extern const std::array<const char *, kNumSources> kSourceNames;

// Called from the thread that runs separate(), never from a worker. Returning
// false asks the engine to stop as soon as it can.
using ProgressFn = std::function<bool(float fraction, const char *stage)>;

struct InputStats
{
    double durationSec = 0.0;
    double peakDb = 0.0;        // sample peak of the mono downmix, dBFS
    double lufs = 0.0;          // BS.1770 integrated loudness of the mono downmix
    bool hasLufs = false;       // false when too short or silent
    bool hasPeak = false;
};

struct SeparateOptions
{
    int threads = 3;            // chunks processed in parallel
    int chunkSeconds = 30;      // unused since segments are scheduled directly
    // Which of the six sources to write, in kSourceNames order.
    std::array<bool, kNumSources> write{{true, true, true, true, true, true}};
};

struct SeparateResult
{
    bool ok = false;
    bool cancelled = false;
    std::string error;
    // RMS of channel 0 per source (same measure desktop uses for presence).
    std::array<double, kNumSources> rms{};
};

// Loads the ggml weights once; safe to call again (no-op when the same path is
// already loaded). Returns an empty string on success.
std::string load_model(const std::string &path);
bool model_loaded();

// Converts raw interleaved float32 PCM at any rate/channel count into the
// engine's input format: interleaved stereo float32 at 44.1 kHz. Mono is
// duplicated; more than two channels are folded to L/R. Returns "" on success.
std::string prepare_input(const std::string &rawPath, int sampleRate,
                          int channels, const std::string &outPath,
                          const ProgressFn &progress);

// Loudness/peak/duration of a prepared (stereo, 44.1 kHz, float32) input.
InputStats analyze_input(const std::string &preparedPath);

// Separates a prepared input into <outDir>/<source>.wav (16-bit stereo).
SeparateResult separate(const std::string &preparedPath,
                        const std::string &outDir,
                        const SeparateOptions &opts,
                        const ProgressFn &progress);

// Asks a running separate()/prepare_input() to stop. Cleared when the next
// call starts.
void request_cancel();

} // namespace sdengine
