// Reading and writing the audio files the analysis stages work on.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sda {

struct Audio {
    std::vector<float> samples;  // interleaved
    int channels = 0;
    int sample_rate = 0;
    size_t frames() const { return channels ? samples.size() / channels : 0; }
};

// 16-bit PCM or 32-bit float WAV, any channel count. Throws std::runtime_error.
Audio read_wav(const std::string& path);
// Raw interleaved float32 (the engine's prepared input: stereo, 44.1 kHz).
Audio read_f32(const std::string& path, int channels, int sample_rate);
// 16-bit PCM WAV, with the same rounding and clipping as the separation engine.
void write_wav16(const std::string& path, const float* interleaved, size_t frames, int channels, int sample_rate);

}  // namespace sda
