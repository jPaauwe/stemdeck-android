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

// 16-bit PCM WAV from samples already converted (no rounding or clipping here).
void write_wav16_pcm(const std::string& path, const int16_t* interleaved, size_t frames, int channels, int sample_rate);

// Reads a 16-bit PCM or 32-bit float WAV a block at a time, so a long stem
// never has to be in memory twice. Samples come out as float (16-bit: x/32768,
// exact) interleaved.
class WavReader {
   public:
    explicit WavReader(const std::string& path);  // throws std::runtime_error
    ~WavReader();
    WavReader(const WavReader&) = delete;
    WavReader& operator=(const WavReader&) = delete;
    int channels() const { return channels_; }
    int sample_rate() const { return rate_; }
    bool is_float() const { return float_; }
    size_t frames() const { return frames_; }
    // Up to `frames` frames into `out` (resized); returns frames read (0 at the end).
    size_t read(std::vector<float>& out, size_t frames);

   private:
    void* f_ = nullptr;
    int channels_ = 0, rate_ = 0;
    bool float_ = false;
    size_t frames_ = 0, left_ = 0;
    std::vector<int16_t> pcm_;
};

}  // namespace sda
