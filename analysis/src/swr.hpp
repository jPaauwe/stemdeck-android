// What ffmpeg does to the audio before desktop StemDeck analyses it
// (`ffmpeg -i stem.wav -ac 1 -ar 22050 -f f32le -`), written here to give
// the same samples: the stereo-to-mono mix at -3 dB per channel and a
// Kaiser-windowed sinc resampler with libswresample's default parameters
// (filter_size 32, cutoff 0.97, beta 9, exact rational ratios) and mirrored
// edges. No FFmpeg code is used; scripts/analysis_parity.py and the tests
// compare the output with ffmpeg 7's (equal to float rounding).
#pragma once
#include <vector>

namespace sda {

// Channel mix to mono as `-ac 1` does it. Stereo is L*0.7071 + R*0.7071 (no
// normalisation for float output); mono passes through.
std::vector<float> downmix_like_ffmpeg(const float* interleaved, size_t frames, int channels);

// `-ar out_rate`. Supports the ratios where libswresample's exact-rational
// filter bank applies (out/in reduces to a numerator <= 1024), which covers
// every integer ratio. Throws std::invalid_argument otherwise.
std::vector<float> resample_like_ffmpeg(const std::vector<float>& x, int in_rate, int out_rate);

}  // namespace sda
