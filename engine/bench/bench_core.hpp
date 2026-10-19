// Benchmark entry points shared by the Android bench app and the desktop
// bench CLI. This file is compiled once against the engine being optimized
// and once against the frozen baseline (bench/src/main/cpp/baseline), so it
// only uses APIs both have.
#pragma once

#include <string>

namespace sdbench
{

// Loads the ggml weights for segment profiling. Returns "" on success.
std::string load_model(const std::string &path);
void free_model();

// Runs the network once on exactly one 7.8 s segment of deterministic test
// audio, on the calling thread. Writes the raw output (6 sources x 2 ch x
// samples, float32) to outPath when it is not empty. Returns JSON:
// {"ms": total, "stages": [[name, ms]...], "ops": [[name, calls, ms]...]}
std::string profile_segment(const std::string &outPath);

// Writes a deterministic test song (stereo float32, 44.1 kHz) for full runs.
bool write_test_song(const std::string &path, double seconds);

// SNR in dB of b against a (reference), per source, for two outputs written
// by profile_segment. Returns JSON {"snr_db": [...6], "max_abs_diff": x}.
std::string compare_outputs(const std::string &a, const std::string &b);

} // namespace sdbench
