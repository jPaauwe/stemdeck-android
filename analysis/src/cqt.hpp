// librosa.feature.chroma_cqt (librosa 0.11, ISC) with tuning estimated as
// librosa does it (piptrack + pitch_tuning). The multirate constant-Q
// transform is librosa's, octave by octave; the one part that differs is the
// 2:1 resampler between octaves, which librosa delegates to libsoxr ("HQ"):
// here it is a linear-phase Kaiser FIR with soxr HQ's band edges (pass 0.913,
// stop 1.0 of the new Nyquist, 120 dB), which agrees with it to within the
// passband ripple of either.
#pragma once
#include <vector>

#include "librosa.hpp"

namespace sda {

// librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=bpo), as a fraction of a bin.
double estimate_tuning(const std::vector<float>& y, double sr, int bins_per_octave);

// librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, fmin=fmin, n_octaves=n_octaves)
// (bins_per_octave 36, tuning estimated, norm inf). frames x 12.
Spec chroma_cqt(const std::vector<float>& y, double sr, int hop, double fmin, int n_octaves);

// librosa.note_to_hz("C1")
constexpr double kC1 = 32.70319566257483;

// librosa.resample(y, orig_sr=2, target_sr=1, res_type="soxr_hq", scale=True) (see above).
std::vector<float> halve_rate(const std::vector<float>& y);

}  // namespace sda
