// The lead/backing vocal split desktop StemDeck runs on the vocals stem:
// UVR-MDX-NET Karaoke 2 through audio-separator's MDXSeparator (0.44.5) with
// the parameters audio-separator uses for it: n_fft 5120, hop 1024, dim_f
// 2048, dim_t 256 (segment size 256), overlap 0.25, compensate 1.065, no
// denoise, peaks normalised to 0.9, 16-bit output.
#pragma once
#include <functional>
#include <string>

namespace sda {

struct VocalSplitResult {
    bool ok = false;
    std::string why;
    double t_model = 0;  // seconds in the model
    double seconds = 0;  // length of the audio
    // RMS over every sample (float, x/32768), for stem presence
    double vocals_rms = 0, lead_rms = 0, backing_rms = 0;
};

// Reads `vocals_wav` (16-bit or float WAV, 44.1 kHz), writes `lead_wav` (the
// model's "Vocals": the mix minus 1.065 x its output) and `backing_wav`
// ("Instrumental": its output) as 16-bit stereo WAVs. Progress 0..1; return
// false to cancel (throws Cancelled). Throws std::runtime_error when the
// model cannot run.
VocalSplitResult split_vocals(const std::string& vocals_wav, const std::string& lead_wav, const std::string& backing_wav,
                              const std::string& model, int threads, const std::function<bool(double)>& progress,
                              const std::string& provider = "");

}  // namespace sda
