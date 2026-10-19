// Song sections as desktop StemDeck finds them: All-In-One (harmonix-all,
// eight folds averaged) on the four-stem spectrogram madmom computes, its
// functional post-processing, then app/pipeline/section_refine.py and
// sections.normalize_sections.
#pragma once
#include <functional>
#include <string>
#include <vector>

namespace sda {

constexpr int kSecFps = 100;
constexpr int kSecBins = 81;
constexpr int kSecFolds = 8;

// madmom's mono signal: a 16-bit stem's channels averaged in float64 and
// truncated back to int16 (np.mean(...).astype(int16)), or a float WAV's
// averaged in float32. `int16` tells which, because madmom scales the STFT
// window by 1/32767 only for integer input.
struct MadmomSignal {
    std::vector<float> x;  // int16 values (exact) or float samples
    bool int16 = true;
};
// One 16-bit stem.
MadmomSignal madmom_load_stem(const std::string& path);
// Several stems summed the way desktop's ffmpeg amix (normalize=0) writes its
// float WAV: per channel, in float32, in the given order.
MadmomSignal madmom_load_mix(const std::vector<std::string>& paths);

// LogarithmicFilterbank(fftfreq bins of a 2048 frame at 44.1 kHz, 12 bands per
// octave, 30 Hz - 17 kHz, norm_filters, unique_filters): 1024 x bands, float32.
std::vector<float> madmom_filterbank(int* bands = nullptr);

// FramedSignal(2048, fps 100) -> STFT(hanning) -> FilteredSpectrogram ->
// LogarithmicSpectrogram(mul 1, add 1): frames x 81, row-major.
// `int16_window`: the Hann window divided by 32767. madmom's STFT processor
// works that out from the first signal it is given and then caches it, and
// desktop runs one processor over bass (16-bit) first, so all four stems,
// including the float "other" mix, get the scaled window.
std::vector<float> madmom_logspec(const MadmomSignal& s, bool int16_window, size_t* frames = nullptr);

// The model input desktop builds (allin1_infer.spectrogram._extract_spectrogram
// on bass, drums, other = other + guitar + piano, vocals): 4 x frames x 81.
std::vector<float> section_features(const std::string& stems_dir, size_t* frames,
                                    const std::function<bool(double)>& progress = nullptr);

struct RawSegment {
    double start = 0, end = 0;
    std::string label;
};

// The ensemble's averaged outputs that post-processing reads.
struct SectionActivations {
    size_t frames = 0;
    std::vector<float> section_logits;   // frames
    std::vector<float> function_logits;  // 10 x frames
    std::vector<double> embedding;       // frames x 96 ([frame][stem * 24 + d]), mean over folds
};

extern const char* const kHarmonixLabels[10];

// postprocess_functional_structure (all-in-one-infer).
std::vector<RawSegment> functional_segments(const SectionActivations& a);

// section_refine.refine_segments. `beats` and `confidence` from beats.json
// (confidence < 0: no grid).
std::vector<RawSegment> refine_segments(const std::vector<RawSegment>& raw, const SectionActivations& a,
                                        const std::vector<double>& beats, double confidence);

// sections.normalize_sections, as the JSON array the job stores ("[]" when
// nothing usable is left).
std::string normalize_sections(const std::vector<RawSegment>& raw, double duration, int* count = nullptr);

struct SectionsResult {
    bool ok = false;
    std::string why;
    std::string sections_json = "[]";
    std::vector<RawSegment> raw, refined;
    int count = 0;
    double t_spec = 0, t_model = 0, t_post = 0;
};

// The whole stage from the six stems in `stems_dir` (bass, drums, vocals,
// other, guitar, piano .wav). `fold_models` are the eight ONNX files.
// Progress 0..1; return false to cancel (throws Cancelled).
SectionsResult run_sections(const std::string& stems_dir, const std::vector<std::string>& fold_models,
                            const std::vector<double>& beats, double confidence, double duration, int threads,
                            const std::function<bool(double)>& progress, const std::string& provider = "");

}  // namespace sda
