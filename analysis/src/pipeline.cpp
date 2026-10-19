#include "pipeline.hpp"

#include <chrono>
#include <stdexcept>

#include "beat_this.hpp"
#include "ort.hpp"
#include "swr.hpp"
#include "wav.hpp"

namespace sda {

std::vector<float> load_mono_22k(const std::string& path, size_t max_input_frames) {
    Audio a = path.size() > 4 && path.compare(path.size() - 4, 4, ".f32") == 0 ? read_f32(path, 2, 44100) : read_wav(path);
    if (max_input_frames && a.frames() > max_input_frames) a.samples.resize(max_input_frames * a.channels);
    auto mono = downmix_like_ffmpeg(a.samples.data(), a.frames(), a.channels);
    a.samples.clear();
    a.samples.shrink_to_fit();
    return resample_like_ffmpeg(mono, a.sample_rate, 22050);
}

BeatGrid run_beat_grid(const std::string& source_path, const std::string& source_label, const std::string& model_path,
                       int threads, const Progress& progress, const std::string& provider) {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a) { return std::chrono::duration<double>(clock::now() - a).count(); };
    auto report = [&](double f) { if (progress && !progress(f)) throw Cancelled{}; };
    report(0.0);
    auto t = clock::now();
    std::vector<float> y = load_mono_22k(source_path);
    double t_load = secs(t);
    report(0.05);
    t = clock::now();
    Spec mel = beat_this_logmel(y);
    double t_mel = secs(t);
    report(0.10);
    std::vector<double> beats, downbeats;
    t = clock::now();
    if (mel.frames > 0) {
        OrtModel model(model_path, threads, provider);
        std::vector<float> bl, dl;
        bool done = beat_this_logits(
            mel,
            [&](const float* in, long frames, float* b, float* d) {
                auto out = model.run(in, {1, frames, 128});
                if (out.size() < 2 || (long)out[0].data.size() != frames || (long)out[1].data.size() != frames)
                    throw std::runtime_error("the beat model returned an unexpected shape");
                std::copy(out[0].data.begin(), out[0].data.end(), b);
                std::copy(out[1].data.begin(), out[1].data.end(), d);
                return true;
            },
            bl, dl, [&](double f) { return !progress || progress(0.10 + 0.75 * f); });
        if (!done) throw Cancelled{};
        beat_this_minimal(bl, dl, beats, downbeats);
    }
    double t_model = secs(t);
    if (beats.size() < 2) {
        BeatGrid g;
        g.why = "the beat model found " + std::to_string(beats.size()) + " beats";
        return g;
    }
    report(0.85);
    t = clock::now();
    BeatGrid g = compute_beat_grid(y, 22050.0, beats, downbeats, "beat_this", source_label);
    g.t_load = t_load;
    g.t_mel = t_mel;
    g.t_model = t_model;
    g.t_grid = secs(t);
    report(1.0);
    return g;
}

}  // namespace sda
