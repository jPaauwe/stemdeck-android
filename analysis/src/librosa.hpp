// The parts of librosa 0.11 (ISC licence) that desktop StemDeck's beat grid
// and key detection call, ported to C++ with numpy's dtypes kept where they
// change results (float32 spectra, float64 statistics).
#pragma once
#include <cstddef>
#include <vector>

namespace sda {

// A frames x bins matrix, row-major ([frame][bin]); librosa's is bins x frames.
struct Spec {
    size_t frames = 0, bins = 0;
    std::vector<float> v;
    float* row(size_t t) { return v.data() + t * bins; }
    const float* row(size_t t) const { return v.data() + t * bins; }
};

// scipy.signal.get_window("hann", n, fftbins=True)
std::vector<double> hann_periodic(int n);

// |librosa.stft(y, n_fft, hop, window="hann", center=True, pad_mode="constant")|^power,
// as float32 like librosa's complex64 output.
Spec stft_power(const std::vector<float>& y, int n_fft, int hop, double power);

// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney"), float32, n_mels x (n_fft/2+1).
std::vector<float> mel_filters(double sr, int n_fft, int n_mels, double fmin, double fmax);

// librosa.onset.onset_strength(y, sr, n_fft, hop_length, n_mels, aggregate=np.median),
// with the mel spectrogram's defaults (power 2, fmax sr/2) and power_to_db(ref=1, top_db=80).
std::vector<float> onset_strength_median(const std::vector<float>& y, double sr, int n_fft, int hop, int n_mels);

// librosa.util.peak_pick (sparse frame indices).
std::vector<int> peak_pick(const std::vector<double>& x, int pre_max, int post_max, int pre_avg, int post_avg,
                           double delta, int wait);

// librosa.onset.onset_detect(onset_envelope, sr, hop_length, units="time").
std::vector<double> onset_detect_times(const std::vector<float>& env, double sr, int hop);

// librosa.feature.rms(y, frame_length, hop_length) (center=True, constant pad).
std::vector<float> rms(const std::vector<float>& y, int frame_length, int hop);

// numpy.percentile(x, q) with the default "linear" method. T is the array's dtype.
template <typename T>
double percentile(const std::vector<T>& x, double q);
// numpy.median, in the array's dtype.
template <typename T>
T median(std::vector<T> x);

}  // namespace sda
