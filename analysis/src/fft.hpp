// Real FFTs in double precision (numpy's pocketfft computes these in double
// for librosa's float64 windows; torch in float, which double rounds the
// same to float32). Plans are cached per thread.
#pragma once
#include <complex>
#include <map>
#include <memory>
#include <vector>

#include <unsupported/Eigen/FFT>

namespace sda {

class RealFFT {
   public:
    // out = rfft(in), size n/2+1. `in` must have size n.
    static void forward(const std::vector<double>& in, std::vector<std::complex<double>>& out) {
        auto& fft = instance();
        fft.fwd(out, in);
        out.resize(in.size() / 2 + 1);
    }
    // Full complex forward FFT.
    static void forward_c(const std::vector<std::complex<double>>& in, std::vector<std::complex<double>>& out) {
        auto& fft = instance();
        fft.fwd(out, in);
    }
    // Inverse complex FFT, unscaled (sum, like numpy's ifft * n).
    static void inverse_c(const std::vector<std::complex<double>>& in, std::vector<std::complex<double>>& out) {
        auto& fft = instance();
        fft.inv(out, in);  // Eigen scales by 1/n
        const double n = (double)in.size();
        for (auto& v : out) v *= n;
    }

    // irfft: real output of size n from a half spectrum of n/2+1 values,
    // scaled by 1/n (numpy's and torch's "backward" norm). The imaginary
    // parts of DC and Nyquist are ignored, as pocketfft's c2r does.
    static void inverse_real(std::vector<std::complex<double>> half, std::vector<double>& out, int n) {
        auto& fft = instance();
        half[0] = std::complex<double>(half[0].real(), 0.0);
        if (n % 2 == 0) half[n / 2] = std::complex<double>(half[n / 2].real(), 0.0);
        fft.inv(out, half, n);
    }

   private:
    static Eigen::FFT<double>& instance() {
        thread_local Eigen::FFT<double> fft;
        thread_local bool init = false;
        if (!init) {
            fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
            init = true;
        }
        return fft;
    }
};

}  // namespace sda
