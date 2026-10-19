// StemDeck for Android: the ORIGINAL demucs.cpp engine, for comparison.
// Same API as the other engines (engine.hpp); only separate() differs.
#include "engine.hpp"

#include "model.hpp"
#include "tensor.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sdengine
{

const std::array<const char *, kNumSources> kSourceNames = {
    "drums", "bass", "other", "vocals", "guitar", "piano"};

namespace
{

std::atomic<bool> g_cancel{false};

struct Cancelled
{
};

std::mutex g_model_mutex;
std::unique_ptr<demucscpp::demucs_model> g_model;
std::string g_model_path;

// ─── A read-only memory-mapped float file ───
class MappedFloats
{
  public:
    explicit MappedFloats(const std::string &path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0)
            return;
        struct stat st
        {
        };
        if (fstat(fd_, &st) != 0 || st.st_size <= 0)
            return;
        bytes_ = static_cast<size_t>(st.st_size);
        void *p = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED)
        {
            bytes_ = 0;
            return;
        }
        madvise(p, bytes_, MADV_SEQUENTIAL);
        data_ = static_cast<const float *>(p);
    }
    ~MappedFloats()
    {
        if (data_)
            munmap(const_cast<float *>(data_), bytes_);
        if (fd_ >= 0)
            ::close(fd_);
    }
    MappedFloats(const MappedFloats &) = delete;
    MappedFloats &operator=(const MappedFloats &) = delete;

    bool ok() const { return data_ != nullptr; }
    const float *data() const { return data_; }
    size_t count() const { return bytes_ / sizeof(float); }

  private:
    int fd_ = -1;
    size_t bytes_ = 0;
    const float *data_ = nullptr;
};

// ─── 16-bit stereo WAV writer, header patched on close ───
class WavWriter
{
  public:
    bool open(const std::string &path)
    {
        path_ = path;
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_)
            return false;
        std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
        uint8_t header[44] = {0};
        return std::fwrite(header, 1, sizeof(header), f_) == sizeof(header);
    }

    bool write(const float *l, const float *r, size_t n)
    {
        buf_.resize(n * 2);
        for (size_t i = 0; i < n; ++i)
        {
            buf_[2 * i] = toPcm(l[i]);
            buf_[2 * i + 1] = toPcm(r[i]);
            sumsq_ += static_cast<double>(l[i]) * l[i];
        }
        frames_ += n;
        return std::fwrite(buf_.data(), sizeof(int16_t), buf_.size(), f_) ==
               buf_.size();
    }

    bool close()
    {
        if (!f_)
            return false;
        const uint32_t dataBytes = static_cast<uint32_t>(frames_ * 4);
        uint8_t h[44];
        auto put32 = [&](int off, uint32_t v) { std::memcpy(h + off, &v, 4); };
        auto put16 = [&](int off, uint16_t v) { std::memcpy(h + off, &v, 2); };
        std::memcpy(h, "RIFF", 4);
        put32(4, 36 + dataBytes);
        std::memcpy(h + 8, "WAVEfmt ", 8);
        put32(16, 16);
        put16(20, 1); // PCM
        put16(22, 2); // stereo
        put32(24, kSampleRate);
        put32(28, kSampleRate * 4);
        put16(32, 4);
        put16(34, 16);
        std::memcpy(h + 36, "data", 4);
        put32(40, dataBytes);
        bool ok = std::fseek(f_, 0, SEEK_SET) == 0 &&
                  std::fwrite(h, 1, sizeof(h), f_) == sizeof(h);
        ok = (std::fclose(f_) == 0) && ok;
        f_ = nullptr;
        return ok;
    }

    void abandon()
    {
        if (f_)
        {
            std::fclose(f_);
            f_ = nullptr;
        }
        std::remove(path_.c_str());
    }

    double rms() const
    {
        return frames_ ? std::sqrt(sumsq_ / static_cast<double>(frames_)) : 0.0;
    }
    const std::string &path() const { return path_; }

    ~WavWriter()
    {
        if (f_)
            std::fclose(f_);
    }

  private:
    static int16_t toPcm(float v)
    {
        if (!(v == v))
            v = 0.0f; // NaN
        v = std::max(-1.0f, std::min(1.0f, v));
        return static_cast<int16_t>(std::lrintf(v * 32767.0f));
    }

    std::string path_;
    std::FILE *f_ = nullptr;
    std::vector<int16_t> buf_;
    size_t frames_ = 0;
    double sumsq_ = 0.0;
};

// ─── Windowed-sinc resampler table ───
class SincTable
{
  public:
    // cutoff is relative to the input Nyquist (<= 1).
    SincTable(double cutoff, int halfWidth, int phases)
        : cutoff_(cutoff), half_(halfWidth), phases_(phases),
          table_(static_cast<size_t>(halfWidth) * phases + 2)
    {
        for (size_t i = 0; i < table_.size(); ++i)
        {
            double x = static_cast<double>(i) / phases_;
            double s = x == 0.0 ? 1.0
                                : std::sin(M_PI * cutoff_ * x) /
                                      (M_PI * cutoff_ * x);
            double u = x / half_;
            double w = u >= 1.0 ? 0.0
                                : 0.42 + 0.5 * std::cos(M_PI * u) +
                                      0.08 * std::cos(2.0 * M_PI * u);
            table_[i] = static_cast<float>(cutoff_ * s * w);
        }
    }
    float at(double x) const
    {
        x = std::fabs(x) * phases_;
        size_t i = static_cast<size_t>(x);
        if (i + 1 >= table_.size())
            return 0.0f;
        float f = static_cast<float>(x - i);
        return table_[i] + (table_[i + 1] - table_[i]) * f;
    }
    int halfWidth() const { return half_; }

  private:
    double cutoff_;
    int half_;
    int phases_;
    std::vector<float> table_;
};

bool cancelled() { return g_cancel.load(std::memory_order_relaxed); }

} // namespace

void request_cancel() { g_cancel.store(true); }

std::string load_model(const std::string &path)
{
    std::lock_guard<std::mutex> lk(g_model_mutex);
    if (g_model && g_model_path == path)
        return "";
    try
    {
        auto model = std::make_unique<demucscpp::demucs_model>();
        if (!demucscpp::load_demucs_model(path, model.get()))
            return "The separation model could not be read.";
        if (model->is_4sources)
            return "The model file is the 4-source model, not the 6-source one.";
        g_model = std::move(model);
        g_model_path = path;
        return "";
    }
    catch (const std::bad_alloc &)
    {
        return "Not enough memory to load the separation model.";
    }
    catch (const std::exception &e)
    {
        return std::string("Loading the model failed: ") + e.what();
    }
}

bool model_loaded()
{
    std::lock_guard<std::mutex> lk(g_model_mutex);
    return g_model != nullptr;
}

std::string prepare_input(const std::string &rawPath, int sampleRate,
                          int channels, const std::string &outPath,
                          const ProgressFn &progress)
{
    g_cancel.store(false);
    if (sampleRate <= 0 || channels <= 0)
        return "The audio format is not supported.";
    MappedFloats in(rawPath);
    if (!in.ok())
        return "The decoded audio could not be read.";
    const size_t inFrames = in.count() / static_cast<size_t>(channels);
    if (inFrames < 1024)
        return "The audio is too short to separate.";
    const float *src = in.data();

    // Fold to stereo. Channel layouts beyond stereo are assumed to start
    // L, R, C (the Android/AAC order); the centre goes to both sides.
    auto frameLR = [&](size_t i, float &l, float &r) {
        const float *p = src + i * channels;
        if (channels == 1)
        {
            l = r = p[0];
        }
        else
        {
            l = p[0];
            r = p[1];
            if (channels >= 3)
            {
                l += 0.7071f * p[2];
                r += 0.7071f * p[2];
            }
        }
    };

    std::FILE *out = std::fopen(outPath.c_str(), "wb");
    if (!out)
        return "Could not write the prepared audio.";
    std::setvbuf(out, nullptr, _IOFBF, 1 << 20);

    const size_t block = 1 << 15;
    std::vector<float> buf(block * 2);
    bool ok = true;

    if (sampleRate == kSampleRate)
    {
        for (size_t start = 0; start < inFrames && ok; start += block)
        {
            size_t n = std::min(block, inFrames - start);
            for (size_t i = 0; i < n; ++i)
                frameLR(start + i, buf[2 * i], buf[2 * i + 1]);
            ok = std::fwrite(buf.data(), sizeof(float), n * 2, out) == n * 2;
            if (cancelled() || !progress(float(start) / inFrames, "Preparing audio"))
            {
                std::fclose(out);
                std::remove(outPath.c_str());
                return "cancelled";
            }
        }
    }
    else
    {
        const double ratio = static_cast<double>(sampleRate) / kSampleRate;
        const double cutoff = std::min(1.0, 1.0 / ratio) * 0.97;
        const int half = static_cast<int>(std::ceil(32.0 / cutoff));
        SincTable table(cutoff, half, 512);
        const size_t outFrames =
            static_cast<size_t>(std::floor(inFrames / ratio));
        for (size_t start = 0; start < outFrames && ok; start += block)
        {
            size_t n = std::min(block, outFrames - start);
            for (size_t j = 0; j < n; ++j)
            {
                const double t = (start + j) * ratio;
                const long center = static_cast<long>(std::floor(t));
                double accL = 0.0, accR = 0.0;
                for (long k = center - half + 1; k <= center + half; ++k)
                {
                    if (k < 0 || k >= static_cast<long>(inFrames))
                        continue;
                    float w = table.at(t - k);
                    if (w == 0.0f)
                        continue;
                    float l, r;
                    frameLR(static_cast<size_t>(k), l, r);
                    accL += w * l;
                    accR += w * r;
                }
                buf[2 * j] = static_cast<float>(accL);
                buf[2 * j + 1] = static_cast<float>(accR);
            }
            ok = std::fwrite(buf.data(), sizeof(float), n * 2, out) == n * 2;
            if (cancelled() || !progress(float(start) / outFrames, "Resampling"))
            {
                std::fclose(out);
                std::remove(outPath.c_str());
                return "cancelled";
            }
        }
    }
    ok = (std::fclose(out) == 0) && ok;
    if (!ok)
    {
        std::remove(outPath.c_str());
        return "Writing the prepared audio failed (is the storage full?).";
    }
    return "";
}

InputStats analyze_input(const std::string &preparedPath)
{
    InputStats st;
    MappedFloats in(preparedPath);
    if (!in.ok())
        return st;
    const size_t n = in.count() / 2;
    const float *p = in.data();
    st.durationSec = static_cast<double>(n) / kSampleRate;

    // K-weighting (ITU-R BS.1770), same filter design as pyloudnorm, which
    // desktop StemDeck uses.
    struct Biquad
    {
        double b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
        double run(double x)
        {
            double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };
    const double fs = kSampleRate;
    Biquad shelf{}, hp{};
    {
        const double G = 4.0, Q = 1.0 / std::sqrt(2.0), fc = 1500.0;
        const double A = std::pow(10.0, G / 40.0);
        const double w0 = 2.0 * M_PI * fc / fs;
        const double alpha = std::sin(w0) / (2.0 * Q);
        const double c = std::cos(w0), sA = std::sqrt(A);
        const double a0 = (A + 1) - (A - 1) * c + 2 * sA * alpha;
        shelf.b0 = A * ((A + 1) + (A - 1) * c + 2 * sA * alpha) / a0;
        shelf.b1 = -2 * A * ((A - 1) + (A + 1) * c) / a0;
        shelf.b2 = A * ((A + 1) + (A - 1) * c - 2 * sA * alpha) / a0;
        shelf.a1 = 2 * ((A - 1) - (A + 1) * c) / a0;
        shelf.a2 = ((A + 1) - (A - 1) * c - 2 * sA * alpha) / a0;
    }
    {
        const double Q = 0.5, fc = 38.0;
        const double w0 = 2.0 * M_PI * fc / fs;
        const double alpha = std::sin(w0) / (2.0 * Q);
        const double c = std::cos(w0);
        const double a0 = 1 + alpha;
        hp.b0 = (1 + c) / 2 / a0;
        hp.b1 = -(1 + c) / a0;
        hp.b2 = (1 + c) / 2 / a0;
        hp.a1 = -2 * c / a0;
        hp.a2 = (1 - alpha) / a0;
    }

    // 100 ms sub-blocks; a 400 ms gating block is four of them.
    const size_t step = kSampleRate / 10;
    std::vector<double> sub;
    sub.reserve(n / step + 1);
    double acc = 0.0, peak = 0.0;
    size_t inSub = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const double m = 0.5 * (static_cast<double>(p[2 * i]) + p[2 * i + 1]);
        peak = std::max(peak, std::fabs(m));
        const double k = hp.run(shelf.run(m));
        acc += k * k;
        if (++inSub == step)
        {
            sub.push_back(acc);
            acc = 0.0;
            inSub = 0;
        }
    }
    if (peak > 1e-9)
    {
        st.peakDb = 20.0 * std::log10(peak);
        st.hasPeak = true;
    }
    if (sub.size() >= 4)
    {
        std::vector<double> z;
        for (size_t b = 0; b + 3 < sub.size(); ++b)
            z.push_back((sub[b] + sub[b + 1] + sub[b + 2] + sub[b + 3]) /
                        (4.0 * step));
        auto loud = [](double ms) { return -0.691 + 10.0 * std::log10(ms); };
        double sumAbs = 0.0;
        size_t cntAbs = 0;
        for (double v : z)
            if (v > 0 && loud(v) > -70.0)
            {
                sumAbs += v;
                ++cntAbs;
            }
        if (cntAbs)
        {
            const double gate = loud(sumAbs / cntAbs) - 10.0;
            double sumRel = 0.0;
            size_t cntRel = 0;
            for (double v : z)
                if (v > 0 && loud(v) > -70.0 && loud(v) > gate)
                {
                    sumRel += v;
                    ++cntRel;
                }
            if (cntRel)
            {
                st.lufs = loud(sumRel / cntRel);
                st.hasLufs = std::isfinite(st.lufs);
            }
        }
    }
    return st;
}

// The original engine: demucs.cpp exactly as upstream ships it
// (sevagh/demucs.cpp f1206e9, unmodified in demucs/). The whole song goes
// through demucscpp::demucs_inference in one call, on one thread, the way
// upstream's own command-line tool runs it. opts.threads and
// opts.chunkSeconds are ignored. This needs much more memory than the
// StemDeck engines: about 1.7 GB for the model's working buffers plus the
// whole song and its six stems as float.
SeparateResult separate(const std::string &preparedPath,
                        const std::string &outDir, const SeparateOptions &opts,
                        const ProgressFn &progress)
{
    SeparateResult res;
    g_cancel.store(false);

    const demucscpp::demucs_model *model = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_model_mutex);
        model = g_model.get();
    }
    if (!model)
    {
        res.error = "The separation model is not loaded.";
        return res;
    }

    MappedFloats in(preparedPath);
    if (!in.ok())
    {
        res.error = "The prepared audio could not be read.";
        return res;
    }
    const long N = static_cast<long>(in.count() / 2);
    const float *pcm = in.data();
    if (N < kSampleRate)
    {
        res.error = "The audio is shorter than one second.";
        return res;
    }

    std::unique_ptr<Eigen::Tensor3dXf> out;
    try
    {
        Eigen::MatrixXf audio(2, N);
        for (long i = 0; i < N; ++i)
        {
            audio(0, i) = pcm[2 * i];
            audio(1, i) = pcm[2 * i + 1];
        }
        // Upstream picks its time shift with rand(); seed it so that every
        // run of the original engine gives the same result.
        std::srand(1);
        demucscpp::ProgressCallback cb = [&progress](float p, const std::string &) {
            if (cancelled() || !progress(std::max(0.0f, std::min(1.0f, p)), "Separating stems"))
            {
                g_cancel.store(true);
                throw Cancelled();
            }
        };
        out = std::make_unique<Eigen::Tensor3dXf>(demucscpp::demucs_inference(*model, audio, cb));
    }
    catch (const Cancelled &)
    {
        res.cancelled = true;
        return res;
    }
    catch (const std::bad_alloc &)
    {
        res.error = "The phone ran out of memory. The original engine separates the "
                    "whole song at once and needs several GB for a long song.";
        return res;
    }
    catch (const std::exception &e)
    {
        res.error = std::string("Separation failed: ") + e.what();
        return res;
    }

    const auto &T = *out;
    const long frames = std::min<long>(N, static_cast<long>(T.dimension(2)));
    std::vector<float> lineL, lineR;
    const long block = 1 << 16;
    for (int s = 0; s < kNumSources; ++s)
    {
        if (!opts.write[s])
            continue;
        WavWriter w;
        if (!w.open(outDir + "/" + kSourceNames[s] + ".wav.part"))
        {
            res.error = "Could not create the stem files.";
            return res;
        }
        bool ok = true;
        for (long i = 0; i < frames && ok; i += block)
        {
            const long n = std::min(block, frames - i);
            lineL.resize(n);
            lineR.resize(n);
            for (long j = 0; j < n; ++j)
            {
                lineL[j] = T(s, 0, i + j);
                lineR[j] = T(s, 1, i + j);
            }
            ok = w.write(lineL.data(), lineR.data(), n);
        }
        res.rms[s] = w.rms();
        const std::string part = w.path();
        const std::string fin = part.substr(0, part.size() - 5); // drop ".part"
        if (!ok || !w.close() || std::rename(part.c_str(), fin.c_str()) != 0)
        {
            w.abandon();
            res.error = "Writing the stems failed (is the storage full?).";
            return res;
        }
    }
    progress(1.0f, "Separating stems");
    res.ok = true;
    return res;
}

} // namespace sdengine
