// StemDeck for Android: separation engine. See engine.hpp.
#include "engine.hpp"

#include "dsp.hpp"
#include "model.hpp"
#include "gpu.hpp"
#include "safetensors.hpp"
#include "pool.hpp"
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

std::string compute_info(const SeparateOptions &opts)
{
    sdpool::set_threads(opts.computeThreads);
    sdgpu::set_enabled(opts.gpu);
    const int n = sdpool::threads();
    return std::to_string(n) + (n == 1 ? " core, " : " cores, ") + sdgpu::info();
}

std::string load_model(const std::string &path)
{
    std::lock_guard<std::mutex> lk(g_model_mutex);
    if (g_model && g_model_path == path)
        return "";
    try
    {
        auto model = std::make_unique<demucscpp::demucs_model>();
        // The weights as the Demucs project publishes them (v4.1.0 on,
        // .safetensors), or demucs.cpp's own format.
        bool loaded;
        if (path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0)
        {
            const std::string tmp = path + ".ggml.tmp";
            const std::string err = safetensors_to_ggml(path, tmp);
            if (!err.empty())
            {
                std::remove(tmp.c_str());
                return err;
            }
            loaded = demucscpp::load_demucs_model(tmp, model.get());
            std::remove(tmp.c_str());
        }
        else
        {
            loaded = demucscpp::load_demucs_model(path, model.get());
        }
        if (!loaded)
            return "The separation model could not be read.";
        if (model->is_4sources)
            return "The model file is the 4-source model, not the 6-source one.";
        sdgpu::clear_weights(); // the old model's weights, if any
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

SeparateResult separate(const std::string &preparedPath,
                        const std::string &outDir, const SeparateOptions &opts,
                        const ProgressFn &progress)
{
    // The whole song is cut into Demucs' own 7.8 s segments with 25 %
    // overlap (split_inference in demucs.cpp), and worker threads take
    // segments in turn. Each worker owns its segment buffers. Finished
    // segments are blended into a sliding window with Demucs' triangular
    // weights; samples no later segment can touch any more are written out.
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

    // Whole-track normalization, as PyTorch Demucs: ref = mix.mean(channels).
    double sum = 0.0, sumsq = 0.0;
    for (long i = 0; i < N; ++i)
    {
        const double m = 0.5 * (static_cast<double>(pcm[2 * i]) + pcm[2 * i + 1]);
        sum += m;
        sumsq += m * m;
    }
    const double mean = sum / N;
    const double var = std::max(0.0, (sumsq - N * mean * mean) / (N - 1));
    const float refMean = static_cast<float>(mean);
    float refStd = static_cast<float>(std::sqrt(var));
    if (!(refStd > 1e-8f))
        refStd = 1.0f;

    const int SEG = static_cast<int>(demucscpp::SEGMENT_LEN_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);
    const long STRIDE = static_cast<long>((1 - demucscpp::OVERLAP) * SEG);

    // Demucs' "shift" (shifts=1): the track is padded with up to 0.5 s of
    // silence in front, so the segment grid does not start on the first
    // sample. Demucs picks the offset at random; it is derived from the
    // length here so results are repeatable. Leaving the shift out measurably
    // hurts separation (vocals leak into other), so it stays.
    const long maxShift = static_cast<long>(demucscpp::MAX_SHIFT_SECS * demucscpp::SUPPORTED_SAMPLE_RATE);
    const long lead = maxShift - static_cast<long>((static_cast<unsigned long long>(N) * 2654435761ULL) % maxShift);
    const long NS = N + lead; // length of the shifted track
    auto sampleAt = [&](long t, int ch) -> float {
        // t is a position in the shifted track
        const long i = t - lead;
        return (i < 0 || i >= N) ? 0.0f : (pcm[2 * i + ch] - refMean) / refStd;
    };
    const int numSegs = static_cast<int>((NS + STRIDE - 1) / STRIDE);
    const int threads = std::max(1, std::min(opts.threads, numSegs));
    // The maths inside the segments uses every core (or computeThreads),
    // and the GPU when it is on.
    sdpool::set_threads(opts.computeThreads);
    sdgpu::set_enabled(opts.gpu);

    // Demucs' transition weight: a triangle over the segment.
    std::vector<float> weight(SEG, 0.0f);
    {
        const int half = SEG / 2;
        for (int k = 0; k < half; ++k)
            weight[k] = static_cast<float>(k + 1);
        for (int k = 0; k < half; ++k)
            weight[SEG - 1 - k] = weight[k];
        const float mx = *std::max_element(weight.begin(), weight.end());
        for (float &v : weight)
            v = std::pow(v / mx, demucscpp::TRANSITION_POWER);
    }

    std::array<WavWriter, kNumSources> writers;
    for (int s = 0; s < kNumSources; ++s)
    {
        if (!opts.write[s])
            continue;
        if (!writers[s].open(outDir + "/" + kSourceNames[s] + ".wav.part"))
        {
            res.error = "Could not create the stem files.";
            for (int t = 0; t < kNumSources; ++t)
                if (opts.write[t])
                    writers[t].abandon();
            return res;
        }
    }

    // Sliding blend window, starting at sample `base`: per source and
    // channel the weighted sum, plus the summed weight.
    std::mutex m;
    std::condition_variable cv;
    long base = 0;                 // first sample not yet written
    std::array<std::vector<float>, kNumSources * 2> acc;
    std::vector<float> wsum;
    std::vector<char> done(numSegs, 0);
    int lowestPending = 0;         // first segment not yet blended
    int inFlightLimit = threads + 2;
    std::atomic<int> nextSeg{0};
    std::atomic<int> segsDone{0};
    std::string workerError;
    bool writeOk = true;

    auto ensureWindow = [&](long end) {
        const size_t need = static_cast<size_t>(end - base);
        if (wsum.size() < need)
        {
            wsum.resize(need, 0.0f);
            for (auto &a : acc)
                a.resize(need, 0.0f);
        }
    };

    // Writes [base, upto) and drops it from the window. Called with m held.
    std::vector<float> lineL, lineR;
    auto flush = [&](long upto) {
        const long n = upto - base;
        if (n <= 0)
            return;
        // The first `lead` samples of the shifted track are the padding.
        const long skip = std::max(0L, std::min(n, lead - base));
        const long nOut = n - skip;
        for (int s = 0; s < kNumSources && writeOk && nOut > 0; ++s)
        {
            if (!opts.write[s])
                continue;
            lineL.resize(nOut);
            lineR.resize(nOut);
            const float *aL = acc[2 * s].data() + skip, *aR = acc[2 * s + 1].data() + skip;
            const float *wsp = wsum.data() + skip;
            for (long i = 0; i < nOut; ++i)
            {
                const float w = wsp[i] > 0.0f ? 1.0f / wsp[i] : 0.0f;
                // undo the normalization: x * std + mean
                lineL[i] = aL[i] * w * refStd + refMean;
                lineR[i] = aR[i] * w * refStd + refMean;
            }
            writeOk = writers[s].write(lineL.data(), lineR.data(), nOut);
        }
        wsum.erase(wsum.begin(), wsum.begin() + n);
        for (auto &a : acc)
            a.erase(a.begin(), a.begin() + n);
        base = upto;
    };

    auto worker = [&]() {
        try
        {
            demucscpp::demucs_segment_buffers buffers(2, SEG, kNumSources);
            demucscpp::stft_buffers stft_buf(buffers.padded_segment_samples);
            for (;;)
            {
                const int k = nextSeg.fetch_add(1);
                if (k >= numSegs || cancelled())
                    return;
                {
                    // Bound the blend window: do not run too far ahead of
                    // the oldest unfinished segment.
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return k < lowestPending + inFlightLimit || cancelled(); });
                }
                if (cancelled())
                    return;

                const long start = static_cast<long>(k) * STRIDE; // in the shifted track
                const int len = static_cast<int>(std::min<long>(SEG, NS - start));
                // symmetric zero padding of a short (last) segment
                const int padL = (SEG - len) / 2;
                buffers.mix.setZero();
                for (int i = 0; i < len; ++i)
                {
                    buffers.mix(0, padL + i) = sampleAt(start + i, 0);
                    buffers.mix(1, padL + i) = sampleAt(start + i, 1);
                }
                demucscpp::ProgressCallback cb = [](float, const std::string &) {
                    if (cancelled())
                        throw Cancelled();
                };
                demucscpp::model_inference(*model, buffers, stft_buf, cb, 0.0f, 1.0f);

                std::lock_guard<std::mutex> lk(m);
                ensureWindow(start + len);
                const long o = start - base;
                for (int s = 0; s < kNumSources; ++s)
                {
                    if (!opts.write[s])
                        continue;
                    float *aL = acc[2 * s].data() + o, *aR = acc[2 * s + 1].data() + o;
                    for (int i = 0; i < len; ++i)
                    {
                        aL[i] += weight[i] * buffers.targets_out(s, 0, padL + i);
                        aR[i] += weight[i] * buffers.targets_out(s, 1, padL + i);
                    }
                }
                float *ws = wsum.data() + o;
                for (int i = 0; i < len; ++i)
                    ws[i] += weight[i];
                done[k] = 1;
                while (lowestPending < numSegs && done[lowestPending])
                    ++lowestPending;
                // Everything before the start of the oldest unfinished
                // segment is final.
                const long final = lowestPending < numSegs
                                       ? static_cast<long>(lowestPending) * STRIDE
                                       : NS;
                flush(final);
                segsDone.fetch_add(1);
                if (!writeOk)
                    g_cancel.store(true);
                cv.notify_all();
            }
        }
        catch (const Cancelled &)
        {
        }
        catch (const std::bad_alloc &)
        {
            std::lock_guard<std::mutex> lk(m);
            if (workerError.empty())
                workerError = "The phone ran out of memory while separating. "
                              "Try fewer parallel threads in Settings.";
            g_cancel.store(true);
        }
        catch (const std::exception &e)
        {
            std::lock_guard<std::mutex> lk(m);
            if (workerError.empty())
                workerError = std::string("Separation failed: ") + e.what();
            g_cancel.store(true);
        }
        cv.notify_all();
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(worker);

    // Report progress from this (the calling) thread only: the JNI callback
    // must not run on a worker.
    while (true)
    {
        {
            std::unique_lock<std::mutex> lk(m);
            cv.wait_for(lk, std::chrono::milliseconds(250));
        }
        const int d = segsDone.load();
        if (d >= numSegs || cancelled())
            break;
        if (!progress(float(d) / numSegs, "Separating stems"))
        {
            g_cancel.store(true);
            cv.notify_all();
            break;
        }
    }
    cv.notify_all();
    for (auto &t : pool)
        t.join();

    bool complete;
    {
        std::lock_guard<std::mutex> lk(m);
        complete = writeOk && workerError.empty() && segsDone.load() == numSegs && base == NS;
        res.error = workerError;
    }
    if (!writeOk && res.error.empty())
        res.error = "Writing the stems failed (is the storage full?).";
    if (!complete)
    {
        res.cancelled = res.error.empty();
        for (int s = 0; s < kNumSources; ++s)
            if (opts.write[s])
                writers[s].abandon();
        return res;
    }

    for (int s = 0; s < kNumSources; ++s)
    {
        if (!opts.write[s])
            continue;
        res.rms[s] = writers[s].rms();
        const std::string part = writers[s].path();
        const std::string fin = part.substr(0, part.size() - 5); // drop ".part"
        if (!writers[s].close() || std::rename(part.c_str(), fin.c_str()) != 0)
        {
            res.error = "Finishing the stem files failed.";
            return res;
        }
    }
    progress(1.0f, "Separating stems");
    res.ok = true;
    return res;
}

} // namespace sdengine
