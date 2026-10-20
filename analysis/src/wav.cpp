#include "wav.hpp"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace sda {
namespace {
struct File {
    FILE* f;
    explicit File(FILE* f) : f(f) {}
    ~File() { if (f) fclose(f); }
};
uint32_t u32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
uint16_t u16(const uint8_t* p) { return p[0] | (p[1] << 8); }
}  // namespace

Audio read_wav(const std::string& path) {
    File file(fopen(path.c_str(), "rb"));
    if (!file.f) throw std::runtime_error("cannot open " + path);
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, file.f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4))
        throw std::runtime_error("not a WAV file: " + path);
    int fmt = 0, channels = 0, rate = 0, bits = 0;
    bool have_fmt = false;
    while (true) {
        uint8_t ch[8];
        if (fread(ch, 1, 8, file.f) != 8) throw std::runtime_error("no data chunk in " + path);
        uint32_t size = u32(ch + 4);
        if (!memcmp(ch, "fmt ", 4)) {
            std::vector<uint8_t> b(size);
            if (fread(b.data(), 1, size, file.f) != size || size < 16) throw std::runtime_error("bad fmt chunk");
            fmt = u16(&b[0]);
            channels = u16(&b[2]);
            rate = (int)u32(&b[4]);
            bits = u16(&b[14]);
            if (fmt == 0xFFFE && size >= 26) fmt = u16(&b[24]);  // WAVE_FORMAT_EXTENSIBLE
            have_fmt = true;
            if (size & 1) fseek(file.f, 1, SEEK_CUR);
        } else if (!memcmp(ch, "data", 4)) {
            if (!have_fmt) throw std::runtime_error("data before fmt in " + path);
            if (channels <= 0) throw std::runtime_error("bad channel count");
            Audio a;
            a.channels = channels;
            a.sample_rate = rate;
            // The engine writes 0xFFFFFFFF-free headers, but tolerate a data size past the end.
            long here = ftell(file.f);
            fseek(file.f, 0, SEEK_END);
            long end = ftell(file.f);
            fseek(file.f, here, SEEK_SET);
            uint64_t avail = (uint64_t)(end - here);
            if (size == 0xFFFFFFFFu || size > avail) size = (uint32_t)avail;
            if (fmt == 1 && bits == 16) {
                size_t n = size / 2;
                n -= n % channels;
                std::vector<int16_t> pcm(n);
                if (fread(pcm.data(), 2, n, file.f) != n) throw std::runtime_error("short read");
                a.samples.resize(n);
                // ffmpeg's s16 -> float: x * (1.0f / (1 << 15))
                for (size_t i = 0; i < n; i++) a.samples[i] = pcm[i] * (1.0f / 32768.0f);
            } else if (fmt == 3 && bits == 32) {
                size_t n = size / 4;
                n -= n % channels;
                a.samples.resize(n);
                if (fread(a.samples.data(), 4, n, file.f) != n) throw std::runtime_error("short read");
            } else {
                throw std::runtime_error("unsupported WAV format in " + path);
            }
            return a;
        } else {
            fseek(file.f, size + (size & 1), SEEK_CUR);
        }
    }
}

void write_wav16_pcm(const std::string& path, const int16_t* x, size_t frames, int channels, int sample_rate) {
    File file(fopen(path.c_str(), "wb"));
    if (!file.f) throw std::runtime_error("cannot write " + path);
    uint32_t data = (uint32_t)(frames * channels * 2);
    uint8_t h[44];
    auto p32 = [&](int o, uint32_t v) { h[o] = v; h[o + 1] = v >> 8; h[o + 2] = v >> 16; h[o + 3] = v >> 24; };
    auto p16 = [&](int o, uint16_t v) { h[o] = v; h[o + 1] = v >> 8; };
    memcpy(h, "RIFF", 4); p32(4, 36 + data); memcpy(h + 8, "WAVEfmt ", 8);
    p32(16, 16); p16(20, 1); p16(22, channels); p32(24, sample_rate);
    p32(28, sample_rate * channels * 2); p16(32, channels * 2); p16(34, 16);
    memcpy(h + 36, "data", 4); p32(40, data);
    if (fwrite(h, 1, 44, file.f) != 44 || fwrite(x, 2, frames * channels, file.f) != frames * channels)
        throw std::runtime_error("could not write " + path);
}

WavReader::WavReader(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    f_ = f;
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fclose(f); f_ = nullptr;
        throw std::runtime_error("not a WAV file: " + path);
    }
    int fmt = 0, bits = 0;
    bool have_fmt = false;
    while (true) {
        uint8_t ch[8];
        if (fread(ch, 1, 8, f) != 8) { fclose(f); f_ = nullptr; throw std::runtime_error("no data chunk in " + path); }
        uint32_t size = u32(ch + 4);
        if (!memcmp(ch, "fmt ", 4)) {
            std::vector<uint8_t> b(size);
            if (fread(b.data(), 1, size, f) != size || size < 16) { fclose(f); f_ = nullptr; throw std::runtime_error("bad fmt chunk"); }
            fmt = u16(&b[0]);
            channels_ = u16(&b[2]);
            rate_ = (int)u32(&b[4]);
            bits = u16(&b[14]);
            if (fmt == 0xFFFE && size >= 26) fmt = u16(&b[24]);
            have_fmt = true;
            if (size & 1) fseek(f, 1, SEEK_CUR);
        } else if (!memcmp(ch, "data", 4)) {
            if (!have_fmt || channels_ <= 0 || !((fmt == 1 && bits == 16) || (fmt == 3 && bits == 32))) {
                fclose(f); f_ = nullptr;
                throw std::runtime_error("unsupported WAV format in " + path);
            }
            float_ = fmt == 3;
            long here = ftell(f);
            fseek(f, 0, SEEK_END);
            long end = ftell(f);
            fseek(f, here, SEEK_SET);
            uint64_t avail = (uint64_t)(end - here);
            if (size == 0xFFFFFFFFu || size > avail) size = (uint32_t)avail;
            frames_ = left_ = size / (float_ ? 4 : 2) / channels_;
            return;
        } else {
            fseek(f, size + (size & 1), SEEK_CUR);
        }
    }
}

WavReader::~WavReader() {
    if (f_) fclose((FILE*)f_);
}

size_t WavReader::read(std::vector<float>& out, size_t frames) {
    size_t n = std::min(frames, left_);
    out.resize(n * channels_);
    if (!n) return 0;
    FILE* f = (FILE*)f_;
    if (float_) {
        if (fread(out.data(), 4, n * channels_, f) != n * channels_) throw std::runtime_error("short read");
    } else {
        pcm_.resize(n * channels_);
        if (fread(pcm_.data(), 2, n * channels_, f) != n * channels_) throw std::runtime_error("short read");
        for (size_t i = 0; i < n * channels_; i++) out[i] = pcm_[i] * (1.0f / 32768.0f);
    }
    left_ -= n;
    return n;
}

Audio read_f32(const std::string& path, int channels, int sample_rate) {
    File file(fopen(path.c_str(), "rb"));
    if (!file.f) throw std::runtime_error("cannot open " + path);
    fseek(file.f, 0, SEEK_END);
    long bytes = ftell(file.f);
    fseek(file.f, 0, SEEK_SET);
    Audio a;
    a.channels = channels;
    a.sample_rate = sample_rate;
    size_t n = (size_t)bytes / 4;
    n -= n % channels;
    a.samples.resize(n);
    if (fread(a.samples.data(), 4, n, file.f) != n) throw std::runtime_error("short read");
    return a;
}

void write_wav16(const std::string& path, const float* x, size_t frames, int channels, int sample_rate) {
    File file(fopen(path.c_str(), "wb"));
    if (!file.f) throw std::runtime_error("cannot write " + path);
    uint32_t data = (uint32_t)(frames * channels * 2);
    uint8_t h[44];
    auto p32 = [&](int o, uint32_t v) { h[o] = v; h[o + 1] = v >> 8; h[o + 2] = v >> 16; h[o + 3] = v >> 24; };
    auto p16 = [&](int o, uint16_t v) { h[o] = v; h[o + 1] = v >> 8; };
    memcpy(h, "RIFF", 4); p32(4, 36 + data); memcpy(h + 8, "WAVEfmt ", 8);
    p32(16, 16); p16(20, 1); p16(22, channels); p32(24, sample_rate);
    p32(28, sample_rate * channels * 2); p16(32, channels * 2); p16(34, 16);
    memcpy(h + 36, "data", 4); p32(40, data);
    fwrite(h, 1, 44, file.f);
    std::vector<int16_t> buf;
    const size_t block = 1 << 15;
    for (size_t i = 0; i < frames * channels; i += block) {
        size_t n = std::min(block, frames * channels - i);
        buf.resize(n);
        for (size_t j = 0; j < n; j++) {
            float v = x[i + j];
            if (!(v == v)) v = 0.0f;
            v = std::max(-1.0f, std::min(1.0f, v));
            buf[j] = (int16_t)std::lrintf(v * 32767.0f);
        }
        fwrite(buf.data(), 2, n, file.f);
    }
}

}  // namespace sda
