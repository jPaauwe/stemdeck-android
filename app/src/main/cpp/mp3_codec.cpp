#include "mp3_codec.hpp"

#include <lame.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <memory>
#include <vector>

namespace sdmp3
{
namespace
{
struct FileCloser
{
    void operator()(FILE *f) const
    {
        if (f)
            std::fclose(f);
    }
};
using File = std::unique_ptr<FILE, FileCloser>;

uint32_t le32(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
uint16_t le16(const unsigned char *p) { return uint16_t(p[0] | (p[1] << 8)); }

struct WavInfo
{
    int rate = 0, channels = 0, bits = 0;
    long dataOffset = 0;
    uint64_t dataBytes = 0;
};

// Walks the RIFF chunks to "fmt " and "data".
bool read_wav_header(FILE *f, WavInfo &w)
{
    unsigned char h[12];
    if (std::fread(h, 1, 12, f) != 12 || std::memcmp(h, "RIFF", 4) || std::memcmp(h + 8, "WAVE", 4))
        return false;
    bool haveFmt = false;
    for (;;)
    {
        unsigned char c[8];
        if (std::fread(c, 1, 8, f) != 8)
            return false;
        const uint32_t size = le32(c + 4);
        if (!std::memcmp(c, "fmt ", 4))
        {
            unsigned char fmt[16];
            if (size < 16 || std::fread(fmt, 1, 16, f) != 16)
                return false;
            if (le16(fmt) != 1) // PCM only
                return false;
            w.channels = le16(fmt + 2);
            w.rate = static_cast<int>(le32(fmt + 4));
            w.bits = le16(fmt + 14);
            haveFmt = true;
            if (std::fseek(f, static_cast<long>(size - 16 + (size & 1)), SEEK_CUR))
                return false;
        }
        else if (!std::memcmp(c, "data", 4))
        {
            if (!haveFmt)
                return false;
            w.dataOffset = std::ftell(f);
            w.dataBytes = size;
            // A streamed WAV may have a 0 or oversized length: use the file.
            std::fseek(f, 0, SEEK_END);
            const long end = std::ftell(f);
            if (w.dataBytes == 0 || w.dataOffset + static_cast<long>(w.dataBytes) > end)
                w.dataBytes = static_cast<uint64_t>(end - w.dataOffset);
            std::fseek(f, w.dataOffset, SEEK_SET);
            return true;
        }
        else if (std::fseek(f, static_cast<long>(size + (size & 1)), SEEK_CUR))
            return false;
    }
}

void put32(unsigned char *p, uint32_t v)
{
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

void wav_header(unsigned char h[44], int rate, int channels, uint64_t frames)
{
    const uint32_t block = static_cast<uint32_t>(channels * 2);
    const uint32_t data = static_cast<uint32_t>(frames * block);
    std::memcpy(h, "RIFF", 4);
    put32(h + 4, 36 + data);
    std::memcpy(h + 8, "WAVEfmt ", 8);
    put32(h + 16, 16);
    h[20] = 1;
    h[21] = 0;
    h[22] = static_cast<unsigned char>(channels);
    h[23] = 0;
    put32(h + 24, static_cast<uint32_t>(rate));
    put32(h + 28, static_cast<uint32_t>(rate) * block);
    h[32] = static_cast<unsigned char>(block);
    h[33] = 0;
    h[34] = 16;
    h[35] = 0;
    std::memcpy(h + 36, "data", 4);
    put32(h + 40, data);
}

struct LameCloser
{
    void operator()(lame_global_flags *g) const
    {
        if (g)
            lame_close(g);
    }
};
} // namespace

std::string encode_wav(const std::string &wavPath, const std::string &mp3Path,
                       int kbps, const Progress &progress)
{
    File in(std::fopen(wavPath.c_str(), "rb"));
    if (!in)
        return "The stem could not be read.";
    WavInfo w;
    if (!read_wav_header(in.get(), w) || w.bits != 16 || w.channels < 1 || w.channels > 2 || w.rate <= 0)
        return "The stem is not a 16-bit WAV file.";

    std::unique_ptr<lame_global_flags, LameCloser> g(lame_init());
    if (!g)
        return "The MP3 encoder could not start.";
    lame_set_num_channels(g.get(), w.channels);
    lame_set_in_samplerate(g.get(), w.rate);
    lame_set_out_samplerate(g.get(), w.rate);
    lame_set_VBR(g.get(), vbr_off);
    lame_set_brate(g.get(), kbps);
    lame_set_mode(g.get(), w.channels == 1 ? MONO : JOINT_STEREO);
    lame_set_quality(g.get(), 2); // LAME's recommended high quality (-q 2)
    lame_set_bWriteVbrTag(g.get(), 1); // info tag: delay and padding, for gapless
    lame_set_write_id3tag_automatic(g.get(), 0);
    if (lame_init_params(g.get()) < 0)
        return "The MP3 encoder does not support this format.";

    File out(std::fopen(mp3Path.c_str(), "wb"));
    if (!out)
        return "The MP3 file could not be written.";

    const int kFrames = 8192;
    std::vector<short> pcm(static_cast<size_t>(kFrames) * w.channels);
    std::vector<unsigned char> mp3(static_cast<size_t>(1.25 * kFrames + 7200));
    const uint64_t total = w.dataBytes / (2u * w.channels);
    uint64_t done = 0;
    while (done < total)
    {
        const int n = static_cast<int>(std::min<uint64_t>(kFrames, total - done));
        if (std::fread(pcm.data(), 2u * w.channels, n, in.get()) != static_cast<size_t>(n))
            return "The stem could not be read.";
        const int bytes = w.channels == 2
            ? lame_encode_buffer_interleaved(g.get(), pcm.data(), n, mp3.data(), static_cast<int>(mp3.size()))
            : lame_encode_buffer(g.get(), pcm.data(), pcm.data(), n, mp3.data(), static_cast<int>(mp3.size()));
        if (bytes < 0)
            return "MP3 encoding failed.";
        if (bytes && std::fwrite(mp3.data(), 1, bytes, out.get()) != static_cast<size_t>(bytes))
            return "The MP3 file could not be written. Is the storage full?";
        done += n;
        if (progress && !progress(static_cast<float>(done) / static_cast<float>(total)))
            return "cancelled";
    }
    const int bytes = lame_encode_flush(g.get(), mp3.data(), static_cast<int>(mp3.size()));
    if (bytes < 0)
        return "MP3 encoding failed.";
    if (bytes && std::fwrite(mp3.data(), 1, bytes, out.get()) != static_cast<size_t>(bytes))
        return "The MP3 file could not be written. Is the storage full?";

    // The first frame was reserved for the info tag; fill it in now.
    unsigned char tag[2880];
    const size_t tagBytes = lame_get_lametag_frame(g.get(), tag, sizeof tag);
    if (tagBytes > 0 && tagBytes <= sizeof tag)
    {
        if (std::fseek(out.get(), 0, SEEK_SET) || std::fwrite(tag, 1, tagBytes, out.get()) != tagBytes)
            return "The MP3 file could not be written.";
    }
    if (std::fflush(out.get()))
        return "The MP3 file could not be written. Is the storage full?";
    return {};
}

std::string decode_to_wav(const std::string &mp3Path, const std::string &wavPath,
                          const Progress &progress)
{
    File in(std::fopen(mp3Path.c_str(), "rb"));
    if (!in)
        return "The MP3 stem could not be read.";
    std::fseek(in.get(), 0, SEEK_END);
    const long inSize = std::ftell(in.get());
    std::fseek(in.get(), 0, SEEK_SET);

    hip_t hip = hip_decode_init();
    if (!hip)
        return "The MP3 decoder could not start.";
    struct HipCloser
    {
        hip_t h;
        ~HipCloser() { hip_decode_exit(h); }
    } closer{hip};

    File out(std::fopen(wavPath.c_str(), "wb"));
    if (!out)
        return "The decoded stem could not be written.";
    unsigned char header[44] = {};
    std::fwrite(header, 1, 44, out.get()); // filled in at the end

    // mpglib decodes at most one frame (1152 samples) per call.
    short left[1152 * 2], right[1152 * 2];
    std::vector<short> inter(1152 * 2 * 2);
    unsigned char buf[4096];
    mp3data_struct md;
    std::memset(&md, 0, sizeof md);
    int encDelay = -1, encPadding = -1;
    int channels = 0, rate = 0;
    long long skip = -1; // samples still to drop at the start
    uint64_t written = 0;
    long readBytes = 0;
    bool eof = false;
    size_t len = 0;

    for (;;)
    {
        // Drain what the decoder has buffered before feeding it more.
        int n = hip_decode1_headersB(hip, buf, len, left, right, &md, &encDelay, &encPadding);
        len = 0;
        if (n < 0)
        {
            if (written == 0)
                return "The MP3 stem could not be decoded.";
            break; // damaged tail: keep what decoded
        }
        if (n == 0)
        {
            if (eof)
                break;
            len = std::fread(buf, 1, sizeof buf, in.get());
            readBytes += static_cast<long>(len);
            if (len == 0)
                eof = true;
            if (progress && inSize > 0 && !progress(static_cast<float>(readBytes) / static_cast<float>(inSize)))
                return "cancelled";
            continue;
        }
        if (md.header_parsed && channels == 0)
        {
            channels = md.stereo;
            rate = md.samplerate;
        }
        if (skip < 0)
            // The encoder delay from the info tag, plus mpglib's own 528 + 1.
            skip = (encDelay >= 0 ? encDelay : 576) + 528 + 1;
        int start = 0;
        if (skip > 0)
        {
            start = static_cast<int>(std::min<long long>(skip, n));
            skip -= start;
        }
        const int count = n - start;
        if (count <= 0)
            continue;
        for (int i = 0; i < count; ++i)
        {
            if (channels == 2)
            {
                inter[2 * i] = left[start + i];
                inter[2 * i + 1] = right[start + i];
            }
            else
                inter[i] = left[start + i];
        }
        const size_t samples = static_cast<size_t>(count) * (channels == 2 ? 2 : 1);
        if (std::fwrite(inter.data(), 2, samples, out.get()) != samples)
            return "The decoded stem could not be written. Is the storage full?";
        written += static_cast<uint64_t>(count);
    }
    if (channels == 0 || rate == 0)
        return "The MP3 stem could not be decoded.";

    // Drop the encoder's padding at the end (minus mpglib's delay, which the
    // start already accounted for).
    if (encPadding > 0)
    {
        const long long drop = encPadding - (528 + 1);
        if (drop > 0 && static_cast<uint64_t>(drop) < written)
            written -= static_cast<uint64_t>(drop);
    }
    const int ch = channels == 2 ? 2 : 1;
    wav_header(header, rate, ch, written);
    if (std::fflush(out.get()))
        return "The decoded stem could not be written.";
    std::fseek(out.get(), 0, SEEK_SET);
    std::fwrite(header, 1, 44, out.get());
    out.reset();
    // Cut the dropped padding off the file.
    if (truncate(wavPath.c_str(), static_cast<off_t>(44 + written * 2 * ch)) != 0)
        return "The decoded stem could not be written.";
    return {};
}
} // namespace sdmp3
