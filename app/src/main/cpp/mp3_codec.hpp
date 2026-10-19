// MP3 encoding and decoding of stems with LAME (libmp3lame + mpglib).
#pragma once

#include <functional>
#include <string>

namespace sdmp3
{
// Called with the fraction done. Return false to stop.
using Progress = std::function<bool(float)>;

// A 16-bit PCM WAV file (as the engine writes them) -> constant-bitrate MP3
// with a LAME info tag, so decoders can remove the encoder delay and padding
// (gapless). Returns "" on success, "cancelled", or an error message.
std::string encode_wav(const std::string &wavPath, const std::string &mp3Path,
                       int kbps, const Progress &progress = nullptr);

// An MP3 file written by encode_wav -> 16-bit PCM WAV with the encoder delay
// and padding removed, so it lines up sample for sample with the original.
std::string decode_to_wav(const std::string &mp3Path, const std::string &wavPath,
                          const Progress &progress = nullptr);
} // namespace sdmp3
