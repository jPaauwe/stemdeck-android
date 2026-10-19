// Desktop StemDeck's key detection (app/pipeline/analyze.py, Apache-2.0):
// Albrecht-Shanahan profiles against the whole song's chroma, and the
// song's edges to tell a key from its relative. Ported with its constants.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace sda {

struct KeyResult {
    bool ok = false;
    std::string key;    // e.g. "G# maj"
    std::string scale;  // "Major", "Natural Minor" or "Harmonic Minor"
    long confidence = 0;
};

using Edge = std::pair<std::vector<double>, std::vector<double>>;  // (chroma, bass chroma), max 1

// _detect_key(chroma_mean, edges)
KeyResult detect_key(const std::vector<double>& chroma_mean, const std::vector<Edge>& edges);

// detect_key_from_audio(harmony, bass, 22050, whole_song); `bass` may be empty.
KeyResult detect_key_from_audio(const std::vector<float>& harmony, const std::vector<float>& bass, bool whole_song);

// refine_key_from_stems: the harmony stems (guitar, piano, other; vocals at
// half weight) and the bass stem, each a WAV path or "" when missing.
KeyResult key_from_stems(const std::vector<std::pair<std::string, double>>& harmony_paths, const std::string& bass_path);

}  // namespace sda
