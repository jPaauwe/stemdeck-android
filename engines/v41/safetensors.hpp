// StemDeck Android: reads Demucs weights in the safetensors format the
// Demucs project publishes since v4.1.0 (huggingface.co/adefossez), and
// writes them in the format demucs.cpp's loader reads: a magic number, then
// per tensor: n_dims, name length, dims (size-1 dims dropped), name, float16
// data. Float32 and bfloat16 tensors are converted to float16.
#pragma once

#include <string>

namespace sdengine
{
// Returns "" on success.
std::string safetensors_to_ggml(const std::string &in, const std::string &out);
} // namespace sdengine
