// StemDeck Android: the GPU, through Vulkan compute.
//
// Vulkan is loaded at run time (dlopen), so the engine still runs where
// there is none. On first use the GPU runs a small self-test against the
// CPU; if anything fails, the GPU is switched off and the engine carries on
// on the CPU.
//
// Work is recorded into a Batch: inputs are copied in when added, the
// recorded operations run in order on run(), and outputs are copied back
// only if everything succeeded, so a caller can fall back to the CPU
// without having changed anything. One Batch runs at a time; other threads
// that want the GPU meanwhile use the CPU instead (try_begin), which keeps
// the GPU and every core busy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace sdgpu
{
// Use the GPU at all. Default on.
void set_enabled(bool on);
bool enabled();

// Starts Vulkan on first call. False when there is no usable GPU, it is
// switched off, or it failed its self-test.
bool available();

// "GPU: <device name>" or why the GPU is not used.
std::string info();

// Forgets the model weights kept on the GPU. Call when the model changes.
void clear_weights();

struct Buf
{
    int id = -1;
    bool valid() const { return id >= 0; }
};

class Batch
{
  public:
    // The GPU if it is free right now; nullptr if busy, off or missing.
    static std::unique_ptr<Batch> try_begin();
    // Waits for the GPU; nullptr if off or missing.
    static std::unique_ptr<Batch> begin();
    ~Batch();

    // A buffer holding a copy of p[0, n).
    Buf input(const float *p, size_t n);
    // Model weights: kept on the GPU between batches, by address.
    Buf weights(const float *p, size_t n);
    // A buffer of n floats, contents undefined.
    Buf scratch(size_t n);

    // C = alpha * op(A) * op(B) + beta * C, column-major, offsets and
    // leading dimensions in floats.
    void gemm(bool transA, bool transB, uint32_t M, uint32_t N, uint32_t K,
              float alpha, Buf A, size_t offA, uint32_t lda, Buf B, size_t offB,
              uint32_t ldb, float beta, Buf C, size_t offC, uint32_t ldc);

    // Softmax down each of `cols` columns of `rows` values (stride ld).
    void softmax_cols(Buf X, size_t off, uint32_t rows, uint32_t cols, uint32_t ld);

    // Copies buf[off, off + n) to dst after a successful run().
    void output(Buf b, size_t off, float *dst, size_t n);

    // Runs everything recorded. False if the GPU failed; then no output was
    // written and the caller should do the work on the CPU.
    bool run();

    struct Impl;
    explicit Batch(Impl *impl); // internal: use try_begin() or begin()

  private:
    Impl *impl_;
};
} // namespace sdgpu
