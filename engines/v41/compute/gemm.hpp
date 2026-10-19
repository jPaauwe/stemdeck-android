// StemDeck Android: one entry point for the engine's large matrix products.
//
// C (M x N) = alpha * op(A) * op(B) + beta * C, column-major, op(X) = X or
// X transposed, as in BLAS. Large products go to the GPU when it is free
// (see gpu.hpp); otherwise they are split over every core (see pool.hpp).
#pragma once

namespace sdgemm
{
enum Flags
{
    kNone = 0,
    kConstA = 1, // A is model weights: kept on the GPU between calls
    kConstB = 2, // B is model weights
};

void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
          const float *A, int lda, const float *B, int ldb, float beta,
          float *C, int ldc, int flags = kNone);

// The same, always on the CPU (all cores).
void gemm_cpu(bool transA, bool transB, int M, int N, int K, float alpha,
              const float *A, int lda, const float *B, int ldb, float beta,
              float *C, int ldc);

// Products smaller than this many multiply-adds stay on the CPU.
void set_gpu_min_macs(double macs);
} // namespace sdgemm
