#include "gemm.hpp"

#include "gpu.hpp"
#include "pool.hpp"

#if __has_include("sdprof.hpp")
#include "sdprof.hpp"
#endif
#ifndef SD_PROF
#define SD_PROF(name)
#endif

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>

namespace sdgemm
{
namespace
{
using CMap = Eigen::Map<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>;
using MMap = Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<>>;

std::atomic<double> g_gpuMinMacs{24.0e6};

// Columns [n0, n0 + nc) of C.
void cpu_cols(bool ta, bool tb, int M, int K, float alpha, const float *A, int lda, const float *B, int ldb,
              float beta, float *C, int ldc, int N, int n0, int nc)
{
    MMap Cc(C + size_t(n0) * ldc, M, nc, Eigen::OuterStride<>(ldc));
    if (beta == 0.0f)
        Cc.setZero();
    else if (beta != 1.0f)
        Cc *= beta;
    if (!ta && !tb)
        Cc.noalias() += alpha * CMap(A, M, K, Eigen::OuterStride<>(lda)) *
                        CMap(B + size_t(n0) * ldb, K, nc, Eigen::OuterStride<>(ldb));
    else if (!ta && tb)
        Cc.noalias() += alpha * CMap(A, M, K, Eigen::OuterStride<>(lda)) *
                        CMap(B, N, K, Eigen::OuterStride<>(ldb)).middleRows(n0, nc).transpose();
    else if (ta && !tb)
        Cc.noalias() += alpha * CMap(A, K, M, Eigen::OuterStride<>(lda)).transpose() *
                        CMap(B + size_t(n0) * ldb, K, nc, Eigen::OuterStride<>(ldb));
    else
        Cc.noalias() += alpha * CMap(A, K, M, Eigen::OuterStride<>(lda)).transpose() *
                        CMap(B, N, K, Eigen::OuterStride<>(ldb)).middleRows(n0, nc).transpose();
}

// Rows [m0, m0 + mc) of C: the same with A's rows sliced.
void cpu_rows(bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
              float beta, float *C, int ldc, int m0, int mc)
{
    MMap Cc(C + m0, mc, N, Eigen::OuterStride<>(ldc));
    if (beta == 0.0f)
        Cc.setZero();
    else if (beta != 1.0f)
        Cc *= beta;
    if (!ta && !tb)
        Cc.noalias() += alpha * CMap(A + m0, mc, K, Eigen::OuterStride<>(lda)) *
                        CMap(B, K, N, Eigen::OuterStride<>(ldb));
    else if (!ta && tb)
        Cc.noalias() += alpha * CMap(A + m0, mc, K, Eigen::OuterStride<>(lda)) *
                        CMap(B, N, K, Eigen::OuterStride<>(ldb)).transpose();
    else if (ta && !tb)
        Cc.noalias() += alpha * CMap(A + size_t(m0) * lda, K, mc, Eigen::OuterStride<>(lda)).transpose() *
                        CMap(B, K, N, Eigen::OuterStride<>(ldb));
    else
        Cc.noalias() += alpha * CMap(A + size_t(m0) * lda, K, mc, Eigen::OuterStride<>(lda)).transpose() *
                        CMap(B, N, K, Eigen::OuterStride<>(ldb)).transpose();
}

bool gpu_gemm(bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
              float beta, float *C, int ldc, int flags)
{
    auto batch = sdgpu::Batch::try_begin();
    if (!batch)
        return false;
    SD_PROF("gpu: matrix product");
    const size_t nA = ta ? size_t(lda) * (M - 1) + K : size_t(lda) * (K - 1) + M;
    const size_t nB = tb ? size_t(ldb) * (K - 1) + N : size_t(ldb) * (N - 1) + K;
    const size_t nC = size_t(ldc) * (N - 1) + M;
    sdgpu::Buf a = (flags & kConstA) ? batch->weights(A, nA) : batch->input(A, nA);
    sdgpu::Buf b = (flags & kConstB) ? batch->weights(B, nB) : batch->input(B, nB);
    sdgpu::Buf c = beta != 0.0f ? batch->input(C, nC) : batch->scratch(nC);
    batch->gemm(ta, tb, M, N, K, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc);
    // Only the M x N block of C is written back, column by column when C
    // is a block of a larger matrix.
    if (ldc == M)
        batch->output(c, 0, C, size_t(M) * N);
    else
        for (int n = 0; n < N; ++n)
            batch->output(c, size_t(n) * ldc, C + size_t(n) * ldc, M);
    return batch->run();
}
} // namespace

void set_gpu_min_macs(double macs) { g_gpuMinMacs.store(macs); }

void gemm_cpu(bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
              float beta, float *C, int ldc)
{
    if (M <= 0 || N <= 0)
        return;
    const double macs = double(M) * N * K;
    const int threads = sdpool::threads();
    // Split into chunks of at least ~2M multiply-adds, about 4 per thread,
    // so faster cores can take more of them.
    int chunks = static_cast<int>(std::min<double>(4.0 * threads, macs / 2.0e6));
    if (chunks <= 1 || K == 0)
    {
        cpu_cols(ta, tb, M, K, alpha, A, lda, B, ldb, beta, C, ldc, N, 0, N);
        return;
    }
    if (N >= M)
    {
        chunks = std::min(chunks, std::max(1, N / 16));
        const int step = (N + chunks - 1) / chunks;
        sdpool::parallel_for((N + step - 1) / step, [&](int i) {
            const int n0 = i * step;
            cpu_cols(ta, tb, M, K, alpha, A, lda, B, ldb, beta, C, ldc, N, n0, std::min(step, N - n0));
        });
    }
    else
    {
        chunks = std::min(chunks, std::max(1, M / 16));
        const int step = (M + chunks - 1) / chunks;
        sdpool::parallel_for((M + step - 1) / step, [&](int i) {
            const int m0 = i * step;
            cpu_rows(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, m0, std::min(step, M - m0));
        });
    }
}

void gemm(bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
          float beta, float *C, int ldc, int flags)
{
    if (M <= 0 || N <= 0)
        return;
    if (K > 0 && double(M) * N * K >= g_gpuMinMacs.load() &&
        gpu_gemm(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, flags))
        return;
    gemm_cpu(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
} // namespace sdgemm
