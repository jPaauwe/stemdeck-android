// Checks the GPU path against the CPU: matrix products of many shapes
// (transposes, offsets, beta) and multi-head attention. Exit code 0 when
// everything matches, 1 on a mismatch, 2 when there is no GPU.
//
// On a computer without a GPU, set SD_GPU_ALLOW_CPU=1 to test on a software
// Vulkan driver (llvmpipe).
#include "gemm.hpp"
#include "gpu.hpp"
#include "layers.hpp"
#include "pool.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double max_rel(const float *a, const float *b, size_t n)
{
    double scale = 1e-6, err = 0;
    for (size_t i = 0; i < n; ++i)
        scale = std::max(scale, double(std::fabs(a[i])));
    for (size_t i = 0; i < n; ++i)
        err = std::max(err, double(std::fabs(a[i] - b[i])));
    return err / scale;
}

int main()
{
    std::printf("%s\n", sdgpu::info().c_str());
    if (!sdgpu::available())
        return 2;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    bool ok = true;

    struct Case { bool ta, tb; int M, N, K; float alpha, beta; int padA, padB, padC; };
    const Case cases[] = {
        {false, false, 64, 64, 16, 1, 0, 0, 0, 0},
        {false, false, 100, 37, 53, 1, 0, 3, 5, 7},
        {false, true, 2688, 384, 384, 1, 0, 0, 0, 0},
        {true, false, 300, 48, 280, 1, 0, 0, 0, 0},
        {false, true, 280, 300, 48, 1, 0, 0, 0, 0},
        {true, true, 65, 129, 77, 0.5f, 0.25f, 2, 1, 3},
        {false, false, 48, 20000, 48, 1, 1, 0, 0, 0},
        {false, false, 1, 1, 1, 1, 0, 0, 0, 0},
        {false, false, 192, 4000, 96, 1, 1, 0, 0, 11},
    };
    for (const Case &c : cases)
    {
        const int lda = (c.ta ? c.K : c.M) + c.padA;
        const int ldb = (c.tb ? c.N : c.K) + c.padB;
        const int ldc = c.M + c.padC;
        std::vector<float> A(size_t(lda) * (c.ta ? c.M : c.K)), B(size_t(ldb) * (c.tb ? c.K : c.N)),
            C0(size_t(ldc) * c.N);
        for (auto &v : A) v = u(rng);
        for (auto &v : B) v = u(rng);
        for (auto &v : C0) v = u(rng);
        std::vector<float> Ccpu = C0, Cgpu = C0;
        sdgemm::gemm_cpu(c.ta, c.tb, c.M, c.N, c.K, c.alpha, A.data(), lda, B.data(), ldb, c.beta, Ccpu.data(), ldc);
        auto g = sdgpu::Batch::begin();
        auto a = g->input(A.data(), A.size()), b = g->input(B.data(), B.size()), cc = g->input(C0.data(), C0.size());
        g->gemm(c.ta, c.tb, c.M, c.N, c.K, c.alpha, a, 0, lda, b, 0, ldb, c.beta, cc, 0, ldc);
        g->output(cc, 0, Cgpu.data(), Cgpu.size());
        const auto t0 = std::chrono::steady_clock::now();
        const bool ran = g->run();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        g.reset();
        const double e = ran ? max_rel(Ccpu.data(), Cgpu.data(), Ccpu.size()) : 1e9;
        const bool pass = ran && e < 1e-4;
        ok = ok && pass;
        std::printf("gemm %c%c M=%d N=%d K=%d alpha=%.2f beta=%.2f: rel err %.2e, %.1f ms  %s\n", c.ta ? 'T' : 'N',
                    c.tb ? 'T' : 'N', c.M, c.N, c.K, c.alpha, c.beta, e, ms, pass ? "ok" : "FAIL");
    }

    // Attention: GPU against CPU.
    for (int T : {300, 2688})
    {
        const int S = T == 300 ? 280 : 2688, C = 384, heads = 8;
        Eigen::MatrixXf Q = Eigen::MatrixXf::NullaryExpr(T, C, [&]() { return 0.3f * u(rng); });
        Eigen::MatrixXf K = Eigen::MatrixXf::NullaryExpr(S, C, [&]() { return 0.3f * u(rng); });
        Eigen::MatrixXf V = Eigen::MatrixXf::NullaryExpr(S, C, [&]() { return u(rng); });
        Eigen::MatrixXf og, oc;
        const auto t0 = std::chrono::steady_clock::now();
        demucscpp::attention_heads(Q, K, V, heads, og);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        sdgpu::set_enabled(false);
        const auto t1 = std::chrono::steady_clock::now();
        demucscpp::attention_heads(Q, K, V, heads, oc);
        const double msc = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        sdgpu::set_enabled(true);
        const double e = max_rel(oc.data(), og.data(), size_t(oc.size()));
        const bool pass = e < 1e-4;
        ok = ok && pass;
        std::printf("attention T=%d S=%d C=%d heads=%d: rel err %.2e, GPU %.0f ms, CPU %.0f ms  %s\n", T, S, C, heads,
                    e, ms, msc, pass ? "ok" : "FAIL");
    }
    std::printf(ok ? "all ok\n" : "MISMATCH\n");
    return ok ? 0 : 1;
}
