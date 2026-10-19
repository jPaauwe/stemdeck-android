// StemDeck Android: fast axis permutations for column-major 3-D tensors.
//
// Every permutation the htdemucs code applies is one or a few 2-D
// transposes over the raw buffer; doing them as cache-blocked transposes is
// several times faster than Eigen's generic Tensor::shuffle.
#pragma once

#include "sdprof.hpp"
#include "tensor.hpp"

#include <array>
#include <unsupported/Eigen/CXX11/Tensor>

namespace demucscpp
{

// out(c, r) = in(r, c) for a rows x cols block, with leading dimensions.
inline void transpose_block(const float *in, Eigen::Index ldIn, float *out,
                            Eigen::Index ldOut, Eigen::Index rows, Eigen::Index cols)
{
    constexpr Eigen::Index B = 32;
    for (Eigen::Index c0 = 0; c0 < cols; c0 += B)
    {
        const Eigen::Index c1 = std::min(cols, c0 + B);
        for (Eigen::Index r0 = 0; r0 < rows; r0 += B)
        {
            const Eigen::Index r1 = std::min(rows, r0 + B);
            for (Eigen::Index r = r0; r < r1; ++r)
            {
                float *o = out + r * ldOut;
                for (Eigen::Index c = c0; c < c1; ++c)
                    o[c] = in[r + c * ldIn];
            }
        }
    }
}

// Same result as x.shuffle(perm) for the permutations htdemucs uses.
inline Eigen::Tensor3dXf permute3(const Eigen::Tensor3dXf &x, std::array<int, 3> perm)
{
    SD_PROF("layout permutes");
    const Eigen::Index A = x.dimension(0), B = x.dimension(1), C = x.dimension(2);
    Eigen::Tensor3dXf y(x.dimension(perm[0]), x.dimension(perm[1]), x.dimension(perm[2]));
    const float *in = x.data();
    float *out = y.data();
    if (perm == std::array<int, 3>{1, 2, 0})
    {
        // (A, B*C) -> (B*C, A)
        transpose_block(in, A, out, B * C, A, B * C);
    }
    else if (perm == std::array<int, 3>{2, 0, 1})
    {
        // (A*B, C) -> (C, A*B)
        transpose_block(in, A * B, out, C, A * B, C);
    }
    else if (perm == std::array<int, 3>{1, 0, 2})
    {
        for (Eigen::Index c = 0; c < C; ++c)
            transpose_block(in + c * A * B, A, out + c * A * B, B, A, B);
    }
    else if (perm == std::array<int, 3>{2, 1, 0})
    {
        // for each b: (a, c) at a + A*b + A*B*c  ->  (c, a) at c + C*b + C*B*a
        for (Eigen::Index b = 0; b < B; ++b)
            transpose_block(in + A * b, A * B, out + C * b, C * B, A, C);
    }
    else if (perm == std::array<int, 3>{0, 2, 1})
    {
        // (a, b, c) -> (a, c, b): move contiguous runs of A
        for (Eigen::Index c = 0; c < C; ++c)
            for (Eigen::Index b = 0; b < B; ++b)
                std::copy(in + A * (b + B * c), in + A * (b + B * c) + A,
                          out + A * (c + C * b));
    }
    else if (perm == std::array<int, 3>{0, 1, 2})
    {
        std::copy(in, in + x.size(), out);
    }
    else
    {
        y = x.shuffle(perm);
    }
    return y;
}

} // namespace demucscpp
