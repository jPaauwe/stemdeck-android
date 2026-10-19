#ifndef CONV_HPP
#define CONV_HPP

#include "gemm.hpp"
#include "gpu.hpp"
#include "pool.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Dense>
#include "sdprof.hpp"
#include "permute.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>

namespace demucscpp
{

template <int kernel_height, int kernel_width, int stride_height,
          int stride_width, int pad_height, int pad_width, int dilation_height,
          int dilation_width>
inline Eigen::MatrixXf im2col(const Eigen::Tensor3dXf &input)
{
    // Adjust the calculation of height_col and width_col for dilation
    int in_channels = input.dimension(0);
    int in_height = input.dimension(1);
    int in_width = input.dimension(2);

    // Apply floating point division before ceiling to correctly calculate
    // dimensions
    int height_col =
        static_cast<int>(std::ceil((in_height + 2 * pad_height -
                                    dilation_height * (kernel_height - 1) - 1) /
                                   float(stride_height)) +
                         1);
    int width_col =
        static_cast<int>(std::ceil((in_width + 2 * pad_width -
                                    dilation_width * (kernel_width - 1) - 1) /
                                   float(stride_width)) +
                         1);

    Eigen::MatrixXf output(height_col * width_col,
                           in_channels * kernel_height * kernel_width);
    output.setZero();

    for (int c = 0; c < in_channels; c++)
    {
        for (int kh = 0; kh < kernel_height; kh++)
        {
            for (int kw = 0; kw < kernel_width; kw++)
            {
                for (int h = 0; h < height_col; h++)
                {
                    for (int w = 0; w < width_col; w++)
                    {
                        int h_pad = h * stride_height + kh * dilation_height -
                                    pad_height;
                        int w_pad =
                            w * stride_width + kw * dilation_width - pad_width;
                        if (h_pad >= 0 && h_pad < in_height && w_pad >= 0 &&
                            w_pad < in_width)
                        {
                            output(h * width_col + w,
                                   c * kernel_height * kernel_width +
                                       kh * kernel_width + kw) =
                                input(c, h_pad, w_pad);
                        }
                    }
                }
            }
        }
    }

    return output;
}

// StemDeck Android: stride-1 convolution (the decoders' 3x3 "rewrite" convs
// and the 1x1 convs) as one GEMM per kernel tap, shifted into the output.
// The upstream im2col for the last frequency decoder's 3x3 conv was a
// (512*336) x (48*9) matrix, ~300 MB per inference thread; this needs only
// one (Cout, H*W) product at a time. Strided convs keep im2col, whose matrix
// is small and whose zero rows would otherwise cost extra multiply-adds.
template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int pad_height, int pad_width, int dilation_height,
          int dilation_width>
Eigen::Tensor3dXf conv2d_stride1_scatter(const Eigen::Tensor3dXf &x,
                                      const Eigen::Tensor4dXf &w,
                                      const Eigen::Tensor1dXf &b,
                                      int out_height, int out_width)
{
    SD_PROF("conv stride 1 (per-tap scatter, fallback)");
    const int in_height = x.dimension(1);
    const int in_width = x.dimension(2);
    Eigen::Map<const Eigen::MatrixXf> X(x.data(), in_channels,
                                        (Eigen::Index)in_height * in_width);

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    float *y = y_out.data();
    {
        const Eigen::Index n = (Eigen::Index)out_height * out_width;
        for (Eigen::Index i = 0; i < n; ++i)
            for (int o = 0; o < out_channels; ++o)
                y[i * out_channels + o] = b(o);
    }

    Eigen::MatrixXf Wk(out_channels, in_channels);
    Eigen::MatrixXf P;
    for (int kh = 0; kh < kernel_height; ++kh)
    {
        for (int kw = 0; kw < kernel_width; ++kw)
        {
            for (int o = 0; o < out_channels; ++o)
                for (int c = 0; c < in_channels; ++c)
                    Wk(o, c) = w(o, c, kh, kw);
            P.resize(out_channels, X.cols()); // contribution of every input position
            sdgemm::gemm(false, false, out_channels, int(X.cols()), in_channels, 1.0f, Wk.data(),
                         out_channels, X.data(), in_channels, 0.0f, P.data(), out_channels);

            // input (ih, iw) feeds output (ih - kh*dh + ph, iw - kw*dw + pw)
            for (int ow = 0; ow < out_width; ++ow)
            {
                const int iw = ow + kw * dilation_width - pad_width;
                if (iw < 0 || iw >= in_width)
                    continue;
                for (int oh = 0; oh < out_height; ++oh)
                {
                    const int ih = oh + kh * dilation_height - pad_height;
                    if (ih < 0 || ih >= in_height)
                        continue;
                    const float *src =
                        P.data() + (ih + (Eigen::Index)in_height * iw) * out_channels;
                    float *dst =
                        y + (oh + (Eigen::Index)out_height * ow) * out_channels;
                    for (int o = 0; o < out_channels; ++o)
                        dst[o] += src[o];
                }
            }
        }
    }
    return y_out;
}

// StemDeck Android: stride-1 convolution as one GEMM per kernel tap on the
// flattened (h, w) positions, accumulating straight into the output.
//
// With "same" padding (out_height == in_height), tap (kh, kw) reads input
// position p + delta for output position p, delta = dh + H*dw. One GEMM
// over all output positions whose w is valid covers the tap, except the
// few rows whose h + dh falls outside [0, H): there the flat index wraps
// into the neighbouring column, so those columns are gathered and their
// contribution subtracted again.
template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int pad_height, int pad_width, int dilation_height,
          int dilation_width>
Eigen::Tensor3dXf conv2d_stride1_taps(const Eigen::Tensor3dXf &x,
                                      const Eigen::Tensor4dXf &w,
                                      const Eigen::Tensor1dXf &b,
                                      int out_height, int out_width)
{
    const int H = x.dimension(1);
    const int W = x.dimension(2);
    if (out_height != H)
    {
        return conv2d_stride1_scatter<in_channels, out_channels, kernel_height,
                                      kernel_width, pad_height, pad_width,
                                      dilation_height, dilation_width>(
            x, w, b, out_height, out_width);
    }
    SD_PROF("conv stride 1 (per-tap GEMM)");
    const int OW = out_width;
    const Eigen::Index P = Eigen::Index(H) * W;
    const Eigen::Index OP = Eigen::Index(H) * OW;

    Eigen::Map<const Eigen::MatrixXf> X(x.data(), in_channels, P);
    Eigen::Tensor3dXf y_out(out_channels, H, OW);
    Eigen::Map<Eigen::MatrixXf> Y(y_out.data(), out_channels, OP);
    Eigen::Map<const Eigen::VectorXf> bias(b.data(), out_channels);
    Y.colwise() = bias;

    struct Tap
    {
        int dh, ow_lo, ow_hi;
        Eigen::Index delta, p_lo, p_hi;
        Eigen::MatrixXf Wk;
    };
    std::vector<Tap> taps;
    for (int kh = 0; kh < kernel_height; ++kh)
    {
        for (int kw = 0; kw < kernel_width; ++kw)
        {
            const int dh = kh * dilation_height - pad_height;
            const int dw = kw * dilation_width - pad_width;
            const int ow_lo = std::max(0, -dw);
            const int ow_hi = std::min(OW, W - dw);
            if (ow_lo >= ow_hi || std::abs(dh) >= H)
                continue;
            const Eigen::Index delta = dh + Eigen::Index(H) * dw;
            Eigen::Index p_lo = Eigen::Index(ow_lo) * H;
            Eigen::Index p_hi = Eigen::Index(ow_hi) * H;
            // keep the input index inside the array
            p_lo = std::max(p_lo, -delta);
            p_hi = std::min(p_hi, P - delta);
            if (p_lo >= p_hi)
                continue;
            Tap t{dh, ow_lo, ow_hi, delta, p_lo, p_hi, Eigen::MatrixXf(out_channels, in_channels)};
            for (int o = 0; o < out_channels; ++o)
                for (int c = 0; c < in_channels; ++c)
                    t.Wk(o, c) = w(o, c, kh, kw);
            taps.push_back(std::move(t));
        }
    }

    // 1. One product per tap, accumulated into Y: as one GPU batch when the
    //    GPU is free and the work is large enough, else on every core.
    bool done = false;
    double macs = 0;
    for (const Tap &t : taps)
        macs += double(out_channels) * in_channels * double(t.p_hi - t.p_lo);
    if (macs >= 24.0e6)
    {
        if (auto g = sdgpu::Batch::try_begin())
        {
            SD_PROF("gpu: conv stride 1");
            sdgpu::Buf xb = g->input(x.data(), size_t(in_channels) * P);
            sdgpu::Buf yb = g->input(Y.data(), size_t(out_channels) * OP);
            for (const Tap &t : taps)
            {
                sdgpu::Buf wb = g->input(t.Wk.data(), size_t(out_channels) * in_channels);
                g->gemm(false, false, out_channels, uint32_t(t.p_hi - t.p_lo), in_channels, 1.0f, wb, 0,
                        out_channels, xb, size_t(t.p_lo + t.delta) * in_channels, in_channels, 1.0f, yb,
                        size_t(t.p_lo) * out_channels, out_channels);
            }
            g->output(yb, 0, Y.data(), size_t(out_channels) * OP);
            done = g->run();
        }
    }
    if (!done)
        for (const Tap &t : taps)
            sdgemm::gemm_cpu(false, false, out_channels, int(t.p_hi - t.p_lo), in_channels, 1.0f, t.Wk.data(),
                             out_channels, x.data() + size_t(t.p_lo + t.delta) * in_channels, in_channels, 1.0f,
                             Y.data() + size_t(t.p_lo) * out_channels, out_channels);

    // 2. Rows whose h + dh is outside [0, H) took a wrapped input: subtract
    //    those contributions again.
    Eigen::MatrixXf G, GW;
    std::vector<Eigen::Index> fix;
    for (const Tap &t : taps)
    {
        const int dh = t.dh;
        if (dh == 0)
            continue;
        fix.clear();
        for (int ow = t.ow_lo; ow < t.ow_hi; ++ow)
        {
            const int r0 = dh < 0 ? 0 : H - dh;
            const int r1 = dh < 0 ? -dh : H;
            for (int r = r0; r < r1; ++r)
            {
                const Eigen::Index p = r + Eigen::Index(H) * ow;
                if (p >= t.p_lo && p < t.p_hi)
                    fix.push_back(p);
            }
        }
        if (fix.empty())
            continue;
        G.resize(in_channels, Eigen::Index(fix.size()));
        for (size_t f = 0; f < fix.size(); ++f)
            G.col(f) = X.col(fix[f] + t.delta);
        GW.noalias() = t.Wk * G;
        for (size_t f = 0; f < fix.size(); ++f)
            Y.col(fix[f]) -= GW.col(f);
    }
    return y_out;
}

template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int stride_height, int stride_width, int pad_height,
          int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d(const Eigen::Tensor3dXf &x, const Eigen::Tensor4dXf &w,
                         const Eigen::Tensor1dXf &b)
{
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    // Calculate output dimensions with the correct application of ceil
    int out_height =
        static_cast<int>(std::ceil(
            (float)(in_height + 2 * pad_height - (kernel_height - 1) - 1) /
            stride_height)) +
        1;
    int out_width =
        static_cast<int>(std::ceil(
            (float)(in_width + 2 * pad_width - (kernel_width - 1) - 1) /
            stride_width)) +
        1;

    if constexpr (stride_height == 1 && stride_width == 1)
    {
        return conv2d_stride1_taps<in_channels, out_channels, kernel_height,
                                   kernel_width, pad_height, pad_width,
                                   dilation_height, dilation_width>(
            x, w, b, out_height, out_width);
    }
    SD_PROF("conv strided (im2col)");

    // Apply im2col
    Eigen::MatrixXf im2col_matrix =
        im2col<kernel_height, kernel_width, stride_height, stride_width,
               pad_height, pad_width, dilation_height, dilation_width>(x);

    // Reshape weights
    // reverse last 3 axes (out chanel x in chan x kernel height x kernel width
    // -> out chan x (kernel width x kernel height x in chan))
    Eigen::Tensor4dXf w_swapped = w.shuffle(Eigen::array<int, 4>({0, 3, 2, 1}));
    // then flatten to the last axis
    Eigen::Tensor2dXf reshaped_weights_tensor =
        w_swapped.reshape(Eigen::array<int, 2>{
            out_channels, in_channels * kernel_width * kernel_height});
    Eigen::MatrixXf reshaped_weights = Eigen::Map<Eigen::MatrixXf>(
        reshaped_weights_tensor.data(), reshaped_weights_tensor.dimension(0),
        reshaped_weights_tensor.dimension(1));

    // Perform matrix multiplication with GEMM
    Eigen::MatrixXf result(im2col_matrix.rows(), out_channels);
    sdgemm::gemm(false, true, int(im2col_matrix.rows()), out_channels, int(im2col_matrix.cols()), 1.0f,
                 im2col_matrix.data(), int(im2col_matrix.rows()), reshaped_weights.data(), out_channels, 0.0f,
                 result.data(), int(im2col_matrix.rows()));

    // Add bias to each column of the result matrix
    for (int chout = 0; chout < out_channels; ++chout)
    {
        result.col(chout).array() += b(chout);
    }

    // Reshape result to 3D output tensor
    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    y_out.setZero();

    for (int chout = 0; chout < out_channels; ++chout)
    {
        for (int h = 0; h < out_height; ++h)
        {
            for (int w_ = 0; w_ < out_width; ++w_)
            {
                int row_idx = h * out_width + w_;
                // Assign the value from the GEMM output to the output tensor
                if (row_idx < result.rows())
                {
                    y_out(chout, h, w_) = result(row_idx, chout);
                }
            }
        }
    }

    return y_out;
}

template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int stride_height, int stride_width, int pad_height,
          int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d_fused_gelu(const Eigen::Tensor3dXf &x,
                                    const Eigen::Tensor4dXf &w,
                                    const Eigen::Tensor1dXf &b)
{
    SD_PROF("conv strided + gelu (im2col)");
    int in_height = x.dimension(1);
    int in_width = x.dimension(2);

    // Calculate output dimensions with the correct application of ceil
    int out_height =
        static_cast<int>(std::ceil(
            (float)(in_height + 2 * pad_height - (kernel_height - 1) - 1) /
            stride_height)) +
        1;
    int out_width =
        static_cast<int>(std::ceil(
            (float)(in_width + 2 * pad_width - (kernel_width - 1) - 1) /
            stride_width)) +
        1;

    // Apply im2col
    Eigen::MatrixXf im2col_matrix =
        im2col<kernel_height, kernel_width, stride_height, stride_width,
               pad_height, pad_width, dilation_height, dilation_width>(x);

    // Reshape weights
    // reverse last 3 axes (out chanel x in chan x kernel height x kernel width
    // -> out chan x (kernel width x kernel height x in chan))
    Eigen::Tensor4dXf w_swapped = w.shuffle(Eigen::array<int, 4>({0, 3, 2, 1}));
    // then flatten to the last axis
    Eigen::Tensor2dXf reshaped_weights_tensor =
        w_swapped.reshape(Eigen::array<int, 2>{
            out_channels, in_channels * kernel_width * kernel_height});
    Eigen::MatrixXf reshaped_weights = Eigen::Map<Eigen::MatrixXf>(
        reshaped_weights_tensor.data(), reshaped_weights_tensor.dimension(0),
        reshaped_weights_tensor.dimension(1));

    // Perform matrix multiplication with GEMM
    Eigen::MatrixXf result(im2col_matrix.rows(), out_channels);
    sdgemm::gemm(false, true, int(im2col_matrix.rows()), out_channels, int(im2col_matrix.cols()), 1.0f,
                 im2col_matrix.data(), int(im2col_matrix.rows()), reshaped_weights.data(), out_channels, 0.0f,
                 result.data(), int(im2col_matrix.rows()));

    // Add bias to each column of the result matrix
    for (int chout = 0; chout < out_channels; ++chout)
    {
        result.col(chout).array() += b(chout);
    }

    // Reshape result to 3D output tensor
    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    y_out.setZero();

    for (int chout = 0; chout < out_channels; ++chout)
    {
        for (int h = 0; h < out_height; ++h)
        {
            for (int w_ = 0; w_ < out_width; ++w_)
            {
                int row_idx = h * out_width + w_;
                // Assign the value from the GEMM output to the output tensor
                // with gelu
                float value = result(row_idx, chout);
                float activated_value =
                    0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
                // Assign the activated value to the output tensor
                y_out(chout, h, w_) = activated_value;
            }
        }
    }

    return y_out;
}

template <int in_channels, int out_channels, int kernel_size, int stride,
          int pad, int dilation>
Eigen::Tensor3dXf conv1d(const Eigen::Tensor3dXf &x, const Eigen::Tensor3dXf &w,
                         const Eigen::Tensor1dXf &b)
{
    // copy w into a 4d tensor with trailing (,1) dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = demucscpp::permute3(x, {1, 2, 0});

    // do 2d convolution inference here
    // treating the in_freq dimension as a width dimension with a no-op kernel
    Eigen::Tensor3dXf y_out =
        demucscpp::conv2d<in_channels, out_channels, kernel_size, 1, stride, 1,
                          pad, 0, dilation, 1>(x_shuff, w_4d, b);

    // move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        demucscpp::permute3(y_out, {2, 0, 1});
    return y_out_shuf;
}

template <int in_channels, int out_channels, int kernel_size, int stride,
          int pad, int dilation>
Eigen::Tensor3dXf conv1d_fused_gelu(const Eigen::Tensor3dXf &x,
                                    const Eigen::Tensor3dXf &w,
                                    const Eigen::Tensor1dXf &b)
{
    // copy w into a 4d tensor with trailing (,1) dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = demucscpp::permute3(x, {1, 2, 0});

    // do 2d convolution inference here
    // treating the in_freq dimension as a width dimension with a no-op kernel
    Eigen::Tensor3dXf y_out =
        demucscpp::conv2d_fused_gelu<in_channels, out_channels, kernel_size, 1,
                                     stride, 1, pad, 0, dilation, 1>(x_shuff,
                                                                     w_4d, b);

    // move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        demucscpp::permute3(y_out, {2, 0, 1});
    return y_out_shuf;
}

template <int kernel_height, int kernel_width, int stride_height,
          int stride_width, int pad_height, int pad_width, int dilation_height,
          int dilation_width>
Eigen::MatrixXf im2col_transposed(const Eigen::Tensor3dXf &input)
{
    int channels = input.dimension(0);
    int input_height = input.dimension(1);
    int input_width = input.dimension(2);

    // Calculate the effective kernel size after dilation
    int effective_kernel_height =
        kernel_height + (kernel_height - 1) * (dilation_height - 1);
    int effective_kernel_width =
        kernel_width + (kernel_width - 1) * (dilation_width - 1);

    // Calculate the expanded output height and width considering dilation
    int expanded_height =
        (input_height - 1) * stride_height + effective_kernel_height;
    int expanded_width =
        (input_width - 1) * stride_width + effective_kernel_width;

    // Initialize the output matrix
    Eigen::MatrixXf output =
        Eigen::MatrixXf::Zero(expanded_height * expanded_width,
                              channels * kernel_height * kernel_width);

    // Populate the output matrix
    for (int c = 0; c < channels; ++c)
    {
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            for (int kw = 0; kw < kernel_width; ++kw)
            {
                for (int h = 0; h < input_height; ++h)
                {
                    for (int w = 0; w < input_width; ++w)
                    {
                        // Adjust calculation for dilation
                        int expanded_h = h * stride_height +
                                         (kh * dilation_height) - pad_height;
                        int expanded_w = w * stride_width +
                                         (kw * dilation_width) - pad_width;

                        // Check if the indices are within the bounds of the
                        // expanded output
                        if (expanded_h >= 0 && expanded_h < expanded_height &&
                            expanded_w >= 0 && expanded_w < expanded_width)
                        {
                            int col_index =
                                expanded_h * expanded_width + expanded_w;
                            int row_index = c * kernel_height * kernel_width +
                                            kh * kernel_width + kw;
                            output(col_index, row_index) = input(c, h, w);
                        }
                    }
                }
            }
        }
    }

    return output;
}

// StemDeck Android: transposed convolution without im2col.
//
// The upstream version built an im2col matrix of (expanded output size) x
// (Cin * Kh * Kw), three quarters zeros for stride 4: about 1 GB for the last
// frequency decoder of one 7.8 s segment, and 0.5 GB for the last time
// decoder. That alone put one inference thread at ~1.6 GB, too much for a
// phone running several. This computes the same sum as one small GEMM per
// kernel tap, scattered into the output: no zero padding, a quarter of the
// multiply-adds, and memory proportional to the input and output only.
template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int stride_height, int stride_width, int pad_height,
          int pad_width, int dilation_height, int dilation_width,
          bool fused_gelu>
Eigen::Tensor3dXf conv2d_tr_scatter(const Eigen::Tensor3dXf &x,
                                    const Eigen::Tensor4dXf &w,
                                    const Eigen::Tensor1dXf &b)
{
    SD_PROF(fused_gelu ? "conv transposed + gelu" : "conv transposed");
    const int in_height = x.dimension(1);
    const int in_width = x.dimension(2);

    const int effective_kernel_height =
        kernel_height + (kernel_height - 1) * (dilation_height - 1);
    const int effective_kernel_width =
        kernel_width + (kernel_width - 1) * (dilation_width - 1);

    const int out_height = (in_height - 1) * stride_height +
                           effective_kernel_height - 2 * pad_height;
    const int out_width = (in_width - 1) * stride_width +
                          effective_kernel_width - 2 * pad_width;

    // Tensors are column-major: x(c, h, w) sits at c + Cin * (h + H * w), so
    // the input is a (Cin, H*W) matrix as it is.
    Eigen::Map<const Eigen::MatrixXf> X(x.data(), in_channels,
                                        (Eigen::Index)in_height * in_width);

    Eigen::Tensor3dXf y_out(out_channels, out_height, out_width);
    y_out.setZero();
    float *y = y_out.data();

    Eigen::MatrixXf Wk(out_channels, in_channels);
    Eigen::MatrixXf P;
    for (int kh = 0; kh < kernel_height; ++kh)
    {
        for (int kw = 0; kw < kernel_width; ++kw)
        {
            // transposed-conv weights are stored (Cin, Cout, Kh, Kw)
            for (int c = 0; c < in_channels; ++c)
                for (int o = 0; o < out_channels; ++o)
                    Wk(o, c) = w(c, o, kh, kw);

            P.resize(out_channels, X.cols()); // (Cout, H*W)
            sdgemm::gemm(false, false, out_channels, int(X.cols()), in_channels, 1.0f, Wk.data(),
                         out_channels, X.data(), in_channels, 0.0f, P.data(), out_channels);

            // Within one tap, each input column lands in a different output
            // column, so the columns can be scattered in parallel.
            sdpool::parallel_for(in_width, [&](int iw) {
                const int ow = iw * stride_width + kw * dilation_width - pad_width;
                if (ow < 0 || ow >= out_width)
                    return;
                for (int ih = 0; ih < in_height; ++ih)
                {
                    const int oh =
                        ih * stride_height + kh * dilation_height - pad_height;
                    if (oh < 0 || oh >= out_height)
                        continue;
                    const float *src =
                        P.data() + (Eigen::Index)(ih + (Eigen::Index)in_height * iw) * out_channels;
                    float *dst =
                        y + (Eigen::Index)(oh + (Eigen::Index)out_height * ow) * out_channels;
                    for (int o = 0; o < out_channels; ++o)
                        dst[o] += src[o];
                }
            });
        }
    }

    const Eigen::Index n = (Eigen::Index)out_height * out_width;
    const Eigen::Index chunk = 4096;
    sdpool::parallel_for(int((n + chunk - 1) / chunk), [&](int t) {
        const Eigen::Index i1 = std::min(n, (t + 1) * chunk);
        for (Eigen::Index i = t * chunk; i < i1; ++i)
        {
            float *v = y + i * out_channels;
            for (int o = 0; o < out_channels; ++o)
            {
                float value = v[o] + b(o);
                if (fused_gelu)
                    value = 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
                v[o] = value;
            }
        }
    });
    return y_out;
}

template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int stride_height, int stride_width, int pad_height,
          int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d_tr(const Eigen::Tensor3dXf &x,
                            const Eigen::Tensor4dXf &w,
                            const Eigen::Tensor1dXf &b)
{
    return conv2d_tr_scatter<in_channels, out_channels, kernel_height,
                             kernel_width, stride_height, stride_width,
                             pad_height, pad_width, dilation_height,
                             dilation_width, false>(x, w, b);
}

template <int in_channels, int out_channels, int kernel_height,
          int kernel_width, int stride_height, int stride_width, int pad_height,
          int pad_width, int dilation_height, int dilation_width>
Eigen::Tensor3dXf conv2d_tr_fused_gelu(const Eigen::Tensor3dXf &x,
                                       const Eigen::Tensor4dXf &w,
                                       const Eigen::Tensor1dXf &b)
{
    return conv2d_tr_scatter<in_channels, out_channels, kernel_height,
                             kernel_width, stride_height, stride_width,
                             pad_height, pad_width, dilation_height,
                             dilation_width, true>(x, w, b);
}

template <int in_channels, int out_channels, int kernel_size, int stride,
          int pad, int dilation>
Eigen::Tensor3dXf conv1d_tr(const Eigen::Tensor3dXf &x,
                            const Eigen::Tensor3dXf &w,
                            const Eigen::Tensor1dXf &b)
{
    // Convert 1D convolution to 2D convolution by adding an extra dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // Move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = demucscpp::permute3(x, {1, 2, 0});

    // Call the 2D transposed convolution function
    Eigen::Tensor3dXf y_out =
        conv2d_tr<in_channels, out_channels, kernel_size, 1, stride, 1, pad, 0,
                  dilation, 1>(x_shuff, w_4d, b);

    // Move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        demucscpp::permute3(y_out, {2, 0, 1});

    return y_out_shuf;
}

template <int in_channels, int out_channels, int kernel_size, int stride,
          int pad, int dilation>
Eigen::Tensor3dXf conv1d_tr_fused_gelu(const Eigen::Tensor3dXf &x,
                                       const Eigen::Tensor3dXf &w,
                                       const Eigen::Tensor1dXf &b)
{
    // Convert 1D convolution to 2D convolution by adding an extra dimension
    Eigen::Tensor4dXf w_4d = w.reshape(Eigen::array<int, 4>{
        {(int)w.dimension(0), (int)w.dimension(1), (int)w.dimension(2), 1}});

    // Move 0 axis to the end
    Eigen::Tensor3dXf x_shuff = demucscpp::permute3(x, {1, 2, 0});

    // Call the 2D transposed convolution function
    Eigen::Tensor3dXf y_out =
        conv2d_tr_fused_gelu<in_channels, out_channels, kernel_size, 1, stride,
                             1, pad, 0, dilation, 1>(x_shuff, w_4d, b);

    // Move end axis to the front
    Eigen::Tensor3dXf y_out_shuf =
        demucscpp::permute3(y_out, {2, 0, 1});

    return y_out_shuf;
}

} // namespace demucscpp

#endif // CONV_HPP
