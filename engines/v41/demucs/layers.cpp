#include "permute.hpp"
#include "layers.hpp"
#include "conv.hpp"
#include "lstm.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include "gemm.hpp"
#include "gpu.hpp"
#include "pool.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// StemDeck Android: the norms, glu and layer scale below are rewritten as
// straight passes over the column-major data (index i + F*(c + C*w)), with
// the per-element divide folded into one multiply-add. Same maths as
// upstream (two-pass mean and unbiased variance), different rounding only.
Eigen::Tensor3dXf demucscpp::group_norm(const Eigen::Tensor3dXf &x,
                                        const Eigen::Tensor1dXf &weight,
                                        const Eigen::Tensor1dXf &b,
                                        int num_groups, float eps)
{
    SD_PROF("group norm");
    const int F = x.dimension(0);
    const int C = x.dimension(1);
    const int W = x.dimension(2);
    const int G = num_groups;
    const int gs = C / G;
    const float *xp = x.data();

    // mean per (i, g), then variance: partial sums over chunks of w, one
    // chunk per task (all cores), added up in chunk order.
    const int chunks = std::max(1, std::min(W, 4 * sdpool::threads()));
    const int step = (W + chunks - 1) / chunks;
    const int tasks = (W + step - 1) / step;
    std::vector<std::vector<double>> part(tasks);
    auto accumulate = [&](bool second, const std::vector<double> *meanp) {
        sdpool::parallel_for(tasks, [&](int t) {
            std::vector<double> &acc = part[t];
            acc.assign(size_t(F) * G, 0.0);
            const int w1 = std::min(W, (t + 1) * step);
            for (int w = t * step; w < w1; ++w)
                for (int c = 0; c < C; ++c)
                {
                    const float *col = xp + size_t(F) * (c + size_t(C) * w);
                    double *a = acc.data() + size_t(F) * (c / gs);
                    if (!second)
                        for (int i = 0; i < F; ++i)
                            a[i] += col[i];
                    else
                    {
                        const double *m = meanp->data() + size_t(F) * (c / gs);
                        for (int i = 0; i < F; ++i)
                        {
                            const double d = col[i] - m[i];
                            a[i] += d * d;
                        }
                    }
                }
        });
    };
    std::vector<double> mean(size_t(F) * G, 0.0), var(size_t(F) * G, 0.0);
    accumulate(false, nullptr);
    for (auto &pt : part)
        for (size_t k = 0; k < mean.size(); ++k)
            mean[k] += pt[k];
    const double n = double(gs) * W;
    for (double &m : mean)
        m /= n;
    accumulate(true, &mean);
    for (auto &pt : part)
        for (size_t k = 0; k < var.size(); ++k)
            var[k] += pt[k];
    std::vector<float> inv(size_t(F) * G), mf(size_t(F) * G);
    for (size_t k = 0; k < inv.size(); ++k)
    {
        inv[k] = float(1.0 / std::sqrt(var[k] / (n - 1) + eps));
        mf[k] = float(mean[k]);
    }

    Eigen::Tensor3dXf y_out(F, C, W);
    float *yp = y_out.data();
    sdpool::parallel_for(tasks, [&](int t) {
        const int w1 = std::min(W, (t + 1) * step);
        for (int w = t * step; w < w1; ++w)
            for (int c = 0; c < C; ++c)
            {
                const size_t off = size_t(F) * (c + size_t(C) * w);
                const float *col = xp + off;
                float *out = yp + off;
                const float *iv = inv.data() + size_t(F) * (c / gs);
                const float *m = mf.data() + size_t(F) * (c / gs);
                const float wc = weight(c), bc = b(c);
                for (int i = 0; i < F; ++i)
                    out[i] = (col[i] - m[i]) * (iv[i] * wc) + bc;
            }
    });
    return y_out;
}

Eigen::Tensor3dXf
demucscpp::group_norm_fused_gelu(const Eigen::Tensor3dXf &x,
                                 const Eigen::Tensor1dXf &weight,
                                 const Eigen::Tensor1dXf &bias, float eps)
{
    SD_PROF("group norm + gelu");
    Eigen::Tensor3dXf y = demucscpp::group_norm(x, weight, bias, 1, eps);
    Eigen::Map<Eigen::ArrayXf> a(y.data(), y.size());
    a = 0.5f * a * (1.0f + (a * 0.70710678118654752f).erf());
    return y;
}

Eigen::Tensor3dXf demucscpp::glu(const Eigen::Tensor3dXf &x, const int dim)
{
    SD_PROF("glu");
    if (x.dimension(dim) % 2 != 0)
    {
        std::cerr << "Dimension size must be evenly divisible by 2"
                  << std::endl;
        std::exit(1);
    }
    const int F = x.dimension(0), C = x.dimension(1), W = x.dimension(2);
    Eigen::array<Eigen::Index, 3> dims = {F, C, W};
    dims[dim] /= 2;
    Eigen::Tensor3dXf y(dims);
    const float *xp = x.data();
    float *yp = y.data();

    // Each output run of `len` floats takes `a` from the first half and
    // the gate from `b` = a + half.
    auto run = [](const float *a, const float *g, float *o, Eigen::Index len) {
        Eigen::Map<const Eigen::ArrayXf> A(a, len), G(g, len);
        Eigen::Map<Eigen::ArrayXf> O(o, len);
        O = A / (1.0f + (-G).exp());
    };
    // Over w in chunks, on every core.
    const int tasks = std::max(1, std::min(W, 4 * sdpool::threads()));
    const int step = (W + tasks - 1) / tasks;
    if (dim == 0)
    {
        const int h = F / 2;
        sdpool::parallel_for((W + step - 1) / step, [&](int t) {
            for (int w = t * step; w < std::min(W, (t + 1) * step); ++w)
                for (int c = 0; c < C; ++c)
                {
                    const float *col = xp + size_t(F) * (c + size_t(C) * w);
                    run(col, col + h, yp + size_t(h) * (c + size_t(C) * w), h);
                }
        });
    }
    else if (dim == 1)
    {
        const int h = C / 2;
        sdpool::parallel_for((W + step - 1) / step, [&](int t) {
            for (int w = t * step; w < std::min(W, (t + 1) * step); ++w)
            {
                const float *blk = xp + size_t(F) * C * w;
                run(blk, blk + size_t(F) * h, yp + size_t(F) * h * w, Eigen::Index(F) * h);
            }
        });
    }
    else
    {
        const int h = W / 2;
        const Eigen::Index plane = Eigen::Index(F) * C;
        run(xp, xp + plane * h, yp, plane * h);
    }
    return y;
}

Eigen::Tensor3dXf demucscpp::layer_norm(const Eigen::Tensor3dXf &x,
                                        const Eigen::Tensor1dXf &weight,
                                        const Eigen::Tensor1dXf &bias,
                                        float eps)
{
    SD_PROF("layer norm");
    int freq = x.dimension(0);
    int channels = x.dimension(1);
    int width = x.dimension(2);

    Eigen::Tensor3dXf y_out(freq, channels, width);

    for (int i = 0; i < freq; ++i)
    {
        for (int c = 0; c < channels; ++c)
        {
            Eigen::Tensor1dXf slice = x.chip(i, 0).chip(c, 0);
            Eigen::Tensor<float, 0> mean_tensor = slice.mean();
            float mean = mean_tensor(0);
            float var = demucscpp::calculate_variance(slice, mean);

            for (int w = 0; w < width; ++w)
            {
                float norm_val = (x(i, c, w) - mean) / std::sqrt(var + eps);
                y_out(i, c, w) = norm_val * weight(w) + bias(w);
            }
        }
    }

    return y_out;
}

void demucscpp::apply_dconv(const struct demucscpp::demucs_model &model,
                            Eigen::Tensor3dXf &y, int freq_idx, int encdec_idx,
                            int layer_idx, int mid_crop)
{
    // store another copy of y to sum back later
    Eigen::Tensor3dXf y_copy = y;

    // now dconv time

    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 6, 3, 1, 1, 1>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 12, 3, 1, 1, 1>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 24, 3, 1, 1, 1>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 48, 3, 1, 1, 1>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    };

    y = demucscpp::group_norm_fused_gelu(
        y,
        model.dconv_layers_1_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [0],
        model.dconv_layers_1_groupnorm_bias[freq_idx][encdec_idx][layer_idx][0],
        1e-05);

    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<6, 96, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 1:
        y = demucscpp::conv1d<12, 192, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 2:
        y = demucscpp::conv1d<24, 384, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    case 3:
        y = demucscpp::conv1d<48, 768, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [0],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [0]);
        break;
    };

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_4_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [0],
        model.dconv_layers_4_groupnorm_bias[freq_idx][encdec_idx][layer_idx][0],
        1, 1e-05);

    y = demucscpp::glu(y, 1);

    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][encdec_idx][layer_idx][0]);

    // now we add y to itself
    y = y + y_copy;

    // store another copy of y to sum back later
    y_copy = y;

    // NEXT ENTIRE SUBSEQUENCE OF DCONV WITH SLIGHTLY DIFFERENT PARAMS

    // Conv1d(48, 6, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 6, 3, 1, 2, 2>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 12, 3, 1, 2, 2>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 24, 3, 1, 2, 2>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 48, 3, 1, 2, 2>(
            y,
            model.dconv_layers_0_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_0_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    };

    Eigen::Tensor3dXf y_cropped =
        y.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                Eigen::array<Eigen::Index, 3>(
                    {y.dimension(0), y.dimension(1), mid_crop}));

    y = y_cropped;

    y = demucscpp::group_norm_fused_gelu(
        y,
        model.dconv_layers_1_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [1],
        model.dconv_layers_1_groupnorm_bias[freq_idx][encdec_idx][layer_idx][1],
        1e-05);

    // Conv1d(6, 96, kernel_size=(1,), stride=(1,))
    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<6, 96, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 1:
        y = demucscpp::conv1d<12, 192, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 2:
        y = demucscpp::conv1d<24, 384, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    case 3:
        y = demucscpp::conv1d<48, 768, 1, 1, 0, 1>(
            y,
            model.dconv_layers_3_conv1d_weight[freq_idx][encdec_idx][layer_idx]
                                              [1],
            model.dconv_layers_3_conv1d_bias[freq_idx][encdec_idx][layer_idx]
                                            [1]);
        break;
    };

    y = demucscpp::group_norm(
        y,
        model.dconv_layers_4_groupnorm_weight[freq_idx][encdec_idx][layer_idx]
                                             [1],
        model.dconv_layers_4_groupnorm_bias[freq_idx][encdec_idx][layer_idx][1],
        1, 1e-05);

    y = demucscpp::glu(y, 1);
    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][encdec_idx][layer_idx][1]);

    // if y_copy is shorter than y in the last dim
    // pad the last dim with zeros to match

    if (y_copy.dimension(2) < y.dimension(2))
    {
        // pad the last dim with zeros to match
        Eigen::Tensor3dXf padded_tensor_copy(
            y_copy.dimension(0), y_copy.dimension(1), y.dimension(2));
        padded_tensor_copy.setZero();
        padded_tensor_copy.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                                 y_copy.dimensions()) = y_copy;
        y_copy = padded_tensor_copy;
    }

    // now sum with itself
    y = y + y_copy;
}

// StemDeck Android: multi-head attention, out = softmax(Q K^T) V per head.
// Qs (T x C) is already scaled; K, V are (S x C); head h is the column
// block [h*hd, (h+1)*hd). On the GPU, each head is two products and a
// softmax in one batch. On the CPU, the (head, block of 128 queries)
// pairs are spread over every core; each computes the scores transposed
// (keys x queries) so each query's softmax runs down one contiguous column.
void demucscpp::attention_heads(const Eigen::MatrixXf &Qs, const Eigen::MatrixXf &K, const Eigen::MatrixXf &V,
                                int num_heads, Eigen::MatrixXf &out)
{
    const int T = static_cast<int>(Qs.rows());
    const int S = static_cast<int>(K.rows());
    const int C = static_cast<int>(Qs.cols());
    const int hd = C / num_heads;
    out.resize(T, C);

    if (double(T) * S * C >= 16.0e6)
    {
        if (auto g = sdgpu::Batch::try_begin())
        {
            SD_PROF("gpu: attention");
            sdgpu::Buf q = g->input(Qs.data(), size_t(T) * C);
            sdgpu::Buf k = g->input(K.data(), size_t(S) * C);
            sdgpu::Buf v = g->input(V.data(), size_t(S) * C);
            sdgpu::Buf sc = g->scratch(size_t(S) * T);
            sdgpu::Buf o = g->scratch(size_t(T) * C);
            for (int h = 0; h < num_heads; ++h)
            {
                const size_t off = size_t(h) * hd;
                // scores (S x T) = K_h (S x hd) * Q_h^T (hd x T)
                g->gemm(false, true, S, T, hd, 1.0f, k, off * S, S, q, off * T, T, 0.0f, sc, 0, S);
                g->softmax_cols(sc, 0, S, T, S);
                // out_h (T x hd) = scores^T (T x S) * V_h (S x hd)
                g->gemm(true, false, T, hd, S, 1.0f, sc, 0, S, v, off * S, S, 0.0f, o, off * T, T);
            }
            g->output(o, 0, out.data(), size_t(T) * C);
            if (g->run())
                return;
        }
    }

    constexpr int kBlock = 128;
    const int blocks = (T + kBlock - 1) / kBlock;
    sdpool::parallel_for(num_heads * blocks, [&](int task) {
        const int h = task / blocks;
        const int q0 = (task % blocks) * kBlock;
        const int nb = std::min(kBlock, T - q0);
        const Eigen::Index off = Eigen::Index(h) * hd;
        Eigen::Map<const Eigen::MatrixXf> Qh(Qs.data() + off * T, T, hd);
        Eigen::Map<const Eigen::MatrixXf> Kh(K.data() + off * S, S, hd);
        Eigen::Map<const Eigen::MatrixXf> Vh(V.data() + off * S, S, hd);
        thread_local Eigen::MatrixXf scores;
        scores.resize(S, nb);
        scores.noalias() = Kh * Qh.middleRows(q0, nb).transpose();
        for (int j = 0; j < nb; ++j)
        {
            auto col = scores.col(j).array();
            const float mx = col.maxCoeff();
            col = (col - mx).exp();
            col *= 1.0f / col.sum();
        }
        out.block(q0, off, nb, hd).noalias() = scores.transpose() * Vh;
    });
}

void demucscpp::common_encoder_layer(
    Eigen::Tensor3dXf &q,       // q = x = frequency
    const Eigen::Tensor3dXf &k, // k = xt = time
    const Eigen::Tensor1dXf &norm1_weight, const Eigen::Tensor1dXf &norm1_bias,
    const Eigen::Tensor1dXf &norm2_weight, const Eigen::Tensor1dXf &norm2_bias,
    const Eigen::MatrixXf &in_proj_weight, const Eigen::VectorXf &in_proj_bias,
    const Eigen::MatrixXf &out_proj_weight,
    const Eigen::VectorXf &out_proj_bias, const Eigen::VectorXf &gamma1_scale,
    const Eigen::Tensor1dXf &norm3_weight, const Eigen::Tensor1dXf &norm3_bias,
    const Eigen::MatrixXf &linear1_weight, const Eigen::VectorXf &linear1_bias,
    const Eigen::MatrixXf &linear2_weight, const Eigen::VectorXf &linear2_bias,
    const Eigen::VectorXf &gamma2_scale,
    const Eigen::Tensor1dXf &norm_out_weight,
    const Eigen::Tensor1dXf &norm_out_bias, const int num_heads,
    // optional params
    float eps, const bool self_attention)
{
    // Normalize x using the norm1 weights and biases
    Eigen::Tensor3dXf q_norm =
        demucscpp::layer_norm(q, norm1_weight, norm1_bias, eps);

    Eigen::Tensor3dXf k_norm;
    if (self_attention)
    {
        k_norm = q_norm;
    }
    else
    {
        k_norm = demucscpp::layer_norm(k, norm2_weight, norm2_bias, eps);
    }

    // Cross-attention block
    // Compute Q, K, V matrices

    int B = q.dimension(0);
    int T = q.dimension(1);
    int C = q.dimension(2);

    int S = k.dimension(1);

    // Reshape q, k to 2D matrix of dimensions (T*B, C)

    // Use Eigen::Map to avoid manual loops for reshaping
    // StemDeck Android: the projections, attention and feed-forward go
    // through sdgemm (GPU when free, otherwise every core), and attention
    // runs as one GPU batch when it can. Same maths as upstream.
    SD_PROF_BEGIN(qkv, "attention: Q/K/V projection");
    const int ldw = static_cast<int>(in_proj_weight.rows()); // 3C
    Eigen::MatrixXf Q(T, C), K(S, C), V(S, C);
    sdgemm::gemm(false, true, T, C, C, 1.0f, q_norm.data(), T, in_proj_weight.data(), ldw, 0.0f, Q.data(), T,
                 sdgemm::kConstB);
    sdgemm::gemm(false, true, S, C, C, 1.0f, k_norm.data(), S, in_proj_weight.data() + C, ldw, 0.0f, K.data(), S,
                 sdgemm::kConstB);
    sdgemm::gemm(false, true, S, C, C, 1.0f, k_norm.data(), S, in_proj_weight.data() + 2 * C, ldw, 0.0f, V.data(),
                 S, sdgemm::kConstB);

    Eigen::VectorXf q_bias = in_proj_bias.segment(0, C);
    Eigen::VectorXf k_bias = in_proj_bias.segment(C, C);
    Eigen::VectorXf v_bias = in_proj_bias.segment(2 * C, C);

    // copied from linear layer: ff1.rowwise() += linear1_bias.transpose();
    Q.rowwise() += q_bias.transpose();
    K.rowwise() += k_bias.transpose();
    V.rowwise() += v_bias.transpose();
    SD_PROF_END(qkv);

    SD_PROF_BEGIN(heads, "attention: scores + softmax + weighted sum");
    const int head_split = C / num_heads;
    const float scale = 1.0f / std::sqrt((float)head_split);
    Eigen::MatrixXf cross_attn_out(T, C);
    Eigen::MatrixXf Qs = Q * scale;
    demucscpp::attention_heads(Qs, K, V, num_heads, cross_attn_out);
    SD_PROF_END(heads);

    // Copy q into q_2d (Map q to 2D matrix)
    Eigen::Map<Eigen::MatrixXf> q_2d(q.data(), T, C);
    SD_PROF_BEGIN(oproj, "attention: output projection");

    // Apply output projection with gamma1_scale
    Eigen::MatrixXf out_proj(T, C);
    sdgemm::gemm(false, true, T, C, C, 1.0f, cross_attn_out.data(), T, out_proj_weight.data(),
                 static_cast<int>(out_proj_weight.rows()), 0.0f, out_proj.data(), T, sdgemm::kConstB);
    out_proj.array().rowwise() += out_proj_bias.transpose().array();
    out_proj = out_proj.array().rowwise() * gamma1_scale.transpose().array();

    // Add to q
    q_2d += out_proj;
    SD_PROF_END(oproj);

    // before feedforward, apply norm3 to x i.e. q
    q_norm = demucscpp::layer_norm(q, norm3_weight, norm3_bias, eps);

    // Feedforward block
    // Linear layer 1
    SD_PROF_BEGIN(ffn1, "transformer feed-forward (linear 1)");
    const int F = static_cast<int>(linear1_weight.rows());
    Eigen::MatrixXf ff1(T, F);
    sdgemm::gemm(false, true, T, F, C, 1.0f, q_norm.data(), T, linear1_weight.data(), F, 0.0f, ff1.data(), T,
                 sdgemm::kConstB);
    ff1.rowwise() += linear1_bias.transpose();
    SD_PROF_END(ffn1);

    ff1 = demucscpp::gelu(ff1);

    // Linear layer 2
    SD_PROF_BEGIN(ffn, "transformer feed-forward (linear 2)");
    Eigen::MatrixXf ff2(T, C);
    sdgemm::gemm(false, true, T, C, F, 1.0f, ff1.data(), T, linear2_weight.data(),
                 static_cast<int>(linear2_weight.rows()), 0.0f, ff2.data(), T, sdgemm::kConstB);
    ff2.rowwise() += linear2_bias.transpose();

    // Apply gamma_2 scale directly on 2D matrix
    ff2 = ff2.array().rowwise() * gamma2_scale.transpose().array();

    // now x = x + self.gamma_2(self._ff_block(self.norm3(q))))
    q_2d += ff2;
    SD_PROF_END(ffn);

    // Map the 2D data back into a 3D tensor with dimensions (T, B, C)
    q = Eigen::TensorMap<Eigen::Tensor3dXf>(q_2d.data(), T, B, C);

    // Swap the first and last dimensions to get a tensor with dimensions (B, C,
    // T)
    Eigen::array<int, 3> permute_dims_3 = {1, 2, 0};
    Eigen::Tensor3dXf q_shuf = demucscpp::permute3(q, permute_dims_3);

    // Normalize the output with norm_out/MyGroupNorm
    q = demucscpp::group_norm(q_shuf, norm_out_weight, norm_out_bias, 1, eps);

    Eigen::array<int, 3> permute_dims_4 = {0, 2, 1};
    q_shuf = demucscpp::permute3(q, permute_dims_4);

    q = q_shuf;
}

void demucscpp_v3::local_attention(
    Eigen::Tensor3dXf &x, // x = frequency, time, or combined
                          // input tensor [B, C, T]
    const Eigen::Tensor3dXf &content_weight,
    const Eigen::Tensor1dXf &content_bias,
    const Eigen::Tensor3dXf &query_weight, const Eigen::Tensor1dXf &query_bias,
    const Eigen::Tensor3dXf &key_weight, const Eigen::Tensor1dXf &key_bias,
    const Eigen::Tensor3dXf &query_decay_weight,
    const Eigen::Tensor1dXf &query_decay_bias,
    const Eigen::Tensor2dXf &query_decay_kernel,
    const Eigen::Tensor3dXf &proj_weight, const Eigen::Tensor1dXf &proj_bias,
    const int hidden_size)
{
    // local-attention block

    int B = x.dimension(0);
    int C = x.dimension(1);
    int T = x.dimension(2);

    const int num_heads = demucscpp_v3::LOCAL_ATTN_N_HEADS;

    // apply query conv1d on x
    Eigen::Tensor3dXf queries;
    Eigen::Tensor3dXf query_decays;
    Eigen::Tensor3dXf keys;
    Eigen::Tensor3dXf content;

    if (hidden_size == 192)
    {
        queries = demucscpp::conv1d<192, 192, 1, 1, 0, 1>(x, query_weight,
                                                          query_bias);
        keys = demucscpp::conv1d<192, 192, 1, 1, 0, 1>(x, key_weight, key_bias);
        query_decays = demucscpp::conv1d<192, 16, 1, 1, 0, 1>(
            x, query_decay_weight, query_decay_bias);
        content = demucscpp::conv1d<192, 192, 1, 1, 0, 1>(x, content_weight,
                                                          content_bias);
    }
    else
    {
        queries = demucscpp::conv1d<384, 384, 1, 1, 0, 1>(x, query_weight,
                                                          query_bias);
        keys = demucscpp::conv1d<384, 384, 1, 1, 0, 1>(x, key_weight, key_bias);
        query_decays = demucscpp::conv1d<384, 16, 1, 1, 0, 1>(
            x, query_decay_weight, query_decay_bias);
        content = demucscpp::conv1d<384, 384, 1, 1, 0, 1>(x, content_weight,
                                                          content_bias);
    }

    // so far, this is correct and matches pytorch

    int features_per_head = C / num_heads;

    // now implement dots calculation

    // Initialize the dots tensor
    Eigen::Tensor4dXf dots(B, num_heads, T, T);
    dots.setZero();

    // Precompute the square root of features_per_head
    float sqrt_features_per_head = std::sqrt(features_per_head);

    // apply a sigmoid activation with a 1/2 incorporated
    query_decays = query_decays.unaryExpr(
        [](float v) { return 0.5f / (1.0f + std::exp(-v)); });

    // Initialize the weights tensor for softmax
    Eigen::Tensor4dXf weights(B, num_heads, T, T);

    // Loop structure to compute both dot products and apply decay
    // simultaneously
    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            for (int t = 0; t < T; ++t)
            {
                for (int s = 0; s < T; ++s)
                {
                    float dot_product = 0.0f;
                    float decay_effect = 0.0f;

                    // Compute the standard dot product
                    for (int c = 0; c < features_per_head; ++c)
                    {
                        int channel_index = h * features_per_head + c;
                        dot_product += queries(b, channel_index, s) *
                                       keys(b, channel_index, t);
                    }
                    dots(b, h, t, s) = dot_product / sqrt_features_per_head;

                    // Calculate decay effect for this dot product
                    for (int n = 0; n < LOCAL_ATTN_N_DECAY; ++n)
                    {
                        int decay_index = std::abs(
                            t - s); // Assuming decay_kernel is indexed by delta
                        float decay_kernel_value =
                            query_decay_kernel(n, decay_index);

                        // Transform query_decay by applying sigmoid directly
                        // here
                        float decay_query_value =
                            query_decays(b, h * LOCAL_ATTN_N_DECAY + n, s);

                        decay_effect += decay_kernel_value * decay_query_value;
                    }

                    // Apply decay effect directly to the dot product
                    if (t != s)
                    {
                        dots(b, h, t, s) += decay_effect;
                    }
                    else
                    {
                        dots(b, h, t, s) = -100.0f;
                    }
                }
            }

            for (int t = 0; t < T; ++t)
            {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int s = 0; s < T; ++s)
                {
                    if (dots(b, h, s, t) > max_val)
                    {
                        max_val = dots(b, h, s, t);
                    }
                }

                float sum_exp = 0.0f;
                // Calculate the exponentials and sum them
                for (int s = 0; s < T; ++s)
                {
                    weights(b, h, s, t) = std::exp(dots(b, h, s, t) - max_val);
                    sum_exp += weights(b, h, s, t);
                }

                // Normalize the weights to form a proper probability
                // distribution
                for (int s = 0; s < T; ++s)
                {
                    weights(b, h, s, t) /= sum_exp;
                }
            }
        }
    }

    // Initialize the reshaped result tensor directly
    Eigen::Tensor3dXf reshaped_result(B, C, T);
    reshaped_result.setZero();

    // Merge computation of result tensor and reshaping
    for (int b = 0; b < B; ++b)
    {
        for (int h = 0; h < num_heads; ++h)
        {
            for (int c = 0; c < C / num_heads; ++c)
            {
                for (int s = 0; s < T; ++s)
                {
                    for (int t = 0; t < T; ++t)
                    {
                        // Directly update the reshaped_result tensor
                        int full_channel_index = h * (C / num_heads) + c;
                        reshaped_result(b, full_channel_index, s) +=
                            weights(b, h, t, s) *
                            content(b, h * (C / num_heads) + c, t);
                    }
                }
            }
        }
    }

    // Apply projection layer
    Eigen::Tensor3dXf projected_result;
    if (hidden_size == 192)
    {
        projected_result = demucscpp::conv1d<192, 192, 1, 1, 0, 1>(
            reshaped_result, proj_weight, proj_bias);
    }
    else
    {
        projected_result = demucscpp::conv1d<384, 384, 1, 1, 0, 1>(
            reshaped_result, proj_weight, proj_bias);
    }

    // Add x to projected_result
    x += projected_result;
}

void demucscpp_v3::apply_dconv_v3(
    const struct demucscpp_v3::demucs_v3_model &model, Eigen::Tensor3dXf &y,
    int freq_idx, int layer_idx, int mid_crop)
{
    // store another copy of y to sum back later
    Eigen::Tensor3dXf y_copy = y;

    // now dconv time

    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 12, 3, 1, 1, 1>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 24, 3, 1, 1, 1>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 48, 3, 1, 1, 1>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 96, 3, 1, 1, 1>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    };

    // y = demucscpp_v3::groupnorm::group_norm_fused_gelu(
    //     y,
    //     model.dconv_layers_1_groupnorm_weight[freq_idx][layer_idx]
    //                                          [0],
    //     model.dconv_layers_1_groupnorm_bias[freq_idx][layer_idx][0],
    //     1,
    //     1e-05);

    y = demucscpp::group_norm_fused_gelu(
        y, model.dconv_layers_1_groupnorm_weight[freq_idx][layer_idx][0],
        model.dconv_layers_1_groupnorm_bias[freq_idx][layer_idx][0], 1e-05);

    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<12, 96, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 1:
        y = demucscpp::conv1d<24, 192, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 2:
        y = demucscpp::conv1d<48, 384, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    case 3:
        y = demucscpp::conv1d<96, 768, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][0],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][0]);
        break;
    };

    y = demucscpp::group_norm(
        y, model.dconv_layers_4_groupnorm_weight[freq_idx][layer_idx][0],
        model.dconv_layers_4_groupnorm_bias[freq_idx][layer_idx][0], 1, 1e-05);

    y = demucscpp::glu(y, 1);

    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][layer_idx][0]);

    // now we add y to itself
    y = y + y_copy;

    // store another copy of y to sum back later
    y_copy = y;

    // NEXT ENTIRE SUBSEQUENCE OF DCONV WITH SLIGHTLY DIFFERENT PARAMS

    // Conv1d(48, 6, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<48, 12, 3, 1, 2, 2>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 1:
        y = demucscpp::conv1d<96, 24, 3, 1, 2, 2>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 2:
        y = demucscpp::conv1d<192, 48, 3, 1, 2, 2>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 3:
        y = demucscpp::conv1d<384, 96, 3, 1, 2, 2>(
            y, model.dconv_layers_0_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_0_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    };

    Eigen::Tensor3dXf y_cropped =
        y.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                Eigen::array<Eigen::Index, 3>(
                    {y.dimension(0), y.dimension(1), mid_crop}));

    y = y_cropped;

    y = demucscpp::group_norm_fused_gelu(
        y, model.dconv_layers_1_groupnorm_weight[freq_idx][layer_idx][1],
        model.dconv_layers_1_groupnorm_bias[freq_idx][layer_idx][1], 1e-05);

    // Conv1d(6, 96, kernel_size=(1,), stride=(1,))
    switch (layer_idx)
    {
    case 0:
        y = demucscpp::conv1d<12, 96, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 1:
        y = demucscpp::conv1d<24, 192, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 2:
        y = demucscpp::conv1d<48, 384, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    case 3:
        y = demucscpp::conv1d<96, 768, 1, 1, 0, 1>(
            y, model.dconv_layers_3_conv1d_weight[freq_idx][layer_idx][1],
            model.dconv_layers_3_conv1d_bias[freq_idx][layer_idx][1]);
        break;
    };

    y = demucscpp::group_norm(
        y, model.dconv_layers_4_groupnorm_weight[freq_idx][layer_idx][1],
        model.dconv_layers_4_groupnorm_bias[freq_idx][layer_idx][1], 1, 1e-05);

    y = demucscpp::glu(y, 1);
    y = demucscpp::layer_scale(
        y, model.dconv_layers_6_scale[freq_idx][layer_idx][1]);

    // if y_copy is shorter than y in the last dim
    // pad the last dim with zeros to match

    if (y_copy.dimension(2) < y.dimension(2))
    {
        // pad the last dim with zeros to match
        Eigen::Tensor3dXf padded_tensor_copy(
            y_copy.dimension(0), y_copy.dimension(1), y.dimension(2));
        padded_tensor_copy.setZero();
        padded_tensor_copy.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                                 y_copy.dimensions()) = y_copy;
        y_copy = padded_tensor_copy;
    }

    // now sum with itself
    y = y + y_copy;
}

void demucscpp_v3::apply_dconv_v3_encoder_4_5(
    const struct demucscpp_v3::demucs_v3_model &model, Eigen::Tensor3dXf &y,
    int encoder_idx, int mid_crop,
    struct demucscpp_v3::demucs_v3_segment_buffers &buffers)
{
    int lstm_hidden_size = encoder_idx == 0 ? demucscpp_v3::LSTM_HIDDEN_SIZE_0
                                            : demucscpp_v3::LSTM_HIDDEN_SIZE_1;

    // store another copy of y to sum back later
    Eigen::Tensor3dXf y_copy = y;

    // now dconv time

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<768, 192, 3, 1, 1, 1>(
            y, model.encoder_4_5_dconv_layers_0_conv1d_weight[encoder_idx][0],
            model.encoder_4_5_dconv_layers_0_conv1d_bias[encoder_idx][0]);
        break;
    case 1:
        y = demucscpp::conv1d<1536, 384, 3, 1, 1, 1>(
            y, model.encoder_4_5_dconv_layers_0_conv1d_weight[encoder_idx][0],
            model.encoder_4_5_dconv_layers_0_conv1d_bias[encoder_idx][0]);
        break;
    };

    y = demucscpp::group_norm_fused_gelu(
        y, model.encoder_4_5_dconv_layers_1_groupnorm_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_1_groupnorm_bias[encoder_idx][0], 1e-05);

    // transpose it to put time seq last
    Eigen::MatrixXf y_mat =
        Eigen::Map<Eigen::MatrixXf>(y.data(), y.dimension(1), y.dimension(2))
            .transpose();

    // then, bilstm
    demucscpp_v3::lstm_forward(model, encoder_idx, 0, y_mat, buffers,
                               lstm_hidden_size);

    // access last element of the last dim which is the output of the bilstm
    Eigen::MatrixXf lstm_out_0 = buffers.lstm_output[encoder_idx][0][1];

    // set lstm state to 0
    demucscpp_v3::lstm_reset_zero(encoder_idx, 0, buffers);

    // apply the linear layer on the lstm_out_0
    lstm_out_0 = (lstm_out_0 *
                  model.encoder_4_5_dconv_layers_3_linear_weight[encoder_idx][0]
                      .transpose())
                     .rowwise() +
                 model.encoder_4_5_dconv_layers_3_linear_bias[encoder_idx][0]
                     .transpose();

    // then apply skip connection
    lstm_out_0 = lstm_out_0 + y_mat;

    // copy it to a original 3d tensor
    y = Eigen::TensorMap<Eigen::Tensor3dXf>(
        lstm_out_0.data(), lstm_out_0.rows(), 1, lstm_out_0.cols());

    // swap dims from 0,1,2 to 1,2,0
    Eigen::Tensor3dXf y_shuff = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // then, localattn
    demucscpp_v3::local_attention(
        y_shuff,
        model.encoder_4_5_dconv_layers_4_content_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_content_bias[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_query_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_query_bias[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_key_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_key_bias[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_query_decay_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_query_decay_bias[encoder_idx][0],
        buffers.local_attn_decay_kernel,
        model.encoder_4_5_dconv_layers_4_proj_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_4_proj_bias[encoder_idx][0],
        lstm_hidden_size);

    y = y_shuff;

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<192, 1536, 1, 1, 0, 1>(
            y, model.encoder_4_5_dconv_layers_5_conv1d_weight[encoder_idx][0],
            model.encoder_4_5_dconv_layers_5_conv1d_bias[encoder_idx][0]);
        break;
    case 1:
        y = demucscpp::conv1d<384, 3072, 1, 1, 0, 1>(
            y, model.encoder_4_5_dconv_layers_5_conv1d_weight[encoder_idx][0],
            model.encoder_4_5_dconv_layers_5_conv1d_bias[encoder_idx][0]);
        break;
    };

    y = demucscpp::group_norm(
        y, model.encoder_4_5_dconv_layers_6_groupnorm_weight[encoder_idx][0],
        model.encoder_4_5_dconv_layers_6_groupnorm_bias[encoder_idx][0], 1,
        1e-05);

    y = demucscpp::glu(y, 1);

    y = demucscpp::layer_scale(
        y, model.encoder_4_5_dconv_layers_8_scale[encoder_idx][0]);

    // now we add y to itself
    y = y + y_copy;

    // store another copy of y to sum back later
    y_copy = y;

    // NEXT ENTIRE SUBSEQUENCE OF DCONV WITH SLIGHTLY DIFFERENT PARAMS
    // now dconv time

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<768, 192, 3, 1, 2, 2>(
            y, model.encoder_4_5_dconv_layers_0_conv1d_weight[encoder_idx][1],
            model.encoder_4_5_dconv_layers_0_conv1d_bias[encoder_idx][1]);
        break;
    case 1:
        y = demucscpp::conv1d<1536, 384, 3, 1, 2, 2>(
            y, model.encoder_4_5_dconv_layers_0_conv1d_weight[encoder_idx][1],
            model.encoder_4_5_dconv_layers_0_conv1d_bias[encoder_idx][1]);
        break;
    };

    Eigen::Tensor3dXf y_cropped =
        y.slice(Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                Eigen::array<Eigen::Index, 3>(
                    {y.dimension(0), y.dimension(1), mid_crop}));

    y = y_cropped;

    y = demucscpp::group_norm_fused_gelu(
        y, model.encoder_4_5_dconv_layers_1_groupnorm_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_1_groupnorm_bias[encoder_idx][1], 1e-05);

    // transpose it to put time seq last
    y_mat =
        Eigen::Map<Eigen::MatrixXf>(y.data(), y.dimension(1), y.dimension(2))
            .transpose();

    // then, bilstm
    demucscpp_v3::lstm_forward(model, encoder_idx, 1, y_mat, buffers,
                               lstm_hidden_size);

    // access last element of the last dim which is the output of the bilstm
    lstm_out_0 = buffers.lstm_output[encoder_idx][1][1];

    // reset lstm state to 0
    demucscpp_v3::lstm_reset_zero(encoder_idx, 1, buffers);

    // apply the linear layer on the lstm_out_0
    lstm_out_0 = (lstm_out_0 *
                  model.encoder_4_5_dconv_layers_3_linear_weight[encoder_idx][1]
                      .transpose())
                     .rowwise() +
                 model.encoder_4_5_dconv_layers_3_linear_bias[encoder_idx][1]
                     .transpose();

    // then apply skip connection
    lstm_out_0 = lstm_out_0 + y_mat;

    // copy it to a original 3d tensor
    y = Eigen::TensorMap<Eigen::Tensor3dXf>(
        lstm_out_0.data(), lstm_out_0.rows(), 1, lstm_out_0.cols());

    // swap dims from 0,1,2 to 1,2,0
    y_shuff = y.shuffle(Eigen::array<int, 3>({1, 2, 0}));

    // then, localattn
    demucscpp_v3::local_attention(
        y_shuff,
        model.encoder_4_5_dconv_layers_4_content_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_content_bias[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_query_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_query_bias[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_key_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_key_bias[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_query_decay_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_query_decay_bias[encoder_idx][1],
        buffers.local_attn_decay_kernel,
        model.encoder_4_5_dconv_layers_4_proj_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_4_proj_bias[encoder_idx][1],
        lstm_hidden_size);

    y = y_shuff;

    switch (encoder_idx)
    {
    case 0:
        y = demucscpp::conv1d<192, 1536, 1, 1, 0, 1>(
            y, model.encoder_4_5_dconv_layers_5_conv1d_weight[encoder_idx][1],
            model.encoder_4_5_dconv_layers_5_conv1d_bias[encoder_idx][1]);
        break;
    case 1:
        y = demucscpp::conv1d<384, 3072, 1, 1, 0, 1>(
            y, model.encoder_4_5_dconv_layers_5_conv1d_weight[encoder_idx][1],
            model.encoder_4_5_dconv_layers_5_conv1d_bias[encoder_idx][1]);
        break;
    };

    y = demucscpp::group_norm(
        y, model.encoder_4_5_dconv_layers_6_groupnorm_weight[encoder_idx][1],
        model.encoder_4_5_dconv_layers_6_groupnorm_bias[encoder_idx][1], 1,
        1e-05);

    y = demucscpp::glu(y, 1);

    y = demucscpp::layer_scale(
        y, model.encoder_4_5_dconv_layers_8_scale[encoder_idx][1]);

    // now sum with itself
    y = y + y_copy;
}
