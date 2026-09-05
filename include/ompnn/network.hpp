// SPDX-License-Identifier: LGPL-3.0-only
// ompnn - OpenMP feed-forward neural network library (1:1 port of syclnn 0.2)
// Copyright (C) 2026 Antonio Napolitano
//
// network.hpp: the same multi-layer perceptron as syclnn, written with OpenMP.
// One code base, two execution paths chosen at run time by Options::device:
//   host  : `#pragma omp parallel for simd` loops + CBLAS (OpenBLAS / oneMKL)
//   target: `#pragma omp target teams distribute parallel for simd` kernels on
//           omp_target_alloc'ed device memory (`is_device_ptr`), GEMM through
//           cuBLAS / hipBLAS on the same pointers (interop), omp_target_memcpy
//           for the transfers.
// Every region is synchronous: OpenMP's portable subset has no event graph, so
// the phases execute back to back and the host timer around each one is the
// profiler.  This is deliberate - it is what the study compares against the
// event-driven SYCL/CUDA implementations.
#pragma once

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ompnn/activations.hpp"
#include "ompnn/blas.hpp"
#include "ompnn/config.hpp"
#include "ompnn/device.hpp"
#include "ompnn/profile.hpp"

namespace ompnn {

// ============================================================================
//                                   kernels
// ============================================================================
// Each kernel exists in a host and a target flavour selected by `gpu`; the loop
// bodies are identical.  No lambdas or STL inside target regions (gcc's nvptx /
// amdgcn offloading rejects most of them), functors from activations.hpp only.
namespace kernels {

/// `tl`: thread_limit of the target teams (Options::workgroup_size, default 256).
template <typename T, typename F>
inline void activate(bool gpu, int dev, int tl, T *net, T *out, const T *bias, std::size_t n, std::size_t M, F f) {
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(net, out, bias) firstprivate(f)
        for (std::size_t i = 0; i < n; ++i) {
            const T z = net[i] + bias[i % M];
            net[i] = z;
            out[i] = f(z);
        }
    } else {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < n; ++i) {
            const T z = net[i] + bias[i % M];
            net[i] = z;
            out[i] = f(z);
        }
    }
}

/// delta = (t - o) f'(net); returns sum 0.5 (t - o)^2 (reduction, or one atomic per element)
template <typename T, typename D>
inline double output_delta_loss(bool gpu, int dev, int tl, bool reduction, const T *targets, const T *out, const T *net,
                                T *delta, std::size_t n, D df) {
    double loss = 0.0;
    if (reduction) {
        if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(targets, out, net, delta) firstprivate(df) reduction(+ : loss) map(tofrom : loss)
            for (std::size_t i = 0; i < n; ++i) {
                const T o = out[i];
                const T e = targets[i] - o;
                delta[i] = e * df(o, net[i]);
                const double ed = double(e);
                loss += 0.5 * ed * ed;
            }
        } else {
#pragma omp parallel for simd reduction(+ : loss)
            for (std::size_t i = 0; i < n; ++i) {
                const T o = out[i];
                const T e = targets[i] - o;
                delta[i] = e * df(o, net[i]);
                const double ed = double(e);
                loss += 0.5 * ed * ed;
            }
        }
        return loss;
    }
    // 0.1-style: one atomic add on a single scalar of type T per element
    T acc = T(0);
    if (gpu) {
#pragma omp target teams distribute parallel for device(dev) thread_limit(tl) is_device_ptr(targets, out, net, delta) firstprivate(df) map(tofrom : acc)
        for (std::size_t i = 0; i < n; ++i) {
            const T o = out[i];
            const T e = targets[i] - o;
            delta[i] = e * df(o, net[i]);
            const T c = static_cast<T>(0.5 * double(e) * double(e));
#pragma omp atomic
            acc += c;
        }
    } else {
#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            const T o = out[i];
            const T e = targets[i] - o;
            delta[i] = e * df(o, net[i]);
            const T c = static_cast<T>(0.5 * double(e) * double(e));
#pragma omp atomic
            acc += c;
        }
    }
    return static_cast<double>(acc);
}

template <typename T, typename D>
inline void hidden_delta(bool gpu, int dev, int tl, T *delta, const T *out, const T *net, std::size_t n, D df) {
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(delta, out, net) firstprivate(df)
        for (std::size_t i = 0; i < n; ++i)
            delta[i] *= df(out[i], net[i]);
    } else {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < n; ++i)
            delta[i] *= df(out[i], net[i]);
    }
}

/// gb[j] = sum_n delta[j + M n] / B (0.1: one work-item per output neuron)
template <typename T> inline void bias_grad(bool gpu, int dev, int tl, const T *delta, T *gb, std::size_t M, std::size_t B) {
    if (gpu) {
#pragma omp target teams distribute parallel for device(dev) thread_limit(tl) is_device_ptr(delta, gb)
        for (std::size_t j = 0; j < M; ++j) {
            T s = T(0);
            for (std::size_t n = 0; n < B; ++n)
                s += delta[j + M * n];
            gb[j] = s / static_cast<T>(B);
        }
    } else {
#pragma omp parallel for
        for (std::size_t j = 0; j < M; ++j) {
            T s = T(0);
            for (std::size_t n = 0; n < B; ++n)
                s += delta[j + M * n];
            gb[j] = s / static_cast<T>(B);
        }
    }
}

template <typename T> struct UpdateParams {
    int strategy;
    bool momentum, l1, l2, host_corr;
    T mu, lr, eps, beta1, beta2, lambda1, lambda2;
    T lr_eff, c1, c2, t;
};

#pragma omp declare target
template <typename T> inline void step_fn(T &p, T grad, T reg_grad, T &m, T &v, const UpdateParams<T> &u) {
    const T g = grad - reg_grad; // ascent direction of (t - o)
    T step;
    switch (u.strategy) {
    case 0: // Constant
    case 1: // LinearDecay
        if (u.momentum) {
            m = u.mu * m + u.lr_eff * g;
            step = m;
        } else {
            step = u.lr_eff * g;
        }
        break;
    case 2: { // AdaGrad
        const T G = -g;
        v += G * G;
        step = (u.lr / (std::sqrt(v) + u.eps)) * g;
    } break;
    case 3: { // RMSProp
        const T G = -g;
        v = u.beta1 * v + (T(1) - u.beta1) * G * G;
        step = (u.lr / (std::sqrt(v) + u.eps)) * g;
    } break;
    default: { // Adam
        const T G = -g;
        m = u.beta1 * m + (T(1) - u.beta1) * G;
        v = u.beta2 * v + (T(1) - u.beta2) * G * G;
        T m_hat, v_hat;
        if (u.host_corr) {
            m_hat = m * u.c1;
            v_hat = v * u.c2;
        } else {
            m_hat = m / (T(1) - std::pow(u.beta1, u.t));
            v_hat = v / (T(1) - std::pow(u.beta2, u.t));
        }
        step = -(u.lr * m_hat / (std::sqrt(v_hat) + u.eps));
    } break;
    }
    p += step;
}
#pragma omp end declare target

/// One fused loop over the weights and biases of one layer.
template <typename T>
inline void update(bool gpu, int dev, int tl, T *W, const T *gW, T *mW, T *vW, std::size_t nw, T *b, const T *gb, T *mb,
                   T *vb, std::size_t nb, UpdateParams<T> u) {
    const std::size_t n = nw + nb;
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(W, gW, mW, vW, b, gb, mb, vb) firstprivate(u)
        for (std::size_t i = 0; i < n; ++i) {
            if (i < nw) {
                const T w = W[i];
                T r = T(0);
                if (u.l1)
                    r += u.lambda1 * sgn(w);
                if (u.l2)
                    r += u.lambda2 * w;
                step_fn(W[i], gW[i], r, mW[i], vW[i], u);
            } else {
                const std::size_t j = i - nw;
                step_fn(b[j], gb[j], T(0), mb[j], vb[j], u);
            }
        }
    } else {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < n; ++i) {
            if (i < nw) {
                const T w = W[i];
                T r = T(0);
                if (u.l1)
                    r += u.lambda1 * sgn(w);
                if (u.l2)
                    r += u.lambda2 * w;
                step_fn(W[i], gW[i], r, mW[i], vW[i], u);
            } else {
                const std::size_t j = i - nw;
                step_fn(b[j], gb[j], T(0), mb[j], vb[j], u);
            }
        }
    }
}

template <typename T>
inline void gather(bool gpu, int dev, int tl, const T *src, T *dst, const std::uint32_t *perm, std::size_t rows, std::size_t total) {
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(src, dst, perm)
        for (std::size_t i = 0; i < total; ++i) {
            const std::size_t c = i / rows, r = i - c * rows;
            dst[i] = src[r + rows * perm[c]];
        }
    } else {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < total; ++i) {
            const std::size_t c = i / rows, r = i - c * rows;
            dst[i] = src[r + rows * perm[c]];
        }
    }
}

template <typename T> inline void fill(bool gpu, int dev, int tl, T *p, T v, std::size_t n) {
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(p)
        for (std::size_t i = 0; i < n; ++i)
            p[i] = v;
    } else {
#pragma omp parallel for simd
        for (std::size_t i = 0; i < n; ++i)
            p[i] = v;
    }
}

template <typename T> inline void copy_d2d(bool gpu, int dev, int tl, T *dst, const T *src, std::size_t n) {
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) thread_limit(tl) is_device_ptr(dst, src)
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = src[i];
    } else {
        std::memcpy(dst, src, n * sizeof(T));
    }
}

} // namespace kernels

// ============================================================================
//                                   helpers
// ============================================================================
namespace detail {

/// Host or device allocation (omp_target_alloc), move-only.
template <typename T> class Buffer {
  public:
    Buffer() = default;
    Buffer(std::size_t n, int dev) : m_n(n), m_dev(dev) {
        if (n == 0)
            return;
        if (dev < 0) {
            m_ptr = static_cast<T *>(std::malloc(n * sizeof(T)));
        } else {
#ifdef OMPNN_HOST_ONLY
            throw std::invalid_argument("ompnn: host-only build, no offload device");
#else
            m_ptr = static_cast<T *>(omp_target_alloc(n * sizeof(T), dev));
#endif
        }
        if (!m_ptr)
            throw std::runtime_error("ompnn: allocation of " + std::to_string(n * sizeof(T)) + " bytes failed");
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&o) noexcept : m_n(o.m_n), m_dev(o.m_dev), m_ptr(o.m_ptr) {
        o.m_ptr = nullptr;
        o.m_n = 0;
    }
    Buffer &operator=(Buffer &&o) noexcept {
        if (this != &o) {
            release();
            m_n = o.m_n;
            m_dev = o.m_dev;
            m_ptr = o.m_ptr;
            o.m_ptr = nullptr;
            o.m_n = 0;
        }
        return *this;
    }
    ~Buffer() { release(); }
    void release() noexcept {
        if (m_ptr) {
            if (m_dev < 0)
                std::free(m_ptr);
#ifndef OMPNN_HOST_ONLY
            else
                omp_target_free(m_ptr, m_dev);
#endif
            m_ptr = nullptr;
            m_n = 0;
        }
    }
    T *data() const { return m_ptr; }
    std::size_t size() const { return m_n; }
    explicit operator bool() const { return m_ptr != nullptr; }

    void copy_in(const T *host, std::size_t n, std::size_t offset = 0) const {
        if (m_dev < 0) {
            std::memcpy(m_ptr + offset, host, n * sizeof(T));
            return;
        }
#ifndef OMPNN_HOST_ONLY
        if (omp_target_memcpy(m_ptr, host, n * sizeof(T), offset * sizeof(T), 0, m_dev,
                              omp_get_initial_device()) != 0)
            throw std::runtime_error("ompnn: omp_target_memcpy (host -> device) failed");
#endif
    }
    void copy_out(T *host, std::size_t n, std::size_t offset = 0) const {
        if (m_dev < 0) {
            std::memcpy(host, m_ptr + offset, n * sizeof(T));
            return;
        }
#ifndef OMPNN_HOST_ONLY
        if (omp_target_memcpy(host, m_ptr, n * sizeof(T), 0, offset * sizeof(T), omp_get_initial_device(), m_dev) != 0)
            throw std::runtime_error("ompnn: omp_target_memcpy (device -> host) failed");
#endif
    }

  private:
    std::size_t m_n = 0;
    int m_dev = -1;
    T *m_ptr = nullptr;
};

inline std::vector<std::uint32_t> shuffle_permutation(std::uint32_t n, std::mt19937 &g) {
    std::vector<std::uint32_t> perm(n);
    for (std::uint32_t i = 0; i < n; ++i)
        perm[i] = i;
    for (std::uint32_t i = 0; i + 1 < n; ++i) {
        std::uint32_t j = i + static_cast<std::uint32_t>(g()) % (n - i);
        std::swap(perm[i], perm[j]);
    }
    return perm;
}

} // namespace detail

// ============================================================================
//                                   Network
// ============================================================================
template <typename T> class Network {
    static_assert(std::is_floating_point_v<T>, "Network<T> requires float or double");

  public:
    using history_t = std::vector<std::vector<std::vector<T>>>;

    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
            const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
            Options options = Options{});
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
            Options options = Options{});
    Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
            AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options = Options{});
    ~Network() noexcept = default;
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;

    std::vector<T> train(const std::vector<T> &input_samples, const std::vector<T> &target_samples, unsigned num_samples,
                         unsigned batch_size, unsigned max_epochs);
    std::vector<T> predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size = 0);

    std::pair<history_t, history_t> weights_biases() const { return {m_weights_history, m_biases_history}; }
    std::vector<std::vector<T>> weights() const;
    std::vector<std::vector<T>> biases() const;
    void set_weights_biases(const std::vector<std::vector<T>> &new_weights, const std::vector<std::vector<T>> &new_biases);

    const Profile &profile() const { return m_prof.profile(); }
    void reset_profile() { m_prof.reset(); }
    const Options &options() const { return m_opts; }
    const std::vector<LayerDescription> &layers() const { return m_layers; }
    std::string device_name() const { return m_info.name; }
    DeviceInfo device_info() const { return m_info; }
    std::string blas_backend() const;
    std::size_t parameter_count() const;
    std::size_t num_layers() const { return m_layers.size(); }

  private:
    using Buf = detail::Buffer<T>;

    // ---- configuration ----
    std::vector<LayerDescription> m_layers;
    T m_lr{};
    Regularization<T> m_reg;
    MomentumConfig<T> m_mom;
    BackPropagation m_bp;
    AdaptiveLearningRate<T> m_adapt;
    StopCriteria<T> m_stop;
    Options m_opts;
    std::size_t m_L = 0;

    // ---- execution path ----
    DeviceInfo m_info;
    bool m_gpu = false;
    int m_dev = -1; ///< omp device number (-1 host)
    BlasKind m_blas = BlasKind::Auto;
    int m_tl = 256; ///< thread_limit of the target teams (Options::workgroup_size)
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
    std::unique_ptr<vendor::Handle> m_vendor;
#endif
    Profiler m_prof;

    // ---- parameters and optimiser state ----
    std::vector<Buf> m_W, m_b, m_mW, m_vW, m_mb, m_vb;

    // ---- workspace ----
    std::size_t m_cap = 0;
    std::vector<Buf> m_act, m_net, m_delta, m_gW, m_gb;
    Buf m_ones;

    // ---- bookkeeping ----
    unsigned m_adam_step = 0;
    history_t m_weights_history, m_biases_history;

    // ---- helpers ----
    void init_device();
    void init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed);
    void ensure_workspace(std::size_t batch);
    void release_workspace();
    void snapshot_history();

    void gemm(bool ta, bool tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c,
              int ldc);
    void gemv(bool ta, int m, int n, T alpha, const T *a, int lda, const T *x, T beta, T *y);
    double asum(int n, const T *x);
    double sumsq(int n, const T *x);

    void forward_layer(std::size_t l, const T *in, std::size_t B);
    double output_delta_loss(const T *targets, std::size_t B);
    void hidden_delta(std::size_t l, std::size_t B);
    void gradients(std::size_t l, const T *in, std::size_t B);
    void update(std::size_t l, unsigned adam_step, unsigned epoch, unsigned max_epochs);
    double penalty();
};

/* ============================================================================
                                  IMPLEMENTATION
============================================================================ */

template <typename T> void Network<T>::init_device() {
    m_info = select_device(m_opts.device);
    m_gpu = m_info.omp_device >= 0;
    m_dev = m_info.omp_device;
    if (m_gpu && m_opts.memory != MemoryKind::Device)
        throw std::invalid_argument("ompnn: only memory=device is available on the OpenMP target path");
    m_blas = parse_blas(m_opts.blas);
    if (m_blas == BlasKind::Vendor && !m_gpu)
        throw std::invalid_argument("ompnn: cuBLAS/rocBLAS need the offload device (device=gpu)");
    if (m_blas == BlasKind::Cblas && m_gpu)
        throw std::invalid_argument("ompnn: CBLAS runs on the host (device=cpu)");
    if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        if (m_blas == BlasKind::Auto || m_blas == BlasKind::Vendor)
            m_vendor = std::make_unique<vendor::Handle>(m_dev);
#else
        if (m_blas == BlasKind::Auto)
            m_blas = BlasKind::Omp; // no vendor BLAS in this build: pure OpenMP GEMM
#endif
        if (m_opts.workgroup_size)
            m_tl = static_cast<int>(m_opts.workgroup_size);
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        // OpenMP device numbers and CUDA/HIP ordinals are assumed to coincide: verify it
        // on a probe allocation (HIP_VISIBLE_DEVICES vs ROCR_VISIBLE_DEVICES can differ).
        {
            void *probe = omp_target_alloc(64, m_dev);
            if (probe) {
                cudaPointerAttributes attr{};
                if (cudaPointerGetAttributes(&attr, probe) == cudaSuccess && attr.device != m_dev) {
                    omp_target_free(probe, m_dev);
                    throw std::runtime_error("ompnn: OpenMP device " + std::to_string(m_dev) + " is vendor device " +
                                             std::to_string(attr.device) + "; set ROCR/CUDA_VISIBLE_DEVICES consistently");
                }
                (void)cudaGetLastError();
                omp_target_free(probe, m_dev);
            }
        }
#endif
    }
    m_prof = Profiler(m_opts.profile);
}

template <typename T> std::string Network<T>::blas_backend() const {
    switch (m_blas) {
    case BlasKind::Omp: return "omp";
    case BlasKind::Tiled: return "tiled";
    case BlasKind::Cblas: return OMPNN_BLAS_NAME;
    case BlasKind::Vendor: return compiled_blas_backends().back();
    case BlasKind::Auto:
    default:
        if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
            return compiled_blas_backends().back();
#else
            return "omp";
#endif
        }
        return OMPNN_BLAS_NAME;
    }
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom,
                    const std::vector<std::vector<T>> &initial_weights, const std::vector<std::vector<T>> &initial_biases,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)) {
    init_device();
    init_parameters(&initial_weights, &initial_biases, 0);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, unsigned seed,
                    Options options)
    : m_layers(std::move(layers)), m_lr(learning_rate), m_reg(reg), m_mom(mom), m_bp(bp), m_adapt(adapt), m_stop(stop),
      m_opts(std::move(options)) {
    init_device();
    init_parameters(nullptr, nullptr, seed);
}

template <typename T>
Network<T>::Network(std::vector<LayerDescription> layers, T learning_rate, Regularization<T> reg, BackPropagation bp,
                    AdaptiveLearningRate<T> adapt, StopCriteria<T> stop, MomentumConfig<T> mom, Options options)
    : Network(std::move(layers), learning_rate, reg, bp, adapt, stop, mom, static_cast<unsigned>(std::time(nullptr)),
              std::move(options)) {}

template <typename T>
void Network<T>::init_parameters(const std::vector<std::vector<T>> *w, const std::vector<std::vector<T>> *b, unsigned seed) {
    if (m_layers.size() < 2)
        throw std::invalid_argument("ompnn: a network needs at least an input and an output layer");
    for (const auto &l : m_layers)
        if (l.neurons == 0)
            throw std::invalid_argument("ompnn: layer neuron count must be greater than 0");
    m_L = m_layers.size() - 1;
    const bool random_init = (w == nullptr || b == nullptr);
    if (!random_init) {
        if (w->size() != m_L)
            throw std::invalid_argument("ompnn: initial weights have " + std::to_string(w->size()) + " layers, expected " +
                                        std::to_string(m_L));
        if (b->size() != m_L)
            throw std::invalid_argument("ompnn: initial biases have " + std::to_string(b->size()) + " layers, expected " +
                                        std::to_string(m_L));
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
            const std::size_t n_out = m_layers[l + 1].neurons;
            if ((*w)[l].size() != nw)
                throw std::invalid_argument("ompnn: initial weights of layer " + std::to_string(l) + " have " +
                                            std::to_string((*w)[l].size()) + " values, expected " + std::to_string(nw));
            if ((*b)[l].size() != n_out)
                throw std::invalid_argument("ompnn: initial biases of layer " + std::to_string(l) + " have " +
                                            std::to_string((*b)[l].size()) + " values, expected " + std::to_string(n_out));
        }
    }
    std::mt19937 rng(seed == 0 ? static_cast<unsigned>(std::time(nullptr)) : seed);
    std::uniform_real_distribution<T> dist(T(-0.1), T(0.1));
    std::vector<std::vector<T>> host_w(m_L), host_b(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::size_t nw = std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons;
        const std::size_t n_out = m_layers[l + 1].neurons;
        if (random_init) {
            host_w[l].resize(nw);
            host_b[l].resize(n_out);
            for (auto &x : host_w[l])
                x = dist(rng);
            for (auto &x : host_b[l])
                x = dist(rng);
        } else {
            host_w[l] = (*w)[l];
            host_b[l] = (*b)[l];
        }
    }
    m_W.clear();
    m_b.clear();
    m_mW.clear();
    m_vW.clear();
    m_mb.clear();
    m_vb.clear();
    for (std::size_t l = 0; l < m_L; ++l) {
        const std::size_t nw = host_w[l].size(), n_out = host_b[l].size();
        m_W.emplace_back(nw, m_dev);
        m_b.emplace_back(n_out, m_dev);
        m_mW.emplace_back(nw, m_dev);
        m_vW.emplace_back(nw, m_dev);
        m_mb.emplace_back(n_out, m_dev);
        m_vb.emplace_back(n_out, m_dev);
        m_prof.record(Phase::H2D, [&] { m_W[l].copy_in(host_w[l].data(), nw); }, nw * sizeof(T));
        m_prof.record(Phase::H2D, [&] { m_b[l].copy_in(host_b[l].data(), n_out); }, n_out * sizeof(T));
        m_prof.record(Phase::Other, [&] { kernels::fill(m_gpu, m_dev, m_tl, m_mW[l].data(), T(0), nw); });
        m_prof.record(Phase::Other, [&] { kernels::fill(m_gpu, m_dev, m_tl, m_vW[l].data(), T(0), nw); });
        m_prof.record(Phase::Other, [&] { kernels::fill(m_gpu, m_dev, m_tl, m_mb[l].data(), T(0), n_out); });
        m_prof.record(Phase::Other, [&] { kernels::fill(m_gpu, m_dev, m_tl, m_vb[l].data(), T(0), n_out); });
    }
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(std::move(host_w));
    m_biases_history.push_back(std::move(host_b));
}

template <typename T> std::size_t Network<T>::parameter_count() const {
    std::size_t n = 0;
    for (std::size_t l = 0; l < m_L; ++l)
        n += std::size_t(m_layers[l].neurons) * m_layers[l + 1].neurons + m_layers[l + 1].neurons;
    return n;
}

// ---------------------------------------------------------------- memory

template <typename T> void Network<T>::ensure_workspace(std::size_t batch) {
    if (batch <= m_cap && !m_act.empty())
        return;
    release_workspace();
    try {
        for (std::size_t l = 0; l <= m_L; ++l) {
            const bool needed = (l > 0) || !m_opts.direct_input;
            m_act.emplace_back(needed ? std::size_t(m_layers[l].neurons) * batch : 0, m_dev);
        }
        for (std::size_t l = 0; l < m_L; ++l) {
            const std::size_t n_in = m_layers[l].neurons, n_out = m_layers[l + 1].neurons;
            m_net.emplace_back(n_out * batch, m_dev);
            m_delta.emplace_back(n_out * batch, m_dev);
            m_gW.emplace_back(n_out * n_in, m_dev);
            m_gb.emplace_back(n_out, m_dev);
        }
        m_ones = Buf(batch, m_dev);
        kernels::fill(m_gpu, m_dev, m_tl, m_ones.data(), T(1), batch);
    } catch (...) {
        release_workspace();
        throw;
    }
    m_cap = batch;
}

template <typename T> void Network<T>::release_workspace() {
    m_act.clear();
    m_net.clear();
    m_delta.clear();
    m_gW.clear();
    m_gb.clear();
    m_ones.release();
    m_cap = 0;
}

template <typename T> std::vector<std::vector<T>> Network<T>::weights() const {
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_W[l].size());
        m_W[l].copy_out(out[l].data(), m_W[l].size());
    }
    return out;
}

template <typename T> std::vector<std::vector<T>> Network<T>::biases() const {
    std::vector<std::vector<T>> out(m_L);
    for (std::size_t l = 0; l < m_L; ++l) {
        out[l].resize(m_b[l].size());
        m_b[l].copy_out(out[l].data(), m_b[l].size());
    }
    return out;
}

template <typename T> void Network<T>::snapshot_history() {
    m_weights_history.push_back(weights());
    m_biases_history.push_back(biases());
}

template <typename T>
void Network<T>::set_weights_biases(const std::vector<std::vector<T>> &new_weights,
                                    const std::vector<std::vector<T>> &new_biases) {
    if (new_weights.size() != m_L || new_biases.size() != m_L)
        throw std::invalid_argument("ompnn: layer count mismatch in set_weights_biases");
    for (std::size_t l = 0; l < m_L; ++l) {
        if (new_weights[l].size() != m_W[l].size())
            throw std::invalid_argument("ompnn: weight size mismatch for layer " + std::to_string(l));
        if (new_biases[l].size() != m_b[l].size())
            throw std::invalid_argument("ompnn: bias size mismatch for layer " + std::to_string(l));
    }
    for (std::size_t l = 0; l < m_L; ++l) {
        m_W[l].copy_in(new_weights[l].data(), m_W[l].size());
        m_b[l].copy_in(new_biases[l].data(), m_b[l].size());
    }
    m_weights_history.clear();
    m_biases_history.clear();
    m_weights_history.push_back(new_weights);
    m_biases_history.push_back(new_biases);
}

// ---------------------------------------------------------------- BLAS dispatch

template <typename T>
void Network<T>::gemm(bool ta, bool tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta,
                      T *c, int ldc) {
    if (m_blas == BlasKind::Omp) {
        ompblas::gemm(m_gpu, m_dev, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    if (m_blas == BlasKind::Tiled) {
        ompblas::gemm_tiled(m_gpu, m_dev, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        m_vendor->gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#else
        ompblas::gemm(true, m_dev, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
        return;
    }
    cblas::gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename T>
void Network<T>::gemv(bool ta, int m, int n, T alpha, const T *a, int lda, const T *x, T beta, T *y) {
    if (m_blas == BlasKind::Omp || m_blas == BlasKind::Tiled) {
        ompblas::gemv(m_gpu, m_dev, ta, m, n, alpha, a, lda, x, beta, y);
        return;
    }
    if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        m_vendor->gemv(ta, m, n, alpha, a, lda, x, beta, y);
#else
        ompblas::gemv(true, m_dev, ta, m, n, alpha, a, lda, x, beta, y);
#endif
        return;
    }
    cblas::gemv(ta, m, n, alpha, a, lda, x, beta, y);
}

template <typename T> double Network<T>::asum(int n, const T *x) {
    if (m_blas == BlasKind::Omp || m_blas == BlasKind::Tiled)
        return ompblas::asum(m_gpu, m_dev, n, x);
    if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        return static_cast<double>(m_vendor->asum(n, x));
#else
        return ompblas::asum(true, m_dev, n, x);
#endif
    }
    return static_cast<double>(cblas::asum(n, x));
}

template <typename T> double Network<T>::sumsq(int n, const T *x) {
    if (m_blas == BlasKind::Omp || m_blas == BlasKind::Tiled)
        return ompblas::sumsq(m_gpu, m_dev, n, x);
    if (m_gpu) {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        const double r = static_cast<double>(m_vendor->nrm2(n, x));
        return r * r;
#else
        return ompblas::sumsq(true, m_dev, n, x);
#endif
    }
    const double r = static_cast<double>(cblas::nrm2(n, x));
    return r * r;
}

// ---------------------------------------------------------------- phases

template <typename T> void Network<T>::forward_layer(std::size_t l, const T *in, std::size_t B) {
    const std::size_t M = m_layers[l + 1].neurons, K = m_layers[l].neurons;
    T *net = m_net[l].data();
    T *out = m_act[l + 1].data();
    const T *bias = m_b[l].data();
    m_prof.record(Phase::Gemm, [&] {
        gemm(false, false, int(M), int(B), int(K), T(1), m_W[l].data(), int(M), in, int(K), T(0), net, int(M));
    });
    const std::size_t n = M * B;
    const ActivationType act = m_layers[l + 1].activation;
    m_prof.record(Phase::Act, [&] {
        if (m_opts.specialized_kernels) {
            dispatch_activation(act, [&](auto tag) {
                constexpr ActivationType A = decltype(tag)::value;
                kernels::activate(m_gpu, m_dev, m_tl, net, out, bias, n, M, ActF<A>{});
            });
        } else {
            kernels::activate(m_gpu, m_dev, m_tl, net, out, bias, n, M, ActRT{act});
        }
    });
}

template <typename T> double Network<T>::output_delta_loss(const T *targets, std::size_t B) {
    const std::size_t l = m_L - 1;
    const std::size_t M = m_layers[m_L].neurons;
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[m_L].data();
    T *delta = m_delta[l].data();
    const ActivationType act = m_layers[m_L].activation;
    const bool from_out = m_opts.derivative_from_output;
    double loss = 0.0;
    m_prof.record(Phase::Loss, [&] {
        if (m_opts.specialized_kernels) {
            loss = dispatch_activation(act, [&](auto tag) {
                constexpr ActivationType A = decltype(tag)::value;
                if (from_out)
                    return kernels::output_delta_loss(m_gpu, m_dev, m_tl, m_opts.loss_reduction, targets, out, net, delta, n,
                                                      DfF<A, true>{});
                return kernels::output_delta_loss(m_gpu, m_dev, m_tl, m_opts.loss_reduction, targets, out, net, delta, n,
                                                  DfF<A, false>{});
            });
        } else {
            loss = kernels::output_delta_loss(m_gpu, m_dev, m_tl, m_opts.loss_reduction, targets, out, net, delta, n,
                                              DfRT{act, from_out});
        }
    });
    return loss;
}

template <typename T> void Network<T>::hidden_delta(std::size_t l, std::size_t B) {
    const std::size_t M = m_layers[l + 1].neurons, K = m_layers[l + 2].neurons;
    T *delta = m_delta[l].data();
    m_prof.record(Phase::Gemm, [&] {
        gemm(true, false, int(M), int(B), int(K), T(1), m_W[l + 1].data(), int(K), m_delta[l + 1].data(), int(K), T(0),
             delta, int(M));
    });
    const std::size_t n = M * B;
    const T *net = m_net[l].data();
    const T *out = m_act[l + 1].data();
    const ActivationType act = m_layers[l + 1].activation;
    const bool from_out = m_opts.derivative_from_output;
    m_prof.record(Phase::Delta, [&] {
        if (m_opts.specialized_kernels) {
            dispatch_activation(act, [&](auto tag) {
                constexpr ActivationType A = decltype(tag)::value;
                if (from_out)
                    kernels::hidden_delta(m_gpu, m_dev, m_tl, delta, out, net, n, DfF<A, true>{});
                else
                    kernels::hidden_delta(m_gpu, m_dev, m_tl, delta, out, net, n, DfF<A, false>{});
            });
        } else {
            kernels::hidden_delta(m_gpu, m_dev, m_tl, delta, out, net, n, DfRT{act, from_out});
        }
    });
}

template <typename T> void Network<T>::gradients(std::size_t l, const T *in, std::size_t B) {
    const std::size_t M = m_layers[l + 1].neurons, N = m_layers[l].neurons;
    const T *delta = m_delta[l].data();
    const T invB = T(1) / static_cast<T>(B);
    m_prof.record(Phase::Gemm, [&] {
        gemm(false, true, int(M), int(N), int(B), invB, delta, int(M), in, int(N), T(0), m_gW[l].data(), int(M));
    });
    T *gb = m_gb[l].data();
    m_prof.record(Phase::BiasGrad, [&] {
        if (m_opts.bias_gemv)
            gemv(false, int(M), int(B), invB, delta, int(M), m_ones.data(), T(0), gb);
        else
            kernels::bias_grad(m_gpu, m_dev, m_tl, delta, gb, M, B);
    });
}

template <typename T> void Network<T>::update(std::size_t l, unsigned adam_step, unsigned epoch, unsigned max_epochs) {
    using Strategy = typename AdaptiveLearningRate<T>::Strategy;
    kernels::UpdateParams<T> u;
    u.strategy = static_cast<int>(m_adapt.strategy);
    u.momentum = m_mom.type == MomentumConfig<T>::Type::Classical;
    u.l1 = m_reg.uses_l1();
    u.l2 = m_reg.uses_l2();
    u.host_corr = m_opts.host_adam_correction;
    u.mu = m_mom.momentum_rate;
    u.lr = m_lr;
    u.eps = m_adapt.epsilon;
    u.beta1 = m_adapt.beta1;
    u.beta2 = m_adapt.beta2;
    u.lambda1 = m_reg.lambda1;
    u.lambda2 = m_reg.lambda2;
    u.lr_eff = m_lr;
    if (m_adapt.strategy == Strategy::LinearDecay) {
        const T gamma = (max_epochs > 1) ? static_cast<T>(epoch) / static_cast<T>(max_epochs - 1) : T(0);
        u.lr_eff = m_lr * (T(1) - gamma) + gamma * m_adapt.final_lr;
    }
    const bool adam = m_adapt.strategy == Strategy::Adam;
    u.t = static_cast<T>(adam_step);
    u.c1 = (adam && u.host_corr) ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(u.beta1), adam_step))) : T(0);
    u.c2 = (adam && u.host_corr) ? T(1) / (T(1) - static_cast<T>(std::pow(static_cast<double>(u.beta2), adam_step))) : T(0);
    m_prof.record(Phase::Update, [&] {
        kernels::update(m_gpu, m_dev, m_tl, m_W[l].data(), m_gW[l].data(), m_mW[l].data(), m_vW[l].data(), m_W[l].size(),
                        m_b[l].data(), m_gb[l].data(), m_mb[l].data(), m_vb[l].data(), m_b[l].size(), u);
    });
}

template <typename T> double Network<T>::penalty() {
    double p = 0.0;
    for (std::size_t l = 0; l < m_L; ++l) {
        const int n = int(m_W[l].size());
        if (m_reg.uses_l1())
            m_prof.record(Phase::Reg, [&] { p += static_cast<double>(m_reg.lambda1) * asum(n, m_W[l].data()); });
        if (m_reg.uses_l2())
            m_prof.record(Phase::Reg, [&] { p += 0.5 * static_cast<double>(m_reg.lambda2) * sumsq(n, m_W[l].data()); });
    }
    return p;
}

// ---------------------------------------------------------------- training

template <typename T>
std::vector<T> Network<T>::train(const std::vector<T> &input_samples, const std::vector<T> &target_samples,
                                 unsigned num_samples, unsigned batch_size, unsigned max_epochs) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        throw std::invalid_argument("ompnn: number of samples cannot be zero");
    if (batch_size == 0)
        throw std::invalid_argument("ompnn: batch size must be greater than 0");
    if (max_epochs == 0)
        throw std::invalid_argument("ompnn: max_epochs must be greater than 0");
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("ompnn: input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in));
    if (target_samples.size() != std::size_t(num_samples) * n_out)
        throw std::invalid_argument("ompnn: targets have " + std::to_string(target_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_out));
    const std::size_t N = num_samples;
    const std::size_t B = std::min<std::size_t>(batch_size, N);
    const std::size_t n_batches = (N + B - 1) / B;
    const bool direct = m_opts.direct_input;
    const bool use_reg = m_reg.uses_l1() || m_reg.uses_l2();

    ensure_workspace(B);
    m_adam_step = 0;

    // ---- dataset upload (once) ----
    Buf X(N * n_in, m_dev), Y(N * n_out, m_dev), Xs, Ys;
    detail::Buffer<std::uint32_t> perm_dev;
    m_prof.record(Phase::H2D, [&] { X.copy_in(input_samples.data(), N * n_in); }, N * n_in * sizeof(T));
    m_prof.record(Phase::H2D, [&] { Y.copy_in(target_samples.data(), N * n_out); }, N * n_out * sizeof(T));
    if (m_opts.shuffle) {
        Xs = Buf(N * n_in, m_dev);
        Ys = Buf(N * n_out, m_dev);
        perm_dev = detail::Buffer<std::uint32_t>(N, m_dev);
    }
    std::mt19937 shuffle_rng(m_opts.shuffle_seed);

    std::vector<T> losses;
    losses.reserve(max_epochs);
    for (unsigned epoch = 0; epoch < max_epochs; ++epoch) {
        const auto t_epoch = Profiler::clock::now();
        const T *Xsrc = X.data();
        const T *Ysrc = Y.data();
        if (m_opts.shuffle) {
            auto perm = detail::shuffle_permutation(static_cast<std::uint32_t>(N), shuffle_rng);
            m_prof.record(Phase::H2D, [&] { perm_dev.copy_in(perm.data(), N); }, N * sizeof(std::uint32_t));
            m_prof.record(Phase::Other, [&] { kernels::gather(m_gpu, m_dev, m_tl, X.data(), Xs.data(), perm_dev.data(), n_in, N * n_in); });
            m_prof.record(Phase::Other, [&] { kernels::gather(m_gpu, m_dev, m_tl, Y.data(), Ys.data(), perm_dev.data(), n_out, N * n_out); });
            Xsrc = Xs.data();
            Ysrc = Ys.data();
        }
        double data_loss = 0.0;
        for (std::size_t bi = 0; bi < n_batches; ++bi) {
            const std::size_t start = bi * B;
            const std::size_t cur = std::min(B, N - start);
            if (m_adapt.strategy == AdaptiveLearningRate<T>::Strategy::Adam)
                ++m_adam_step;
            const T *x_batch = Xsrc + start * n_in;
            const T *targets = Ysrc + start * n_out;
            const T *in = x_batch;
            if (!direct) {
                m_prof.record(Phase::Other, [&] { kernels::copy_d2d(m_gpu, m_dev, m_tl, m_act[0].data(), x_batch, cur * n_in); });
                in = m_act[0].data();
            }
            for (std::size_t l = 0; l < m_L; ++l)
                forward_layer(l, (l == 0) ? in : m_act[l].data(), cur);
            data_loss += output_delta_loss(targets, cur);
            for (std::size_t l = m_L - 1; l-- > 0;)
                hidden_delta(l, cur);
            for (std::size_t l = 0; l < m_L; ++l)
                gradients(l, (l == 0) ? in : m_act[l].data(), cur);
            for (std::size_t l = 0; l < m_L; ++l)
                update(l, m_adam_step, epoch, max_epochs);
        }
        const double total = data_loss / static_cast<double>(N) + (use_reg ? penalty() : 0.0);
        losses.push_back(static_cast<T>(total));
        if (m_opts.record_history)
            snapshot_history();
        // always recorded (a clock read per epoch): the steady-state epoch metric needs it
        m_prof.profile().epoch_wall_ns.push_back(Profiler::elapsed_ns(t_epoch));
        ++m_prof.profile().epochs;
        m_prof.profile().batches += n_batches;
        bool stop = false;
        if (m_stop.type == StopCriteria<T>::Type::MinError) {
            stop = losses.back() < m_stop.threshold;
        } else if (m_stop.type == StopCriteria<T>::Type::MinErrorChange && epoch > 0) {
            stop = std::abs(losses[epoch - 1] - losses[epoch]) < m_stop.threshold;
        }
        if (stop)
            break;
    }
    if (!m_opts.record_history)
        snapshot_history();
    if (!m_opts.persistent_workspace)
        release_workspace();
    m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    return losses;
}

// ---------------------------------------------------------------- inference

template <typename T>
std::vector<T> Network<T>::predict(const std::vector<T> &input_samples, unsigned num_samples, unsigned batch_size) {
    const auto t_start = Profiler::clock::now();
    const std::size_t n_in = m_layers.front().neurons, n_out = m_layers.back().neurons;
    if (num_samples == 0)
        return {};
    if (input_samples.size() != std::size_t(num_samples) * n_in)
        throw std::invalid_argument("ompnn: predict input has " + std::to_string(input_samples.size()) + " values, expected " +
                                    std::to_string(std::size_t(num_samples) * n_in));
    const std::size_t N = num_samples;
    const std::size_t B = (batch_size == 0) ? N : std::min<std::size_t>(batch_size, N);
    ensure_workspace(B);
    Buf X(N * n_in, m_dev);
    m_prof.record(Phase::H2D, [&] { X.copy_in(input_samples.data(), N * n_in); }, N * n_in * sizeof(T));
    std::vector<T> out(N * n_out);
    for (std::size_t start = 0; start < N; start += B) {
        const std::size_t cur = std::min(B, N - start);
        const T *in = X.data() + start * n_in;
        if (!m_opts.direct_input) {
            m_prof.record(Phase::Other, [&] { kernels::copy_d2d(m_gpu, m_dev, m_tl, m_act[0].data(), in, cur * n_in); });
            in = m_act[0].data();
        }
        for (std::size_t l = 0; l < m_L; ++l)
            forward_layer(l, (l == 0) ? in : m_act[l].data(), cur);
        m_prof.record(Phase::D2H, [&] { m_act[m_L].copy_out(out.data() + start * n_out, cur * n_out); },
                      cur * n_out * sizeof(T));
    }
    if (!m_opts.persistent_workspace)
        release_workspace();
    m_prof.profile().wall_ns += Profiler::elapsed_ns(t_start);
    return out;
}

} // namespace ompnn
