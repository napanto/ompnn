// SPDX-License-Identifier: LGPL-3.0-only
// ompnn - BLAS for the two execution paths:
//   host  : CBLAS (OpenBLAS or Intel oneMKL, chosen at link time, OMPNN_BLAS)
//   target: cuBLAS (OMPNN_TARGET_NVIDIA) or hipBLAS/rocBLAS (OMPNN_TARGET_AMD) on
//           the device pointers obtained from omp_target_alloc - the "interop"
//           path that keeps the GEMM library identical to syclnn's and cudann's;
//   omp   : a hand-written OpenMP GEMM/GEMV (`Options::blas = "omp"`), the pure
//           pragma-based measurement, on both paths.
// Every call is synchronous with respect to the host (vendor calls are followed
// by a device synchronisation), because OpenMP offers no portable way to order
// a foreign stream with its target regions without the 5.1 interop API.
#pragma once

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cblas.h>

#include "ompnn/vendor.hpp"

namespace ompnn {

#ifndef OMPNN_BLAS_NAME
#define OMPNN_BLAS_NAME "cblas"
#endif

inline std::vector<std::string> compiled_blas_backends() {
    std::vector<std::string> v{OMPNN_BLAS_NAME, "omp", "tiled"};
#if defined(OMPNN_TARGET_NVIDIA)
    v.push_back("cublas");
#elif defined(OMPNN_TARGET_AMD)
    v.push_back("rocblas");
#endif
    return v;
}

enum class BlasKind { Auto, Cblas, Omp, Tiled, Vendor };

inline BlasKind parse_blas(const std::string &name) {
    if (name.empty() || name == "auto")
        return BlasKind::Auto;
    if (name == "omp" || name == "openmp")
        return BlasKind::Omp;
    if (name == "tiled" || name == "handwritten")
        return BlasKind::Tiled;
    if (name == "openblas" || name == "mkl" || name == "mklcpu" || name == "netlib" || name == "cblas") {
        if (name != "cblas" && name != OMPNN_BLAS_NAME && !(name == "mklcpu" && std::string(OMPNN_BLAS_NAME) == "mkl") &&
            !(name == "netlib" && std::string(OMPNN_BLAS_NAME) == "openblas"))
            throw std::invalid_argument("ompnn: this build links " OMPNN_BLAS_NAME ", not '" + name + "'");
        return BlasKind::Cblas;
    }
    if (name == "cublas" || name == "rocblas" || name == "hipblas") {
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
        return BlasKind::Vendor;
#else
        throw std::invalid_argument("ompnn: this build has no GPU BLAS (" + name + ")");
#endif
    }
    throw std::invalid_argument("ompnn: unknown BLAS backend '" + name + "'");
}

// ---------------------------------------------------------------- host CBLAS
namespace cblas {
inline void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                 float beta, float *c, int ldc) {
    cblas_sgemm(CblasColMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
}
inline void gemm(bool ta, bool tb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc) {
    cblas_dgemm(CblasColMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b,
                ldb, beta, c, ldc);
}
inline void gemv(bool ta, int m, int n, float alpha, const float *a, int lda, const float *x, float beta, float *y) {
    cblas_sgemv(CblasColMajor, ta ? CblasTrans : CblasNoTrans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}
inline void gemv(bool ta, int m, int n, double alpha, const double *a, int lda, const double *x, double beta, double *y) {
    cblas_dgemv(CblasColMajor, ta ? CblasTrans : CblasNoTrans, m, n, alpha, a, lda, x, 1, beta, y, 1);
}
inline float asum(int n, const float *x) { return cblas_sasum(n, x, 1); }
inline double asum(int n, const double *x) { return cblas_dasum(n, x, 1); }
inline float nrm2(int n, const float *x) { return cblas_snrm2(n, x, 1); }
inline double nrm2(int n, const double *x) { return cblas_dnrm2(n, x, 1); }
} // namespace cblas

// ---------------------------------------------------------------- hand-written OpenMP BLAS
namespace ompblas {
/// C = alpha op(A) op(B) + beta C, column-major, one work-item per C element
/// (coalesced along m).  `gpu`: target teams on device `dev`, else host threads.
template <typename T>
inline void gemm(bool gpu, int dev, bool ta, bool tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b,
                 int ldb, T beta, T *c, int ldc) {
    const std::size_t total = std::size_t(m) * n;
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) is_device_ptr(a, b, c)
        for (std::size_t idx = 0; idx < total; ++idx) {
            const int j = int(idx / m), i = int(idx - std::size_t(j) * m);
            T acc = T(0);
            for (int p = 0; p < k; ++p) {
                const T av = ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda];
                const T bv = tb ? b[j + std::size_t(p) * ldb] : b[p + std::size_t(j) * ldb];
                acc += av * bv;
            }
            c[i + std::size_t(j) * ldc] = beta == T(0) ? alpha * acc : alpha * acc + beta * c[i + std::size_t(j) * ldc];
        }
    } else {
#pragma omp parallel for
        for (int j = 0; j < n; ++j) {
#pragma omp simd
            for (int i = 0; i < m; ++i) {
                T acc = T(0);
                for (int p = 0; p < k; ++p) {
                    const T av = ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda];
                    const T bv = tb ? b[j + std::size_t(p) * ldb] : b[p + std::size_t(j) * ldb];
                    acc += av * bv;
                }
                c[i + std::size_t(j) * ldc] = beta == T(0) ? alpha * acc : alpha * acc + beta * c[i + std::size_t(j) * ldc];
            }
        }
    }
}
/// The 16x16 tiled GEMM (same tile as syclnn's local-memory and cudann's
/// shared-memory versions): one team per C tile, team-shared tiles of op(A) and
/// op(B) refilled every 16 k-steps by a parallel loop, the team's threads each
/// own one C element.  On the host the same loop nest runs with the tiles in
/// the cache.
constexpr int TILE = 16;
#define OMPNN_PRAGMA_(x) _Pragma(#x)
#define OMPNN_PRAGMA(x) OMPNN_PRAGMA_(x)
#if defined(__clang__)
#define OMPNN_PTEAM_CLAUSES uses_allocators(omp_pteam_mem_alloc)
#else
#define OMPNN_PTEAM_CLAUSES
#endif
template <typename T>
inline void gemm_tiled(bool gpu, int dev, bool ta, bool tb, int m, int n, int k, T alpha, const T *a, int lda, const T *b,
                       int ldb, T beta, T *c, int ldc) {
    const int tiles_m = (m + TILE - 1) / TILE, tiles_n = (n + TILE - 1) / TILE;
    if (gpu) {
        // one team per C tile, one parallel region per team: the TILE*TILE threads each
        // own a C element and keep the accumulator in a register; the op(A)/op(B) tiles
        // live in the team's low-latency memory (omp_pteam_mem_alloc = shared memory / LDS)
        T As[TILE][TILE], Bs[TILE][TILE];
        // clang requires the predefined allocator to be listed in uses_allocators, gcc 14 rejects the clause
        OMPNN_PRAGMA(omp target teams distribute collapse(2) device(dev) is_device_ptr(a, b, c) thread_limit(256)
                         OMPNN_PTEAM_CLAUSES private(As, Bs) allocate(omp_pteam_mem_alloc : As, Bs))
        for (int tj = 0; tj < tiles_n; ++tj) {
            for (int ti = 0; ti < tiles_m; ++ti) {
#pragma omp parallel num_threads(TILE * TILE) shared(As, Bs)
                {
                    const int tid = omp_get_thread_num();
                    const int li = tid % TILE, lj = tid / TILE;
                    const int i = ti * TILE + li, j = tj * TILE + lj;
                    T acc = T(0);
                    for (int t = 0; t < k; t += TILE) {
                        int p = t + lj;
                        As[li][lj] = (i < m && p < k) ? (ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda]) : T(0);
                        p = t + li;
                        Bs[li][lj] = (p < k && j < n) ? (tb ? b[j + std::size_t(p) * ldb] : b[p + std::size_t(j) * ldb]) : T(0);
#pragma omp barrier
                        for (int kk = 0; kk < TILE; ++kk)
                            acc += As[li][kk] * Bs[kk][lj];
#pragma omp barrier
                    }
                    if (i < m && j < n) {
                        T *cc = c + i + std::size_t(j) * ldc;
                        *cc = (beta == T(0)) ? alpha * acc : alpha * acc + beta * (*cc);
                    }
                }
            }
        }
    } else {
        // cache-tiled host version, Bs stored transposed so the k loop is contiguous
#pragma omp parallel for collapse(2)
        for (int tj = 0; tj < tiles_n; ++tj) {
            for (int ti = 0; ti < tiles_m; ++ti) {
                T As[TILE][TILE], BsT[TILE][TILE], Cs[TILE][TILE] = {};
                for (int t = 0; t < k; t += TILE) {
                    for (int lj = 0; lj < TILE; ++lj)
                        for (int li = 0; li < TILE; ++li) {
                            const int i = ti * TILE + li, j = tj * TILE + lj;
                            int p = t + lj;
                            As[li][lj] = (i < m && p < k) ? (ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda]) : T(0);
                            p = t + li;
                            BsT[lj][li] = (p < k && j < n) ? (tb ? b[j + std::size_t(p) * ldb] : b[p + std::size_t(j) * ldb]) : T(0);
                        }
                    for (int lj = 0; lj < TILE; ++lj)
                        for (int li = 0; li < TILE; ++li) {
                            T acc = Cs[li][lj];
#pragma omp simd reduction(+ : acc)
                            for (int kk = 0; kk < TILE; ++kk)
                                acc += As[li][kk] * BsT[lj][kk];
                            Cs[li][lj] = acc;
                        }
                }
                for (int lj = 0; lj < TILE; ++lj)
                    for (int li = 0; li < TILE; ++li) {
                        const int i = ti * TILE + li, j = tj * TILE + lj;
                        if (i < m && j < n) {
                            T *cc = c + i + std::size_t(j) * ldc;
                            *cc = (beta == T(0)) ? alpha * Cs[li][lj] : alpha * Cs[li][lj] + beta * (*cc);
                        }
                    }
            }
        }
    }
}

template <typename T>
inline void gemv(bool gpu, int dev, bool ta, int m, int n, T alpha, const T *a, int lda, const T *x, T beta, T *y) {
    const int rows = ta ? n : m, inner = ta ? m : n;
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) is_device_ptr(a, x, y)
        for (int i = 0; i < rows; ++i) {
            T acc = T(0);
            for (int p = 0; p < inner; ++p)
                acc += (ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda]) * x[p];
            y[i] = beta == T(0) ? alpha * acc : alpha * acc + beta * y[i];
        }
    } else {
#pragma omp parallel for simd
        for (int i = 0; i < rows; ++i) {
            T acc = T(0);
            for (int p = 0; p < inner; ++p)
                acc += (ta ? a[p + std::size_t(i) * lda] : a[i + std::size_t(p) * lda]) * x[p];
            y[i] = beta == T(0) ? alpha * acc : alpha * acc + beta * y[i];
        }
    }
}
template <typename T> inline double asum(bool gpu, int dev, int n, const T *x) {
    double s = 0.0;
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) is_device_ptr(x) reduction(+ : s) map(tofrom : s)
        for (int i = 0; i < n; ++i)
            s += double(x[i] < T(0) ? -x[i] : x[i]);
    } else {
#pragma omp parallel for simd reduction(+ : s)
        for (int i = 0; i < n; ++i)
            s += double(x[i] < T(0) ? -x[i] : x[i]);
    }
    return s;
}
template <typename T> inline double sumsq(bool gpu, int dev, int n, const T *x) {
    double s = 0.0;
    if (gpu) {
#pragma omp target teams distribute parallel for simd device(dev) is_device_ptr(x) reduction(+ : s) map(tofrom : s)
        for (int i = 0; i < n; ++i)
            s += double(x[i]) * double(x[i]);
    } else {
#pragma omp parallel for simd reduction(+ : s)
        for (int i = 0; i < n; ++i)
            s += double(x[i]) * double(x[i]);
    }
    return s;
}
} // namespace ompblas

// ---------------------------------------------------------------- vendor BLAS on the device pointers
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
namespace vendor {
inline void check(cublasStatus_t st, const char *what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("ompnn: ") + what + " failed (status " + std::to_string(int(st)) + ")");
}
inline void sync(const char *what) {
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(std::string("ompnn: device synchronisation after ") + what + " failed");
}
class Handle {
  public:
    explicit Handle(int dev) {
        (void)cudaSetDevice(dev);
        check(cublasCreate(&m_h), "cublasCreate");
        check(cublasSetPointerMode(m_h, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    }
    ~Handle() {
        if (m_h)
            (void)cublasDestroy(m_h);
    }
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    static cublasOperation_t op(bool t) { return t ? CUBLAS_OP_T : CUBLAS_OP_N; }
    void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
              float beta, float *c, int ldc) {
        check(cublasSgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasSgemm");
        sync("gemm");
    }
    void gemm(bool ta, bool tb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
              double beta, double *c, int ldc) {
        check(cublasDgemm(m_h, op(ta), op(tb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc), "cublasDgemm");
        sync("gemm");
    }
    void gemv(bool ta, int m, int n, float alpha, const float *a, int lda, const float *x, float beta, float *y) {
        check(cublasSgemv(m_h, op(ta), m, n, &alpha, a, lda, x, 1, &beta, y, 1), "cublasSgemv");
        sync("gemv");
    }
    void gemv(bool ta, int m, int n, double alpha, const double *a, int lda, const double *x, double beta, double *y) {
        check(cublasDgemv(m_h, op(ta), m, n, &alpha, a, lda, x, 1, &beta, y, 1), "cublasDgemv");
        sync("gemv");
    }
    float asum(int n, const float *x) {
        float r = 0;
        check(cublasSasum(m_h, n, x, 1, &r), "cublasSasum");
        return r;
    }
    double asum(int n, const double *x) {
        double r = 0;
        check(cublasDasum(m_h, n, x, 1, &r), "cublasDasum");
        return r;
    }
    float nrm2(int n, const float *x) {
        float r = 0;
        check(cublasSnrm2(m_h, n, x, 1, &r), "cublasSnrm2");
        return r;
    }
    double nrm2(int n, const double *x) {
        double r = 0;
        check(cublasDnrm2(m_h, n, x, 1, &r), "cublasDnrm2");
        return r;
    }

  private:
    cublasHandle_t m_h = nullptr;
};
} // namespace vendor
#endif

} // namespace ompnn
