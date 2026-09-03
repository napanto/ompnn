# Changelog

## [0.1.0] - 2026-09

First release: OpenMP translation of syclnn 0.2.0 with the same public API, file
layout and numerics.

- `ompnn/config.hpp`: identical to syclnn's (namespace apart).
- `ompnn/activations.hpp`: the same functions inside `omp declare target`, plus
  functor forms usable in target regions (no lambdas there: gcc's nvptx/amdgcn
  offloading rejects them).
- `ompnn/device.hpp`: the host plus every `omp_get_num_devices()` device;
  vendor names/memory through the CUDA/HIP runtime when built for a GPU.
- `ompnn/blas.hpp`: CBLAS on the host (OpenBLAS or oneMKL, `OMPNN_BLAS`),
  cuBLAS/hipBLAS on the `omp_target_alloc` pointers (interop, synchronised
  after every call), and a hand-written OpenMP GEMM/GEMV/asum (`blas="omp"`).
- `ompnn/profile.hpp`: host timers around each synchronous region.
- `ompnn/network.hpp`: every kernel in a host (`parallel for simd`) and a target
  (`target teams distribute parallel for simd` + `is_device_ptr`) flavour;
  reductions for the loss (`loss_reduction=false` = `omp atomic` per element),
  fused update loop, gather for shuffling; all syclnn ablation switches with an
  OpenMP meaning are honoured (`bias_gemv`, `direct_input`,
  `specialized_kernels`, `derivative_from_output`, `host_adam_correction`,
  `workgroup_size` → `omp_set_teams_thread_limit`, `persistent_workspace`,
  `loss_reduction`); `queue`, `streams`, `fine_deps`, `join_kernels`,
  `pinned_host` have no counterpart; only `memory=device` on the target path.
- Build: `OMPNN_TARGET=cpu|nvidia|amd`, `OMPNN_OFFLOAD_ARCH`, `OMPNN_BLAS`,
  per-compiler offload flags (clang `-fopenmp-targets`/`--offload-arch`, gcc
  `-foffload`, nvc++ `-mp=gpu`), `build_info()` with compiler / flags / target /
  CBLAS / `_OPENMP` / device count; same pybind11 module layout as syclnn,
  tests through `fnn-testkit`, C++ driver, ctest, CI, container image.
