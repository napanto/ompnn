# Changelog

## [0.1.0] - 2026-09

First release: OpenMP translation of syclnn 0.2.0 with the same public API, file
layout and numerics.

- `Options.blas = "tiled"`: hand-written BLAS (a 16x16 tiled GEMM with team-shared tiles in target teams (and a cache-tiled host loop),
  a row-per-work-item GEMV, reductions for asum/nrm2) for the "same kernel in
  the three programming models" comparison (E7 of the study).

- `ompnn/config.hpp`: identical to syclnn's (namespace apart).
- `ompnn/activations.hpp`: the same functions inside `omp declare target`, plus
  functor forms usable in target regions (no lambdas there: gcc's nvptx/amdgcn
  offloading rejects them).
- `ompnn/device.hpp`: the host plus every `omp_get_num_devices()` device;
  vendor names/memory through the CUDA/HIP runtime when built for a GPU.
- `ompnn/blas.hpp`: CBLAS on the host (OpenBLAS or oneMKL, `OMPNN_BLAS`),
  cuBLAS/hipBLAS on the `omp_target_alloc` pointers (interop, synchronised
  after every call), and a hand-written OpenMP GEMM/GEMV/asum (`blas="omp"`).
- `ompnn/profile.hpp`: host timers around each synchronous region (one
  `launches` count per kernel/BLAS call, like syclnn's per-event count).
- `ompnn/network.hpp`: every kernel in a host (`parallel for simd`) and a target
  (`target teams distribute parallel for simd` + `is_device_ptr`) flavour;
  reductions for the loss (`loss_reduction=false` = `omp atomic` per element),
  fused update loop, gather for shuffling; all syclnn ablation switches with an
  OpenMP meaning are honoured (`bias_gemv`, `direct_input`,
  `specialized_kernels`, `derivative_from_output`, `host_adam_correction`,
  `workgroup_size` → `thread_limit(N)` clause on every target kernel (default 256),
  `persistent_workspace`,
  `loss_reduction`); `queue`, `streams`, `fine_deps`, `join_kernels`,
  `pinned_host` have no counterpart; only `memory=device` on the target path.
- Kernel arguments (activation functors, the fused-update parameter struct) are
  `firstprivate` in the target regions: with the default data-sharing rules an
  aggregate is mapped `tofrom`, i.e. a device allocation, an H2D and a D2H copy
  per launch (found in code review, it inflated every element-wise phase).
- The OpenMP device number is checked against the CUDA/HIP ordinal of a probe
  allocation at construction (`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`
  can disagree); mismatch raises.
- `Options.sync_every` accepted for parity (every target operation is already
  synchronous, so it has no effect).
- `Options.sync_ops` accepted for parity; always reported `true` (every target
  region and vendor BLAS call is synchronous).
- Known numerical differences vs syclnn (within the parity tolerances):
  `sumsq` accumulates in double, the host tiled GEMM reassociates the 16-term
  partial sums (`simd reduction`), and gcc offload builds run the `omp atomic`
  loss and the serial bias gradient at one lane per wavefront (those two
  ablation rows are not compared with SYCL/CUDA).
- Build: the Python module carries an rpath to the directory of the CBLAS it was
  configured with. `libopenblas.so.0` used to resolve through the distro
  alternatives symlink to the pthread build even when `OMPNN_BLAS_ROOT` named the
  OpenMP one; the two thread pools contend (1.5x slower on a 16-core host).
- Build: `OMPNN_TARGET=cpu|nvidia|amd`, `OMPNN_OFFLOAD_ARCH`, `OMPNN_BLAS`,
  per-compiler offload flags (clang `-fopenmp-targets`/`--offload-arch`, gcc
  `-foffload`, nvc++ `-mp=gpu`), `build_info()` with compiler / flags / target /
  CBLAS / `_OPENMP` / device count; same pybind11 module layout as syclnn,
  tests through `fnn-testkit`, C++ driver, ctest, CI, container image.
