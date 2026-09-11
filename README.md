# ompnn

OpenMP feed-forward neural network library: the 1:1 port of
[syclnn](https://github.com/napanto/syclnn) (and sibling of
[cudann](https://github.com/napanto/cudann)) to pragma-based OpenMP, with the
same public API and numerics (validated by the shared
[fnn-testkit](https://github.com/napanto/fnn-bench) parity suite). One code base,
two execution paths selected at run time:

* **host** (`device="cpu"`): `#pragma omp parallel for simd` kernels + CBLAS
  (OpenBLAS or Intel oneMKL, chosen at build time with `OMPNN_BLAS`);
* **target** (`device="gpu"`): `#pragma omp target teams distribute parallel for
  simd` kernels on `omp_target_alloc` memory (`is_device_ptr`), transfers with
  `omp_target_memcpy`, GEMM through cuBLAS / hipBLAS on the same device pointers
  (interop), so the BLAS library is the one syclnn and cudann use on that GPU;
* `blas="omp"`: a hand-written OpenMP GEMM/GEMV instead of any library (the
  pure pragma measurement), on both paths.

Every region is synchronous: portable OpenMP has no event graph, so the phases
run back to back and a host timer around each one is the profiler
(`profile=True`). This is the point of the comparison with the event-driven
SYCL and CUDA implementations.

## Quick start

```python
import numpy as np, ompnn

layers = [ompnn.LayerDescription(784), ompnn.LayerDescription(512, ompnn.ActivationType.ReLU),
          ompnn.LayerDescription(10, ompnn.ActivationType.Sigmoid)]
net = ompnn.Network(layers, 0.1, dtype="float", device="gpu", profile=True, seed=1,
                    momentum=ompnn.MomentumConfig_float(ompnn.Classical, 0.9))
losses = net.train(X.ravel(), Y.ravel(), n_samples=len(X), batch_size=256, max_epochs=5)
print(net.device_name, net.profile)
print(ompnn.devices())      # [host, offload devices...]
print(ompnn.build_info())   # compiler, target, offload arch, CBLAS, OpenMP version
```

The API is that of syclnn 0.2 (`Network_double` / `Network_float`, the `*_double`
/ `*_float` configuration classes, `Options`, `devices()`, `build_info()`).
`Options.device`: `cpu`/`host`, `gpu`, `index:N`, `omp:N`, or a name substring.
`Options.blas`: `auto` (CBLAS on the host, vendor BLAS on the GPU), `omp`,
`openblas`/`mkl` (must match the linked one), `cublas`/`rocblas`. Switches
without an OpenMP counterpart (`queue`, `streams`, `fine_deps`, `join_kernels`,
`pinned_host`, `memory` other than `device`) are ignored or rejected.
`workgroup_size` sets `omp_set_teams_thread_limit`.

## Run the published image

`ghcr.io/napanto/ompnn` is the library installed in the `fnn-cuda` toolchain image, built by CI from
the `Containerfile` on every push, in three virtual environments, one per compiler:
`/opt/venvs/gcc14` (host + nvptx offload), `/opt/venvs/clang18` (host + nvptx offload) and
`/opt/venvs/clang22` (host). One command:

```sh
# CPU only (any x86-64 host with a container runtime):
podman run --rm -it ghcr.io/napanto/ompnn /opt/venvs/clang22/bin/python -c "import ompnn; print(ompnn.devices())"
# with an NVIDIA GPU (driver >= 525 and the NVIDIA container toolkit's CDI spec on the host):
podman run --rm -it --device nvidia.com/gpu=all ghcr.io/napanto/ompnn /opt/venvs/clang18/bin/python -c "import ompnn; print(ompnn.devices())"
```

The AMD builds (amdclang++, gcc-14 amdgcn) are made from the `fnn-rocm` toolchain image
(`fnn-bench/containers`) and are not published as a library image.

## Building

```sh
# host only, gcc-14 + OpenBLAS
CXX=g++-14 OMPNN_BLAS=openblas OMPNN_BLAS_ROOT=/opt/openblas-openmp pip install -v .
# host only, clang-22 + oneMKL
CXX=clang++-22 OMPNN_BLAS=mkl OMPNN_BLAS_ROOT=/opt/venv pip install -v .
# NVIDIA offload: clang-18 (Ubuntu, ships libomptarget nvptx), gcc-14-offload-nvptx, nvc++
CXX=clang++-18 OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH="sm_61;sm_80" pip install -v .
CXX=g++-14     OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_80 pip install -v .
CXX=nvc++      OMPNN_TARGET=nvidia OMPNN_OFFLOAD_ARCH=sm_80 pip install -v .
# AMD offload: amdclang++ (ROCm), clang-18, gcc-14-offload-amdgcn
CXX=amdclang++ OMPNN_TARGET=amd OMPNN_OFFLOAD_ARCH=gfx1100 OMPNN_BLAS_ROOT=$HOME/.local/opt/fnn-rocm/openblas-openmp pip install -v .

# C++ driver + ctest smoke test
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DOMPNN_BLAS_ROOT=/opt/openblas-openmp
cmake --build build -j8 && ctest --test-dir build
./build/train_bench --layers 784,1024,10 --samples 8192 --batch 256 --epochs 5 --dtype float --device cpu --profile
```

`build_info()` records the compiler, flags, offload architecture, CBLAS and
OpenMP version of each build; the benchmark harness stores it with every result,
so one virtual environment per compiler is enough for the compiler matrix.

## Tests

```sh
pip install "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit"
pytest --device cpu                     # both dtypes on the host
pytest --device gpu --option blas=omp   # target path with the hand-written GEMM
```

Status: see `fnn-bench/docs/toolchains.md` for the compiler × device matrix.

## License

LGPL-3.0-only. Copyright (C) 2026 Antonio Napolitano.
