# ompnn installed into a toolchain image of the study (fnn-bench/containers), one virtual environment per
# compiler under /opt/venvs/<name>. BUILDS = "|"-separated "<name> <compiler> <target> <offload-arch|->" entries.
#   default   BASE=ghcr.io/napanto/fnn-cuda:latest   gcc-14 and clang-18 with nvptx offload (sm_61, sm_80), clang-22 host
#   rocm      BASE=ghcr.io/napanto/fnn-rocm:latest   BUILDS="amdclang amdclang++ amd gfx1100|gcc14amd g++-14 amd gfx1100"
#             BLAS_ROOT=/opt/fnn-rocm/openblas-openmp (the base ships those two venvs with the build dependencies)
ARG BASE=ghcr.io/napanto/fnn-cuda:latest
FROM ${BASE}
ARG IMAGE_NAME=ompnn:latest
ARG IMAGE_BUILT=unknown
ARG BUILDS="gcc14 g++-14 nvidia sm_61;sm_80|clang22 clang++-22 cpu -|clang18 clang++-18 nvidia sm_61;sm_80"
ARG BLAS_ROOT=/opt/openblas-openmp
ENV CMAKE_BUILD_PARALLEL_LEVEL=8 OMPNN_BLAS_ROOT=${BLAS_ROOT}
COPY . /opt/src/ompnn
WORKDIR /opt/src/ompnn
RUN mkdir -p /opt/venvs && old_ifs=$IFS && IFS='|' && set -f && for cfg in $BUILDS; do IFS=$old_ifs; set +f; set -- $cfg; \
        if [ -x /opt/fnn-rocm/venv-$1/bin/python ]; then ln -sfn /opt/fnn-rocm/venv-$1 /opt/venvs/$1; else python -m venv --system-site-packages /opt/venvs/$1; fi && \
        CXX=$2 OMPNN_TARGET=$3 OMPNN_OFFLOAD_ARCH=$([ "$4" = "-" ] && echo "" || echo "$4") \
          /opt/venvs/$1/bin/pip install --no-cache-dir -v . && \
        /opt/venvs/$1/bin/pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" && \
        if [ "$3" = amd ] && [ "$2" = g++-14 ] && [ ! -e /dev/kfd ]; then echo "$1: import check skipped (gcc's GCN plugin needs the GPU at load time)"; \
        else /opt/venvs/$1/bin/python -c "import ompnn; print(ompnn.build_info())"; fi || exit 1; \
        IFS='|'; set -f; done
ENV FNN_IMAGE=${IMAGE_NAME} FNN_IMAGE_BUILT=${IMAGE_BUILT}
LABEL org.opencontainers.image.source=https://github.com/napanto/ompnn fnn.image="${IMAGE_NAME}" fnn.image.built="${IMAGE_BUILT}"
CMD ["/bin/bash"]
