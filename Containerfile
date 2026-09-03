# ompnn container: one wheel per compiler in its own venv, all in the fnn-cuda image.
#
#   podman build --memory=20g -t ompnn .
#   podman run --rm -it ompnn /opt/venvs/gcc14/bin/pytest --device cpu
#   podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable ompnn /opt/venvs/clang18/bin/pytest --device gpu
ARG BASE=ghcr.io/napanto/fnn-cuda:latest
FROM ${BASE}

ENV CMAKE_BUILD_PARALLEL_LEVEL=8 OMPNN_BLAS_ROOT=/opt/openblas-openmp
COPY . /opt/src/ompnn
WORKDIR /opt/src/ompnn

# gcc-14: host + nvptx offload; clang-22: host; clang-18: host + nvptx offload
RUN for cfg in "gcc14 g++-14 nvidia sm_61;sm_80" "clang22 clang++-22 cpu -" "clang18 clang++-18 nvidia sm_61;sm_80"; do \
        set -- $cfg; \
        python -m venv --system-site-packages /opt/venvs/$1 && \
        CXX=$2 OMPNN_TARGET=$3 OMPNN_OFFLOAD_ARCH=$([ "$4" = "-" ] && echo "" || echo "$4") \
          /opt/venvs/$1/bin/pip install --no-cache-dir -v . && \
        /opt/venvs/$1/bin/pip install --no-cache-dir "fnn-testkit @ git+https://github.com/napanto/fnn-bench#subdirectory=testkit" && \
        /opt/venvs/$1/bin/python -c "import ompnn; print(ompnn.build_info())"; \
    done

CMD ["/bin/bash"]
