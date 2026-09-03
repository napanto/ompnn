"""Backend-specific smoke tests; the full parity suite is collected from fnn_testkit."""

import numpy as np
import pytest

ompnn = pytest.importorskip("ompnn")


def test_import_and_devices():
    assert ompnn.__version__.startswith("0.1")
    devs = ompnn.devices()
    assert devs[0]["type"] == "cpu" and devs[0]["omp_device"] == -1
    info = ompnn.build_info()
    for k in ("target", "cblas", "openmp", "omp_num_devices", "omp_max_threads"):
        assert k in info, k


def test_cpu_and_omp_blas_agree():
    layers = [ompnn.LayerDescription(6), ompnn.LayerDescription(5, ompnn.ActivationType.Tanh),
              ompnn.LayerDescription(2, ompnn.ActivationType.Sigmoid)]
    X = np.random.default_rng(0).uniform(-1, 1, (16, 6))
    Y = np.zeros((16, 2))
    a = ompnn.Network(layers, 0.1, dtype="double", seed=1, device="cpu")
    b = ompnn.Network(layers, 0.1, dtype="double", seed=1, device="cpu", blas="omp")
    la = a.train(X.ravel(), Y.ravel(), 16, 5, 3)
    lb = b.train(X.ravel(), Y.ravel(), 16, 5, 3)
    np.testing.assert_allclose(la, lb, rtol=1e-12)


def test_wrong_blas_rejected():
    layers = [ompnn.LayerDescription(2), ompnn.LayerDescription(1, ompnn.ActivationType.Tanh)]
    linked = ompnn.build_info()["cblas"]
    other = "mkl" if linked == "openblas" else "openblas"
    with pytest.raises(ValueError):
        ompnn.Network(layers, 0.1, blas=other, seed=1, device="cpu")
