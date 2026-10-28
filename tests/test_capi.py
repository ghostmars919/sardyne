import numpy as np

from engine import capi


def test_version():
    assert capi.version() == "0.0.1"


def test_add_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1023).astype(np.float32)
    b = rng.standard_normal(1023).astype(np.float32)
    np.testing.assert_array_equal(capi.add(a, b), a + b)
