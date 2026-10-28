import numpy as np
import pytest

from engine import capi

pytestmark = pytest.mark.skipif(not capi.available(), reason="libengine non compilata: eseguire make")


def test_version():
    assert capi.version() == "0.0.1"


def test_add_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1023).astype(np.float32)
    b = rng.standard_normal(1023).astype(np.float32)
    np.testing.assert_array_equal(capi.add(a, b), a + b)


def test_missing_library_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ENGINE_LIB", str(tmp_path / "manca.so"))
    monkeypatch.setattr(capi, "_lib", None)
    assert not capi.available()
    with pytest.raises(OSError, match="make"):
        capi.version()
