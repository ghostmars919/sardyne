"""Binding ctypes verso libengine."""
import ctypes
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIBNAME = "libengine.dylib" if sys.platform == "darwin" else "libengine.so"

lib = ctypes.CDLL(str(_ROOT / "build" / _LIBNAME))

_f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

lib.engine_version.restype = ctypes.c_char_p
lib.engine_version.argtypes = []
lib.engine_add_f32.restype = None
lib.engine_add_f32.argtypes = [_f32p, _f32p, _f32p, ctypes.c_size_t]


def version() -> str:
    return lib.engine_version().decode()


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    assert a.shape == b.shape
    out = np.empty_like(a)
    lib.engine_add_f32(a, b, out, a.size)
    return out
