"""Binding ctypes verso libsardyne.

La libreria viene caricata al primo uso: da build/ oppure dal percorso in SARDYNE_LIB
(utile per usare la build con i sanitizer, vedi `make debug`). Così importare questo modulo
non fallisce se la libreria non è ancora compilata.
"""
import ctypes
import os
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LIBNAME = "libsardyne.dylib" if sys.platform == "darwin" else "libsardyne.so"

_f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_u64 = ctypes.c_uint64
_lib = None


class _GGUFTensor(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("n_dims", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("ne", ctypes.c_uint64 * 4), ("offset", ctypes.c_uint64), ("nbytes", ctypes.c_uint64),
                ("data", ctypes.c_void_p)]


class _GGUFKV(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("type", ctypes.c_uint32), ("arr_type", ctypes.c_uint32),
                ("n", ctypes.c_uint64), ("ptr", ctypes.c_void_p), ("str_pos", ctypes.c_void_p)]


_SIGNATURES = {
    "engine_version": (ctypes.c_char_p, []),
    "engine_add_f32": (None, [_f32p, _f32p, _f32p, ctypes.c_size_t]),
    "gguf_open": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]),
    "gguf_close": (None, [ctypes.c_void_p]),
    "gguf_version": (ctypes.c_uint32, [ctypes.c_void_p]),
    "gguf_alignment": (ctypes.c_uint32, [ctypes.c_void_p]),
    "gguf_data_offset": (_u64, [ctypes.c_void_p]),
    "gguf_n_kv": (_u64, [ctypes.c_void_p]),
    "gguf_n_tensors": (_u64, [ctypes.c_void_p]),
    "gguf_get_kv": (ctypes.POINTER(_GGUFKV), [ctypes.c_void_p, _u64]),
    "gguf_find_kv": (ctypes.c_int64, [ctypes.c_void_p, ctypes.c_char_p]),
    "gguf_get_tensor": (ctypes.POINTER(_GGUFTensor), [ctypes.c_void_p, _u64]),
    "gguf_find_tensor": (ctypes.c_int64, [ctypes.c_void_p, ctypes.c_char_p]),
    "gguf_kv_get_u64": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(_u64)]),
    "gguf_kv_get_f64": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]),
    "gguf_kv_get_str": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(_u64)]),
    "gguf_kv_get_arr_str": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int64, _u64, ctypes.POINTER(_u64)]),
    "gguf_type_name": (ctypes.c_char_p, [ctypes.c_uint32]),
}


class EngineModelInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in
                ("n_vocab", "dim", "n_layer", "n_head", "n_kv_head", "head_dim", "ffn_dim", "max_pos")] + \
               [("norm_eps", ctypes.c_float), ("rope_theta", ctypes.c_float)] + \
               [(name, ctypes.c_int32) for name in ("rope_interleaved", "qkv_bias", "tied_embeddings", "qk_norm",
                                                     "n_expert", "n_expert_used", "ffn_exp_dim")]


_vp, _i32, _fp = ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)
_SIGNATURES.update({
    # runtime (csrc/engine.h)
    "engine_model_load": (_vp, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]),
    "engine_model_free": (None, [_vp]),
    "engine_model_get_info": (None, [_vp, ctypes.POINTER(EngineModelInfo)]),
    "engine_ctx_new": (_vp, [_vp, _i32, _i32, _i32, _i32, ctypes.c_char_p, ctypes.c_size_t]),
    "engine_ctx_n_threads": (_i32, [_vp]),
    "engine_ctx_free": (None, [_vp]),
    "engine_eval": (ctypes.c_int, [_vp, ctypes.POINTER(_i32), _i32]),
    "engine_logits": (_fp, [_vp]),
    "engine_n_logits": (_i32, [_vp]),
    "engine_n_past": (_i32, [_vp]),
    "engine_reset": (None, [_vp]),
    "engine_truncate": (ctypes.c_int, [_vp, _i32]),
    "engine_last_error": (ctypes.c_char_p, [_vp]),
    # kernel (csrc/ops.h), esposti per i test di confronto con NumPy
    "op_simd": (ctypes.c_int, []),
    "op_set_simd": (ctypes.c_int, [ctypes.c_int]),
    "op_f16_to_f32_n": (None, [_vp, _vp, ctypes.c_size_t]),
    "op_matmul_rows": (None, [_vp, _vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int]),
    "op_dot_f32": (ctypes.c_float, [_vp, _vp, ctypes.c_int]),
    "op_f32_to_f16": (ctypes.c_uint16, [ctypes.c_float]),
    "op_row_to_f32": (None, [_vp, ctypes.c_int, _vp, ctypes.c_size_t]),
    "op_row_size": (ctypes.c_size_t, [ctypes.c_int, ctypes.c_int]),
    "op_quantize_rows": (None, [ctypes.c_int, _vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
    "op_matmul_q_rows": (None, [_vp, _vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int]),
    "op_vec_dot_q": (ctypes.c_float, [ctypes.c_int, _vp, _vp, ctypes.c_int]),
    "op_axpy": (None, [_vp, ctypes.c_float, _vp, ctypes.c_int]),
    "op_matmul": (None, [_vp, _vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
    "op_rmsnorm": (None, [_vp, _vp, _vp, ctypes.c_int, ctypes.c_float]),
    "op_softmax": (None, [_vp, ctypes.c_int]),
    "op_silu_mul": (None, [_vp, _vp, ctypes.c_int]),
    "op_rope_table": (None, [_vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_float]),
    "op_rope": (None, [_vp, ctypes.c_int, ctypes.c_int, _vp, _vp, ctypes.c_int]),
    "op_attention": (None, [_vp, _vp, _vp, _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, _vp]),
})


def lib_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("SARDYNE_LIB", _ROOT / "build" / _LIBNAME))


def available() -> bool:
    return lib_path().exists()


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        path = lib_path()
        if not path.exists():
            raise OSError(f"{path} non trovata: compilare la libreria con `make`")
        handle = ctypes.CDLL(str(path))
        for name, (res, args) in _SIGNATURES.items():
            fn = getattr(handle, name)
            fn.restype, fn.argtypes = res, args
        _lib = handle
    return _lib


def version() -> str:
    return lib().engine_version().decode()


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    assert a.shape == b.shape
    out = np.empty_like(a)
    lib().engine_add_f32(a, b, out, a.size)
    return out


# ---------------------------------------------------------------- GGUF (csrc/gguf.h)

class CGGUFFile:
    """Il parser C visto da Python: serve a confrontarlo con sardyne/loader/gguf.py.

    Tutti i valori restituiti sono copie (stringhe, interi, byte dei tensori): niente resta legato
    alla mmap del C, quindi nulla diventa invalido dopo close().
    """

    def __init__(self, path):
        self._lib = lib()
        self._h = None
        err = ctypes.create_string_buffer(256)
        self._h = self._lib.gguf_open(str(path).encode(), err, len(err))
        if not self._h:
            raise ValueError(err.value.decode("utf-8", errors="replace"))
        self.version = self._lib.gguf_version(self._h)
        self.alignment = self._lib.gguf_alignment(self._h)
        self.data_offset = self._lib.gguf_data_offset(self._h)

    def close(self):
        if self._h:
            self._lib.gguf_close(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _handle(self):
        if not self._h:
            raise ValueError("file GGUF già chiuso")
        return self._h

    def keys(self) -> list[str]:
        h = self._handle()
        # Chiavi e nomi sono UTF-8 valido: il parser C rifiuta il file altrimenti.
        return [self._lib.gguf_get_kv(h, i).contents.key.decode() for i in range(self._lib.gguf_n_kv(h))]

    def _kv(self, key):
        h = self._handle()
        i = self._lib.gguf_find_kv(h, key.encode())
        if i < 0:
            raise KeyError(key)
        return h, i, self._lib.gguf_get_kv(h, i).contents

    def kv_type(self, key) -> tuple[int, int, int]:
        _, _, kv = self._kv(key)
        return kv.type, kv.arr_type, kv.n

    def get_u64(self, key) -> int:
        h, i, _ = self._kv(key)
        out = _u64()
        if self._lib.gguf_kv_get_u64(h, i, ctypes.byref(out)):
            raise TypeError(key)
        return out.value

    def get_f64(self, key) -> float:
        h, i, _ = self._kv(key)
        out = ctypes.c_double()
        if self._lib.gguf_kv_get_f64(h, i, ctypes.byref(out)):
            raise TypeError(key)
        return out.value

    def get_str(self, key) -> str:
        h, i, _ = self._kv(key)
        n = _u64()
        p = self._lib.gguf_kv_get_str(h, i, ctypes.byref(n))
        if not p:
            raise TypeError(key)
        return ctypes.string_at(p, n.value).decode("utf-8", errors="replace")

    def get_arr_str(self, key, j) -> str:
        h, i, _ = self._kv(key)
        n = _u64()
        p = self._lib.gguf_kv_get_arr_str(h, i, j, ctypes.byref(n))
        if not p:
            raise IndexError(j)
        return ctypes.string_at(p, n.value).decode("utf-8", errors="replace")

    def tensors(self) -> list[dict]:
        h = self._handle()
        out = []
        for i in range(self._lib.gguf_n_tensors(h)):
            t = self._lib.gguf_get_tensor(h, i).contents
            out.append({"name": t.name.decode(), "ne": tuple(t.ne[:t.n_dims]),
                        "type": self._lib.gguf_type_name(t.type).decode(), "offset": t.offset,
                        "nbytes": t.nbytes})
        return out

    def tensor_bytes(self, name) -> np.ndarray:
        """Copia dei byte del tensore, letti dal puntatore `data` del C (dentro la mmap)."""
        h = self._handle()
        i = self._lib.gguf_find_tensor(h, name.encode())
        if i < 0:
            raise KeyError(name)
        t = self._lib.gguf_get_tensor(h, i).contents
        return np.frombuffer(ctypes.string_at(t.data, t.nbytes), dtype=np.uint8)
