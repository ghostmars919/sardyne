"""Scrittura di file GGUF v3 (formato descritto in engine/loader/gguf.py).

- I dati dei tensori possono essere prodotti al momento della scrittura (add_lazy_tensor):
  l'header dipende solo da nomi, dimensioni e tipi, quindi il modello convertito non deve
  stare tutto in memoria.
- Il file viene scritto accanto alla destinazione e poi rinominato (scrittura atomica): chi ha
  già mappato il file precedente continua a leggerlo intatto, e un errore a metà non lascia
  un file corrotto.
"""
import math
import os
import struct
from pathlib import Path

import numpy as np

from engine.loader.gguf import DEFAULT_ALIGNMENT, GGML_TYPE_ID, GGUF_MAGIC, VALUE_TYPES, tensor_nbytes

_VT = {name: tid for tid, (name, _) in VALUE_TYPES.items()}
_NP_TO_VT = {np.dtype(np.uint8): "UINT8", np.dtype(np.int8): "INT8", np.dtype(np.uint16): "UINT16",
             np.dtype(np.int16): "INT16", np.dtype(np.uint32): "UINT32", np.dtype(np.int32): "INT32",
             np.dtype(np.float32): "FLOAT32", np.dtype(np.bool_): "BOOL", np.dtype(np.uint64): "UINT64",
             np.dtype(np.int64): "INT64", np.dtype(np.float64): "FLOAT64"}
_INT_RANGE = {"UINT8": (0, 2 ** 8), "INT8": (-2 ** 7, 2 ** 7), "UINT16": (0, 2 ** 16),
              "INT16": (-2 ** 15, 2 ** 15), "UINT32": (0, 2 ** 32), "INT32": (-2 ** 31, 2 ** 31),
              "UINT64": (0, 2 ** 64), "INT64": (-2 ** 63, 2 ** 63)}
MAX_ALIGNMENT = 1 << 20


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bit bf16 (uint16) con arrotondamento al pari più vicino, come fa PyTorch.

    I NaN sono un caso a parte: l'arrotondamento li farebbe traboccare in ±0 o inf, quindi
    diventano NaN silenziosi con lo stesso segno.
    """
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = 0x7FFF + ((bits >> 16) & 1)
    out = ((bits + rounding) >> 16).astype(np.uint16)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    if nan.any():
        out[nan] = ((bits[nan] >> 16) | 0x0040).astype(np.uint16)
    return out


def _check_name(name: str, what: str):
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError(f"{what} non valido: {name!r}")


def _infer_type(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "BOOL"
    if isinstance(value, (int, np.integer)):
        return "UINT32" if 0 <= value < 2 ** 32 else ("UINT64" if value >= 0 else "INT64")
    if isinstance(value, (float, np.floating)):
        return "FLOAT32"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "ARRAY"
    raise TypeError(f"tipo di metadato non supportato: {type(value).__name__}")


def _check_scalar(key: str, vtype: str, value):
    if vtype in _INT_RANGE:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{key}: {vtype} richiede un intero, ricevuto {type(value).__name__}")
        lo, hi = _INT_RANGE[vtype]
        if not lo <= int(value) < hi:
            raise ValueError(f"{key}: {value} fuori dall'intervallo di {vtype}")
    elif vtype in ("FLOAT32", "FLOAT64"):
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
            raise TypeError(f"{key}: {vtype} richiede un numero")
    elif vtype == "BOOL" and not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{key}: BOOL richiede un booleano")
    elif vtype == "STRING" and not isinstance(value, str):
        raise TypeError(f"{key}: STRING richiede una stringa")


class GGUFWriter:
    def __init__(self, path, alignment: int = DEFAULT_ALIGNMENT):
        if not 0 < alignment <= MAX_ALIGNMENT or alignment & (alignment - 1):
            raise ValueError(f"allineamento {alignment} non valido: serve una potenza di 2 <= {MAX_ALIGNMENT}")
        self.path = Path(path)
        self.alignment = alignment
        self._kv = {}  # chiave -> (tipo, valore, tipo elementi)
        self._tensors = {}  # nome -> (ne, tipo ggml, nbytes, funzione che produce i byte)
        if alignment != DEFAULT_ALIGNMENT:
            self.add("general.alignment", alignment, "UINT32")

    def add(self, key: str, value, vtype: str | None = None, elem_type: str | None = None):
        _check_name(key, "chiave")
        if key in self._kv:
            raise ValueError(f"metadato duplicato: {key}")
        vtype = vtype or _infer_type(value)
        if vtype not in _VT:
            raise ValueError(f"tipo {vtype} sconosciuto")
        if vtype == "ARRAY":
            if isinstance(value, np.ndarray):
                if value.ndim != 1:
                    raise ValueError(f"{key}: solo array monodimensionali")
                if elem_type is None:
                    if value.dtype not in _NP_TO_VT:
                        raise TypeError(f"{key}: dtype {value.dtype} non rappresentabile in GGUF")
                    elem_type = _NP_TO_VT[value.dtype]
                value = value.tolist()
            value = list(value)
            if elem_type is None:
                types = {_infer_type(v) for v in value} or {"UINT32"}
                if types <= {"UINT32", "UINT64", "INT64"} and len(types) > 1:
                    types = {"INT64"} if any(v < 0 for v in value) else {"UINT64"}
                if len(types) != 1:
                    raise TypeError(f"{key}: array con elementi di tipi diversi {sorted(types)}")
                elem_type = types.pop()
            if elem_type == "ARRAY":
                raise TypeError(f"{key}: array di array non supportati")
            for v in value:
                _check_scalar(key, elem_type, v)
        else:
            _check_scalar(key, vtype, value)
        self._kv[key] = (vtype, value, elem_type)

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: str):
        """data: array NumPy nel formato di destinazione (BF16 come uint16). Shape NumPy -> ne rovesciato."""
        expected = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}
        if ggml_type not in expected:
            raise ValueError("per i tipi quantizzati usare add_raw_tensor o add_lazy_tensor")
        data = np.ascontiguousarray(data, dtype=expected[ggml_type])
        self.add_raw_tensor(name, tuple(reversed(data.shape)), ggml_type, data.tobytes())

    def add_raw_tensor(self, name: str, ne: tuple, ggml_type: str, raw: bytes):
        if ggml_type not in GGML_TYPE_ID:
            raise ValueError(f"tipo ggml {ggml_type} sconosciuto")
        expected = tensor_nbytes(tuple(ne), GGML_TYPE_ID[ggml_type])
        if len(raw) != expected:
            raise ValueError(f"{name}: {len(raw)} byte, attesi {expected}")
        self.add_lazy_tensor(name, ne, ggml_type, lambda: raw)

    def add_lazy_tensor(self, name: str, ne: tuple, ggml_type: str, produce):
        """Registra un tensore i cui byte vengono prodotti da produce() solo durante write()."""
        _check_name(name, "nome di tensore")
        if name in self._tensors:
            raise ValueError(f"tensore duplicato: {name}")
        if ggml_type not in GGML_TYPE_ID:
            raise ValueError(f"tipo ggml {ggml_type} sconosciuto")
        ne = tuple(int(d) for d in ne)
        if not 1 <= len(ne) <= 4 or min(ne) < 1:
            raise ValueError(f"{name}: dimensioni non valide {ne}")
        tid = GGML_TYPE_ID[ggml_type]
        self._tensors[name] = (ne, tid, tensor_nbytes(ne, tid), produce)

    def _check_size(self, name, n):
        expected = self._tensors[name][2]
        if n != expected:
            raise ValueError(f"{name}: {n} byte, attesi {expected}")

    # ------------------------------------------------------------ serializzazione

    @staticmethod
    def _string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, vtype: str, value, elem_type: str | None) -> bytes:
        if vtype == "STRING":
            return self._string(value)
        if vtype == "ARRAY":
            out = [struct.pack("<IQ", _VT[elem_type], len(value))]
            if elem_type == "STRING":
                out += [self._string(s) for s in value]
            else:
                fmt = VALUE_TYPES[_VT[elem_type]][1]
                out.append(np.asarray(value, dtype=np.dtype(fmt)).tobytes())
            return b"".join(out)
        if vtype in ("FLOAT32", "FLOAT64"):
            value = float(value)
        elif vtype == "BOOL":
            value = bool(value)
        else:
            value = int(value)
        return struct.pack(VALUE_TYPES[_VT[vtype]][1], value)

    def _header(self) -> bytes:
        header = [GGUF_MAGIC, struct.pack("<IQQ", 3, len(self._tensors), len(self._kv))]
        for key, (vtype, value, elem_type) in self._kv.items():
            header += [self._string(key), struct.pack("<I", _VT[vtype]), self._value(vtype, value, elem_type)]
        offset = 0
        for name, (ne, tid, nbytes, _) in self._tensors.items():
            header += [self._string(name), struct.pack("<I", len(ne)), struct.pack(f"<{len(ne)}Q", *ne),
                       struct.pack("<IQ", tid, offset)]
            offset += math.ceil(nbytes / self.alignment) * self.alignment
        return b"".join(header)

    def write(self):
        header = self._header()
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(header)
                f.write(b"\0" * (-len(header) % self.alignment))
                for name, (_, _, nbytes, produce) in self._tensors.items():
                    raw = produce()
                    raw = raw.tobytes() if isinstance(raw, np.ndarray) else raw
                    self._check_size(name, len(raw))
                    f.write(raw)
                    f.write(b"\0" * (-nbytes % self.alignment))
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
