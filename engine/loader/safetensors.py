"""Lettore minimale del formato safetensors (Hugging Face), senza dipendenze oltre NumPy.

Layout del file:
    [8 byte]  N = lunghezza dell'header, intero little-endian senza segno
    [N byte]  header JSON: {nome: {"dtype": "BF16", "shape": [...], "data_offsets": [inizio, fine]}, ...}
              più una voce opzionale "__metadata__"
    [resto]   i dati grezzi dei tensori, row-major; gli offset sono relativi all'inizio di questa zona

Il file viene mappato in memoria (mmap): leggere un tensore tocca solo le sue pagine.
L'header viene validato all'apertura (offset, dtype, shape coerenti con la dimensione del file),
perché i file arrivano da Internet.
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # NumPy non ha bfloat16: si leggono i bit e si convertono a mano
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
MAX_HEADER = 100 << 20  # come la libreria safetensors


class SafeTensorsError(ValueError):
    pass


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    # bf16 è la metà alta di un float32 (stesso esponente a 8 bit, mantissa troncata a 7 bit):
    # basta spostare i 16 bit nella parte alta di un intero a 32 bit e reinterpretarlo.
    return (bits.astype(np.uint32) << 16).view(np.float32)


class SafeTensorsFile:
    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size < 8:
            raise SafeTensorsError(f"{self.path.name}: file troppo corto")
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n > min(size - 8, MAX_HEADER):
                raise SafeTensorsError(f"{self.path.name}: lunghezza dell'header {n} non valida")
            try:
                header = json.loads(f.read(n))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise SafeTensorsError(f"{self.path.name}: header JSON non valido ({e})") from None
        if not isinstance(header, dict):
            raise SafeTensorsError(f"{self.path.name}: l'header non è un oggetto JSON")
        self.metadata = header.pop("__metadata__", {})
        self._data_start = 8 + n
        data_size = size - self._data_start
        for name, t in header.items():
            self._validate(name, t, data_size)
        self.tensors = header
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r") if size > 0 else None

    def _validate(self, name, t, data_size):
        where = f"{self.path.name}: {name}"
        if not isinstance(t, dict) or t.get("dtype") not in _DTYPES:
            raise SafeTensorsError(f"{where}: dtype {t.get('dtype') if isinstance(t, dict) else t!r} non supportato")
        shape, offsets = t.get("shape"), t.get("data_offsets")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise SafeTensorsError(f"{where}: shape non valida {shape!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(o, int) for o in offsets)):
            raise SafeTensorsError(f"{where}: data_offsets non validi {offsets!r}")
        begin, end = offsets
        if not 0 <= begin <= end <= data_size:
            raise SafeTensorsError(f"{where}: data_offsets {offsets} fuori dal file")
        expected = math.prod(shape) * np.dtype(_DTYPES[t["dtype"]]).itemsize
        if end - begin != expected:
            raise SafeTensorsError(f"{where}: {end - begin} byte per shape {shape}, attesi {expected}")

    def keys(self):
        return self.tensors.keys()

    def info(self, name):
        return self.tensors[name]

    def shape(self, name) -> tuple:
        return tuple(self.tensors[name]["shape"])

    def raw(self, name) -> np.ndarray:
        """Il tensore nel suo dtype originale (BF16 come uint16), come vista in sola lettura sulla mmap."""
        t = self.tensors[name]
        begin, end = t["data_offsets"]
        buf = self._mm[self._data_start + begin:self._data_start + end]
        return buf.view(_DTYPES[t["dtype"]]).reshape(t["shape"])

    def get_f32(self, name) -> np.ndarray:
        """Il tensore convertito in float32: sempre una copia in RAM, scrivibile, anche se era già F32."""
        a = self.raw(name)
        if self.tensors[name]["dtype"] == "BF16":
            return bf16_to_f32(a)
        return np.array(a, dtype=np.float32, copy=True)


class SafeTensorsDir:
    """Un modello HF in una cartella: un solo model.safetensors oppure più shard con un indice."""

    def __init__(self, path):
        path = Path(path)
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = {}
            for f in sorted(set(weight_map.values())):
                # L'indice arriva da Internet: i file devono stare nella cartella del modello.
                if Path(f).name != f or not f.endswith(".safetensors"):
                    raise SafeTensorsError(f"nome di shard non valido nell'indice: {f!r}")
                files[f] = SafeTensorsFile(path / f)
            self._where = {}
            for name, f in weight_map.items():
                if name not in files[f].tensors:
                    raise SafeTensorsError(f"{name} indicato in {f} ma assente nel file")
                self._where[name] = files[f]
        else:
            single = SafeTensorsFile(path / "model.safetensors")
            self._where = {name: single for name in single.keys()}

    def __contains__(self, name):
        return name in self._where

    def keys(self):
        return self._where.keys()

    def shape(self, name) -> tuple:
        return self._where[name].shape(name)

    def get_f32(self, name) -> np.ndarray:
        return self._where[name].get_f32(name)
