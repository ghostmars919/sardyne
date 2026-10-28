"""Parser del formato GGUF (llama.cpp / ggml), versioni 2 e 3, senza dipendenze oltre NumPy.

Layout del file (tutti gli interi little-endian):

    magic     4 byte  "GGUF"
    version   u32     (3 è la versione corrente)
    n_tensors u64
    n_kv      u64
    n_kv volte:       chiave (stringa), tipo del valore (u32), valore
    n_tensors volte:  nome (stringa), n_dims (u32), ne[n_dims] (u64), tipo ggml (u32), offset (u64)
    padding fino a un multiplo di `general.alignment` (default 32)
    dati dei tensori: ogni offset è relativo all'inizio di questa zona ed è allineato

    stringa = lunghezza u64 + byte UTF-8 (senza terminatore)
    array   = tipo degli elementi (u32) + numero di elementi (u64) + elementi

Dimensioni: ne[0] è la dimensione più interna (contigua). Una matrice HF [d_out, d_in] ha
ne = [d_in, d_out]: la shape NumPy è ne rovesciato.

Uso da riga di comando: python -m engine.loader.gguf file.gguf [--all]
"""
import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.loader.safetensors import bf16_to_f32

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# Tipi dei valori nei metadati: id -> (nome, formato struct o None)
VALUE_TYPES = {
    0: ("UINT8", "<B"), 1: ("INT8", "<b"), 2: ("UINT16", "<H"), 3: ("INT16", "<h"),
    4: ("UINT32", "<I"), 5: ("INT32", "<i"), 6: ("FLOAT32", "<f"), 7: ("BOOL", "<?"),
    8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", "<Q"), 11: ("INT64", "<q"),
    12: ("FLOAT64", "<d"),
}
_INT_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}
_NP_SCALAR = {0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "?",
              10: "<u8", 11: "<i8", 12: "<f8"}

# Tipi dei tensori ggml: id -> (nome, elementi per blocco, byte per blocco).
# I tipi quantizzati salvano blocchi di valori con le loro scale (Fase 5).
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}
GGML_TYPE_ID = {name: tid for tid, (name, _, _) in GGML_TYPES.items()}

MAX_DIMS = 4


class GGUFError(ValueError):
    pass


@dataclass
class TensorInfo:
    name: str
    ne: tuple  # dimensioni in ordine ggml (ne[0] = più interna)
    ggml_type: int
    offset: int  # relativo all'inizio della zona dati
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type][0]

    @property
    def shape(self) -> tuple:
        # Ordine NumPy (row-major): la dimensione più interna va per ultima.
        return tuple(reversed(self.ne))

    @property
    def n_elements(self) -> int:
        return math.prod(self.ne)


def tensor_nbytes(ne, ggml_type: int) -> int:
    name, block, size = GGML_TYPES[ggml_type]
    if ne[0] % block:
        raise GGUFError(f"ne[0]={ne[0]} non multiplo del blocco {block} di {name}")
    n = 1
    for d in ne:
        n *= d
    return n // block * size


class _Reader:
    """Cursore sul buffer con controllo dei limiti a ogni lettura."""

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def need(self, n: int):
        if n < 0 or self.pos + n > len(self.buf):
            raise GGUFError(f"file troncato: servono {n} byte all'offset {self.pos}")

    def scalar(self, fmt: str):
        n = struct.calcsize(fmt)
        self.need(n)
        (v,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += n
        return v

    def string(self) -> str:
        n = self.scalar("<Q")
        self.need(n)
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def name(self, what: str) -> str:
        """Chiave o nome di tensore: UTF-8 valido e senza NUL, come richiede anche il parser C."""
        n = self.scalar("<Q")
        self.need(n)
        raw = bytes(self.buf[self.pos:self.pos + n])
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise GGUFError(f"{what} all'offset {self.pos}: UTF-8 non valido") from None
        if "\0" in s:
            raise GGUFError(f"{what} all'offset {self.pos}: contiene un byte NUL")
        self.pos += n
        return s

    def value(self, vtype: int):
        if vtype not in VALUE_TYPES:
            raise GGUFError(f"tipo di valore sconosciuto {vtype}")
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.scalar("<I")
            count = self.scalar("<Q")
            if etype in _NP_SCALAR:  # array numerico: lettura in blocco
                dt = np.dtype(_NP_SCALAR[etype])
                self.need(count * dt.itemsize)
                arr = np.frombuffer(self.buf, dtype=dt, count=count, offset=self.pos).copy()
                self.pos += count * dt.itemsize
                return arr
            if etype == 9:
                raise GGUFError("array di array non supportati")
            self.need(count * 8)  # ogni stringa occupa almeno gli 8 byte della lunghezza
            return [self.value(etype) for _ in range(count)]
        return self.scalar(VALUE_TYPES[vtype][1])


class GGUFFile:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.stat().st_size < 4:  # np.memmap rifiuta i file vuoti
            raise GGUFError("file troppo corto")
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        r = _Reader(self._mm)

        if bytes(self._mm[:4]) != GGUF_MAGIC:
            raise GGUFError("non è un file GGUF (magic errato)")
        r.pos = 4
        self.version = r.scalar("<I")
        if self.version & 0xFFFF == 0 and 1 <= self.version >> 24 <= 3:
            raise GGUFError("file GGUF big-endian non supportato")
        if self.version not in (2, 3):
            raise GGUFError(f"versione GGUF {self.version} non supportata")
        n_tensors = r.scalar("<Q")
        n_kv = r.scalar("<Q")

        self.metadata = {}
        self.metadata_types = {}  # chiave -> (tipo, tipo elementi se array)
        for _ in range(n_kv):
            key = r.name("chiave")
            if key in self.metadata:
                raise GGUFError(f"metadato duplicato: {key}")
            vtype = r.scalar("<I")
            etype_pos = r.pos
            self.metadata[key] = r.value(vtype)
            etype = struct.unpack_from("<I", self._mm, etype_pos)[0] if vtype == 9 else None
            self.metadata_types[key] = (vtype, etype)

        self.alignment = DEFAULT_ALIGNMENT
        if "general.alignment" in self.metadata:
            if self.metadata_types["general.alignment"][0] not in _INT_TYPES:
                raise GGUFError("general.alignment non è un intero")
            self.alignment = int(self.metadata["general.alignment"])
        if not 0 < self.alignment <= 1 << 20 or self.alignment & (self.alignment - 1):
            raise GGUFError(f"allineamento {self.alignment} non valido")

        infos = []
        for _ in range(n_tensors):
            name = r.name("nome di tensore")
            n_dims = r.scalar("<I")
            if not 1 <= n_dims <= MAX_DIMS:
                raise GGUFError(f"{name}: n_dims={n_dims}")
            ne = tuple(r.scalar("<Q") for _ in range(n_dims))
            if 0 in ne:
                raise GGUFError(f"{name}: dimensione nulla")
            ttype = r.scalar("<I")
            if ttype not in GGML_TYPES:
                raise GGUFError(f"{name}: tipo ggml sconosciuto {ttype}")
            offset = r.scalar("<Q")
            infos.append(TensorInfo(name, ne, ttype, offset, tensor_nbytes(ne, ttype)))
        names = [t.name for t in infos]
        if len(set(names)) != len(names):
            raise GGUFError("tensore duplicato")

        # La zona dati inizia al primo multiplo dell'allineamento dopo l'header.
        self.data_offset = -(-r.pos // self.alignment) * self.alignment
        data_size = len(self._mm) - self.data_offset
        if data_size < 0:
            raise GGUFError("zona dati oltre la fine del file")
        self.tensors = {}
        for t in infos:
            if t.offset % self.alignment:
                raise GGUFError(f"{t.name}: offset {t.offset} non allineato")
            if t.offset + t.nbytes > data_size:
                raise GGUFError(f"{t.name}: dati oltre la fine del file")
            self.tensors[t.name] = t

    def close(self):
        """Rilascia la mmap. Le viste restituite da raw() la tengono comunque in vita."""
        self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def arch(self) -> str:
        return self.metadata["general.architecture"]

    def raw(self, name) -> np.ndarray:
        """Vista senza copie sui dati del tensore.

        F32/F16/BF16 (BF16 come uint16): shape NumPy del tensore.
        Tipi quantizzati: uint8 [righe, byte per riga] (la decodifica arriva in Fase 5).
        """
        if self._mm is None:
            raise ValueError("file GGUF già chiuso")
        t = self.tensors[name]
        start = self.data_offset + t.offset
        buf = self._mm[start:start + t.nbytes]
        if t.type_name in ("F32", "F16", "BF16"):
            dt = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}[t.type_name]
            return buf.view(dt).reshape(t.shape)
        rows = t.n_elements // t.ne[0]
        return buf.reshape(rows, t.nbytes // rows)

    def get_f32(self, name) -> np.ndarray:
        t = self.tensors[name]
        a = self.raw(name)
        if t.type_name == "F32":
            return np.array(a)
        if t.type_name == "F16":
            return a.astype(np.float32)
        if t.type_name == "BF16":
            return bf16_to_f32(a)
        raise NotImplementedError(f"{name}: dequantizzazione {t.type_name} non ancora implementata (Fase 5)")


def _fmt_value(v, limit=6) -> str:
    if isinstance(v, (list, np.ndarray)):
        items = list(v[:limit])
        more = f", ... (+{len(v) - limit})" if len(v) > limit else ""
        return f"[{len(v)}] " + "[" + ", ".join(repr(x) if isinstance(x, str) else str(x) for x in items) + more + "]"
    if isinstance(v, str) and len(v) > 80:
        return repr(v[:80]) + f"... ({len(v)} caratteri)"
    return repr(v) if isinstance(v, str) else str(v)


def main():
    ap = argparse.ArgumentParser(description="Mostra metadati e tensori di un file GGUF")
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true", help="elenca tutti i tensori (default: primi 2 strati)")
    args = ap.parse_args()

    f = GGUFFile(args.path)
    size = f.path.stat().st_size
    print(f"{f.path.name}: GGUF v{f.version}, {len(f.metadata)} metadati, {len(f.tensors)} tensori, "
          f"{size / 1e6:.1f} MB, dati da offset {f.data_offset} (allineamento {f.alignment})\n")
    for k, v in f.metadata.items():
        vt, et = f.metadata_types[k]
        tname = VALUE_TYPES[vt][0] + (f"<{VALUE_TYPES[et][0]}>" if et is not None else "")
        print(f"  {k:45s} {tname:16s} {_fmt_value(v)}")

    print(f"\n  {'tensore':32s} {'tipo':6s} {'shape (NumPy)':>18s} {'MB':>8s}")
    by_type = {}
    for t in f.tensors.values():
        by_type.setdefault(t.type_name, [0, 0])
        by_type[t.type_name][0] += 1
        by_type[t.type_name][1] += t.nbytes
        is_block = t.name.startswith("blk.")
        if args.all or not is_block or t.name.split(".")[1] in ("0", "1"):
            print(f"  {t.name:32s} {t.type_name:6s} {str(t.shape):>18s} {t.nbytes / 1e6:8.2f}")
    n_params = sum(t.n_elements for t in f.tensors.values())
    print(f"\n  parametri: {n_params / 1e6:.1f}M  |  " +
          ", ".join(f"{k}: {n} tensori {b / 1e6:.0f} MB" for k, (n, b) in by_type.items()))


if __name__ == "__main__":
    main()
