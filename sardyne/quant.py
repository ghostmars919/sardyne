"""Formati quantizzati di ggml in NumPy: la reference dei kernel C (csrc/quant.c).

Ogni formato divide una riga di pesi in blocchi; ogni blocco ha una o più scale e dei valori
interi piccoli. Le formule seguono ggml-quants.c di llama.cpp; i calcoli in float32 sono fatti
nello stesso ordine, così i risultati coincidono bit per bit con il C.

| tipo | blocco | byte | valore |
|---|---|---|---|
| Q8_0 | 32  | 34  | d·q, q int8 |
| Q4_0 | 32  | 18  | d·(q − 8), q a 4 bit |
| Q5_0 | 32  | 22  | d·(q − 16), q a 5 bit (4 in qs, il quinto in qh) |
| Q4_K | 256 | 144 | 8 sotto-blocchi da 32: d·sc·q − dmin·m, sc e m a 6 bit |
| Q6_K | 256 | 210 | 16 sotto-blocchi da 16: d·sc·(q − 32), q a 6 bit, sc int8 |
| Q8_K | 256 | 292 | solo per le attivazioni: d (float32)·q, più le somme di q per gruppi di 16 |
"""
import numpy as np

QK = {"Q8_0": 32, "Q4_0": 32, "Q5_0": 32, "Q4_K": 256, "Q6_K": 256, "Q8_K": 256}
BLOCK_BYTES = {"Q8_0": 34, "Q4_0": 18, "Q5_0": 22, "Q4_K": 144, "Q6_K": 210, "Q8_K": 292}
# Tipo delle attivazioni con cui ogni tipo di pesi fa il prodotto scalare (come in llama.cpp).
VEC_DOT_TYPE = {"Q8_0": "Q8_0", "Q4_0": "Q8_0", "Q5_0": "Q8_0", "Q4_K": "Q8_K", "Q6_K": "Q8_K"}


def _f16(b: np.ndarray) -> np.ndarray:
    """Colonne di 2 byte (uint8) -> float32 da f16."""
    return np.ascontiguousarray(b).view(np.float16).astype(np.float32)[..., 0]


def _blocks(raw: np.ndarray, kind: str) -> np.ndarray:
    raw = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
    if raw.size % BLOCK_BYTES[kind]:
        raise ValueError(f"{raw.size} byte non sono un numero intero di blocchi {kind}")
    return raw.reshape(-1, BLOCK_BYTES[kind])


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """[nb, 16] byte -> [nb, 32] valori: i 4 bit bassi sono gli elementi 0..15, gli alti 16..31."""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


# ---------------------------------------------------------------- dequantizzazione

def dequant_q8_0(raw) -> np.ndarray:
    b = _blocks(raw, "Q8_0")
    return (b[:, 2:].view(np.int8).astype(np.float32) * _f16(b[:, :2])[:, None]).reshape(-1)


def dequant_q4_0(raw) -> np.ndarray:
    b = _blocks(raw, "Q4_0")
    q = _nibbles(b[:, 2:]).astype(np.int32) - 8
    return (q.astype(np.float32) * _f16(b[:, :2])[:, None]).reshape(-1)


def _q5_0_values(b: np.ndarray) -> np.ndarray:
    qh = b[:, 2:6].copy().view("<u4")[:, 0]
    high = (qh[:, None] >> np.arange(32, dtype=np.uint32)) & 1  # bit e = quinto bit dell'elemento e
    return (_nibbles(b[:, 6:]).astype(np.int32) | (high.astype(np.int32) << 4)) - 16


def dequant_q5_0(raw) -> np.ndarray:
    b = _blocks(raw, "Q5_0")
    return (_q5_0_values(b).astype(np.float32) * _f16(b[:, :2])[:, None]).reshape(-1)


def q4_k_scales(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """12 byte -> 8 scale e 8 minimi a 6 bit (get_scale_min_k4 di ggml).

    I primi 4 valori stanno nei 6 bit bassi dei byte 0..3 (scale) e 4..7 (minimi); gli ultimi 4
    usano i 4 bit dei byte 8..11 più i 2 bit alti dei byte 0..7.
    """
    s = s.astype(np.int32)
    sc = np.empty((s.shape[0], 8), np.int32)
    m = np.empty_like(sc)
    sc[:, :4] = s[:, 0:4] & 63
    m[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, m


def _q4_k_values(b: np.ndarray) -> np.ndarray:
    """[nb, 256] valori 0..15: per ogni gruppo di 32 byte, prima i 4 bit bassi poi gli alti."""
    qs = b[:, 16:].reshape(-1, 4, 32)
    return np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(-1, 256).astype(np.int32)


def dequant_q4_k(raw) -> np.ndarray:
    b = _blocks(raw, "Q4_K")
    d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    sc, m = q4_k_scales(b[:, 4:16])
    d1 = (d[:, None] * sc.astype(np.float32))  # come ggml: d1 = d*sc, m1 = min*m, y = d1*q - m1
    m1 = (dmin[:, None] * m.astype(np.float32))
    q = _q4_k_values(b).reshape(-1, 8, 32).astype(np.float32)
    return (d1[:, :, None] * q - m1[:, :, None]).reshape(-1)


def _q6_k_values(b: np.ndarray) -> np.ndarray:
    """[nb, 256] valori −32..31. Ogni metà da 128: 64 byte di ql (4 bit) e 32 di qh (2 bit × 4)."""
    out = np.empty((b.shape[0], 256), np.int32)
    for half in range(2):
        ql = b[:, half * 64:half * 64 + 64].astype(np.int32)
        qh = b[:, 128 + half * 32:128 + half * 32 + 32].astype(np.int32)
        base = half * 128
        out[:, base + 0:base + 32] = (ql[:, :32] & 0xF) | (((qh >> 0) & 3) << 4)
        out[:, base + 32:base + 64] = (ql[:, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
        out[:, base + 64:base + 96] = (ql[:, :32] >> 4) | (((qh >> 4) & 3) << 4)
        out[:, base + 96:base + 128] = (ql[:, 32:] >> 4) | (((qh >> 6) & 3) << 4)
    return out - 32


def dequant_q6_k(raw) -> np.ndarray:
    b = _blocks(raw, "Q6_K")
    d = _f16(b[:, 208:210])
    sc = b[:, 192:208].view(np.int8).astype(np.float32)  # 16 scale, una ogni 16 valori
    q = _q6_k_values(b).reshape(-1, 16, 16).astype(np.float32)
    return ((d[:, None] * sc)[:, :, None] * q).reshape(-1)  # come ggml: (d*sc)*q


def dequant_q8_k(raw) -> np.ndarray:
    """Attivazioni Q8_K -> float (serve solo alla reference del modello con attivazioni quantizzate)."""
    b = _blocks(raw, "Q8_K")
    d = b[:, 0:4].copy().view("<f4")[:, 0]
    return (b[:, 4:260].view(np.int8).astype(np.float32) * d[:, None]).reshape(-1)


def fake_quant(x: np.ndarray, kind: str) -> np.ndarray:
    """Quantizza e dequantizza le righe di x: l'errore che il runtime C introduce nelle attivazioni."""
    deq = dequant_q8_0 if kind == "Q8_0" else dequant_q8_k
    return deq(quantize(x, kind)).reshape(x.shape)


DEQUANT = {"Q8_0": dequant_q8_0, "Q4_0": dequant_q4_0, "Q5_0": dequant_q5_0,
           "Q4_K": dequant_q4_k, "Q6_K": dequant_q6_k}


def dequantize(raw, kind: str, shape) -> np.ndarray:
    return DEQUANT[kind](raw).reshape(shape)


# ---------------------------------------------------------------- quantizzazione

def _round_away(v: np.ndarray) -> np.ndarray:
    """roundf del C: metà arrotondata lontano da zero (np.round arrotonda al pari).

    La somma con 0,5 si fa in float64, dove è esatta: in float32 0,49999997 + 0,5 darebbe 1.
    """
    v64 = v.astype(np.float64)
    return np.sign(v64) * np.floor(np.abs(v64) + 0.5)


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Come quantize_row_q8_0_ref: d = max|x| / 127 per blocco, q = round(x / d)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = _round_away(x * inv[:, None]).astype(np.int8)
    out = np.empty((x.shape[0], 34), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Come quantize_row_q4_0_ref: d = (valore di modulo massimo, col segno) / −8."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    idx = np.abs(x).argmax(axis=1)  # primo indice di modulo massimo, come il ciclo del C
    vmax = x[np.arange(x.shape[0]), idx]
    d = vmax / np.float32(-8)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    v = (x * inv[:, None]) + np.float32(8.5)  # float32, poi troncamento verso zero come (int8_t)
    q = np.minimum(15, np.trunc(v).astype(np.int32)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def quantize_q8_k(x: np.ndarray) -> bytes:
    """Attivazioni per i K-quant (quantize_row_q8_K_ref): d = max / −127 col segno, float32.

    nearest_int di ggml arrotonda al pari (np.rint); bsums = somme dei q a gruppi di 16.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 256)
    idx = np.abs(x).argmax(axis=1)
    vmax = x[np.arange(x.shape[0]), idx]
    out = np.zeros((x.shape[0], 292), np.uint8)
    nz = vmax != 0
    iscale = np.zeros_like(vmax)
    iscale[nz] = np.float32(-127) / vmax[nz]
    q = np.minimum(127, np.rint(x * iscale[:, None]).astype(np.int32)).astype(np.int8)
    q[~nz] = 0
    d = np.zeros_like(vmax)
    d[nz] = np.float32(1) / iscale[nz]
    out[:, 0:4] = d.astype("<f4").view(np.uint8).reshape(-1, 4)
    out[:, 4:260] = q.view(np.uint8)
    out[:, 260:292] = q.astype(np.int16).reshape(-1, 16, 16).sum(axis=2).astype("<i2").view(np.uint8)
    return out.tobytes()


QUANTIZE = {"Q8_0": quantize_q8_0, "Q4_0": quantize_q4_0, "Q8_K": quantize_q8_k}


def quantize(x: np.ndarray, kind: str) -> bytes:
    """Quantizza le righe di x (l'ultima dimensione deve essere multipla del blocco)."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] % QK[kind]:
        raise ValueError(f"{kind}: la riga ({x.shape[-1]}) non è multipla del blocco {QK[kind]}")
    return QUANTIZE[kind](x)


# ---------------------------------------------------------------- prodotti scalari di riferimento

def vec_dot(w_raw, kind: str, x_raw) -> float:
    """Prodotto scalare peso·attivazione come lo calcola il C: interi esatti per blocco,
    poi scale in float. Restituisce un float64 (il C accumula in float32: piccola differenza)."""
    vt = VEC_DOT_TYPE[kind]
    w, x = _blocks(w_raw, kind), _blocks(x_raw, vt)
    if vt == "Q8_0":
        xq = x[:, 2:].view(np.int8).astype(np.int64)
        dx = _f16(x[:, :2]).astype(np.float64)
        if kind == "Q8_0":
            wq = w[:, 2:].view(np.int8).astype(np.int64)
        elif kind == "Q4_0":
            wq = _nibbles(w[:, 2:]).astype(np.int64) - 8
        else:
            wq = _q5_0_values(w).astype(np.int64)
        isum = (wq * xq).sum(axis=1)
        return float((isum * _f16(w[:, :2]).astype(np.float64) * dx).sum())

    dx = x[:, 0:4].copy().view("<f4")[:, 0].astype(np.float64)
    xq = x[:, 4:260].view(np.int8).astype(np.int64)
    bsums = x[:, 260:292].copy().view("<i2").astype(np.int64)
    if kind == "Q4_K":
        sc, m = q4_k_scales(w[:, 4:16])
        q = _q4_k_values(w).astype(np.int64).reshape(-1, 8, 32)
        isum = (sc * (q * xq.reshape(-1, 8, 32)).sum(axis=2)).sum(axis=1)
        msum = (m * bsums.reshape(-1, 8, 2).sum(axis=2)).sum(axis=1)
        d, dmin = _f16(w[:, 0:2]).astype(np.float64), _f16(w[:, 2:4]).astype(np.float64)
        return float((dx * (d * isum - dmin * msum)).sum())
    sc = w[:, 192:208].view(np.int8).astype(np.int64)
    q = _q6_k_values(w).astype(np.int64).reshape(-1, 16, 16)
    isum = (sc * (q * xq.reshape(-1, 16, 16)).sum(axis=2)).sum(axis=1)
    return float((_f16(w[:, 208:210]).astype(np.float64) * dx * isum).sum())
