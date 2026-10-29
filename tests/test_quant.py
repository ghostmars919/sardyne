"""Formati quantizzati (Fase 5): reference NumPy e kernel C."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from engine import capi, quant
from engine.loader.gguf import GGML_TYPE_ID, GGUFFile

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models" / "gguf"
needs_c = pytest.mark.skipif(not capi.available(), reason="libengine non compilata: eseguire make")
WEIGHT_TYPES = ["Q8_0", "Q4_0", "Q5_0", "Q4_K", "Q6_K"]


def gguf(name):
    path = GGUF / name
    if not path.exists():
        pytest.skip(f"{name} non scaricato")
    return GGUFFile(path)


def random_blocks(kind: str, n_blocks: int, seed: int = 0) -> np.ndarray:
    """Blocchi con byte casuali ma scale f16 ragionevoli (niente inf/NaN)."""
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 256, (n_blocks, quant.BLOCK_BYTES[kind]), dtype=np.uint8)
    scale = lambda: (rng.uniform(1e-4, 1e-2, n_blocks).astype(np.float16)).view(np.uint8).reshape(-1, 2)
    if kind in ("Q8_0", "Q4_0", "Q5_0"):
        b[:, 0:2] = scale()
    elif kind == "Q4_K":
        b[:, 0:2], b[:, 2:4] = scale(), scale()
    elif kind == "Q6_K":
        b[:, 208:210] = scale()
        b[:, 192:208] = rng.integers(-60, 60, (n_blocks, 16)).astype(np.int8).view(np.uint8)
    return b


# ---------------------------------------------------------------- NumPy contro i file pubblicati

def test_published_q8_0_matches_f16_within_half_step():
    f16, q8 = gguf("SmolLM2-135M-Instruct-f16.gguf"), gguf("SmolLM2-135M-Instruct-Q8_0.gguf")
    for name in ["blk.0.attn_q.weight", "blk.29.ffn_down.weight", "token_embd.weight"]:
        w = f16.get_f32(name).reshape(-1, 32)
        deq = q8.get_f32(name).reshape(-1, 32)
        d = np.abs(w).max(axis=1, keepdims=True) / 127
        # mezzo passo, più l'errore dovuto alla scala salvata in f16 (2^-11 relativo, × 127)
        assert np.all(np.abs(deq - w) <= d * (0.5 + 127 * 2 ** -11) + 1e-12), name


def test_published_q4_k_m_error_by_type():
    f16, q4 = gguf("SmolLM2-135M-Instruct-f16.gguf"), gguf("SmolLM2-135M-Instruct-Q4_K_M.gguf")
    limits = {"Q8_0": 0.01, "Q6_K": 0.03, "Q5_0": 0.07, "Q4_K": 0.11}  # errore RMS relativo massimo
    seen = set()
    for name, t in q4.tensors.items():
        if t.type_name in limits and t.type_name not in seen:
            w, deq = f16.get_f32(name), q4.get_f32(name)
            rel = np.sqrt(np.mean((deq - w) ** 2) / np.mean(w ** 2))
            assert rel < limits[t.type_name], (name, t.type_name, rel)
            seen.add(t.type_name)
    assert seen == set(limits)


def test_our_q8_0_quantizer_matches_llama_cpp():
    f16, q8 = gguf("SmolLM2-135M-Instruct-f16.gguf"), gguf("SmolLM2-135M-Instruct-Q8_0.gguf")
    for name in ["blk.0.attn_q.weight", "blk.15.ffn_up.weight", "token_embd.weight"]:
        assert quant.quantize(f16.get_f32(name), "Q8_0") == q8.raw(name).tobytes(), name


# ---------------------------------------------------------------- quantizzatori

@pytest.mark.parametrize("kind", ["Q8_0", "Q4_0"])
def test_quantize_roundtrip(kind):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((8, 128)) * rng.uniform(0.01, 10, (8, 1))).astype(np.float32)
    x[0, :32] = 0  # blocco tutto a zero: scala 0, nessuna divisione per zero
    raw = quant.quantize(x, kind)
    assert len(raw) == x.size // 32 * quant.BLOCK_BYTES[kind]
    deq = quant.dequantize(np.frombuffer(raw, np.uint8), kind, x.shape)
    assert np.all(deq[0, :32] == 0)
    blocks = x.reshape(-1, 32)
    step = np.abs(blocks).max(axis=1, keepdims=True) / (127 if kind == "Q8_0" else 8)
    tol = step * (0.5 + 2 ** -10 * (127 if kind == "Q8_0" else 8)) + 1e-12
    assert np.all(np.abs(deq.reshape(-1, 32) - blocks) <= (tol if kind == "Q8_0" else 2 * tol))
    with pytest.raises(ValueError):
        quant.quantize(np.ones(40, np.float32), kind)


def test_q8_k_activation_layout():
    x = np.random.default_rng(2).standard_normal(512).astype(np.float32)
    raw = np.frombuffer(quant.quantize(x, "Q8_K"), np.uint8).reshape(2, 292)
    q = raw[:, 4:260].view(np.int8)
    bsums = raw[:, 260:].copy().view("<i2")
    assert np.array_equal(bsums, q.astype(np.int32).reshape(2, 16, 16).sum(axis=2))
    assert np.abs(quant.dequant_q8_k(raw) - x).max() <= np.abs(x).max() / 127


# ---------------------------------------------------------------- kernel C contro NumPy

def lib():
    return capi.lib()


def ptr(a):
    return a.ctypes.data


@pytest.fixture(params=[False, True], ids=["scalare", "avx2"])
def kernels(request):
    from engine.runtime import set_simd

    if request.param and not set_simd(True):
        pytest.skip("CPU senza AVX2")
    set_simd(request.param)
    yield request.param
    set_simd(True)


@needs_c
def test_f32_to_f16_matches_numpy():
    rng = np.random.default_rng(3)
    v = (rng.standard_normal(100000) * 10.0 ** rng.integers(-9, 6, 100000)).astype(np.float32)
    v = np.concatenate([v, np.array([65504, 65519, 65520, 7e4, 6e-8, 3e-8, 2.98e-8, 0, -0.0, 5.96e-8,
                                     np.inf, -np.inf], np.float32)])
    c = np.array([lib().op_f32_to_f16(float(x)) for x in v], np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 7e4 -> inf
        assert np.array_equal(c, v.astype(np.float16).view(np.uint16))


@needs_c
@pytest.mark.parametrize("kind", WEIGHT_TYPES)
def test_dequant_row_matches_numpy(kind):
    n = 512
    row = random_blocks(kind, n // quant.QK[kind], seed=4).reshape(-1)
    out = np.empty(n, np.float32)
    lib().op_row_to_f32(ptr(row), GGML_TYPE_ID[kind], ptr(out), n)
    assert np.array_equal(out, quant.dequantize(row, kind, (n,)))  # bit per bit


@needs_c
@pytest.mark.parametrize("vt", ["Q8_0", "Q8_K"])
def test_activation_quantization_matches_numpy(vt):
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((3, 512)) * np.array([[1e-3], [1.0], [300.0]])).astype(np.float32)
    x[1, :256] = 0  # blocco nullo
    x[2, 7] = 1e4   # un valore dominante
    n_rows, n = x.shape
    out = np.zeros(lib().op_row_size(GGML_TYPE_ID[vt], n) * n_rows, np.uint8)
    lib().op_quantize_rows(GGML_TYPE_ID[vt], ptr(x), ptr(out), n, 0, n_rows)
    assert out.tobytes() == quant.quantize(x, vt)


@needs_c
@pytest.mark.parametrize("kind", WEIGHT_TYPES)
def test_vec_dot_matches_numpy(kernels, kind):
    n = 1024
    w = random_blocks(kind, n // quant.QK[kind], seed=6).reshape(-1)
    x = np.random.default_rng(7).standard_normal(n).astype(np.float32)
    xq = np.frombuffer(quant.quantize(x, quant.VEC_DOT_TYPE[kind]), np.uint8).copy()
    got = lib().op_vec_dot_q(GGML_TYPE_ID[kind], ptr(w), ptr(xq), n)
    assert got == pytest.approx(quant.vec_dot(w, kind, xq), rel=1e-5, abs=1e-5)
    # e il prodotto con le attivazioni quantizzate approssima quello in float: l'errore int8 delle
    # attivazioni (≤ 1/254 del massimo del blocco) resta una piccola frazione di ‖w‖·‖x‖
    wd = quant.dequantize(w, kind, (n,)).astype(np.float64)
    assert abs(got - float(wd @ x)) <= 0.01 * float(np.linalg.norm(wd) * np.linalg.norm(x))


@needs_c
@pytest.mark.parametrize("kind", ["Q4_K", "Q6_K"])
def test_k_quants_scalar_and_avx2_are_identical(kind):
    from engine.runtime import set_simd

    if not set_simd(True):
        pytest.skip("CPU senza AVX2")
    n = 2048
    w = random_blocks(kind, n // 256, seed=8).reshape(-1)
    xq = np.frombuffer(quant.quantize(np.random.default_rng(9).standard_normal(n).astype(np.float32), "Q8_K"),
                       np.uint8).copy()
    res = {}
    for simd in (False, True):
        set_simd(simd)
        res[simd] = lib().op_vec_dot_q(GGML_TYPE_ID[kind], ptr(w), ptr(xq), n)
    set_simd(True)
    assert res[False] == res[True]  # parte intera esatta, stessa combinazione in fp32


@needs_c
@pytest.mark.parametrize("kind", WEIGHT_TYPES)
def test_matmul_q_rows_split(kernels, kind):
    n_in, n_out, n_tok = 512, 13, 3
    w = random_blocks(kind, n_out * n_in // quant.QK[kind], seed=10).reshape(n_out, -1)
    x = np.random.default_rng(11).standard_normal((n_tok, n_in)).astype(np.float32)
    vt = quant.VEC_DOT_TYPE[kind]
    xq = np.frombuffer(quant.quantize(x, vt), np.uint8).copy()
    out = np.empty((n_tok, n_out), np.float32)
    lib().op_matmul_q_rows(ptr(out), ptr(xq), ptr(w), GGML_TYPE_ID[kind], n_tok, n_in, n_out, 0, n_out)
    xrow = len(xq) // n_tok
    for t in range(n_tok):
        for j in range(n_out):
            assert out[t, j] == pytest.approx(quant.vec_dot(w[j], kind, xq[t * xrow:(t + 1) * xrow]), rel=1e-5, abs=1e-5)
    parts = np.full_like(out, np.nan)
    for r0, r1 in [(0, 5), (5, 6), (6, n_out)]:
        lib().op_matmul_q_rows(ptr(parts), ptr(xq), ptr(w), GGML_TYPE_ID[kind], n_tok, n_in, n_out, r0, r1)
    assert np.array_equal(parts, out)


@needs_c
@pytest.mark.parametrize("name", ["SmolLM2-135M-Instruct-Q4_K_M.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"])
def test_real_rows(kernels, name):
    f = gguf(name)
    rng = np.random.default_rng(12)
    done = set()
    for tname, t in f.tensors.items():
        if t.type_name not in WEIGHT_TYPES or t.type_name in done:
            continue
        done.add(t.type_name)
        row = np.ascontiguousarray(f.raw(tname)[5])
        n, tid = t.ne[0], GGML_TYPE_ID[t.type_name]
        out = np.empty(n, np.float32)
        lib().op_row_to_f32(ptr(row), tid, ptr(out), n)
        assert np.array_equal(out, quant.dequantize(row, t.type_name, (n,)))
        x = rng.standard_normal(n).astype(np.float32)
        xq = np.frombuffer(quant.quantize(x, quant.VEC_DOT_TYPE[t.type_name]), np.uint8).copy()
        got = lib().op_vec_dot_q(tid, ptr(row), ptr(xq), n)
        assert got == pytest.approx(quant.vec_dot(row, t.type_name, xq), rel=1e-5, abs=1e-5)
    assert done
