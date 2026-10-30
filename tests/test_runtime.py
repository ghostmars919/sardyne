"""Runtime C (Fase 3): kernel confrontati con NumPy e forward completo confrontato con la reference."""
import gc
from pathlib import Path

import numpy as np
import pytest

from engine import capi

pytestmark = pytest.mark.skipif(not capi.available(), reason="libengine non compilata: eseguire make")

from engine.loader.convert import convert  # noqa: E402
from engine.loader.gguf_writer import f32_to_bf16  # noqa: E402
from engine.ref import ops  # noqa: E402
from engine.ref.llama import Llama  # noqa: E402
from engine.runtime import Context, Model, generate, set_simd, simd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "models" / "gguf"
F32, F16, BF16 = 0, 1, 30


def lib():
    return capi.lib()


def ptr(a: np.ndarray) -> int:
    assert a.flags["C_CONTIGUOUS"]
    return a.ctypes.data


HAS_SIMD = set_simd(True)


@pytest.fixture(params=[False, True], ids=["scalare", "avx2"])
def kernels(request):
    """Esegue il test con i kernel scalari e con quelli AVX2 (se la CPU li ha)."""
    if request.param and not HAS_SIMD:
        pytest.skip("CPU senza AVX2/FMA/F16C")
    assert set_simd(request.param) == request.param
    yield request.param
    set_simd(True)


# ---------------------------------------------------------------- kernel

def test_f16_conversion_all_values():
    bits = np.arange(1 << 16, dtype=np.uint16)
    out = np.empty(1 << 16, np.float32)
    lib().op_f16_to_f32_n(ptr(bits), ptr(out), bits.size)
    expected = bits.view(np.float16).astype(np.float32)
    nan = np.isnan(expected)
    assert np.array_equal(np.isnan(out), nan)
    assert np.array_equal(out[~nan].view(np.uint32), expected[~nan].view(np.uint32))  # bit per bit, anche ±0


@pytest.mark.parametrize("wtype", [F32, F16, BF16], ids=["f32", "f16", "bf16"])
@pytest.mark.parametrize("n_tok", [1, 2, 3, 4, 5, 8, 9])  # resti dei blocchi da 4 token
@pytest.mark.parametrize("n_in", [67, 96])  # con e senza coda non multipla di 8 e di 32
def test_matmul(kernels, wtype, n_tok, n_in):
    rng = np.random.default_rng(0)
    n_out = 45  # dispari: l'ultima riga resta fuori dalle coppie del micro-kernel
    x = rng.standard_normal((n_tok, n_in)).astype(np.float32)
    w32 = rng.standard_normal((n_out, n_in)).astype(np.float32)
    if wtype == F16:
        w = w32.astype(np.float16)
        w_ref = w.astype(np.float32)
    elif wtype == BF16:
        w = f32_to_bf16(w32)
        w_ref = (w.astype(np.uint32) << 16).view(np.float32)
    else:
        w, w_ref = w32, w32
    out = np.empty((n_tok, n_out), np.float32)
    lib().op_matmul(ptr(out), ptr(x), ptr(w), wtype, n_tok, n_in, n_out)
    np.testing.assert_allclose(out, x @ w_ref.T, rtol=1e-5, atol=1e-5)

    # Righe calcolate a pezzi (come fanno i thread): stesso risultato, bit per bit.
    parts = np.full_like(out, np.nan)
    for r0, r1 in [(0, 10), (10, 30), (30, n_out)]:
        lib().op_matmul_rows(ptr(parts), ptr(x), ptr(w), wtype, n_tok, n_in, n_out, r0, r1)
    assert np.array_equal(parts, out)


@pytest.mark.parametrize("n", [1, 7, 8, 64, 100])
def test_dot_and_axpy(kernels, n):
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((2, n)).astype(np.float32)
    assert lib().op_dot_f32(ptr(a), ptr(b), n) == pytest.approx(float(a.astype(np.float64) @ b), rel=1e-5, abs=1e-5)
    y = rng.standard_normal(n).astype(np.float32)
    expected = y + np.float32(0.5) * a
    lib().op_axpy(ptr(y), 0.5, ptr(a), n)
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)


def test_simd_selection():
    assert set_simd(False) is False and simd() is False
    assert set_simd(True) == HAS_SIMD


def test_rmsnorm_softmax_silu():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(300) * 5).astype(np.float32)
    w = rng.standard_normal(300).astype(np.float32)
    out = np.empty_like(x)
    lib().op_rmsnorm(ptr(out), ptr(x), ptr(w), x.size, 1e-5)
    np.testing.assert_allclose(out, ops.rms_norm(x, w, 1e-5), rtol=1e-5, atol=1e-6)

    s = np.array([1000.0, 1001.0, -50.0, 3.0], np.float32)  # valori grandi: serve la sottrazione del max
    expected = ops.softmax(s)
    lib().op_softmax(ptr(s), s.size)
    np.testing.assert_allclose(s, expected, rtol=1e-6)

    g = np.array([-100.0, -1.0, 0.0, 2.0, 50.0], np.float32)
    up = np.array([3.0, 2.0, 1.0, -1.0, 0.5], np.float32)
    expected = ops.silu(g) * up
    lib().op_silu_mul(ptr(g), ptr(up), g.size)
    np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-30)


@pytest.mark.parametrize("interleaved", [False, True], ids=["neox", "interleaved"])
def test_rope(interleaved):
    n_pos, hd, n_heads, theta = 40, 16, 3, 10000.0
    cos = np.empty((n_pos, hd // 2), np.float32)
    sin = np.empty_like(cos)
    lib().op_rope_table(ptr(cos), ptr(sin), n_pos, hd, theta)
    ref_cos, ref_sin = ops.rope_cos_sin(n_pos, hd, theta)
    np.testing.assert_allclose(cos, ref_cos[:, :hd // 2], atol=1e-7)
    np.testing.assert_allclose(sin, ref_sin[:, :hd // 2], atol=1e-7)

    x = np.random.default_rng(2).standard_normal((n_heads, hd)).astype(np.float32)
    pos = 33
    rope = ops.apply_rope_interleaved if interleaved else ops.apply_rope
    expected = rope(x[:, None, :], ref_cos[pos:pos + 1], ref_sin[pos:pos + 1])[:, 0]
    row_c, row_s = np.ascontiguousarray(cos[pos]), np.ascontiguousarray(sin[pos])
    lib().op_rope(ptr(x), n_heads, hd, ptr(row_c), ptr(row_s), int(interleaved))
    np.testing.assert_allclose(x, expected, rtol=1e-5, atol=1e-6)


def test_attention_gqa(kernels):
    rng = np.random.default_rng(3)
    n_head, n_kv_head, hd, stride, n_kv = 6, 2, 8, 20, 13  # cache con 20 posti, 13 occupati
    q = rng.standard_normal((n_head, hd)).astype(np.float32)
    kc = rng.standard_normal((n_kv_head, stride, hd)).astype(np.float32)
    vc = rng.standard_normal((n_kv_head, stride, hd)).astype(np.float32)
    out = np.empty_like(q)
    scores = np.empty(stride, np.float32)
    lib().op_attention(ptr(out), ptr(q), ptr(kc), ptr(vc), n_head, n_kv_head, hd, n_kv, stride, ptr(scores))
    n_rep = n_head // n_kv_head
    k = ops.repeat_kv(kc[:, :n_kv], n_rep)
    v = ops.repeat_kv(vc[:, :n_kv], n_rep)
    expected = ops.scaled_dot_product_attention(q[:, None, :], k, v)[:, 0]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- modelli sintetici

TOKENS = [3, 14, 15, 9, 26, 5, 35, 8, 2, 7, 1, 60]


@pytest.fixture(scope="module", params=[
    ("llama", "f32"), ("llama", "f16"), ("llama", "bf16"),
    ("llama-bias-headdim", "bf16"), ("qwen2", "f32"), ("qwen2", "bf16"),
    ("qwen3", "bf16"), ("qwen3", "q8_0"),
    ("llama", "q8_0"), ("llama", "q4_0"), ("llama-bias-headdim", "q8_0"), ("qwen2", "q4_0")],
    ids=lambda p: "-".join(p))
def synthetic(request, tmp_path_factory):
    from test_convert import make_hf

    variant, outtype = request.param
    kwargs = {"llama": {}, "llama-bias-headdim": dict(bias=True, tied=False, head_dim=16),
              "qwen2": dict(arch="Qwen2ForCausalLM", tied=False),
              "qwen3": dict(arch="Qwen3ForCausalLM", head_dim=16)}[variant]
    tmp = tmp_path_factory.mktemp(f"{variant}-{outtype}")
    hf = make_hf(tmp / "hf", **kwargs)
    path = tmp / "m.gguf"
    convert(hf, path, outtype)
    return path


def test_forward_matches_numpy(synthetic, kernels):
    # Con pesi quantizzati il riferimento emula anche le attivazioni int8 del runtime: su 2 strati
    # la differenza resta all'ordine dell'arrotondamento fp32.
    ref = Llama.from_gguf(synthetic, max_seq=32, emulate_quant=True).forward(TOKENS)
    ctx = Context(Model(synthetic), max_seq=32, n_batch=len(TOKENS), logits_all=True)
    np.testing.assert_allclose(ctx.eval(TOKENS), ref, atol=1e-4, rtol=0)


def test_results_do_not_depend_on_thread_count(synthetic):
    model = Model(synthetic)
    outs = []
    for n_threads in (1, 2, 3, 4, 7):
        ctx = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True, n_threads=n_threads)
        assert ctx.n_threads == n_threads
        logits = ctx.eval(TOKENS[:8])
        outs.append(np.concatenate([logits, ctx.eval(TOKENS[8:])]))  # prefill + seguito
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])  # bit per bit


def test_many_threads_stress(synthetic):
    # Più thread che core e molte operazioni brevi: cerca race nel thread pool.
    model = Model(synthetic)
    ref = Context(model, max_seq=32, n_threads=1)
    ctx = Context(model, max_seq=32, n_threads=8)
    for _ in range(20):
        ref.reset()
        ctx.reset()
        a, b = ref.eval(TOKENS[:6]), ctx.eval(TOKENS[:6])
        for t in TOKENS[6:]:
            a, b = ref.eval([t]), ctx.eval([t])
        assert np.array_equal(a, b)


def test_info_matches_numpy_config(synthetic):
    cfg = Llama.from_gguf(synthetic, max_seq=8).cfg
    i = Model(synthetic).info
    assert (i.n_vocab, i.dim, i.n_layer, i.n_head, i.n_kv_head, i.head_dim, i.ffn_dim) == \
           (cfg.vocab_size, cfg.dim, cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.ffn_dim)
    assert (bool(i.rope_interleaved), bool(i.qkv_bias), bool(i.tied_embeddings)) == \
           (cfg.rope_interleaved, cfg.qkv_bias, cfg.tie_embeddings)
    assert i.rope_theta == pytest.approx(cfg.rope_theta) and i.norm_eps == pytest.approx(cfg.norm_eps)


def test_kv_cache_and_batching(synthetic):
    model = Model(synthetic)
    full = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    # Prefill in blocchi da 5 (5 + 5 + 2) con logits_all, poi gli stessi token uno alla volta.
    chunked = Context(model, max_seq=32, n_batch=5, logits_all=True).eval(TOKENS)
    np.testing.assert_allclose(chunked, full, atol=1e-5, rtol=0)
    ctx = Context(model, max_seq=32, n_batch=4)
    last = ctx.eval(TOKENS[:7])  # 7 token in blocchi da 4: solo i logits dell'ultimo
    np.testing.assert_allclose(last, full[6], atol=1e-5, rtol=0)
    for i, t in enumerate(TOKENS[7:], start=7):
        np.testing.assert_allclose(ctx.eval([t]), full[i], atol=1e-5, rtol=0)
    assert ctx.n_past == len(TOKENS)
    ctx.reset()
    assert ctx.n_past == 0
    np.testing.assert_allclose(ctx.eval(TOKENS), full[-1], atol=1e-5, rtol=0)


def test_truncate_and_batching(synthetic):
    model = Model(synthetic)
    full = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    ctx = Context(model, max_seq=32, n_batch=4)
    ctx.eval(TOKENS[:7])
    ctx.truncate(3)  # si riparte da metà, con token diversi da quelli di prima
    assert ctx.n_past == 3
    ctx.eval([60, 61])
    ctx.truncate(3)
    np.testing.assert_allclose(ctx.eval(TOKENS[3:]), full[-1], atol=1e-5, rtol=0)
    ctx.truncate(ctx.n_past)  # nessun cambiamento
    assert ctx.n_past == len(TOKENS)
    for bad in (-1, len(TOKENS) + 1):
        with pytest.raises(ValueError, match="posizione"):
            ctx.truncate(bad)
        assert ctx.n_past == len(TOKENS)


def test_generate_matches_numpy(synthetic):
    ref = Llama.from_gguf(synthetic, max_seq=32, emulate_quant=True)
    ctx = Context(Model(synthetic), max_seq=32)
    assert generate(ctx, TOKENS[:4], 12) == ref.generate(TOKENS[:4], 12)
    assert generate(ctx, TOKENS[:4], 0) == []
    stop = ref.generate(TOKENS[:4], 12)[5]
    assert generate(ctx, TOKENS[:4], 12, eos_ids={stop}) == ref.generate(TOKENS[:4], 12, eos_ids={stop})


def test_errors_leave_context_unchanged(synthetic):
    model = Model(synthetic)
    ctx = Context(model, max_seq=8, n_batch=4)
    ctx.eval([1, 2, 3])
    for bad, msg in [([1] * 6, "contesto pieno"), ([64], "vocabolario"), ([-1], "vocabolario"), ([], "nessun")]:
        with pytest.raises(ValueError, match=msg):
            ctx.eval(bad)
        assert ctx.n_past == 3
    with pytest.raises(ValueError, match="n_batch"):
        Context(model, max_seq=8, n_batch=0)
    with pytest.raises(ValueError, match="max_seq"):
        Context(model, max_seq=0)
    with pytest.raises(ValueError, match="n_threads"):
        Context(model, max_seq=8, n_threads=-1)
    all_ctx = Context(model, max_seq=16, n_batch=2, logits_all=True)
    assert all_ctx.eval([1, 2, 3, 4, 5]).shape == (5, model.info.n_vocab)  # Python divide in blocchi


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="impossibile aprire"):
        Model(tmp_path / "manca.gguf")


def test_rope_freqs_and_unknown_tensors_are_rejected(tmp_path):
    from engine.loader.gguf import GGUFFile, VALUE_TYPES
    from engine.loader.gguf_writer import GGUFWriter
    from test_convert import make_hf

    hf = make_hf(tmp_path / "hf")
    src_path = tmp_path / "m.gguf"
    convert(hf, src_path, "f32")
    src = GGUFFile(src_path)
    for extra, msg in [("rope_freqs.weight", "rope_freqs"), ("blk.0.strano.weight", "non supportato")]:
        w = GGUFWriter(tmp_path / f"{extra}.gguf")
        for k, v in src.metadata.items():
            vt, et = src.metadata_types[k]
            w.add(k, v, VALUE_TYPES[vt][0], VALUE_TYPES[et][0] if et is not None else None)
        for name in src.tensors:
            w.add_tensor(name, src.get_f32(name), "F32")
        w.add_tensor(extra, np.ones(4, np.float32), "F32")
        w.write()
        with pytest.raises(ValueError, match=msg):
            Model(w.path)


# ---------------------------------------------------------------- modelli reali (se scaricati)

PROMPT_IDS = {"SmolLM2-135M-Instruct-f16.gguf": [504, 3575, 282, 4649, 314, 253, 1789, 30, 378, 5402],
              "qwen2.5-0.5b-instruct-fp16.gguf": [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279],
              "qwen3-0.6b-bf16.gguf": [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279]}


@pytest.mark.parametrize("name", sorted(PROMPT_IDS))
def test_real_model_matches_numpy(name):
    path = GGUF_DIR / name
    if not path.exists():
        pytest.skip(f"{name} non scaricato")
    ids = PROMPT_IDS[name]
    ref = Llama.from_gguf(path, max_seq=64)
    ref_logits, ref_gen = ref.forward(ids), ref.generate(ids, 20)
    del ref
    gc.collect()
    model = Model(path)
    ctx = Context(model, max_seq=64, n_batch=len(ids), logits_all=True)
    logits = ctx.eval(ids)
    np.testing.assert_allclose(logits, ref_logits, atol=1e-3, rtol=0)
    assert generate(Context(model, max_seq=64), ids, 20) == ref_gen
    one = Context(model, max_seq=64, n_batch=len(ids), logits_all=True, n_threads=1).eval(ids)
    assert np.array_equal(one, logits)  # 1 thread o tutti: stesso risultato


# ---------------------------------------------------------------- modelli quantizzati reali (Fase 5)

QUANT_FILES = {"SmolLM2-135M-Instruct-Q8_0.gguf": PROMPT_IDS["SmolLM2-135M-Instruct-f16.gguf"],
               "SmolLM2-135M-Instruct-Q4_K_M.gguf": PROMPT_IDS["SmolLM2-135M-Instruct-f16.gguf"],
               "qwen2.5-0.5b-instruct-q8_0.gguf": PROMPT_IDS["qwen2.5-0.5b-instruct-fp16.gguf"],
               "qwen2.5-0.5b-instruct-q4_k_m.gguf": PROMPT_IDS["qwen2.5-0.5b-instruct-fp16.gguf"]}


@pytest.mark.parametrize("name", sorted(QUANT_FILES))
def test_real_quantized_model(name):
    """Su 24-30 strati l'errore delle attivazioni int8 si amplifica (anche in NumPy, una perturbazione
    di un ulp sposta i logits di decimi): si verificano il token più probabile e il determinismo."""
    path = GGUF_DIR / name
    if not path.exists():
        pytest.skip(f"{name} non scaricato")
    ids = QUANT_FILES[name]
    ref = Llama.from_gguf(path, max_seq=64, emulate_quant=True).forward(ids)
    gc.collect()
    model = Model(path)
    logits = Context(model, max_seq=64, n_batch=len(ids), logits_all=True).eval(ids)
    assert np.mean(logits.argmax(-1) == ref.argmax(-1)) >= 0.8
    assert np.abs(logits - ref).mean() < 0.5
    one = Context(model, max_seq=64, n_batch=len(ids), logits_all=True, n_threads=1).eval(ids)
    assert np.array_equal(one, logits)  # 1 thread o tutti: identici bit per bit


def test_real_large_model():
    """Qwen2.5-3B Q4_K_M: K-quant in tutte le matrici. La reference NumPy in fp32 non ci sta in RAM
    (12 GB): si verificano una previsione ovvia e il determinismo tra thread."""
    path = GGUF_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
    if not path.exists():
        pytest.skip(f"{path.name} non scaricato")
    ids = PROMPT_IDS["qwen2.5-0.5b-instruct-fp16.gguf"]  # "The capital of France is Paris. It is the"
    model = Model(path)
    logits = Context(model, max_seq=64, n_batch=len(ids), logits_all=True).eval(ids)
    assert logits[4].argmax() == ids[5]  # "... France is" -> " Paris"
    one = Context(model, max_seq=64, n_batch=len(ids), logits_all=True, n_threads=1).eval(ids)
    assert np.array_equal(one, logits)


def test_unsupported_quant_type_is_rejected(tmp_path):
    from engine.loader.gguf import GGUFFile, VALUE_TYPES
    from engine.loader.gguf_writer import GGUFWriter
    from test_convert import make_hf

    hf = make_hf(tmp_path / "hf")
    src_path = tmp_path / "m.gguf"
    convert(hf, src_path, "f32")
    src = GGUFFile(src_path)
    w = GGUFWriter(tmp_path / "q4_1.gguf")
    for k, v in src.metadata.items():
        vt, et = src.metadata_types[k]
        w.add(k, v, VALUE_TYPES[vt][0], VALUE_TYPES[et][0] if et is not None else None)
    for name, t in src.tensors.items():
        if name == "blk.0.attn_q.weight":  # Q4_1: 32 valori in 20 byte, non supportato
            w.add_raw_tensor(name, t.ne, "Q4_1", bytes(t.n_elements // 32 * 20))
        else:
            w.add_tensor(name, src.get_f32(name), "F32")
    w.write()
    with pytest.raises(ValueError, match="Q4_1 non supportato"):
        Model(w.path)
