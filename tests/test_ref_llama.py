import gc
import math
from pathlib import Path

import numpy as np
import pytest

from sardyne.loader.safetensors import SafeTensorsFile, bf16_to_f32
from sardyne.ref import ops
from sardyne.ref.llama import Llama, LlamaConfig

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["SmolLM2-135M", "Qwen2.5-0.5B"]


# ---------------------------------------------------------------- ops

def test_rms_norm_formula():
    x = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
    w = np.linspace(0.5, 2, 8, dtype=np.float32)
    expected = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(ops.rms_norm(x, w, 1e-6), expected, rtol=1e-6)


def test_rope_preserves_norm():
    # Una rotazione non cambia la lunghezza del vettore.
    cos, sin = ops.rope_cos_sin(64, 16, 10000.0)
    x = np.random.default_rng(0).standard_normal((2, 64, 16)).astype(np.float32)
    y = ops.apply_rope(x, cos, sin)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)


def test_rope_dot_product_depends_only_on_distance():
    # <rope(q, m), rope(k, n)> dipende solo da m - n: spostare entrambe le posizioni non cambia nulla.
    hd = 32
    cos, sin = ops.rope_cos_sin(128, hd, 10000.0)
    rng = np.random.default_rng(1)
    q, k = rng.standard_normal((2, 1, 1, hd))

    def score(m, n):
        qm = ops.apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = ops.apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return float((qm * kn).sum())

    for shift in (1, 17, 90):
        assert math.isclose(score(10, 3), score(10 + shift, 3 + shift), rel_tol=1e-5)
    assert not math.isclose(score(10, 3), score(10, 4), rel_tol=1e-3)


def test_rope_half_equals_interleaved_after_permutation():
    # HF ruota le coppie (i, i + hd/2), llama.cpp le coppie adiacenti (2i, 2i+1).
    # Sono la stessa operazione a meno di una permutazione delle dimensioni: per questo
    # il convertitore GGUF permuta le righe di W_q e W_k (lo ritroveremo in Fase 2).
    hd, n_pos = 16, 10
    cos, sin = ops.rope_cos_sin(n_pos, hd, 10000.0)
    x = np.random.default_rng(2).standard_normal((1, n_pos, hd)).astype(np.float32)
    perm = np.stack([np.arange(hd // 2), np.arange(hd // 2) + hd // 2], axis=1).reshape(-1)

    def rope_interleaved(v):
        out = np.empty_like(v)
        c, s = cos[:, :hd // 2], sin[:, :hd // 2]
        a, b = v[..., 0::2], v[..., 1::2]
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return out

    np.testing.assert_allclose(rope_interleaved(x[..., perm]), ops.apply_rope(x, cos, sin)[..., perm],
                               rtol=1e-5, atol=1e-6)


def test_repeat_kv_maps_query_head_to_kv_head():
    kv = np.arange(3)[:, None, None] * np.ones((3, 2, 4))
    rep = ops.repeat_kv(kv, 4)
    assert rep.shape == (12, 2, 4)
    for h in range(12):
        assert np.all(rep[h] == h // 4)


def test_swiglu_formula():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    g, u, d = rng.standard_normal((16, 8)), rng.standard_normal((16, 8)), rng.standard_normal((8, 16))
    a = x @ g.T
    expected = ((a / (1 + np.exp(-a))) * (x @ u.T)) @ d.T
    np.testing.assert_allclose(ops.swiglu_mlp(x, g, u, d), expected, rtol=1e-5)


def test_bf16_conversion():
    # bf16 = 16 bit alti del float32: valori rappresentabili esattamente tornano identici.
    vals = np.array([0.0, 1.0, -2.5, 0.15625, 3.0e38, -1.0e-30], np.float32)
    bits = (vals.view(np.uint32) >> 16).astype(np.uint16)
    back = bf16_to_f32(bits)
    np.testing.assert_array_equal(back[:4], vals[:4])
    np.testing.assert_allclose(back, vals, rtol=2 ** -7)  # 7 bit di mantissa


def test_safetensors_roundtrip(tmp_path):
    safetensors_np = pytest.importorskip("safetensors.numpy")
    rng = np.random.default_rng(4)
    tensors = {"a": rng.standard_normal((3, 5)).astype(np.float32),
               "b": rng.standard_normal(7).astype(np.float16)}
    path = tmp_path / "t.safetensors"
    safetensors_np.save_file(tensors, str(path))
    f = SafeTensorsFile(path)
    assert set(f.keys()) == {"a", "b"}
    np.testing.assert_array_equal(f.get_f32("a"), tensors["a"])
    np.testing.assert_array_equal(f.get_f32("b"), tensors["b"].astype(np.float32))


# ---------------------------------------------------------------- modello piccolo con pesi casuali

@pytest.fixture(scope="module", params=[False, True], ids=["rope-neox", "rope-interleaved"])
def tiny(request):
    # Ogni test sul modello piccolo gira con entrambe le convenzioni della RoPE.
    cfg = LlamaConfig(vocab_size=64, dim=32, n_layer=2, n_head=4, n_kv_head=2, head_dim=8,
                      ffn_dim=48, norm_eps=1e-5, rope_theta=10000.0, tie_embeddings=True,
                      qkv_bias=True, max_pos=64, rope_interleaved=request.param)
    rng = np.random.default_rng(5)

    def w(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    embed = w(cfg.vocab_size, cfg.dim)
    layers = []
    for _ in range(cfg.n_layer):
        qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
        layers.append({
            "attn_norm": np.ones(cfg.dim, np.float32), "mlp_norm": np.ones(cfg.dim, np.float32),
            "wq": w(qd, cfg.dim), "wk": w(kvd, cfg.dim), "wv": w(kvd, cfg.dim), "wo": w(cfg.dim, qd),
            "bq": w(qd), "bk": w(kvd), "bv": w(kvd),
            "w_gate": w(cfg.ffn_dim, cfg.dim), "w_up": w(cfg.ffn_dim, cfg.dim),
            "w_down": w(cfg.dim, cfg.ffn_dim),
        })
    params = {"embed": embed, "lm_head": embed, "norm": np.ones(cfg.dim, np.float32), "layers": layers}
    return Llama(cfg, params, max_seq=32)


def test_tiny_stays_float32(tiny):
    # Regressione: uno scalare np.float64 promuoveva tutto a float64 (20x più lento).
    assert tiny.forward([1, 2, 3]).dtype == np.float32


def test_tiny_causality(tiny):
    a = [5, 9, 2, 33, 7, 1]
    b = a.copy()
    b[4] = 60
    la, lb = tiny.forward(a), tiny.forward(b)
    np.testing.assert_allclose(la[:4], lb[:4], rtol=1e-5, atol=1e-5)
    assert not np.allclose(la[4:], lb[4:])


def test_tiny_kv_cache_matches_full_forward(tiny):
    tokens = [3, 14, 15, 9, 26, 5, 35, 8]
    full = tiny.forward(tokens)
    cache = tiny.new_cache()
    parts = [tiny.forward(tokens[:3], cache)] + [tiny.forward([t], cache) for t in tokens[3:]]
    np.testing.assert_allclose(np.concatenate(parts), full, rtol=1e-4, atol=1e-5)
    assert cache.length == len(tokens)


def test_tiny_context_limit(tiny):
    cache = tiny.new_cache()
    tiny.forward(list(range(32)), cache)
    with pytest.raises(ValueError):
        tiny.forward([1], cache)


def test_grouped_attention_matches_repeat_kv(tiny):
    # La GQA senza copie deve dare lo stesso risultato della versione esplicita con repeat_kv.
    c, layer = tiny.cfg, tiny.p["layers"][0]
    t = 6
    x = np.random.default_rng(9).standard_normal((t, c.dim)).astype(np.float32)
    got = tiny.attention(x, layer, 0, 0, None, ops.causal_mask(t, t))

    q = ops.split_heads(ops.linear(x, layer["wq"], layer["bq"]), c.n_head)
    k = ops.split_heads(ops.linear(x, layer["wk"], layer["bk"]), c.n_kv_head)
    v = ops.split_heads(ops.linear(x, layer["wv"], layer["bv"]), c.n_kv_head)
    rope = ops.apply_rope_interleaved if c.rope_interleaved else ops.apply_rope
    q, k = rope(q, tiny.cos[:t], tiny.sin[:t]), rope(k, tiny.cos[:t], tiny.sin[:t])
    n_rep = c.n_head // c.n_kv_head
    out = ops.scaled_dot_product_attention(q, ops.repeat_kv(k, n_rep), ops.repeat_kv(v, n_rep),
                                           ops.causal_mask(t, t))
    expected = ops.linear(ops.merge_heads(out), layer["wo"])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_tiny_generate_stops(tiny):
    full = tiny.generate([1, 2, 3], 10)
    assert len(full) == 10 and tiny.generate([1, 2, 3], 0) == []
    # Si ferma sul primo token che appartiene all'insieme degli EOS (incluso nel risultato).
    stop = full[3]
    first = full.index(stop)
    assert tiny.generate([1, 2, 3], 10, eos_ids={999, stop}) == full[:first + 1]
    assert tiny.generate([1, 2, 3], 10, eos_ids=stop) == full[:first + 1]
    # Contesto pieno: max_seq=32, prompt di 30 token -> al massimo 3 token generati.
    assert len(tiny.generate(list(range(30)), 10)) == 3


@pytest.mark.parametrize("tokens", [[], [-1], [64], [1, 2, 64]])
def test_tiny_rejects_invalid_tokens(tiny, tokens):
    with pytest.raises(ValueError):
        tiny.forward(tokens)


def test_silu_has_no_overflow_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = ops.silu(np.array([-100.0, 0.0, 100.0], np.float32))
    np.testing.assert_allclose(y, [0.0, 0.0, 100.0], atol=1e-30)


# ---------------------------------------------------------------- confronto con Hugging Face

PROMPT = "In 1905 Albert Einstein published four papers that changed physics. The first one"


@pytest.mark.parametrize("name", MODELS)
def test_matches_huggingface(name):
    """Logits e generazione greedy uguali a transformers (fp32). Richiede scripts/download_model.py."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    path = ROOT / "models" / name
    if not (path / "config.json").exists():
        pytest.skip(f"modello {name} non scaricato")
    from sardyne.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.from_hf_dir(path)
    ids = tok.encode(PROMPT)
    ours = Llama.from_dir(path, max_seq=128)
    logits = ours.forward(ids)
    gen = ours.generate(ids, 30, eos_ids=tok.eos_ids)
    del ours
    gc.collect()

    hf = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor([ids])).logits[0].numpy()
        hf_gen = hf.generate(torch.tensor([ids]), max_new_tokens=30, do_sample=False)[0, len(ids):].tolist()
    del hf
    gc.collect()

    # Logits fino a ~30: differenze ~1e-4 sono dovute solo all'ordine delle somme in fp32.
    np.testing.assert_allclose(logits, hf_logits, atol=1e-3, rtol=0)
    assert gen == hf_gen


@pytest.mark.parametrize("name", MODELS)
def test_kv_cache_on_real_model(name):
    path = ROOT / "models" / name
    if not (path / "config.json").exists():
        pytest.skip(f"modello {name} non scaricato")
    model = Llama.from_dir(path, max_seq=64)
    tokens = list(range(100, 112))
    full = model.forward(tokens)
    cache = model.new_cache()
    parts = [model.forward(tokens[:5], cache)] + [model.forward([t], cache) for t in tokens[5:]]
    np.testing.assert_allclose(np.concatenate(parts), full, atol=1e-3, rtol=0)
