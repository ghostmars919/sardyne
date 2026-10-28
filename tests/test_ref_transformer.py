import numpy as np
import pytest

from engine.ref import ops
from engine.ref.transformer import Config, KVCache, Transformer, init_params


@pytest.fixture(scope="module")
def model():
    cfg = Config(vocab_size=50, d_model=32, n_layer=2, n_head=4, d_ff=64, max_len=32)
    return Transformer(cfg, init_params(cfg, seed=1))


# ---------------------------------------------------------------- ops

def test_softmax_is_stable_and_normalized():
    x = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]], np.float32)
    y = ops.softmax(x)
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y.sum(-1), 1.0, rtol=1e-6)
    # Lo spostamento di una costante non cambia il risultato.
    np.testing.assert_allclose(y[0], ops.softmax(x[0] - 1000.0), rtol=1e-6)


def test_layer_norm_zero_mean_unit_var():
    x = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32) * 7 + 3
    y = ops.layer_norm(x, np.ones(16, np.float32), np.zeros(16, np.float32))
    np.testing.assert_allclose(y.mean(-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(y.var(-1), 1.0, rtol=1e-3)


def test_positional_encoding_formula():
    pe = ops.sinusoidal_positional_encoding(10, 8)
    pos, i = 7, 2
    np.testing.assert_allclose(pe[pos, 2 * i], np.sin(pos / 10000 ** (2 * i / 8)), rtol=1e-6)
    np.testing.assert_allclose(pe[pos, 2 * i + 1], np.cos(pos / 10000 ** (2 * i / 8)), rtol=1e-6)


def test_positional_encoding_shift_is_linear():
    # Sez. 3.5: PE[pos+k] è una funzione lineare di PE[pos]. Ogni coppia (sin, cos)
    # ruota di un angolo che dipende solo da k, non da pos.
    d, k = 16, 5
    pe = ops.sinusoidal_positional_encoding(64, d).astype(np.float64)
    freqs = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    for pos in (0, 3, 40):
        for j, w in enumerate(freqs):
            s, c = pe[pos, 2 * j], pe[pos, 2 * j + 1]
            rot = np.array([[np.cos(k * w), np.sin(k * w)], [-np.sin(k * w), np.cos(k * w)]])
            np.testing.assert_allclose(rot @ [s, c], pe[pos + k, 2 * j:2 * j + 2], atol=1e-6)


def test_attention_matches_naive_loops():
    rng = np.random.default_rng(0)
    t, dk, dv = 6, 8, 5
    q, k = rng.standard_normal((2, t, dk))
    v = rng.standard_normal((t, dv))
    out = ops.scaled_dot_product_attention(q, k, v, ops.causal_mask(t, t))
    for i in range(t):
        # La query i vede solo le chiavi 0..i.
        scores = np.array([q[i] @ k[j] / np.sqrt(dk) for j in range(i + 1)])
        w = np.exp(scores - scores.max())
        w /= w.sum()
        np.testing.assert_allclose(out[i], sum(w[j] * v[j] for j in range(i + 1)), rtol=1e-6)


def test_causal_mask_with_offset():
    # 2 query nuove su 5 chiavi totali: posizioni assolute 3 e 4.
    m = ops.causal_mask(2, 5)
    assert np.isinf(m[0, 4]) and m[0, 3] == 0
    assert np.all(m[1] == 0)


def test_multi_head_equals_separate_heads(model):
    # Sez. 3.2.2: le teste sono attenzioni indipendenti sulle proprie fette di W^Q, W^K, W^V.
    cfg, layer = model.cfg, model.p["layers"][0]
    x = np.random.default_rng(2).standard_normal((7, cfg.d_model)).astype(np.float32)
    dh, mask = cfg.d_head, ops.causal_mask(7, 7)
    heads = []
    for i in range(cfg.n_head):
        sl = slice(i * dh, (i + 1) * dh)
        heads.append(ops.scaled_dot_product_attention(
            x @ layer["wq"][:, sl], x @ layer["wk"][:, sl], x @ layer["wv"][:, sl], mask))
    expected = np.concatenate(heads, axis=-1) @ layer["wo"]
    np.testing.assert_allclose(model.attention(x, layer, 0, None), expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- modello

def test_forward_shape_and_finite(model):
    logits = model.forward([1, 2, 3, 4])
    assert logits.shape == (4, model.cfg.vocab_size)
    assert np.all(np.isfinite(logits))


def test_causality(model):
    # Cambiare il token in posizione 5 non deve influenzare i logits delle posizioni 0..4.
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    b = a.copy()
    b[5] = 7
    la, lb = model.forward(a), model.forward(b)
    np.testing.assert_allclose(la[:5], lb[:5], rtol=1e-5, atol=1e-5)
    assert not np.allclose(la[5:], lb[5:])


def test_kv_cache_matches_full_forward(model):
    tokens = [7, 3, 9, 1, 0, 12, 5, 8, 2]
    full = model.forward(tokens)
    cache = KVCache(model.cfg)
    step = [model.forward(tokens[:4], cache)]  # prefill
    for t in tokens[4:]:  # decode token per token
        step.append(model.forward([t], cache))
    np.testing.assert_allclose(np.concatenate(step), full, rtol=1e-4, atol=1e-4)
    assert cache.length == len(tokens)


def test_generate_is_greedy_and_deterministic(model):
    out = model.generate([1, 2, 3], 5)
    assert len(out) == 8 and out == model.generate([1, 2, 3], 5)
    # Ogni token generato è l'argmax dei logits sull'intero prefisso.
    for i in range(3, 8):
        assert out[i] == int(np.argmax(model.forward(out[:i])[-1]))


def test_max_len_enforced(model):
    with pytest.raises(AssertionError):
        model.forward(list(range(model.cfg.max_len + 1)))
