"""Mixture of Experts (architettura GGUF "qwen3moe"): reference NumPy, caricamento lazy e runtime C."""
from pathlib import Path

import numpy as np
import pytest

from sardyne import capi, quant
from sardyne.loader.gguf import GGUFFile, LazyTensor
from sardyne.loader.gguf_writer import GGUFWriter, f32_to_bf16
from sardyne.ref import ops
from sardyne.ref.llama import Llama
from sardyne.runtime import Context, Model, generate, set_simd

DIM, N_HEAD, N_KV, HEAD_DIM, FFN_EXP, VOCAB, N_LAYER = 32, 4, 2, 16, 64, 50, 2
TOKENS = [3, 14, 15, 9, 26, 5, 35, 8, 2, 7, 1, 40]
needs_c = pytest.mark.skipif(not capi.available(), reason="libsardyne non compilata: eseguire make")


def weights(n_expert, seed=0, same_experts=False, dim=DIM, head_dim=HEAD_DIM, ffn_exp=FFN_EXP):
    """Pesi casuali: dizionario nome GGUF -> array fp32 (esperti in 3D, come nel file)."""
    rng = np.random.default_rng(seed)

    def w(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    def stack(*shape):
        return np.broadcast_to(w(*shape), (n_expert, *shape)).copy() if same_experts else w(n_expert, *shape)

    q, kv = N_HEAD * head_dim, N_KV * head_dim
    t = {"token_embd.weight": w(VOCAB, dim), "output_norm.weight": 1 + w(dim), "output.weight": w(VOCAB, dim)}
    for i in range(N_LAYER):
        p = f"blk.{i}."
        t.update({p + "attn_norm.weight": 1 + w(dim), p + "ffn_norm.weight": 1 + w(dim),
                  p + "attn_q.weight": w(q, dim), p + "attn_k.weight": w(kv, dim), p + "attn_v.weight": w(kv, dim),
                  p + "attn_output.weight": w(dim, q), p + "attn_q_norm.weight": 1 + w(head_dim),
                  p + "attn_k_norm.weight": 1 + w(head_dim), p + "ffn_gate_inp.weight": 3 * w(n_expert, dim),
                  p + "ffn_gate_exps.weight": stack(ffn_exp, dim), p + "ffn_up_exps.weight": stack(ffn_exp, dim),
                  p + "ffn_down_exps.weight": stack(dim, ffn_exp)})
    return t


def write_gguf(path, tensors, arch="qwen3moe", n_expert=8, n_used=3, outtype="f32", dim=DIM, head_dim=HEAD_DIM,
               ffn_exp=FFN_EXP, random_kinds=None):
    """random_kinds: nome -> tipo quantizzato con blocchi casuali (non abbiamo quantizzatori K-quant)."""
    w = GGUFWriter(path)
    w.add("general.architecture", arch)
    w.add(f"{arch}.vocab_size", VOCAB)
    w.add(f"{arch}.context_length", 64)
    w.add(f"{arch}.embedding_length", dim)
    w.add(f"{arch}.block_count", N_LAYER)
    w.add(f"{arch}.feed_forward_length", ffn_exp * max(n_used, 1) if arch == "qwen3moe" else ffn_exp)
    w.add(f"{arch}.attention.head_count", N_HEAD)
    w.add(f"{arch}.attention.head_count_kv", N_KV)
    w.add(f"{arch}.attention.key_length", head_dim)
    w.add(f"{arch}.attention.value_length", head_dim)
    w.add(f"{arch}.rope.freq_base", 10000.0)
    w.add(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    if arch == "qwen3moe":
        if n_expert:
            w.add(f"{arch}.expert_count", n_expert)
        w.add(f"{arch}.expert_used_count", n_used)
        w.add(f"{arch}.expert_feed_forward_length", ffn_exp)
    for i, (name, data) in enumerate(tensors.items()):
        ne = tuple(reversed(data.shape))
        kind = random_kinds(name) if random_kinds else None
        if kind:
            w.add_raw_tensor(name, ne, kind, random_blocks(kind, data.size // quant.QK[kind], seed=i).tobytes())
        elif data.ndim == 1 or outtype == "f32":
            w.add_tensor(name, data, "F32")
        elif outtype == "bf16":
            w.add_tensor(name, f32_to_bf16(data), "BF16")
        else:
            kind = outtype.upper()
            w.add_raw_tensor(name, ne, kind, quant.quantize(data.reshape(-1, data.shape[-1]), kind))
    w.write()
    return path


def random_blocks(kind, n_blocks, seed):
    from test_quant import random_blocks as blocks
    return blocks(kind, n_blocks, seed)


def naive_moe(h, layer, k):
    """La formula scritta token per token ed esperto per esperto, in float64."""
    h = h.astype(np.float64)
    out = np.zeros_like(h)
    for t in range(len(h)):
        logits = np.asarray(layer["gate_inp"], np.float64) @ h[t]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top = np.argsort(-p)[:k]
        for e in top:
            gate = np.asarray(layer["exp_gate"][e], np.float64) @ h[t]
            up = np.asarray(layer["exp_up"][e], np.float64) @ h[t]
            y = np.asarray(layer["exp_down"][e], np.float64) @ (gate / (1 + np.exp(-gate)) * up)
            out[t] += p[e] / p[top].sum() * y
    return out


@pytest.fixture(scope="module")
def tensors():
    return weights(8)


@pytest.fixture(scope="module")
def moe_path(tmp_path_factory, tensors):
    return write_gguf(tmp_path_factory.mktemp("moe") / "moe.gguf", tensors)


def test_config_from_gguf(moe_path):
    c = Llama.from_gguf(moe_path, max_seq=32).cfg
    assert (c.n_expert, c.n_expert_used, c.ffn_exp_dim) == (8, 3, FFN_EXP)
    assert c.qk_norm and not c.rope_interleaved


def test_ffn_exp_dim_defaults_like_llama_cpp(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(path, weights(8))
    with GGUFFile(path) as gf:
        src = dict(gf.metadata)
    w = GGUFWriter(tmp_path / "n.gguf")
    for k, v in src.items():
        if k != "qwen3moe.expert_feed_forward_length":
            w.add(k, v)
    for name, data in weights(8).items():
        w.add_tensor(name, data, "F32")
    w.write()
    assert Llama.from_gguf(tmp_path / "n.gguf", max_seq=32).cfg.ffn_exp_dim == FFN_EXP


def test_router_formula(moe_path):
    model = Llama.from_gguf(moe_path, max_seq=32)
    h = np.random.default_rng(1).standard_normal((12, DIM)).astype(np.float32)
    for layer in model.p["layers"]:
        np.testing.assert_allclose(model.moe(h, layer), naive_moe(h, layer, 3), atol=2e-5, rtol=0)


def test_tokens_choose_different_experts(moe_path):
    # Altrimenti gli altri test non proverebbero il raggruppamento per esperto.
    model = Llama.from_gguf(moe_path, max_seq=32)
    h = np.random.default_rng(1).standard_normal((12, DIM)).astype(np.float32)
    probs = ops.softmax(h @ model.p["layers"][0]["gate_inp"].T)
    assert len(set(probs.argmax(-1).tolist())) > 3


@pytest.mark.parametrize("n_expert,n_used", [(1, 1), (5, 5), (5, 2)], ids=["one", "all", "topk"])
def test_identical_experts_equal_dense_qwen3(tmp_path, n_expert, n_used):
    """Con esperti tutti uguali i pesi sommano a 1: il risultato è quello dell'MLP denso."""
    t = weights(n_expert, same_experts=True)
    dense = {k: v for k, v in t.items() if "_exps" not in k and "gate_inp" not in k}
    for i in range(N_LAYER):
        for gname in ("gate", "up", "down"):
            dense[f"blk.{i}.ffn_{gname}.weight"] = t[f"blk.{i}.ffn_{gname}_exps.weight"][0]
    moe = Llama.from_gguf(write_gguf(tmp_path / "moe.gguf", t, n_expert=n_expert, n_used=n_used), max_seq=32)
    ref = Llama.from_gguf(write_gguf(tmp_path / "d.gguf", dense, arch="qwen3", n_used=n_used), max_seq=32)
    np.testing.assert_allclose(moe.forward(TOKENS), ref.forward(TOKENS), atol=2e-4, rtol=0)


def test_cache_matches_full_forward(moe_path):
    model = Llama.from_gguf(moe_path, max_seq=32)
    full = model.forward(TOKENS)
    cache = model.new_cache()
    steps = [model.forward(TOKENS[:5], cache)] + [model.forward([t], cache) for t in TOKENS[5:]]
    np.testing.assert_allclose(np.concatenate(steps), full, atol=1e-4, rtol=0)


@pytest.mark.parametrize("outtype", ["f32", "bf16", "q8_0", "q4_0"])
@pytest.mark.parametrize("emulate", [False, True], ids=["pesi", "attivazioni"])
def test_lazy_equals_eager(tmp_path, tensors, outtype, emulate):
    path = write_gguf(tmp_path / "m.gguf", tensors, outtype=outtype)
    lazy = Llama.from_gguf(path, max_seq=32, emulate_quant=emulate, lazy=True)
    assert isinstance(lazy.p["layers"][0]["exp_gate"], LazyTensor)
    eager = Llama.from_gguf(path, max_seq=32, emulate_quant=emulate)
    assert np.array_equal(lazy.forward(TOKENS), eager.forward(TOKENS))
    assert lazy.generate(TOKENS, 8) == eager.generate(TOKENS, 8)


def test_emulated_activations_reach_the_experts(tmp_path, tensors, monkeypatch):
    """Una quantizzazione per ogni matmul: lm_head, 4 dell'attention e il router per strato, gate, up
    e down per ogni esperto scelto."""
    path = write_gguf(tmp_path / "m.gguf", tensors, outtype="q8_0")
    model = Llama.from_gguf(path, max_seq=32, emulate_quant=True, lazy=True)
    fake_quant, get_f32 = quant.fake_quant, GGUFFile.get_f32
    n_quant, n_experts = [], []

    def count_quant(x, kind):
        n_quant.append(kind)
        return fake_quant(x, kind)

    def count_experts(self, name, idx=None):
        if "ffn_down_exps" in name:
            n_experts.append(idx)
        return get_f32(self, name, idx)

    monkeypatch.setattr(quant, "fake_quant", count_quant)
    monkeypatch.setattr(GGUFFile, "get_f32", count_experts)
    model.forward(TOKENS)
    assert len(n_experts) >= N_LAYER
    assert n_quant == ["Q8_0"] * (1 + 5 * N_LAYER + 3 * len(n_experts))


def test_unused_experts_are_never_decoded(tmp_path, tensors, moe_path):
    """I pesi degli esperti non scelti diventano NaN: se venissero usati, si vedrebbe nel risultato."""
    model = Llama.from_gguf(moe_path, max_seq=32)
    used = [set() for _ in range(N_LAYER)]
    real_moe = model.moe

    def spy(h, layer):
        top = np.argsort(-ops.softmax(h @ layer["gate_inp"].T), axis=-1, kind="stable")[:, :3]
        used[next(i for i, l in enumerate(model.p["layers"]) if l is layer)].update(top.ravel().tolist())
        return real_moe(h, layer)

    model.moe = spy
    ref = model.forward(TOKENS)

    poisoned = dict(tensors)
    for i, chosen in enumerate(used):
        unused = sorted(set(range(8)) - chosen)
        assert unused
        for gname in ("gate", "up", "down"):
            name = f"blk.{i}.ffn_{gname}_exps.weight"
            poisoned[name] = tensors[name].copy()
            poisoned[name][unused] = np.nan
    lazy = Llama.from_gguf(write_gguf(tmp_path / "p.gguf", poisoned), max_seq=32, lazy=True)
    assert np.array_equal(lazy.forward(TOKENS), ref)


@pytest.mark.parametrize("outtype", ["f32", "bf16", "q8_0", "q4_0"])
def test_get_f32_slices(tmp_path, tensors, outtype):
    path = write_gguf(tmp_path / "m.gguf", tensors, outtype=outtype)
    with GGUFFile(path) as gf:
        for name in ("blk.0.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "token_embd.weight"):
            full = gf.get_f32(name)
            assert np.array_equal(gf.get_f32(name, 3), full[3])
            idx = np.array([5, 1, 1])
            assert np.array_equal(gf.get_f32(name, idx), full[idx])


def test_lazy_tensor_behaves_like_array(moe_path):
    with GGUFFile(moe_path) as gf:
        lazy = LazyTensor(gf, "blk.0.ffn_gate_exps.weight")
        full = gf.get_f32(lazy.name)
        assert lazy.shape == full.shape and lazy.size == full.size
        assert np.array_equal(np.asarray(lazy), full)


@pytest.mark.parametrize("n_expert,n_used,msg", [(0, 3, "expert_count"), (8, 0, "n_expert_used"),
                                                 (8, 9, "n_expert_used")])
def test_bad_expert_counts(tmp_path, tensors, n_expert, n_used, msg):
    path = write_gguf(tmp_path / "m.gguf", tensors, n_expert=n_expert, n_used=n_used)
    with pytest.raises(ValueError, match=msg):
        Llama.from_gguf(path, max_seq=32)


def test_wrong_expert_shape_is_rejected(tmp_path, tensors):
    bad = dict(tensors)
    bad["blk.1.ffn_up_exps.weight"] = tensors["blk.1.ffn_up_exps.weight"][:-1]
    with pytest.raises(ValueError, match="exp_up"):
        Llama.from_gguf(write_gguf(tmp_path / "m.gguf", bad), max_seq=32)


# ---------------------------------------------------------------- runtime C

def real_file_kinds(name):
    """I tipi di un Q4_K_M reale: K-quant dove le righe sono multiple di 256, Q5_0 in ffn_down."""
    if "gate_inp" in name or name.endswith("norm.weight") or "token_embd" in name or name == "output.weight":
        return None
    if "ffn_down" in name:
        return "Q5_0"
    return "Q6_K" if "ffn_up" in name or "attn_output" in name else "Q4_K"


@pytest.fixture(scope="module", params=[("f32", 8, 3), ("bf16", 8, 3), ("q8_0", 8, 3), ("q4_0", 8, 3), ("q8_0", 5, 5),
                                        ("f32", 1, 1), ("q8_0", 128, 8), ("kquant", 8, 3)],
                ids=lambda p: "-".join(map(str, p)))
def c_model(request, tmp_path_factory):
    outtype, n_expert, n_used = request.param
    path = tmp_path_factory.mktemp("c") / "m.gguf"
    if outtype == "kquant":
        sizes = dict(dim=256, head_dim=64, ffn_exp=160)
        write_gguf(path, weights(n_expert, **sizes), n_expert=n_expert, n_used=n_used, random_kinds=real_file_kinds, **sizes)
    else:
        write_gguf(path, weights(n_expert), n_expert=n_expert, n_used=n_used, outtype=outtype)
    return path


@pytest.fixture(params=[False, True], ids=["scalare", "avx2"])
def kernels(request):
    if request.param and not set_simd(True):
        pytest.skip("CPU senza AVX2")
    set_simd(request.param)
    yield request.param
    set_simd(True)


def assert_close(c, ref):
    # I K-quant casuali danno logits grandi: la tolleranza segue la scala.
    np.testing.assert_allclose(c, ref, atol=1e-4 * max(1.0, float(np.abs(ref).max())), rtol=0)


@needs_c
def test_c_matches_numpy(c_model, kernels):
    ref = Llama.from_gguf(c_model, max_seq=32, emulate_quant=True).forward(TOKENS)
    ctx = Context(Model(c_model), max_seq=32, n_batch=len(TOKENS), logits_all=True)
    assert_close(ctx.eval(TOKENS), ref)


@needs_c
def test_c_info_matches_numpy_config(c_model):
    cfg = Llama.from_gguf(c_model, max_seq=8).cfg
    i = Model(c_model).info
    assert (i.n_expert, i.n_expert_used, i.ffn_exp_dim, bool(i.qk_norm)) == \
           (cfg.n_expert, cfg.n_expert_used, cfg.ffn_exp_dim, cfg.qk_norm)


@needs_c
def test_c_generate_matches_numpy(c_model):
    ref = Llama.from_gguf(c_model, max_seq=32, emulate_quant=True)
    assert generate(Context(Model(c_model), max_seq=32), TOKENS[:4], 12) == ref.generate(TOKENS[:4], 12)


@needs_c
def test_thread_count_does_not_change_results(c_model):
    model = Model(c_model)
    outs = []
    for n_threads in (1, 2, 3, 4, 7):
        ctx = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True, n_threads=n_threads)
        logits = ctx.eval(TOKENS[:8])
        outs.append(np.concatenate([logits, ctx.eval(TOKENS[8:])]))  # prefill + seguito
    for o in outs[1:]:
        assert np.array_equal(o, outs[0])  # bit per bit


@needs_c
def test_many_threads_stress(c_model):
    model = Model(c_model)
    ref, ctx = Context(model, max_seq=32, n_threads=1), Context(model, max_seq=32, n_threads=8)
    for _ in range(10):
        ref.reset()
        ctx.reset()
        a, b = ref.eval(TOKENS[:6]), ctx.eval(TOKENS[:6])
        for t in TOKENS[6:]:
            a, b = ref.eval([t]), ctx.eval([t])
        assert np.array_equal(a, b)


@needs_c
def test_batching_and_cache(c_model):
    model = Model(c_model)
    full = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    chunked = Context(model, max_seq=32, n_batch=5, logits_all=True).eval(TOKENS)
    assert_close(chunked, full)
    ctx = Context(model, max_seq=32, n_batch=4)
    last = ctx.eval(TOKENS[:7])
    assert_close(last, full[6])
    for i, t in enumerate(TOKENS[7:], start=7):
        assert_close(ctx.eval([t]), full[i])


@needs_c
def test_prefetch_does_not_change_results(c_model, monkeypatch):
    model = Model(c_model)
    with_prefetch = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    monkeypatch.setenv("ENGINE_NO_PREFETCH", "1")
    without = Context(model, max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    assert np.array_equal(with_prefetch, without)


@needs_c
def test_c_uses_only_chosen_experts(tmp_path, tensors):
    """Stessa prova della reference: i pesi degli esperti non scelti diventano NaN."""
    ref = Llama.from_gguf(write_gguf(tmp_path / "a.gguf", tensors), max_seq=32)
    used = [set() for _ in range(N_LAYER)]
    real_moe = ref.moe

    def spy(h, layer):
        top = np.argsort(-ops.softmax(h @ layer["gate_inp"].T), axis=-1, kind="stable")[:, :3]
        used[next(i for i, l in enumerate(ref.p["layers"]) if l is layer)].update(top.ravel().tolist())
        return real_moe(h, layer)

    ref.moe = spy
    ref.forward(TOKENS)
    poisoned = dict(tensors)
    for i, chosen in enumerate(used):
        unused = sorted(set(range(8)) - chosen)
        assert unused
        for gname in ("gate", "up", "down"):
            name = f"blk.{i}.ffn_{gname}_exps.weight"
            poisoned[name] = tensors[name].copy()
            poisoned[name][unused] = np.nan
    clean = Context(Model(tmp_path / "a.gguf"), max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS)
    dirty = Context(Model(write_gguf(tmp_path / "b.gguf", poisoned)), max_seq=32, n_batch=len(TOKENS),
                    logits_all=True).eval(TOKENS)
    assert np.array_equal(clean, dirty)


@needs_c
def test_ties_choose_the_lowest_experts(tmp_path, tensors):
    """Router a zero: probabilità tutte uguali, si scelgono i primi k esperti, come nella reference."""
    t = dict(tensors)
    for i in range(N_LAYER):
        t[f"blk.{i}.ffn_gate_inp.weight"] = np.zeros_like(tensors[f"blk.{i}.ffn_gate_inp.weight"])
    path = write_gguf(tmp_path / "m.gguf", t)
    ref = Llama.from_gguf(path, max_seq=32).forward(TOKENS)
    assert_close(Context(Model(path), max_seq=32, n_batch=len(TOKENS), logits_all=True).eval(TOKENS), ref)


@needs_c
@pytest.mark.parametrize("n_expert,n_used,msg", [(0, 3, "expert_count"), (8, 0, "expert_used_count"),
                                                 (8, 9, "expert_used_count")])
def test_c_rejects_bad_expert_counts(tmp_path, tensors, n_expert, n_used, msg):
    path = write_gguf(tmp_path / "m.gguf", tensors, n_expert=n_expert, n_used=n_used)
    with pytest.raises(ValueError, match=msg):
        Model(path)


@needs_c
@pytest.mark.parametrize("name,cut", [("blk.1.ffn_up_exps.weight", lambda a: a[:-1]),
                                      ("blk.0.ffn_down_exps.weight", lambda a: a[0]),
                                      ("blk.0.ffn_gate_inp.weight", lambda a: a[:-1])],
                         ids=["meno-esperti", "2d", "router"])
def test_c_rejects_wrong_shapes(tmp_path, tensors, name, cut):
    bad = dict(tensors)
    bad[name] = cut(tensors[name])
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        Model(write_gguf(tmp_path / "m.gguf", bad))


@needs_c
def test_c_rejects_unknown_expert_tensor(tmp_path, tensors):
    bad = dict(tensors)
    bad["blk.0.ffn_gate_shexp.weight"] = np.ones((FFN_EXP, DIM), np.float32)
    with pytest.raises(ValueError, match="ffn_gate_shexp"):
        Model(write_gguf(tmp_path / "m.gguf", bad))


@needs_c
def test_real_quantized_moe():
    """Qwen3-30B-A3B Q4_K_M: reference lazy contro C, come per gli altri file reali. Serve qualche minuto."""
    path = Path(__file__).resolve().parent.parent / "models" / "gguf" / "Qwen3-30B-A3B-Q4_K_M.gguf"
    if not path.exists():
        pytest.skip("Qwen3-30B-A3B-Q4_K_M.gguf non scaricato")
    ids = [785, 6722, 315, 9625, 374, 12095, 13, 1084, 374, 279]  # "The capital of France is Paris. It is the"
    model = Model(path)
    logits = Context(model, max_seq=16, n_batch=len(ids), logits_all=True).eval(ids)
    ref = Llama.from_gguf(path, max_seq=16, emulate_quant=True, lazy=True).forward(ids)
    assert np.mean(logits.argmax(-1) == ref.argmax(-1)) >= 0.8
    assert np.abs(logits - ref).mean() < 0.5
    one = Context(model, max_seq=16, n_batch=len(ids), logits_all=True, n_threads=1).eval(ids)
    assert np.array_equal(one, logits)  # 1 thread o tutti: identici bit per bit
