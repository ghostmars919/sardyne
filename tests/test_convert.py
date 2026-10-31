"""Convertitore HF -> GGUF su modelli sintetici minuscoli: non servono modelli scaricati."""
import json

import numpy as np
import pytest

from sardyne.loader import hf_config
from sardyne.loader.convert import convert
from sardyne.loader.gguf import GGUFFile
from sardyne.loader.gguf_writer import GGUFWriter
from sardyne.ref.llama import Llama, LlamaConfig

safetensors_np = pytest.importorskip("safetensors.numpy")


def make_hf(path, arch="LlamaForCausalLM", bias=False, tied=True, head_dim=None, rope="old",
            theta=5000.0, eos=(5, 7)):
    """Un modello HF finto ma completo: config.json, generation_config.json, model.safetensors."""
    path.mkdir(parents=True, exist_ok=True)
    dim, n_head, n_kv, ffn, vocab, n_layer = 32, 4, 2, 48, 64, 2
    hd = head_dim or dim // n_head
    cfg = {"architectures": [arch], "hidden_size": dim, "num_attention_heads": n_head,
           "num_key_value_heads": n_kv, "intermediate_size": ffn, "vocab_size": vocab,
           "num_hidden_layers": n_layer, "rms_norm_eps": 1e-5, "max_position_embeddings": 64,
           "tie_word_embeddings": tied, "bos_token_id": 1}
    if arch == "LlamaForCausalLM":
        cfg["attention_bias"] = bias
    if head_dim:
        cfg["head_dim"] = head_dim
    if rope == "old":
        cfg["rope_theta"] = theta
    elif rope == "v5":  # forma di transformers 5
        cfg["rope_parameters"] = {"rope_theta": theta, "rope_type": "default"}
    elif rope == "llama3":
        cfg["rope_theta"] = theta
        cfg["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0}
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": list(eos)}))

    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    t = {"model.embed_tokens.weight": w(vocab, dim), "model.norm.weight": 1 + w(dim)}
    if not tied:
        t["lm_head.weight"] = w(vocab, dim)
    has_bias = bias or arch == "Qwen2ForCausalLM"
    for i in range(n_layer):
        p = f"model.layers.{i}."
        t.update({p + "input_layernorm.weight": 1 + w(dim), p + "post_attention_layernorm.weight": 1 + w(dim),
                  p + "self_attn.q_proj.weight": w(n_head * hd, dim), p + "self_attn.k_proj.weight": w(n_kv * hd, dim),
                  p + "self_attn.v_proj.weight": w(n_kv * hd, dim), p + "self_attn.o_proj.weight": w(dim, n_head * hd),
                  p + "mlp.gate_proj.weight": w(ffn, dim), p + "mlp.up_proj.weight": w(ffn, dim),
                  p + "mlp.down_proj.weight": w(dim, ffn)})
        if has_bias:
            # Bias grandi: se non fossero permutati come i pesi, l'errore sarebbe evidente.
            for n, d in (("q", n_head * hd), ("k", n_kv * hd), ("v", n_kv * hd)):
                t[p + f"self_attn.{n}_proj.bias"] = 2 * rng.standard_normal(d).astype(np.float32)
        if arch == "Qwen3ForCausalLM":
            t.update({p + "self_attn.q_norm.weight": 1 + w(hd), p + "self_attn.k_norm.weight": 1 + w(hd)})
    safetensors_np.save_file(t, str(path / "model.safetensors"))
    return path


TOKENS = [3, 14, 15, 9, 26, 5, 35, 8, 2, 7]


@pytest.mark.parametrize("variant", [
    dict(arch="LlamaForCausalLM"),
    dict(arch="LlamaForCausalLM", bias=True, tied=False, head_dim=16),  # bias permutati, head_dim esplicito
    dict(arch="Qwen2ForCausalLM", tied=False),
    dict(arch="LlamaForCausalLM", rope="v5"),
], ids=["llama", "llama-bias-headdim", "qwen2", "rope_parameters"])
def test_roundtrip_matches_safetensors(tmp_path, variant):
    hf = make_hf(tmp_path / "hf", **variant)
    out = tmp_path / "m.gguf"
    convert(hf, out, "f32")
    a, b = Llama.from_dir(hf, max_seq=32), Llama.from_gguf(out, max_seq=32)
    assert b.cfg.head_dim == a.cfg.head_dim and b.cfg.rope_theta == a.cfg.rope_theta == 5000.0
    assert b.cfg.rope_interleaved == (variant["arch"] == "LlamaForCausalLM")
    np.testing.assert_allclose(b.forward(TOKENS), a.forward(TOKENS), atol=1e-5, rtol=0)

    # E anche con la KV cache (RoPE interleaved inclusa).
    cache = b.new_cache()
    parts = [b.forward(TOKENS[:4], cache)] + [b.forward([t], cache) for t in TOKENS[4:]]
    np.testing.assert_allclose(np.concatenate(parts), a.forward(TOKENS), atol=1e-5, rtol=0)


def test_metadata(tmp_path):
    hf = make_hf(tmp_path / "hf", head_dim=16, eos=(5, 7))
    out = tmp_path / "m.gguf"
    convert(hf, out, "bf16")
    f = GGUFFile(out)
    m = f.metadata
    assert m["llama.attention.key_length"] == m["llama.attention.value_length"] == 16
    assert m["llama.rope.dimension_count"] == 16 and m["llama.rope.freq_base"] == 5000.0
    assert m["tokenizer.ggml.eos_token_id"] == 5 and m["tokenizer.ggml.bos_token_id"] == 1
    assert "output.weight" not in f.tensors  # embedding condivisi
    assert f.tensors["blk.0.attn_q.weight"].type_name == "BF16"
    assert f.tensors["blk.0.attn_norm.weight"].type_name == "F32"


def test_bf16_conversion_is_close(tmp_path):
    hf = make_hf(tmp_path / "hf")
    out = tmp_path / "m.gguf"
    convert(hf, out, "bf16")
    a, b = Llama.from_dir(hf, max_seq=32), Llama.from_gguf(out, max_seq=32)
    assert np.abs(b.forward(TOKENS) - a.forward(TOKENS)).max() < 0.05


def test_rope_scaling_is_rejected(tmp_path):
    hf = make_hf(tmp_path / "hf", rope="llama3")
    with pytest.raises(NotImplementedError):
        Llama.from_dir(hf)
    with pytest.raises(NotImplementedError):
        convert(hf, tmp_path / "m.gguf")
    with pytest.raises(NotImplementedError):
        hf_config.rope_theta({"rope_parameters": {"rope_theta": 1e4, "rope_type": "yarn"}})
    assert hf_config.rope_theta({"rope_parameters": {"full_attention": {"rope_theta": 3.0}}}) == 3.0
    assert hf_config.rope_theta({}) == 10000.0


def test_rope_freqs_tensor_is_rejected(tmp_path):
    # Llama 3.1/3.2 in GGUF: la RoPE "llama3" è un tensore rope_freqs.weight, senza rope.scaling.type.
    hf = make_hf(tmp_path / "hf")
    out = tmp_path / "m.gguf"
    convert(hf, out, "f32")
    src = GGUFFile(out)
    w = GGUFWriter(tmp_path / "llama3.gguf")
    for k, v in src.metadata.items():
        vt, et = src.metadata_types[k]
        from sardyne.loader.gguf import VALUE_TYPES
        w.add(k, v, VALUE_TYPES[vt][0], VALUE_TYPES[et][0] if et is not None else None)
    for name in src.tensors:
        w.add_tensor(name, src.get_f32(name), "F32")
    w.add_tensor("rope_freqs.weight", np.ones(4, np.float32), "F32")
    w.write()
    with pytest.raises(NotImplementedError, match="rope_freqs"):
        Llama.from_gguf(w.path)


def test_f16_overflow_is_an_error(tmp_path):
    hf = make_hf(tmp_path / "hf")
    st = safetensors_np.load_file(str(hf / "model.safetensors"))
    st["model.layers.0.mlp.up_proj.weight"][0, 0] = 70000.0
    safetensors_np.save_file(st, str(hf / "model.safetensors"))
    out = tmp_path / "m.gguf"
    with pytest.raises(ValueError, match="f16"):
        convert(hf, out, "f16")
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
    convert(hf, out, "bf16")  # bf16 ha lo stesso esponente di f32: nessun problema


def test_eos_ids(tmp_path):
    hf = make_hf(tmp_path / "hf", eos=(5, 7))
    assert hf_config.eos_ids(hf) == [5, 7]
    (hf / "generation_config.json").unlink()
    assert hf_config.eos_ids(hf) == []
    cfg = json.loads((hf / "config.json").read_text())
    cfg["eos_token_id"] = 9
    (hf / "config.json").write_text(json.dumps(cfg))
    assert hf_config.eos_ids(hf) == [9]


def test_shape_mismatch_is_reported(tmp_path):
    hf = make_hf(tmp_path / "hf")
    cfg = json.loads((hf / "config.json").read_text())
    cfg["intermediate_size"] = 50  # non coerente con i pesi
    (hf / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="w_gate"):
        Llama.from_dir(hf)
    with pytest.raises(ValueError):
        LlamaConfig.from_hf({**cfg, "intermediate_size": 48, "num_key_value_heads": 3})  # 4 % 3 != 0
