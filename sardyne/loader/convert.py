"""Conversione di un modello HF (safetensors) in GGUF, con le stesse convenzioni di llama.cpp.

Uso: python -m sardyne.loader.convert models/SmolLM2-135M models/gguf/SmolLM2-135M-bf16.gguf --outtype bf16

Regole (come convert_hf_to_gguf.py di llama.cpp):
- i tensori 1D (norm, bias) restano sempre F32; le matrici prendono il tipo scelto;
- architettura "llama": righe di W_q e W_k (e dei loro bias, se ci sono) permutate, perché
  llama.cpp applica la RoPE alle coppie adiacenti (2i, 2i+1) mentre HF alle coppie (i, i + hd/2);
- architetture "qwen2" e "qwen3": nessuna permutazione, llama.cpp usa la RoPE "neox" come HF;
- con gli embedding condivisi, output.weight non viene scritto;
- head_dim esplicito -> attention.key_length / value_length (oltre a rope.dimension_count).

I tensori vengono letti e convertiti uno alla volta durante la scrittura: in memoria c'è un solo
tensore alla volta, non l'intero modello.

Se la cartella contiene tokenizer.json, nel file finiscono anche vocabolario, merges e chat template.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sardyne import quant
from sardyne.loader import hf_config
from sardyne.loader.gguf_writer import GGUFWriter, f32_to_bf16
from sardyne.loader.safetensors import SafeTensorsDir

HF_TO_GGUF_ARCH = {"LlamaForCausalLM": "llama", "Qwen2ForCausalLM": "qwen2", "Qwen3ForCausalLM": "qwen3"}
FILE_TYPE = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 7, "bf16": 32}  # general.file_type (enum llama_ftype)
QUANT_TYPES = {"q8_0": "Q8_0", "q4_0": "Q4_0"}
F16_MAX = float(np.finfo(np.float16).max)

# nome HF (dentro model.layers.N.) -> nome GGUF (dentro blk.N.)
LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
PERMUTED = ("attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias")


def permute_for_interleaved_rope(w: np.ndarray, n_head: int) -> np.ndarray:
    """Riordina le righe di ogni testa da [metà 1 | metà 2] a coppie alternate (come llama.cpp).

    Riga j della prima metà e riga j della seconda metà diventano le righe 2j e 2j+1.
    Vale sia per le matrici [n_head*hd, d_in] sia per i bias [n_head*hd].
    """
    out_dim = w.shape[0]
    return w.reshape(n_head, 2, out_dim // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_name(hf_name: str) -> str | None:
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    if hf_name.startswith("model.layers."):
        _, _, idx, rest = hf_name.split(".", 3)
        if rest in LAYER_NAMES:
            return f"blk.{idx}.{LAYER_NAMES[rest]}"
    return None


def target_type(shape, outtype: str) -> str:
    """Tipo ggml di un tensore. Vettori sempre F32; con q8_0/q4_0, le matrici con righe non
    multiple di 32 restano in F16 (come i ripieghi di llama-quantize)."""
    if len(shape) == 1 or outtype == "f32":
        return "F32"
    if outtype in QUANT_TYPES:
        return QUANT_TYPES[outtype] if shape[-1] % quant.QK[QUANT_TYPES[outtype]] == 0 else "F16"
    return {"f16": "F16", "bf16": "BF16"}[outtype]


def _encode(name: str, data: np.ndarray, outtype: str) -> tuple[str, np.ndarray]:
    """Tipo ggml e dati codificati per un tensore."""
    kind = target_type(data.shape, outtype)
    if kind == "F32":
        return "F32", data.astype(np.float32)
    if kind in quant.QK:
        return kind, np.frombuffer(quant.quantize(data, kind), dtype=np.uint8)
    if kind == "F16" and outtype != "f16":
        outtype = "f16"  # ripiego di una matrice non quantizzabile
    if outtype == "f16":
        big = float(np.abs(data).max()) if data.size else 0.0
        if big > F16_MAX:  # (i NaN non superano il confronto e restano NaN, senza diventare inf)
            raise ValueError(f"{name}: valore {big:g} oltre il massimo di f16 ({F16_MAX:g}); usare --outtype bf16")
        return "F16", data.astype(np.float16)
    return "BF16", f32_to_bf16(data)


def tokenizer_metadata(w: GGUFWriter, hf_dir, vocab_size: int):
    """Scrive il tokenizer con le stesse chiavi di llama.cpp."""
    from sardyne.tokenizer.bpe import BPETokenizer

    tok = BPETokenizer.from_hf_dir(hf_dir, vocab_size)
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", tok.pre)
    w.add("tokenizer.ggml.tokens", tok.tokens, "ARRAY", "STRING")
    w.add("tokenizer.ggml.token_type", tok.token_types, "ARRAY", "INT32")
    w.add("tokenizer.ggml.merges", tok.merges, "ARRAY", "STRING")
    if tok.chat_template:
        w.add("tokenizer.chat_template", tok.chat_template)


def convert(hf_dir, out_path, outtype: str = "bf16"):
    hf_dir = Path(hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    arch = HF_TO_GGUF_ARCH.get(cfg["architectures"][0])
    if arch is None:
        raise NotImplementedError(f"architettura {cfg['architectures'][0]} non supportata")
    if outtype not in FILE_TYPE:
        raise ValueError(f"outtype {outtype} non supportato")
    theta = hf_config.rope_theta(cfg)  # solleva NotImplementedError con RoPE scalata

    n_head = cfg["num_attention_heads"]
    n_kv_head = cfg.get("num_key_value_heads", n_head)
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_head
    tied = cfg.get("tie_word_embeddings", False)

    w = GGUFWriter(out_path)
    w.add("general.architecture", arch)
    w.add("general.name", hf_dir.name)
    w.add(f"{arch}.vocab_size", cfg["vocab_size"])
    w.add(f"{arch}.context_length", cfg.get("max_position_embeddings", 2048))
    w.add(f"{arch}.embedding_length", cfg["hidden_size"])
    w.add(f"{arch}.block_count", cfg["num_hidden_layers"])
    w.add(f"{arch}.feed_forward_length", cfg["intermediate_size"])
    w.add(f"{arch}.attention.head_count", n_head)
    w.add(f"{arch}.attention.head_count_kv", n_kv_head)
    if cfg.get("head_dim"):
        w.add(f"{arch}.attention.key_length", head_dim)
        w.add(f"{arch}.attention.value_length", head_dim)
    w.add(f"{arch}.rope.freq_base", theta)
    w.add(f"{arch}.rope.dimension_count", head_dim)
    w.add(f"{arch}.attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]))
    w.add("general.file_type", FILE_TYPE[outtype])
    if isinstance(cfg.get("bos_token_id"), int):
        w.add("tokenizer.ggml.bos_token_id", cfg["bos_token_id"])
    eos = hf_config.eos_ids(hf_dir)
    if eos:
        # Come llama.cpp: il primo è l'EOS "ufficiale"; gli altri (eot/eom) arriveranno con la Fase 7.
        w.add("tokenizer.ggml.eos_token_id", eos[0])

    if (hf_dir / "tokenizer.json").exists():
        tokenizer_metadata(w, hf_dir, cfg["vocab_size"])

    st = SafeTensorsDir(hf_dir)
    for hf_name in sorted(st.keys(), key=_layer_order):
        name = gguf_name(hf_name)
        if name is None or (name == "output.weight" and tied):
            continue
        shape = st.shape(hf_name)
        ggml_type = target_type(shape, outtype)

        def produce(hf_name=hf_name, name=name):
            data = st.get_f32(hf_name)
            if arch == "llama" and name.endswith(PERMUTED):
                data = permute_for_interleaved_rope(data, n_head if ".attn_q." in name else n_kv_head)
            return _encode(name, data, outtype)[1]

        w.add_lazy_tensor(name, tuple(reversed(shape)), ggml_type, produce)
    w.write()


def _layer_order(name: str):
    # Ordine: embedding, strati in ordine numerico, poi norm finale e output.
    if name.startswith("model.layers."):
        return (1, int(name.split(".")[2]), name)
    return (0 if "embed" in name else 2, 0, name)


def main():
    ap = argparse.ArgumentParser(description="Converte un modello HF (safetensors) in GGUF")
    ap.add_argument("hf_dir")
    ap.add_argument("out")
    ap.add_argument("--outtype", choices=sorted(FILE_TYPE), default="bf16")
    args = ap.parse_args()
    convert(args.hf_dir, args.out, args.outtype)
    print(f"scritto {args.out}")


if __name__ == "__main__":
    main()
