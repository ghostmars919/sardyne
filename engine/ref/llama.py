"""Modello Llama-like in NumPy (fp32): la reference numerica del motore.

Rispetto al Transformer del paper (engine/ref/transformer.py):
- pre-norm con RMSNorm:   x = x + Attn(RMSNorm(x));  x = x + MLP(RMSNorm(x))
- RoPE applicata a q e k in ogni strato, al posto della PE sommata all'embedding
- GQA: n_kv_head <= n_head teste per chiavi e valori
- MLP SwiGLU senza bias
- RMSNorm finale prima della proiezione sul vocabolario

Architetture supportate (config.json di HF): LlamaForCausalLM (Llama 2/3, SmolLM2) e
Qwen2ForCausalLM (Qwen2/2.5: uguale a Llama più un bias su q, k, v).

Uso: python -m engine.ref.llama models/SmolLM2-135M "The capital of France is" -n 30
"""
import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.loader import hf_config
from engine.loader.gguf import GGUFFile
from engine.loader.safetensors import SafeTensorsDir
from engine.ref import ops

SUPPORTED = {"LlamaForCausalLM", "Qwen2ForCausalLM"}

# Architetture GGUF supportate -> RoPE a coppie adiacenti? Come in llama.cpp: "llama" usa la
# RoPE "normale" (pesi q/k permutati dal convertitore), "qwen2" la RoPE "neox" (pesi invariati).
GGUF_ARCHS = {"llama": True, "qwen2": False}

# Tensori che sappiamo usare. Un tensore diverso (es. rope_freqs.weight dei Llama 3.1) indica una
# variante dell'architettura non supportata: meglio un errore che un modello sbagliato in silenzio.
_GGUF_TENSOR = re.compile(r"(token_embd|output_norm|output)\.weight"
                          r"|blk\.\d+\.(attn_norm|attn_output|ffn_norm|ffn_gate|ffn_up|ffn_down)\.weight"
                          r"|blk\.\d+\.attn_[qkv]\.(weight|bias)")


@dataclass
class LlamaConfig:
    vocab_size: int
    dim: int  # d_model
    n_layer: int
    n_head: int
    n_kv_head: int
    head_dim: int
    ffn_dim: int
    norm_eps: float
    rope_theta: float
    tie_embeddings: bool
    qkv_bias: bool
    max_pos: int
    # False: RoPE sulle coppie (i, i+hd/2) come in HF ("neox"); True: coppie adiacenti (2i, 2i+1).
    rope_interleaved: bool = False

    def __post_init__(self):
        if self.n_head % self.n_kv_head:
            raise ValueError(f"n_head={self.n_head} non è multiplo di n_kv_head={self.n_kv_head}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} dispari: la RoPE ruota coppie di dimensioni")

    @classmethod
    def from_hf(cls, cfg: dict) -> "LlamaConfig":
        arch = cfg["architectures"][0]
        if arch not in SUPPORTED:
            raise NotImplementedError(f"architettura {arch} non supportata (ancora)")
        n_head = cfg["num_attention_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            dim=cfg["hidden_size"],
            n_layer=cfg["num_hidden_layers"],
            n_head=n_head,
            n_kv_head=cfg.get("num_key_value_heads", n_head),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // n_head,
            ffn_dim=cfg["intermediate_size"],
            norm_eps=cfg["rms_norm_eps"],
            rope_theta=hf_config.rope_theta(cfg),  # gestisce anche rope_parameters (transformers 5)
            tie_embeddings=cfg.get("tie_word_embeddings", False),
            qkv_bias=arch == "Qwen2ForCausalLM" or cfg.get("attention_bias", False),
            max_pos=cfg.get("max_position_embeddings", 2048),
        )

    @classmethod
    def from_gguf(cls, gf: GGUFFile) -> "LlamaConfig":
        arch = gf.arch
        if arch not in GGUF_ARCHS:
            raise NotImplementedError(f"architettura GGUF {arch} non supportata (ancora)")
        m = gf.metadata
        # La RoPE "llama3" non scrive rope.scaling.type ma il tensore rope_freqs.weight.
        if m.get(f"{arch}.rope.scaling.type", "none") != "none" or "rope_freqs.weight" in gf.tensors:
            raise NotImplementedError("rope scaling (anche 'llama3' via rope_freqs.weight) arriva in Fase 6")
        unknown = [n for n in gf.tensors if not _GGUF_TENSOR.fullmatch(n)]
        if unknown:
            raise NotImplementedError(f"tensori non supportati: {unknown[:5]}")

        dim = int(m[f"{arch}.embedding_length"])
        n_head = int(m[f"{arch}.attention.head_count"])
        head_dim = int(m.get(f"{arch}.attention.key_length", dim // n_head))
        if int(m.get(f"{arch}.attention.value_length", head_dim)) != head_dim:
            raise NotImplementedError("key_length diverso da value_length")
        if int(m.get(f"{arch}.rope.dimension_count", head_dim)) != head_dim:
            raise NotImplementedError("RoPE parziale (rope.dimension_count < head_dim) non supportata")
        vocab = m.get(f"{arch}.vocab_size")
        if vocab is None:
            vocab = gf.tensors["token_embd.weight"].shape[0]
        return cls(
            vocab_size=int(vocab),
            dim=dim,
            n_layer=int(m[f"{arch}.block_count"]),
            n_head=n_head,
            n_kv_head=int(m.get(f"{arch}.attention.head_count_kv", n_head)),
            head_dim=head_dim,
            ffn_dim=int(m[f"{arch}.feed_forward_length"]),
            norm_eps=float(m[f"{arch}.attention.layer_norm_rms_epsilon"]),
            rope_theta=float(m.get(f"{arch}.rope.freq_base", 10000.0)),
            tie_embeddings="output.weight" not in gf.tensors,
            qkv_bias="blk.0.attn_q.bias" in gf.tensors,
            max_pos=int(m.get(f"{arch}.context_length", 2048)),
            rope_interleaved=GGUF_ARCHS[arch],
        )

    def expected_shapes(self) -> dict:
        """Shape attese dei pesi, per chiave del dizionario dei parametri."""
        q, kv = self.n_head * self.head_dim, self.n_kv_head * self.head_dim
        layer = {"attn_norm": (self.dim,), "mlp_norm": (self.dim,),
                 "wq": (q, self.dim), "wk": (kv, self.dim), "wv": (kv, self.dim), "wo": (self.dim, q),
                 "w_gate": (self.ffn_dim, self.dim), "w_up": (self.ffn_dim, self.dim),
                 "w_down": (self.dim, self.ffn_dim)}
        if self.qkv_bias:
            layer.update(bq=(q,), bk=(kv,), bv=(kv,))
        return {"embed": (self.vocab_size, self.dim), "lm_head": (self.vocab_size, self.dim),
                "norm": (self.dim,), "layer": layer}


def check_shapes(cfg: LlamaConfig, p: dict):
    """Pesi incoerenti con la configurazione -> errore chiaro, invece di un reshape fallito più avanti."""
    exp = cfg.expected_shapes()
    for key in ("embed", "lm_head", "norm"):
        if p[key].shape != exp[key]:
            raise ValueError(f"{key}: shape {p[key].shape}, attesa {exp[key]}")
    for i, layer in enumerate(p["layers"]):
        for key, shape in exp["layer"].items():
            if layer[key].shape != shape:
                raise ValueError(f"strato {i}, {key}: shape {layer[key].shape}, attesa {shape}")


def load_weights_gguf(cfg: LlamaConfig, gf: GGUFFile) -> dict:
    """Nomi dei tensori GGUF -> stesso dizionario di load_weights. Tutto convertito in fp32."""
    p = {
        "embed": gf.get_f32("token_embd.weight"),
        "norm": gf.get_f32("output_norm.weight"),
        "layers": [],
    }
    p["lm_head"] = p["embed"] if cfg.tie_embeddings else gf.get_f32("output.weight")
    for i in range(cfg.n_layer):
        pre = f"blk.{i}."
        layer = {
            "attn_norm": gf.get_f32(pre + "attn_norm.weight"),
            "wq": gf.get_f32(pre + "attn_q.weight"),
            "wk": gf.get_f32(pre + "attn_k.weight"),
            "wv": gf.get_f32(pre + "attn_v.weight"),
            "wo": gf.get_f32(pre + "attn_output.weight"),
            "mlp_norm": gf.get_f32(pre + "ffn_norm.weight"),
            "w_gate": gf.get_f32(pre + "ffn_gate.weight"),
            "w_up": gf.get_f32(pre + "ffn_up.weight"),
            "w_down": gf.get_f32(pre + "ffn_down.weight"),
        }
        if cfg.qkv_bias:
            for n in ("q", "k", "v"):
                layer["b" + n] = gf.get_f32(pre + f"attn_{n}.bias")
        p["layers"].append(layer)
    return p


def load_weights(cfg: LlamaConfig, st: SafeTensorsDir) -> dict:
    """Nomi dei tensori HF -> dizionario compatto. Tutto convertito in fp32."""
    p = {
        "embed": st.get_f32("model.embed_tokens.weight"),  # [vocab, dim]
        "norm": st.get_f32("model.norm.weight"),
        "layers": [],
    }
    # Con gli embedding condivisi la proiezione di output è la stessa matrice (nessuna copia).
    p["lm_head"] = p["embed"] if cfg.tie_embeddings else st.get_f32("lm_head.weight")
    for i in range(cfg.n_layer):
        pre = f"model.layers.{i}."
        layer = {
            "attn_norm": st.get_f32(pre + "input_layernorm.weight"),
            "wq": st.get_f32(pre + "self_attn.q_proj.weight"),  # [n_head * hd, dim]
            "wk": st.get_f32(pre + "self_attn.k_proj.weight"),  # [n_kv_head * hd, dim]
            "wv": st.get_f32(pre + "self_attn.v_proj.weight"),
            "wo": st.get_f32(pre + "self_attn.o_proj.weight"),  # [dim, n_head * hd]
            "mlp_norm": st.get_f32(pre + "post_attention_layernorm.weight"),
            "w_gate": st.get_f32(pre + "mlp.gate_proj.weight"),  # [ffn, dim]
            "w_up": st.get_f32(pre + "mlp.up_proj.weight"),  # [ffn, dim]
            "w_down": st.get_f32(pre + "mlp.down_proj.weight"),  # [dim, ffn]
        }
        if cfg.qkv_bias:
            for n in ("q", "k", "v"):
                layer["b" + n] = st.get_f32(pre + f"self_attn.{n}_proj.bias")
        p["layers"].append(layer)
    return p


class KVCache:
    """KV cache preallocata: [n_layer][n_kv_head, max_seq, head_dim] per K e per V.

    Allocarla una volta sola, e scrivere in place, è lo schema che userà anche il runtime C.
    Memoria: 2 * n_layer * n_kv_head * max_seq * head_dim * 4 byte.
    """

    def __init__(self, cfg: LlamaConfig, max_seq: int):
        shape = (cfg.n_kv_head, max_seq, cfg.head_dim)
        self.k = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]
        self.v = [np.zeros(shape, np.float32) for _ in range(cfg.n_layer)]
        self.max_seq = max_seq
        self.length = 0  # token già presenti

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) * 2


class Llama:
    def __init__(self, cfg: LlamaConfig, params: dict, max_seq: int = 2048):
        check_shapes(cfg, params)
        self.cfg = cfg
        self.p = params
        self.max_seq = min(max_seq, cfg.max_pos)
        self.cos, self.sin = ops.rope_cos_sin(self.max_seq, cfg.head_dim, cfg.rope_theta)

    @classmethod
    def from_dir(cls, path, max_seq: int = 2048) -> "Llama":
        path = Path(path)
        cfg = LlamaConfig.from_hf(json.loads((path / "config.json").read_text()))
        return cls(cfg, load_weights(cfg, SafeTensorsDir(path)), max_seq)

    @classmethod
    def from_gguf(cls, path, max_seq: int = 2048) -> "Llama":
        with GGUFFile(path) as gf:
            cfg = LlamaConfig.from_gguf(gf)
            params = load_weights_gguf(cfg, gf)
        return cls(cfg, params, max_seq)

    @classmethod
    def load(cls, path, max_seq: int = 2048) -> "Llama":
        """Un file .gguf oppure una cartella HF con safetensors."""
        if str(path).endswith(".gguf"):
            return cls.from_gguf(path, max_seq)
        return cls.from_dir(path, max_seq)

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg, self.max_seq)

    def attention(self, x, layer, li, pos0, cache, mask):
        c = self.cfg
        t, hd = x.shape[0], c.head_dim
        q = ops.split_heads(ops.linear(x, layer["wq"], layer.get("bq")), c.n_head)  # [nh, T, hd]
        k = ops.split_heads(ops.linear(x, layer["wk"], layer.get("bk")), c.n_kv_head)  # [nkv, T, hd]
        v = ops.split_heads(ops.linear(x, layer["wv"], layer.get("bv")), c.n_kv_head)

        cos, sin = self.cos[pos0:pos0 + t], self.sin[pos0:pos0 + t]
        rope = ops.apply_rope_interleaved if c.rope_interleaved else ops.apply_rope
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)  # in cache finisce k già ruotato

        if cache is not None:
            cache.k[li][:, pos0:pos0 + t] = k
            cache.v[li][:, pos0:pos0 + t] = v
            k = cache.k[li][:, :pos0 + t]
            v = cache.v[li][:, :pos0 + t]

        # GQA senza copiare K e V: le n_rep teste di query di un gruppo diventano una dimensione
        # in più, e K/V (una testa per gruppo) vengono condivisi per broadcasting.
        n_rep = c.n_head // c.n_kv_head
        qg = q.reshape(c.n_kv_head, n_rep, t, hd)  # la testa h appartiene al gruppo h // n_rep
        out = ops.scaled_dot_product_attention(qg, k[:, None], v[:, None], mask)
        return ops.linear(ops.merge_heads(out.reshape(c.n_head, t, hd)), layer["wo"])

    def forward(self, tokens, cache: KVCache | None = None) -> np.ndarray:
        """Logits [T, vocab] per i token dati. Con una cache, i token proseguono quelli già visti."""
        tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if tokens.size == 0:
            raise ValueError("nessun token in ingresso")
        if tokens.min() < 0 or tokens.max() >= self.cfg.vocab_size:
            raise ValueError(f"id di token fuori dal vocabolario [0, {self.cfg.vocab_size})")
        pos0 = cache.length if cache is not None else 0
        t = len(tokens)
        if pos0 + t > self.max_seq:
            raise ValueError(f"contesto pieno: {pos0 + t} > max_seq={self.max_seq}")
        c = self.cfg

        # La maschera è la stessa per tutti gli strati; con un solo token nuovo non serve
        # (l'ultima posizione vede tutte le precedenti).
        mask = ops.causal_mask(t, pos0 + t) if t > 1 else None
        x = self.p["embed"][tokens]  # nessuna scala per sqrt(d), nessuna PE sommata
        for li, layer in enumerate(self.p["layers"]):
            x = x + self.attention(ops.rms_norm(x, layer["attn_norm"], c.norm_eps), layer, li, pos0, cache, mask)
            h = ops.rms_norm(x, layer["mlp_norm"], c.norm_eps)
            x = x + ops.swiglu_mlp(h, layer["w_gate"], layer["w_up"], layer["w_down"])
        x = ops.rms_norm(x, self.p["norm"], c.norm_eps)

        if cache is not None:
            cache.length += t
        return ops.linear(x, self.p["lm_head"])

    def generate(self, prompt, n_new: int, eos_ids=(), on_token=None) -> list[int]:
        """Greedy: prefill del prompt in un solo passo, poi decode un token alla volta.

        Restituisce solo i token nuovi. Si ferma al primo token in eos_ids (incluso nel risultato)
        o quando il contesto è pieno.
        """
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        out = []
        if n_new <= 0:
            return out
        cache = self.new_cache()
        logits = self.forward(prompt, cache)[-1]
        for _ in range(n_new):
            nxt = int(np.argmax(logits))
            out.append(nxt)
            if on_token:
                on_token(nxt)
            if nxt in eos_ids or cache.length >= self.max_seq:
                break
            logits = self.forward([nxt], cache)[-1]
        return out


class _Streamer:
    """Stampa il testo generato man mano, senza ridecodificare tutta la sequenza a ogni token.

    Un token può contenere solo una parte di un carattere UTF-8 (il decoder restituisce allora
    U+FFFD): si aspetta il token successivo. I token si accumulano fino a fine riga, così il
    costo resta proporzionale alla lunghezza della riga, non dell'intero testo.
    """

    def __init__(self, tok):
        self.tok = tok
        self.cache = []
        self.printed = 0

    def __call__(self, tid):
        self.cache.append(tid)
        text = self.tok.decode(self.cache)
        if text.endswith("�"):
            return
        print(text[self.printed:], end="", flush=True)
        if text.endswith("\n"):
            self.cache, self.printed = [], 0
        else:
            self.printed = len(text)


def main():
    from engine.tokenizer.hf import Tokenizer

    ap = argparse.ArgumentParser(description="Generazione greedy con la reference NumPy")
    ap.add_argument("model", help="file .gguf oppure cartella HF (config.json, *.safetensors, tokenizer.json)")
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=40, help="token da generare")
    ap.add_argument("--tokenizer", help="cartella HF con tokenizer.json (necessaria con i .gguf fino alla Fase 7)")
    args = ap.parse_args()

    tok_dir = args.tokenizer or args.model
    if not (Path(tok_dir) / "tokenizer.json").exists():
        ap.error("serve un tokenizer: con un .gguf passa --tokenizer models/<cartella HF>")
    if args.n < 0:
        ap.error("-n deve essere >= 0")

    t0 = time.perf_counter()
    model = Llama.load(args.model, max_seq=1024)
    tok = Tokenizer.from_dir(tok_dir)
    if args.model.endswith(".gguf"):
        with GGUFFile(args.model) as gf:
            eos = gf.metadata.get("tokenizer.ggml.eos_token_id")
        if eos is not None:
            tok.eos_ids = sorted(set(tok.eos_ids) | {int(eos)})
    t_load = time.perf_counter() - t0
    c = model.cfg
    n_params = sum(a.size for a in [model.p["embed"], model.p["norm"]]) + sum(
        a.size for layer in model.p["layers"] for a in layer.values())
    if not c.tie_embeddings:
        n_params += model.p["lm_head"].size
    print(f"[{c.n_layer} strati, dim {c.dim}, teste {c.n_head}/{c.n_kv_head} kv, "
          f"{n_params / 1e6:.0f}M parametri, caricato in {t_load:.1f}s]")

    ids = tok.encode(args.prompt)
    if not ids:
        ap.error("il prompt non produce nessun token")
    print(args.prompt, end="", flush=True)

    streamer = _Streamer(tok)
    times = []

    def on_token(tid):
        times.append(time.perf_counter())
        streamer(tid)

    t0 = time.perf_counter()
    generated = model.generate(ids, args.n, eos_ids=tok.eos_ids, on_token=on_token)
    if not generated:
        print("\n\n[nessun token generato]")
        return
    t_end = time.perf_counter()
    n_dec = len(generated) - 1
    decode = f"{n_dec / (t_end - times[0]):.1f} token/s" if n_dec and t_end > times[0] else "n/d"
    print(f"\n\n[prefill: {len(ids)} token in {times[0] - t0:.2f}s | decode: {decode}]")


if __name__ == "__main__":
    main()
