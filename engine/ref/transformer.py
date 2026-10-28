"""Transformer decoder-only in NumPy, fedele a "Attention Is All You Need".

Il paper descrive un encoder-decoder per la traduzione; i modelli linguistici usati
oggi (GPT, Llama, Qwen...) tengono solo la pila del decoder, senza il sotto-strato di
attenzione encoder-decoder. Questo file implementa quella pila con le scelte del paper:

- post-norm: x = LayerNorm(x + Sublayer(x))                       (sez. 3.1)
- multi-head self-attention causale                                (sez. 3.2)
- FFN con ReLU                                                      (sez. 3.3)
- embedding condivisi con la proiezione di output, scalati per sqrt(d_model)  (sez. 3.4)
- positional encoding sinusoidale sommato agli embedding            (sez. 3.5)

I modelli Llama-like della Fase 1 cambiano: pre-norm con RMSNorm, RoPE al posto della PE
sommata, SwiGLU al posto della ReLU, GQA. La struttura resta la stessa.

Il modello ha pesi casuali: serve a studiare il flusso dei dati e le shape, non a
generare testo sensato. I pesi veri arrivano in Fase 1.
"""
import math
from dataclasses import dataclass

import numpy as np

from engine.ref import ops


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    d_ff: int = 256
    max_len: int = 128

    @property
    def d_head(self) -> int:
        # d_k = d_v = d_model / h (sez. 3.2.2)
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


def init_params(cfg: Config, seed: int = 0) -> dict:
    """Pesi casuali con varianza ~1/d_in, così le attivazioni restano di scala ragionevole."""
    rng = np.random.default_rng(seed)

    def w(d_in, d_out):
        return (rng.standard_normal((d_in, d_out)) / np.sqrt(d_in)).astype(np.float32)

    d, f = cfg.d_model, cfg.d_ff
    params = {
        # Embedding [vocab, d_model]; la stessa matrice (trasposta) fa da proiezione di output.
        "tok_emb": (rng.standard_normal((cfg.vocab_size, d)) / np.sqrt(d)).astype(np.float32),
        "layers": [],
    }
    for _ in range(cfg.n_layer):
        params["layers"].append({
            # Le proiezioni di tutte le teste sono impilate: W^Q = [W_1^Q | ... | W_h^Q].
            "wq": w(d, d), "wk": w(d, d), "wv": w(d, d), "wo": w(d, d),
            "ln1_g": np.ones(d, np.float32), "ln1_b": np.zeros(d, np.float32),
            "w1": w(d, f), "b1": np.zeros(f, np.float32),
            "w2": w(f, d), "b2": np.zeros(d, np.float32),
            "ln2_g": np.ones(d, np.float32), "ln2_b": np.zeros(d, np.float32),
        })
    return params


def count_params(params: dict) -> int:
    n = params["tok_emb"].size
    for layer in params["layers"]:
        n += sum(a.size for a in layer.values())
    return n


class KVCache:
    """Chiavi e valori già calcolati, per ogni strato: [n_head, T, d_head].

    In generazione ogni nuovo token deve vedere K e V di tutti i token precedenti;
    tenerli in memoria evita di ricalcolare l'intero prefisso a ogni passo.
    """

    def __init__(self, cfg: Config):
        self.k = [np.zeros((cfg.n_head, 0, cfg.d_head), np.float32) for _ in range(cfg.n_layer)]
        self.v = [np.zeros((cfg.n_head, 0, cfg.d_head), np.float32) for _ in range(cfg.n_layer)]

    @property
    def length(self) -> int:
        return self.k[0].shape[1]


class Transformer:
    def __init__(self, cfg: Config, params: dict):
        self.cfg = cfg
        self.p = params
        self.pe = ops.sinusoidal_positional_encoding(cfg.max_len, cfg.d_model)

    def attention(self, x, layer, li, cache):
        """Multi-head self-attention causale (sez. 3.2.2) su x: [T, d_model]."""
        h = self.cfg.n_head
        q = ops.split_heads(x @ layer["wq"], h)  # [h, T, d_head]
        k = ops.split_heads(x @ layer["wk"], h)
        v = ops.split_heads(x @ layer["wv"], h)
        if cache is not None:
            k = np.concatenate([cache.k[li], k], axis=1)
            v = np.concatenate([cache.v[li], v], axis=1)
            cache.k[li], cache.v[li] = k, v
        mask = ops.causal_mask(q.shape[1], k.shape[1])
        out = ops.scaled_dot_product_attention(q, k, v, mask)  # [h, T, d_head]
        return ops.merge_heads(out) @ layer["wo"]

    def forward(self, tokens, cache: KVCache | None = None) -> np.ndarray:
        """tokens: sequenza di id. Restituisce i logits [T, vocab].

        Con una cache, `tokens` sono solo i token nuovi: le loro posizioni partono da cache.length.
        """
        tokens = np.asarray(tokens)
        start = cache.length if cache is not None else 0
        t = len(tokens)
        if start + t > self.cfg.max_len:
            raise ValueError(f"sequenza più lunga di max_len={self.cfg.max_len}")

        # Sez. 3.4 e 3.5: embedding * sqrt(d_model) + positional encoding.
        x = self.p["tok_emb"][tokens] * math.sqrt(self.cfg.d_model) + self.pe[start:start + t]

        for li, layer in enumerate(self.p["layers"]):
            # Post-norm come nel paper: LayerNorm(x + Sublayer(x)).
            x = ops.layer_norm(x + self.attention(x, layer, li, cache), layer["ln1_g"], layer["ln1_b"])
            ffn = ops.feed_forward(x, layer["w1"], layer["b1"], layer["w2"], layer["b2"])
            x = ops.layer_norm(x + ffn, layer["ln2_g"], layer["ln2_b"])

        # Proiezione di output con la matrice di embedding condivisa (sez. 3.4).
        return x @ self.p["tok_emb"].T

    def generate(self, prompt, n_new: int) -> list[int]:
        """Generazione greedy: un passo di prefill sul prompt, poi un token alla volta (decode).

        Restituisce solo i token nuovi, come Llama.generate.
        """
        cache = KVCache(self.cfg)
        logits = self.forward(prompt, cache)  # prefill
        out = []
        for _ in range(n_new):
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            logits = self.forward([nxt], cache)  # decode: solo il token nuovo
        return out


if __name__ == "__main__":
    cfg = Config()
    model = Transformer(cfg, init_params(cfg))
    print(cfg)
    print(f"parametri: {count_params(model.p):,}")
    prompt = list(b"Attention")
    print("generato (pesi casuali):", prompt + model.generate(prompt, 10))
