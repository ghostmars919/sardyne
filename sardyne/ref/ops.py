"""Operazioni di base del Transformer in NumPy (reference, fp32).

Riferimenti alle sezioni di "Attention Is All You Need" (docs/1706.03762).
Convenzione: le matrici di peso hanno shape [d_in, d_out] e si applicano come x @ W.
"""
import math

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Sottrarre il massimo non cambia il risultato (si semplifica nel rapporto)
    # ma evita overflow di exp() con valori grandi. Righe tutte a -inf danno NaN:
    # non succede con la maschera causale, perché ogni query vede almeno se stessa.
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # Normalizza ogni vettore (ultima dimensione) a media 0 e varianza 1, poi riscala. [1]
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def sinusoidal_positional_encoding(n_pos: int, d_model: int) -> np.ndarray:
    """Sez. 3.5: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...). Shape [n_pos, d_model]."""
    assert d_model % 2 == 0
    pos = np.arange(n_pos, dtype=np.float64)[:, None]
    two_i = np.arange(0, d_model, 2, dtype=np.float64)[None, :]
    angle = pos / np.power(10000.0, two_i / d_model)
    pe = np.empty((n_pos, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe.astype(np.float32)


def causal_mask(t_q: int, t_k: int) -> np.ndarray:
    """Maschera additiva [t_q, t_k]: 0 dove consentito, -inf sulle posizioni future (sez. 3.2.3).

    Le query sono le ultime t_q posizioni di una sequenza lunga t_k (utile con la KV cache):
    la query i si trova alla posizione assoluta t_k - t_q + i e vede le chiavi j <= quella posizione.
    """
    q_pos = np.arange(t_k - t_q, t_k)[:, None]
    k_pos = np.arange(t_k)[None, :]
    return np.where(k_pos <= q_pos, 0.0, -np.inf).astype(np.float32)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Eq. (1): softmax(Q K^T / sqrt(d_k)) V.

    q: [..., t_q, d_k], k: [..., t_k, d_k], v: [..., t_k, d_v], mask: broadcastabile a [..., t_q, t_k].
    """
    d_k = q.shape[-1]
    # math.sqrt e non np.sqrt: np.sqrt(int) è uno scalare np.float64 che, con le regole di
    # promozione di NumPy 2 (NEP 50), trasformerebbe tutto il calcolo in float64.
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(d_k)  # [..., t_q, t_k]
    if mask is not None:
        scores = scores + mask
    return softmax(scores, axis=-1) @ v


def split_heads(x: np.ndarray, n_head: int) -> np.ndarray:
    # [T, n_head * d_head] -> [n_head, T, d_head]
    t, d = x.shape
    return x.reshape(t, n_head, d // n_head).transpose(1, 0, 2)


def merge_heads(x: np.ndarray) -> np.ndarray:
    # [n_head, T, d_head] -> [T, n_head * d_head]  (è il Concat della sez. 3.2.2)
    h, t, dh = x.shape
    return x.transpose(1, 0, 2).reshape(t, h * dh)


def feed_forward(x, w1, b1, w2, b2):
    # Eq. (2): FFN(x) = max(0, x W1 + b1) W2 + b2, applicata a ogni posizione indipendentemente.
    return relu(x @ w1 + b1) @ w2 + b2


# ---------------------------------------------------------------- operazioni Llama-like (Fase 1)

def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Strato lineare con pesi nel layout di PyTorch/HF: w ha shape [d_out, d_in], quindi x @ w.T.

    È anche il layout di GGUF e del futuro runtime C: ogni riga di w è un neurone di output,
    contigua in memoria, e il prodotto matrice-vettore diventa una serie di prodotti scalari.
    """
    y = x @ w.T
    return y if b is None else y + b


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    # RMSNorm (Zhang & Sennrich 2019): come LayerNorm ma senza sottrarre la media e senza bias.
    # x / sqrt(mean(x^2) + eps) * weight
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def silu(x: np.ndarray) -> np.ndarray:
    # SiLU (o swish): x * sigmoid(x). Per x < -88 exp(-x) trabocca in float32: il risultato
    # (x / inf = -0.0) è comunque quello giusto, quindi si silenzia solo l'avviso.
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


def swiglu_mlp(x, w_gate, w_up, w_down):
    # SwiGLU (Shazeer 2020): down( silu(gate(x)) * up(x) ). Tre matrici invece di due, nessun bias.
    return linear(silu(linear(x, w_gate)) * linear(x, w_up), w_down)


def rope_cos_sin(n_pos: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Tabelle cos/sin della RoPE (RoFormer, Su et al. 2021), shape [n_pos, head_dim].

    Frequenze: inv_freq[i] = theta^(-2i/head_dim), i = 0..head_dim/2-1 (come la PE sinusoidale
    del paper Transformer, con base theta al posto di 10000). Le tabelle sono duplicate sulle due
    metà perché si usa la convenzione "rotate_half" di HF (vedi apply_rope).
    """
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.arange(n_pos, dtype=np.float64)[:, None] * inv_freq[None, :]  # [n_pos, hd/2]
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Ruota ogni coppia di dimensioni (i, i + head_dim/2) dell'angolo pos * inv_freq[i].

    x: [n_head, T, head_dim], cos/sin: [T, head_dim] per le posizioni di quei T token.
    Convenzione HF "rotate_half": le coppie sono (i, i + hd/2). llama.cpp/GGUF usa invece coppie
    adiacenti (2i, 2i+1), e il convertitore permuta le righe di W_q e W_k per compensare (Fase 2).
    Il prodotto scalare tra q ruotato in posizione m e k ruotato in posizione n dipende solo da m - n:
    l'informazione di posizione diventa relativa.
    """
    return x * cos + rotate_half(x) * sin


def apply_rope_interleaved(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """RoPE sulle coppie adiacenti (2i, 2i+1): la convenzione di llama.cpp per l'architettura "llama".

    Stessi cos/sin di rope_cos_sin (si usa solo la prima metà, l'altra è un duplicato).
    Il convertitore GGUF permuta le righe di W_q e W_k in modo che questa rotazione dia
    esattamente lo stesso risultato di apply_rope sui pesi originali di HF.
    """
    half = x.shape[-1] // 2
    c, s = cos[..., :half], sin[..., :half]
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """GQA (Ainslie et al. 2023): [n_kv_head, T, hd] -> [n_kv_head * n_rep, T, hd].

    Gruppi di n_rep teste di query condividono la stessa testa di chiavi/valori: la query h usa
    la testa KV h // n_rep. La KV cache si riduce di un fattore n_rep.
    """
    return x if n_rep == 1 else np.repeat(x, n_rep, axis=0)
