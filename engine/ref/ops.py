"""Operazioni di base del Transformer in NumPy (reference, fp32).

Riferimenti alle sezioni di "Attention Is All You Need" (docs/1706.03762).
Convenzione: le matrici di peso hanno shape [d_in, d_out] e si applicano come x @ W.
"""
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
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)  # [..., t_q, t_k]
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
