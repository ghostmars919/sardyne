"""Scelta del prossimo token a partire dai logits.

Ordine dei filtri, lo stesso di llama.cpp:
1. repetition penalty sui token recenti (CTRL): logit positivo diviso per p, negativo moltiplicato;
2. top-k: solo i k token più probabili;
3. top-p (nucleus): i più probabili fino a coprire una probabilità cumulata p;
4. min-p: solo i token con probabilità almeno min_p volte quella del migliore;
5. temperature: si divide per T e si estrae dalla distribuzione rimasta.

I filtri 2-4 lavorano sulle probabilità prima della temperature. Con temperature 0 si sceglie il
migliore: è il greedy, deterministico, e i filtri 2-4 non cambierebbero il risultato.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0  # 0 = disattivato
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    seed: int | None = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature={self.temperature} negativa")
        if self.top_k < 0:
            raise ValueError(f"top_k={self.top_k} negativo")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p={self.top_p} fuori da (0, 1]")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p={self.min_p} fuori da [0, 1]")
        if self.repeat_penalty <= 0:
            raise ValueError(f"repeat_penalty={self.repeat_penalty} non positiva")
        if self.repeat_last_n < 0:
            raise ValueError(f"repeat_last_n={self.repeat_last_n} negativo")


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    p = np.exp(x)
    return p / p.sum()


def apply_repeat_penalty(logits: np.ndarray, history, penalty: float, last_n: int) -> np.ndarray:
    if penalty == 1.0 or last_n == 0 or not len(history):
        return logits
    ids = np.unique(np.asarray(history[-last_n:], dtype=np.int64))
    out = logits.copy()
    out[ids] = np.where(out[ids] > 0, out[ids] / penalty, out[ids] * penalty)
    return out


def keep_mask(probs: np.ndarray, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0) -> np.ndarray:
    """Token che superano top-k, top-p e min-p. Il più probabile resta sempre."""
    order = np.argsort(-probs, kind="stable")
    n = len(order)
    if top_k:
        n = min(n, top_k)
    if top_p < 1.0:
        cum = np.cumsum(probs[order[:n]])
        # Il primo token che raggiunge top_p entra: copre la massa richiesta.
        n = min(n, int(np.searchsorted(cum, top_p)) + 1)
    if min_p > 0.0:
        n = max(1, min(n, int(np.count_nonzero(probs[order[:n]] >= min_p * probs[order[0]]))))
    keep = np.zeros(len(probs), bool)
    keep[order[:n]] = True
    return keep


class Sampler:
    def __init__(self, params: SamplingParams | None = None):
        self.params = params or SamplingParams()
        self.rng = np.random.default_rng(self.params.seed)

    def __call__(self, logits: np.ndarray, history=()) -> int:
        """Il prossimo token. history: i token già visti, per la repetition penalty."""
        p = self.params
        x = apply_repeat_penalty(np.asarray(logits, np.float64), history, p.repeat_penalty, p.repeat_last_n)
        if p.temperature == 0:
            return int(np.argmax(x))
        keep = keep_mask(softmax(x), p.top_k, p.top_p, p.min_p)
        ids = np.flatnonzero(keep)
        return int(self.rng.choice(ids, p=softmax(x[ids] / p.temperature)))
