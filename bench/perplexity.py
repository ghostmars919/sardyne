"""Perplexity su wikitext-2 con il runtime C, per confrontare la qualità delle quantizzazioni.

Uso: python bench/perplexity.py models/gguf/qwen2.5-0.5b-instruct-{fp16,q8_0,q4_k_m}.gguf [--ctx 512] [--chunks 8]

Perplexity = exp(media di −log p(token corretto)): quanto il modello è "sorpreso" in media dal testo
vero. Come llama-perplexity, il testo si divide in finestre da --ctx token e si contano solo i token
della seconda metà di ogni finestra, che hanno almeno ctx/2 token di contesto. Tutti i file vengono
valutati sulla stessa sequenza di token, quindi i numeri sono confrontabili tra loro (non con quelli
pubblicati per altre impostazioni).
Il testo si scarica con scripts/download_wikitext.py.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sardyne.runtime import Context, Model  # noqa: E402
from sardyne.tokenizer.bpe import BPETokenizer  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "wikitext-2-raw" / "wiki.test.raw"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def perplexity(path, tokens, ctx_len, n_chunks, n_batch, n_threads):
    model = Model(path)
    ctx = Context(model, max_seq=ctx_len, n_batch=n_batch, logits_all=True, n_threads=n_threads)
    nll, count = [], 0
    t0 = time.perf_counter()
    for c in range(n_chunks):
        chunk = tokens[c * ctx_len:(c + 1) * ctx_len]
        ctx.reset()
        first = ctx_len // 2
        for start in range(0, ctx_len, n_batch):  # a blocchi: logits di n_batch token alla volta
            piece = chunk[start:start + n_batch]
            lp = log_softmax(ctx.eval(piece))
            for i in range(len(piece)):
                pos = start + i  # i logits in pos predicono il token pos + 1
                if first <= pos < ctx_len - 1:
                    nll.append(-lp[i, chunk[pos + 1]])
        count += 1
        ppl = math.exp(np.mean(nll))
        print(f"\r  {Path(path).name}: finestra {count}/{n_chunks}, perplexity {ppl:.4f}", end="", flush=True)
    dt = time.perf_counter() - t0
    nll = np.array(nll)
    ppl = math.exp(nll.mean())
    err = ppl * nll.std(ddof=1) / math.sqrt(len(nll))  # errore standard (propagato da quello della media)
    print(f"\r  {Path(path).name}: perplexity {ppl:.4f} ± {err:.4f} ({len(nll)} token, {dt:.0f}s)          ")
    return ppl, err


def main():
    ap = argparse.ArgumentParser(description="Perplexity su wikitext-2 (runtime C)")
    ap.add_argument("models", nargs="+", help="file .gguf della stessa famiglia (stesso tokenizer)")
    ap.add_argument("--tokenizer", help="cartella HF (default: il tokenizer del primo file)")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if not DATA.exists():
        ap.error(f"{DATA} mancante: eseguire python scripts/download_wikitext.py")

    tokens = BPETokenizer.load(args.models[0], args.tokenizer).encode(DATA.read_text(encoding="utf-8"))
    n_chunks = min(args.chunks, len(tokens) // args.ctx)
    print(f"wikitext-2 test: {len(tokens)} token, {n_chunks} finestre da {args.ctx} "
          f"(contati gli ultimi {args.ctx // 2 - 1} di ogni finestra)")
    results = {Path(m).name: perplexity(m, tokens, args.ctx, n_chunks, args.batch, args.threads) for m in args.models}

    base = next(iter(results.values()))[0]
    print(f"\n  {'file':40s} {'perplexity':>12s} {'rispetto al primo':>18s}")
    for name, (ppl, err) in results.items():
        print(f"  {name:40s} {ppl:8.4f} ± {err:.3f} {100 * (ppl / base - 1):+16.2f}%")


if __name__ == "__main__":
    main()
