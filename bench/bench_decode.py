"""Velocità di prefill e decode: reference NumPy contro runtime C, sullo stesso file GGUF.

Uso: python bench/bench_decode.py models/gguf/SmolLM2-135M-Instruct-f16.gguf [--prompt 64] [--decode 32] [--skip-numpy]

Oltre ai token/s stampa la banda "effettiva" del decode: byte dei pesi letti per token × token/s.
Il decode legge tutti i pesi a ogni token, quindi il suo limite teorico è la banda della RAM
(sezione roofline in notes/03-runtime.md).
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.loader.gguf import GGUFFile  # noqa: E402
from engine.ref.llama import Llama  # noqa: E402
from engine.runtime import Context, Model  # noqa: E402


def weight_bytes_per_token(path, as_f32=False) -> int:
    """Byte dei pesi letti per un token: tutte le matrici, ma dell'embedding solo una riga.

    as_f32: byte nella reference NumPy, che tiene tutti i pesi in fp32 (4 byte per peso).
    """
    with GGUFFile(path) as f:
        size = {name: (t.n_elements * 4 if as_f32 else t.nbytes) for name, t in f.tensors.items()}
        total = sum(size.values())
        emb = f.tensors["token_embd.weight"]
        if "output.weight" in f.tensors:  # embedding separato: si legge una riga sola
            total -= size["token_embd.weight"] - size["token_embd.weight"] // emb.shape[0]
        return total


def bench(name, prefill, step, prompt, n_decode):
    t0 = time.perf_counter()
    prefill(prompt)
    t_prefill = time.perf_counter() - t0
    tok = 1
    t0 = time.perf_counter()
    for _ in range(n_decode):
        step(tok)
    t_decode = time.perf_counter() - t0
    return {"name": name, "prefill": len(prompt) / t_prefill, "decode": n_decode / t_decode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--skip-numpy", action="store_true")
    args = ap.parse_args()

    prompt = list(range(100, 100 + args.prompt))
    max_seq = args.prompt + args.decode + 1
    results = []

    model = Model(args.model)
    ctx = Context(model, max_seq=max_seq, n_batch=64)
    results.append(bench("runtime C (fp32 scalare, 1 thread)", ctx.eval, lambda t: ctx.eval([t]),
                         prompt, args.decode) | {"gb": weight_bytes_per_token(args.model) / 1e9})
    del ctx, model
    gc.collect()

    if not args.skip_numpy:
        ref = Llama.from_gguf(args.model, max_seq=max_seq)
        cache = ref.new_cache()
        results.append(bench("reference NumPy (OpenBLAS, pesi fp32)", lambda p: ref.forward(p, cache),
                             lambda t: ref.forward([t], cache), prompt, args.decode)
                       | {"gb": weight_bytes_per_token(args.model, as_f32=True) / 1e9})

    print(f"{Path(args.model).name}: prompt {args.prompt} token, decode {args.decode} token")
    print(f"  {'motore':40s} {'prefill tok/s':>14s} {'decode tok/s':>13s} {'GB/token':>9s} {'banda decode':>13s}")
    for r in results:
        print(f"  {r['name']:40s} {r['prefill']:14.1f} {r['decode']:13.2f} {r['gb']:9.2f} "
              f"{r['decode'] * r['gb']:8.2f} GB/s")


if __name__ == "__main__":
    np.random.seed(0)
    main()
