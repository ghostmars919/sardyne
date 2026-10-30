"""Il runtime C visto da Python: modello, contesto, generazione greedy e CLI.

Il forward gira interamente in C (csrc/model_llama.c); qui restano il ciclo di generazione,
la scelta del token e il tokenizer.

Kernel SIMD (AVX2) e thread: vedi set_simd() e il parametro n_threads di Context.

Uso: python -m engine.runtime models/gguf/SmolLM2-135M-Instruct-f16.gguf "The capital of France is" \\
         -n 40 [--compare]
"""
import argparse
import ctypes
import time
from pathlib import Path

import numpy as np

from engine import capi


def simd() -> bool:
    """True se il runtime usa i kernel AVX2."""
    return bool(capi.lib().op_simd())


def set_simd(enable: bool) -> bool:
    """Forza i kernel scalari (False) o AVX2 se la CPU li supporta (True). Restituisce lo stato effettivo.

    Da chiamare quando nessun eval è in corso: serve a test e benchmark.
    """
    return bool(capi.lib().op_set_simd(int(enable)))


class Model:
    """Pesi letti con mmap da un GGUF (architetture llama, qwen2, qwen3 e qwen3moe; pesi F32/F16/BF16 o quantizzati
    Q8_0/Q4_0/Q5_0/Q4_K/Q6_K, quindi anche i file Q8_0 e Q4_K_M)."""

    def __init__(self, path):
        self._lib = capi.lib()
        self._h = None
        err = ctypes.create_string_buffer(512)
        self._h = self._lib.engine_model_load(str(path).encode(), err, len(err))
        if not self._h:
            raise ValueError(err.value.decode("utf-8", errors="replace"))
        self.info = capi.EngineModelInfo()
        self._lib.engine_model_get_info(self._h, ctypes.byref(self.info))
        self.path = Path(path)

    def __del__(self):
        if self._h:
            self._lib.engine_model_free(self._h)
            self._h = None


class Context:
    """KV cache e memoria di lavoro di una sessione. Tiene un riferimento al modello."""

    def __init__(self, model: Model, max_seq: int = 2048, n_batch: int = 64, logits_all: bool = False,
                 n_threads: int = 0):
        self.model = model
        self._lib = model._lib
        self._h = None
        self.max_seq = min(max_seq, model.info.max_pos)
        self.n_batch = n_batch
        self.logits_all = logits_all
        err = ctypes.create_string_buffer(512)
        self._h = self._lib.engine_ctx_new(model._h, self.max_seq, n_batch, int(logits_all), n_threads,
                                           err, len(err))
        if not self._h:
            raise ValueError(err.value.decode("utf-8", errors="replace"))
        self.n_threads = self._lib.engine_ctx_n_threads(self._h)  # 0 = uno per core

    def __del__(self):
        if self._h:
            self._lib.engine_ctx_free(self._h)
            self._h = None

    @property
    def n_past(self) -> int:
        return self._lib.engine_n_past(self._h)

    def reset(self):
        self._lib.engine_reset(self._h)

    def eval(self, tokens) -> np.ndarray:
        """Logits (copia) [n_token, vocab] con logits_all, altrimenti [vocab] dell'ultimo token.

        Con logits_all i token vengono passati al C in blocchi di n_batch.
        """
        tokens = np.ascontiguousarray(np.asarray(tokens, dtype=np.int64).reshape(-1))
        if tokens.size == 0:
            raise ValueError("nessun token in ingresso")
        if tokens.min() < np.iinfo(np.int32).min or tokens.max() > np.iinfo(np.int32).max:
            raise ValueError("id di token fuori dall'intervallo int32")
        tokens = tokens.astype(np.int32)
        step = self.n_batch if self.logits_all else tokens.size
        outs = []
        for i in range(0, tokens.size, step):
            part = tokens[i:i + step]
            ptr = part.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if self._lib.engine_eval(self._h, ptr, part.size) != 0:
                raise ValueError(self._lib.engine_last_error(self._h).decode("utf-8", errors="replace"))
            n = self._lib.engine_n_logits(self._h)
            logits = np.ctypeslib.as_array(self._lib.engine_logits(self._h), shape=(n, self.model.info.n_vocab))
            outs.append(logits.copy())
        out = np.concatenate(outs)
        return out if self.logits_all else out[-1]


def generate(ctx: Context, prompt, n_new: int, eos_ids=(), on_token=None) -> list[int]:
    """Greedy come Llama.generate: prefill del prompt, poi un token alla volta. Solo i token nuovi."""
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    out = []
    if n_new <= 0:
        return out
    ctx.reset()
    logits = ctx.eval(prompt)
    if ctx.logits_all:
        logits = logits[-1]
    for _ in range(n_new):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if nxt in eos_ids or ctx.n_past >= ctx.max_seq:
            break
        logits = ctx.eval([nxt])
        if ctx.logits_all:
            logits = logits[-1]
    return out


def main():
    from engine.tokenizer.bpe import BPETokenizer, Streamer

    ap = argparse.ArgumentParser(description="Generazione greedy con il runtime C")
    ap.add_argument("model", help="file .gguf (llama, qwen2, qwen3, qwen3moe; F32/F16/BF16, Q8_0, Q4_K_M...)")
    ap.add_argument("prompt")
    ap.add_argument("--tokenizer", help="cartella HF con tokenizer.json (default: quello nel file .gguf)")
    ap.add_argument("-n", type=int, default=40, help="token da generare")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=0, help="thread di calcolo (0 = uno per core)")
    ap.add_argument("--no-simd", action="store_true", help="usa i kernel scalari")
    ap.add_argument("--compare", action="store_true", help="confronta i logits con la reference NumPy")
    args = ap.parse_args()
    if args.n < 0:
        ap.error("-n deve essere >= 0")

    if args.no_simd:
        set_simd(False)
    t0 = time.perf_counter()
    model = Model(args.model)
    ctx = Context(model, max_seq=args.max_seq, n_threads=args.threads)
    tok = BPETokenizer.load(args.model, args.tokenizer)
    i = model.info
    print(f"[runtime C: {i.n_layer} strati, dim {i.dim}, teste {i.n_head}/{i.n_kv_head} kv, "
          f"{ctx.n_threads} thread, {'AVX2' if simd() else 'scalare'}, caricato in {time.perf_counter() - t0:.2f}s]")

    ids = tok.encode(args.prompt)
    if not ids:
        ap.error("il prompt non produce nessun token")

    if args.compare:
        from engine.ref.llama import Llama

        ref = Llama.from_gguf(args.model, max_seq=len(ids), emulate_quant=True, lazy=bool(model.info.n_expert))
        c_all = Context(model, max_seq=len(ids), n_batch=len(ids), logits_all=True)
        d = np.abs(c_all.eval(ids) - ref.forward(ids))
        print(f"[confronto con NumPy sul prompt: max |diff| {d.max():.2e}, media {d.mean():.2e} "
              f"(con pesi quantizzati l'errore delle attivazioni int8 cresce con la profondità)]")

    print(args.prompt, end="", flush=True)
    streamer = Streamer(tok)
    times = []

    def on_token(tid):
        times.append(time.perf_counter())
        streamer(tid)

    t0 = time.perf_counter()
    generated = generate(ctx, ids, args.n, eos_ids=tok.eos_ids, on_token=on_token)
    if not generated:
        print("\n\n[nessun token generato]")
        return
    t_end = time.perf_counter()
    n_dec = len(generated) - 1
    decode = f"{n_dec / (t_end - times[0]):.2f} token/s" if n_dec and t_end > times[0] else "n/d"
    print(f"\n\n[prefill: {len(ids)} token in {times[0] - t0:.2f}s | decode: {decode}]")


if __name__ == "__main__":
    main()
