"""Chat da terminale.

Uso: python -m engine.cli models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf [--system "..."] [--temperature 0.7]

Comandi: /reset ricomincia la conversazione, /exit esce (come Ctrl-D).
"""
import argparse
import time

from engine.chat import THINK_CLOSE, THINK_OPEN, ChatEngine
from engine.runtime import add_sampling_args, sampling_params
from engine.sampling import Sampler


class ThinkFilter:
    """Toglie <think>...</think> dal testo in streaming, anche con i tag spezzati tra i pezzi."""

    def __init__(self):
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buf += text
        out = []
        while True:
            tag = THINK_CLOSE if self._inside else THINK_OPEN
            pos = self._buf.find(tag)
            if pos >= 0:
                if not self._inside:
                    out.append(self._buf[:pos])
                self._buf = self._buf[pos + len(tag):]
                self._inside = not self._inside
                continue
            keep = next((i for i in range(max(0, len(self._buf) - len(tag) + 1), len(self._buf))
                         if tag.startswith(self._buf[i:])), len(self._buf))
            if not self._inside:
                out.append(self._buf[:keep])
            self._buf = self._buf[keep:]
            return "".join(out)

    def finish(self) -> str:
        rest, self._buf = "" if self._inside else self._buf, ""
        return rest


def main():
    ap = argparse.ArgumentParser(description="Chat con un modello GGUF")
    ap.add_argument("model")
    ap.add_argument("--tokenizer", help="cartella HF (default: il tokenizer nel file .gguf)")
    ap.add_argument("--system", help="messaggio di sistema")
    ap.add_argument("--max-seq", type=int, default=4096, help="lunghezza del contesto")
    ap.add_argument("--max-tokens", type=int, default=1024, help="token massimi per risposta")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-think", action="store_true", help="chiede al modello di non ragionare (Qwen3)")
    ap.add_argument("--hide-think", action="store_true", help="non mostra il blocco <think>")
    add_sampling_args(ap, temperature=0.7)
    args = ap.parse_args()

    engine = ChatEngine(args.model, args.max_seq, args.threads, args.tokenizer)
    sampler = Sampler(sampling_params(args))
    base = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)
    print("[pronto: /reset ricomincia, /exit esce]")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/exit":
            return
        if line == "/reset":
            messages = list(base)
            print("[conversazione azzerata]")
            continue

        messages.append({"role": "user", "content": line})
        think = ThinkFilter() if args.hide_think else None
        reply = []
        t0 = time.perf_counter()
        try:
            for piece in engine.stream(messages, sampler=sampler, max_tokens=args.max_tokens,
                                       enable_thinking=False if args.no_think else None):
                reply.append(piece)
                print(think.feed(piece) if think else piece, end="", flush=True)
        except ValueError as e:
            messages.pop()
            print(f"[errore: {e}; /reset per ricominciare]")
            continue
        except KeyboardInterrupt:
            print("\n[interrotto]")
        if think:
            print(think.finish(), end="")
        messages.append({"role": "assistant", "content": "".join(reply)})
        s = engine.stats
        print(f"\n[{s.prompt_tokens} token di prompt, {s.completion_tokens} generati, "
              f"{time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main()
