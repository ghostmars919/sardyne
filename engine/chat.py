"""Chat: template dei metadati, token speciali protetti, generazione in streaming e tool calling.

Il testo dei messaggi non è fidato: se un utente o uno strumento scrive "<|im_end|>", deve restare
testo, altrimenti potrebbe chiudere il turno e inventarne altri. Nei messaggi user e tool si inserisce
GUARD dentro ogni token speciale, così la ricerca dei token non lo riconosce; encode lo toglie prima del BPE.

Strumenti: il template li descrive al modello, che risponde con blocchi
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>
Il testo dentro <think> non conta: lì il modello ragiona, non chiama.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from engine.runtime import Context, Model
from engine.sampling import Sampler
from engine.tokenizer.bpe import BPETokenizer, Streamer

GUARD = ""  # area privata: nessun testo normale lo contiene, e viene comunque tolto dall'ingresso
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_TOOL_BLOCK = re.compile(re.escape(TOOL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CLOSE), re.DOTALL)


def _raise(message):
    raise ValueError(f"chat template: {message}")


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class ChatTemplate:
    """Il template viene da un file scaricato: gira in un ambiente Jinja che non può toccare il sistema."""

    def __init__(self, source: str, bos_token: str = "", eos_token: str = ""):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals.update(raise_exception=_raise, strftime_now=lambda fmt: datetime.now().strftime(fmt))
        self._template = env.from_string(source)
        self.bos_token, self.eos_token = bos_token, eos_token

    def render(self, messages, tools=None, add_generation_prompt: bool = True, enable_thinking=None) -> str:
        extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                     bos_token=self.bos_token, eos_token=self.eos_token, **extra)


def guard_message(msg: dict, tok: BPETokenizer) -> dict:
    """Copia del messaggio con i token speciali del testo resi innocui (solo user e tool)."""
    content = msg.get("content")
    if msg.get("role") not in ("user", "tool") or not isinstance(content, str):
        return msg
    return {**msg, "content": tok.guard_special(content.replace(GUARD, ""), GUARD)}


def _valid_call(raw: str):
    try:
        call = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
        return None
    args = call.get("arguments", {})
    if isinstance(args, str):  # alcuni modelli scrivono gli argomenti come stringa JSON
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return {"name": call["name"], "arguments": args} if isinstance(args, dict) else None


def split_reasoning(text: str) -> tuple[str, str]:
    """(ragionamento fino a </think> compreso, risposta). Senza </think> è tutto risposta."""
    head, sep, tail = text.rpartition(THINK_CLOSE)
    return head + sep, tail if sep else text


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """(testo senza i blocchi validi, chiamate). Un blocco malformato o non chiuso resta testo."""
    head, tail = split_reasoning(text)
    calls = []

    def take(m):
        call = _valid_call(m.group(1))
        if call is None:
            return m.group(0)
        calls.append(call)
        return ""

    rest = _TOOL_BLOCK.sub(take, tail)
    return (head + rest.strip(), calls) if calls else (text, [])


class ToolCallStream:
    """parse_tool_calls per lo streaming: feed(testo) -> eventi ("text", s) o ("tool_call", call)."""

    def __init__(self):
        self._buf = ""
        self._mode = "text"  # text, think, tool

    def _tag_may_start(self) -> int:
        """Da dove tenere il buffer: la coda potrebbe essere l'inizio di un tag spezzato tra i token."""
        tags = (TOOL_OPEN, THINK_OPEN) if self._mode == "text" else (THINK_CLOSE,)
        for i in range(max(0, len(self._buf) - max(map(len, tags)) + 1), len(self._buf)):
            if any(t.startswith(self._buf[i:]) for t in tags):
                return i
        return len(self._buf)

    def feed(self, text: str) -> list[tuple]:
        self._buf += text
        out = []
        while True:
            if self._mode == "tool":
                end = self._buf.find(TOOL_CLOSE)
                if end < 0:
                    return out
                end += len(TOOL_CLOSE)
                block, self._buf = self._buf[:end], self._buf[end:]
                call = _valid_call(block[len(TOOL_OPEN):-len(TOOL_CLOSE)].strip())
                out.append(("tool_call", call) if call else ("text", block))
                self._mode = "text"
                continue
            tag = (TOOL_OPEN, THINK_OPEN) if self._mode == "text" else (THINK_CLOSE,)
            found = [(self._buf.find(t), t) for t in tag if self._buf.find(t) >= 0]
            if not found:
                keep = self._tag_may_start()
                if keep:
                    out.append(("text", self._buf[:keep]))
                    self._buf = self._buf[keep:]
                return out
            pos, t = min(found)
            if t == TOOL_OPEN:
                if pos:
                    out.append(("text", self._buf[:pos]))
                self._buf = self._buf[pos:]
                self._mode = "tool"
            else:
                out.append(("text", self._buf[:pos + len(t)]))
                self._buf = self._buf[pos + len(t):]
                self._mode = "think" if t == THINK_OPEN else "text"

    def finish(self) -> list[tuple]:
        """A fine generazione ciò che resta è testo: per esempio un <tool_call> mai chiuso."""
        rest, self._buf = self._buf, ""
        return [("text", rest)] if rest else []


@dataclass
class ChatStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # stop o length


class ChatEngine:
    """Modello, contesto e tokenizer di una chat. Una generazione alla volta."""

    def __init__(self, model_path, max_seq: int = 4096, n_threads: int = 0, tokenizer_dir=None, n_batch: int = 64):
        self.model_path = model_path
        self.model = Model(model_path)
        self.ctx = Context(self.model, max_seq=max_seq, n_batch=n_batch, n_threads=n_threads)
        self.tok = BPETokenizer.load(model_path, tokenizer_dir)
        if not self.tok.chat_template:
            raise ValueError("il modello non ha un chat template")
        bos = self.tok.tokens[self.tok.bos_id] if self.tok.bos_id is not None else ""
        eos = self.tok.tokens[self.tok.eos_ids[0]] if self.tok.eos_ids else ""
        self.template = ChatTemplate(self.tok.chat_template, bos, eos)
        self.stats = ChatStats()

    def prompt_ids(self, messages, tools=None, enable_thinking=None) -> list[int]:
        safe = [guard_message(m, self.tok) for m in messages]
        return self.tok.encode(self.template.render(safe, tools, enable_thinking=enable_thinking), guard=GUARD)

    def stream(self, messages, tools=None, sampler=None, max_tokens: int = 1024, enable_thinking=None):
        """Generatore di pezzi di testo. A fine corsa i conteggi sono in self.stats.

        Chi smette di leggere interrompe la generazione: serve al server quando il client si disconnette.
        """
        ids = self.prompt_ids(messages, tools, enable_thinking)
        room = self.ctx.max_seq - len(ids)
        if room < 1:
            raise ValueError(f"prompt di {len(ids)} token oltre il contesto di {self.ctx.max_seq}")
        self.stats = ChatStats(prompt_tokens=len(ids), finish_reason="length")
        sampler = sampler or Sampler()
        history = list(ids)
        decoder = Streamer(self.tok)
        self.ctx.reset()
        logits = self.ctx.eval(ids)
        for i in range(min(max_tokens, room)):
            tid = sampler(logits, history)
            if tid in self.tok.eos_ids:
                self.stats.finish_reason = "stop"
                break
            history.append(tid)
            self.stats.completion_tokens += 1
            text = decoder.text(tid)
            if text:
                yield text
            if i + 1 < min(max_tokens, room):
                logits = self.ctx.eval([tid])
        rest = decoder.flush()
        if rest:
            yield rest
