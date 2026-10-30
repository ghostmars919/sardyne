"""Tokenizer BPE byte-level (GPT-2), letto dai metadati GGUF o da tokenizer.json.

Passi di encode, nello stesso ordine di `tokenizers` di HF:
1. i token speciali si cercano nel testo grezzo e dividono il testo in parti;
2. Qwen normalizza ogni parte in NFC (SmolLM no);
3. una regex divide la parte in pezzi; SmolLM isola prima ogni cifra;
4. ogni pezzo diventa byte, ogni byte un carattere stampabile (mappa di GPT-2), e le coppie di simboli
   adiacenti si fondono in ordine di priorità finché nessuna coppia è tra i merges.

La normalizzazione e la divisione non sono nei metadati GGUF: dipendono da `tokenizer.ggml.pre`.

Un token può contenere solo una parte di un carattere UTF-8: per lo streaming si usa Streamer, che
lavora sui byte.
"""
import codecs
import json
import unicodedata
from pathlib import Path

import regex

from engine.loader import hf_config
from engine.loader.gguf import GGUFFile

# Tipi dei token nei GGUF (enum llama_token_type).
NORMAL, CONTROL, USER_DEFINED, UNUSED = 1, 3, 4, 5

_QWEN2 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
          r"|\s*[\r\n]+|\s+(?!\S)|\s+")
_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

# pre -> (regex, normalizzazione NFC, cifre isolate prima della regex)
PRE = {"qwen2": (regex.compile(_QWEN2), True, False), "smollm": (regex.compile(_GPT2), False, True)}
_DIGITS = regex.compile(r"\p{N}|\P{N}+")

# Token che chiudono un turno oltre a quello indicato nei metadati, come fa llama.cpp.
_END_OF_TURN = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|eom_id|>")


def bytes_to_unicode() -> dict:
    """Byte -> carattere: quelli stampabili restano uguali, gli altri vanno oltre U+00FF (GPT-2)."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, extra = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars)))


class BPETokenizer:
    def __init__(self, tokens, token_types, merges, pre, eos_ids=(), chat_template=None, bos_id=None):
        if pre not in PRE:
            raise NotImplementedError(f"tokenizer '{pre}' non supportato (supportati: {sorted(PRE)})")
        if len(tokens) != len(token_types):
            raise ValueError(f"{len(tokens)} token ma {len(token_types)} tipi")
        self.tokens = list(tokens)
        self.token_types = [int(t) for t in token_types]
        self.merges = list(merges)
        self.pre = pre
        self.chat_template = chat_template
        self.bos_id = bos_id
        self._split, self._nfc, self._digits = PRE[pre]
        self._ids = {t: i for i, t in enumerate(self.tokens)}
        self._ranks = {tuple(m.split(" ", 1)): r for r, m in enumerate(self.merges)}
        self._byte_char = bytes_to_unicode()
        self._char_byte = {c: b for b, c in self._byte_char.items()}
        self._cache = {}

        special = [t for t, k in zip(self.tokens, self.token_types) if k in (CONTROL, USER_DEFINED)]
        # In un'alternativa vince la prima: i più lunghi prima, come il leftmost-longest di HF.
        self._special = regex.compile("|".join(regex.escape(t) for t in sorted(special, key=len, reverse=True))) \
            if special else None
        ends = [self._ids[t] for t in _END_OF_TURN if t in self._ids and self._is_special(self._ids[t])]
        self.eos_ids = sorted({int(e) for e in eos_ids} | set(ends))

    def _is_special(self, tid: int) -> bool:
        return self.token_types[tid] != NORMAL

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_gguf(cls, gf: GGUFFile) -> "BPETokenizer":
        m = gf.metadata
        kind = m.get("tokenizer.ggml.model")
        if "tokenizer.ggml.tokens" not in m:
            raise ValueError(f"{gf.path.name} non contiene il tokenizer: indicare la cartella HF")
        if kind != "gpt2":
            raise NotImplementedError(f"tokenizer '{kind}' non supportato (supportato: gpt2)")
        eos = m.get("tokenizer.ggml.eos_token_id")
        return cls(m["tokenizer.ggml.tokens"], m["tokenizer.ggml.token_type"], m["tokenizer.ggml.merges"],
                   m.get("tokenizer.ggml.pre"), [] if eos is None else [eos], m.get("tokenizer.chat_template"),
                   m.get("tokenizer.ggml.bos_token_id"))

    @classmethod
    def from_hf_dir(cls, path, vocab_size: int | None = None) -> "BPETokenizer":
        """Da tokenizer.json. vocab_size: dimensione del modello, se maggiore del vocabolario (gli id
        in più diventano [PADn], come nei GGUF)."""
        path = Path(path)
        t = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        if t["model"]["type"] != "BPE":
            raise NotImplementedError(f"tokenizer {t['model']['type']} non supportato")
        id_type = {i: NORMAL for i in t["model"]["vocab"].values()}
        id_text = {i: s for s, i in t["model"]["vocab"].items()}
        for a in t["added_tokens"]:
            id_text[a["id"]] = a["content"]
            id_type[a["id"]] = CONTROL if a["special"] else USER_DEFINED
        n = max(vocab_size or 0, max(id_text) + 1)
        tokens = [id_text.get(i, f"[PAD{i}]") for i in range(n)]
        types = [id_type.get(i, UNUSED) for i in range(n)]
        merges = [m if isinstance(m, str) else " ".join(m) for m in t["model"]["merges"]]

        cfg_path = path / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        return cls(tokens, types, merges, _hf_pre(t), hf_config.eos_ids(path), cfg.get("chat_template"))

    @classmethod
    def load(cls, model=None, hf_dir=None) -> "BPETokenizer":
        """Dalla cartella HF, se indicata, altrimenti dai metadati del file .gguf."""
        if hf_dir:
            return cls.from_hf_dir(hf_dir)
        with GGUFFile(model) as gf:
            return cls.from_gguf(gf)

    def _bpe(self, piece: str) -> list[int]:
        out = self._cache.get(piece)
        if out is not None:
            return out
        # Il vocabolario di SmolLM non ha tutti i byte: come in HF, quelli assenti si scartano subito.
        word = [c for c in map(self._byte_char.get, piece.encode("utf-8")) if c in self._ids]
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self._ranks.get(p, float("inf")))
            if best not in self._ranks:
                break
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        out = self._cache[piece] = [self._ids[s] for s in word]
        return out

    def _encode_part(self, text: str, out: list):
        if self._nfc:
            text = unicodedata.normalize("NFC", text)
        for part in (_DIGITS.findall(text) if self._digits else (text,)):
            for piece in self._split.findall(part):
                out.extend(self._bpe(piece))

    def encode(self, text: str, guard: str | None = None) -> list[int]:
        """Riconosce i token speciali scritti nel testo. guard: il carattere inserito da guard_special,
        che impedisce il riconoscimento e non finisce nel risultato."""
        out, pos = [], 0

        def part(s):
            self._encode_part(s.replace(guard, "") if guard else s, out)

        if self._special is not None:
            for m in self._special.finditer(text):
                part(text[pos:m.start()])
                out.append(self._ids[m.group()])
                pos = m.end()
        part(text[pos:])
        return out

    def guard_special(self, text: str, guard: str) -> str:
        """Inserisce guard dentro ogni token speciale del testo: per encode resta testo normale."""
        return self._special.sub(lambda m: m.group()[0] + guard + m.group()[1:], text) if self._special else text

    def encode_plain(self, text: str) -> list[int]:
        """Come encode, ma i token speciali scritti nel testo restano testo normale."""
        out = []
        self._encode_part(text, out)
        return out

    def token_bytes(self, tid: int) -> bytes:
        s = self.tokens[tid]
        return s.encode("utf-8") if self._is_special(tid) else bytes(self._char_byte[c] for c in s)

    def decode(self, ids) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")


def _hf_pre(t: dict) -> str:
    """Il valore di tokenizer.ggml.pre per un tokenizer.json che sappiamo riprodurre."""
    steps = (t.get("pre_tokenizer") or {}).get("pretokenizers", [])
    kinds = [s["type"] for s in steps]
    norm = (t.get("normalizer") or {}).get("type")
    if kinds == ["Split", "ByteLevel"] and steps[0]["pattern"].get("Regex") == _QWEN2 and norm == "NFC":
        return "qwen2"
    if kinds == ["Digits", "ByteLevel"] and steps[0].get("individual_digits") and norm is None:
        return "smollm"
    raise NotImplementedError(f"pre-tokenizer non supportato: {kinds}, normalizer {norm}")


class Streamer:
    """Stampa i token man mano che arrivano. I byte di un carattere diviso tra due token restano in
    attesa nel decoder incrementale."""

    def __init__(self, tok: BPETokenizer):
        self.tok = tok
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def text(self, tid: int) -> str:
        return self._dec.decode(self.tok.token_bytes(tid))

    def flush(self) -> str:
        """I byte rimasti a metà di un carattere, a fine generazione."""
        return self._dec.decode(b"", final=True)

    def __call__(self, tid: int):
        print(self.text(tid), end="", flush=True)
