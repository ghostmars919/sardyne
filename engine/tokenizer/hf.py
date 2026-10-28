"""Tokenizer provvisorio: usa la libreria `tokenizers` di HF su tokenizer.json.

In Fase 7 verrà sostituito da un BPE scritto da noi, letto dai metadati GGUF.
"""
from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer

from engine.loader import hf_config


class Tokenizer:
    def __init__(self, tok: _HFTokenizer, eos_ids):
        self._tok = tok
        self.eos_ids = list(eos_ids)  # tutti i token di fine generazione

    @classmethod
    def from_dir(cls, path) -> "Tokenizer":
        path = Path(path)
        tok = _HFTokenizer.from_file(str(path / "tokenizer.json"))
        return cls(tok, hf_config.eos_ids(path))

    def encode(self, text: str) -> list[int]:
        # Senza token speciali aggiunti automaticamente (BOS ecc.): li gestiremo esplicitamente
        # con i chat template in Fase 7.
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)
