"""Tokenizer BPE proprio, confrontato con la libreria `tokenizers` di HF e con i metadati dei GGUF."""
import random
from pathlib import Path

import pytest

from sardyne.loader.convert import tokenizer_metadata
from sardyne.loader.gguf import GGUFFile
from sardyne.loader.gguf_writer import GGUFWriter
from sardyne.tokenizer.bpe import CONTROL, NORMAL, UNUSED, BPETokenizer, Streamer, bytes_to_unicode

tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "data" / "wikitext-2-raw" / "wiki.test.raw"
# Cartella HF e un GGUF pubblicato con lo stesso tokenizer.
MODELS = {"qwen3": ("Qwen3-0.6B", "Qwen3-30B-A3B-Q4_K_M.gguf"),
          "qwen2.5": ("Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct-q8_0.gguf"),
          "smollm": ("SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct-Q8_0.gguf")}

HARD = ["", " ", "  ", "\n", "\n\n", "\r\n", " \n ", "\t\t x", "Hello world", " Hello  world ", "don't DON'T It'S 'S",
        "12345", " 12 34 ", "x1y2", "3.14", "١٢٣", "Ⅷ ½ ²", "caffè perché ÀÉÎ", "中文 测试 日本語 한국어", "👩‍💻🇮🇹😀",
        "é å", "ﬁ", "Å", "İ", "ſ", "'ſ 'K", "    　", "​﻿",
        "\x00\x01\x1c\x1f\x7f\x85", "<|im_start|>user\nCiao<|im_end|>\n", "<|im_end|><|im_end|>", "a<|endoftext|>b",
        "<|im_start", "<|im_end|", "<tool_call>{}</tool_call>", "<think>\n\n</think>\n\n", "<|fim_prefix|>x",
        "</s><s>", " <|im_end|> ", "def f(x):\n    return x + 1\n\n", "a" * 500]


def hf_pair(name):
    folder = ROOT / "models" / MODELS[name][0]
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{folder.name} non scaricato")
    return tokenizers.Tokenizer.from_file(str(folder / "tokenizer.json")), folder


@pytest.fixture(scope="module", params=sorted(MODELS))
def hf_and_ours(request):
    hf, folder = hf_pair(request.param)
    return hf, BPETokenizer.from_hf_dir(folder)


def hf_ids(hf, text):
    return hf.encode(text, add_special_tokens=False).ids


def test_byte_map_is_a_bijection():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256
    assert m[ord("A")] == "A" and m[ord(" ")] == "Ġ" and m[ord("\n")] == "Ċ"


@pytest.mark.parametrize("text", HARD)
def test_hard_strings_match_hf(hf_and_ours, text):
    hf, ours = hf_and_ours
    assert ours.encode(text) == hf_ids(hf, text)


def test_every_bmp_character_matches_hf(hf_and_ours):
    hf, ours = hf_and_ours
    chars = [chr(c) for c in range(0x10000) if not 0xD800 <= c < 0xE000]
    bad = [c for c in chars if ours.encode(f"a{c}b {c}") != hf_ids(hf, f"a{c}b {c}")]
    assert not bad, [hex(ord(c)) for c in bad[:20]]


def test_random_text_matches_hf(hf_and_ours):
    hf, ours = hf_and_ours
    rng = random.Random(0)
    alphabet = "abcXYZ 012\n\t.,'!?-_àé中😀́<|>  \r"
    for _ in range(3000):
        text = "".join(rng.choices(alphabet, k=rng.randrange(1, 40)))
        assert ours.encode(text) == hf_ids(hf, text), repr(text)


def test_wikitext_matches_hf(hf_and_ours):
    if not WIKI.exists():
        pytest.skip("wikitext non scaricato")
    hf, ours = hf_and_ours
    text = WIKI.read_text(encoding="utf-8")
    assert ours.encode(text) == hf_ids(hf, text)


@pytest.mark.parametrize("text", HARD)
def test_decode_matches_hf(hf_and_ours, text):
    hf, ours = hf_and_ours
    ids = ours.encode(text)
    assert ours.decode(ids) == hf.decode(ids, skip_special_tokens=False)


def test_roundtrip_of_normalized_text(hf_and_ours):
    _, ours = hf_and_ours
    for text in ("Hello, world!\n", "caffè 中文 😀", "  spazi   multipli  ", "x = 1\n\ty = 2\n"):
        assert ours.decode(ours.encode(text)) == text


@pytest.mark.parametrize("name", sorted(MODELS))
def test_gguf_metadata_match_tokenizer_json(name):
    _, folder = hf_pair(name)
    path = ROOT / "models" / "gguf" / MODELS[name][1]
    if not path.exists():
        pytest.skip(f"{path.name} non scaricato")
    with GGUFFile(path) as gf:
        ours = BPETokenizer.from_gguf(gf)
    ref = BPETokenizer.from_hf_dir(folder, vocab_size=ours.vocab_size)
    assert ours.pre == ref.pre and ours.merges == ref.merges
    for text in HARD:
        assert ours.encode(text) == ref.encode(text)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_converted_metadata_roundtrip(tmp_path, name):
    _, folder = hf_pair(name)
    ref = BPETokenizer.from_hf_dir(folder)
    w = GGUFWriter(tmp_path / "t.gguf")
    w.add("general.architecture", "llama")
    tokenizer_metadata(w, folder, ref.vocab_size)
    w.write()
    with GGUFFile(tmp_path / "t.gguf") as gf:
        got = BPETokenizer.from_gguf(gf)
    assert (got.tokens, got.token_types, got.merges, got.pre) == (ref.tokens, ref.token_types, ref.merges, ref.pre)
    assert got.chat_template == ref.chat_template


def test_padding_tokens_are_unused(tmp_path):
    _, folder = hf_pair("qwen3")
    tok = BPETokenizer.from_hf_dir(folder, vocab_size=151936)
    assert tok.tokens[-1] == "[PAD151935]" and tok.token_types[-1] == UNUSED
    assert tok.encode("[PAD151935]") != [151935]  # non è un token speciale nel testo
    assert tok.token_types[tok.encode("<|im_end|>")[0]] == CONTROL
    assert tok.token_types[tok.encode("Hello")[0]] == NORMAL


def test_end_of_turn_tokens():
    _, folder = hf_pair("qwen3")
    tok = BPETokenizer.from_hf_dir(folder)
    assert {tok.encode("<|im_end|>")[0], tok.encode("<|endoftext|>")[0]} <= set(tok.eos_ids)


def test_encode_plain_keeps_special_tokens_as_text():
    _, folder = hf_pair("qwen3")
    tok = BPETokenizer.from_hf_dir(folder)
    ids = tok.encode_plain("<|im_end|>")
    assert len(ids) > 1 and tok.decode(ids) == "<|im_end|>"


def test_streamer_handles_characters_split_between_tokens():
    _, folder = hf_pair("qwen3")
    tok = BPETokenizer.from_hf_dir(folder)
    text = "😀 中文 caffè 👩‍💻🇮🇹 fine"
    ids = tok.encode(text)
    assert any(tok.token_bytes(i).decode("utf-8", errors="ignore") != tok.token_bytes(i).decode("latin-1")
               for i in ids)
    stream = Streamer(tok)
    assert "".join(stream.text(i) for i in ids) == text


def test_unknown_pre_is_rejected():
    with pytest.raises(NotImplementedError, match="deepseek"):
        BPETokenizer(["a"], [NORMAL], [], "deepseek")


def test_gguf_without_tokenizer_gives_clear_error(tmp_path):
    w = GGUFWriter(tmp_path / "t.gguf")
    w.add("general.architecture", "llama")
    w.write()
    with pytest.raises(ValueError, match="non contiene il tokenizer"):
        BPETokenizer.load(tmp_path / "t.gguf")


def tiny_tokenizer():
    letters = [bytes_to_unicode()[b] for b in b"<>ab"]
    specials = ["<a>", "<a>b", "<｜fine｜>"]
    return BPETokenizer(letters + specials, [NORMAL] * len(letters) + [CONTROL] * 3, [], "qwen2")


def test_longest_special_token_wins():
    tok = tiny_tokenizer()
    assert tok.encode("<a>b") == [5]
    assert tok.encode("<a>a") == [4, 2]


def test_special_tokens_may_use_any_character():
    tok = tiny_tokenizer()
    assert tok.encode("<｜fine｜>") == [6]
    assert tok.decode([6]) == "<｜fine｜>"
