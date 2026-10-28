import gc
import struct
from pathlib import Path

import numpy as np
import pytest

from engine import capi
from engine.loader.convert import convert, permute_for_interleaved_rope
from engine.loader.gguf import GGUFError, GGUFFile
from engine.loader.gguf_writer import GGUFWriter, f32_to_bf16
from engine.ref import ops
from engine.ref.llama import Llama

ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = ROOT / "models" / "gguf"
DOWNLOADED = sorted(p for p in GGUF_DIR.glob("*.gguf")) if GGUF_DIR.exists() else []
needs_c = pytest.mark.skipif(not capi.available(), reason="libengine non compilata: eseguire make")


def CGGUFFile(path):
    return capi.CGGUFFile(path)


def model_dir(name):
    path = ROOT / "models" / name
    if not (path / "config.json").exists():
        pytest.skip(f"modello {name} non scaricato")
    return path


# ---------------------------------------------------------------- file sintetico

@pytest.fixture
def small_gguf(tmp_path):
    """File con un valore di ogni tipo di metadato e tensori F32/F16/BF16."""
    rng = np.random.default_rng(0)
    w = GGUFWriter(tmp_path / "small.gguf")
    w.add("general.architecture", "test")
    w.add("u8", 200, "UINT8")
    w.add("i8", -100, "INT8")
    w.add("u16", 60000, "UINT16")
    w.add("i16", -30000, "INT16")
    w.add("u32", 4_000_000_000, "UINT32")
    w.add("i32", -2_000_000_000, "INT32")
    w.add("f32", 0.5, "FLOAT32")
    w.add("flag", True)
    w.add("u64", 2 ** 63 + 5, "UINT64")
    w.add("i64", -(2 ** 40), "INT64")
    w.add("f64", 3.141592653589793, "FLOAT64")
    w.add("name", "càffè ✓")
    w.add("words", ["", "a", "ciao", "Ġthe"])
    w.add("ints", np.arange(5, dtype=np.int32))
    tensors = {
        "a": rng.standard_normal((3, 5)).astype(np.float32),
        "b": rng.standard_normal((4, 2, 8)).astype(np.float16),
        "c": f32_to_bf16(rng.standard_normal(7)),
    }
    w.add_tensor("a", tensors["a"], "F32")
    w.add_tensor("b", tensors["b"], "F16")
    w.add_tensor("c", tensors["c"], "BF16")
    w.write()
    return w.path, tensors


def test_roundtrip_python(small_gguf):
    path, tensors = small_gguf
    f = GGUFFile(path)
    m = f.metadata
    assert (m["u8"], m["i8"], m["u16"], m["i16"]) == (200, -100, 60000, -30000)
    assert (m["u32"], m["i32"], m["u64"], m["i64"]) == (4_000_000_000, -2_000_000_000, 2 ** 63 + 5, -(2 ** 40))
    assert m["f32"] == 0.5 and m["f64"] == 3.141592653589793 and m["flag"] is True
    assert m["name"] == "càffè ✓" and m["words"] == ["", "a", "ciao", "Ġthe"]
    np.testing.assert_array_equal(m["ints"], np.arange(5))
    assert f.tensors["b"].ne == (8, 2, 4) and f.tensors["b"].shape == (4, 2, 8)
    for name, data in tensors.items():
        np.testing.assert_array_equal(f.raw(name), data)
    for t in f.tensors.values():
        assert t.offset % f.alignment == 0


@needs_c
def test_roundtrip_c(small_gguf):
    path, tensors = small_gguf
    c = CGGUFFile(path)
    assert c.get_u64("u8") == 200 and c.get_u64("u64") == 2 ** 63 + 5
    with pytest.raises(TypeError):
        c.get_u64("i8")  # negativo: il getter accetta solo interi >= 0
    assert c.get_f64("f32") == 0.5 and c.get_f64("f64") == 3.141592653589793
    assert c.get_str("name") == "càffè ✓"
    assert [c.get_arr_str("words", j) for j in range(4)] == ["", "a", "ciao", "Ġthe"]
    with pytest.raises(IndexError):
        c.get_arr_str("words", 4)
    with pytest.raises(KeyError):
        c.get_str("inesistente")
    for name, data in tensors.items():
        np.testing.assert_array_equal(c.tensor_bytes(name), np.frombuffer(data.tobytes(), np.uint8))


@needs_c
def test_custom_alignment(tmp_path):
    w = GGUFWriter(tmp_path / "al.gguf", alignment=64)
    w.add_tensor("x", np.ones(3, np.float32), "F32")
    w.add_tensor("y", np.ones(5, np.float32), "F32")
    w.write()
    f, c = GGUFFile(w.path), CGGUFFile(w.path)
    assert f.alignment == c.alignment == 64
    assert f.data_offset % 64 == 0 and f.tensors["y"].offset == 64
    np.testing.assert_array_equal(f.raw("y"), np.ones(5))


def test_bf16_rounding():
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5], np.float32)  # 1+2^-8 e 1+3·2^-8: casi di pareggio
    bits = f32_to_bf16(x)
    back = (bits.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(back, [1.0, 1.0, 1.015625, -2.5])  # pareggio -> mantissa pari


# ---------------------------------------------------------------- file malformati: errori, mai crash

def _both_parse(path):
    """(esito Python, esito C): riassunto del file oppure None se il file viene rifiutato.

    Il riassunto contiene allineamento, zona dati, chiavi con i loro tipi e i tensori con nome,
    dimensioni, tipo, offset e byte: tutto ciò che i due parser devono leggere allo stesso modo.
    """
    try:
        f = GGUFFile(path)
        py = (f.alignment, f.data_offset,
              [(k, f.metadata_types[k][0]) for k in f.metadata],
              [(t.name, t.ne, t.type_name, t.offset, t.nbytes) for t in f.tensors.values()])
    except GGUFError:
        py = None
    try:
        with CGGUFFile(path) as c:
            cc = (c.alignment, c.data_offset, [(k, c.kv_type(k)[0]) for k in c.keys()],
                  [(t["name"], t["ne"], t["type"], t["offset"], t["nbytes"]) for t in c.tensors()])
    except ValueError:
        cc = None
    return py, cc


@needs_c
def test_truncated_files_are_rejected(small_gguf, tmp_path):
    path, _ = small_gguf
    data = path.read_bytes()
    f = GGUFFile(path)
    last = max(f.tensors.values(), key=lambda t: t.offset)
    data_end = f.data_offset + last.offset + last.nbytes  # dopo c'è solo il padding finale
    bad = tmp_path / "trunc.gguf"
    for n in list(range(0, f.data_offset, 7)) + [data_end - 1]:
        bad.write_bytes(data[:n])
        py, c = _both_parse(bad)
        assert py is None and c is None, f"troncato a {n} byte ma accettato"


@needs_c
def test_corrupted_files_never_crash(small_gguf, tmp_path):
    # Fuzzing: byte casuali alterati nell'header. Un crash del C ucciderebbe il processo di pytest.
    path, _ = small_gguf
    data = bytearray(path.read_bytes())
    header_end = GGUFFile(path).data_offset
    rng = np.random.default_rng(42)
    bad = tmp_path / "fuzz.gguf"
    for _ in range(1500):
        mutated = bytearray(data)
        for pos in rng.integers(4, header_end, size=rng.integers(1, 4)):
            mutated[pos] = rng.integers(0, 256)
        bad.write_bytes(bytes(mutated))
        py, c = _both_parse(bad)
        # Un byte alterato può anche produrre un file ancora valido: allora i due parser devono concordare.
        assert py == c, "i due parser leggono il file in modo diverso"


@needs_c
def test_bad_magic_and_version(small_gguf, tmp_path):
    path, _ = small_gguf
    data = bytearray(path.read_bytes())
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUX" + data[4:])
    assert _both_parse(bad) == (None, None)
    data[4] = 99  # versione 99
    bad.write_bytes(bytes(data))
    assert _both_parse(bad) == (None, None)


# ---------------------------------------------------------------- file scaricati da HF

@pytest.mark.parametrize("path", DOWNLOADED, ids=lambda p: p.name)
@needs_c
def test_python_and_c_parsers_agree(path):
    py, c = GGUFFile(path), CGGUFFile(path)
    assert (py.version, py.alignment, py.data_offset) == (c.version, c.alignment, c.data_offset)
    assert list(py.metadata) == c.keys()
    for key, value in py.metadata.items():
        vtype, etype = py.metadata_types[key]
        if vtype == 8:
            assert c.get_str(key) == value
        elif vtype in (6, 12):
            assert c.get_f64(key) == pytest.approx(value, rel=1e-7)
        elif vtype in (0, 2, 4, 10) or (vtype in (1, 3, 5, 11) and value >= 0):
            assert c.get_u64(key) == value
        elif vtype == 9 and etype == 8:
            assert c.kv_type(key)[2] == len(value)
            for j in (0, len(value) // 2, len(value) - 1):
                assert c.get_arr_str(key, j) == value[j]

    ct = c.tensors()
    assert [t["name"] for t in ct] == list(py.tensors)
    for t in ct:
        p = py.tensors[t["name"]]
        assert (t["ne"], t["type"], t["offset"], t["nbytes"]) == (p.ne, p.type_name, p.offset, p.nbytes)
    for name in list(py.tensors)[:3] + list(py.tensors)[-3:]:
        np.testing.assert_array_equal(c.tensor_bytes(name), py.raw(name).reshape(-1).view(np.uint8))


@pytest.mark.parametrize("path", DOWNLOADED, ids=lambda p: p.name)
def test_tensor_layout_is_contiguous(path):
    # Ordinati per offset, i tensori non si sovrappongono e tra uno e l'altro c'è solo il padding:
    # verifica la tabella delle dimensioni dei blocchi per i tipi presenti nel file.
    f = GGUFFile(path)
    ts = sorted(f.tensors.values(), key=lambda t: t.offset)
    for a, b in zip(ts, ts[1:]):
        gap = b.offset - (a.offset + a.nbytes)
        assert 0 <= gap < f.alignment, f"{a.name} -> {b.name}: gap {gap}"
    end = f.data_offset + ts[-1].offset + ts[-1].nbytes
    assert 0 <= f.path.stat().st_size - end < f.alignment


# ---------------------------------------------------------------- conversione e modello

def test_permutation_matches_interleaved_rope():
    # W_q permutata + RoPE a coppie adiacenti == W_q originale + RoPE "rotate_half", a meno
    # della stessa permutazione su q e k, che lascia invariati i prodotti scalari q·k.
    rng = np.random.default_rng(7)
    n_head, hd, dim, t = 3, 8, 12, 5
    wq = rng.standard_normal((n_head * hd, dim)).astype(np.float32)
    x = rng.standard_normal((t, dim)).astype(np.float32)
    cos, sin = ops.rope_cos_sin(t, hd, 10000.0)
    q_hf = ops.apply_rope(ops.split_heads(x @ wq.T, n_head), cos, sin)
    q_gg = ops.apply_rope_interleaved(ops.split_heads(x @ permute_for_interleaved_rope(wq, n_head).T, n_head), cos, sin)
    np.testing.assert_allclose(q_hf @ q_hf.transpose(0, 2, 1), q_gg @ q_gg.transpose(0, 2, 1), rtol=1e-4, atol=1e-4)


def test_converter_matches_llama_cpp_bit_for_bit(tmp_path):
    """Il nostro convertitore produce gli stessi byte del file f16 pubblicato (fatto con llama.cpp)."""
    ref_path = GGUF_DIR / "SmolLM2-135M-Instruct-f16.gguf"
    if not ref_path.exists():
        pytest.skip("GGUF di riferimento non scaricato")
    out = tmp_path / "smol-f16.gguf"
    convert(model_dir("SmolLM2-135M-Instruct"), out, "f16")
    ours, ref = GGUFFile(out), GGUFFile(ref_path)
    assert set(ours.tensors) == set(ref.tensors)
    for name, t in ref.tensors.items():
        assert ours.tensors[name].ne == t.ne and ours.tensors[name].ggml_type == t.ggml_type, name
        assert np.array_equal(ours.raw(name).view(np.uint8), ref.raw(name).view(np.uint8)), name
    for key in ("llama.block_count", "llama.context_length", "llama.embedding_length",
                "llama.feed_forward_length", "llama.attention.head_count", "llama.attention.head_count_kv",
                "llama.rope.freq_base", "llama.rope.dimension_count", "llama.attention.layer_norm_rms_epsilon",
                "llama.vocab_size", "general.file_type", "general.architecture",
                "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        assert ours.metadata[key] == ref.metadata[key], key


PROMPT = "In 1905 Albert Einstein published four papers that changed physics. The first one"


@pytest.mark.parametrize("name", ["SmolLM2-135M", "Qwen2.5-0.5B"])
def test_gguf_model_matches_safetensors(name, tmp_path):
    """bf16 -> GGUF bf16 è senza perdite: stessi logits della Fase 1 (llama: con pesi permutati)."""
    from engine.tokenizer.hf import Tokenizer

    hf = model_dir(name)
    ids = Tokenizer.from_dir(hf).encode(PROMPT)
    out = tmp_path / f"{name}-bf16.gguf"
    convert(hf, out, "bf16")

    a = Llama.from_dir(hf, max_seq=64)
    la = a.forward(ids)
    del a
    gc.collect()
    b = Llama.from_gguf(out, max_seq=64)
    assert b.cfg.rope_interleaved == (name == "SmolLM2-135M")
    lb = b.forward(ids)
    np.testing.assert_allclose(lb, la, atol=1e-4, rtol=0)


def test_published_gguf_matches_hf():
    """Il file f16 di bartowski (llama.cpp) dà la stessa generazione greedy dei pesi HF."""
    path = GGUF_DIR / "SmolLM2-135M-Instruct-f16.gguf"
    if not path.exists():
        pytest.skip("GGUF non scaricato")
    from engine.tokenizer.hf import Tokenizer

    hf = model_dir("SmolLM2-135M-Instruct")
    ids = Tokenizer.from_dir(hf).encode(PROMPT)
    a, b = Llama.from_dir(hf, max_seq=96), Llama.from_gguf(path, max_seq=96)
    np.testing.assert_allclose(b.forward(ids), a.forward(ids), atol=1e-3, rtol=0)  # pesi arrotondati a f16
    assert b.generate(ids, 30) == a.generate(ids, 30)


def test_quantized_tensors_not_yet_supported():
    path = GGUF_DIR / "SmolLM2-135M-Instruct-Q8_0.gguf"
    if not path.exists():
        pytest.skip("GGUF Q8_0 non scaricato")
    f = GGUFFile(path)
    assert f.tensors["blk.0.attn_q.weight"].type_name == "Q8_0"
    assert f.raw("blk.0.attn_q.weight").shape == (576, 576 // 32 * 34)
    with pytest.raises(NotImplementedError):
        f.get_f32("blk.0.attn_q.weight")


# ---------------------------------------------------------------- regole di rifiuto (identiche nei due parser)

def _raw_gguf(path, kvs=(), tensors=(), version=3):
    """Scrive a mano un GGUF, anche non valido. kvs: [(chiave, tipo, valore)] in byte;
    tensors: [(nome, ne, tipo ggml, offset)] con 64 byte di dati a zero per ciascuno."""
    def s(b):
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs))
    for k, t, v in kvs:
        out += s(k) + struct.pack("<I", t) + v
    for n, ne, t, off in tensors:
        out += s(n) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne) + struct.pack("<IQ", t, off)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(out + b"\0" * 64 * max(1, len(tensors)))
    return path


U32 = 4


@needs_c
@pytest.mark.parametrize("case", [
    "valido", "nul_nella_chiave", "chiave_duplicata", "utf8_non_valido_chiave", "utf8_sovralungo",
    "surrogato", "tensore_duplicato", "nul_nel_nome", "dimensione_nulla", "big_endian"])
def test_rejection_rules_are_the_same(case, tmp_path):
    kv = [(b"general.architecture", 8, struct.pack("<Q", 4) + b"test")]
    t = [(b"a", (4,), 0, 0)]
    if case == "nul_nella_chiave":
        kv.append((b"general.alignment\0x", U32, struct.pack("<I", 64)))
    elif case == "chiave_duplicata":
        kv += [(b"k", U32, struct.pack("<I", 1)), (b"k", U32, struct.pack("<I", 2))]
    elif case == "utf8_non_valido_chiave":
        kv.append((b"caf\xe9", U32, struct.pack("<I", 1)))
    elif case == "utf8_sovralungo":
        kv.append((b"\xc0\xaf", U32, struct.pack("<I", 1)))  # "/" codificato in 2 byte
    elif case == "surrogato":
        kv.append((b"\xed\xa0\x80", U32, struct.pack("<I", 1)))  # U+D800
    elif case == "tensore_duplicato":
        t.append((b"a", (4,), 0, 32))
    elif case == "nul_nel_nome":
        t = [(b"a\0b", (4,), 0, 0)]
    elif case == "dimensione_nulla":
        t = [(b"a", (4, 0), 0, 0)]
    path = _raw_gguf(tmp_path / f"{case}.gguf", kv, t, version=0x03000000 if case == "big_endian" else 3)
    py, c = _both_parse(path)
    assert py == c
    assert (py is not None) == (case == "valido")


@needs_c
def test_error_messages(tmp_path):
    path = _raw_gguf(tmp_path / "be.gguf", version=0x03000000)
    with pytest.raises(GGUFError, match="big-endian"):
        GGUFFile(path)
    with pytest.raises(ValueError, match="big-endian"):
        CGGUFFile(path)
    with pytest.raises(ValueError, match="No such file|impossibile aprire"):
        CGGUFFile(tmp_path / "inesistente.gguf")


# ---------------------------------------------------------------- binding C: ciclo di vita

@needs_c
def test_c_binding_lifecycle(small_gguf):
    path, tensors = small_gguf
    # Oggetto temporaneo: i byte restituiti sono una copia e restano validi.
    data = CGGUFFile(path).tensor_bytes("a")
    gc.collect()
    np.testing.assert_array_equal(data, np.frombuffer(tensors["a"].tobytes(), np.uint8))
    # Dopo close() ogni metodo solleva un errore invece di passare NULL al C.
    c = CGGUFFile(path)
    view = c.tensor_bytes("b")
    c.close()
    c.close()  # idempotente
    for call in (c.keys, c.tensors, lambda: c.get_u64("u8"), lambda: c.tensor_bytes("a")):
        with pytest.raises(ValueError, match="chiuso"):
            call()
    assert view.sum() == np.frombuffer(tensors["b"].tobytes(), np.uint8).sum()
    with CGGUFFile(path) as c2:
        assert c2.get_str("name") == "càffè ✓"
    with pytest.raises(ValueError):
        c2.keys()


def test_python_file_close(small_gguf):
    path, tensors = small_gguf
    with GGUFFile(path) as f:
        view = f.raw("a")
    np.testing.assert_array_equal(view, tensors["a"])  # la vista tiene in vita la mmap
    with pytest.raises(ValueError, match="chiuso"):
        f.raw("a")


# ---------------------------------------------------------------- writer

def test_writer_rejects_invalid_input(tmp_path):
    with pytest.raises(ValueError):
        GGUFWriter(tmp_path / "x.gguf", alignment=24)
    w = GGUFWriter(tmp_path / "x.gguf")
    w.add("k", 1)
    for bad in [lambda: w.add("k", 2),                      # chiave duplicata
                lambda: w.add("a\0b", 1),                   # NUL nella chiave
                lambda: w.add("big", 2 ** 64),              # fuori da UINT64
                lambda: w.add("u8", 256, "UINT8"),
                lambda: w.add("mix", [1, "a"]),             # array eterogeneo
                lambda: w.add("nested", [[1, 2], [3]]),
                lambda: w.add("f16", np.ones(3, np.float16)),
                lambda: w.add("obj", object())]:
        with pytest.raises((ValueError, TypeError)):
            bad()
    w.add("np_scalar", np.float32(1.5))  # gli scalari NumPy sono accettati
    w.add("wide", [1, 2 ** 33])  # interi misti -> UINT64
    w.add_tensor("t", np.ones(4, np.float32), "F32")
    with pytest.raises(ValueError):
        w.add_tensor("t", np.ones(4, np.float32), "F32")
    with pytest.raises(ValueError):
        w.add_raw_tensor("q", (32,), "Q8_0", b"\0" * 33)  # Q8_0: 34 byte per 32 valori
    w.write()
    f = GGUFFile(w.path)
    assert f.metadata["np_scalar"] == 1.5 and list(f.metadata["wide"]) == [1, 2 ** 33]
    assert f.metadata_types["wide"] == (9, 10)


def test_writer_is_atomic(tmp_path):
    path = tmp_path / "m.gguf"
    w = GGUFWriter(path)
    w.add_tensor("t", np.arange(8192, dtype=np.float32), "F32")
    w.write()
    with GGUFFile(path) as old:
        view = old.raw("t")
        # Riscrittura dello stesso percorso mentre il vecchio file è mappato: niente SIGBUS,
        # la vista continua a vedere i dati vecchi (nuovo inode + rename).
        w2 = GGUFWriter(path)
        w2.add_tensor("t", np.zeros(4, np.float32), "F32")
        w2.write()
        assert view[-1] == 8191
    assert GGUFFile(path).tensors["t"].ne == (4,)

    # Un errore durante la scrittura lascia intatto il file esistente e nessun .tmp.
    w3 = GGUFWriter(path)
    w3.add_lazy_tensor("t", (4,), "F32", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        w3.write()
    assert GGUFFile(path).tensors["t"].ne == (4,)
    assert not list(tmp_path.glob("*.tmp"))


def test_bf16_special_values():
    x = np.array([np.inf, -np.inf, np.nan, -np.nan, np.finfo(np.float32).max, 0.0, -0.0], np.float32)
    x = np.concatenate([x, np.array([0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001], np.uint32).view(np.float32)])
    back = (f32_to_bf16(x).astype(np.uint32) << 16).view(np.float32)
    assert back[0] == np.inf and back[1] == -np.inf
    assert np.isnan(back[[2, 3, 7, 8, 9]]).all()  # i NaN restano NaN (anche con mantissa piena)
    assert np.signbit(back[8]) and not np.signbit(back[7])
    assert back[4] == np.inf  # FLT_MAX arrotonda a inf: corretto in bf16
    assert back[5] == 0 and np.signbit(back[6])
