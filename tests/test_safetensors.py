import json
import struct

import numpy as np
import pytest

from engine.loader.safetensors import SafeTensorsDir, SafeTensorsError, SafeTensorsFile


def _write(path, header: dict, data: bytes = b"", header_len: int | None = None):
    h = json.dumps(header).encode()
    n = len(h) if header_len is None else header_len
    path.write_bytes(struct.pack("<Q", n) + h + data)
    return path


def test_valid_file_and_copy_semantics(tmp_path):
    a = np.arange(6, dtype=np.float32)
    p = _write(tmp_path / "ok.safetensors", {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}},
               a.tobytes())
    f = SafeTensorsFile(p)
    out = f.get_f32("a")
    np.testing.assert_array_equal(out, a.reshape(2, 3))
    out[0, 0] = 99  # get_f32 restituisce sempre una copia scrivibile
    assert f.get_f32("a")[0, 0] == 0
    assert f.shape("a") == (2, 3)


@pytest.mark.parametrize("header,data,why", [
    ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]}}, b"\0" * 8, "offset negativo"),
    ({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b"\0" * 8, "oltre il file"),
    ({"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, b"\0" * 8, "shape incoerente"),
    ({"a": {"dtype": "F8_E4M3", "shape": [8], "data_offsets": [0, 8]}}, b"\0" * 8, "dtype sconosciuto"),
    ({"a": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}}, b"\0" * 8, "shape negativa"),
    ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0]}}, b"\0" * 8, "offset incompleti"),
    ({"a": "x"}, b"", "voce non oggetto"),
])
def test_malformed_headers_are_rejected(tmp_path, header, data, why):
    with pytest.raises(SafeTensorsError):
        SafeTensorsFile(_write(tmp_path / "bad.safetensors", header, data))


def test_bad_lengths_and_json(tmp_path):
    (tmp_path / "short.safetensors").write_bytes(b"\1\2\3")
    with pytest.raises(SafeTensorsError):
        SafeTensorsFile(tmp_path / "short.safetensors")
    with pytest.raises(SafeTensorsError):  # lunghezza enorme: nessuna allocazione da 2^62 byte
        SafeTensorsFile(_write(tmp_path / "huge.safetensors", {}, header_len=2 ** 62))
    (tmp_path / "json.safetensors").write_bytes(struct.pack("<Q", 3) + b"{x}")
    with pytest.raises(SafeTensorsError):
        SafeTensorsFile(tmp_path / "json.safetensors")


def test_index_cannot_escape_model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    _write(tmp_path / "outside.safetensors", {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
           b"\0" * 4)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(SafeTensorsError):
        SafeTensorsDir(model)
