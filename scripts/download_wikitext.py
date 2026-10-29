"""Scarica wikitext-2-raw (lo stesso file usato da llama-perplexity) in data/wikitext-2-raw/.

Uso: python scripts/download_wikitext.py
Il file di test (wiki.test.raw, ~1,3 MB di testo) serve a bench/perplexity.py.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
DEST = Path(__file__).resolve().parent.parent / "data" / "wikitext-2-raw"

with urllib.request.urlopen(URL) as r:
    data = r.read()
DEST.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for info in z.infolist():
        name = Path(info.filename).name
        if name.endswith(".raw"):  # solo i file di testo, con un nome semplice: niente percorsi dal file zip
            (DEST / name).write_bytes(z.read(info))
            print(f"{name}: {info.file_size / 1e6:.1f} MB -> {DEST / name}")
