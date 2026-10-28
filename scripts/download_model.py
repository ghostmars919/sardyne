"""Scarica modelli da Hugging Face.

    python scripts/download_model.py HuggingFaceTB/SmolLM2-135M          -> models/SmolLM2-135M/
    python scripts/download_model.py Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-fp16.gguf
                                                                         -> models/gguf/<file>

Con "repo" scarica config, tokenizer e pesi safetensors; con "repo:file" solo quel file.
"""
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent

for arg in sys.argv[1:]:
    if ":" in arg:
        repo, filename = arg.split(":", 1)
        dest = ROOT / "models" / "gguf"
        hf_hub_download(repo, filename, local_dir=dest)
        print(f"{arg} -> {dest / filename}")
    else:
        dest = ROOT / "models" / arg.split("/")[-1]
        snapshot_download(arg, local_dir=dest,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
        print(f"{arg} -> {dest}")
