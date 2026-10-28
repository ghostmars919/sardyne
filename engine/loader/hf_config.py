"""Lettura dei config.json di Hugging Face, condivisa da LlamaConfig.from_hf e dal convertitore.

Gli stessi parametri compaiono in forme diverse a seconda della versione di transformers:
- fino a transformers 4: `rope_theta` e `rope_scaling` al primo livello;
- da transformers 5: un dizionario `rope_parameters` ({"rope_theta": ..., "rope_type": "default"}),
  eventualmente diviso per tipo di strato ({"full_attention": {...}, "sliding_attention": {...}}).
"""
import json
from pathlib import Path


def rope_theta(cfg: dict) -> float:
    """La base della RoPE. Solleva NotImplementedError se il modello usa una RoPE scalata."""
    params = cfg.get("rope_parameters") or {}
    if params and all(isinstance(v, dict) for v in params.values()):
        if "full_attention" not in params:
            raise NotImplementedError(f"rope_parameters per tipo di strato non supportati: {sorted(params)}")
        params = params["full_attention"]

    scaling = cfg.get("rope_scaling") or {}
    for p in (params, scaling):
        kind = p.get("rope_type", p.get("type", "default"))
        if kind not in ("default", None):
            raise NotImplementedError(f"rope scaling '{kind}' arriva in Fase 6")

    theta = params.get("rope_theta", cfg.get("rope_theta", 10000.0))
    return float(theta)


def eos_ids(model_dir) -> list[int]:
    """Tutti i token di fine generazione: da generation_config.json, altrimenti da config.json.

    Molti modelli instruct ne hanno più di uno (Llama 3: <|end_of_text|>, <|eom_id|>, <|eot_id|>).
    """
    model_dir = Path(model_dir)
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return [int(e) for e in (eos if isinstance(eos, list) else [eos])]
    return []
