"""Lectura de dos márgenes para el calibrador binario 6/7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluar_ensamble_calidad import cargar
from src.red_convolucional import preparar_lote

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "models/selectivo_01.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def threshold() -> float:
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    if rule.get("fraccion_entre_predichas_6_7") != 0.2:
        raise ValueError("Regla de activación inesperada")
    return float(rule["umbral_margen_6_7"])


def load_models():
    models = {name: cargar(name) for name in ("exp033", "exp051")}
    for model, artifact, _ in models.values():
        if not artifact.get("sin_atajos") or artifact.get("sin_escala", False) or artifact.get("sin_grupo", False):
            raise ValueError("Preprocesado distinto del plan")
        model.eval()
    return models


def gather(dataset, models, limit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve predicción base, índices consultados y [margen base, margen exp051]."""
    base_model, _, base_bias = models["exp033"]
    second_model, _, second_bias = models["exp051"]
    predictions = []
    indexes = []
    features = []
    with torch.inference_mode():
        for start in range(0, len(dataset), 1024):
            end = min(start + 1024, len(dataset))
            series, auxiliary = preparar_lote(
                dataset.series[start:end], dataset.meta["sample_rate"][start:end],
                dataset.meta["level"][start:end], dataset.meta["group"][start:end],
                sin_atajos=True,
            )
            series = torch.from_numpy(series)
            auxiliary = torch.from_numpy(auxiliary)
            base_logits = base_model(series, auxiliary).numpy() + base_bias
            base_pred = base_logits.argmax(1)
            active = np.isin(base_pred, (6, 7)) & (np.abs(base_logits[:, 6] - base_logits[:, 7]) < limit)
            predictions.append(base_pred)
            if active.any():
                selected = torch.from_numpy(active)
                second_logits = second_model(series[selected], auxiliary[selected]).numpy() + second_bias
                features.append(np.column_stack((base_logits[active, 6] - base_logits[active, 7],
                                                 second_logits[:, 6] - second_logits[:, 7])))
                indexes.append(start + np.flatnonzero(active))
    return (np.concatenate(predictions), np.concatenate(indexes), np.concatenate(features))
