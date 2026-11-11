"""Ajusta un calibrador 6/7 con la mitad de selección, sin abrir la auditoría."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset
from calibrador_67_common import RULE, gather, load_models, sha256, threshold

OUTPUT = ROOT / "models/calibrador_67_01.json"
SELECT = ROOT / "data/val_exp052_select.npz"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"El calibrador ya existe: {OUTPUT}")
    torch.set_num_threads(1)
    data = load_dataset(SELECT)
    labels = np.asarray(data.labels, dtype=np.int64)
    limit = threshold()
    models = load_models()
    _, active_indexes, all_features = gather(data, models, limit)
    useful = np.isin(labels[active_indexes], (6, 7))
    x = all_features[useful]
    y = labels[active_indexes][useful]
    counts = np.bincount(y, minlength=8)
    if counts[6] < 30 or counts[7] < 30:
        raise ValueError(f"Muy pocos ejemplos para calibrar 6/7: {counts[6:8].tolist()}")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    z = (x - mean) / scale
    fit = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                             random_state=2026)
    fit.fit(z, y)
    if fit.classes_.tolist() != [6, 7]:
        raise ValueError("Clases inesperadas")
    artifact = {
        "format": "gdg-calibrador-67-v1", "threshold": limit,
        "features": ["exp033_logit6_minus_logit7", "exp051_logit6_minus_logit7"],
        "mean": mean.tolist(), "scale": scale.tolist(),
        "coef": fit.coef_[0].tolist(), "intercept": float(fit.intercept_[0]),
        "class_if_decision_nonnegative": 7, "class_otherwise": 6,
        "C": 0.1, "class_weight": "balanced", "seed": 2026,
        "n_activated": int(len(active_indexes)), "n_fit": int(len(y)),
        "n_fit_6": int(counts[6]), "n_fit_7": int(counts[7]),
        "source_sha256": {
            "data/val_exp052_select.npz": sha256(SELECT),
            "models/selectivo_01.json": sha256(RULE),
            **{f"models/{name}.pt": sha256(ROOT / f"models/{name}.pt") for name in models},
        },
        "limit": "Ajustado solo en la mitad de selección; score en este bloque no es validación independiente.",
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Calibrador guardado: {OUTPUT.relative_to(ROOT)}; "
          f"activadas={len(active_indexes)}, ajuste 6/7={counts[6]}/{counts[7]}")


if __name__ == "__main__":
    main()
