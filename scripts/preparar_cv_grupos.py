"""Fija tres pliegues por grupo sin abrir test_demo ni entrenar modelos.

En cada pliegue A/B/C, train_90k de los otros dos grupos sirve para ajustar;
val_examen_v2 de esos mismos grupos sirve solo para elegir época. El grupo
reservado de val_10k queda fuera de ambas decisiones y se audita luego en
versiones cruda y simulada de las mismas señales base.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = "ABC"
PREFIX = "cv_grupos_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def counts(labels: np.ndarray, rates: np.ndarray, indices: np.ndarray) -> dict:
    return {
        "n": int(len(indices)),
        "por_clase": np.bincount(labels[indices], minlength=8).tolist(),
        "por_tasa": {str(int(rate)): int(np.sum(rates[indices] == rate)) for rate in (1.0, 2.0)},
    }


def main() -> None:
    data = ROOT / "data"
    manifest_path = ROOT / "experiments" / f"{PREFIX}_preparacion.json"
    targets = [manifest_path]
    for group in GROUPS:
        targets.extend((data / f"{PREFIX}_sin_{group.lower()}_train.npy",
                        data / f"{PREFIX}_sin_{group.lower()}_select.npz",
                        data / f"{PREFIX}_{group.lower()}_audit.npy"))
    existing = [str(path.relative_to(ROOT)) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Ya existen artefactos de CV; no se sobrescriben: {existing}")

    with np.load(data / "train_90k.npz", allow_pickle=False) as file:
        train = {key: file[key] for key in ("labels", "sample_index", "meta__group", "meta__sample_rate")}
    with np.load(data / "val_10k.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in ("labels", "sample_index", "meta__group", "meta__sample_rate")}
    with np.load(data / "val_examen_v2.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}

    for key in raw:
        if not np.array_equal(raw[key], simulated[key]):
            raise ValueError(f"val_10k y val_examen_v2 no están alineados en {key}")
    if len(np.intersect1d(train["sample_index"], raw["sample_index"])):
        raise ValueError("train_90k y val_10k comparten sample_index")
    if set(np.unique(train["meta__group"])) != set(GROUPS) or set(np.unique(raw["meta__group"])) != set(GROUPS):
        raise ValueError("Los grupos disponibles no son exactamente A/B/C")

    folds = {}
    for group in GROUPS:
        short = group.lower()
        train_idx = np.flatnonzero(train["meta__group"] != group).astype(np.int64)
        select_idx = np.flatnonzero(raw["meta__group"] != group)
        audit_idx = np.flatnonzero(raw["meta__group"] == group).astype(np.int64)
        if min(np.bincount(train["labels"][train_idx], minlength=8)) < 1000:
            raise ValueError(f"Falta representación de una clase en el entrenamiento sin {group}")
        if min(np.bincount(raw["labels"][audit_idx], minlength=8)) < 250:
            raise ValueError(f"Falta representación de una clase en auditoría de {group}")
        if len(np.intersect1d(select_idx, audit_idx)) or len(select_idx) + len(audit_idx) != len(raw["labels"]):
            raise AssertionError(f"Selección y auditoría se solapan en {group}")

        paths = {
            "train_indices": data / f"{PREFIX}_sin_{short}_train.npy",
            "select": data / f"{PREFIX}_sin_{short}_select.npz",
            "audit_indices": data / f"{PREFIX}_{short}_audit.npy",
        }
        np.save(paths["train_indices"], train_idx, allow_pickle=False)
        np.savez(paths["select"], **{key: value[select_idx] for key, value in simulated.items()})
        np.save(paths["audit_indices"], audit_idx, allow_pickle=False)
        folds[short] = {
            "grupo_reservado": group,
            "train": counts(train["labels"], train["meta__sample_rate"], train_idx),
            "seleccion_epoca": counts(raw["labels"], raw["meta__sample_rate"], select_idx),
            "auditoria": counts(raw["labels"], raw["meta__sample_rate"], audit_idx),
            "archivos": {key: {"ruta": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                         for key, path in paths.items()},
        }

    if sum(fold["auditoria"]["n"] for fold in folds.values()) != len(raw["labels"]):
        raise AssertionError("Los tres pliegues no cubren val_10k exactamente una vez")
    manifest = {
        "protocolo": PREFIX,
        "regla": "Dejar fuera cada grupo A/B/C; elegir época con los otros dos grupos; anular auxiliares de grupo",
        "fuentes": {name: {"ruta": f"data/{name}.npz", "sha256": sha256(data / f"{name}.npz")}
                    for name in ("train_90k", "val_10k", "val_examen_v2")},
        "folds": folds,
        "limites": [
            "La auditoría usa el mismo origen de 100k señales que train; el pliegue evita uso directo de sus etiquetas, pero no crea un examen nuevo.",
            "El simulador v2 fue calibrado históricamente con train y demo; esta CV no valida esa calibración de forma independiente.",
            "Las vistas cruda y simulada comparten las mismas señales base; no se cuentan como dos pruebas independientes.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({group: {part: fold[part]["n"] for part in ("train", "seleccion_epoca", "auditoria")}
                      for group, fold in folds.items()}, ensure_ascii=False, indent=2))
    print(f"Manifiesto: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
