"""Audita una vez la consulta selectiva frente a exp033, sin medir latencia."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset
from model_selectivo import Model

PLAN = ROOT / "plans/selectivo_01.json"
OUTPUT = ROOT / "experiments/selectivo_01_auditoria.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def resumir(labels: np.ndarray, predicted: np.ndarray) -> dict:
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    n = confusion.sum(axis=1)
    correct = np.diag(confusion)
    term = np.sqrt(n) * (correct - 0.33 * (n - correct)) / np.maximum(n, 1) * (correct / np.maximum(n, 1)) ** 2
    score = float(np.exp(np.log(np.maximum(0.01, term)).mean()))
    return {
        "n": len(labels), "score_mismo_n": score,
        "accuracy": float((labels == predicted).mean()),
        "aciertos_por_clase": correct.tolist(),
        "n_por_clase": n.tolist(), "confusion": confusion.tolist(),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"No se sobrescribe {OUTPUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "models/selectivo_01.json").read_text(encoding="utf-8"))
    if plan.get("protocolo") != "selectivo_01" or config.get("fraccion_entre_predichas_6_7") != 0.2:
        raise ValueError("Plan o regla inesperados")
    for name, expected in plan["sha256_fuentes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Cambió la fuente: {name}")

    os.environ["GDG_PREP_JIT"] = "0"
    os.environ["GDG_JIT_BUFFERS"] = "0"
    os.environ["GDG_ORT_IOBINDING"] = "1"
    model = Model()
    result = {
        "protocolo": "selectivo_01", "umbral": model.umbral,
        "limite": "Bloque y demo consultados antes; vistas cruda y simulada comparten señales. No estima el test oculto.",
    }
    for view, filename in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        dataset = load_dataset(ROOT / f"data/{filename}.npz")
        labels = np.asarray(dataset.labels, dtype=np.int64)
        base = np.empty(len(dataset), dtype=np.int64)
        candidate = np.empty(len(dataset), dtype=np.int64)
        activations = 0
        for i, sample in enumerate(dataset):
            candidate[i] = model.predict(sample)
            base[i] = int(model.base.logits_buffer.argmax())
            logits = model.base.logits_buffer[0]
            activations += (base[i] in (6, 7)
                            and abs(float(logits[6] - logits[7])) < model.umbral)
        result[view] = {
            "exp033": resumir(labels, base),
            "selectivo": resumir(labels, candidate),
            "activaciones": activations,
            "predicciones_cambiadas": int(np.count_nonzero(base != candidate)),
        }
        print(f"{view}: base={result[view]['exp033']['score_mismo_n']:.6f}, "
              f"selectivo={result[view]['selectivo']['score_mismo_n']:.6f}, "
              f"activaciones={activations}, cambios={result[view]['predicciones_cambiadas']}", flush=True)

    base_sim = result["sim"]["exp033"]
    new_sim = result["sim"]["selectivo"]
    old_hits, new_hits = base_sim["aciertos_por_clase"], new_sim["aciertos_por_clase"]
    result["deltas"] = {
        "score_sim": new_sim["score_mismo_n"] - base_sim["score_mismo_n"],
        "score_raw": result["raw"]["selectivo"]["score_mismo_n"] - result["raw"]["exp033"]["score_mismo_n"],
        "aciertos_6_sim": new_hits[6] - old_hits[6],
        "aciertos_7_sim": new_hits[7] - old_hits[7],
        "aciertos_6_7_sim": sum(new_hits[6:8]) - sum(old_hits[6:8]),
    }
    limits = plan["filtros_calidad"]
    result["filtros"] = {
        "score_sim": result["deltas"]["score_sim"] >= limits["delta_score_sim_minimo"],
        "score_raw": result["deltas"]["score_raw"] >= limits["delta_score_raw_minimo"],
        "clase_6_sim": result["deltas"]["aciertos_6_sim"] >= limits["delta_aciertos_6_sim_minimo"],
        "clase_7_sim": result["deltas"]["aciertos_7_sim"] >= limits["delta_aciertos_7_sim_minimo"],
        "clases_6_7_sim": result["deltas"]["aciertos_6_7_sim"] >= limits["delta_aciertos_6_7_sim_minimo"],
    }
    result["pasa_calidad"] = all(result["filtros"].values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Pasa calidad: {result['pasa_calidad']}; informe: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
