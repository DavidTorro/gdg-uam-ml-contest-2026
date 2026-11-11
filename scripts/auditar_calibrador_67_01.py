"""Evalúa una vez el calibrador 6/7 en el bloque reservado."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset
from calibrador_67_common import RULE, gather, load_models, sha256, threshold
from evaluar_ensamble_calidad import resumir

ARTIFACT = ROOT / "models/calibrador_67_01.json"
OUTPUT = ROOT / "experiments/calibrador_67_01_auditoria.json"
PREVIOUS = ROOT / "experiments/selectivo_01_auditoria.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"La auditoría ya existe: {OUTPUT}")
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if (artifact.get("format") != "gdg-calibrador-67-v1"
            or artifact["threshold"] != threshold()
            or artifact["C"] != 0.1
            or artifact["source_sha256"]["models/selectivo_01.json"] != sha256(RULE)):
        raise ValueError("Calibrador con receta distinta del plan")
    for name in ("exp033", "exp051"):
        relative = f"models/{name}.pt"
        if artifact["source_sha256"][relative] != sha256(ROOT / relative):
            raise ValueError(f"Cambió {relative}")
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    models = load_models()
    result = {"protocol": "calibrador_67_01", "calibrator_sha256": sha256(ARTIFACT),
              "limit": "Auditoría históricamente consultada; vistas cruda y simulada comparten señales. No estima el test oculto."}
    mean = np.asarray(artifact["mean"], dtype=np.float64)
    scale = np.asarray(artifact["scale"], dtype=np.float64)
    coef = np.asarray(artifact["coef"], dtype=np.float64)
    intercept = float(artifact["intercept"])
    if mean.shape != (2,) or scale.shape != (2,) or coef.shape != (2,) or np.any(scale <= 0):
        raise ValueError("Coeficientes del calibrador inválidos")
    for view, filename in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        path = ROOT / f"data/{filename}.npz"
        data = load_dataset(path)
        labels = np.asarray(data.labels, dtype=np.int64)
        base, indexes, features = gather(data, models, artifact["threshold"])
        candidate = base.copy()
        decision = ((features - mean) / scale) @ coef + intercept
        candidate[indexes] = np.where(decision >= 0, 7, 6)
        base_result = resumir(labels, base)
        candidate_result = resumir(labels, candidate)
        previous_base = prior[view]["exp033"]
        if (abs(base_result["score_mismo_n"] - previous_base["score_mismo_n"]) > 1e-5
                or base_result["aciertos_por_clase"] != previous_base["aciertos_por_clase"]):
            raise ValueError(f"exp033 no coincide con la evaluación anterior: {view}")
        result[view] = {"data_sha256": sha256(path), "base": base_result,
                        "candidate": candidate_result, "activations": int(len(indexes)),
                        "changed_predictions": int(np.count_nonzero(base != candidate))}
        print(f"{view}: {base_result['score_mismo_n']:.6f} → "
              f"{candidate_result['score_mismo_n']:.6f}; "
              f"activaciones={len(indexes)}; cambios={result[view]['changed_predictions']}", flush=True)
    sim, raw = result["sim"], result["raw"]
    before = sim["base"]["aciertos_por_clase"]
    after = sim["candidate"]["aciertos_por_clase"]
    delta = {"score_sim": sim["candidate"]["score_mismo_n"] - sim["base"]["score_mismo_n"],
             "score_raw": raw["candidate"]["score_mismo_n"] - raw["base"]["score_mismo_n"],
             "class_6": after[6] - before[6], "class_7": after[7] - before[7]}
    delta["classes_6_7"] = delta["class_6"] + delta["class_7"]
    result["delta"] = delta
    result["passes"] = {"score_sim": delta["score_sim"] >= 0.5,
                        "score_raw": delta["score_raw"] >= 0,
                        "class_6": delta["class_6"] >= -5,
                        "class_7": delta["class_7"] >= -5,
                        "classes_6_7": delta["classes_6_7"] >= 10}
    result["passes_quality"] = all(result["passes"].values())
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Δ={delta}; filtros={result['passes']}; pasa={result['passes_quality']}")


if __name__ == "__main__":
    main()
