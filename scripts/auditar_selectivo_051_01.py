"""Audita una segunda opinión de exp051 solo en dudas 6/7 de exp033."""

from __future__ import annotations

import hashlib
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
from evaluar_ensamble_calidad import cargar, resumir
from src.red_convolucional import preparar_lote

OUTPUT = ROOT / "experiments/selectivo_051_01_auditoria.json"
CONFIG = ROOT / "models/selectivo_01.json"
PREVIOUS = ROOT / "experiments/selectivo_01_auditoria.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def predict(dataset, base, second, threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    base_model, base_artifact, base_bias = base
    second_model, second_artifact, second_bias = second
    for artifact in (base_artifact, second_artifact):
        if not artifact.get("sin_atajos") or artifact.get("sin_escala", False) or artifact.get("sin_grupo", False):
            raise ValueError("Preprocesado incompatible con la cascada")
    original = []
    selective = []
    activations = 0
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
            uncertain = (np.isin(base_pred, (6, 7))
                         & (np.abs(base_logits[:, 6] - base_logits[:, 7]) < threshold))
            candidate = base_pred.copy()
            if uncertain.any():
                selected = torch.from_numpy(uncertain)
                second_logits = (second_model(series[selected], auxiliary[selected]).numpy()
                                 + second_bias)
                first_prob = torch.softmax(torch.from_numpy(base_logits[uncertain]), dim=1).numpy()
                second_prob = torch.softmax(torch.from_numpy(second_logits), dim=1).numpy()
                candidate[uncertain] = (first_prob + second_prob).argmax(1)
                activations += int(uncertain.sum())
            original.append(base_pred)
            selective.append(candidate)
    return np.concatenate(original), np.concatenate(selective), activations


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"No se sobrescribe {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("fraccion_entre_predichas_6_7") != 0.2:
        raise ValueError("Regla de activación inesperada")
    threshold = float(config["umbral_margen_6_7"])
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    models = {name: cargar(name) for name in ("exp033", "exp051")}
    result = {
        "protocol": "selectivo_051_01", "threshold": threshold,
        "models": {name: sha256(ROOT / f"models/{name}.pt") for name in models},
        "activation_rule_sha256": sha256(CONFIG),
        "limit": "Bloque de auditoría estudiado antes; crudo y simulado comparten señales. Demo no leído. No estima el oculto.",
    }
    for view, filename in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        path = ROOT / f"data/{filename}.npz"
        dataset = load_dataset(path)
        labels = np.asarray(dataset.labels, dtype=np.int64)
        base, candidate, activations = predict(dataset, models["exp033"], models["exp051"], threshold)
        base_result = resumir(labels, base)
        candidate_result = resumir(labels, candidate)
        previous_base = prior[view]["exp033"]
        if (abs(base_result["score_mismo_n"] - previous_base["score_mismo_n"]) > 1e-5
                or base_result["aciertos_por_clase"] != previous_base["aciertos_por_clase"]):
            raise ValueError(f"exp033 no coincide con la auditoría ONNX anterior: {view}")
        result[view] = {"data_sha256": sha256(path), "base": base_result,
                        "candidate": candidate_result, "activations": activations,
                        "changed_predictions": int(np.count_nonzero(base != candidate))}
        print(f"{view}: {base_result['score_mismo_n']:.6f} → "
              f"{candidate_result['score_mismo_n']:.6f}; "
              f"activaciones={activations}; cambios={result[view]['changed_predictions']}", flush=True)
    sim = result["sim"]
    raw = result["raw"]
    old_hits = sim["base"]["aciertos_por_clase"]
    new_hits = sim["candidate"]["aciertos_por_clase"]
    delta = {
        "score_sim": sim["candidate"]["score_mismo_n"] - sim["base"]["score_mismo_n"],
        "score_raw": raw["candidate"]["score_mismo_n"] - raw["base"]["score_mismo_n"],
        "class_6": new_hits[6] - old_hits[6],
        "class_7": new_hits[7] - old_hits[7],
    }
    delta["classes_6_7"] = delta["class_6"] + delta["class_7"]
    result["delta"] = delta
    result["passes"] = {
        "score_sim": delta["score_sim"] >= 0.5,
        "score_raw": delta["score_raw"] >= 0,
        "class_6": delta["class_6"] >= -5,
        "class_7": delta["class_7"] >= -5,
        "classes_6_7": delta["classes_6_7"] >= 10,
    }
    result["passes_quality"] = all(result["passes"].values())
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Δ={delta}; filtros={result['passes']}; pasa={result['passes_quality']}")


if __name__ == "__main__":
    main()
