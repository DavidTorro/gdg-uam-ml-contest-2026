"""Compara corr67_26/27 con los controles prefijados, sin abrir el demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from auditar_tanda_nocturna_01 import by_group, predict
from evaluar_ensamble_calidad import resumir
from tanda_nocturna_01 import PLAN, sha256

REPORT = ROOT / "experiments/corr67_01_auditoria.json"
PREVIOUS = ROOT / "experiments/tanda_nocturna_01_auditoria.json"
PRED = ROOT / "data/tanda_nocturna_01_pred.npz"
SEEDS = {2026: ("n01_b_ref_26", "corr67_26"),
         2027: ("n01_b_ref_27", "corr67_27")}


def verify_training(control: str, candidate: str, seed: int,
                    flag_key: str, flag_value: object) -> dict:
    base = json.loads((ROOT / f"experiments/{control}/entrenamiento.json").read_text())
    path = ROOT / f"experiments/{candidate}/entrenamiento.json"
    if not path.is_file() or not (ROOT / f"models/{candidate}.pt").is_file():
        raise FileNotFoundError(f"Falta el entrenamiento completo de {candidate}")
    current = json.loads(path.read_text())
    fields = ("muestras_seleccionadas", "iteraciones", "batch_size", "simular_examen",
              "sin_atajos", "sin_escala", "sin_grupo", "equilibrar", "elegir_epoca",
              "archivo_examen", "ancho", "red", "lr", "semilla",
              "parametros_simulador", "prob_grupo_cero", "prob_escala_cero",
              "peso_clase_6", "igualar_correlacion_67", "peso_consistencia_vistas", "mixup_alpha",
              "peso_alineacion_demo")
    different = [key for key in fields if key != flag_key and current.get(key) != base.get(key)]
    if (different or current["semilla"] != seed or current.get(flag_key) != flag_value
            or base.get(flag_key) not in (False, 0.0)):
        raise ValueError(f"Receta de {candidate} distinta del control: {different}")
    artifact = torch.load(ROOT / f"models/{candidate}.pt", map_location="cpu", weights_only=True)
    if artifact.get(flag_key) != flag_value or artifact.get("red") != "base":
        raise ValueError(f"Peso de {candidate} con receta inesperada")
    return {"id": candidate, "red": "base", "semilla": seed}


def run_audit(protocol: str, seeds: dict[int, tuple[str, str]],
              flag_key: str, flag_value: object, report_path: Path,
              plan_path: str) -> None:
    if report_path.exists():
        raise FileExistsError(f"La auditoría ya existe: {report_path}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    for relative in ("data/train_90k.npz", "data/val_exp052_select.npz",
                     "data/val_exp052_audit.npz", "data/val_exp052_audit_raw.npz"):
        if sha256(ROOT / relative) != plan["source_sha256"][relative]:
            raise ValueError(f"Cambió la fuente: {relative}")
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if sha256(PRED) != previous["predicciones"]["sha256"]:
        raise ValueError("Cambiaron las predicciones de referencia")
    runs = {seed: verify_training(*pair, seed, flag_key, flag_value)
            for seed, pair in seeds.items()}

    with np.load(ROOT / "data/val_exp052_audit.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}
    with np.load(ROOT / "data/val_exp052_audit_raw.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in file.files}
    if not (np.array_equal(simulated["sample_index"], raw["sample_index"])
            and np.array_equal(simulated["labels"], raw["labels"])):
        raise ValueError("Vistas de auditoría no alineadas")
    y = raw["labels"].astype(np.int64)
    torch.set_num_threads(1)
    results = {}
    with np.load(PRED, allow_pickle=False) as reference:
        if not (np.array_equal(reference["sample_index"], raw["sample_index"])
                and np.array_equal(reference["labels"], y)):
            raise ValueError("Las predicciones de referencia no corresponden a la auditoría")
        for seed, (control, candidate) in seeds.items():
            entry = {"control": control, "candidate": candidate}
            for view, data in (("sim", simulated), ("raw", raw)):
                before = reference[f"{control}_{view}"]
                after = predict(runs[seed], data)
                base = resumir(y, before)
                new = resumir(y, after)
                old_groups = by_group(data, y, before)
                new_groups = by_group(data, y, after)
                entry[view] = {
                    "control": base,
                    "candidate": new,
                    "delta_score": new["score_mismo_n"] - base["score_mismo_n"],
                    "delta_accuracy": new["accuracy"] - base["accuracy"],
                    "delta_correct_by_class": [a - b for a, b in zip(
                        new["aciertos_por_clase"], base["aciertos_por_clase"])],
                    "delta_score_by_group": {g: new_groups[g]["score_mismo_n"]
                                             - old_groups[g]["score_mismo_n"] for g in "ABC"},
                }
            sim, actual = entry["sim"], entry["raw"]
            c = sim["delta_correct_by_class"]
            entry["passes"] = {
                "sim_score": sim["delta_score"] >= 0.5,
                "raw_score": actual["delta_score"] >= 0,
                "class_6_7": c[6] >= -10 and c[7] >= -10 and c[6] + c[7] >= 10,
                "groups": min(sim["delta_score_by_group"].values()) >= -0.3,
            }
            entry["passes_all"] = all(entry["passes"].values())
            results[str(seed)] = entry
            print(f"{candidate}: Δsim={sim['delta_score']:+.3f}, "
                  f"Δraw={actual['delta_score']:+.3f}, "
                  f"Δ6/7={c[6]:+d}/{c[7]:+d}, "
                  f"filtros={entry['passes']}", flush=True)

    report = {
        "protocol": protocol,
        "n": len(y),
        "plan": plan_path,
        "reference_predictions_sha256": sha256(PRED),
        "results": results,
        "worth_further_testing": all(entry["passes_all"] for entry in results.values()),
        "limit": "Auditoría reutilizada de 5k señales; las vistas cruda y simulada comparten señales. El demo no se abrió. No estima el oculto.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Informe: {report_path.relative_to(ROOT)}")


def main() -> None:
    run_audit("corr67_01", SEEDS, "igualar_correlacion_67", True,
              REPORT, "plans/corr67_01.md")


if __name__ == "__main__":
    main()
