"""Auditoría prefijada de la puerta de calidad, después de los cinco ajustes.

Compara las dos semillas de 90k con referencias de igual semilla y las mismas
5k señales. No abre demo ni el bloque audit antes de verificar los cinco pesos.
"""

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

from evaluar_ensamble_calidad import resumir
from auditar_tanda_nocturna_01 import by_group
from src.red_convolucional import preparar_lote
from src.red_dos_rutas_gate import RedDosRutasGate

PLAN = ROOT / "plans/gate_calidad_01.json"
SALIDA = ROOT / "experiments/gate_calidad_01_auditoria.json"
PRED = ROOT / "data/gate_calidad_01_pred.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cargar_validado(run: dict) -> RedDosRutasGate:
    name = run["id"]
    report_path = ROOT / f"experiments/{name}/entrenamiento.json"
    weight_path = ROOT / f"models/{name}.pt"
    if not report_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Falta entrenamiento completo: {name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "id": name, "red": "dos_rutas_gate", "fuente": "train_90k.npz",
        "muestras_entrenamiento": run["muestras"], "iteraciones": 200,
        "batch_size": 256, "ancho": 1, "lr": 0.002, "semilla": run["semilla"],
        "archivo_examen": run["archivo_examen"], "elegir_epoca": "examen",
        "simular_examen": True, "sin_atajos": True, "sin_grupo": run["sin_grupo"],
        "equilibrar": True, "prob_grupo_cero": 0.0, "prob_escala_cero": 0.0,
        "peso_clase_6": 1.0, "peso_consistencia_vistas": 0.0,
        "mixup_alpha": 0.0, "peso_alineacion_demo": 0.0,
        "indices_entrenamiento": run.get("indices_entrenamiento"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name}: {key}={report.get(key)!r}; esperado {value!r}")
    artifact = torch.load(weight_path, map_location="cpu", weights_only=True)
    if artifact.get("red") != "dos_rutas_gate" or artifact.get("sin_grupo") != run["sin_grupo"]:
        raise ValueError(f"Peso de receta distinta: {name}")
    model = RedDosRutasGate(ancho=1)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def predecir(model: RedDosRutasGate, data: dict) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), 1024):
            stop = start + 1024
            series, aux = preparar_lote(
                data["series"][start:stop], data["meta__sample_rate"][start:stop],
                data["meta__level"][start:stop], data["meta__group"][start:stop], sin_atajos=True,
            )
            predictions.append(model(torch.from_numpy(series), torch.from_numpy(aux)).argmax(1).numpy())
    return np.concatenate(predictions)


def main() -> None:
    if SALIDA.exists() or PRED.exists():
        raise FileExistsError("La auditoría ya existe; no se sobrescribe")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("protocolo") != "gate_calidad_01" or len(plan["runs"]) != 5:
        raise ValueError("Plan incorrecto")
    for path, expected in plan["sha256_fuentes"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Cambió la fuente: {path}")
    models = {run["id"]: cargar_validado(run) for run in plan["runs"]}
    with np.load(ROOT / "data/val_exp052_audit.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}
    with np.load(ROOT / "data/val_exp052_audit_raw.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in file.files}
    if not np.array_equal(raw["sample_index"], simulated["sample_index"]):
        raise ValueError("Las vistas no comparten señales base")
    y = raw["labels"]
    with np.load(ROOT / "data/tanda_nocturna_01_pred.npz", allow_pickle=False) as file:
        if not np.array_equal(file["sample_index"], raw["sample_index"]):
            raise ValueError("La tanda histórica no está alineada")
        baseline = {key: file[key] for key in file.files if key.endswith("_sim") or key.endswith("_raw")}
    torch.set_num_threads(1)
    report = {"protocolo": "gate_calidad_01", "resultados": {}, "comparaciones": {},
              "limite": "Auditoría exploratoria ya consultada históricamente; crudo y simulado comparten señales. No mirar demo ni ajustar la puerta a este bloque."}
    arrays = {"sample_index": raw["sample_index"], "labels": y}
    for run in plan["runs"][:2]:
        name = run["id"]
        seed = str(run["semilla"])[-2:]
        report["resultados"][name] = {}
        report["comparaciones"][name] = {}
        for view, data in (("sim", simulated), ("raw", raw)):
            pred = predecir(models[name], data)
            arrays[f"{name}_{view}"] = pred
            result = {"total": resumir(y, pred), "grupos": by_group(data, y, pred)}
            report["resultados"][name][view] = result
            report["comparaciones"][name][view] = {}
            for arch in ("b", "d"):
                reference = baseline[f"n01_{arch}_ref_{seed}_{view}"]
                report["comparaciones"][name][view][f"n01_{arch}_ref_{seed}"] = {
                    "delta_score": result["total"]["score_mismo_n"] - resumir(y, reference)["score_mismo_n"],
                    "delta_accuracy": result["total"]["accuracy"] - float(np.mean(reference == y)),
                    "delta_6": int(np.sum((pred == y) & (y == 6)) - np.sum((reference == y) & (y == 6))),
                    "delta_7": int(np.sum((pred == y) & (y == 7)) - np.sum((reference == y) & (y == 7))),
                }
    np.savez_compressed(PRED, **arrays)
    report["predicciones"] = {"ruta": str(PRED.relative_to(ROOT)), "sha256": sha256(PRED)}
    SALIDA.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Informe: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
