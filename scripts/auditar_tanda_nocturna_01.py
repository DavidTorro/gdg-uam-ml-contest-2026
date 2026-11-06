"""Audita la tanda completa una sola vez en las 5k señales bloqueadas, sin demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluar_ensamble_calidad import resumir
from tanda_nocturna_01 import PLAN, sha256, estado
from src.red_convolucional import RedConvolucional, preparar_lote
from src.red_dos_rutas import RedDosRutas

REPORT = ROOT / "experiments/tanda_nocturna_01_auditoria.json"
PRED = ROOT / "data/tanda_nocturna_01_pred.npz"


def predict(run: dict, data: dict) -> np.ndarray:
    artifact = torch.load(ROOT / f"models/{run['id']}.pt", map_location="cpu", weights_only=True)
    if (artifact.get("red") != run["red"] or artifact.get("sin_grupo", False) != run.get("sin_grupo", False)
            or artifact.get("prob_grupo_cero", 0.0) != run.get("prob_grupo_cero", 0.0)):
        raise ValueError(f"Peso de receta distinta: {run['id']}")
    model = (RedDosRutas(ancho=1) if run["red"] == "dos_rutas" else RedConvolucional(ancho=1))
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), 1024):
            end = start + 1024
            series, auxiliary = preparar_lote(
                data["series"][start:end], data["meta__sample_rate"][start:end],
                data["meta__level"][start:end], data["meta__group"][start:end],
                sin_atajos=True,
            )
            if artifact.get("sin_grupo", False):
                auxiliary[:, 14:17] = 0.0
            results.append(model(torch.from_numpy(series), torch.from_numpy(auxiliary)).argmax(1).numpy())
    return np.concatenate(results)


def by_group(data: dict, y: np.ndarray, prediction: np.ndarray) -> dict:
    groups = data["meta__group"]
    rates = data["meta__sample_rate"]
    result = {}
    for group in "ABC":
        mask = groups == group
        result[group] = resumir(y[mask], prediction[mask])
        result[group]["tasas"] = {}
        for rate in sorted(np.unique(rates[mask]).tolist()):
            subset = mask & (rates == rate)
            result[group]["tasas"][str(rate)] = resumir(y[subset], prediction[subset])
    return result


def main() -> None:
    if REPORT.exists() or PRED.exists():
        raise FileExistsError("La auditoría de la tanda ya existe; no se sobrescribe")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    for relative, expected in plan["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Cambió la fuente: {relative}")
    runs = plan["runs"]
    incomplete = [run["id"] for run in runs if estado(run) != "completo"]
    if incomplete:
        raise RuntimeError(f"No se abre la auditoría con corridas pendientes: {incomplete}")
    with np.load(ROOT / "data/val_exp052_audit.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}
    with np.load(ROOT / "data/val_exp052_audit_raw.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in file.files}
    if not np.array_equal(simulated["sample_index"], raw["sample_index"]) or not np.array_equal(simulated["labels"], raw["labels"]):
        raise ValueError("Las dos vistas de auditoría no están alineadas")
    y = raw["labels"].astype(np.int64)
    torch.set_num_threads(1)
    predictions = {}
    results = {}
    for run in runs:
        name = run["id"]
        results[name] = {"red": run["red"], "condicion": run["condicion"], "semilla": run["semilla"]}
        for view, data in (("sim", simulated), ("raw", raw)):
            pred = predict(run, data)
            predictions[f"{name}_{view}"] = pred
            results[name][view] = {"total": resumir(y, pred), "grupos": by_group(data, y, pred)}
        print(f"{name}: sim {results[name]['sim']['total']['score_mismo_n']:.4f}, "
              f"crudo {results[name]['raw']['total']['score_mismo_n']:.4f}", flush=True)
    comparisons = {}
    for run in runs:
        if run["condicion"] == "ref":
            continue
        ref = next(other for other in runs if other["red"] == run["red"]
                   and other["semilla"] == run["semilla"] and other["condicion"] == "ref")
        comparisons[run["id"]] = {"referencia": ref["id"]}
        for view in ("sim", "raw"):
            current = results[run["id"]][view]
            baseline = results[ref["id"]][view]
            comparisons[run["id"]][view] = {
                "delta_score": current["total"]["score_mismo_n"] - baseline["total"]["score_mismo_n"],
                "delta_accuracy": current["total"]["accuracy"] - baseline["total"]["accuracy"],
                "delta_score_grupo": {group: current["grupos"][group]["score_mismo_n"]
                                      - baseline["grupos"][group]["score_mismo_n"]
                                      for group in "ABC"},
            }
    np.savez_compressed(PRED, sample_index=raw["sample_index"], labels=y, **predictions)
    report = {"protocolo": "tanda_nocturna_01", "plan_sha256": sha256(PLAN),
              "fuentes": plan["source_sha256"], "resultados": results, "comparaciones": comparisons,
              "predicciones": {"ruta": str(PRED.relative_to(ROOT)), "sha256": sha256(PRED)},
              "limite": "Exploración de 20 corridas sobre 5k señales ya estudiadas históricamente; vistas cruda y simulada comparten señales. No seleccionar por demo ni presentar esta auditoría como test externo."}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Auditoría: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
