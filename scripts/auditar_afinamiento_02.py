"""Audita una vez las dos réplicas del afinamiento sin destilación.

El bloque de 5k ya se consultó antes: esta comparación mide estabilidad entre
semillas en ese bloque, no rendimiento independiente del examen oculto.
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
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset
from evaluar_ensamble_calidad import cargar, evaluar, resumir

PLAN = ROOT / "plans/afinamiento_02.json"
SALIDA = ROOT / "experiments/afinamiento_02_auditoria.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validar_corrida(name: str, seed: int, plan: dict) -> None:
    report_path = ROOT / f"experiments/{name}/entrenamiento.json"
    config_path = ROOT / f"experiments/{name}/configuracion.json"
    weights_path = ROOT / f"models/{name}.pt"
    for path in (report_path, config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Falta corrida completa: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "id": name, "red": "base", "fuente": "train_90k.npz",
        "muestras_entrenamiento": plan["muestras"], "iteraciones": plan["epocas"],
        "batch_size": plan["batch_size"], "lr": plan["lr"], "semilla": seed,
        "simular_examen": True, "sin_atajos": True, "sin_grupo": False,
        "sin_escala": False, "equilibrar": True,
        "elegir_epoca": "examen", "archivo_examen": "data/val_exp052_select.npz",
        "inicializar_pesos": "models/exp033.pt",
        "sha256_inicializar_pesos": plan["sha256_fuentes"]["models/exp033.pt"],
        "peso_destilacion": 0.0, "temperatura_destilacion": 2.0,
        "peso_consistencia_vistas": 0.0, "mixup_alpha": 0.0,
        "peso_alineacion_demo": 0.0, "aumento_ruido": False,
        "prob_grupo_cero": 0.0, "prob_escala_cero": 0.0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name}: {key}={report.get(key)!r}; esperado {value!r}")
    if report.get("sha256_profesores"):
        raise ValueError(f"{name}: no debe cargar maestros")
    artifact = torch.load(weights_path, map_location="cpu", weights_only=True)
    if artifact.get("red") != "base" or artifact.get("peso_destilacion") != 0.0:
        raise ValueError(f"{name}: checkpoint de otra receta")
    if config.get("entrenamiento", {}).get("peso_destilacion") != 0.0:
        raise ValueError(f"{name}: configuración distinta")


def main() -> None:
    if SALIDA.exists():
        raise FileExistsError(f"Ya existe {SALIDA}; no se sobrescribe")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("protocolo") != "afinamiento_02" or plan.get("replicas") != {
        "d02_control_27": 2027, "d02_control_28": 2028
    }:
        raise ValueError("Plan inesperado")
    for name, digest in plan["sha256_fuentes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Cambió la fuente: {name}")
    validar_corrida(plan["control_existente"], 2026, plan)
    for name, seed in plan["replicas"].items():
        validar_corrida(name, seed, plan)

    torch.set_num_threads(1)
    names = [plan["base"], plan["control_existente"], *plan["replicas"]]
    models = {name: cargar(name) for name in names}
    report = {
        "protocolo": plan["protocolo"], "modelos": names,
        "limite": "El bloque de auditoría y el simulador se consultaron antes; las vistas cruda y simulada comparten señales. Comparación exploratoria de estabilidad, no test nuevo."
    }
    for source, file in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        data = load_dataset(ROOT / f"data/{file}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions = evaluar(data, models, {})
        report[source] = {name: resumir(labels, predictions[name]) for name in names}
        print(source, *(f"{name}={report[source][name]['score_mismo_n']:.6f}" for name in names), flush=True)

    base = plan["base"]
    filters = {}
    for name in plan["replicas"]:
        sim = report["sim"][name]
        base_sim = report["sim"][base]
        sim_delta = sim["score_mismo_n"] - base_sim["score_mismo_n"]
        raw_delta = (report["raw"][name]["score_mismo_n"]
                     - report["raw"][base]["score_mismo_n"])
        combined = sum(sim["aciertos_por_clase"][class_id] for class_id in (6, 7))
        base_combined = sum(base_sim["aciertos_por_clase"][class_id] for class_id in (6, 7))
        filters[name] = {
            "delta_score_sim": sim_delta,
            "delta_score_raw": raw_delta,
            "delta_aciertos_6_7_sim": combined - base_combined,
            "score_sim_mas_0_5": sim_delta >= 0.5,
            "score_raw_no_baja": raw_delta >= 0,
            "aciertos_6_7_sim_no_bajan": combined >= base_combined,
        }
    report["filtros_replica"] = filters
    report["reproducido_en_bloque"] = all(
        result["score_sim_mas_0_5"] and result["score_raw_no_baja"]
        and result["aciertos_6_7_sim_no_bajan"] for result in filters.values()
    )
    SALIDA.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Reproducido en este bloque: {report['reproducido_en_bloque']}; "
          f"informe: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
