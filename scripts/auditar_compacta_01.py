"""Audita la candidata compacta una vez, antes de medir velocidad o demo."""

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

PLAN = ROOT / "plans/compacta_01.json"
OUTPUT = ROOT / "experiments/compacta_01_auditoria.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validar_candidato(plan: dict) -> str:
    name = plan["candidato"]
    report_path = ROOT / f"experiments/{name}/entrenamiento.json"
    config_path = ROOT / f"experiments/{name}/configuracion.json"
    weights_path = ROOT / f"models/{name}.pt"
    for path in (report_path, config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Falta corrida completa: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "id": name, "red": "compacta", "fuente": "train_90k.npz",
        "muestras_entrenamiento": plan["muestras"], "iteraciones": plan["epocas"],
        "batch_size": plan["batch_size"], "lr": plan["lr"],
        "semilla": plan["semilla"], "simular_examen": True,
        "sin_atajos": True, "sin_escala": False, "sin_grupo": False,
        "equilibrar": True, "elegir_epoca": "examen",
        "archivo_examen": "data/val_exp052_select.npz",
        "inicializar_pesos": "models/c01_pruned_init.pt",
        "sha256_inicializar_pesos": plan["sha256_fuentes"]["models/c01_pruned_init.pt"],
        "peso_destilacion": 0.0, "peso_retencion_67": 0.0,
        "peso_consistencia_vistas": 0.0, "mixup_alpha": 0.0,
        "peso_alineacion_demo": 0.0, "peso_clase_6": 1.0,
        "aumento_ruido": False, "prob_grupo_cero": 0.0,
        "prob_escala_cero": 0.0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name}: {key}={report.get(key)!r}; esperado {value!r}")
    if config.get("entrenamiento", {}).get("red") != "compacta":
        raise ValueError("Configuración distinta de la red compacta")
    artifact = torch.load(weights_path, map_location="cpu", weights_only=True)
    initial = torch.load(ROOT / "models/c01_pruned_init.pt", map_location="cpu", weights_only=True)
    if artifact.get("red") != "compacta" or artifact.get("sha256_inicializar_pesos") != expected["sha256_inicializar_pesos"]:
        raise ValueError("Checkpoint de otra receta")
    if not torch.equal(artifact["state_dict"]["sesgo_fijo"], initial["state_dict"]["sesgo_fijo"]):
        raise ValueError("Se modificó el sesgo fijo")
    if (ROOT / f"models/{name}.sesgo.json").exists():
        raise ValueError("No se admite un segundo sesgo")
    return sha256(weights_path)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"No se sobrescribe {OUTPUT}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("protocolo") != "compacta_01" or plan.get("candidato") != "c01_compacta_28":
        raise ValueError("Plan inesperado")
    for name, expected in plan["sha256_fuentes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Cambió la fuente: {name}")
    candidate_hash = validar_candidato(plan)

    torch.set_num_threads(1)
    names = (plan["base"], plan["candidato"])
    models = {name: cargar(name) for name in names}
    report = {
        "protocolo": plan["protocolo"], "candidato": plan["candidato"],
        "sha256_candidato": candidate_hash,
        "limite": "Bloque y simulador ya consultados; crudo y simulado comparten señales. Filtro práctico, no test independiente.",
    }
    for view, filename in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        data = load_dataset(ROOT / f"data/{filename}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions = evaluar(data, models, {})
        report[view] = {name: resumir(labels, predictions[name]) for name in names}
        print(view, *(f"{name}={report[view][name]['score_mismo_n']:.6f}" for name in names), flush=True)

    base, candidate = names
    old_sim, new_sim = report["sim"][base], report["sim"][candidate]
    limits = plan["filtros_calidad"]
    old_hits, new_hits = old_sim["aciertos_por_clase"], new_sim["aciertos_por_clase"]
    report["deltas"] = {
        "score_sim": new_sim["score_mismo_n"] - old_sim["score_mismo_n"],
        "score_raw": report["raw"][candidate]["score_mismo_n"] - report["raw"][base]["score_mismo_n"],
        "aciertos_6_sim": new_hits[6] - old_hits[6],
        "aciertos_7_sim": new_hits[7] - old_hits[7],
        "aciertos_6_7_sim": sum(new_hits[6:8]) - sum(old_hits[6:8]),
    }
    report["filtros"] = {
        "score_sim": report["deltas"]["score_sim"] >= limits["delta_score_sim_minimo"],
        "score_raw": report["deltas"]["score_raw"] >= limits["delta_score_raw_minimo"],
        "clase_6_sim": report["deltas"]["aciertos_6_sim"] >= limits["delta_aciertos_6_sim_minimo"],
        "clase_7_sim": report["deltas"]["aciertos_7_sim"] >= limits["delta_aciertos_7_sim_minimo"],
        "clases_6_7_sim": report["deltas"]["aciertos_6_7_sim"] >= limits["delta_aciertos_6_7_sim_minimo"],
    }
    report["pasa_calidad"] = all(report["filtros"].values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Pasa calidad: {report['pasa_calidad']}; informe: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
