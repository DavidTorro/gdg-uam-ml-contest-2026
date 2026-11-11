"""Audita una vez el afinamiento con retención 6/7 frente a exp033."""

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

PLAN = ROOT / "plans/retencion_67_01.json"
SALIDA = ROOT / "experiments/retencion_67_01_auditoria.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verificar_candidato(plan: dict) -> str:
    name = plan["candidato"]
    report_path = ROOT / f"experiments/{name}/entrenamiento.json"
    config_path = ROOT / f"experiments/{name}/configuracion.json"
    weights_path = ROOT / f"models/{name}.pt"
    for path in (report_path, config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Falta corrida completa: {path}")
    if (ROOT / f"models/{name}.sesgo.json").exists():
        raise ValueError("El candidato debe evaluarse sin sesgo añadido")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "id": name, "red": "base", "fuente": "train_90k.npz",
        "muestras_entrenamiento": plan["muestras"], "iteraciones": plan["epocas"],
        "batch_size": plan["batch_size"], "lr": plan["lr"],
        "semilla": plan["semilla"], "simular_examen": True,
        "sin_atajos": True, "sin_grupo": False, "sin_escala": False,
        "equilibrar": True, "elegir_epoca": "examen",
        "archivo_examen": "data/val_exp052_select.npz",
        "inicializar_pesos": "models/exp033.pt",
        "sha256_inicializar_pesos": plan["sha256_fuentes"]["models/exp033.pt"],
        "peso_retencion_67": plan["peso_retencion_67"],
        "sha256_sesgo_retencion": plan["sha256_fuentes"]["models/exp033.sesgo.json"],
        "peso_destilacion": 0.0, "peso_consistencia_vistas": 0.0,
        "mixup_alpha": 0.0, "peso_alineacion_demo": 0.0,
        "aumento_ruido": False, "prob_grupo_cero": 0.0,
        "prob_escala_cero": 0.0, "peso_clase_6": 1.0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name}: {key}={report.get(key)!r}; esperado {value!r}")
    if config.get("entrenamiento", {}).get("peso_retencion_67") != plan["peso_retencion_67"]:
        raise ValueError("Configuración de entrenamiento distinta")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    for key in ("peso_retencion_67", "sha256_sesgo_retencion"):
        if checkpoint.get(key) != expected[key]:
            raise ValueError(f"Checkpoint con {key} distinto")
    return sha256(weights_path)


def main() -> None:
    if SALIDA.exists():
        raise FileExistsError(f"Ya existe {SALIDA}; no se sobrescribe")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("protocolo") != "retencion_67_01" or plan.get("candidato") != "r67_retencion_28":
        raise ValueError("Plan inesperado")
    for name, digest in plan["sha256_fuentes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Cambió la fuente: {name}")
    candidate_hash = verificar_candidato(plan)

    torch.set_num_threads(1)
    names = [plan["base"], plan["control_afinamiento"], plan["candidato"]]
    models = {name: cargar(name) for name in names}
    result = {
        "protocolo": plan["protocolo"], "candidato": plan["candidato"],
        "sha256_candidato": candidate_hash,
        "limite": "Auditoría y simulador consultados antes; las vistas comparten señales base. No estima el test oculto.",
    }
    for view, filename in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        data = load_dataset(ROOT / f"data/{filename}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions = evaluar(data, models, {})
        result[view] = {name: resumir(labels, predictions[name]) for name in names}
        print(view, *(f"{name}={result[view][name]['score_mismo_n']:.6f}" for name in names), flush=True)

    base = plan["base"]
    candidate = plan["candidato"]
    base_sim = result["sim"][base]
    new_sim = result["sim"][candidate]
    limits = plan["filtros"]
    result["deltas"] = {
        "score_sim": new_sim["score_mismo_n"] - base_sim["score_mismo_n"],
        "score_raw": result["raw"][candidate]["score_mismo_n"] - result["raw"][base]["score_mismo_n"],
        "aciertos_6_sim": new_sim["aciertos_por_clase"][6] - base_sim["aciertos_por_clase"][6],
        "aciertos_6_7_sim": sum(new_sim["aciertos_por_clase"][6:8]) - sum(base_sim["aciertos_por_clase"][6:8]),
    }
    result["filtros"] = {
        "score_sim": result["deltas"]["score_sim"] >= limits["delta_score_sim_minimo"],
        "score_raw": result["deltas"]["score_raw"] >= limits["delta_score_raw_minimo"],
        "clase_6_sim": result["deltas"]["aciertos_6_sim"] >= limits["delta_aciertos_6_sim_minimo"],
        "clases_6_7_sim": result["deltas"]["aciertos_6_7_sim"] >= limits["delta_aciertos_6_7_sim_minimo"],
    }
    result["pasa"] = all(result["filtros"].values())
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Pasa el filtro: {result['pasa']}; informe: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
