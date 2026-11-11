"""Auditoría prefijada de la destilación, solo tras terminar ambas corridas.

No abre el demo. La auditoría ya se había inspeccionado en rondas anteriores;
sus resultados son comparativos y no estiman un test externo nuevo.
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
from evaluar_ensamble_calidad import bootstrap_delta, cargar, evaluar, resumir

PLAN = ROOT / "plans/destilacion_01.json"
SALIDA = ROOT / "experiments/destilacion_01_auditoria.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validar_corrida(name: str, weight: float, plan: dict) -> None:
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
        "batch_size": plan["batch_size"], "lr": plan["lr"], "semilla": plan["semilla"],
        "simular_examen": True, "sin_atajos": True, "sin_grupo": False,
        "sin_escala": False, "equilibrar": True,
        "elegir_epoca": "examen", "archivo_examen": "data/val_exp052_select.npz",
        "inicializar_pesos": "models/exp033.pt",
        "sha256_inicializar_pesos": plan["sha256_fuentes"]["models/exp033.pt"],
        "peso_destilacion": weight,
        "temperatura_destilacion": plan["temperatura_destilacion"],
        "peso_consistencia_vistas": 0.0, "mixup_alpha": 0.0,
        "peso_alineacion_demo": 0.0, "aumento_ruido": False,
        "prob_grupo_cero": 0.0, "prob_escala_cero": 0.0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{name}: {key}={report.get(key)!r}; esperado {value!r}")
    teacher_hashes = report.get("sha256_profesores", {})
    if weight:
        for teacher in plan["maestros"]:
            path = f"models/{teacher}.pt"
            if teacher_hashes.get(path) != plan["sha256_fuentes"][path]:
                raise ValueError(f"{name}: maestro distinto: {path}")
    elif teacher_hashes:
        raise ValueError(f"{name}: el control no debe cargar maestros")
    artifact = torch.load(weights_path, map_location="cpu", weights_only=True)
    if artifact.get("red") != "base" or artifact.get("peso_destilacion") != weight:
        raise ValueError(f"{name}: checkpoint de otra receta")
    if config.get("entrenamiento", {}).get("peso_destilacion") != weight:
        raise ValueError(f"{name}: configuración distinta")


def main() -> None:
    if SALIDA.exists():
        raise FileExistsError(f"Ya existe {SALIDA}; no se sobrescribe")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("protocolo") != "destilacion_01" or plan["peso_destilacion"] != 0.5:
        raise ValueError("Plan inesperado")
    for name, digest in plan["sha256_fuentes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Cambió la fuente: {name}")
    validar_corrida(plan["control"], 0.0, plan)
    validar_corrida(plan["candidato"], plan["peso_destilacion"], plan)

    torch.set_num_threads(1)
    models = {name: cargar(name) for name in ("exp033", plan["control"], plan["candidato"])}
    report = {"protocolo": plan["protocolo"], "modelos": list(models),
              "regla": "Comparación pareada a igual N; control y candidata comparten inicio, semilla y receta",
              "limite": "La auditoría y el simulador ya se consultaron antes; exp033 y los maestros también tienen dependencia histórica del demo."}
    for source, file in (("sim", "val_exp052_audit"), ("raw", "val_exp052_audit_raw")):
        data = load_dataset(ROOT / f"data/{file}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions = evaluar(data, models, {})
        metrics = {name: resumir(labels, predictions[name]) for name in models}
        metrics["delta_kd_control"] = (metrics[plan["candidato"]]["score_mismo_n"]
                                       - metrics[plan["control"]]["score_mismo_n"])
        metrics["delta_kd_exp033"] = (metrics[plan["candidato"]]["score_mismo_n"]
                                      - metrics["exp033"]["score_mismo_n"])
        if source == "sim":
            metrics["bootstrap_pareado"] = bootstrap_delta(
                labels, predictions[plan["control"]], predictions[plan["candidato"]], draws=2000)
        report[source] = metrics
        print(f"{source}: exp033 {metrics['exp033']['score_mismo_n']:.6f}, "
              f"control {metrics[plan['control']]['score_mismo_n']:.6f}, "
              f"destilada {metrics[plan['candidato']]['score_mismo_n']:.6f}", flush=True)

    sim = report["sim"]
    control = sim[plan["control"]]
    candidate = sim[plan["candidato"]]
    filters = {
        "sim_vs_control_mas_0_5": sim["delta_kd_control"] >= 0.5,
        "sim_vs_exp033_mas_0_5": sim["delta_kd_exp033"] >= 0.5,
        "raw_no_inferior_control": report["raw"]["delta_kd_control"] >= 0,
        "clase_6_no_pierde_mas_de_5": candidate["aciertos_por_clase"][6] >= control["aciertos_por_clase"][6] - 5,
        "clase_7_no_pierde_mas_de_5": candidate["aciertos_por_clase"][7] >= control["aciertos_por_clase"][7] - 5,
    }
    report["filtros_prefijados"] = filters
    report["pasa_filtro"] = all(filters.values())
    SALIDA.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Pasa filtro: {report['pasa_filtro']}; informe: {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
