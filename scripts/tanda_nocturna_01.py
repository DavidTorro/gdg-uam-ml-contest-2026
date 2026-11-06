"""Lanza una tanda fija de 20 CNN; David ejecuta este script con caffeinate.

No evalúa ni consulta el bloque de auditoría. Salta únicamente corridas completas
cuya receta coincide exactamente y se detiene si encuentra artefactos parciales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "plans/tanda_nocturna_01.json"
GIB = 1024 ** 3


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def comando(run: dict, min_libre_gib: float) -> list[str]:
    args = [
        sys.executable, str(ROOT / "scripts/ejecutar_experimento.py"),
        "--id", run["id"], "--arquitectura", "cnn", "--red", run["red"],
        "--muestras", "90000", "--epocas", "200", "--batch-size", "256",
        "--simular-examen", "--sin-atajos", "--equilibrar",
        "--elegir-epoca", "examen", "--archivo-examen", "data/val_exp052_select.npz",
        "--validacion", "val_exp052_select", "--lr", "0.002", "--ancho", "1",
        "--semilla", str(run["semilla"]), "--runs", "1",
        "--min-libre-gib", str(min_libre_gib),
    ]
    if run.get("sin_grupo"):
        args.append("--sin-grupo")
    if run.get("prob_grupo_cero"):
        args.extend(("--prob-grupo-cero", str(run["prob_grupo_cero"])))
    if run.get("ajustes_simulador"):
        args.extend(("--ajustes-simulador", json.dumps(run["ajustes_simulador"], sort_keys=True)))
    return args


def estado(run: dict) -> str:
    name = run["id"]
    weight = ROOT / f"models/{name}.pt"
    training_dir = ROOT / f"experiments/{name}"
    results_dir = ROOT / f"results/{name}"
    training = training_dir / "entrenamiento.json"
    config = training_dir / "configuracion.json"
    evaluation = results_dir / "model/aggregate.json"
    paths = (weight, training, config, evaluation)
    if not any(path.exists() for path in (*paths, training_dir, results_dir)):
        return "pendiente"
    if not all(path.is_file() for path in paths):
        return "PARCIAL"
    data = json.loads(training.read_text(encoding="utf-8"))
    configuration = json.loads(config.read_text(encoding="utf-8"))
    configured = configuration["entrenamiento"]
    checks = {
        "red": run["red"], "semilla": run["semilla"],
        "sin_grupo": bool(run.get("sin_grupo", False)),
        "prob_grupo_cero": float(run.get("prob_grupo_cero", 0.0)),
        "iteraciones": 200, "muestras_seleccionadas": 90000,
        "archivo_examen": "data/val_exp052_select.npz",
        "parametros_simulador": None,
    }
    from src.simulador import PARAMETROS
    checks["parametros_simulador"] = {**PARAMETROS, **run.get("ajustes_simulador", {})}
    if any(data.get(key) != value for key, value in checks.items()):
        return "RECETA DISTINTA"
    if configured.get("sin_grupo") != checks["sin_grupo"] or configured.get("prob_grupo_cero") != checks["prob_grupo_cero"]:
        return "RECETA DISTINTA"
    if configuration.get("sha256", {}).get("pesos") != sha256(weight):
        return "PESO CAMBIADO"
    return "completo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listar", action="store_true", help="Muestra el estado sin entrenar")
    parser.add_argument("--min-libre-gib", type=float, default=6.0)
    parser.add_argument("--max-horas", type=float, default=7.5,
                        help="No iniciar otra corrida tras este tiempo; no corta la activa")
    args = parser.parse_args()
    if args.min_libre_gib <= 0 or args.max_horas <= 0:
        parser.error("Los límites deben ser positivos")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or len(plan.get("runs", [])) != 20:
        raise ValueError("Plan nocturno incompleto o de otra versión")
    runs = plan["runs"]
    if len({run["id"] for run in runs}) != 20:
        raise ValueError("IDs duplicados en el plan")
    for relative, expected in plan["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Cambió la fuente fijada: {relative}")
    states = [(run, estado(run)) for run in runs]
    for run, state in states:
        print(f"{run['id']:<18} {run['red']:<10} {run['condicion']:<18} {state}", flush=True)
    if args.listar:
        return 0 if all(state in ("pendiente", "completo") for _, state in states) else 2
    if any(state not in ("pendiente", "completo") for _, state in states):
        print("Hay una corrida parcial o de receta distinta; revisar antes de continuar.", file=sys.stderr)
        return 2
    start = monotonic()
    for run, state in states:
        if state == "completo":
            continue
        elapsed_h = (monotonic() - start) / 3600
        if elapsed_h >= args.max_horas:
            print(f"Límite de {args.max_horas:g} h alcanzado; no se inicia {run['id']}.")
            return 0
        free_gib = shutil.disk_usage(ROOT).free / GIB
        if free_gib < args.min_libre_gib:
            print(f"Disco bajo: {free_gib:.1f} GiB; no se inicia {run['id']}.", file=sys.stderr)
            return 2
        print(f"\nIniciando {run['id']} · {elapsed_h:.2f} h transcurridas · {free_gib:.1f} GiB libres", flush=True)
        result = subprocess.run(comando(run, args.min_libre_gib), cwd=ROOT, check=False)
        if result.returncode:
            print(f"{run['id']} terminó con código {result.returncode}; se detiene la tanda.", file=sys.stderr)
            return result.returncode
        if estado(run) != "completo":
            print(f"{run['id']} no dejó artefactos completos; se detiene la tanda.", file=sys.stderr)
            return 2
    print("\nTanda completa. No se ha abierto el bloque de auditoría ni el demo.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
