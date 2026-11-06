"""Lanza una sola vez los seis entrenamientos de CV por grupos con caffeinate.

Se puede reanudar: omite un peso completo y validado, pero se detiene si hay
artefactos parciales o si alguno de los archivos congelados cambió.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from auditar_cv_grupos import load_model, sha256

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "cv_grupos_v1"
RUNS = [("base", "a"), ("base", "b"), ("base", "c"),
        ("dos_rutas", "a"), ("dos_rutas", "b"), ("dos_rutas", "c")]


def check_files(manifest: dict) -> None:
    if manifest.get("protocolo") != PREFIX:
        raise ValueError("El manifiesto no corresponde a cv_grupos_v1")
    for source in manifest["fuentes"].values():
        if sha256(ROOT / source["ruta"]) != source["sha256"]:
            raise ValueError(f"Cambió el origen {source['ruta']}; no iniciar entrenamientos")
    for fold in manifest["folds"].values():
        for item in fold["archivos"].values():
            if sha256(ROOT / item["ruta"]) != item["sha256"]:
                raise ValueError(f"Cambió {item['ruta']}; no iniciar entrenamientos")


def command(architecture: str, group: str, fold: dict) -> list[str]:
    model_id = f"cvg_{'base' if architecture == 'base' else 'dos'}_{group}"
    return [
        "caffeinate", "-dimsu", sys.executable, str(ROOT / "scripts/entrenar_cnn.py"),
        "--id", model_id, "--red", architecture,
        "--indices-entrenamiento", fold["archivos"]["train_indices"]["ruta"],
        "--muestras", str(fold["train"]["n"]),
        "--archivo-examen", fold["archivos"]["select"]["ruta"],
        "--epocas", "200", "--batch-size", "256", "--simular-examen",
        "--sin-atajos", "--sin-grupo", "--equilibrar", "--elegir-epoca", "examen",
        "--ancho", "1", "--lr", "0.002", "--semilla", "2026", "--min-libre-gib", "6",
    ]


def status(model_id: str, group: str, fold: dict, architecture: str) -> str:
    weight = ROOT / f"models/{model_id}.pt"
    folder = ROOT / f"experiments/{model_id}"
    report = folder / "entrenamiento.json"
    if weight.exists() and report.is_file():
        _, artifact = load_model(model_id, group, fold)
        if artifact["red"] != architecture:
            raise ValueError(f"{model_id}: arquitectura inesperada")
        return "completo"
    if weight.exists() or folder.exists():
        return "parcial"
    return "pendiente"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listar", action="store_true", help="Solo muestra el estado; no entrena")
    args = parser.parse_args()
    if not args.listar and shutil.which("caffeinate") is None:
        parser.error("No se encuentra caffeinate; este lanzador es para macOS")
    manifest = json.loads((ROOT / f"experiments/{PREFIX}_preparacion.json").read_text(encoding="utf-8"))
    check_files(manifest)
    for number, (architecture, group) in enumerate(RUNS, 1):
        fold = manifest["folds"][group]
        model_id = f"cvg_{'base' if architecture == 'base' else 'dos'}_{group}"
        state = status(model_id, group, fold, architecture)
        print(f"[{number}/6] {model_id}: {state}", flush=True)
        if state == "parcial":
            raise RuntimeError(f"{model_id} tiene artefactos parciales. Revísalos antes de reanudar; no se borran automáticamente.")
        if state == "completo" or args.listar:
            continue
        subprocess.run(command(architecture, group, fold), cwd=ROOT, check=True)
        if status(model_id, group, fold, architecture) != "completo":
            raise RuntimeError(f"{model_id} terminó sin peso e informe completos")
    if not args.listar:
        print("Seis pliegues terminados. La auditoría pareada está lista para ejecutarse por separado.")


if __name__ == "__main__":
    main()
