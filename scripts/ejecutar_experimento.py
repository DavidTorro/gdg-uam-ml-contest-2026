"""Entrena y evalúa un experimento para importarlo en el dashboard"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "ml-contest-materials" / "harness.py"


def sha256(ruta: Path) -> str:
    # El hash identifica los archivos exactos usados en esta ejecución
    with ruta.open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="Identificador único, por ejemplo exp001")
    parser.add_argument("--muestras", type=int, default=5_000)
    parser.add_argument("--iteraciones", type=int, default=80)
    parser.add_argument("--arquitectura", choices=("histgb", "cnn"), default="histgb")
    parser.add_argument("--epocas", type=int, default=12, help="Épocas de la CNN")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamaño de lote de la CNN")
    parser.add_argument("--aumento-ruido", action="store_true", help="HGB: copias ruidosas; CNN: ruido y huecos por lote")
    parser.add_argument("--aumento-ritmo", action="store_true", help="Varía el ritmo de las copias ruidosas")
    parser.add_argument("--temporales", action="store_true", help="Añade características de tramos temporales")
    parser.add_argument("--extendidas", action="store_true", help="Añade características estadísticas y espectrales detalladas")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-muestras", type=int, default=1_000)
    parser.add_argument("--presupuesto-s", type=float, default=2.0)
    args = parser.parse_args()

    if args.id is None:
        if len(sys.argv) > 1:
            parser.error("Indica --id o ejecuta el script sin argumentos para abrir el menú")
        from menu_entrenamiento import menu_entrenamiento

        try:
            seleccion = menu_entrenamiento()
        except ValueError as error:
            parser.error(str(error))
        if seleccion is None:
            print("Entrenamiento cancelado")
            return
        for campo, valor in seleccion.items():
            setattr(args, campo, valor)

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("--id debe usar letras minúsculas, números, guiones o guiones bajos")
    if (args.muestras < 8 or args.iteraciones < 1 or args.epocas < 1 or args.batch_size < 1
            or args.runs < 1 or args.max_muestras < 1 or args.presupuesto_s <= 0):
        parser.error("--muestras debe ser al menos 8; los demás valores numéricos deben ser positivos")
    if args.arquitectura == "cnn" and args.muestras < 80:
        parser.error("La CNN requiere al menos 80 muestras para separar control interno")
    if args.arquitectura == "cnn" and (args.aumento_ritmo or args.temporales or args.extendidas):
        parser.error("--aumento-ritmo, --temporales y --extendidas solo se usan con histgb")
    if args.aumento_ritmo and not args.aumento_ruido:
        parser.error("--aumento-ritmo requiere --aumento-ruido")

    # Cada experimento conserva sus pesos y JSON oficiales propios
    carpeta = ROOT / "experiments" / args.id
    salida = ROOT / "results" / args.id
    extension = ".pt" if args.arquitectura == "cnn" else ".pkl"
    pesos = ROOT / "models" / f"{args.id}{extension}"
    for ruta in (carpeta, salida, pesos):
        if ruta.exists():
            parser.error(f"Ya existe {ruta}; elige otro --id para no sobrescribir el historial")

    particion = ROOT / "experiments/particion_90_10_2026.json"
    validacion = ROOT / "data/val_10k.npz"
    for ruta in (HARNESS, ROOT / "data/train_90k.npz", validacion, particion):
        if not ruta.is_file():
            parser.error(f"Falta {ruta}")

    # Se usa el mismo intérprete Python que inició este script
    # El entrenamiento solo lee train_90k y guarda pesos independientes
    script_entrenamiento = "entrenar_cnn.py" if args.arquitectura == "cnn" else "entrenar_modelo.py"
    entrenamiento = [
        sys.executable, str(ROOT / "scripts" / script_entrenamiento),
        "--id", args.id, "--muestras", str(args.muestras),
    ]
    if args.arquitectura == "cnn":
        entrenamiento += ["--epocas", str(args.epocas), "--batch-size", str(args.batch_size)]
        if args.aumento_ruido:
            entrenamiento.append("--aumento-ruido")
    else:
        entrenamiento += ["--iteraciones", str(args.iteraciones)]
        if args.aumento_ruido:
            entrenamiento.append("--aumento-ruido")
        if args.aumento_ritmo:
            entrenamiento.append("--aumento-ritmo")
        if args.temporales:
            entrenamiento.append("--temporales")
        if args.extendidas:
            entrenamiento.append("--extendidas")
    print("\n1/2 Entrenamiento", flush=True)
    subprocess.run(entrenamiento, cwd=ROOT, check=True)

    # Guardamos la configuración antes de evaluar para diagnosticar fallos
    datos_particion = json.loads(particion.read_text(encoding="utf-8"))
    configuracion = {
        "id": args.id,
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrenamiento": {
            "arquitectura": args.arquitectura,
            "muestras": args.muestras,
            "iteraciones": args.epocas if args.arquitectura == "cnn" else args.iteraciones,
            "batch_size": args.batch_size if args.arquitectura == "cnn" else None,
            "aumento_ruido": args.aumento_ruido,
            "aumento_ritmo": args.aumento_ritmo,
            "temporales": args.temporales,
            "extendidas": args.extendidas,
        },
        "evaluacion": {
            "source": "file:data/val_10k.npz",
            "runs": args.runs,
            "max_scored_samples": args.max_muestras,
            "time_budget_s": args.presupuesto_s,
        },
        "particion": {
            "sha256_indices_validacion": datos_particion.get("sha256_indices_validacion"),
            "semilla": datos_particion.get("semilla"),
            "sha256_archivo_validacion": sha256(validacion),
        },
        "sha256": {
            "model.py": sha256(ROOT / "model.py"),
            "src/caracteristicas.py" if args.arquitectura == "histgb" else "src/red_convolucional.py":
                sha256(ROOT / "src" / ("caracteristicas.py" if args.arquitectura == "histgb" else "red_convolucional.py")),
            f"scripts/{script_entrenamiento}": sha256(ROOT / "scripts" / script_entrenamiento),
            "harness.py": sha256(HARNESS),
            "pesos": sha256(pesos),
        },
    }
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # El evaluador recibe solo validación y carga los pesos de este experimento
    # Los límites indicados aquí son de prueba rápida, no de evaluación final
    entorno = os.environ.copy()
    entorno["GDG_MODEL_PATH"] = f"models/{args.id}{extension}"
    evaluacion = [
        sys.executable, str(HARNESS), "evaluate",
        "--model", "model.py",
        "--source", "file:data/val_10k.npz",
        "--runs", str(args.runs),
        "--out", str(salida),
        "--set", f"max_scored_samples={args.max_muestras}",
        "--set", f"time_budget_s={args.presupuesto_s}",
    ]
    print("\n2/2 Evaluación oficial sobre validación", flush=True)
    subprocess.run(evaluacion, cwd=ROOT, env=entorno, check=True)
    print(
        f"\nExperimento {args.id} terminado. Importa results/{args.id}/model "
        f"y experiments/{args.id} en el dashboard"
    )


if __name__ == "__main__":
    main()
