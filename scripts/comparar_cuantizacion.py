"""Compara ONNX float e INT8 con el harness sobre 10 000 ejemplos repetidos.

La repetición solo mide latencia bajo el mismo flujo; no estima nota de examen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness.py"
if not HARNESS.is_file():
    HARNESS = ROOT.parent / "ml-contest-materials" / "harness.py"
DEMO = ROOT / "data/test_demo.npz"
MODELOS = {
    "float": "models/exp033.sesgo.onnx",
    "int8": "models/exp033.sesgo.int8_qdq.onnx",
}


def ejecutar(nombre: str, fuente: Path, salida: Path) -> dict:
    entorno = os.environ.copy()
    entorno["GDG_MODEL_PATH"] = MODELOS[nombre]
    comando = [
        sys.executable, str(HARNESS), "evaluate", "--model", str(ROOT / "model.py"),
        "--source", f"file:{fuente}", "--runs", "1", "--out", str(salida),
        "--set", "time_budget_s=10", "--set", "max_scored_samples=10000",
    ]
    proceso = subprocess.run(comando, cwd=ROOT, env=entorno, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proceso.returncode:
        raise RuntimeError(f"Falló {nombre}:\n{proceso.stdout}")
    datos = json.loads((salida / "model/run_0.json").read_text(encoding="utf-8"))
    if datos["status"] != "ok" or datos["n_scored"] != 10000:
        raise RuntimeError(f"Evaluación incompleta ({nombre}): {datos['status']}, "
                           f"{datos['n_scored']} muestras\n{proceso.stdout}")
    return datos


def main() -> None:
    for ruta in [HARNESS, DEMO, ROOT / "model.py", *(ROOT / p for p in MODELOS.values())]:
        if not ruta.is_file():
            raise SystemExit(f"Falta {ruta}")
    with tempfile.TemporaryDirectory(prefix="gdg_int8_") as temporal:
        destino = Path(temporal)
        with np.load(DEMO, allow_pickle=False) as datos:
            repetidos = {clave: np.tile(datos[clave], (25,) + (1,) * (datos[clave].ndim - 1))
                         for clave in datos.files}
        repetidos["sample_index"] = np.arange(10000, dtype=np.int64)
        fuente = destino / "demo_repetido.npz"
        np.savez_compressed(fuente, **repetidos)

        resultados: dict[str, list[dict]] = {"float": [], "int8": []}
        for indice, nombre in enumerate(("float", "int8", "int8", "float")):
            datos = ejecutar(nombre, fuente, destino / f"run_{indice}_{nombre}")
            resultados[nombre].append(datos)
            print(f"{indice + 1}/4 {nombre}: {datos['latency_s']['mean'] * 1000:.4f} ms, "
                  f"score repetido {datos['score']:.6f}, acierto {datos['accuracy_all']:.2%}", flush=True)

        tiempos = {nombre: float(np.mean([item["latency_s"]["mean"] for item in grupo]))
                   for nombre, grupo in resultados.items()}
        mejora = (tiempos["float"] / tiempos["int8"]) ** 0.5 - 1.0
        print(f"Latencia media: float {tiempos['float'] * 1000:.4f} ms, "
              f"INT8 {tiempos['int8'] * 1000:.4f} ms", flush=True)
        print(f"Proyección solo por velocidad: {mejora * 100:+.2f} % de score a 60 s, "
              "si el throughput y la distribución se mantienen. El score repetido no es nota de examen.",
              flush=True)


if __name__ == "__main__":
    main()
