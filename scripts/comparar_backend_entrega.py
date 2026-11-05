"""Compara NumPy y ONNX con el harness oficial sobre un demo repetido.

La repetición sirve solo para medir tiempo; no estima el score del examen.
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
DEMO = ROOT / "data" / "test_demo.npz"
MODELOS = {
    "numpy": "models/exp033.sesgo.npz",
    "onnx": "models/exp033.sesgo.onnx",
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
    resultado = salida / "model" / "run_0.json"
    if not resultado.is_file():
        raise RuntimeError(f"No se creó {resultado}:\n{proceso.stdout}")
    datos = json.loads(resultado.read_text(encoding="utf-8"))
    if datos["status"] != "ok" or datos["n_scored"] != 10000:
        raise RuntimeError(f"Evaluación incompleta ({nombre}): {datos['status']}, "
                           f"{datos['n_scored']} muestras\n{proceso.stdout}")
    return datos


def main() -> None:
    for ruta in [HARNESS, DEMO, ROOT / "model.py", *(ROOT / p for p in MODELOS.values())]:
        if not ruta.is_file():
            raise SystemExit(f"Falta {ruta}")
    with tempfile.TemporaryDirectory(prefix="gdg_backend_") as temporal:
        destino = Path(temporal)
        with np.load(DEMO, allow_pickle=False) as datos:
            repetidos = {clave: np.tile(datos[clave], (25,) + (1,) * (datos[clave].ndim - 1))
                         for clave in datos.files}
        repetidos["sample_index"] = np.arange(10000, dtype=np.int64)
        fuente = destino / "demo_repetido.npz"
        np.savez_compressed(fuente, **repetidos)

        resultados: dict[str, list[dict]] = {"numpy": [], "onnx": []}
        for indice, nombre in enumerate(("numpy", "onnx", "onnx", "numpy")):
            datos = ejecutar(nombre, fuente, destino / f"run_{indice}_{nombre}")
            resultados[nombre].append(datos)
            print(f"{indice + 1}/4 {nombre}: {datos['latency_s']['mean'] * 1000:.4f} ms, "
                  f"score {datos['score']:.6f}", flush=True)

        matrices = [item["confusion"] for grupo in resultados.values() for item in grupo]
        if any(matriz != matrices[0] for matriz in matrices[1:]):
            raise SystemExit("ERROR: NumPy y ONNX no dieron las mismas predicciones")
        tiempos = {nombre: float(np.mean([item["latency_s"]["mean"] for item in grupo]))
                   for nombre, grupo in resultados.items()}
        mejora = (tiempos["numpy"] / tiempos["onnx"]) ** 0.5 - 1.0
        print(f"Paridad: 10000/10000 predicciones; latencia media NumPy "
              f"{tiempos['numpy'] * 1000:.4f} ms, ONNX {tiempos['onnx'] * 1000:.4f} ms")
        print(f"Proyección por velocidad: {mejora * 100:+.2f} % de score a 60 s "
              "si la distribución de clases y el throughput se mantienen. "
              "No es una nota medida del examen.")


if __name__ == "__main__":
    main()
