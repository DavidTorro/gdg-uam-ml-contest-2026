"""Compara ONNX con preparación NumPy y JIT en el harness oficial.

Repite el demo solo para medir tiempo. La puntuación no estima el examen.
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
PESOS = ROOT / "models" / "exp033.sesgo.onnx"


def ejecutar(jit: bool, fuente: Path, salida: Path) -> dict:
    entorno = os.environ.copy()
    entorno["GDG_MODEL_PATH"] = "models/exp033.sesgo.onnx"
    entorno["GDG_PREP_JIT"] = "1" if jit else "0"
    comando = [
        sys.executable, str(HARNESS), "evaluate", "--model", str(ROOT / "model.py"),
        "--source", f"file:{fuente}", "--runs", "1", "--out", str(salida),
        "--set", "time_budget_s=10", "--set", "max_scored_samples=10000",
    ]
    proceso = subprocess.run(comando, cwd=ROOT, env=entorno, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proceso.returncode:
        raise RuntimeError(f"Falló {'JIT' if jit else 'NumPy'}:\n{proceso.stdout}")
    resultado = salida / "model" / "run_0.json"
    if not resultado.is_file():
        raise RuntimeError(f"No se creó {resultado}:\n{proceso.stdout}")
    datos = json.loads(resultado.read_text(encoding="utf-8"))
    if datos["status"] != "ok" or datos["n_scored"] != 10000:
        raise RuntimeError(f"Evaluación incompleta: {datos['status']}, "
                           f"{datos['n_scored']} muestras\n{proceso.stdout}")
    return datos


def main() -> None:
    for ruta in (HARNESS, DEMO, PESOS, ROOT / "model.py"):
        if not ruta.is_file():
            raise SystemExit(f"Falta {ruta}")
    with tempfile.TemporaryDirectory(prefix="gdg_jit_") as temporal:
        destino = Path(temporal)
        with np.load(DEMO, allow_pickle=False) as datos:
            repetidos = {clave: np.tile(datos[clave], (25,) + (1,) * (datos[clave].ndim - 1))
                         for clave in datos.files}
        repetidos["sample_index"] = np.arange(10000, dtype=np.int64)
        fuente = destino / "demo_repetido.npz"
        np.savez_compressed(fuente, **repetidos)

        resultados: dict[bool, list[dict]] = {False: [], True: []}
        for indice, jit in enumerate((False, True, True, False)):
            datos = ejecutar(jit, fuente, destino / f"run_{indice}")
            resultados[jit].append(datos)
            print(f"{indice + 1}/4 {'JIT' if jit else 'NumPy'}: "
                  f"{datos['latency_s']['mean'] * 1000:.4f} ms, "
                  f"score {datos['score']:.6f}", flush=True)

        matrices = [item["confusion"] for grupo in resultados.values() for item in grupo]
        if any(matriz != matrices[0] for matriz in matrices[1:]):
            raise SystemExit("ERROR: las preparaciones dieron predicciones diferentes")
        tiempos = {jit: float(np.mean([item["latency_s"]["mean"] for item in grupo]))
                   for jit, grupo in resultados.items()}
        mejora = (tiempos[False] / tiempos[True]) ** 0.5 - 1.0
        print(f"Paridad: 10000/10000 predicciones; latencia ONNX+NumPy "
              f"{tiempos[False] * 1000:.4f} ms, ONNX+JIT {tiempos[True] * 1000:.4f} ms")
        print(f"Proyección por velocidad: {mejora * 100:+.2f} % de score a 60 s "
              "si throughput y distribución se mantienen. No es nota del examen.")


if __name__ == "__main__":
    main()
