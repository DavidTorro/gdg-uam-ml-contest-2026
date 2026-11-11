"""Compara la preparación JIT en buffers con la ruta ONNX actual."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "ml-contest-materials/harness.py"
DEMO = ROOT / "data/test_demo.npz"
WEIGHTS = ROOT / "models/exp033.sesgo.onnx"


def run(mode: str, source: Path, out: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    env = os.environ.copy()
    env["GDG_MODEL_PATH"] = "models/exp033.sesgo.onnx"
    env["GDG_ORT_IOBINDING"] = "1"
    env["GDG_JIT_BUFFERS"] = mode
    env["GDG_PREP_JIT"] = "1"
    command = [
        sys.executable, str(HARNESS), "evaluate",
        "--model", str(ROOT / "model.py"),
        "--source", f"file:{source}",
        "--runs", "1", "--out", str(out),
        "--set", "time_budget_s=60",
        "--set", "max_scored_samples=10000",
        "--set", "save_per_sample=true",
    ]
    process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Falló el modo {mode}:\n{process.stdout}")
    result = json.loads((out / "model/run_0.json").read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["n_scored"] != 10_000:
        raise RuntimeError(f"Evaluación incompleta del modo {mode}: "
                           f"{result['status']}, {result['n_scored']}/10000\n{process.stdout}")
    with np.load(out / "model/run_0.samples.npz", allow_pickle=False) as samples:
        return result, samples["sample_index"].copy(), samples["y_pred"].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/jit_buffers_harness.json")
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if (Path(args.out).is_absolute() or ROOT not in output.parents
            or output.exists()):
        parser.error("--out debe ser una ruta nueva dentro del repositorio")
    for path in (HARNESS, DEMO, WEIGHTS, ROOT / "model.py"):
        if not path.is_file():
            parser.error(f"Falta {path}")

    with tempfile.TemporaryDirectory(prefix="gdg_jit_buffers_") as temp_dir:
        temp = Path(temp_dir)
        with np.load(DEMO, allow_pickle=False) as data:
            if len(data["labels"]) != 400:
                raise ValueError("Se esperaban 400 señales en el demo")
            repeated = {key: np.tile(data[key], (25,) + (1,) * (data[key].ndim - 1))
                        for key in data.files}
        repeated["sample_index"] = np.arange(10_000, dtype=np.int64)
        source = temp / "demo_25x.npz"
        np.savez_compressed(source, **repeated)

        modes = ("0", "1", "1", "0")
        rows = []
        reference_indices = reference_predictions = reference_stream = None
        for index, mode in enumerate(modes):
            result, indices, predictions = run(mode, source, temp / f"run_{index}")
            if reference_indices is None:
                reference_indices = indices
                reference_predictions = predictions
                reference_stream = result["stream_hash"]
            elif (result["stream_hash"] != reference_stream
                  or not np.array_equal(indices, reference_indices)
                  or not np.array_equal(predictions, reference_predictions)):
                raise RuntimeError(f"El modo {mode} cambió el flujo o las predicciones")
            latency = result["latency_s"]["mean"] * 1000
            rows.append({"modo": mode, "latencia_ms": latency,
                         "score_demo_repetido_no_usar": result["score"]})
            print(f"{index + 1}/4: modo {mode}, {latency:.5f} ms, "
                  "10.000 predicciones idénticas", flush=True)

    base = [row["latencia_ms"] for row in rows if row["modo"] == "0"]
    reuse = [row["latencia_ms"] for row in rows if row["modo"] == "1"]
    pairs = ((base[0], reuse[0]), (base[1], reuse[1]))
    reduction = 1 - mean(reuse) / mean(base)
    report = {
        "protocolo": "jit_buffers_01",
        "python": sys.version.split()[0],
        "corridas": rows,
        "predicciones_identicas": 10_000,
        "latencia_base_ms": mean(base),
        "latencia_reutilizacion_ms": mean(reuse),
        "reduccion_relativa": reduction,
        "ambas_parejas_mas_rapidas": all(new < old for old, new in pairs),
        "pasa": all(new < old for old, new in pairs) and reduction >= 0.03,
        "limite": "25 copias de 400 señales miden velocidad, no calidad independiente. El score del demo repetido no se usa.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Reducción: {reduction:.2%}; pasa: {report['pasa']}; informe: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
