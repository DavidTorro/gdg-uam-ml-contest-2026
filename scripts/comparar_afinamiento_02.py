"""Compara cuatro ONNX con el harness oficial; lo ejecuta David.

El demo original mide acierto de contrato. Una repetición de las mismas 400
señales mide latencia, pero NO aporta 10.000 ejemplos independientes.
"""

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
HARNESS = ROOT.parent / "ml-contest-materials" / "harness.py"
DEMO = ROOT / "data/test_demo.npz"
MODELS = (
    "exp033.sesgo",
    "d01_control",
    "d02_control_27",
    "d02_control_28",
)


def evaluate(model: str, source: Path, count: int, out: Path) -> dict:
    env = os.environ.copy()
    env["GDG_MODEL_PATH"] = f"models/{model}.onnx"
    command = [
        sys.executable, str(HARNESS), "evaluate",
        "--model", str(ROOT / "model.py"),
        "--source", f"file:{source}", "--runs", "1", "--out", str(out),
        "--set", "time_budget_s=60", "--set", f"max_scored_samples={count}",
    ]
    process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Falló {model}:\n{process.stdout}")
    result = json.loads((out / "model/run_0.json").read_text(encoding="utf-8"))
    if result["status"] != "ok" or result["n_scored"] != count:
        raise RuntimeError(f"Evaluación incompleta de {model}: "
                           f"{result['status']}, {result['n_scored']}/{count}\n{process.stdout}")
    return result


def summary(result: dict) -> dict:
    return {
        "score": result["score"], "accuracy": result["accuracy_all"],
        "n_scored": result["n_scored"],
        "latency_ms": result["latency_s"]["mean"] * 1000,
        "confusion": result["confusion"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="experiments/afinamiento_02_harness.json",
                        help="Informe JSON nuevo, relativo al repositorio; no se sobrescribe")
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if output.exists():
        parser.error(f"Ya existe {output}; usa --out con otro nombre")
    if Path(args.out).is_absolute() or ROOT not in output.parents:
        parser.error("--out debe quedar dentro del repositorio")
    for path in (HARNESS, DEMO, ROOT / "model.py",
                 *(ROOT / "models" / f"{name}.onnx" for name in MODELS)):
        if not path.is_file():
            parser.error(f"Falta {path}")

    with tempfile.TemporaryDirectory(prefix="gdg_afinamiento_02_") as temporary:
        temp = Path(temporary)
        with np.load(DEMO, allow_pickle=False) as data:
            original_count = len(data["labels"])
            repeated = {key: np.tile(data[key], (25,) + (1,) * (data[key].ndim - 1))
                        for key in data.files}
        if original_count != 400:
            raise ValueError(f"Se esperaban 400 señales en el demo: {original_count}")
        repeated["sample_index"] = np.arange(10_000, dtype=np.int64)
        repeat_file = temp / "demo_repetido_25x.npz"
        np.savez_compressed(repeat_file, **repeated)

        demo = {}
        for name in MODELS:
            result = evaluate(name, DEMO, 400, temp / f"demo_{name}")
            demo[name] = summary(result)
            print(f"Demo {name}: {result['n_correct']}/400, "
                  f"score {result['score']:.6f}", flush=True)

        speed = {name: [] for name in MODELS}
        for index, name in enumerate((*MODELS, *reversed(MODELS))):
            result = evaluate(name, repeat_file, 10_000, temp / f"speed_{index}_{name}")
            speed[name].append(summary(result))
            print(f"Latencia {index + 1}/8 {name}: "
                  f"{result['latency_s']['mean'] * 1000:.4f} ms", flush=True)

    report = {
        "protocolo": "afinamiento_02_harness",
        "python": sys.version.split()[0],
        "demo": demo,
        "speed_repeated_demo": {
            name: {"runs": runs, "mean_latency_ms": mean(run["latency_ms"] for run in runs)}
            for name, runs in speed.items()
        },
        "limite": "El demo se ha consultado históricamente. Las 10.000 señales de velocidad son 25 copias de las mismas 400; su score no se usa para estimar el examen oculto. La medición depende de este equipo.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Informe: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
