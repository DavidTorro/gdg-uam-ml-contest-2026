"""Fija el presupuesto del segundo modelo con señales demo, sin leer etiquetas."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import Sample
from src.red_onnx import RedOnnx

DEMO = ROOT / "data/test_demo.npz"
WEIGHTS = ROOT / "models/exp033.sesgo.onnx"
OUTPUT = ROOT / "models/selectivo_01.json"
FRACTION = 0.20


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Ya existe {OUTPUT}; no se sobrescribe")
    # La calibración local usa la preparación NumPy; la entrega mantiene JIT.
    os.environ["GDG_PREP_JIT"] = "0"
    os.environ["GDG_JIT_BUFFERS"] = "0"
    red = RedOnnx(WEIGHTS)
    margins = []
    with np.load(DEMO, allow_pickle=False) as data:
        series = data["series"]
        rates = data["meta__sample_rate"]
        levels = data["meta__level"]
        groups = data["meta__group"]
        # data["labels"] no se abre ni se consulta.
        for i in range(len(series)):
            sample = Sample(series[i], {
                "sample_rate": rates[i], "level": levels[i], "group": groups[i],
            })
            label = red.predecir(sample)
            if label in (6, 7):
                logits = red.logits_buffer[0]
                margins.append(abs(float(logits[6] - logits[7])))
    if len(margins) < 10:
        raise ValueError("Muy pocas señales 6/7 para fijar el presupuesto")
    threshold = float(np.quantile(margins, FRACTION))
    result = {
        "protocolo": "selectivo_01", "fraccion_entre_predichas_6_7": FRACTION,
        "umbral_margen_6_7": threshold,
        "n_demo_sin_etiquetas": len(series), "n_predichas_6_7": len(margins),
        "n_activaciones_demo": int(sum(m < threshold for m in margins)),
        "sha256_fuentes": {
            "data/test_demo.npz": sha256(DEMO),
            "models/exp033.sesgo.onnx": sha256(WEIGHTS),
        },
        "limite": "El demo se usa solo para fijar la fracción de activaciones; no se leen sus etiquetas.",
    }
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Umbral {threshold:.6f}; {result['n_activaciones_demo']}/{len(series)} "
          f"activaciones sin usar etiquetas; {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
