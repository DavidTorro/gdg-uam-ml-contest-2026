"""Compara exp053 y su aporte al ensamble fijo de calidad.

No ajusta pesos de mezcla: los cuatro modelos pesan 1/4. La comparación base
es el ensamble uniforme exp033+exp034+exp051, fijado antes de entrenar exp053.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluar_ensamble_calidad import bootstrap_delta, cargar, evaluar, resumir

ROOT = Path(__file__).resolve().parents[1]
TRIPLE = ("exp033", "exp034", "exp051")
CUADRUPLE = (*TRIPLE, "exp053")
FUENTES = ("val_exp052_audit", "val_exp052_audit_raw", "test_demo", "val_10k", "pseudo_test")


def main() -> None:
    destino = ROOT / "experiments/exp053_auditoria_calidad.json"
    if destino.exists():
        raise FileExistsError(f"Ya existe {destino}; no se recalcula ni sobrescribe")
    if not (ROOT / "models/exp053.pt").is_file():
        raise FileNotFoundError("Falta models/exp053.pt")
    torch.set_num_threads(1)
    modelos = {name: cargar(name) for name in CUADRUPLE}
    if modelos["exp053"][1].get("red") != "dos_rutas":
        raise ValueError("exp053 no es la red de dos rutas registrada")
    from competition_api import load_dataset

    report = {"hipotesis": "ruta conjunta y ruta por canal complementan al ensamble fijo",
              "triple_fijo": list(TRIPLE), "cuadruple_fijo": list(CUADRUPLE),
              "regla": "media uniforme de probabilidades softmax; sesgos históricos propios",
              "limites": "demo consultado repetidamente; pseudo deriva de val_10k; velocidad fuera del objetivo actual"}
    for source in FUENTES:
        data = load_dataset(ROOT / f"data/{source}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        pred = evaluar(data, modelos, {"triple": TRIPLE, "cuadruple": CUADRUPLE})
        row = {name: resumir(labels, pred[name]) for name in ("exp053", "triple", "cuadruple")}
        row["delta_score_cuadruple_vs_triple"] = row["cuadruple"]["score_mismo_n"] - row["triple"]["score_mismo_n"]
        row["delta_accuracy_cuadruple_vs_triple"] = row["cuadruple"]["accuracy"] - row["triple"]["accuracy"]
        if source in ("val_exp052_audit", "test_demo"):
            row["bootstrap_pareado_estratificado"] = bootstrap_delta(labels, pred["triple"], pred["cuadruple"])
        report[source] = row
        print(f"{source}: triple {row['triple']['score_mismo_n']:.6f} → "
              f"cuádruple {row['cuadruple']['score_mismo_n']:.6f}; "
              f"exp053 solo {row['exp053']['score_mismo_n']:.6f}", flush=True)
    destino.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Informe guardado en {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
