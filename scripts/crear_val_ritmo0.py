"""Crea una validación de estrés para el ritmo del canal 0 (clases 6 y 7).

Transforma las 10 000 señales reservadas con el simulador v2 más una variación
aleatoria del espectro del canal 0, igual para todas las clases. No es test real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulador import simular_examen

AJUSTES = {"prob_ritmo_0": 0.7, "ritmo_0_max": 2.4}
SEMILLA = 2027


def main() -> None:
    origen = ROOT / "data/val_10k.npz"
    destino = ROOT / "data/val_estres_ritmo0.npz"
    informe = ROOT / "experiments/val_estres_ritmo0.json"
    if destino.exists() or informe.exists():
        raise FileExistsError("La validación de estrés ya existe; no se sobrescribe")
    with np.load(origen, allow_pickle=False) as datos:
        series, niveles = simular_examen(
            datos["series"], np.random.default_rng(SEMILLA), parametros=AJUSTES, mezclas=False
        )
        otros = {clave: datos[clave] for clave in datos.files if clave not in ("series", "meta__level")}
        np.savez(destino, series=series, meta__level=niveles, **otros)
    informe.parent.mkdir(exist_ok=True)
    informe.write_text(json.dumps({
        "origen": "data/val_10k.npz",
        "archivo": "data/val_estres_ritmo0.npz",
        "semilla": SEMILLA,
        "ajustes": AJUSTES,
        "transformacion": "src.simulador.simular_examen(mezclas=False)",
        "limite": "Validación de estrés: deliberadamente más variable que el demo; no estima el score real.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Creado {destino} ({len(series)} muestras)")


if __name__ == "__main__":
    main()
