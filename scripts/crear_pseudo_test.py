"""Genera 50 000 ejemplos de examen a partir de la validación reservada.

Las transformaciones son independientes, pero reutilizan las 10 000 señales base.
El simulador fue calibrado con el demo; este archivo no es un test independiente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.simulador import simular_examen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestras", type=int, default=50_000)
    parser.add_argument("--semilla", type=int, default=20261103)
    args = parser.parse_args()
    if args.muestras < 8 or args.muestras % 8:
        parser.error("--muestras debe ser un múltiplo de 8")

    origen = ROOT / "data/val_10k.npz"
    destino = ROOT / "data/pseudo_test.npz"
    informe = ROOT / "experiments/pseudo_test.json"
    if destino.exists() or informe.exists():
        parser.error("El pseudo-test o su informe ya existe; no se sobrescribe")

    rng = np.random.default_rng(args.semilla)
    with np.load(origen, allow_pickle=False) as datos:
        etiquetas_base = datos["labels"]
        por_clase = args.muestras // 8
        seleccion = np.concatenate([
            rng.choice(np.flatnonzero(etiquetas_base == clase), por_clase, replace=True)
            for clase in range(8)
        ])
        rng.shuffle(seleccion)
        series = np.empty((args.muestras, 4, 256), dtype=np.float32)
        niveles = np.empty(args.muestras, dtype=np.float64)
        for inicio in range(0, args.muestras, 1024):
            fin = min(inicio + 1024, args.muestras)
            series[inicio:fin], niveles[inicio:fin] = simular_examen(
                datos["series"][seleccion[inicio:fin]], rng, mezclas=False
            )
        np.savez(
            destino,
            series=series,
            labels=etiquetas_base[seleccion],
            sample_index=np.arange(args.muestras, dtype=np.int64),
            origin_sample_index=datos["sample_index"][seleccion],
            meta__sample_rate=datos["meta__sample_rate"][seleccion],
            meta__group=datos["meta__group"][seleccion],
            meta__level=niveles,
        )
    sha = hashlib.sha256()
    with origen.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    informe.parent.mkdir(exist_ok=True)
    informe.write_text(json.dumps({
        "archivo": "data/pseudo_test.npz",
        "origen": "data/val_10k.npz",
        "sha256_origen": sha.hexdigest(),
        "semilla": args.semilla,
        "muestras": args.muestras,
        "por_clase": por_clase,
        "senales_base_unicas": int(np.unique(seleccion).size),
        "transformacion": "src.simulador.simular_examen(mezclas=False)",
        "limite": "Simulador calibrado con test_demo; señales base repetidas; no son muestras independientes.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Creado {destino} ({args.muestras} muestras, {np.unique(seleccion).size} señales base únicas)")


if __name__ == "__main__":
    main()
