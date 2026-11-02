"""Entrena un primer clasificador pequeño con la partición de entrenamiento"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

# El cargador oficial está en la carpeta de referencias y no se copia al proyecto
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.caracteristicas import extraer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="modelo_inicial")
    parser.add_argument("--muestras", type=int, default=5_000)
    parser.add_argument("--iteraciones", type=int, default=80)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        raise ValueError("--id debe usar letras minúsculas, números, guiones o guiones bajos")

    fuente = ROOT / "data/train_90k.npz"
    dataset = load_dataset(fuente)
    if not 8 <= args.muestras <= len(dataset):
        raise ValueError(f"--muestras debe estar entre 8 y {len(dataset)}")
    if args.iteraciones < 1:
        raise ValueError("--iteraciones debe ser mayor que cero")

    destino = ROOT / "models" / f"{args.id}.pkl"
    informe = ROOT / "experiments" / args.id / "entrenamiento.json"
    if destino.exists() or informe.exists():
        raise FileExistsError(f"El experimento {args.id} ya existe; usa otro --id")

    # Elegimos un subconjunto estratificado solo dentro de train_90k
    # val_10k no se abre ni se usa durante el entrenamiento
    indices = np.arange(len(dataset))
    if args.muestras < len(dataset):
        indices, _ = train_test_split(
            indices,
            train_size=args.muestras,
            random_state=2026,
            stratify=dataset.labels,
        )
    indices = np.sort(indices)

    # La misma función extraer se utilizará después para cada predicción
    inicio = perf_counter()
    matriz = np.stack([extraer(dataset[int(i)]) for i in indices])
    segundos_caracteristicas = perf_counter() - inicio
    etiquetas = np.asarray(dataset.labels[indices], dtype=np.int64)

    # El boosting aprende reglas sobre las características, no sobre la serie cruda
    clasificador = HistGradientBoostingClassifier(
        max_iter=args.iteraciones,
        learning_rate=0.08,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=2026,
    )
    inicio = perf_counter()
    clasificador.fit(matriz, etiquetas)
    segundos_ajuste = perf_counter() - inicio

    destino.parent.mkdir(exist_ok=True)
    with destino.open("wb") as file:
        pickle.dump(clasificador, file, protocol=pickle.HIGHEST_PROTOCOL)

    # El evaluador no registra el tiempo de entrenamiento, así que lo guardamos aquí
    resumen = {
        "modelo": "HistGradientBoostingClassifier",
        "id": args.id,
        "fuente": fuente.name,
        "muestras_entrenamiento": int(len(indices)),
        "clases_entrenamiento": np.bincount(etiquetas, minlength=8).tolist(),
        "caracteristicas_por_muestra": int(matriz.shape[1]),
        "iteraciones": args.iteraciones,
        "semilla": 2026,
        "segundos_caracteristicas": segundos_caracteristicas,
        "segundos_ajuste": segundos_ajuste,
        "segundos_entrenamiento_total": segundos_caracteristicas + segundos_ajuste,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pesos": str(destino.relative_to(ROOT)),
    }
    informe.parent.mkdir(exist_ok=True)
    informe.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
