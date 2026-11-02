"""Crea una validación fija de 10.000 muestras sin mezclarlas con entrenamiento"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# La API oficial vive en la carpeta de referencias, fuera de este repositorio
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset

# La semilla permite repetir la misma selección en ejecuciones posteriores
SEED = 2026
SOURCE = ROOT / "data/train_100k.npz"
SOURCE_SHA256 = "1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668"


def guardar(dataset, indices: np.ndarray, destino: Path) -> None:
    # np.savez conserva el formato que entiende load_dataset y el evaluador oficial
    # sample_index permite localizar cada fila en el archivo original
    np.savez(
        destino,
        series=dataset.series[indices],
        labels=dataset.labels[indices],
        sample_index=dataset.sample_index[indices],
        **{
            f"meta__{nombre}": valores[indices]
            for nombre, valores in dataset.meta.items()
        },
    )


def main() -> None:
    # El hash detecta un archivo incompleto o una versión distinta de los datos
    with SOURCE.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("El archivo de entrenamiento no coincide con el oficial")

    dataset = load_dataset(SOURCE)
    indices = np.arange(len(dataset))

    # stratify conserva aproximadamente la proporción de las ocho clases
    # La validación no se usará para ajustar el modelo durante el entrenamiento
    train, val = train_test_split(
        indices,
        test_size=10_000,
        random_state=SEED,
        stratify=dataset.labels,
    )

    # Ordenar cambia el orden de lectura pero no qué filas pertenecen a cada grupo
    train = np.sort(train)
    val = np.sort(val)

    # Ninguna fila puede aparecer a la vez en entrenamiento y validación
    if len(train) + len(val) != len(dataset) or np.intersect1d(train, val).size:
        raise ValueError("La partición contiene índices repetidos o perdidos")

    # Los NPZ generados son locales y están excluidos de Git
    guardar(dataset, train, ROOT / "data/train_90k.npz")
    guardar(dataset, val, ROOT / "data/val_10k.npz")

    resumen = {
        "fuente": SOURCE.name,
        "sha256_fuente": digest,
        "semilla": SEED,
        "metodo": "aleatorio estratificado por clase",
        "sha256_indices_validacion": hashlib.sha256(val.tobytes()).hexdigest(),
        "entrenamiento": {
            "muestras": len(train),
            "clases": np.bincount(dataset.labels[train], minlength=8).tolist(),
        },
        "validacion": {
            "muestras": len(val),
            "clases": np.bincount(dataset.labels[val], minlength=8).tolist(),
        },
    }

    # Guardamos la semilla, el hash y los tamaños para reproducir esta partición
    salida = ROOT / "experiments/particion_90_10_2026.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
