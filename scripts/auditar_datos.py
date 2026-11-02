"""Compara los datos de entrenamiento y demo sin entrenar ningún modelo"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Resolvemos rutas desde este archivo para poder ejecutar el script desde cualquier carpeta
ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT.parent / "ml-contest-materials"
# Usamos la misma función de lectura que proporciona el concurso
sys.path.insert(0, str(MATERIALS))

from competition_api import load_dataset


def sha256(path: Path) -> str:
    # Leemos por bloques para calcular la huella sin cargar todo el archivo en memoria
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auditar(path: Path) -> dict:
    # load_dataset abre los NPZ oficiales sin copiar todas las series a memoria
    dataset = load_dataset(path)
    series = dataset.series
    labels = np.asarray(dataset.labels, dtype=np.int64)
    n = len(dataset)

    if series.shape[1:] != (4, 256):
        raise ValueError(f"Forma inesperada en {path}: {series.shape}")
    if np.any((labels < 0) | (labels > 7)):
        raise ValueError(f"Etiquetas fuera de 0..7 en {path}")

    # Guardamos una medida por muestra para poder resumir también cada clase
    nan_por_muestra = np.empty(n, dtype=np.int32)
    amplitud_por_muestra = np.empty(n, dtype=np.float32)
    nan_por_canal = np.zeros(4, dtype=np.int64)

    # Analizamos por bloques para limitar el uso de memoria
    for start in range(0, n, 2048):
        end = min(start + 2048, n)
        chunk = series[start:end]
        # NaN mide los huecos y valid excluye cualquier valor no finito de la amplitud
        missing = np.isnan(chunk)
        valid = np.isfinite(chunk)
        clean = np.where(valid, chunk, 0.0)

        nan_por_muestra[start:end] = missing.sum(axis=(1, 2))
        nan_por_canal += missing.sum(axis=(0, 2))

        # Evitamos una división por cero si una muestra no tiene ningún valor válido
        counts = valid.sum(axis=(1, 2))
        amplitud_por_muestra[start:end] = 0
        np.divide(
            np.abs(clean).sum(axis=(1, 2)),
            counts,
            out=amplitud_por_muestra[start:end],
            where=counts > 0,
        )

    # bincount produce una posición por clase, incluidas las clases ausentes
    class_counts = np.bincount(labels, minlength=8)
    metadata = {}

    # Resumimos cada metadato sin asumir que todos sean numéricos
    for name, values in dataset.meta.items():
        array = np.asarray(values)
        unique, counts = np.unique(array, return_counts=True)
        top = np.argsort(counts)[-5:][::-1]
        summary = {
            "tipo": str(array.dtype),
            "valores_distintos": int(len(unique)),
            "mas_frecuentes": [
                {"valor": str(unique[i]), "muestras": int(counts[i])}
                for i in top
            ] if len(unique) < len(array) else [],
        }
        # Para valores continuos como level son más útiles los percentiles que la moda
        if np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            if len(finite):
                summary["percentiles_10_50_90"] = [
                    float(x) for x in np.quantile(finite, [0.1, 0.5, 0.9])
                ]
        metadata[name] = summary

    return {
        "archivo": path.name,
        "sha256": sha256(path),
        "muestras": n,
        "forma": list(series.shape),
        "tipo": str(series.dtype),
        "clases": {str(k): int(class_counts[k]) for k in range(8)},
        # Usamos fracciones entre 0 y 1 para comparar conjuntos de distinto tamaño
        "fraccion_nan": float(nan_por_muestra.sum() / series.size),
        "fraccion_nan_por_canal": [
            float(value / (n * 256)) for value in nan_por_canal
        ],
        "fraccion_nan_por_clase": {
            str(k): (
                float(nan_por_muestra[labels == k].sum() / (class_counts[k] * 1024))
                if class_counts[k] else None
            )
            for k in range(8)
        },
        "amplitud_absoluta_percentiles_10_50_90": [
            float(x) for x in np.quantile(amplitud_por_muestra, [0.1, 0.5, 0.9])
        ],
        "metadatos": metadata,
    }


def main() -> None:
    # El JSON queda en experiments, que es privado y está excluido de Git
    report = {
        "entrenamiento": auditar(ROOT / "data/train_100k.npz"),
        "demo": auditar(ROOT / "data/test_demo.npz"),
    }
    output = ROOT / "experiments/auditoria_datos.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nInforme local: {output}")


if __name__ == "__main__":
    main()
