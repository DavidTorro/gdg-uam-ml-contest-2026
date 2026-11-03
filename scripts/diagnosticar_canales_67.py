"""Mide la sensibilidad de exp033 al ocultar un canal de las clases 6 y 7.

Poner a cero un canal normalizado es una intervención artificial: sirve para
formular hipótesis, no para estimar la calidad de un modelo nuevo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_numpy import RedNumpy, preparar_rapida


def evaluar(nombre: str, red: RedNumpy) -> dict:
    datos = load_dataset(ROOT / "data" / nombre)
    indices = np.flatnonzero(np.isin(datos.labels, (6, 7)))
    etiquetas = datos.labels[indices]
    salidas = np.empty((5, len(indices), 8), dtype=np.float32)
    for numero, indice in enumerate(indices):
        serie, auxiliares = preparar_rapida(datos[int(indice)], red.sin_atajos)
        salidas[0, numero] = red.salida(serie, auxiliares)
        for canal in range(4):
            sin_canal = serie.copy()
            sin_canal[canal] = 0.0
            salidas[canal + 1, numero] = red.salida(sin_canal, auxiliares)
    resultado = {}
    for variante, valores in zip(("original", "sin_canal_0", "sin_canal_1", "sin_canal_2", "sin_canal_3"), salidas):
        pred = valores.argmax(axis=1)
        margen = valores[:, 6] - valores[:, 7]
        resultado[variante] = {
            str(clase): {
                "n": int((etiquetas == clase).sum()),
                "accuracy": float((pred[etiquetas == clase] == clase).mean()),
                "mediana_margen_6_menos_7": float(np.median(margen[etiquetas == clase])),
            }
            for clase in (6, 7)
        }
    return resultado


def main() -> None:
    red = RedNumpy(ROOT / "models/exp033.sesgo.npz")
    informe = {
        "modelo": "models/exp033.sesgo.npz",
        "val_examen_v2": evaluar("val_examen_v2.npz", red),
        "test_demo": evaluar("test_demo.npz", red),
        "limite": "Ocultar un canal crea entradas fuera de entrenamiento; la sensibilidad no prueba causalidad ni generalización.",
    }
    destino = ROOT / "experiments/diagnostico_canales_67.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(informe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
