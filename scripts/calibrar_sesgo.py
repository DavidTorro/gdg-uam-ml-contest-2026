"""Calcula un sesgo para que la CNN prediga una clase con la frecuencia esperada.

No lee las etiquetas: solo mira qué clases predice la red sobre las señales.
Sube el valor de una clase hasta que la red la elige en 1/8 de las muestras,
porque el demo, generado como el test, trae las 8 clases en igual proporción.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_muestra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", default="models/exp016.pt")
    parser.add_argument("--datos", default="data/test_demo.npz")
    parser.add_argument("--clase", type=int, default=6)
    args = parser.parse_args()

    artefacto = torch.load(ROOT / args.pesos, map_location="cpu", weights_only=True)
    if artefacto.get("formato") != "gdg-cnn-v1":
        raise ValueError(f"Formato de pesos CNN desconocido: {args.pesos}")
    # Los pesos desde exp022 guardan su ancho y si se entrenaron sin las auxiliares tramposas
    red = RedConvolucional(ancho=int(artefacto.get("ancho", 1)))
    sin_atajos = bool(artefacto.get("sin_atajos", False))
    red.load_state_dict(artefacto["state_dict"])
    red.eval()

    # Preparamos cada muestra igual que model.py y calculamos la salida de la red
    datos = load_dataset(str(ROOT / args.datos))
    series, auxiliares = zip(*(preparar_muestra(datos[j], sin_atajos=sin_atajos) for j in range(len(datos))))
    with torch.inference_mode():
        salida = red(torch.from_numpy(np.stack(series)), torch.from_numpy(np.stack(auxiliares))).numpy()

    objetivo = len(salida) / 8

    def veces_elegida(valor: float) -> int:
        ajustada = salida.copy()
        ajustada[:, args.clase] += valor
        return int((ajustada.argmax(axis=1) == args.clase).sum())

    # La clase se elige más cuanto mayor es su sesgo: buscamos por bisección
    bajo, alto = 0.0, 20.0
    for _ in range(50):
        medio = (bajo + alto) / 2
        if veces_elegida(medio) < objetivo:
            bajo = medio
        else:
            alto = medio
    valor = round(alto, 4)

    sesgo = [0.0] * 8
    sesgo[args.clase] = valor
    antes = np.bincount(salida.argmax(axis=1), minlength=8)
    ajustada = salida.copy()
    ajustada[:, args.clase] += valor
    despues = np.bincount(ajustada.argmax(axis=1), minlength=8)
    print(f"Predicciones por clase sin sesgo: {antes.tolist()}")
    print(f"Predicciones por clase con sesgo: {despues.tolist()}")
    print(f"Sesgo de la clase {args.clase}: {valor}")

    destino = ROOT / args.pesos.replace(".pt", ".sesgo.json")
    destino.write_text(json.dumps({
        "formato": "gdg-sesgo-v1",
        "pesos": args.pesos,
        "datos": args.datos,
        "clase": args.clase,
        "sesgo": sesgo,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Guardado en {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
