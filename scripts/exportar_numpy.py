"""Convierte los pesos de una CNN (.pt) y su empujón opcional en un .npz que no necesita PyTorch.

Después comprueba con el demo y con val_examen_v2 que la versión NumPy elige las mismas
clases que PyTorch y mide cuánto tarda cada una por muestra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_muestra
from src.red_numpy import RedNumpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", required=True, help="Por ejemplo models/exp033.pt")
    parser.add_argument("--sesgo", help="Por ejemplo models/exp033.sesgo.json; sin él no se empuja ninguna clase")
    parser.add_argument("--destino", help="Por defecto, el mismo nombre con .npz (o .sesgo.npz si lleva empujón)")
    args = parser.parse_args()

    artefacto = torch.load(ROOT / args.pesos, map_location="cpu", weights_only=True)
    if artefacto.get("formato") != "gdg-cnn-v1":
        raise ValueError(f"Formato de pesos CNN desconocido: {args.pesos}")
    ancho = int(artefacto.get("ancho", 1))
    fusion_tardia = bool(artefacto.get("fusion_tardia", False))
    sin_atajos = bool(artefacto.get("sin_atajos", False))
    sin_escala = bool(artefacto.get("sin_escala", False))
    sesgo = np.zeros(8, dtype=np.float32)
    if args.sesgo:
        datos = json.loads((ROOT / args.sesgo).read_text(encoding="utf-8"))
        if datos.get("formato") != "gdg-sesgo-v1" or len(datos["sesgo"]) != 8:
            raise ValueError(f"Formato de sesgo desconocido: {args.sesgo}")
        sesgo = np.asarray(datos["sesgo"], dtype=np.float32)

    destino = ROOT / (args.destino or args.pesos.replace(".pt", ".sesgo.npz" if args.sesgo else ".npz"))
    np.savez(destino, formato=np.array("gdg-numpy-v1"), origen=np.array(args.pesos), ancho=np.array(ancho),
             sin_atajos=np.array(sin_atajos), sin_escala=np.array(sin_escala),
             fusion_tardia=np.array(fusion_tardia), sesgo=sesgo,
             **{clave: tensor.numpy() for clave, tensor in artefacto["state_dict"].items()})
    print(f"Guardado {destino.relative_to(ROOT)} · ancho {ancho} · sin atajos {sin_atajos} · sesgo {sesgo.tolist()}")

    torch.set_num_threads(1)
    red = RedConvolucional(ancho=ancho, fusion_tardia=fusion_tardia)
    red.load_state_dict(artefacto["state_dict"])
    red.eval()
    rapida = RedNumpy(destino)
    sesgo_torch = torch.from_numpy(sesgo)

    def con_torch(muestra) -> int:
        serie, auxiliares = preparar_muestra(muestra, sin_atajos=sin_atajos, sin_escala=sin_escala)
        with torch.inference_mode():
            salida = red(torch.from_numpy(serie[None]), torch.from_numpy(auxiliares[None])) + sesgo_torch
        return int(salida.argmax(dim=1).item())

    for nombre in ("test_demo", "val_examen_v2"):
        datos = load_dataset(str(ROOT / "data" / f"{nombre}.npz"))
        muestras = [datos[j] for j in range(min(len(datos), 3000))]
        tiempos = {}
        clases = {}
        for etiqueta, funcion in (("PyTorch", con_torch), ("NumPy", rapida.predecir)):
            funcion(muestras[0])
            inicio = perf_counter()
            clases[etiqueta] = np.array([funcion(muestra) for muestra in muestras])
            tiempos[etiqueta] = (perf_counter() - inicio) / len(muestras) * 1000
        iguales = (clases["PyTorch"] == clases["NumPy"]).mean()
        print(f"{nombre:14s} {len(muestras)} muestras · mismas clases {iguales:.2%} · PyTorch "
              f"{tiempos['PyTorch']:.3f} ms · NumPy {tiempos['NumPy']:.3f} ms por muestra")
        if iguales < 0.999:
            raise SystemExit("La versión NumPy no coincide con PyTorch: no la uses")


if __name__ == "__main__":
    main()
