"""Reajusta la capa final de una CNN con las 400 muestras etiquetadas del demo.

Las capas convolucionales no cambian: solo se vuelve a entrenar la capa final,
mezclando muestras de train_90k (con las clases equilibradas) y del demo.
Primero mide por validación cruzada de 5 partes cómo rinde con muestras del demo
que no ha visto. Con --guardar entrena además con las 400 y guarda un peso nuevo.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_muestra


def preparar(dataset, indices, sin_atajos: bool = False) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    series, auxiliares = zip(*(preparar_muestra(dataset[int(j)], sin_atajos=sin_atajos) for j in indices))
    etiquetas = np.asarray(dataset.labels[indices], dtype=np.int64)
    return torch.from_numpy(np.stack(series)), torch.from_numpy(np.stack(auxiliares)), etiquetas


def resumir(red: RedConvolucional, series: torch.Tensor, auxiliares: torch.Tensor) -> torch.Tensor:
    """Calcula lo que la capa final recibe, igual que RedConvolucional.forward."""
    partes = []
    with torch.inference_mode():
        for inicio in range(0, len(series), 2048):
            oculto = red.convoluciones(series[inicio:inicio + 2048])
            partes.append(torch.cat((
                red.media(oculto).flatten(1), red.maximo(oculto).flatten(1),
                auxiliares[inicio:inicio + 2048],
            ), dim=1))
    return torch.cat(partes)


def puntuacion(confusion: np.ndarray, escala: float = 1.0) -> float:
    """Fórmula oficial; escala multiplica las muestras para estimar una ejecución completa."""
    n = confusion.sum(axis=1) * escala
    c = np.diag(confusion) * escala
    w = n - c
    t = np.sqrt(n) * (c - 0.33 * w) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, t)).mean()))


def ajustar(cabeza: nn.Module, resumen: torch.Tensor, etiquetas: np.ndarray,
            pesos: np.ndarray, pasos: int, semilla: int) -> nn.Module:
    """Entrena una copia de la capa final, empezando desde los pesos originales."""
    torch.manual_seed(semilla)
    nueva = copy.deepcopy(cabeza)
    nueva.train()
    y = torch.from_numpy(etiquetas)
    w = torch.from_numpy(pesos.astype(np.float32))
    optimizador = torch.optim.AdamW(nueva.parameters(), lr=3e-4, weight_decay=1e-3)
    criterio = nn.CrossEntropyLoss(reduction="none")
    for _ in range(pasos):
        optimizador.zero_grad(set_to_none=True)
        perdida = (criterio(nueva(resumen), y) * w).sum() / w.sum()
        perdida.backward()
        optimizador.step()
    nueva.eval()
    return nueva


def predecir(cabeza: nn.Module, resumen: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return cabeza(resumen).argmax(dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", default="models/exp016.pt")
    parser.add_argument("--muestras-train", type=int, default=20_000)
    parser.add_argument("--pesos-demo", default="0,10,30,100",
                        help="Valores a comparar: cuánto pesa cada muestra del demo frente a una de train")
    parser.add_argument("--pasos", type=int, default=300)
    parser.add_argument("--guardar", help="ID del nuevo peso, por ejemplo exp021")
    parser.add_argument("--peso-demo", type=float, help="Valor de --pesos-demo usado al guardar")
    args = parser.parse_args()
    if args.guardar:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.guardar):
            parser.error("--guardar debe usar letras minúsculas, números, guiones o guiones bajos")
        if args.peso_demo is None:
            parser.error("--guardar necesita --peso-demo")
        if (ROOT / "models" / f"{args.guardar}.pt").exists():
            parser.error(f"El experimento {args.guardar} ya existe; usa otro ID")
    valores = [float(v) for v in args.pesos_demo.split(",")]
    if args.peso_demo is not None and args.peso_demo not in valores:
        valores.append(args.peso_demo)

    artefacto = torch.load(ROOT / args.pesos, map_location="cpu", weights_only=True)
    if artefacto.get("formato") != "gdg-cnn-v1":
        raise ValueError(f"Formato de pesos CNN desconocido: {args.pesos}")
    red = RedConvolucional(ancho=int(artefacto.get("ancho", 1)))
    sin_atajos = bool(artefacto.get("sin_atajos", False))
    red.load_state_dict(artefacto["state_dict"])
    red.eval()

    inicio = perf_counter()
    train = load_dataset(str(ROOT / "data/train_90k.npz"))
    elegidos, _ = train_test_split(
        np.arange(len(train)), train_size=args.muestras_train,
        stratify=train.labels, random_state=2028,
    )
    serie_t, aux_t, y_t = preparar(train, np.sort(elegidos), sin_atajos)
    demo = load_dataset(str(ROOT / "data/test_demo.npz"))
    serie_d, aux_d, y_d = preparar(demo, np.arange(len(demo)), sin_atajos)
    resumen_t = resumir(red, serie_t, aux_t)
    resumen_d = resumir(red, serie_d, aux_d)
    print(f"Muestras preparadas en {perf_counter() - inicio:.1f} s", flush=True)

    # Train se equilibra por clase porque el test trae las 8 clases por igual
    conteo = np.bincount(y_t, minlength=8)
    pesos_t = len(y_t) / (8 * conteo[y_t])

    original = predecir(red.salida, resumen_d)
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (y_d, original), 1)
    filas = [("sin reajustar", confusion)]

    partes = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2026).split(y_d, y_d))
    for valor in valores:
        confusion = np.zeros((8, 8), dtype=np.int64)
        for numero, (ajuste_idx, prueba_idx) in enumerate(partes):
            resumen = torch.cat((resumen_t, resumen_d[ajuste_idx]))
            etiquetas = np.concatenate((y_t, y_d[ajuste_idx]))
            pesos = np.concatenate((pesos_t, np.full(len(ajuste_idx), valor)))
            cabeza = ajustar(red.salida, resumen, etiquetas, pesos, args.pasos, 2026 + numero)
            np.add.at(confusion, (y_d[prueba_idx], predecir(cabeza, resumen_d[prueba_idx])), 1)
        filas.append((f"peso demo {valor:g}", confusion))

    print("\nDemo con validación cruzada (cada muestra se predice sin haberla visto)")
    print(f"{'versión':16s} {'nota demo':>9s} {'nota oficial est.':>17s} {'acierto':>8s} {'clase 6':>8s} {'clase 7':>8s}")
    resultados = []
    for nombre, confusion in filas:
        aciertos = np.diag(confusion)
        fila = {
            "version": nombre,
            "nota_demo": puntuacion(confusion),
            "nota_oficial_estimada": puntuacion(confusion, 1000),
            "acierto": float(aciertos.sum() / confusion.sum()),
            "aciertos_por_clase": aciertos.tolist(),
        }
        resultados.append(fila)
        print(f"{nombre:16s} {fila['nota_demo']:9.3f} {fila['nota_oficial_estimada']:17.1f} "
              f"{fila['acierto']:8.1%} {aciertos[6]:5d}/50 {aciertos[7]:5d}/50")

    if not args.guardar:
        return

    # Peso final: misma red, capa final entrenada con las 400 muestras del demo
    resumen = torch.cat((resumen_t, resumen_d))
    etiquetas = np.concatenate((y_t, y_d))
    pesos = np.concatenate((pesos_t, np.full(len(y_d), args.peso_demo)))
    cabeza = ajustar(red.salida, resumen, etiquetas, pesos, args.pasos, 2026)
    red.salida = cabeza
    destino = ROOT / "models" / f"{args.guardar}.pt"
    estado = {nombre: tensor.detach().cpu().clone() for nombre, tensor in red.state_dict().items()}
    torch.save({"formato": "gdg-cnn-v1", "state_dict": estado}, destino)

    informe = ROOT / "experiments" / args.guardar / "entrenamiento.json"
    informe.parent.mkdir(parents=True, exist_ok=True)
    informe.write_text(json.dumps({
        "modelo": "Red convolucional 1D con capa final reajustada con el demo",
        "id": args.guardar,
        "pesos_base": args.pesos,
        "muestras_train": int(len(y_t)),
        "muestras_demo": int(len(y_d)),
        "peso_demo": args.peso_demo,
        "pasos": args.pasos,
        "validacion_cruzada_demo": resultados,
        "aviso": "Este peso ha visto el demo: su nota en test_demo ya no es una medida justa",
        "pesos": str(destino.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nGuardado {destino.relative_to(ROOT)} con peso demo {args.peso_demo:g}")


if __name__ == "__main__":
    main()
