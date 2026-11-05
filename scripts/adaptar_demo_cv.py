"""Evalúa adaptación supervisada al dominio del demo sin puntuar muestras vistas.

Cada uno de cinco pliegues parte de exp033 y ajusta la última convolución y la
cabeza con 4/5 del demo, mezclados con train simulado. La época y los parámetros
son fijos: el pliegue reservado se usa solo para medir, nunca para elegir pesos.
No produce un modelo de entrega ni modifica model.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_lote, preparar_muestra
from src.simulador import simular_examen


def nota(confusion: np.ndarray) -> float:
    n = confusion.sum(axis=1)
    c = np.diag(confusion)
    t = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, t)).mean()))


def matriz(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (y, p), 1)
    return confusion


def cargar_base() -> RedConvolucional:
    artefacto = torch.load(ROOT / "models/exp033.pt", map_location="cpu", weights_only=True)
    if artefacto.get("formato") != "gdg-cnn-v1" or artefacto.get("sin_atajos") is not True:
        raise ValueError("exp033.pt no tiene el formato esperado")
    red = RedConvolucional(ancho=int(artefacto.get("ancho", 1)))
    red.load_state_dict(artefacto["state_dict"])
    sesgo = json.loads((ROOT / "models/exp033.sesgo.json").read_text(encoding="utf-8"))
    if sesgo.get("formato") != "gdg-sesgo-v1" or len(sesgo["sesgo"]) != 8:
        raise ValueError("Sesgo de exp033 inválido")
    with torch.no_grad():
        red.salida[2].bias += torch.tensor(sesgo["sesgo"], dtype=torch.float32)
    return red


def preparar_demo() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    datos = load_dataset(str(ROOT / "data/test_demo.npz"))
    preparados = [preparar_muestra(datos[i], sin_atajos=True) for i in range(len(datos))]
    serie = torch.from_numpy(np.stack([s for s, _ in preparados]))
    auxiliares = torch.from_numpy(np.stack([a for _, a in preparados]))
    return serie, auxiliares, np.asarray(datos.labels, dtype=np.int64)


def fuente_por_epoca(datos: dict, posiciones: list[np.ndarray], rng: np.random.Generator,
                     muestras: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if muestras % 8:
        raise ValueError("--muestras-fuente debe ser múltiplo de 8")
    por_clase = muestras // 8
    indices = np.concatenate([rng.choice(pos, por_clase, replace=False) for pos in posiciones])
    rng.shuffle(indices)
    series, nivel = simular_examen(datos["series"][indices], rng)
    serie, aux = preparar_lote(
        series, datos["meta__sample_rate"][indices], nivel,
        datos["meta__group"][indices], sin_atajos=True,
    )
    return torch.from_numpy(serie), torch.from_numpy(aux), torch.from_numpy(datos["labels"][indices].astype(np.int64))


def entrenar_pliegue(base: RedConvolucional, demo_s: torch.Tensor, demo_a: torch.Tensor,
                     demo_y: np.ndarray, train_idx: np.ndarray, fuente: dict,
                     posiciones: list[np.ndarray], *, epocas: int, muestras_fuente: int,
                     semilla: int, dispositivo: torch.device) -> RedConvolucional:
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    red = RedConvolucional().to(dispositivo)
    red.load_state_dict(base.state_dict())
    for parametro in red.parameters():
        parametro.requires_grad_(False)
    for modulo in (red.convoluciones[6], red.salida):
        for parametro in modulo.parameters():
            parametro.requires_grad_(True)
    optimizador = torch.optim.AdamW(
        (p for p in red.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01,
    )
    criterio = nn.CrossEntropyLoss()
    target_s = demo_s[train_idx].to(dispositivo)
    target_a = demo_a[train_idx].to(dispositivo)
    target_y = torch.from_numpy(demo_y[train_idx]).to(dispositivo)

    for epoca in range(epocas):
        source_s, source_a, source_y = fuente_por_epoca(fuente, posiciones, rng, muestras_fuente)
        orden = rng.permutation(muestras_fuente)
        red.train()
        perdida_total = 0.0
        pasos = 0
        for inicio in range(0, muestras_fuente, 256):
            ids = orden[inicio:inicio + 256]
            s = source_s[ids].to(dispositivo)
            a = source_a[ids].to(dispositivo)
            y = source_y[ids].to(dispositivo)
            ids_demo = torch.from_numpy(rng.integers(0, len(train_idx), 64)).to(dispositivo)
            sd = torch.clamp(target_s[ids_demo] + 0.05 * torch.randn_like(target_s[ids_demo]), -6, 6)
            ad = target_a[ids_demo]
            yd = target_y[ids_demo]
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(red(s, a), y) + criterio(red(sd, ad), yd)
            perdida.backward()
            optimizador.step()
            perdida_total += float(perdida.item())
            pasos += 1
        print(f"  época {epoca + 1}/{epocas}: pérdida mixta {perdida_total / pasos:.4f}", flush=True)
    return red.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=8)
    parser.add_argument("--muestras-fuente", type=int, default=5_120)
    parser.add_argument("--semilla", type=int, default=20261104)
    parser.add_argument("--humo", action="store_true", help="Un pliegue, una época y 80 muestras; no guarda informe")
    args = parser.parse_args()
    if args.epocas < 1 or args.muestras_fuente < 80 or args.muestras_fuente % 8:
        parser.error("--epocas debe ser positivo y --muestras-fuente múltiplo de 8 y al menos 80")
    if args.humo:
        args.epocas, args.muestras_fuente = 1, 80
    destino = ROOT / "experiments/exp045_adaptacion_cv.json"
    if destino.exists() and not args.humo:
        parser.error(f"Ya existe {destino}; no se repite ni se sobrescribe")

    torch.set_num_threads(1)
    dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = cargar_base().eval()
    demo_s, demo_a, demo_y = preparar_demo()
    with torch.inference_mode():
        base_pred = base(demo_s, demo_a).argmax(dim=1).numpy()
    print(f"Base exp033: score {nota(matriz(demo_y, base_pred)):.6f}; "
          f"acierto {(base_pred == demo_y).mean():.2%}", flush=True)

    fuente = dict(np.load(ROOT / "data/train_90k.npz"))
    posiciones = [np.flatnonzero(fuente["labels"] == k) for k in range(8)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.semilla)
    predicciones = np.full(len(demo_y), -1, dtype=np.int64)
    for numero, (train_idx, test_idx) in enumerate(cv.split(np.arange(len(demo_y)), demo_y), start=1):
        print(f"Pliegue {numero}/5 · {len(train_idx)} entrenamiento demo, "
              f"{len(test_idx)} reservado · {dispositivo}", flush=True)
        red = entrenar_pliegue(
            base, demo_s, demo_a, demo_y, train_idx, fuente, posiciones,
            epocas=args.epocas, muestras_fuente=args.muestras_fuente,
            semilla=args.semilla + numero, dispositivo=dispositivo,
        )
        with torch.inference_mode():
            predicciones[test_idx] = red(demo_s[test_idx].to(dispositivo),
                                         demo_a[test_idx].to(dispositivo)).argmax(dim=1).cpu().numpy()
        if args.humo:
            print("Prueba de humo completada; no se guarda ningún modelo ni informe")
            return
    if (predicciones < 0).any():
        raise RuntimeError("Hay muestras sin predicción fuera de pliegue")

    confusion = matriz(demo_y, predicciones)
    informe = {
        "descripcion": "Adaptación de exp033 a demo; predicciones fuera de pliegue",
        "modelo_base": "models/exp033.pt + models/exp033.sesgo.json",
        "pliegues": 5, "semilla": args.semilla,
        "epocas_fijas": args.epocas, "muestras_fuente_por_epoca": args.muestras_fuente,
        "capas_entrenadas": ["convoluciones.6", "salida"],
        "lr": 1e-4, "peso_loss_fuente": 1.0, "peso_loss_demo": 1.0,
        "ruido_demo_normalizado": 0.05,
        "score_base": nota(matriz(demo_y, base_pred)),
        "score_oof": nota(confusion),
        "accuracy_base": float((base_pred == demo_y).mean()),
        "accuracy_oof": float((predicciones == demo_y).mean()),
        "aciertos_por_clase_oof": np.diag(confusion).tolist(),
        "confusion_oof": confusion.tolist(),
        "arreglados": int(((base_pred != demo_y) & (predicciones == demo_y)).sum()),
        "estropeados": int(((base_pred == demo_y) & (predicciones != demo_y)).sum()),
        "predicciones_oof": predicciones.tolist(),
    }
    if destino.exists():
        raise FileExistsError(f"Ya existe {destino}; no se sobrescribe")
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OOF: score {informe['score_oof']:.6f}; acierto {informe['accuracy_oof']:.2%}; "
          f"arreglados {informe['arreglados']}, estropeados {informe['estropeados']}")
    print(f"Informe: {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
