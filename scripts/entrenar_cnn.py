"""Entrena una CNN 1D sin utilizar la validación oficial ni el test demo."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_muestra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--muestras", type=int, default=90_000)
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("--id debe usar letras minúsculas, números, guiones o guiones bajos")
    if args.epocas < 1 or args.batch_size < 1:
        parser.error("--epocas y --batch-size deben ser positivos")

    fuente = ROOT / "data/train_90k.npz"
    dataset = load_dataset(fuente)
    if not 80 <= args.muestras <= len(dataset):
        parser.error(f"--muestras debe estar entre 80 y {len(dataset)}")
    destino = ROOT / "models" / f"{args.id}.pt"
    informe = ROOT / "experiments" / args.id / "entrenamiento.json"
    if destino.exists() or informe.exists():
        parser.error(f"El experimento {args.id} ya existe; usa otro --id")

    np.random.seed(2026)
    torch.manual_seed(2026)
    indices = np.arange(len(dataset))
    if args.muestras < len(dataset):
        indices, _ = train_test_split(
            indices, train_size=args.muestras, stratify=dataset.labels, random_state=2026
        )
    indices = np.sort(indices)
    etiquetas = np.asarray(dataset.labels[indices], dtype=np.int64)
    entrenamiento_idx, control_idx = train_test_split(
        np.arange(len(indices)), test_size=0.1, stratify=etiquetas, random_state=2027
    )

    inicio_total = perf_counter()
    senales = np.empty((len(indices), 4, 256), dtype=np.float32)
    auxiliares = np.empty((len(indices), 17), dtype=np.float32)
    for posicion, indice in enumerate(indices):
        senales[posicion], auxiliares[posicion] = preparar_muestra(dataset[int(indice)])
    segundos_preparacion = perf_counter() - inicio_total

    tensores = TensorDataset(
        torch.from_numpy(senales), torch.from_numpy(auxiliares), torch.from_numpy(etiquetas)
    )
    entrenamiento = DataLoader(
        Subset(tensores, entrenamiento_idx.tolist()),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    control = DataLoader(
        Subset(tensores, control_idx.tolist()),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    red = RedConvolucional().to(dispositivo)
    optimizador = torch.optim.AdamW(red.parameters(), lr=0.001, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)
    criterio = nn.CrossEntropyLoss()
    mejor_perdida = float("inf")
    mejor_epoca = 0
    mejores_pesos = None
    inicio_ajuste = perf_counter()

    for epoca in range(1, args.epocas + 1):
        red.train()
        perdida_entrenamiento = 0.0
        for serie, auxiliar, etiqueta in entrenamiento:
            serie = serie.to(dispositivo)
            auxiliar = auxiliar.to(dispositivo)
            etiqueta = etiqueta.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(red(serie, auxiliar), etiqueta)
            perdida.backward()
            optimizador.step()
            perdida_entrenamiento += perdida.item() * len(etiqueta)
        plan.step()

        red.eval()
        perdida_control = 0.0
        aciertos = 0
        with torch.inference_mode():
            for serie, auxiliar, etiqueta in control:
                serie = serie.to(dispositivo)
                auxiliar = auxiliar.to(dispositivo)
                etiqueta = etiqueta.to(dispositivo)
                predicciones = red(serie, auxiliar)
                perdida_control += criterio(predicciones, etiqueta).item() * len(etiqueta)
                aciertos += (predicciones.argmax(dim=1) == etiqueta).sum().item()
        perdida_control /= len(control_idx)
        if perdida_control < mejor_perdida:
            mejor_perdida = perdida_control
            mejor_epoca = epoca
            mejores_pesos = {nombre: tensor.detach().cpu().clone() for nombre, tensor in red.state_dict().items()}
        print(
            f"Época {epoca}/{args.epocas}: pérdida entrenamiento "
            f"{perdida_entrenamiento / len(entrenamiento_idx):.4f}, "
            f"control {perdida_control:.4f}, acierto control {aciertos / len(control_idx):.2%}",
            flush=True,
        )

    segundos_ajuste = perf_counter() - inicio_ajuste
    destino.parent.mkdir(exist_ok=True)
    torch.save({"formato": "gdg-cnn-v1", "state_dict": mejores_pesos}, destino)
    resumen = {
        "modelo": "Red convolucional 1D",
        "id": args.id,
        "fuente": fuente.name,
        "muestras_seleccionadas": int(len(indices)),
        "muestras_entrenamiento": int(len(entrenamiento_idx)),
        "muestras_control_interno": int(len(control_idx)),
        "clases_entrenamiento": np.bincount(etiquetas[entrenamiento_idx], minlength=8).tolist(),
        "caracteristicas_por_muestra": 4 * 256 + 17,
        "iteraciones": args.epocas,
        "mejor_epoca": mejor_epoca,
        "perdida_control_interno": mejor_perdida,
        "semilla": 2026,
        "dispositivo_entrenamiento": str(dispositivo),
        "segundos_caracteristicas": segundos_preparacion,
        "segundos_ajuste": segundos_ajuste,
        "segundos_entrenamiento_total": segundos_preparacion + segundos_ajuste,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pesos": str(destino.relative_to(ROOT)),
    }
    informe.parent.mkdir(parents=True, exist_ok=True)
    informe.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
