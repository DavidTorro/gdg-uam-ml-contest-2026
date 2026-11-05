"""Prueba si adaptar filtros tempranos corrige el cambio 6↔7 del demo.

Parte de exp033, ajusta las cuatro convoluciones y la cabeza con train simulado
y cuatro quintos del demo etiquetado. Predice el quinto reservado sin usarlo
para entrenar ni escoger época. No crea un modelo de entrega.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from adaptar_demo_cv import cargar_base, fuente_por_epoca, matriz, nota, preparar_demo
from src.red_convolucional import RedConvolucional


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def comprobar_disco(min_libre_gib: float) -> None:
    disponible = shutil.disk_usage(ROOT).free / GIB
    if disponible < min_libre_gib:
        raise RuntimeError(f"Disco bajo: {disponible:.2f} GiB libres < {min_libre_gib:.2f} GiB; se corta")


def entrenar_pliegue(base: RedConvolucional, demo_s: torch.Tensor, demo_a: torch.Tensor,
                     demo_y: np.ndarray, train_idx: np.ndarray, fuente: dict,
                     posiciones: list[np.ndarray], *, epocas: int, muestras_fuente: int,
                     semilla: int, dispositivo: torch.device, min_libre_gib: float) -> RedConvolucional:
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    red = RedConvolucional().to(dispositivo)
    red.load_state_dict(base.state_dict())
    # exp045 dejaba fijas las primeras tres convoluciones. Aquí se adaptan todas.
    optimizador = torch.optim.AdamW([
        {"params": red.convoluciones.parameters(), "lr": 2e-5},
        {"params": red.salida.parameters(), "lr": 1e-4},
    ], weight_decay=0.01)
    criterio = nn.CrossEntropyLoss()
    target_s = demo_s[train_idx].to(dispositivo)
    target_a = demo_a[train_idx].to(dispositivo)
    target_y = torch.from_numpy(demo_y[train_idx]).to(dispositivo)

    for epoca in range(epocas):
        comprobar_disco(min_libre_gib)
        source_s, source_a, source_y = fuente_por_epoca(fuente, posiciones, rng, muestras_fuente)
        orden = rng.permutation(muestras_fuente)
        red.train()
        perdida_fuente = perdida_demo = 0.0
        pasos = 0
        for inicio in range(0, muestras_fuente, 256):
            if pasos % 4 == 0:
                comprobar_disco(min_libre_gib)
            ids = orden[inicio:inicio + 256]
            s = source_s[ids].to(dispositivo)
            a = source_a[ids].to(dispositivo)
            y = source_y[ids].to(dispositivo)
            ids_demo = torch.from_numpy(rng.integers(0, len(train_idx), 64)).to(dispositivo)
            sd = torch.clamp(target_s[ids_demo] + 0.05 * torch.randn_like(target_s[ids_demo]), -6, 6)
            ad = target_a[ids_demo]
            yd = target_y[ids_demo]
            optimizador.zero_grad(set_to_none=True)
            lf = criterio(red(s, a), y)
            ld = criterio(red(sd, ad), yd)
            (0.5 * lf + ld).backward()
            optimizador.step()
            perdida_fuente += float(lf.item())
            perdida_demo += float(ld.item())
            pasos += 1
        print(f"  época {epoca + 1}/{epocas}: pérdida fuente {perdida_fuente / pasos:.4f}, "
              f"demo entrenamiento {perdida_demo / pasos:.4f}", flush=True)
    return red.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--muestras-fuente", type=int, default=5_120)
    parser.add_argument("--semilla", type=int, default=20261105)
    parser.add_argument("--min-libre-gib", type=float, default=6.0)
    parser.add_argument("--humo", action="store_true", help="Un pliegue y una época; no guarda nada")
    args = parser.parse_args()
    if args.epocas < 1 or args.muestras_fuente < 80 or args.muestras_fuente % 8:
        parser.error("--epocas debe ser positivo y --muestras-fuente múltiplo de 8 y al menos 80")
    if not 0 < args.min_libre_gib < 1000 or not np.isfinite(args.min_libre_gib):
        parser.error("--min-libre-gib debe ser positivo y finito")
    if args.humo:
        args.epocas, args.muestras_fuente = 1, 80
    destino = ROOT / "experiments/exp050_adaptacion_profunda_cv.json"
    if destino.exists() and not args.humo:
        parser.error(f"Ya existe {destino}; no se repite ni se sobrescribe")
    comprobar_disco(args.min_libre_gib)

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
        print(f"Pliegue {numero}/5 · {len(train_idx)} demo para entrenar, "
              f"{len(test_idx)} reservadas · {dispositivo}", flush=True)
        antes = base.convoluciones[0].weight.detach().cpu().clone()
        red = entrenar_pliegue(
            base, demo_s, demo_a, demo_y, train_idx, fuente, posiciones,
            epocas=args.epocas, muestras_fuente=args.muestras_fuente,
            semilla=args.semilla + numero, dispositivo=dispositivo,
            min_libre_gib=args.min_libre_gib,
        )
        cambio = float((red.convoluciones[0].weight.detach().cpu() - antes).abs().max())
        if cambio <= 0 or not np.isfinite(cambio):
            raise RuntimeError("La primera convolución no se actualizó")
        with torch.inference_mode():
            predicciones[test_idx] = red(demo_s[test_idx].to(dispositivo),
                                         demo_a[test_idx].to(dispositivo)).argmax(dim=1).cpu().numpy()
        print(f"  cambio máximo primera convolución: {cambio:.6g}", flush=True)
        if args.humo:
            print("Prueba de humo completada; no se ha guardado ningún peso ni informe")
            return
    if (predicciones < 0).any():
        raise RuntimeError("Hay muestras sin predicción fuera de pliegue")

    confusion = matriz(demo_y, predicciones)
    # Umbral fijado antes de mirar el resultado: exigimos una ganancia amplia y
    # que la corrección de la clase 6 no se pague empeorando la 7.
    pasa_filtro = (nota(confusion) >= 4.2 and (predicciones == demo_y).mean() >= 0.865
                   and confusion[6, 6] >= 35 and confusion[7, 7] >= 32)
    informe = {
        "descripcion": "Adaptación de todas las convoluciones de exp033; predicciones fuera de pliegue",
        "modelo_base": "models/exp033.pt + models/exp033.sesgo.json",
        "pliegues": 5, "semilla": args.semilla,
        "epocas_fijas": args.epocas, "muestras_fuente_por_epoca": args.muestras_fuente,
        "capas_entrenadas": ["convoluciones.0", "convoluciones.2", "convoluciones.4", "convoluciones.6", "salida"],
        "lr_convoluciones": 2e-5, "lr_cabeza": 1e-4,
        "peso_loss_fuente": 0.5, "peso_loss_demo": 1.0,
        "ruido_demo_normalizado": 0.05,
        "score_base": nota(matriz(demo_y, base_pred)),
        "score_oof": nota(confusion),
        "accuracy_base": float((base_pred == demo_y).mean()),
        "accuracy_oof": float((predicciones == demo_y).mean()),
        "aciertos_por_clase_oof": np.diag(confusion).tolist(),
        "confusion_oof": confusion.tolist(),
        "arreglados": int(((base_pred != demo_y) & (predicciones == demo_y)).sum()),
        "estropeados": int(((base_pred == demo_y) & (predicciones != demo_y)).sum()),
        "filtro_predefinido": {"score_min": 4.2, "accuracy_min": 0.865,
                               "aciertos_clase_6_min": 35, "aciertos_clase_7_min": 32,
                               "superado": bool(pasa_filtro)},
        "predicciones_base": base_pred.tolist(),
        "predicciones_oof": predicciones.tolist(),
    }
    comprobar_disco(args.min_libre_gib)
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OOF: score {informe['score_oof']:.6f}; acierto {informe['accuracy_oof']:.2%}; "
          f"arreglados {informe['arreglados']}, estropeados {informe['estropeados']}", flush=True)
    print(f"Filtro predefinido para considerar un modelo final: {'SUPERADO' if pasa_filtro else 'NO superado'}",
          flush=True)
    print(f"Informe: {destino.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
