"""Entrena una CNN 1D sin utilizar la validación oficial ni el test demo.

Sin opciones nuevas reproduce exp015 a exp020: separa un 10 % de control interno y elige
la época de menor pérdida en él. Desde exp022 se puede entrenar como para el examen:
--simular-examen pasa cada época por src/simulador.py, --sin-atajos anula las auxiliares
que delatan la clase solo en train, --equilibrar hace que todas las clases pesen igual y
--elegir-epoca examen elige la época con la mejor nota en el archivo de --archivo-examen.
"""

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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.red_convolucional import RedConvolucional, preparar_lote
from src.simulador import PARAMETROS, simular_examen


def aumentar_senales(serie: torch.Tensor, auxiliares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simula ruido rápido y huecos extra únicamente en los lotes de entrenamiento."""
    cantidad, canales, _ = serie.shape
    activo = (torch.rand((cantidad, 1, 1), device=serie.device) < 0.7).to(serie.dtype)
    intensidad = (0.08 + 0.22 * torch.rand((cantidad, canales, 1), device=serie.device)) * activo
    aumentada = serie + torch.randn_like(serie) * intensidad

    # Sustituimos pocos puntos por la media de sus vecinos, como una
    # aproximación de la interpolación que se hace con los huecos reales.
    huecos = torch.rand(serie.shape, device=serie.device) < 0.008
    izquierda = torch.cat((aumentada[:, :, :1], aumentada[:, :, :-1]), dim=2)
    derecha = torch.cat((aumentada[:, :, 1:], aumentada[:, :, -1:]), dim=2)
    aumentada = torch.where(huecos, (izquierda + derecha) * 0.5, aumentada)
    auxiliares = auxiliares.clone()
    auxiliares[:, 8:12] = torch.clamp(
        auxiliares[:, 8:12] + huecos.float().mean(dim=2), max=1.0
    )
    return aumentada.clamp(-6.0, 6.0), auxiliares


def nota_oficial(confusion: np.ndarray) -> float:
    """Fórmula del concurso sobre una matriz de confusión, sin el efecto de la velocidad"""
    n = confusion.sum(axis=1)
    c = np.diag(confusion)
    t = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, t)).mean()))


def preparar(datos: dict, indices: np.ndarray, sin_atajos: bool, series: np.ndarray | None = None,
             nivel: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepara señales y auxiliares en bloques para no disparar la memoria"""
    partes_s, partes_a = [], []
    for inicio in range(0, len(indices), 20_000):
        bloque = indices[inicio:inicio + 20_000]
        s, a = preparar_lote(
            datos["series"][bloque] if series is None else series[inicio:inicio + 20_000],
            datos["meta__sample_rate"][bloque],
            datos["meta__level"][bloque] if nivel is None else nivel[inicio:inicio + 20_000],
            datos["meta__group"][bloque], sin_atajos=sin_atajos,
        )
        partes_s.append(s)
        partes_a.append(a)
    return torch.from_numpy(np.concatenate(partes_s)), torch.from_numpy(np.concatenate(partes_a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True)
    parser.add_argument("--muestras", type=int, default=90_000)
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aumento-ruido", action="store_true", help="Añade ruido y huecos nuevos a los lotes de entrenamiento")
    parser.add_argument("--simular-examen", action="store_true", help="Pasa cada época por el simulador de examen")
    parser.add_argument("--sin-atajos", action="store_true", help="Anula las medias y el level en las auxiliares")
    parser.add_argument("--equilibrar", action="store_true", help="Todas las clases pesan igual en la pérdida")
    parser.add_argument("--elegir-epoca", choices=("control", "examen"), default="control",
                        help="control: menor pérdida en el 10 %% interno; examen: mejor nota en --archivo-examen")
    parser.add_argument("--archivo-examen", default="data/val_examen_v2.npz",
                        help="Examen de prácticas para elegir la época; val_examen.npz es el de exp022, con huella")
    parser.add_argument("--ancho", type=int, default=1, help="Multiplica los filtros de la red")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--semilla", type=int, default=2026)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("--id debe usar letras minúsculas, números, guiones o guiones bajos")
    if args.epocas < 1 or args.batch_size < 1 or args.ancho < 1:
        parser.error("--epocas, --batch-size y --ancho deben ser positivos")

    fuente = ROOT / "data/train_90k.npz"
    examen = ROOT / args.archivo_examen
    if args.elegir_epoca == "examen" and not examen.is_file():
        parser.error(f"Falta {args.archivo_examen}: créalo con scripts/calibrar_simulador.py --guardar-val")
    datos = dict(np.load(fuente))
    total = len(datos["labels"])
    if not 80 <= args.muestras <= total:
        parser.error(f"--muestras debe estar entre 80 y {total}")
    destino = ROOT / "models" / f"{args.id}.pt"
    carpeta = ROOT / "experiments" / args.id
    informe = carpeta / "entrenamiento.json"
    if destino.exists() or informe.exists():
        parser.error(f"El experimento {args.id} ya existe; usa otro --id")

    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    rng = np.random.default_rng(args.semilla)
    indices = np.arange(total)
    if args.muestras < total:
        indices, _ = train_test_split(indices, train_size=args.muestras, stratify=datos["labels"], random_state=2026)
    indices = np.sort(indices)
    etiquetas_todas = np.asarray(datos["labels"][indices], dtype=np.int64)

    # Con control interno se reserva un 10 %; eligiendo con val_examen se entrena con todo
    if args.elegir_epoca == "control":
        entrenamiento_pos, control_pos = train_test_split(
            np.arange(len(indices)), test_size=0.1, stratify=etiquetas_todas, random_state=2027
        )
    else:
        entrenamiento_pos, control_pos = np.arange(len(indices)), np.array([], dtype=np.int64)
    entrenamiento_idx = indices[np.sort(entrenamiento_pos)]
    etiquetas = np.asarray(datos["labels"][entrenamiento_idx], dtype=np.int64)

    inicio_total = perf_counter()
    if args.elegir_epoca == "control":
        control_idx = indices[np.sort(control_pos)]
        control_s, control_a = preparar(datos, control_idx, args.sin_atajos)
        control_y = torch.from_numpy(np.asarray(datos["labels"][control_idx], dtype=np.int64))
        nombre_control = "control interno"
    else:
        datos_examen = dict(np.load(examen))
        todos = np.arange(len(datos_examen["labels"]))
        control_s, control_a = preparar(datos_examen, todos, args.sin_atajos)
        control_y = torch.from_numpy(np.asarray(datos_examen["labels"], dtype=np.int64))
        nombre_control = "examen de prácticas"
    fijas = None if args.simular_examen else preparar(datos, entrenamiento_idx, args.sin_atajos)
    segundos_preparacion = perf_counter() - inicio_total

    dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    red = RedConvolucional(ancho=args.ancho).to(dispositivo)
    optimizador = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)
    conteo = np.bincount(etiquetas, minlength=8)
    pesos_clase = torch.tensor(len(etiquetas) / (8 * conteo), dtype=torch.float32, device=dispositivo) if args.equilibrar else None
    criterio = nn.CrossEntropyLoss(weight=pesos_clase)
    criterio_control = nn.CrossEntropyLoss()
    mejor = {"epoca": 0, "perdida": float("inf"), "nota": -1.0}
    mejores_pesos = None
    historial = []
    segundos_simulador = 0.0
    inicio_ajuste = perf_counter()
    carpeta.mkdir(parents=True, exist_ok=True)
    progreso = carpeta / "progreso.json"

    for epoca in range(1, args.epocas + 1):
        inicio_epoca = perf_counter()
        if args.simular_examen:
            # Cada época ve una versión distinta y sucia de las mismas muestras
            antes = perf_counter()
            series, niveles = [], []
            for inicio in range(0, len(entrenamiento_idx), 10_000):
                bloque = entrenamiento_idx[inicio:inicio + 10_000]
                s, nv = simular_examen(datos["series"][bloque], rng)
                series.append(s)
                niveles.append(nv)
            serie_e, aux_e = preparar(datos, entrenamiento_idx, args.sin_atajos,
                                      series=np.concatenate(series), nivel=np.concatenate(niveles))
            segundos_simulador += perf_counter() - antes
        else:
            serie_e, aux_e = fijas
        cargador = DataLoader(TensorDataset(serie_e, aux_e, torch.from_numpy(etiquetas)),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)

        red.train()
        perdida_entrenamiento = 0.0
        for serie, auxiliar, etiqueta in cargador:
            serie = serie.to(dispositivo)
            auxiliar = auxiliar.to(dispositivo)
            etiqueta = etiqueta.to(dispositivo)
            if args.aumento_ruido:
                serie, auxiliar = aumentar_senales(serie, auxiliar)
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(red(serie, auxiliar), etiqueta)
            perdida.backward()
            optimizador.step()
            perdida_entrenamiento += perdida.item() * len(etiqueta)
        plan.step()

        red.eval()
        perdida_control = 0.0
        confusion = np.zeros((8, 8), dtype=np.int64)
        with torch.inference_mode():
            for inicio in range(0, len(control_y), 2048):
                fin = inicio + 2048
                y = control_y[inicio:fin].to(dispositivo)
                salida = red(control_s[inicio:fin].to(dispositivo), control_a[inicio:fin].to(dispositivo))
                perdida_control += criterio_control(salida, y).item() * len(y)
                np.add.at(confusion, (y.cpu().numpy(), salida.argmax(dim=1).cpu().numpy()), 1)
        perdida_control /= len(control_y)
        acierto_clase = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
        fila = {
            "epoca": epoca,
            "perdida_entrenamiento": perdida_entrenamiento / len(etiquetas),
            "perdida_control": perdida_control,
            "acierto_control": float(np.trace(confusion) / confusion.sum()),
            "nota_control": nota_oficial(confusion),
            "acierto_por_clase": [round(float(v), 4) for v in acierto_clase],
            "segundos": perf_counter() - inicio_epoca,
        }
        historial.append(fila)
        mejora = (fila["nota_control"] > mejor["nota"]) if args.elegir_epoca == "examen" else (perdida_control < mejor["perdida"])
        if mejora:
            mejor = {"epoca": epoca, "perdida": perdida_control, "nota": fila["nota_control"]}
            mejores_pesos = {nombre: tensor.detach().cpu().clone() for nombre, tensor in red.state_dict().items()}
        print(
            f"Época {epoca}/{args.epocas}: pérdida {fila['perdida_entrenamiento']:.4f} · {nombre_control}: "
            f"acierto {fila['acierto_control']:.2%}, nota {fila['nota_control']:.3f}, clase 6 {acierto_clase[6]:.0%}"
            f"{' ← mejor' if mejora else ''} ({fila['segundos']:.0f} s)",
            flush=True,
        )
        # Estado en directo para seguir el entrenamiento desde fuera
        progreso.write_text(json.dumps({
            "id": args.id, "epoca": epoca, "epocas": args.epocas, "mejor_epoca": mejor["epoca"],
            "elegir_epoca": args.elegir_epoca, "historial": historial,
            "segundos_transcurridos": perf_counter() - inicio_total,
        }, ensure_ascii=False) + "\n", encoding="utf-8")

    segundos_ajuste = perf_counter() - inicio_ajuste
    destino.parent.mkdir(exist_ok=True)
    torch.save({
        "formato": "gdg-cnn-v1", "state_dict": mejores_pesos,
        "ancho": args.ancho, "sin_atajos": args.sin_atajos,
    }, destino)
    resumen = {
        "modelo": "Red convolucional 1D" + (" · simulador de examen" if args.simular_examen else ""),
        "id": args.id,
        "fuente": fuente.name,
        "muestras_seleccionadas": int(len(indices)),
        "muestras_entrenamiento": int(len(entrenamiento_idx)),
        "muestras_control_interno": int(len(control_pos)),
        "aumento_ruido": args.aumento_ruido,
        "simular_examen": args.simular_examen,
        "parametros_simulador": PARAMETROS if args.simular_examen else None,
        "sin_atajos": args.sin_atajos,
        "equilibrar": args.equilibrar,
        "elegir_epoca": args.elegir_epoca,
        "archivo_examen": args.archivo_examen if args.elegir_epoca == "examen" else None,
        "ancho": args.ancho,
        "lr": args.lr,
        "clases_entrenamiento": conteo.tolist(),
        "caracteristicas_por_muestra": 4 * 256 + 17,
        "iteraciones": args.epocas,
        "mejor_epoca": mejor["epoca"],
        "perdida_control_interno": mejor["perdida"],
        "nota_control_mejor": mejor["nota"],
        "historial": historial,
        "semilla": args.semilla,
        "dispositivo_entrenamiento": str(dispositivo),
        "segundos_caracteristicas": segundos_preparacion + segundos_simulador,
        "segundos_ajuste": segundos_ajuste - segundos_simulador,
        "segundos_entrenamiento_total": segundos_preparacion + segundos_ajuste,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pesos": str(destino.relative_to(ROOT)),
    }
    informe.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nMejor época: {mejor['epoca']} · nota en {nombre_control} {mejor['nota']:.3f} · "
          f"{resumen['segundos_entrenamiento_total'] / 60:.1f} min · {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
