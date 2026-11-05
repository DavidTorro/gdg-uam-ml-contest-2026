"""Ejecuta exp048 y exp049 secuencialmente y corta si escasea el disco."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
EXPERIMENTOS = (
    ("exp048", ("--mixup-alpha", "0.2")),
    ("exp049", ("--peso-alineacion-demo", "0.5")),
)


def libre_gib() -> float:
    return shutil.disk_usage(ROOT).free / GIB


def detener_grupo(proceso: subprocess.Popen) -> None:
    if proceso.poll() is not None:
        return
    for senal, espera in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proceso.pid, senal)
        except ProcessLookupError:
            break
        try:
            proceso.wait(timeout=espera)
            break
        except subprocess.TimeoutExpired:
            continue


def comando(nombre: str, extra: tuple[str, str]) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts/ejecutar_experimento.py"),
        "--id", nombre, "--arquitectura", "cnn", "--muestras", "90000",
        "--epocas", "200", "--batch-size", "256", "--simular-examen",
        "--sin-atajos", "--equilibrar", "--elegir-epoca", "examen",
        "--validacion", "val_examen_v2", "--ancho", "1", "--lr", "0.002",
        "--semilla", "2026", "--runs", "1", *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-libre-gib", type=float, default=6.0,
                        help="Detener el entrenamiento si el espacio libre baja de este umbral")
    args = parser.parse_args()
    if not 0 < args.min_libre_gib < 1000:
        parser.error("--min-libre-gib debe ser positivo y razonable")
    for nombre, _ in EXPERIMENTOS:
        for ruta in (ROOT / f"models/{nombre}.pt", ROOT / f"experiments/{nombre}",
                     ROOT / f"results/{nombre}"):
            if ruta.exists():
                parser.error(f"Ya existe {ruta}; no se sobrescribirá")
    disponible = libre_gib()
    print(f"Espacio libre inicial: {disponible:.2f} GiB; límite: {args.min_libre_gib:.2f} GiB",
          flush=True)
    if disponible < args.min_libre_gib:
        parser.error("Espacio libre insuficiente antes de empezar")

    for nombre, extra in EXPERIMENTOS:
        print(f"\nIniciando {nombre}: {extra[0]} {extra[1]}", flush=True)
        proceso = subprocess.Popen(comando(nombre, extra), cwd=ROOT, start_new_session=True)
        try:
            while True:
                estado = proceso.poll()
                if estado is not None:
                    if estado:
                        print(f"{nombre} terminó con código {estado}; se detiene la serie", flush=True)
                        return estado
                    print(f"{nombre} terminado. Espacio libre: {libre_gib():.2f} GiB", flush=True)
                    break
                disponible = libre_gib()
                if disponible < args.min_libre_gib:
                    print(f"DISCO BAJO: {disponible:.2f} GiB < {args.min_libre_gib:.2f} GiB. "
                          f"Deteniendo {nombre} y cancelando los siguientes.", flush=True)
                    detener_grupo(proceso)
                    return 2
                time.sleep(5)
        except KeyboardInterrupt:
            print(f"Interrupción manual: deteniendo {nombre}", flush=True)
            detener_grupo(proceso)
            return 130
        if libre_gib() < args.min_libre_gib:
            print("Disco por debajo del límite; no se inicia el siguiente experimento", flush=True)
            return 2
    print("\nExp048 y exp049 completos. Entrega actual sin modificar.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
