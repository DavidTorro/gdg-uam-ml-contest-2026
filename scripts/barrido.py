"""Entrena varias configuraciones de la CNN seguidas, por ejemplo durante la noche.

Cada configuración usa la receta de exp023 (simulador v2, sin atajos, clases iguales y
época elegida con val_examen_v2) y cambia el ancho, las épocas, la semilla, la tasa de
aprendizaje o los ajustes del simulador. Al terminar cada una:
- calcula la nota estimada a 60 s, que tiene en cuenta la velocidad como el concurso,
- la importa en el dashboard si encuentra ../ml-dashboard,
- guarda el ranking en experiments/barrido/<fecha>.json.
Si una configuración falla, sigue con la siguiente. No usa test_demo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT.parent / "ml-dashboard"

# Plan por defecto; cada entrada solo indica lo que cambia respecto a exp023
PLAN = [
    {"nombre": "ancho 2", "ancho": 2, "epocas": 150},
    {"nombre": "ancho 3", "ancho": 3, "epocas": 150},
    {"nombre": "ancho 4", "ancho": 4, "epocas": 150},
    {"nombre": "ancho 1 · 200 épocas", "ancho": 1, "epocas": 200},
    {"nombre": "ancho 3 · 300 épocas", "ancho": 3, "epocas": 300},
    {"nombre": "ancho 3 · otra semilla", "ancho": 3, "epocas": 150, "semilla": 2027},
    {"nombre": "como exp023 · otra semilla", "ancho": 1, "epocas": 100, "semilla": 2027},
    {"nombre": "ancho 3 · mezclas suaves", "ancho": 3, "epocas": 150,
     "ajustes": {"prob_ritmo_23": 0.5, "ritmo_23_max": 1.6}},
    {"nombre": "ancho 3 · lr 0,002", "ancho": 3, "epocas": 150, "lr": 0.002},
]

COMUNES = ["--arquitectura", "cnn", "--muestras", "90000", "--batch-size", "256", "--simular-examen",
           "--sin-atajos", "--equilibrar", "--elegir-epoca", "examen", "--validacion", "val_examen_v2", "--runs", "1"]


def ids_libres(cantidad: int) -> list[str]:
    """Siguientes expNNN que no existen en models, experiments ni results"""
    usados = {ruta.stem for carpeta in ("models", "experiments", "results") for ruta in (ROOT / carpeta).glob("exp[0-9]*")}
    numero = max([int(nombre[3:6]) for nombre in usados if nombre[3:6].isdigit()] + [0]) + 1
    return [f"exp{numero + i:03d}" for i in range(cantidad)]


def minutos_estimados(config: dict) -> float:
    # Medido: ~5 s por época con ancho 1 y ~6 s con ancho 3; el simulador pesa más que la red
    return config["epocas"] * (4.5 + 0.5 * config.get("ancho", 1)) / 60 + 1.5


def nota_60s(confusion: np.ndarray, latencia_s: float) -> float:
    """Fórmula oficial escalando las muestras a las que respondería en 60 s"""
    escala = (60.0 / latencia_s) / confusion.sum()
    n = confusion.sum(axis=1) * escala
    c = np.diag(confusion) * escala
    t = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, t)).mean()))


def resultado(id_: str) -> dict | None:
    try:
        run = json.loads((ROOT / "results" / id_ / "model" / "run_0.json").read_text())
        entrenamiento = json.loads((ROOT / "experiments" / id_ / "entrenamiento.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    confusion = np.array(run["confusion"])[:, :8].astype(float)
    latencia = run["latency_s"]["mean"]
    return {
        "nota_practicas": run["score"],
        "acierto": run["accuracy_all"],
        "latencia_ms": latencia * 1000,
        "nota_60s": nota_60s(confusion, latencia),
        "acierto_por_clase": np.round(np.diag(confusion) / confusion.sum(axis=1), 3).tolist(),
        "mejor_epoca": entrenamiento.get("mejor_epoca"),
        "minutos": entrenamiento.get("segundos_entrenamiento_total", 0) / 60,
    }


def importar(id_: str, nombre: str) -> str:
    """Importa en el dashboard y le pone un nombre legible; nunca detiene el barrido"""
    importador = DASHBOARD / "scripts" / "importar_experimento.mjs"
    if not importador.is_file() or shutil.which("node") is None:
        return "sin dashboard"
    salida = subprocess.run(["node", str(importador), id_], cwd=DASHBOARD, capture_output=True, text=True)
    if salida.returncode != 0:
        return "falló la importación"
    anotaciones = DASHBOARD / "data" / "anotaciones.json"
    try:
        datos = json.loads(anotaciones.read_text(encoding="utf-8")) if anotaciones.is_file() else {}
        datos.setdefault(id_, {"name": f"PRÁCTICAS v2 · {id_} · {nombre}"[:40],
                               "note": "Barrido nocturno con la receta de exp023"})
        temporal = anotaciones.with_suffix(".tmp")
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporal.replace(anotaciones)
    except (OSError, json.JSONDecodeError):
        return "importado sin nombre"
    return "importado"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", help="JSON con una lista de configuraciones; por defecto, el PLAN del script")
    parser.add_argument("--solo-mostrar", action="store_true", help="Muestra el plan y el tiempo estimado sin entrenar")
    parser.add_argument("--sin-importar", action="store_true", help="No importa en el dashboard")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else PLAN
    ids = ids_libres(len(plan))
    total = sum(minutos_estimados(config) for config in plan)
    print(f"Barrido de {len(plan)} configuraciones · unos {total / 60:.1f} h en total")
    for id_, config in zip(ids, plan):
        print(f"  {id_}: {config['nombre']} (~{minutos_estimados(config):.0f} min)")
    if args.solo_mostrar:
        return

    fecha = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    registro = ROOT / "experiments" / "barrido" / f"{fecha}.json"
    registro.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    for numero, (id_, config) in enumerate(zip(ids, plan), start=1):
        print(f"\n===== {numero}/{len(plan)} · {id_} · {config['nombre']} =====", flush=True)
        orden = [sys.executable, str(ROOT / "scripts/ejecutar_experimento.py"), "--id", id_, *COMUNES,
                 "--ancho", str(config.get("ancho", 1)), "--epocas", str(config["epocas"]),
                 "--lr", str(config.get("lr", 0.001)), "--semilla", str(config.get("semilla", 2026)),
                 "--ajustes-simulador", json.dumps(config.get("ajustes", {}))]
        inicio = perf_counter()
        estado = "ok" if subprocess.run(orden, cwd=ROOT).returncode == 0 else "falló"
        fila = {"id": id_, **config, "estado": estado, "minutos_reales": (perf_counter() - inicio) / 60}
        datos = resultado(id_) if estado == "ok" else None
        if datos:
            fila.update(datos)
            if not args.sin_importar:
                fila["dashboard"] = importar(id_, config["nombre"])
        filas.append(fila)
        registro.write_text(json.dumps({"fecha_utc": fecha, "filas": filas}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

        # Ranking provisional: la nota a 60 s premia acertar y también ser rápido
        print("\nRanking hasta ahora (nota estimada a 60 s en val_examen_v2):")
        for fila in sorted((f for f in filas if "nota_60s" in f), key=lambda f: -f["nota_60s"]):
            print(f"  {fila['id']} · {fila['nombre']:28s} nota 60 s {fila['nota_60s']:7.1f} · acierto "
                  f"{fila['acierto']:.1%} · {fila['latencia_ms']:.3f} ms · clase 6 {fila['acierto_por_clase'][6]:.0%}")
        for fila in filas:
            if fila["estado"] != "ok":
                print(f"  {fila['id']} · {fila['nombre']}: falló")
    print(f"\nBarrido terminado. Registro en {registro.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
