"""Comprueba si el simulador de examen hace que train se parezca a test_demo.

Calcula rasgos sencillos de cada muestra (ruido, picos, huecos, relaciones entre
canales...) en train, en train simulado y en el demo, y guarda un informe JSON que
dibuja la sección Simulador del dashboard. No entrena ni evalúa ningún modelo.
Con --guardar-val crea además data/val_examen.npz: val_10k pasado por el simulador.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.simulador import PARAMETROS, simular_examen

BANDAS = [(1, 4, "ritmo muy lento"), (4, 8, "ritmo lento"), (8, 16, "ritmo medio"),
          (16, 32, "ritmo rápido"), (32, 64, "ritmo muy rápido"), (64, 129, "ruido rápido")]


def rasgos(series: np.ndarray, nivel: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Devuelve una matriz (muestras, rasgos) y, por rasgo, su clave, nombre y grupo."""
    crudo = series.astype(np.float64)
    with np.errstate(all="ignore"):
        x = np.where(np.isfinite(crudo), crudo, np.nanmean(crudo, axis=2, keepdims=True))
    x = np.nan_to_num(x)
    media = x.mean(2)
    tamano = x.std(2) + 1e-9
    z = (x - media[:, :, None]) / tamano[:, :, None]
    columnas, info = [], []

    def anadir(valor, clave, nombre, grupo):
        columnas.append(valor)
        info.append((clave, nombre, grupo))

    anadir(nivel, "level", "Nivel (dato level)", "desplazamiento")
    for c in range(4):
        anadir(media[:, c] - nivel, f"desplazamiento_c{c}", f"Desplazamiento propio · canal {c}", "desplazamiento")
        anadir(np.log(tamano[:, c]), f"tamano_c{c}", f"Tamaño · canal {c}", "tamaño")
    for c in range(1, 4):
        anadir(np.log(tamano[:, c] / tamano[:, 0]), f"proporcion_c{c}", f"Tamaño canal {c} frente al 0", "tamaño")
    for a in range(4):
        for b in range(a + 1, 4):
            anadir((z[:, a] * z[:, b]).mean(1), f"relacion_c{a}c{b}", f"Relación canales {a} y {b}", "relación entre canales")
    potencia = np.abs(np.fft.rfft(z, axis=2)) ** 2
    potencia /= potencia.sum(2, keepdims=True)
    for c in range(4):
        for bajo, alto, texto in BANDAS:
            anadir(np.log(potencia[:, c, bajo:alto].sum(1) + 1e-6), f"banda{bajo}-{alto}_c{c}",
                   f"{texto[0].upper()}{texto[1:]} · canal {c}", "ruido" if bajo >= 64 else "ritmo")
        anadir(np.abs(z[:, c]).max(1), f"pico_c{c}", f"Pico más alto · canal {c}", "picos")
        anadir((z[:, c] ** 4).mean(1), f"extremos_c{c}", f"Valores extremos · canal {c}", "picos")
        envolvente = z[:, c].reshape(len(z), 8, 32).std(2)
        anadir(envolvente.std(1) / (envolvente.mean(1) + 1e-9), f"volumen_c{c}", f"Cambios de volumen · canal {c}", "volumen")
    anadir(np.isnan(crudo).mean((1, 2)), "huecos", "Huecos (valores que faltan)", "huecos")
    return np.stack(columnas, 1), info


def efecto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diferencia de medias en desviaciones típicas (d de Cohen), por rasgo."""
    escala = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9
    return (a.mean(0) - b.mean(0)) / escala


def ejemplo(serie: np.ndarray) -> list[list[float | None]]:
    return [[None if not np.isfinite(v) else round(float(v), 3) for v in canal] for canal in serie]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muestras", type=int, default=20_000)
    parser.add_argument("--semilla", type=int, default=2026)
    parser.add_argument("--guardar-val", action="store_true")
    parser.add_argument("--sin-trasplantes", action="store_true", help="No cambia canales de las clases 3 y 6")
    args = parser.parse_args()

    rng = np.random.default_rng(args.semilla)
    train = np.load(ROOT / "data/train_90k.npz")
    demo = np.load(ROOT / "data/test_demo.npz")
    elegidos = np.sort(rng.choice(len(train["labels"]), args.muestras, replace=False))
    serie_t = train["series"][elegidos]
    nivel_t = train["meta__level"][elegidos]
    y_t = train["labels"][elegidos]
    y_d = demo["labels"]
    serie_s, nivel_s = simular_examen(serie_t, rng, etiquetas=None if args.sin_trasplantes else y_t)

    f_t, info = rasgos(serie_t, nivel_t)
    f_s, _ = rasgos(serie_s, nivel_s)
    f_d, _ = rasgos(demo["series"], demo["meta__level"])

    # Parecido global: cuánto se separa cada conjunto del demo en cada rasgo
    d_train = efecto(f_t, f_d)
    d_sim = efecto(f_s, f_d)
    globales = [{
        "clave": clave, "nombre": nombre, "grupo": grupo,
        "mediana_train": float(np.median(f_t[:, i])), "mediana_simulado": float(np.median(f_s[:, i])),
        "mediana_demo": float(np.median(f_d[:, i])),
        "distancia_train": float(abs(d_train[i])), "distancia_simulado": float(abs(d_sim[i])),
    } for i, (clave, nombre, grupo) in enumerate(info)]

    # Por clase: pistas que en train separan la clase y en el demo no (trampas) o sí (reales)
    trampas, reales = [], []
    for k in range(8):
        dt = efecto(f_t[y_t == k], f_t[y_t != k])
        ds = efecto(f_s[y_t == k], f_s[y_t != k])
        dd = efecto(f_d[y_d == k], f_d[y_d != k])
        for i, (clave, nombre, grupo) in enumerate(info):
            fila = {"clase": k, "clave": clave, "nombre": nombre, "grupo": grupo,
                    "d_train": float(dt[i]), "d_simulado": float(ds[i]), "d_demo": float(dd[i])}
            if abs(dt[i]) > 0.6 and abs(dd[i]) < 0.35 * abs(dt[i]):
                # Rota si el simulador la deja al menos a mitad de camino del demo
                fila["rota"] = bool(abs(ds[i] - dd[i]) <= 0.5 * abs(dt[i] - dd[i]))
                trampas.append(fila)
            elif abs(dd[i]) > 0.5 and np.sign(dd[i]) == np.sign(dt[i]) and abs(dt[i]) > 0.3:
                # Conservada si el simulador no le quita más de la mitad de su fuerza
                fila["conservada"] = bool(np.sign(ds[i]) == np.sign(dt[i]) and abs(ds[i]) >= 0.5 * abs(dt[i]))
                reales.append(fila)

    ejemplos = []
    for k in range(8):
        i = int(np.where(y_t == k)[0][0])
        j = int(np.where(y_d == k)[0][0])
        ejemplos.append({"clase": k, "original": ejemplo(serie_t[i]), "simulada": ejemplo(serie_s[i]),
                         "demo": ejemplo(demo["series"][j])})

    resumen = {
        "distancia_media_train": float(np.mean(np.abs(d_train))),
        "distancia_media_simulado": float(np.mean(np.abs(d_sim))),
        "rasgos_lejos_train": int((np.abs(d_train) > 0.25).sum()),
        "rasgos_lejos_simulado": int((np.abs(d_sim) > 0.25).sum()),
        "rasgos": len(info),
        "trampas": len(trampas),
        "trampas_rotas": sum(fila["rota"] for fila in trampas),
        "reales": len(reales),
        "reales_conservadas": sum(fila["conservada"] for fila in reales),
    }
    informe = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "muestras": {"train": int(len(y_t)), "demo": int(len(y_d))},
        "semilla": args.semilla,
        "parametros": PARAMETROS,
        "resumen": resumen,
        "globales": globales,
        "trampas": trampas,
        "reales": reales,
        "ejemplos": ejemplos,
    }
    destino = ROOT / "experiments/simulador/informe.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(informe, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Distancia media al demo: train {resumen['distancia_media_train']:.3f} → "
          f"simulado {resumen['distancia_media_simulado']:.3f}")
    print(f"Rasgos lejos del demo (|d|>0,25): train {resumen['rasgos_lejos_train']} → "
          f"simulado {resumen['rasgos_lejos_simulado']} de {resumen['rasgos']}")
    print(f"Trampas rotas: {resumen['trampas_rotas']} de {resumen['trampas']}; "
          f"pistas reales conservadas: {resumen['reales_conservadas']} de {resumen['reales']}")
    peores = sorted(globales, key=lambda fila: -fila["distancia_simulado"])[:8]
    print("Rasgos más lejos del demo tras simular:")
    for fila in peores:
        print(f"  {fila['nombre']:38s} train {fila['distancia_train']:.2f} → simulado {fila['distancia_simulado']:.2f}")
    for fila in trampas:
        if not fila["rota"]:
            print(f"  Trampa sin romper · clase {fila['clase']} · {fila['nombre']}: "
                  f"{fila['d_train']:+.2f} → {fila['d_simulado']:+.2f} (demo {fila['d_demo']:+.2f})")
    for fila in reales:
        if not fila["conservada"]:
            print(f"  Pista real debilitada · clase {fila['clase']} · {fila['nombre']}: "
                  f"{fila['d_train']:+.2f} → {fila['d_simulado']:+.2f} (demo {fila['d_demo']:+.2f})")
    print(f"Informe guardado en {destino.relative_to(ROOT)}")

    if args.guardar_val:
        val = np.load(ROOT / "data/val_10k.npz")
        serie_v, nivel_v = simular_examen(val["series"], np.random.default_rng(args.semilla + 1),
                                          etiquetas=None if args.sin_trasplantes else val["labels"])
        extra = {clave: val[clave] for clave in val.files if clave not in ("series", "meta__level")}
        np.savez(ROOT / "data/val_examen.npz", series=serie_v, meta__level=nivel_v, **extra)
        print("Guardado data/val_examen.npz")


if __name__ == "__main__":
    main()
