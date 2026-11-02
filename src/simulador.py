"""Simulador de examen: transforma muestras de entrenamiento para que se parezcan al test.

El fichero de entrenamiento trae pistas que el test no tiene (desplazamientos y tamaños
de canal ligados a la clase) y está más limpio que el test. Aquí se rompen esas pistas
con cambios aleatorios iguales para todas las clases y se añade la suciedad medida en
test_demo: ruido rápido, picos, cambios de volumen y huecos. No usa las etiquetas.
"""

from __future__ import annotations

import numpy as np

# Valores medidos comparando train_90k con test_demo; scripts/calibrar_simulador.py los comprueba
PARAMETROS = {
    # Desplazamiento común de los 4 canales, igual que el meta level del test
    "nivel_desv": 2.24,
    "desplazamiento_canal_desv": 0.3,
    # Tamaño de la señal: factor global y factor propio de cada canal (escala logarítmica)
    "ganancia_media": 0.1,
    "ganancia_desv": 0.37,
    "ganancia_canal_desv": 0.27,
    # Ruido rápido, como fracción del tamaño de cada canal
    "ruido_min": 0.1,
    "ruido_max": 0.5,
    # Cambios de volumen lentos a lo largo de la muestra
    "envolvente_max": 0.3,
    # Picos aislados, en tamaños de canal
    "prob_picos": 0.35,
    "picos_media": 3.0,
    "pico_mediana": 4.5,
    "pico_desv_log": 0.58,
    # Huecos nuevos por canal; la mayoría de un punto y algunos largos
    "huecos_por_canal": 0.56,
    "prob_hueco_largo": 0.25,
    "hueco_largo_max": 16,
}

# Canales cuya forma solo distingue la clase en train: en el demo se parecen a los del resto
TRASPLANTES = {3: (2, 3), 6: (2,)}


def trasplantar(series: np.ndarray, etiquetas: np.ndarray, rng: np.random.Generator,
                trasplantes: dict[int, tuple[int, ...]] = TRASPLANTES) -> np.ndarray:
    """Cambia esos canales por los de una muestra al azar de otra clase. Sí usa etiquetas"""
    x = np.array(series, copy=True)
    for clase, canales in trasplantes.items():
        destino = np.where(etiquetas == clase)[0]
        otras = np.where(etiquetas != clase)[0]
        if not len(destino) or not len(otras):
            continue
        origen = rng.choice(otras, len(destino))
        for canal in canales:
            x[destino, canal] = series[origen, canal]
    return x


def simular_examen(series: np.ndarray, rng: np.random.Generator,
                   parametros: dict | None = None,
                   etiquetas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Recibe (N, 4, 256) con NaN y devuelve la serie simulada y su nuevo level.

    Si se pasan etiquetas, también aplica los trasplantes de canal de TRASPLANTES.
    """
    p = {**PARAMETROS, **(parametros or {})}
    if etiquetas is not None:
        series = trasplantar(series, etiquetas, rng)
    x = np.array(series, dtype=np.float64)
    n, canales, largo = x.shape
    t = np.arange(largo)

    # Quitamos el desplazamiento original, que en train depende de la clase
    with np.errstate(all="ignore"):
        x -= np.nanmean(x, axis=2, keepdims=True)

    # Tamaño aleatorio: rompe la pista de canales más grandes en ciertas clases
    ganancia = np.exp(rng.normal(p["ganancia_media"], p["ganancia_desv"], (n, 1, 1))
                      + rng.normal(0.0, p["ganancia_canal_desv"], (n, canales, 1)))
    x *= ganancia

    # Volumen que sube y baja despacio
    amplitud = rng.uniform(0.0, p["envolvente_max"], (n, canales, 1))
    frecuencia = rng.uniform(0.5, 3.0, (n, canales, 1))
    fase = rng.uniform(0.0, 2 * np.pi, (n, canales, 1))
    x *= 1.0 + amplitud * np.sin(2 * np.pi * frecuencia * t / largo + fase)

    # Ruido rápido proporcional al tamaño de cada canal
    with np.errstate(all="ignore"):
        tamano = np.nan_to_num(np.nanstd(x, axis=2, keepdims=True), nan=1.0)
    x += rng.normal(0.0, 1.0, x.shape) * rng.uniform(p["ruido_min"], p["ruido_max"], (n, 1, 1)) * tamano

    # Picos aislados en algunas muestras
    con_picos = rng.random(n) < p["prob_picos"]
    cuantos = np.where(con_picos, np.maximum(1, rng.poisson(p["picos_media"], n)), 0)
    total = int(cuantos.sum())
    if total:
        muestra = np.repeat(np.arange(n), cuantos)
        canal = rng.integers(0, canales, total)
        instante = rng.integers(0, largo, total)
        signo = rng.choice((-1.0, 1.0), total)
        altura = np.exp(rng.normal(np.log(p["pico_mediana"]), p["pico_desv_log"], total))
        np.add.at(x, (muestra, canal, instante), signo * altura * tamano[muestra, canal, 0])

    # Nuevo desplazamiento independiente de la clase
    nivel = rng.normal(0.0, p["nivel_desv"], n)
    x += nivel[:, None, None] + rng.normal(0.0, p["desplazamiento_canal_desv"], (n, canales, 1))

    # Huecos nuevos: se suman a los que ya traía la muestra
    maximo = 4
    huecos = np.minimum(rng.poisson(p["huecos_por_canal"], (n, canales)), maximo)
    inicio = rng.integers(0, largo, (n, canales, maximo))
    largos = np.where(rng.random((n, canales, maximo)) < p["prob_hueco_largo"],
                      rng.integers(2, p["hueco_largo_max"] + 1, (n, canales, maximo)), 1)
    activo = np.arange(maximo) < huecos[..., None]
    dentro = (t >= inicio[..., None]) & (t < (inicio + largos)[..., None]) & activo[..., None]
    x[dentro.any(axis=2)] = np.nan

    return x.astype(np.float32), nivel
