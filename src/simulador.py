"""Simulador de examen: transforma muestras de entrenamiento para que se parezcan al test.

El fichero de entrenamiento trae pistas que el test no tiene (desplazamientos y tamaños
de canal ligados a la clase) y está más limpio que el test. Aquí se rompen esas pistas
con cambios aleatorios iguales para todas las clases y se añade la suciedad medida en
test_demo: ruido rápido, picos, cambios de volumen y huecos. No usa las etiquetas.

Versión 2: la v1 (exp022) cambiaba canales de las clases 3 y 6 por los de otra muestra,
y eso dejaba una huella que delataba esas clases. La v2 imita sus trampas en todas las
clases por igual (mezclas), así dejan de distinguir ninguna clase.
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
    # Mezclas de entrenamiento, iguales para todas las clases. Imitan las trampas de train:
    # la clase 6 lleva una señal lenta común en los canales 1 y 2, y la clase 3 tiene
    # cambiado el ritmo de los canales 2 y 3
    "prob_comun_12": 0.5,
    "comun_12_max": 0.9,
    "prob_ritmo_23": 0.7,
    "ritmo_23_max": 2.4,
}

def mezclar(x: np.ndarray, rng: np.random.Generator, p: dict) -> np.ndarray:
    """Aplica a muestras al azar de cualquier clase las mezclas que en train solo tienen la 3 y la 6"""
    n, _, largo = x.shape
    t = np.arange(largo)

    # Señal lenta común en los canales 1 y 2: el canal 3 de otra muestra con volumen variable
    elegidas = np.where(rng.random(n) < p["prob_comun_12"])[0]
    if len(elegidas):
        donante = np.nan_to_num(x[rng.integers(0, n, len(elegidas)), 3])
        donante /= donante.std(axis=1, keepdims=True) + 1e-9
        donante *= 1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0, (len(elegidas), 1)) * t / largo
                                      + rng.uniform(0.0, 2 * np.pi, (len(elegidas), 1)))
        with np.errstate(all="ignore"):
            tamano = np.nan_to_num(np.nanstd(x[elegidas, 0], axis=1, keepdims=True), nan=1.0)
        comun = donante * rng.uniform(0.3, p["comun_12_max"], (len(elegidas), 1)) * tamano
        x[elegidas, 1] += comun
        x[elegidas, 2] += comun

    # Ritmo cambiado en los canales 2 y 3: inclina su espectro hacia lo lento o lo rápido
    for canal in (2, 3):
        elegidas = np.where(rng.random(n) < p["prob_ritmo_23"])[0]
        if not len(elegidas):
            continue
        senal = x[elegidas, canal]
        faltan = ~np.isfinite(senal)
        with np.errstate(all="ignore"):
            tamano = np.nanstd(senal, axis=1, keepdims=True)
        espectro = np.fft.rfft(np.where(faltan, 0.0, senal), axis=1)
        frecuencias = np.arange(espectro.shape[1]) / espectro.shape[1]
        inclinacion = rng.uniform(-p["ritmo_23_max"], p["ritmo_23_max"], (len(elegidas), 1))
        nueva = np.fft.irfft(espectro * np.exp(inclinacion * (frecuencias - 0.25) * 4), n=largo, axis=1)
        nueva *= np.nan_to_num(tamano, nan=1.0) / (nueva.std(axis=1, keepdims=True) + 1e-9)
        x[elegidas, canal] = np.where(faltan, np.nan, nueva)
    return x


def simular_examen(series: np.ndarray, rng: np.random.Generator,
                   parametros: dict | None = None, mezclas: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Recibe (N, 4, 256) con NaN y devuelve la serie simulada y su nuevo level.

    mezclas solo es para entrenar: el test no las tiene y val_examen tampoco las lleva.
    """
    p = {**PARAMETROS, **(parametros or {})}
    x = np.array(series, dtype=np.float64)
    n, canales, largo = x.shape
    t = np.arange(largo)

    # Quitamos el desplazamiento original, que en train depende de la clase
    with np.errstate(all="ignore"):
        x -= np.nanmean(x, axis=2, keepdims=True)
    if mezclas:
        x = mezclar(x, rng, p)

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
