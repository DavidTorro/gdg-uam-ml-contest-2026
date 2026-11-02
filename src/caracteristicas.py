"""Convierte una serie de cuatro canales en un vector numérico para el clasificador"""

from __future__ import annotations

import numpy as np

from competition_api import Sample


def extraer(sample: Sample) -> np.ndarray:
    """Resume una muestra siempre con las mismas características y en el mismo orden"""
    series = sample.series
    tiempos = np.arange(series.shape[1])
    faltantes = ~np.isfinite(series)
    completa = np.empty(series.shape, dtype=np.float64)

    # Interpolamos los huecos por canal sin modificar la muestra original
    # Un canal completamente vacío se representa con ceros
    for canal in range(series.shape[0]):
        presentes = ~faltantes[canal]
        if presentes.any():
            completa[canal] = np.interp(
                tiempos, tiempos[presentes], series[canal, presentes]
            )
        else:
            completa[canal] = 0.0

    # Conservamos nivel y escala antes de normalizar la forma de la señal
    medias = completa.mean(axis=1)
    desviaciones = completa.std(axis=1)
    normalizada = (completa - medias[:, None]) / (desviaciones[:, None] + 1e-6)

    # Los cuantiles describen la distribución sin depender de un instante concreto
    cuantiles = np.percentile(normalizada, [10, 50, 90], axis=1).T

    # Las diferencias y la autocorrelación describen cambios rápidos y persistencia
    rugosidad = np.abs(np.diff(normalizada, axis=1)).mean(axis=1)
    autocorrelacion = (normalizada[:, 1:] * normalizada[:, :-1]).mean(axis=1)

    # La energía espectral resume cuánto varía la señal a distintas frecuencias
    potencia = np.abs(np.fft.rfft(normalizada, axis=1))[:, 1:] ** 2
    bandas = np.array_split(potencia, 4, axis=1)
    espectro = np.log1p(np.stack([banda.sum(axis=1) for banda in bandas], axis=1))

    # Las correlaciones permiten detectar relaciones entre pares de canales
    correlaciones = normalizada @ normalizada.T / normalizada.shape[1]
    pares = correlaciones[np.triu_indices(4, k=1)]

    # group es texto y se codifica con tres indicadores separados
    grupo = str(sample.meta.get("group", ""))
    metadatos = np.array(
        [
            float(sample.meta.get("sample_rate", 0.0)),
            float(sample.meta.get("level", 0.0)),
            float(grupo == "A"),
            float(grupo == "B"),
            float(grupo == "C"),
        ],
        dtype=np.float64,
    )

    vector = np.concatenate(
        [
            medias,
            desviaciones,
            cuantiles.ravel(),
            rugosidad,
            autocorrelacion,
            espectro.ravel(),
            faltantes.mean(axis=1),
            pares,
            metadatos,
        ]
    ).astype(np.float32)

    # El modelo recibe un vector finito incluso en una muestra extrema
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
