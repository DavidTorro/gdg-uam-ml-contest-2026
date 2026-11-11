"""Preparación opcional compilada para reducir el coste por muestra de ONNX."""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from competition_api import Sample


@njit
def _preparar(serie: np.ndarray, tasa: float, nivel: float, grupo: int,
             sin_atajos: bool, sin_escala: bool) -> tuple[np.ndarray, np.ndarray]:
    completa = np.empty((4, 256), np.float32)
    normalizada = np.empty((4, 256), np.float32)
    auxiliares = np.zeros(17, np.float32)
    for canal in range(4):
        presentes = np.empty(256, np.int64)
        total = 0
        for i in range(256):
            if math.isfinite(float(serie[canal, i])):
                presentes[total] = i
                total += 1
        auxiliares[8 + canal] = np.float32((256 - total) / 256.0)
        siguiente = 0
        for i in range(256):
            valor = serie[canal, i]
            if math.isfinite(float(valor)):
                completa[canal, i] = valor
                continue
            while siguiente < total and presentes[siguiente] < i:
                siguiente += 1
            if total == 0:
                completa[canal, i] = 0.0
            elif siguiente == 0:
                completa[canal, i] = serie[canal, presentes[0]]
            elif siguiente == total:
                completa[canal, i] = serie[canal, presentes[total - 1]]
            else:
                izquierda = presentes[siguiente - 1]
                derecha = presentes[siguiente]
                v0 = float(serie[canal, izquierda])
                v1 = float(serie[canal, derecha])
                completa[canal, i] = np.float32(
                    v0 + (v1 - v0) * ((i - izquierda) / (derecha - izquierda))
                )
        media = np.float32(0.0)
        for i in range(256):
            media = np.float32(media + completa[canal, i])
        media = np.float32(media / 256.0)
        cuadrados = np.float32(0.0)
        for i in range(256):
            centro = np.float32(completa[canal, i] - media)
            cuadrados = np.float32(cuadrados + np.float32(centro * centro))
        desviacion = np.float32(math.sqrt(float(np.float32(cuadrados / 256.0))))
        if not sin_escala:
            auxiliares[4 + canal] = np.float32(math.log1p(float(desviacion)))
        if not sin_atajos:
            auxiliares[canal] = np.float32(math.asinh(float(media)))
        divisor = np.float32(desviacion + np.float32(1e-6))
        for i in range(256):
            valor = np.float32((completa[canal, i] - media) / divisor)
            normalizada[canal, i] = min(6.0, max(-6.0, valor))
    if math.isfinite(tasa):
        auxiliares[12] = np.float32(math.copysign(math.log1p(abs(tasa)), tasa))
    if not sin_atajos and math.isfinite(nivel):
        auxiliares[13] = np.float32(math.copysign(math.log1p(abs(nivel)), nivel))
    if grupo >= 0:
        auxiliares[14 + grupo] = 1.0
    return normalizada, auxiliares


@njit
def _preparar_en_buffers(serie: np.ndarray, tasa: float, nivel: float, grupo: int,
                        sin_atajos: bool, sin_escala: bool, completa: np.ndarray,
                        normalizada: np.ndarray, auxiliares: np.ndarray) -> None:
    auxiliares[:] = np.float32(0.0)
    for canal in range(4):
        presentes = np.empty(256, np.int64)
        total = 0
        for i in range(256):
            if math.isfinite(float(serie[canal, i])):
                presentes[total] = i
                total += 1
        auxiliares[8 + canal] = np.float32((256 - total) / 256.0)
        siguiente = 0
        for i in range(256):
            valor = serie[canal, i]
            if math.isfinite(float(valor)):
                completa[canal, i] = valor
                continue
            while siguiente < total and presentes[siguiente] < i:
                siguiente += 1
            if total == 0:
                completa[canal, i] = 0.0
            elif siguiente == 0:
                completa[canal, i] = serie[canal, presentes[0]]
            elif siguiente == total:
                completa[canal, i] = serie[canal, presentes[total - 1]]
            else:
                izquierda = presentes[siguiente - 1]
                derecha = presentes[siguiente]
                v0 = float(serie[canal, izquierda])
                v1 = float(serie[canal, derecha])
                completa[canal, i] = np.float32(
                    v0 + (v1 - v0) * ((i - izquierda) / (derecha - izquierda))
                )
        media = np.float32(0.0)
        for i in range(256):
            media = np.float32(media + completa[canal, i])
        media = np.float32(media / 256.0)
        cuadrados = np.float32(0.0)
        for i in range(256):
            centro = np.float32(completa[canal, i] - media)
            cuadrados = np.float32(cuadrados + np.float32(centro * centro))
        desviacion = np.float32(math.sqrt(float(np.float32(cuadrados / 256.0))))
        if not sin_escala:
            auxiliares[4 + canal] = np.float32(math.log1p(float(desviacion)))
        if not sin_atajos:
            auxiliares[canal] = np.float32(math.asinh(float(media)))
        divisor = np.float32(desviacion + np.float32(1e-6))
        for i in range(256):
            valor = np.float32((completa[canal, i] - media) / divisor)
            normalizada[canal, i] = min(6.0, max(-6.0, valor))
    if math.isfinite(tasa):
        auxiliares[12] = np.float32(math.copysign(math.log1p(abs(tasa)), tasa))
    if not sin_atajos and math.isfinite(nivel):
        auxiliares[13] = np.float32(math.copysign(math.log1p(abs(nivel)), nivel))
    if grupo >= 0:
        auxiliares[14 + grupo] = 1.0


def preparar_jit(sample: Sample, sin_atajos: bool, sin_escala: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Tiene la misma entrada y salida que ``preparar_rapida``."""
    serie = np.asarray(sample.series, dtype=np.float32)
    if serie.shape != (4, 256):
        raise ValueError(f"Se esperaba una señal (4, 256), recibida {serie.shape}")
    grupo = str(sample.meta.get("group", ""))
    indice_grupo = {"A": 0, "B": 1, "C": 2}.get(grupo, -1)
    return _preparar(
        serie,
        float(sample.meta.get("sample_rate", 0.0)),
        float(sample.meta.get("level", 0.0)),
        indice_grupo, sin_atajos, sin_escala,
    )


def preparar_jit_en_buffers(sample: Sample, sin_atajos: bool, sin_escala: bool,
                            completa: np.ndarray, normalizada: np.ndarray,
                            auxiliares: np.ndarray) -> None:
    """Escribe la misma preparación directamente en buffers reutilizados."""
    serie = np.asarray(sample.series, dtype=np.float32)
    if serie.shape != (4, 256):
        raise ValueError(f"Se esperaba una señal (4, 256), recibida {serie.shape}")
    grupo = str(sample.meta.get("group", ""))
    indice_grupo = {"A": 0, "B": 1, "C": 2}.get(grupo, -1)
    _preparar_en_buffers(
        serie,
        float(sample.meta.get("sample_rate", 0.0)),
        float(sample.meta.get("level", 0.0)),
        indice_grupo, sin_atajos, sin_escala,
        completa, normalizada, auxiliares,
    )


def calentar_jit() -> None:
    """Compila en setup, que el evaluador no cronometra."""
    _preparar(np.zeros((4, 256), dtype=np.float32), 0.0, 0.0, -1, True, False)


def calentar_jit_en_buffers() -> None:
    """Compila la variante con buffers durante setup."""
    _preparar_en_buffers(
        np.zeros((4, 256), dtype=np.float32), 0.0, 0.0, -1, True, False,
        np.empty((4, 256), dtype=np.float32),
        np.empty((4, 256), dtype=np.float32),
        np.empty(17, dtype=np.float32),
    )
