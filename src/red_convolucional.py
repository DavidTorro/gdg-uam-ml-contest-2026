"""Red pequeña para clasificar las cuatro series completas de una muestra."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from competition_api import Sample

# Posiciones de las auxiliares que en train delatan la clase y en el test no: medias y level
ATAJOS = [0, 1, 2, 3, 13]


def preparar_muestra(sample: Sample, sin_atajos: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Interpola huecos y devuelve la señal normalizada y sus datos auxiliares."""
    serie = np.asarray(sample.series, dtype=np.float32)
    if serie.shape != (4, 256):
        raise ValueError(f"Se esperaba una señal (4, 256), recibida {serie.shape}")
    tiempos = np.arange(256)
    completa = np.zeros((4, 256), dtype=np.float32)
    ausentes = ~np.isfinite(serie)
    for canal in range(4):
        presentes = ~ausentes[canal]
        if presentes.any():
            completa[canal] = np.interp(tiempos, tiempos[presentes], serie[canal, presentes])

    medias = completa.mean(axis=1)
    desviaciones = completa.std(axis=1)
    normalizada = (completa - medias[:, None]) / (desviaciones[:, None] + 1e-6)
    normalizada = np.clip(normalizada, -6.0, 6.0).astype(np.float32)

    grupo = str(sample.meta.get("group", ""))
    tasa = float(sample.meta.get("sample_rate", 0.0))
    nivel = float(sample.meta.get("level", 0.0))
    auxiliares = np.concatenate((
        np.arcsinh(medias),
        np.log1p(desviaciones),
        ausentes.mean(axis=1),
        [np.sign(tasa) * np.log1p(abs(tasa)), np.sign(nivel) * np.log1p(abs(nivel))],
        [float(grupo == letra) for letra in "ABC"],
    )).astype(np.float32)
    auxiliares = np.nan_to_num(auxiliares, nan=0.0, posinf=0.0, neginf=0.0)
    if sin_atajos:
        auxiliares[ATAJOS] = 0.0
    return normalizada, auxiliares


def preparar_lote(series: np.ndarray, tasa: np.ndarray, nivel: np.ndarray, grupo: np.ndarray,
                  sin_atajos: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Igual que preparar_muestra, pero para (N, 4, 256) de una vez; solo se usa al entrenar."""
    serie = np.asarray(series, dtype=np.float32)
    n, canales, largo = serie.shape
    tiempos = np.arange(largo)
    presentes = np.isfinite(serie)
    ausentes = ~presentes

    # Interpolación lineal entre el valor presente anterior y el siguiente, como np.interp
    anterior = np.maximum.accumulate(np.where(presentes, tiempos, -1), axis=2)
    siguiente = np.minimum.accumulate(np.where(presentes, tiempos, largo)[..., ::-1], axis=2)[..., ::-1]
    sin_anterior = anterior < 0
    sin_siguiente = siguiente >= largo
    anterior = np.where(sin_anterior, siguiente, anterior)
    siguiente = np.where(sin_siguiente, anterior, siguiente)
    vacio = sin_anterior & sin_siguiente
    seguros = np.where(presentes, serie, 0.0)
    izquierda = np.take_along_axis(seguros, np.clip(anterior, 0, largo - 1), axis=2)
    derecha = np.take_along_axis(seguros, np.clip(siguiente, 0, largo - 1), axis=2)
    paso = (siguiente - anterior).astype(np.float32)
    peso = np.divide(tiempos - anterior, paso, out=np.zeros_like(paso), where=paso > 0)
    completa = np.where(presentes, serie, izquierda + peso * (derecha - izquierda))
    completa = np.where(vacio, 0.0, completa).astype(np.float32)

    medias = completa.mean(axis=2)
    desviaciones = completa.std(axis=2)
    normalizada = (completa - medias[..., None]) / (desviaciones[..., None] + 1e-6)
    normalizada = np.clip(normalizada, -6.0, 6.0).astype(np.float32)

    tasa = np.asarray(tasa, dtype=np.float64)
    nivel = np.asarray(nivel, dtype=np.float64)
    grupo = np.asarray(grupo).astype(str)
    auxiliares = np.concatenate((
        np.arcsinh(medias),
        np.log1p(desviaciones),
        ausentes.mean(axis=2),
        (np.sign(tasa) * np.log1p(np.abs(tasa)))[:, None],
        (np.sign(nivel) * np.log1p(np.abs(nivel)))[:, None],
        np.stack([(grupo == letra).astype(np.float64) for letra in "ABC"], axis=1),
    ), axis=1).astype(np.float32)
    auxiliares = np.nan_to_num(auxiliares, nan=0.0, posinf=0.0, neginf=0.0)
    if sin_atajos:
        auxiliares[:, ATAJOS] = 0.0
    return normalizada, auxiliares


class RedConvolucional(nn.Module):
    """CNN 1D de pocos parámetros; recibe señal (N, 4, 256) y 17 auxiliares.

    ancho multiplica el número de filtros; con 1 es la red de exp015 a exp020.
    """

    def __init__(self, ancho: int = 1) -> None:
        super().__init__()
        a = ancho
        self.convoluciones = nn.Sequential(
            nn.Conv1d(4, 16 * a, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16 * a, 32 * a, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32 * a, 48 * a, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(48 * a, 64 * a, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.media = nn.AdaptiveAvgPool1d(4)
        self.maximo = nn.AdaptiveMaxPool1d(4)
        self.salida = nn.Sequential(
            nn.Linear(64 * a * 8 + 17, 64 * a), nn.ReLU(),
            nn.Linear(64 * a, 8),
        )

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor) -> torch.Tensor:
        oculto = self.convoluciones(serie)
        resumen = torch.cat((self.media(oculto).flatten(1), self.maximo(oculto).flatten(1), auxiliares), dim=1)
        return self.salida(resumen)
