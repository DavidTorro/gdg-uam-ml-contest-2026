"""Red pequeña para clasificar las cuatro series completas de una muestra."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from competition_api import Sample


def preparar_muestra(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
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
    return normalizada, auxiliares


class RedConvolucional(nn.Module):
    """CNN 1D de pocos parámetros; recibe señal (N, 4, 256) y 17 auxiliares."""

    def __init__(self) -> None:
        super().__init__()
        self.convoluciones = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.media = nn.AdaptiveAvgPool1d(4)
        self.maximo = nn.AdaptiveMaxPool1d(4)
        self.salida = nn.Sequential(
            nn.Linear(64 * 8 + 17, 64), nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor) -> torch.Tensor:
        oculto = self.convoluciones(serie)
        resumen = torch.cat((self.media(oculto).flatten(1), self.maximo(oculto).flatten(1), auxiliares), dim=1)
        return self.salida(resumen)
