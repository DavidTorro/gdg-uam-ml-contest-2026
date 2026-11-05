"""CNN temporal con dos escalas iniciales y contexto largo residual.

Mantiene la preparación y el vector auxiliar de la CNN base. La rama larga
observa 33 puntos desde el inicio y el bloque dilatado combina regiones
separadas antes del resumen final, conservando mezcla entre canales.
"""

from __future__ import annotations

import torch
from torch import nn


class RedMultiescala(nn.Module):
    def __init__(self, ancho: int = 1) -> None:
        super().__init__()
        if ancho < 1:
            raise ValueError("ancho debe ser positivo")
        a = ancho
        self.corta = nn.Sequential(nn.Conv1d(4, 8 * a, 9, stride=2, padding=4), nn.ReLU())
        self.larga = nn.Sequential(nn.Conv1d(4, 8 * a, 33, stride=2, padding=16), nn.ReLU())
        self.convoluciones = nn.Sequential(
            nn.Conv1d(16 * a, 32 * a, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32 * a, 48 * a, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(48 * a, 64 * a, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64 * a, 64 * a, 5, padding=4, dilation=2, groups=64 * a), nn.ReLU(),
            nn.Conv1d(64 * a, 64 * a, 1),
        )
        self.activacion = nn.ReLU()
        self.media = nn.AdaptiveAvgPool1d(4)
        self.maximo = nn.AdaptiveMaxPool1d(4)
        self.salida = nn.Sequential(
            nn.Linear(64 * a * 8 + 17, 64 * a), nn.ReLU(),
            nn.Linear(64 * a, 8),
        )

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor,
                devolver_oculto: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ramas = torch.cat((self.corta(serie), self.larga(serie)), dim=1)
        oculto = self.convoluciones(ramas)
        oculto = self.activacion(oculto + self.temporal(oculto))
        resumen = torch.cat((self.media(oculto).flatten(1), self.maximo(oculto).flatten(1), auxiliares), dim=1)
        if devolver_oculto:
            representacion = self.salida[:2](resumen)
            return self.salida[2](representacion), representacion
        return self.salida(resumen)
