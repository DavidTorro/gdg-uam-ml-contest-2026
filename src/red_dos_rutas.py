"""CNN de dos rutas: relaciones entre canales y forma propia de cada canal.

La ruta conjunta conserva la mezcla temprana que funciona en exp033. La ruta
por canal mantiene separadas las cuatro señales hasta resumirlas. La cabeza
recibe ambas, sin imponer que una sustituya a la otra.
"""

from __future__ import annotations

import torch
from torch import nn


class RedDosRutas(nn.Module):
    def __init__(self, ancho: int = 1) -> None:
        super().__init__()
        a = ancho
        self.conjunta = nn.Sequential(
            nn.Conv1d(4, 16 * a, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16 * a, 32 * a, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32 * a, 48 * a, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(48 * a, 64 * a, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.por_canal = nn.Sequential(
            nn.Conv1d(4, 32 * a, 9, stride=2, padding=4, groups=4), nn.ReLU(),
            nn.Conv1d(32 * a, 64 * a, 7, stride=2, padding=3, groups=4), nn.ReLU(),
            nn.Conv1d(64 * a, 64 * a, 5, stride=2, padding=2, groups=4), nn.ReLU(),
            nn.Conv1d(64 * a, 64 * a, 3, stride=2, padding=1, groups=4), nn.ReLU(),
        )
        self.media = nn.AdaptiveAvgPool1d(4)
        self.maximo = nn.AdaptiveMaxPool1d(4)
        self.salida = nn.Sequential(
            nn.Linear(64 * a * 16 + 17, 96 * a), nn.ReLU(),
            nn.Linear(96 * a, 8),
        )

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor,
                devolver_oculto: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        conjunta = self.conjunta(serie)
        por_canal = self.por_canal(serie)
        resumen = torch.cat((
            self.media(conjunta).flatten(1), self.maximo(conjunta).flatten(1),
            self.media(por_canal).flatten(1), self.maximo(por_canal).flatten(1),
            auxiliares,
        ), dim=1)
        oculto = self.salida[:2](resumen)
        salida = self.salida[2](oculto)
        return (salida, oculto) if devolver_oculto else salida
