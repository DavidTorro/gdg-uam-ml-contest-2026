"""CNN base con menos canales y el sesgo histórico fijo en la salida."""

from __future__ import annotations

import torch
from torch import nn

from src.red_convolucional import RedConvolucional


class RedCompacta(RedConvolucional):
    """Misma secuencia de operaciones que exp033, con anchuras 12/24/36/48."""

    def __init__(self) -> None:
        super().__init__()
        self.convoluciones = nn.Sequential(
            nn.Conv1d(4, 12, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(12, 24, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(24, 36, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(36, 48, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.salida = nn.Sequential(
            nn.Linear(48 * 8 + 17, 48), nn.ReLU(),
            nn.Linear(48, 8),
        )
        self.register_buffer("sesgo_fijo", torch.zeros(8, dtype=torch.float32))

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor,
                devolver_oculto: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        resultado = super().forward(serie, auxiliares, devolver_oculto=devolver_oculto)
        if devolver_oculto:
            logits, oculto = resultado
            return logits + self.sesgo_fijo, oculto
        return resultado + self.sesgo_fijo
