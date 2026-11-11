"""Dos expertos de señal con mezcla condicionada por calidad observada.

La puerta no recibe etiquetas ni logits. Ve la fracción de huecos, rugosidad
normalizada y tasa; las dos rutas conservan la información de la señal. Las
cabezas se supervisan también por separado durante entrenamiento para evitar
que una de ellas quede sin aprender cuando la puerta favorece a la otra.
"""

from __future__ import annotations

import torch
from torch import nn


class RedDosRutasGate(nn.Module):
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
        self.cabeza_conjunta = nn.Sequential(nn.Linear(64 * a * 8 + 17, 64 * a), nn.ReLU(), nn.Linear(64 * a, 8))
        self.cabeza_canal = nn.Sequential(nn.Linear(64 * a * 8 + 17, 64 * a), nn.ReLU(), nn.Linear(64 * a, 8))
        # 4 tasas de huecos, 4 rugosidades y sample_rate. La última capa
        # comienza en cero: la mezcla inicial es exactamente 50/50.
        self.puerta = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.puerta[-1].weight)
        nn.init.zeros_(self.puerta[-1].bias)

    def forward(self, serie: torch.Tensor, auxiliares: torch.Tensor,
                devolver_componentes: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conjunta = self.conjunta(serie)
        canal = self.por_canal(serie)
        z_conjunta = torch.cat((self.media(conjunta).flatten(1), self.maximo(conjunta).flatten(1), auxiliares), dim=1)
        z_canal = torch.cat((self.media(canal).flatten(1), self.maximo(canal).flatten(1), auxiliares), dim=1)
        logits_conjunta = self.cabeza_conjunta(z_conjunta)
        logits_canal = self.cabeza_canal(z_canal)
        rugosidad = (serie[:, :, 1:] - serie[:, :, :-1]).square().mean(dim=2).clamp(0, 8)
        calidad = torch.cat((auxiliares[:, 8:12], rugosidad, auxiliares[:, 12:13]), dim=1)
        peso_conjunta = torch.sigmoid(self.puerta(calidad))
        logits = peso_conjunta * logits_conjunta + (1 - peso_conjunta) * logits_canal
        if devolver_componentes:
            return logits, logits_conjunta, logits_canal, peso_conjunta
        return logits
