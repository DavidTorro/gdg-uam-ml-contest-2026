"""Candidata: consulta exp034 solo en decisiones 6/7 poco seguras."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from competition_api import Sample
from src.red_onnx import RedOnnx

ROOT = Path(__file__).resolve().parent


class Model:
    def __init__(self) -> None:
        config = json.loads((ROOT / "models/selectivo_01.json").read_text(encoding="utf-8"))
        if config.get("protocolo") != "selectivo_01":
            raise ValueError("Configuración selectiva inesperada")
        self.umbral = float(config["umbral_margen_6_7"])
        self.base = RedOnnx(ROOT / "models/exp033.sesgo.onnx")
        self.segunda = RedOnnx(ROOT / "models/exp034.sesgo.onnx")
        if self.base.io_binding is None or self.segunda.io_binding is None:
            raise RuntimeError("La candidata selectiva requiere I/O Binding")

    def predict(self, sample: Sample) -> int:
        clase = self.base.predecir(sample)
        if clase not in (6, 7):
            return clase
        logits_base = self.base.logits_buffer[0]
        if abs(float(logits_base[6] - logits_base[7])) >= self.umbral:
            return clase
        np.copyto(self.segunda.serie_buffer, self.base.serie_buffer)
        np.copyto(self.segunda.aux_buffer, self.base.aux_buffer)
        self.segunda.sesion.run_with_iobinding(self.segunda.io_binding)
        logits_segunda = self.segunda.logits_buffer[0]
        prob_base = np.exp(logits_base - logits_base.max())
        prob_segunda = np.exp(logits_segunda - logits_segunda.max())
        combinada = prob_base / prob_base.sum() + prob_segunda / prob_segunda.sum()
        return int(combinada.argmax())
