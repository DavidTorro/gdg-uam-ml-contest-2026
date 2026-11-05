"""Inferencia de la CNN con ONNX Runtime en un solo hilo de CPU."""

from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort

from competition_api import Sample
from src.red_numpy import preparar_rapida


class RedOnnx:
    def __init__(self, ruta: Path) -> None:
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = 1
        opciones.inter_op_num_threads = 1
        opciones.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesion = ort.InferenceSession(str(ruta), sess_options=opciones,
                                          providers=["CPUExecutionProvider"])
        meta = self.sesion.get_modelmeta().custom_metadata_map
        if meta.get("formato") != "gdg-onnx-v1":
            raise ValueError(f"Formato de pesos ONNX desconocido: {ruta}")
        self.sin_atajos = meta.get("sin_atajos") == "true"
        self.sin_escala = meta.get("sin_escala") == "true"
        self.preparar = preparar_rapida
        if os.environ.get("GDG_PREP_JIT", "1") == "1":
            from src.preparacion_numba import calentar_jit, preparar_jit

            calentar_jit()
            self.preparar = preparar_jit
        if [entrada.name for entrada in self.sesion.get_inputs()] != ["serie", "auxiliares"]:
            raise ValueError(f"Entradas ONNX inesperadas: {ruta}")

    def predecir(self, sample: Sample) -> int:
        serie, auxiliares = self.preparar(sample, self.sin_atajos, self.sin_escala)
        logits = self.sesion.run(None, {"serie": serie[None], "auxiliares": auxiliares[None]})[0]
        return int(logits.argmax())
