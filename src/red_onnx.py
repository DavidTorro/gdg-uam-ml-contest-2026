"""Inferencia de la CNN con ONNX Runtime en un solo hilo de CPU."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
        self.io_binding = None
        if os.environ.get("GDG_ORT_IOBINDING", "1") == "1":
            # Buffers de tamaño fijo: la evaluación llama a predict de forma secuencial.
            self.serie_buffer = np.empty((1, 4, 256), dtype=np.float32)
            self.aux_buffer = np.empty((1, 17), dtype=np.float32)
            self.logits_buffer = np.empty((1, 8), dtype=np.float32)
            self.io_binding = self.sesion.io_binding()
            self.io_binding.bind_cpu_input("serie", self.serie_buffer)
            self.io_binding.bind_cpu_input("auxiliares", self.aux_buffer)
            self.io_binding.bind_output(
                "logits", "cpu", 0, np.float32,
                self.logits_buffer.shape, self.logits_buffer.ctypes.data,
            )
        self.preparar_en_buffers = None
        jit_activo = os.environ.get("GDG_PREP_JIT", "1") == "1"
        buffers_por_defecto = "1" if self.io_binding is not None and jit_activo else "0"
        if os.environ.get("GDG_JIT_BUFFERS", buffers_por_defecto) == "1":
            if self.io_binding is None or not jit_activo:
                raise ValueError("GDG_JIT_BUFFERS requiere I/O Binding y preparación JIT")
            from src.preparacion_numba import calentar_jit_en_buffers, preparar_jit_en_buffers

            self.completa_buffer = np.empty((4, 256), dtype=np.float32)
            calentar_jit_en_buffers()
            self.preparar_en_buffers = preparar_jit_en_buffers

    def predecir(self, sample: Sample) -> int:
        if self.preparar_en_buffers is not None:
            self.preparar_en_buffers(
                sample, self.sin_atajos, self.sin_escala,
                self.completa_buffer, self.serie_buffer[0], self.aux_buffer[0],
            )
            self.sesion.run_with_iobinding(self.io_binding)
            return int(self.logits_buffer.argmax())
        serie, auxiliares = self.preparar(sample, self.sin_atajos, self.sin_escala)
        if self.io_binding is not None:
            np.copyto(self.serie_buffer[0], serie)
            np.copyto(self.aux_buffer[0], auxiliares)
            self.sesion.run_with_iobinding(self.io_binding)
            return int(self.logits_buffer.argmax())
        logits = self.sesion.run(None, {"serie": serie[None], "auxiliares": auxiliares[None]})[0]
        return int(logits.argmax())
