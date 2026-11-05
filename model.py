"""Punto de entrada que cargará el evaluador oficial"""

from __future__ import annotations

import json
import pickle
import os
from pathlib import Path

from competition_api import Sample
from src.caracteristicas import extraer

ROOT = Path(__file__).resolve().parent


class Model:
    def __init__(self) -> None:
        # La carga ocurre en setup y debe terminar antes de 120 segundos
        # El flujo local elige los pesos del experimento mediante una ruta relativa
        # Por defecto, la entrega: CNN exp033 en ONNX con el empujón de la clase 6 incluido
        ruta = os.environ.get("GDG_MODEL_PATH", "models/exp033.sesgo.onnx")
        pesos = ROOT / ruta
        if Path(ruta).is_absolute() or not pesos.is_file():
            raise FileNotFoundError(f"No se encuentran los pesos del modelo: {ruta}")
        self.red = None
        self.red_numpy = None
        self.red_onnx = None
        if pesos.suffix == ".npz":
            # CNN exportada con scripts/exportar_numpy.py: mismas clases, más rápida y sin PyTorch
            # El empujón de clase, si lo hay, va dentro del propio .npz
            from src.red_numpy import RedNumpy

            self.red_numpy = RedNumpy(pesos)
            return
        if pesos.suffix == ".onnx":
            from src.red_onnx import RedOnnx

            self.red_onnx = RedOnnx(pesos)
            return
        if pesos.suffix == ".pt":
            import torch
            from src.red_convolucional import RedConvolucional, preparar_muestra
            from src.red_multiescala import RedMultiescala

            torch.set_num_threads(1)
            artefacto = torch.load(pesos, map_location="cpu", weights_only=True)
            if artefacto.get("formato") != "gdg-cnn-v1":
                raise ValueError(f"Formato de pesos CNN desconocido: {ruta}")
            # Los pesos desde exp022 guardan su ancho y si se entrenaron sin las auxiliares tramposas
            tipo_red = artefacto.get("red", "base")
            if tipo_red == "multiescala":
                self.red = RedMultiescala(ancho=int(artefacto.get("ancho", 1)))
            elif tipo_red == "base":
                self.red = RedConvolucional(ancho=int(artefacto.get("ancho", 1)),
                                            fusion_tardia=bool(artefacto.get("fusion_tardia", False)))
            else:
                raise ValueError(f"Red CNN desconocida: {tipo_red}")
            self.red.load_state_dict(artefacto["state_dict"])
            self.red.eval()
            self.torch = torch
            self.preparar_muestra = preparar_muestra
            self.sin_atajos = bool(artefacto.get("sin_atajos", False))
            self.sin_escala = bool(artefacto.get("sin_escala", False))

            # Sesgo opcional que se suma a la salida de la red antes de elegir clase
            # Lo calcula scripts/calibrar_sesgo.py; sin GDG_SESGO_PATH la red no cambia
            self.sesgo = None
            ruta_sesgo = os.environ.get("GDG_SESGO_PATH")
            if ruta_sesgo:
                fichero = ROOT / ruta_sesgo
                if Path(ruta_sesgo).is_absolute() or not fichero.is_file():
                    raise FileNotFoundError(f"No se encuentra el sesgo: {ruta_sesgo}")
                datos = json.loads(fichero.read_text(encoding="utf-8"))
                if datos.get("formato") != "gdg-sesgo-v1" or len(datos["sesgo"]) != 8:
                    raise ValueError(f"Formato de sesgo desconocido: {ruta_sesgo}")
                self.sesgo = torch.tensor(datos["sesgo"], dtype=torch.float32)
            return
        with pesos.open("rb") as file:
            artefacto = pickle.load(file)
        if isinstance(artefacto, dict) and artefacto.get("formato") == "gdg-modelo-v2":
            self.clasificador = artefacto["clasificador"]
            self.temporales = bool(artefacto["temporales"])
            self.extendidas = bool(artefacto.get("extendidas", False))
        else:
            self.clasificador = artefacto
            self.temporales = False
            self.extendidas = False

    def predict(self, x: Sample) -> int:
        # El evaluador cronometra esta llamada para una muestra cada vez
        if self.red_numpy is not None:
            return self.red_numpy.predecir(x)
        if self.red_onnx is not None:
            return self.red_onnx.predecir(x)
        if self.red is not None:
            serie, auxiliares = self.preparar_muestra(
                x, sin_atajos=self.sin_atajos, sin_escala=self.sin_escala
            )
            with self.torch.inference_mode():
                salida = self.red(
                    self.torch.from_numpy(serie[None, ...]),
                    self.torch.from_numpy(auxiliares[None, ...]),
                )
                if self.sesgo is not None:
                    salida = salida + self.sesgo
            return int(salida.argmax(dim=1).item())
        vector = extraer(x, temporales=self.temporales, extendidas=self.extendidas).reshape(1, -1)
        prediccion = self.clasificador.predict(vector)[0]

        # Convertimos el escalar de NumPy a int de Python como exige el contrato
        return int(prediccion)
