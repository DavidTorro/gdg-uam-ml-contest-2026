"""Punto de entrada que cargará el evaluador oficial"""

from __future__ import annotations

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
        # exp001 es la base provisional; la entrega final usará el modelo elegido
        ruta = os.environ.get("GDG_MODEL_PATH", "models/exp001.pkl")
        pesos = ROOT / ruta
        if Path(ruta).is_absolute() or not pesos.is_file():
            raise FileNotFoundError(f"No se encuentran los pesos del modelo: {ruta}")
        self.red = None
        if pesos.suffix == ".pt":
            import torch
            from src.red_convolucional import RedConvolucional, preparar_muestra

            torch.set_num_threads(1)
            artefacto = torch.load(pesos, map_location="cpu", weights_only=True)
            if artefacto.get("formato") != "gdg-cnn-v1":
                raise ValueError(f"Formato de pesos CNN desconocido: {ruta}")
            self.red = RedConvolucional()
            self.red.load_state_dict(artefacto["state_dict"])
            self.red.eval()
            self.torch = torch
            self.preparar_muestra = preparar_muestra
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
        if self.red is not None:
            serie, auxiliares = self.preparar_muestra(x)
            with self.torch.inference_mode():
                salida = self.red(
                    self.torch.from_numpy(serie[None, ...]),
                    self.torch.from_numpy(auxiliares[None, ...]),
                )
            return int(salida.argmax(dim=1).item())
        vector = extraer(x, temporales=self.temporales, extendidas=self.extendidas).reshape(1, -1)
        prediccion = self.clasificador.predict(vector)[0]

        # Convertimos el escalar de NumPy a int de Python como exige el contrato
        return int(prediccion)
