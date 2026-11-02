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
        with pesos.open("rb") as file:
            self.clasificador = pickle.load(file)

    def predict(self, x: Sample) -> int:
        # El evaluador cronometra esta llamada para una muestra cada vez
        vector = extraer(x).reshape(1, -1)
        prediccion = self.clasificador.predict(vector)[0]

        # Convertimos el escalar de NumPy a int de Python como exige el contrato
        return int(prediccion)
