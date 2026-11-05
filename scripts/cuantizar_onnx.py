"""Cuantiza exp033 a INT8 con 512 muestras simuladas equilibradas, sin demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.red_convolucional import preparar_lote


class Lector(CalibrationDataReader):
    def __init__(self, serie: np.ndarray, auxiliares: np.ndarray) -> None:
        self.serie = serie
        self.auxiliares = auxiliares
        self.indice = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.indice == len(self.serie):
            return None
        i = self.indice
        self.indice += 1
        return {"serie": self.serie[i:i + 1], "auxiliares": self.auxiliares[i:i + 1]}

    def rewind(self) -> None:
        self.indice = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", default="models/exp033.sesgo.int8_qdq.onnx")
    args = parser.parse_args()
    origen = ROOT / "models/exp033.sesgo.onnx"
    fuente = ROOT / "data/val_examen_v2.npz"
    destino = ROOT / args.destino
    if Path(args.destino).is_absolute() or destino.suffix != ".onnx":
        parser.error("--destino debe ser un archivo .onnx relativo al repositorio")
    if destino.exists():
        parser.error(f"Ya existe {destino}; no se sobrescribe")
    for ruta in (origen, fuente):
        if not ruta.is_file():
            parser.error(f"Falta {ruta}")

    with np.load(fuente) as datos:
        rng = np.random.default_rng(20261105)
        indices = np.concatenate([
            rng.choice(np.flatnonzero(datos["labels"] == clase), 64, replace=False)
            for clase in range(8)
        ])
        serie, auxiliares = preparar_lote(
            datos["series"][indices], datos["meta__sample_rate"][indices],
            datos["meta__level"][indices], datos["meta__group"][indices],
            sin_atajos=True,
        )
    quantize_static(
        str(origen), str(destino), Lector(serie, auxiliares),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm"], per_channel=True,
    )
    print(f"Guardado {destino.relative_to(ROOT)} ({destino.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
