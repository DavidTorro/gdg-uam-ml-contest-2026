"""Exporta una CNN PyTorch a ONNX; solo requiere onnx durante la exportación."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.red_convolucional import RedConvolucional
from src.red_multiescala import RedMultiescala


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pesos", required=True, help="Ruta relativa a un peso CNN .pt")
    parser.add_argument("--sesgo", help="Ruta relativa al sesgo .json opcional")
    parser.add_argument("--destino", help="Ruta relativa .onnx; por defecto toma el nombre del peso")
    args = parser.parse_args()
    origen = ROOT / args.pesos
    if origen.suffix != ".pt" or not origen.is_file():
        parser.error("--pesos debe ser un archivo .pt existente")
    artefacto = torch.load(origen, map_location="cpu", weights_only=True)
    if artefacto.get("formato") != "gdg-cnn-v1":
        parser.error("Formato de peso CNN desconocido")
    tipo_red = artefacto.get("red", "base")
    if tipo_red == "multiescala":
        red = RedMultiescala(ancho=int(artefacto.get("ancho", 1))).eval()
    elif tipo_red == "base":
        red = RedConvolucional(
            ancho=int(artefacto.get("ancho", 1)),
            fusion_tardia=bool(artefacto.get("fusion_tardia", False)),
        ).eval()
    else:
        parser.error(f"Tipo de red desconocido: {tipo_red}")
    red.load_state_dict(artefacto["state_dict"])
    if args.sesgo:
        sesgo = json.loads((ROOT / args.sesgo).read_text(encoding="utf-8"))
        if sesgo.get("formato") != "gdg-sesgo-v1" or len(sesgo.get("sesgo", [])) != 8:
            parser.error("Sesgo inválido")
        with torch.no_grad():
            red.salida[2].bias += torch.tensor(sesgo["sesgo"], dtype=torch.float32)
    destino = ROOT / (args.destino or args.pesos.replace(
        ".pt", ".sesgo.onnx" if args.sesgo else ".onnx"
    ))
    if destino.exists():
        parser.error(f"Ya existe {destino}; no se sobrescribe")
    torch.onnx.export(
        red, (torch.zeros(1, 4, 256), torch.zeros(1, 17)), str(destino),
        input_names=["serie", "auxiliares"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    modelo = onnx.load(destino)
    onnx.helper.set_model_props(modelo, {
        "formato": "gdg-onnx-v1", "origen": args.pesos,
        "red": tipo_red,
        "sin_atajos": str(bool(artefacto.get("sin_atajos", False))).lower(),
        "sin_escala": str(bool(artefacto.get("sin_escala", False))).lower(),
    })
    onnx.checker.check_model(modelo)
    onnx.save(modelo, destino)
    print(f"Guardado {destino.relative_to(ROOT)} ({destino.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
