"""Crea una subred 3/4 de exp033 sin mirar validación ni demo."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.red_compacta import RedCompacta

SOURCE = ROOT / "models/exp033.pt"
BIAS = ROOT / "models/exp033.sesgo.json"
DEST = ROOT / "models/c01_pruned_init.pt"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def top(score: torch.Tensor, count: int) -> torch.Tensor:
    # Orden original tras elegir: las capas siguientes conservan sus posiciones.
    return torch.argsort(score, descending=True, stable=True)[:count].sort().values


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Ya existe {DEST}; no se sobrescribe")
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if (source.get("formato") != "gdg-cnn-v1" or source.get("red", "base") != "base"
            or source.get("ancho", 1) != 1 or not source.get("sin_atajos")
            or source.get("sin_escala") or source.get("sin_grupo")):
        raise ValueError("exp033 no tiene la arquitectura y preparación esperadas")
    bias = json.loads(BIAS.read_text(encoding="utf-8"))
    if (bias.get("formato") != "gdg-sesgo-v1" or bias.get("pesos") != "models/exp033.pt"
            or len(bias.get("sesgo", [])) != 8):
        raise ValueError("Sesgo histórico inesperado")
    old = source["state_dict"]
    conv = [old[f"convoluciones.{i}.weight"] for i in (0, 2, 4, 6)]
    head = old["salida.0.weight"]
    output = old["salida.2.weight"]
    expected_shapes = ((16, 4, 9), (32, 16, 7), (48, 32, 5), (64, 48, 3))
    if tuple(tuple(w.shape) for w in conv) != expected_shapes or tuple(head.shape) != (64, 529):
        raise ValueError("Tensores de exp033 con formas inesperadas")

    keep = [
        top(conv[0].abs().sum((1, 2)) * conv[1].abs().sum((0, 2)), 12),
        top(conv[1].abs().sum((1, 2)) * conv[2].abs().sum((0, 2)), 24),
        top(conv[2].abs().sum((1, 2)) * conv[3].abs().sum((0, 2)), 36),
        top(conv[3].abs().sum((1, 2)) * head[:, :512].abs().reshape(64, 2, 64, 4).sum((0, 1, 3)), 48),
        top(head.abs().sum(1) * output.abs().sum(0), 48),
    ]
    compact = RedCompacta()
    new = compact.state_dict()
    previous = torch.arange(4)
    for layer, selected in zip((0, 2, 4, 6), keep[:4]):
        new[f"convoluciones.{layer}.weight"] = old[f"convoluciones.{layer}.weight"].index_select(0, selected).index_select(1, previous).clone()
        new[f"convoluciones.{layer}.bias"] = old[f"convoluciones.{layer}.bias"].index_select(0, selected).clone()
        previous = selected
    pooled = torch.tensor(
        [offset + 4 * int(channel) + position
         for offset in (0, 256) for channel in keep[3] for position in range(4)]
        + list(range(512, 529)), dtype=torch.long,
    )
    new["salida.0.weight"] = head.index_select(0, keep[4]).index_select(1, pooled).clone()
    new["salida.0.bias"] = old["salida.0.bias"].index_select(0, keep[4]).clone()
    new["salida.2.weight"] = output.index_select(1, keep[4]).clone()
    new["salida.2.bias"] = old["salida.2.bias"].clone()
    new["sesgo_fijo"] = torch.tensor(bias["sesgo"], dtype=torch.float32)
    compact.load_state_dict(new, strict=True)
    torch.save({
        "formato": "gdg-cnn-v1", "state_dict": new,
        "red": "compacta", "ancho": 1, "sin_atajos": True,
        "sin_escala": False, "sin_grupo": False, "fusion_tardia": False,
        "origen": "models/exp033.pt", "sha256_origen": sha256(SOURCE),
        "sha256_sesgo": sha256(BIAS),
        "canales_conservados": [item.tolist() for item in keep],
    }, DEST)
    print(f"Creado {DEST.relative_to(ROOT)} ({DEST.stat().st_size} bytes); "
          f"{sum(p.numel() for p in compact.parameters())} parámetros", flush=True)


if __name__ == "__main__":
    main()
