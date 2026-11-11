"""Crea una media fija de exp033 y sus tres afinamientos compatibles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("exp033", "d01_control", "d02_control_27", "d02_control_28")
COEFFICIENTS = (0.5, 1 / 6, 1 / 6, 1 / 6)
WEIGHTS = ROOT / "models/s01_sopa.pt"
BIAS = ROOT / "models/s01_sopa.sesgo.json"
BASE_BIAS = ROOT / "models/exp033.sesgo.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in (WEIGHTS, BIAS):
        if path.exists():
            raise FileExistsError(f"No se sobrescribe {path}")
    paths = [ROOT / "models" / f"{name}.pt" for name in SOURCES]
    for path in (*paths, BASE_BIAS):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifacts = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    for path, artifact in zip(paths, artifacts):
        if (artifact.get("formato") != "gdg-cnn-v1"
                or artifact.get("red", "base") != "base"
                or int(artifact.get("ancho", 1)) != 1
                or bool(artifact.get("fusion_tardia", False))
                or not bool(artifact.get("sin_atajos", False))
                or bool(artifact.get("sin_escala", False))
                or bool(artifact.get("sin_grupo", False))):
            raise ValueError(f"Arquitectura o preparación incompatible: {path}")
    states = [artifact["state_dict"] for artifact in artifacts]
    keys = states[0].keys()
    if any(state.keys() != keys for state in states[1:]):
        raise ValueError("Las capas de los checkpoints no coinciden")
    mixture = {}
    for key in keys:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape or tensor.dtype != tensors[0].dtype
               for tensor in tensors[1:]):
            raise ValueError(f"Tensor incompatible: {key}")
        if not tensors[0].is_floating_point():
            raise ValueError(f"Tensor no interpolable: {key}")
        mixture[key] = sum((tensor.to(torch.float64) * coefficient
                            for tensor, coefficient in zip(tensors, COEFFICIENTS))).to(tensors[0].dtype)

    original_bias = json.loads(BASE_BIAS.read_text(encoding="utf-8"))
    if (original_bias.get("formato") != "gdg-sesgo-v1"
            or original_bias.get("pesos") != "models/exp033.pt"
            or len(original_bias.get("sesgo", [])) != 8):
        raise ValueError("Sesgo base inesperado")
    bias = [value * COEFFICIENTS[0] for value in original_bias["sesgo"]]
    artifact = {
        "formato": "gdg-cnn-v1", "red": "base", "ancho": 1,
        "fusion_tardia": False, "sin_atajos": True,
        "sin_escala": False, "sin_grupo": False,
        "state_dict": mixture,
        "mezcla": dict(zip(SOURCES, COEFFICIENTS)),
        "sha256_fuentes": {f"models/{name}.pt": sha256(path)
                            for name, path in zip(SOURCES, paths)},
    }
    torch.save(artifact, WEIGHTS)
    BIAS.write_text(json.dumps({
        "formato": "gdg-sesgo-v1", "pesos": "models/s01_sopa.pt",
        "sesgo": bias, "regla": "mitad del sesgo de exp033",
        "sha256_fuente": sha256(BASE_BIAS),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Creados {WEIGHTS.relative_to(ROOT)} y {BIAS.relative_to(ROOT)}")
    print("Mezcla:", artifact["mezcla"], "sesgo clase 6:", bias[6])


if __name__ == "__main__":
    main()
