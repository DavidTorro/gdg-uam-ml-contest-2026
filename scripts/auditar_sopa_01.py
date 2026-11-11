"""Audita la mezcla fija s01_sopa antes de consultar el demo."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))

from competition_api import load_dataset
from evaluar_ensamble_calidad import cargar, evaluar, resumir


OUTPUT = ROOT / "experiments/sopa_01_auditoria.json"
EXPECTED_MIX = {
    "exp033": 0.5,
    "d01_control": 1 / 6,
    "d02_control_27": 1 / 6,
    "d02_control_28": 1 / 6,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"No se sobrescribe {OUTPUT}")
    candidate_path = ROOT / "models/s01_sopa.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if candidate.get("mezcla") != EXPECTED_MIX:
        raise ValueError("La mezcla no coincide con el plan")
    for name, digest in candidate["sha256_fuentes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Cambió la fuente: {name}")
    bias_path = ROOT / "models/s01_sopa.sesgo.json"
    bias = json.loads(bias_path.read_text(encoding="utf-8"))
    if bias.get("sesgo") != [0.0] * 6 + [0.4707, 0.0]:
        raise ValueError("El sesgo no coincide con el plan")

    torch.set_num_threads(1)
    models = {name: cargar(name) for name in ("exp033", "s01_sopa")}
    report = {
        "protocolo": "sopa_01",
        "mezcla": EXPECTED_MIX,
        "sha256_candidato": sha256(candidate_path),
        "limite": "El bloque se ha consultado antes. Las vistas simulada y cruda comparten señales base; esta auditoría no estima el test oculto.",
    }
    for view, filename in (("sim", "val_exp052_audit"),
                           ("raw", "val_exp052_audit_raw")):
        data = load_dataset(ROOT / f"data/{filename}.npz")
        labels = np.asarray(data.labels, dtype=np.int64)
        predictions = evaluar(data, models, {})
        report[view] = {name: resumir(labels, predictions[name]) for name in models}
        print(view, *(f"{name}={report[view][name]['score_mismo_n']:.6f}"
                      for name in models), flush=True)

    base_sim = report["sim"]["exp033"]
    mix_sim = report["sim"]["s01_sopa"]
    base_raw = report["raw"]["exp033"]
    mix_raw = report["raw"]["s01_sopa"]
    b = base_sim["aciertos_por_clase"]
    m = mix_sim["aciertos_por_clase"]
    report["delta_score_sim"] = mix_sim["score_mismo_n"] - base_sim["score_mismo_n"]
    report["delta_score_raw"] = mix_raw["score_mismo_n"] - base_raw["score_mismo_n"]
    report["delta_aciertos_6_sim"] = m[6] - b[6]
    report["delta_aciertos_6_7_sim"] = m[6] + m[7] - b[6] - b[7]
    report["filtros"] = {
        "score_sim_mas_0_5": report["delta_score_sim"] >= 0.5,
        "score_raw_no_baja": report["delta_score_raw"] >= 0,
        "clase_6_pierde_maximo_15": report["delta_aciertos_6_sim"] >= -15,
        "clases_6_7_ganan_15": report["delta_aciertos_6_7_sim"] >= 15,
    }
    report["pasa"] = all(report["filtros"].values())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Pasa el filtro: {report['pasa']}; informe: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
