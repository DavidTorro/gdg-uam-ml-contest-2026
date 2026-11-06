"""Audita exp052 en las 5000 señales bloqueadas, sin leer el demo.

Compara predicciones pareadas de exp033 y exp052 sobre la misma señal base en
crudo y simulada. El subgrupo difícil de clase 6 se definió antes de entrenar
con umbrales de train. El score calculado aquí usa el mismo número de muestras
para ambos modelos; la velocidad se comprueba aparte con el harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from model import Model
from scripts.calibrar_simulador import rasgos


def score(y: np.ndarray, pred: np.ndarray) -> float:
    n = np.bincount(y, minlength=8)
    c = np.bincount(y[y == pred], minlength=8)
    term = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, term)).mean()))


def predictions(path: str, data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    old_model = os.environ.get("GDG_MODEL_PATH")
    old_bias = os.environ.pop("GDG_SESGO_PATH", None)
    os.environ["GDG_MODEL_PATH"] = path
    try:
        model = Model()
        data = load_dataset(data_path)
        pred = np.fromiter((model.predict(data[i]) for i in range(len(data))), dtype=np.int64,
                           count=len(data))
        return np.asarray(data.labels, dtype=np.int64), pred
    finally:
        if old_model is None:
            os.environ.pop("GDG_MODEL_PATH", None)
        else:
            os.environ["GDG_MODEL_PATH"] = old_model
        if old_bias is not None:
            os.environ["GDG_SESGO_PATH"] = old_bias


def paired_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray,
              metric: str, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(2000)
    for i in range(len(values)):
        positions = np.concatenate([rng.choice(np.flatnonzero(y == k),
                                                size=int((y == k).sum()), replace=True)
                                    for k in np.unique(y)])
        if metric == "score":
            values[i] = score(y[positions], candidate[positions]) - score(y[positions], base[positions])
        else:
            values[i] = (candidate[positions] == y[positions]).mean() - (base[positions] == y[positions]).mean()
    return [round(float(v), 5) for v in np.quantile(values, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidato", default="models/exp052.pt")
    args = parser.parse_args()
    if Path(args.candidato).is_absolute() or not (ROOT / args.candidato).resolve().is_relative_to(ROOT / "models"):
        parser.error("Usa una ruta relativa dentro de models/")
    candidate_path = ROOT / args.candidato
    if not candidate_path.is_file():
        parser.error(f"Falta {candidate_path}")
    manifest = json.loads((ROOT / "experiments/exp052_preparacion.json").read_text(encoding="utf-8"))
    threshold = manifest["umbrales_de_train_clase_7"]
    raw_path = ROOT / "data/val_exp052_audit_raw.npz"
    simulated_path = ROOT / "data/val_exp052_audit.npz"
    for key in ("indices", "audit", "audit_raw"):
        record = manifest["archivos"][key]
        path = ROOT / record["ruta"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"El hash de {path} no coincide con la preparación de exp052")
    training_report = ROOT / "experiments/exp052/entrenamiento.json"
    if args.candidato == "models/exp052.pt":
        if not training_report.is_file():
            raise FileNotFoundError(f"Falta el informe de entrenamiento: {training_report}")
        training = json.loads(training_report.read_text(encoding="utf-8"))
        if training["sha256_indices_entrenamiento"] != manifest["archivos"]["indices"]["sha256"]:
            raise ValueError("El entrenamiento exp052 usó un archivo de índices diferente")
    with np.load(raw_path, allow_pickle=False) as file:
        y = file["labels"].astype(np.int64)
        indices = np.flatnonzero(np.isin(y, (6, 7)))
        feats, info = rasgos(file["series"][indices], file["meta__level"][indices])
    col = {item[0]: i for i, item in enumerate(info)}
    clean_67 = ((feats[:, col["relacion_c1c2"]] <= threshold["corr12_max"])
                & (feats[:, col["proporcion_c2"]] <= threshold["log_size2_over_0_max"])
                & (feats[:, col["relacion_c0c1"]] >= threshold["corr01_min"]))
    clean_6 = np.zeros(len(y), dtype=bool)
    clean_6[indices] = clean_67 & (y[indices] == 6)
    masks = {"clase_6_limpia": clean_6, "clase_6_toda": y == 6, "clase_7_toda": y == 7}
    report = {"id": "exp052", "candidato": args.candidato,
              "base": "models/exp033.sesgo.npz", "bootstrap": "2000 remuestreos pareados estratificados por clase",
              "nota": "val_10k completo influyó históricamente en exp033; el bloque audit queda fuera de la elección de época de exp052"}
    for name, path in (("simulada", simulated_path), ("cruda", raw_path)):
        y_base, base = predictions("models/exp033.sesgo.npz", path)
        y_candidate, candidate = predictions(args.candidato, path)
        if not np.array_equal(y_base, y_candidate) or not np.array_equal(y, y_base):
            raise ValueError("Las etiquetas del bloque raw y simulado no coinciden")
        result = {"n": len(y), "score_base": score(y, base), "score_candidato": score(y, candidate),
                  "delta_score": score(y, candidate) - score(y, base),
                  "ic95_delta_score": paired_ci(y, base, candidate, "score", 20261105),
                  "accuracy_base": float((base == y).mean()),
                  "accuracy_candidato": float((candidate == y).mean()),
                  "subgrupos": {}}
        for subgroup, mask in masks.items():
            local_y = y[mask]
            local_base = base[mask]
            local_candidate = candidate[mask]
            result["subgrupos"][subgroup] = {
                "n": int(mask.sum()),
                "accuracy_base": float((local_base == local_y).mean()),
                "accuracy_candidato": float((local_candidate == local_y).mean()),
                "delta_accuracy": float((local_candidate == local_y).mean() - (local_base == local_y).mean()),
                "ic95_delta_accuracy": paired_ci(local_y, local_base, local_candidate, "accuracy", 20261105),
            }
        report[name] = result
    sim = report["simulada"]
    raw = report["cruda"]
    report["pasa_filtro_predefinido"] = bool(
        sim["delta_score"] >= 0.5
        and sim["subgrupos"]["clase_6_limpia"]["delta_accuracy"] >= 0.08
        and sim["subgrupos"]["clase_7_toda"]["delta_accuracy"] >= -0.02
        and raw["delta_score"] >= 0.0
    )
    destination = ROOT / "experiments/exp052_auditoria.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
