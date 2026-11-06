"""Audita tres CNN entrenadas dejando fuera A/B/C, sin abrir test_demo.

Guarda predicciones OOF en el orden de val_10k. Con --referencia compara dos
arquitecturas entrenadas con los mismos pliegues mediante predicciones pareadas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from evaluar_ensamble_calidad import resumir, score
from src.red_convolucional import RedConvolucional, preparar_lote
from src.red_dos_rutas import RedDosRutas
from src.simulador import PARAMETROS

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "cv_grupos_v1"
GROUPS = "abc"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_model(model_id: str, group: str, fold: dict) -> tuple[torch.nn.Module, dict]:
    train_path = ROOT / f"experiments/{model_id}/entrenamiento.json"
    weight_path = ROOT / f"models/{model_id}.pt"
    if not train_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Faltan entrenamiento o pesos de {model_id}")
    train = json.loads(train_path.read_text(encoding="utf-8"))
    expected = fold["archivos"]["train_indices"]
    select = fold["archivos"]["select"]
    checks = {
        "indices_entrenamiento": expected["ruta"],
        "sha256_indices_entrenamiento": expected["sha256"],
        "fuente": "train_90k.npz",
        "muestras_seleccionadas": fold["train"]["n"],
        "muestras_entrenamiento": fold["train"]["n"],
        "archivo_examen": select["ruta"],
        "simular_examen": True,
        "sin_atajos": True,
        "sin_grupo": True,
        "sin_escala": False,
        "equilibrar": True,
        "elegir_epoca": "examen",
        "iteraciones": 200,
        "batch_size": 256,
        "ancho": 1,
        "fusion_tardia": False,
        "lr": 0.002,
        "semilla": 2026,
        "prob_escala_cero": 0.0,
        "peso_clase_6": 1.0,
        "igualar_correlacion_67": False,
        "peso_consistencia_vistas": 0.0,
        "mixup_alpha": 0.0,
        "peso_alineacion_demo": 0.0,
        "parametros_simulador": PARAMETROS,
    }
    for key, value in checks.items():
        if train.get(key) != value:
            raise ValueError(f"{model_id}: {key}={train.get(key)!r}, se esperaba {value!r}")
    if sha256(ROOT / select["ruta"]) != select["sha256"]:
        raise ValueError(f"Cambió el control de época del grupo {group.upper()}")
    artifact = torch.load(weight_path, map_location="cpu", weights_only=True)
    if (artifact.get("sin_grupo") is not True or artifact.get("red") != train["red"]
            or artifact.get("batch_size") != train["batch_size"]):
        raise ValueError(f"{model_id}: los pesos no coinciden con el protocolo de entrenamiento")
    if train["red"] == "base":
        model = RedConvolucional(ancho=int(artifact["ancho"]))
    elif train["red"] == "dos_rutas":
        model = RedDosRutas(ancho=int(artifact["ancho"]))
    else:
        raise ValueError(f"{model_id}: arquitectura fuera de esta comparación fijada")
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def predict(model: torch.nn.Module, artifact: dict, data: dict, indices: np.ndarray) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(indices), 1024):
            batch = indices[start:start + 1024]
            series, aux = preparar_lote(
                data["series"][batch], data["meta__sample_rate"][batch],
                data["meta__level"][batch], data["meta__group"][batch],
                sin_atajos=bool(artifact["sin_atajos"]),
                sin_escala=bool(artifact.get("sin_escala", False)),
            )
            aux[:, 14:17] = 0.0
            predictions.append(model(torch.from_numpy(series), torch.from_numpy(aux)).argmax(1).numpy())
    return np.concatenate(predictions)


def paired_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, groups: np.ndarray,
              draws: int = 2000) -> list[float]:
    rng = np.random.default_rng(20261105)
    strata = [np.flatnonzero((groups == group) & (y == label))
              for group in "ABC" for label in range(8)]
    deltas = np.empty(draws)
    for draw in range(draws):
        take = np.concatenate([rng.choice(part, len(part), replace=True) for part in strata])
        deltas[draw] = score(y[take], candidate[take]) - score(y[take], base[take])
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nombre", required=True, help="Etiqueta única del conjunto, p. ej. base o dos_rutas")
    parser.add_argument("--ids", nargs=3, required=True, metavar=("ID_A", "ID_B", "ID_C"),
                        help="Modelo entrenado dejando fuera A, B y C, respectivamente")
    parser.add_argument("--referencia", help="Nombre ya auditado para comparación pareada")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]{1,24}", args.nombre):
        parser.error("--nombre solo admite minúsculas, números y guion bajo")
    if args.referencia is not None and not re.fullmatch(r"[a-z0-9_]{1,24}", args.referencia):
        parser.error("--referencia no es válida")
    report_path = ROOT / f"experiments/{PREFIX}_{args.nombre}.json"
    pred_path = ROOT / f"data/{PREFIX}_{args.nombre}_pred.npz"
    if report_path.exists() or pred_path.exists():
        raise FileExistsError("La auditoría ya existe; no se sobrescribe")
    manifest = json.loads((ROOT / f"experiments/{PREFIX}_preparacion.json").read_text(encoding="utf-8"))
    if manifest["protocolo"] != PREFIX:
        raise ValueError("Manifiesto de otro protocolo")
    for name in ("train_90k", "val_10k", "val_examen_v2"):
        source = manifest["fuentes"][name]
        if sha256(ROOT / source["ruta"]) != source["sha256"]:
            raise ValueError(f"Cambió {name} desde la preparación")
    with np.load(ROOT / "data/val_10k.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in file.files}
    with np.load(ROOT / "data/val_examen_v2.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}
    y = np.asarray(raw["labels"], dtype=np.int64)
    if not np.array_equal(y, simulated["labels"]) or not np.array_equal(raw["sample_index"], simulated["sample_index"]):
        raise ValueError("Las dos vistas de validación no están alineadas")
    torch.set_num_threads(1)
    oof = {"raw": np.full(len(y), -1, dtype=np.int64), "sim": np.full(len(y), -1, dtype=np.int64)}
    folds = {}
    architectures = set()
    for group, model_id in zip(GROUPS, args.ids):
        fold = manifest["folds"][group]
        model, artifact = load_model(model_id, group, fold)
        architectures.add(artifact["red"])
        audit_file = fold["archivos"]["audit_indices"]
        if sha256(ROOT / audit_file["ruta"]) != audit_file["sha256"]:
            raise ValueError(f"Cambió la auditoría del grupo {group.upper()}")
        indices = np.load(ROOT / audit_file["ruta"], allow_pickle=False)
        if not np.all(raw["meta__group"][indices] == group.upper()):
            raise ValueError(f"Índices de auditoría incorrectos para {group.upper()}")
        row = {"id": model_id, "n": len(indices)}
        for view, data in (("raw", raw), ("sim", simulated)):
            prediction = predict(model, artifact, data, indices)
            oof[view][indices] = prediction
            row[view] = resumir(y[indices], prediction)
        folds[group] = row
        print(f"{group.upper()}: crudo {row['raw']['score_mismo_n']:.4f}, "
              f"simulado {row['sim']['score_mismo_n']:.4f}", flush=True)
    if len(architectures) != 1 or any(np.any(pred < 0) for pred in oof.values()):
        raise ValueError("Los tres pliegues no forman una predicción OOF completa de una arquitectura")

    aggregate = {view: resumir(y, pred) for view, pred in oof.items()}
    report = {"protocolo": PREFIX, "nombre": args.nombre, "ids": args.ids,
              "arquitectura": architectures.pop(), "regla": "grupo reservado fuera de train y selección de época; auxiliares de grupo anuladas",
              "folds": folds, "oof": aggregate,
              "limite": "Los tres grupos y el simulador no son un test externo nuevo; el IC remuestrea ejemplos dentro de grupo y clase."}
    if args.referencia:
        reference_file = ROOT / f"data/{PREFIX}_{args.referencia}_pred.npz"
        reference_report = ROOT / f"experiments/{PREFIX}_{args.referencia}.json"
        if not reference_file.is_file() or not reference_report.is_file():
            raise FileNotFoundError("Falta la auditoría de referencia")
        reference = json.loads(reference_report.read_text(encoding="utf-8"))
        if reference.get("protocolo") != PREFIX or reference.get("predicciones", {}).get("sha256") != sha256(reference_file):
            raise ValueError("La referencia no corresponde a este protocolo o cambió desde su auditoría")
        with np.load(reference_file, allow_pickle=False) as file:
            if not np.array_equal(file["sample_index"], raw["sample_index"]) or not np.array_equal(file["labels"], y):
                raise ValueError("Referencia no alineada con este protocolo")
            base = {view: file[view] for view in ("raw", "sim")}
        comparison = {"referencia": args.referencia}
        for view in ("raw", "sim"):
            comparison[view] = {
                "delta_score_oof": aggregate[view]["score_mismo_n"] - score(y, base[view]),
                "delta_accuracy_oof": aggregate[view]["accuracy"] - float(np.mean(base[view] == y)),
                "ic95_delta_score_pareado": paired_ci(y, base[view], oof[view], raw["meta__group"]),
                "delta_score_por_grupo": {group: folds[group][view]["score_mismo_n"]
                                          - score(y[raw["meta__group"] == group.upper()],
                                                  base[view][raw["meta__group"] == group.upper()])
                                          for group in GROUPS},
            }
        report["comparacion"] = comparison
    np.savez(pred_path, sample_index=raw["sample_index"], labels=y, raw=oof["raw"], sim=oof["sim"])
    report["predicciones"] = {"ruta": str(pred_path.relative_to(ROOT)), "sha256": sha256(pred_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OOF crudo {aggregate['raw']['score_mismo_n']:.4f}, simulado {aggregate['sim']['score_mismo_n']:.4f}")
    print(f"Informe: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
