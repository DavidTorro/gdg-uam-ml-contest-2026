"""Evalúa un ensamble fijo de calidad, sin cambiar model.py ni medir velocidad.

Regla congelada: media aritmética de probabilidades softmax de exp033, exp034
y exp051, cada uno con el sesgo ya existente si lo tiene. Compara contra
exp033 sesgado en los mismos ejemplos. No busca pesos ni umbrales en demo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.red_convolucional import RedConvolucional, preparar_lote
from src.red_compacta import RedCompacta
from src.red_dos_rutas import RedDosRutas
from src.red_multiescala import RedMultiescala

MODELOS = ("exp033", "exp034", "exp051")
FUENTES = ("test_demo", "val_exp052_audit", "val_10k", "pseudo_test")


def score(y: np.ndarray, pred: np.ndarray) -> float:
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    n = confusion.sum(1)
    c = np.diag(confusion)
    term = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, term)).mean()))


def resumir(y: np.ndarray, pred: np.ndarray) -> dict:
    confusion = np.zeros((8, 8), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    return {"n": len(y), "score_mismo_n": score(y, pred),
            "accuracy": float((pred == y).mean()),
            "aciertos_por_clase": np.diag(confusion).tolist(),
            "n_por_clase": confusion.sum(1).tolist(),
            "confusion": confusion.tolist()}


def cargar(nombre: str) -> tuple[torch.nn.Module, dict, np.ndarray]:
    artefacto = torch.load(ROOT / f"models/{nombre}.pt", map_location="cpu", weights_only=True)
    if artefacto.get("red") == "multiescala":
        red = RedMultiescala(ancho=int(artefacto.get("ancho", 1)))
    elif artefacto.get("red") == "compacta":
        red = RedCompacta()
    elif artefacto.get("red") == "dos_rutas":
        red = RedDosRutas(ancho=int(artefacto.get("ancho", 1)))
    else:
        red = RedConvolucional(ancho=int(artefacto.get("ancho", 1)),
                               fusion_tardia=bool(artefacto.get("fusion_tardia", False)))
    red.load_state_dict(artefacto["state_dict"])
    red.eval()
    sesgo_file = ROOT / f"models/{nombre}.sesgo.json"
    sesgo = (np.asarray(json.loads(sesgo_file.read_text(encoding="utf-8"))["sesgo"], dtype=np.float32)
             if sesgo_file.is_file() else np.zeros(8, dtype=np.float32))
    return red, artefacto, sesgo


def evaluar(datos, modelos: dict, combinaciones: dict[str, tuple[str, ...]]) -> dict[str, np.ndarray]:
    salidas = {nombre: [] for nombre in modelos}
    with torch.inference_mode():
        for start in range(0, len(datos), 1024):
            end = min(start + 1024, len(datos))
            prepared = {}
            for nombre, (red, artefacto, sesgo) in modelos.items():
                flags = (bool(artefacto.get("sin_atajos", False)),
                         bool(artefacto.get("sin_escala", False)))
                if flags not in prepared:
                    serie, aux = preparar_lote(
                        datos.series[start:end], datos.meta["sample_rate"][start:end],
                        datos.meta["level"][start:end], datos.meta["group"][start:end],
                        sin_atajos=flags[0], sin_escala=flags[1],
                    )
                    prepared[flags] = (torch.from_numpy(serie), torch.from_numpy(aux))
                serie, aux = prepared[flags]
                logits = red(serie, aux).numpy() + sesgo
                salidas[nombre].append(torch.softmax(torch.from_numpy(logits), dim=1).numpy())
    probabilities = {nombre: np.concatenate(parts) for nombre, parts in salidas.items()}
    for name, members in combinaciones.items():
        probabilities[name] = np.mean([probabilities[member] for member in members], axis=0)
    return {nombre: probs.argmax(1) for nombre, probs in probabilities.items()}


def bootstrap_delta(y: np.ndarray, base: np.ndarray, ensemble: np.ndarray,
                    *, draws: int = 3000) -> dict:
    rng = np.random.default_rng(20261105)
    strata = [np.flatnonzero(y == k) for k in range(8)]
    deltas_score = np.empty(draws)
    deltas_acc = np.empty(draws)
    for i in range(draws):
        index = np.concatenate([rng.choice(part, len(part), replace=True) for part in strata])
        deltas_score[i] = score(y[index], ensemble[index]) - score(y[index], base[index])
        deltas_acc[i] = (ensemble[index] == y[index]).mean() - (base[index] == y[index]).mean()
    return {"delta_score_ic95": np.quantile(deltas_score, (0.025, 0.975)).tolist(),
            "delta_accuracy_ic95": np.quantile(deltas_acc, (0.025, 0.975)).tolist()}


def main() -> None:
    destino = ROOT / "experiments/ensamble_033_034_051.json"
    if destino.exists():
        raise FileExistsError(f"Ya existe {destino}; no se recalcula ni sobrescribe")
    torch.set_num_threads(1)
    modelos = {nombre: cargar(nombre) for nombre in MODELOS}
    informe = {"modelos": list(MODELOS), "regla": "media uniforme de softmax, sesgos históricos conservados",
               "objetivo": "calidad predictiva; sin inferencia cronometrada ni cambio de entrega",
               "limites": "demo consultado repetidamente; pseudo reutiliza señales de val_10k; la auditoría exp052 ya fue inspeccionada"}
    for fuente in FUENTES:
        datos = load_dataset(ROOT / f"data/{fuente}.npz")
        y = np.asarray(datos.labels, dtype=np.int64)
        pred = evaluar(datos, modelos, {"ensamble": MODELOS})
        row = {name: resumir(y, pred[name]) for name in (*MODELOS, "ensamble")}
        row["delta_score_vs_exp033"] = row["ensamble"]["score_mismo_n"] - row["exp033"]["score_mismo_n"]
        row["delta_accuracy_vs_exp033"] = row["ensamble"]["accuracy"] - row["exp033"]["accuracy"]
        if fuente in ("test_demo", "val_exp052_audit"):
            row["bootstrap_pareado_estratificado"] = bootstrap_delta(y, pred["exp033"], pred["ensamble"])
        informe[fuente] = row
        print(f"{fuente}: exp033 {row['exp033']['score_mismo_n']:.6f} → "
              f"ensamble {row['ensamble']['score_mismo_n']:.6f}; "
              f"accuracy {row['exp033']['accuracy']:.2%} → {row['ensamble']['accuracy']:.2%}", flush=True)
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Informe guardado: {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
