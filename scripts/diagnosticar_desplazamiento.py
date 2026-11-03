"""Comprueba si centrar y escalar cada canal elimina el cambio train→demo.

Compara 400 ejemplos de train equilibrados por clase con los 400 del demo.
Un AUC de 0,5 significa que un clasificador no distingue los dos orígenes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def rasgos(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bruto, forma, huecos = [], [], []
    tiempo = np.arange(256)
    for muestra in series:
        limpia = np.empty((4, 256), dtype=np.float64)
        falta = ~np.isfinite(muestra)
        for canal in range(4):
            valido = ~falta[canal]
            limpia[canal] = (np.interp(tiempo, tiempo[valido], muestra[canal, valido])
                             if valido.any() else 0.0)
        media = limpia.mean(axis=1)
        desviacion = limpia.std(axis=1)
        z = (limpia - media[:, None]) / (desviacion[:, None] + 1e-6)
        fft = abs(np.fft.rfft(z, axis=1)) ** 2
        energia_alta = fft[:, 64:].sum(axis=1) / (fft[:, 1:].sum(axis=1) + 1e-9)
        autocorr_1 = (z[:, 1:] * z[:, :-1]).mean(axis=1)
        autocorr_8 = (z[:, 8:] * z[:, :-8]).mean(axis=1)
        rugosidad = abs(np.diff(z, axis=1)).mean(axis=1)
        curtosis = (z ** 4).mean(axis=1)
        bruto.append(np.concatenate([media, desviacion]))
        forma.append(np.concatenate([energia_alta, autocorr_1, autocorr_8, rugosidad, curtosis]))
        huecos.append(falta.mean(axis=1))
    return np.asarray(bruto), np.asarray(forma), np.asarray(huecos)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261103)
    prob = cross_val_predict(modelo, x, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, prob))


def main() -> None:
    rng = np.random.default_rng(20261103)
    with np.load(ROOT / "data/train_100k.npz", allow_pickle=False) as train:
        y = train["labels"]
        indices = np.concatenate([rng.choice(np.flatnonzero(y == k), 50, replace=False)
                                  for k in range(8)])
        train_series = train["series"][indices]
    with np.load(ROOT / "data/test_demo.npz", allow_pickle=False) as demo:
        demo_series = demo["series"]
    with np.load(ROOT / "data/pseudo_test.npz", allow_pickle=False) as pseudo:
        y_p = pseudo["labels"]
        indices_p = np.concatenate([rng.choice(np.flatnonzero(y_p == k), 50, replace=False)
                                    for k in range(8)])
        pseudo_series = pseudo["series"][indices_p]
    if len(demo_series) != 400:
        raise ValueError("El diagnóstico espera 400 muestras del demo")
    bruto_t, forma_t, huecos_t = rasgos(train_series)
    bruto_d, forma_d, huecos_d = rasgos(demo_series)
    bruto_p, forma_p, huecos_p = rasgos(pseudo_series)
    dominio = np.r_[np.zeros(400, dtype=int), np.ones(400, dtype=int)]
    forma = np.vstack([forma_t, forma_d])
    huecos = np.vstack([huecos_t, huecos_d])
    bruto = np.vstack([bruto_t, bruto_d])
    resultado = {
        "train_equilibrado": 400,
        "demo": 400,
        "semilla": 20261103,
        "auc_bruto_y_forma": auc(np.hstack([bruto, forma, huecos]), dominio),
        "auc_forma_tras_zscore": auc(forma, dominio),
        "auc_forma_y_huecos_tras_zscore": auc(np.hstack([forma, huecos]), dominio),
        "auc_pseudo_demo_bruto_y_forma": auc(np.vstack([
            np.hstack([bruto_p, forma_p, huecos_p]),
            np.hstack([bruto_d, forma_d, huecos_d]),
        ]), dominio),
        "auc_pseudo_demo_forma_tras_zscore": auc(np.vstack([forma_p, forma_d]), dominio),
        "nan_train": float(np.isnan(train_series).mean()),
        "nan_demo": float(np.isnan(demo_series).mean()),
        "nan_pseudo": float(np.isnan(pseudo_series).mean()),
        "energia_alta_train": float(np.median(forma_t[:, :4])),
        "energia_alta_demo": float(np.median(forma_d[:, :4])),
        "energia_alta_pseudo": float(np.median(forma_p[:, :4])),
        "rugosidad_train": float(np.median(forma_t[:, 12:16])),
        "rugosidad_demo": float(np.median(forma_d[:, 12:16])),
        "rugosidad_pseudo": float(np.median(forma_p[:, 12:16])),
        "limite": "AUC de 5 pliegues sobre solo 400 muestras por origen; demuestra un cambio residual, no identifica su causa.",
    }
    destino = ROOT / "experiments/auditoria_desplazamiento.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
