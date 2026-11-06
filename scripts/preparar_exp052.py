"""Prepara una única selección de train y una auditoría disjunta para exp052.

Los umbrales salen exclusivamente de la clase 7 de train_90k. Las etiquetas y
señales de test_demo no se abren. Los dos controles se separan por señal base
antes de entrenar; el bloque audit queda fuera de la elección de época.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from calibrar_simulador import rasgos

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261105


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guardar_bloque(path: Path, source: dict[str, np.ndarray], indices: np.ndarray) -> None:
    np.savez(path, **{key: value[indices] for key, value in source.items()})


def main() -> None:
    data = ROOT / "data"
    paths = {
        "indices": data / "exp052_indices.npy",
        "select": data / "val_exp052_select.npz",
        "audit": data / "val_exp052_audit.npz",
        "audit_raw": data / "val_exp052_audit_raw.npz",
        "manifest": ROOT / "experiments/exp052_preparacion.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Ya existen artefactos de exp052: {existing}")
    if (ROOT / "models/exp052.pt").exists() or (ROOT / "experiments/exp052").exists():
        raise FileExistsError("exp052 ya se usó; elige otro ID")

    rng = np.random.default_rng(SEED)
    with np.load(data / "train_90k.npz", allow_pickle=False) as file:
        labels = file["labels"]
        group = file["meta__group"]
        rate = file["meta__sample_rate"]
        indices_67 = np.flatnonzero(np.isin(labels, (6, 7)))
        feats, info = rasgos(file["series"][indices_67], file["meta__level"][indices_67])
    column = {item[0]: i for i, item in enumerate(info)}
    seven = labels[indices_67] == 7
    corr12 = feats[:, column["relacion_c1c2"]]
    size2 = feats[:, column["proporcion_c2"]]
    corr01 = feats[:, column["relacion_c0c1"]]
    thresholds = {
        "corr12_max": float(np.quantile(corr12[seven], 0.75)),
        "log_size2_over_0_max": float(np.quantile(size2[seven], 0.75)),
        "corr01_min": float(np.quantile(corr01[seven], 0.25)),
    }
    clean = ((corr12 <= thresholds["corr12_max"])
             & (size2 <= thresholds["log_size2_over_0_max"])
             & (corr01 >= thresholds["corr01_min"]))
    six_all = indices_67[~seven]
    six_clean = indices_67[(~seven) & clean]
    if len(six_clean) < 500:
        raise ValueError("Hay pocos ejemplos de clase 6 sin la señal común; revisar los datos")

    # Preserva la distribución de group y sample_rate de la clase 6. Cada
    # estrato conserva su tamaño: mitad desde el subgrupo limpio, mitad desde
    # todo el estrato. Las otras siete clases entran exactamente una vez.
    six_selected = []
    strata = []
    for g in np.unique(group[six_all]):
        for r in np.unique(rate[six_all]):
            all_stratum = six_all[(group[six_all] == g) & (rate[six_all] == r)]
            clean_stratum = six_clean[(group[six_clean] == g) & (rate[six_clean] == r)]
            if not len(clean_stratum):
                raise ValueError(f"Sin ejemplos limpios para group={g}, sample_rate={r}")
            n_clean = len(all_stratum) // 2
            picked_clean = rng.choice(clean_stratum, n_clean, replace=n_clean > len(clean_stratum))
            picked_all = rng.choice(all_stratum, len(all_stratum) - n_clean, replace=False)
            six_selected.extend((picked_clean, picked_all))
            strata.append({"group": str(g), "sample_rate": float(r),
                           "total": int(len(all_stratum)), "limpios_disponibles": int(len(clean_stratum)),
                           "plazas_limpias": int(n_clean)})
    selected = np.sort(np.concatenate([np.flatnonzero(labels != 6), *six_selected]).astype(np.int64))
    if len(selected) != len(labels) or np.any(np.bincount(labels[selected], minlength=8)
                                               != np.bincount(labels, minlength=8)):
        raise AssertionError("El remuestreo cambió el tamaño o las frecuencias por clase")

    with np.load(data / "val_10k.npz", allow_pickle=False) as file:
        raw = {key: file[key] for key in file.files}
    with np.load(data / "val_examen_v2.npz", allow_pickle=False) as file:
        simulated = {key: file[key] for key in file.files}
    if not np.array_equal(raw["labels"], simulated["labels"]):
        raise ValueError("val_10k y val_examen_v2 no están alineados")
    positions = np.arange(len(raw["labels"]))
    joint = np.array([f"{y}:{g}:{r:g}" for y, g, r in zip(
        raw["labels"], raw["meta__group"], raw["meta__sample_rate"])])
    select, audit = train_test_split(positions, train_size=0.5, stratify=joint, random_state=SEED)
    select, audit = np.sort(select), np.sort(audit)
    if np.intersect1d(select, audit).size or len(select) + len(audit) != len(positions):
        raise AssertionError("La validación de selección y auditoría se solapan")

    np.save(paths["indices"], selected, allow_pickle=False)
    guardar_bloque(paths["select"], simulated, select)
    guardar_bloque(paths["audit"], simulated, audit)
    guardar_bloque(paths["audit_raw"], raw, audit)
    manifest = {
        "id": "exp052", "seed": SEED,
        "regla": "50 % de las plazas de clase 6 por estrato desde ejemplos sin artefacto, 50 % desde todos",
        "umbrales_de_train_clase_7": thresholds,
        "clase_6_total": int(len(six_all)),
        "clase_6_limpios_disponibles": int(len(six_clean)),
        "clase_6_limpios_seleccionados": int(np.isin(selected[labels[selected] == 6], six_clean).sum()),
        "clase_6_unicos_seleccionados": int(len(np.unique(selected[labels[selected] == 6]))),
        "estratos": strata,
        "validacion": {"seleccion": int(len(select)), "auditoria": int(len(audit)),
                       "sha256_indices_seleccion": hashlib.sha256(select.tobytes()).hexdigest(),
                       "sha256_indices_auditoria": hashlib.sha256(audit.tobytes()).hexdigest()},
        "archivos": {key: {"ruta": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                     for key, path in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": "exp052", "clase_6_limpios_disponibles": len(six_clean),
                      "clase_6_limpios_seleccionados": manifest["clase_6_limpios_seleccionados"],
                      "clase_6_unicos_seleccionados": manifest["clase_6_unicos_seleccionados"],
                      "validacion_seleccion": len(select), "validacion_auditoria": len(audit)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
