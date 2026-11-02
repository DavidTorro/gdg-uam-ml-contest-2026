"""Compara las características por clase entre entrenamiento, validación y demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from competition_api import load_dataset
from src.caracteristicas import extraer


def nombres_caracteristicas() -> list[str]:
    nombres = [f"canal_{c}.media" for c in range(4)]
    nombres += [f"canal_{c}.desviacion" for c in range(4)]
    nombres += [f"canal_{c}.percentil_{p}" for c in range(4) for p in (10, 50, 90)]
    nombres += [f"canal_{c}.rugosidad" for c in range(4)]
    nombres += [f"canal_{c}.autocorrelacion" for c in range(4)]
    nombres += [f"canal_{c}.banda_{b}" for c in range(4) for b in range(4)]
    nombres += [f"canal_{c}.fraccion_faltante" for c in range(4)]
    nombres += [f"correlacion_{a}_{b}" for a in range(4) for b in range(a + 1, 4)]
    nombres += ["sample_rate", "level", "grupo_A", "grupo_B", "grupo_C"]
    return nombres


def cargar(nombre: str) -> tuple[np.ndarray, np.ndarray]:
    ruta = ROOT / "data" / nombre
    dataset = load_dataset(ruta)
    print(f"Extrayendo características: {nombre} ({len(dataset)} muestras)", flush=True)
    matriz = np.stack([extraer(dataset[i]) for i in range(len(dataset))])
    etiquetas = np.asarray(dataset.labels, dtype=np.int64)
    return matriz, etiquetas


def resumir_clase(
    clase: int,
    nombres: list[str],
    entrenamiento: tuple[np.ndarray, np.ndarray],
    validacion: tuple[np.ndarray, np.ndarray],
    demo: tuple[np.ndarray, np.ndarray],
) -> dict:
    train_x, train_y = entrenamiento
    val_x, val_y = validacion
    demo_x, demo_y = demo
    base = train_x[train_y == clase]
    control = val_x[val_y == clase]
    examen = demo_x[demo_y == clase]
    if not len(base) or not len(control) or not len(examen):
        return {"clase": clase, "error": "Faltan muestras de esta clase en algún conjunto"}

    p05, p25, mediana, p75, p95 = np.percentile(base, [5, 25, 50, 75, 95], axis=0)
    escala = np.maximum(p75 - p25, np.maximum(np.abs(mediana) * 0.1, 1e-3))
    control_mediana = np.median(control, axis=0)
    demo_mediana = np.median(examen, axis=0)
    cambio_control = np.abs(control_mediana - mediana) / escala
    cambio_demo = np.abs(demo_mediana - mediana) / escala
    exceso = cambio_demo - cambio_control
    fuera_rango = np.mean((examen < p05) | (examen > p95), axis=0)

    # El ranking resta el cambio observado en la validación habitual.
    # Demo solo tiene 50 muestras por clase: esto señala hipótesis, no demuestra causas.
    principales = np.argsort(exceso)[::-1][:8]
    return {
        "clase": clase,
        "muestras": {"entrenamiento": int(len(base)), "validacion": int(len(control)), "demo": int(len(examen))},
        "caracteristicas_con_mayor_cambio": [
            {
                "nombre": nombres[int(i)],
                "mediana_entrenamiento": float(mediana[i]),
                "mediana_validacion": float(control_mediana[i]),
                "mediana_demo": float(demo_mediana[i]),
                "cambio_demo_en_escalas": float(cambio_demo[i]),
                "cambio_extra_frente_a_validacion": float(exceso[i]),
                "demo_fuera_del_5_95_por_ciento_entrenamiento": float(fuera_rango[i]),
            }
            for i in principales
        ],
    }


def main() -> None:
    nombres = nombres_caracteristicas()
    entrenamiento = cargar("train_90k.npz")
    validacion = cargar("val_10k.npz")
    demo = cargar("test_demo.npz")
    if any(matriz.shape[1] != len(nombres) for matriz, _ in (entrenamiento, validacion, demo)):
        raise ValueError("Los nombres ya no coinciden con las características de src/caracteristicas.py")

    informe = {
        "explicacion": "Cambios de medianas por clase; cifras altas sugieren diferencias entre conjuntos, no causalidad.",
        "clases": [
            resumir_clase(clase, nombres, entrenamiento, validacion, demo)
            for clase in range(8)
        ],
    }
    destino = ROOT / "experiments" / "diagnostico_generalizacion.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in informe["clases"]:
        print(f"\nClase {item['clase']}:")
        if "error" in item:
            print(item["error"])
            continue
        for dato in item["caracteristicas_con_mayor_cambio"][:5]:
            print(f"  {dato['nombre']}: cambio extra {dato['cambio_extra_frente_a_validacion']:.2f}")
    print(f"\nInforme completo: {destino}")


if __name__ == "__main__":
    main()
