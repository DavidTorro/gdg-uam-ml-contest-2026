"""Crea controles de estrés para los atajos de la clase 6 frente a la 7.

Solo modifica la clase 6 de val_examen_v2. Usa muestras de clase 7 de esa
misma validación como referencia de correlación y tamaño de canales. No lee
test_demo y los archivos resultantes NO estiman el score del test real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103
MODOS = ("correlacion", "escala", "ambas")


def completar(muestra: np.ndarray) -> np.ndarray:
    tiempo = np.arange(muestra.shape[1])
    salida = np.empty_like(muestra, dtype=np.float64)
    for canal in range(muestra.shape[0]):
        valido = np.isfinite(muestra[canal])
        salida[canal] = (np.interp(tiempo, tiempo[valido], muestra[canal, valido])
                         if valido.any() else 0.0)
    return salida


def estadisticas(muestra: np.ndarray) -> tuple[float, float, float]:
    llena = completar(muestra)
    desv = llena.std(axis=1)
    z1 = (llena[1] - llena[1].mean()) / (desv[1] + 1e-9)
    z2 = (llena[2] - llena[2].mean()) / (desv[2] + 1e-9)
    return (float(np.mean(z1 * z2)), float(desv[1] / (desv[0] + 1e-9)),
            float(desv[2] / (desv[0] + 1e-9)))


def transformar(muestra: np.ndarray, referencia: tuple[float, float, float], modo: str) -> np.ndarray:
    salida = muestra.copy()
    llena = completar(muestra)
    medias = llena.mean(axis=1)
    desv = llena.std(axis=1)
    rho_obj, ratio_1, ratio_2 = referencia

    if modo in ("correlacion", "ambas") and min(desv[1], desv[2]) > 1e-8:
        z1 = (llena[1] - medias[1]) / desv[1]
        z2 = (llena[2] - medias[2]) / desv[2]
        rho = float(np.mean(z1 * z2))
        residual = z2 - rho * z1
        residual /= residual.std() + 1e-9
        objetivo = float(np.clip(rho_obj, -0.95, 0.95))
        nuevo_2 = medias[2] + desv[2] * (objetivo * z1 + np.sqrt(1 - objetivo**2) * residual)
        validos = np.isfinite(salida[2])
        salida[2, validos] = nuevo_2[validos]

    if modo in ("escala", "ambas"):
        for canal, ratio in ((1, ratio_1), (2, ratio_2)):
            if desv[canal] <= 1e-8:
                continue
            validos = np.isfinite(salida[canal])
            actual = salida[canal, validos].astype(np.float64)
            salida[canal, validos] = medias[canal] + (actual - medias[canal]) * (
                desv[0] * ratio / desv[canal]
            )
    return salida


def main() -> None:
    origen = ROOT / "data/val_examen_v2.npz"
    destinos = {modo: ROOT / f"data/val_estres_6_{modo}.npz" for modo in MODOS}
    informe = ROOT / "experiments/validacion_estres_6.json"
    if informe.exists() or any(ruta.exists() for ruta in destinos.values()):
        raise FileExistsError("La validación de estrés ya existe; no se sobrescribe")

    rng = np.random.default_rng(SEED)
    with np.load(origen, allow_pickle=False) as datos:
        etiquetas = datos["labels"]
        indice_6 = np.flatnonzero(etiquetas == 6)
        indice_7 = np.flatnonzero(etiquetas == 7)
        donantes = rng.choice(indice_7, len(indice_6), replace=True)
        referencias = [estadisticas(datos["series"][i]) for i in donantes]
        base = datos["series"]
        resumen = {}
        for modo, destino in destinos.items():
            series = base.copy()
            for numero, indice in enumerate(indice_6):
                series[indice] = transformar(base[indice], referencias[numero], modo)
            otros = {clave: datos[clave] for clave in datos.files if clave != "series"}
            np.savez(destino, series=series, **otros)
            medidas = np.array([estadisticas(series[i]) for i in indice_6])
            resumen[modo] = {
                "archivo": str(destino.relative_to(ROOT)),
                "mediana_corr_12_clase_6": float(np.median(medidas[:, 0])),
                "mediana_std_1_sobre_0_clase_6": float(np.median(medidas[:, 1])),
                "mediana_std_2_sobre_0_clase_6": float(np.median(medidas[:, 2])),
            }
        original_6 = np.array([estadisticas(base[i]) for i in indice_6])
        original_7 = np.array([estadisticas(base[i]) for i in indice_7])
    informe.parent.mkdir(exist_ok=True)
    informe.write_text(json.dumps({
        "origen": str(origen.relative_to(ROOT)),
        "semilla": SEED,
        "muestras_clase_6": int(len(indice_6)),
        "muestras_clase_7": int(len(indice_7)),
        "medianas_original_6": np.median(original_6, axis=0).tolist(),
        "medianas_referencia_7": np.median(original_7, axis=0).tolist(),
        "orden_medidas": ["corr_12", "std_1_sobre_0", "std_2_sobre_0"],
        "variantes": resumen,
        "limite": "Prueba de estrés con etiquetas conocidas: no aproxima la distribución real ni sirve como score de selección.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
