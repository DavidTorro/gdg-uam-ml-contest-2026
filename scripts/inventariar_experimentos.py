"""Inventario reproducible de entrenamientos, pesos, datasets y evaluaciones locales.

Uso: .venv/bin/python scripts/inventariar_experimentos.py
No ejecuta entrenamiento ni inferencia. Los alias de evaluación se identifican por el
campo `pesos`; las rutas de results sin configuración se atribuyen por nombre y se
marcan explícitamente como inferencia, nunca como entrenamiento nuevo.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "experiments/inventario_experimentos.json"
# Identificados por los informes históricos del engram.md raíz. Son evaluaciones
# o ablaciones de inferencia del peso exp033, nunca ajustes independientes.
ALIAS_EXP033 = ("auditoria_base_", "estres_6_", "estres_ritmo0_base",
                "opt_aux_finitos_", "opt_interp_huecos_", "opt_ventanas_",
                "prueba_sin_escala_")


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def rel(ruta: Path) -> str:
    return str(ruta.relative_to(ROOT))


def modelo_base(nombre: str, ids: set[str]) -> tuple[str | None, str]:
    if nombre in ids:
        return nombre, "id_exacto"
    coincidencias = re.findall(r"exp\d{3}", nombre)
    if coincidencias and coincidencias[0] in ids:
        return coincidencias[0], "inferido_del_nombre"
    return None, "sin_atribucion"


def main() -> None:
    carpetas = sorted(d for d in (ROOT / "experiments").iterdir() if d.is_dir())
    entrenamientos = []
    alias = []
    peso_a_id: dict[str, str] = {}
    for carpeta in carpetas:
        ruta = carpeta / "entrenamiento.json"
        if not ruta.is_file():
            continue
        informe = leer(ruta)
        peso = informe.get("pesos")
        configuracion = carpeta / "configuracion.json"
        cfg = leer(configuracion) if configuracion.is_file() else {}
        registro = {
            "id": carpeta.name,
            "informe": rel(ruta),
            "configuracion": rel(configuracion) if configuracion.is_file() else None,
            "peso_declarado": peso,
            "fuente_train": informe.get("fuente"),
            "n_train": informe.get("muestras_entrenamiento"),
            "semilla": informe.get("semilla"),
            "red": informe.get("red", informe.get("modelo")),
            "mejor_epoca": informe.get("mejor_epoca"),
            "control_epoca": informe.get("archivo_examen"),
            "opciones": cfg.get("entrenamiento"),
            "evaluacion_inicial": cfg.get("evaluacion"),
        }
        if peso and Path(peso).stem == carpeta.name:
            entrenamientos.append(registro)
            peso_a_id[peso] = carpeta.name
        else:
            registro["entrenamiento_origen"] = peso_a_id.get(peso)
            alias.append(registro)
    # La ordenación de carpetas puede poner un alias antes del origen.
    for registro in alias:
        registro["entrenamiento_origen"] = peso_a_id.get(registro["peso_declarado"])

    pesos = []
    for ruta in sorted((ROOT / "models").iterdir()):
        if ruta.is_file():
            pesos.append({"ruta": rel(ruta), "bytes": ruta.stat().st_size,
                          "sha256": sha256(ruta), "entrenamiento_declarado": peso_a_id.get(rel(ruta))})

    datasets = []
    for ruta in sorted((ROOT / "data").iterdir()):
        if not ruta.is_file():
            continue
        registro = {"ruta": rel(ruta), "bytes": ruta.stat().st_size, "sha256": sha256(ruta)}
        if ruta.suffix == ".npz":
            with np.load(ruta, allow_pickle=False) as archivo:
                registro["arrays"] = {k: {"forma": list(archivo[k].shape),
                                           "tipo": str(archivo[k].dtype)} for k in archivo.files}
                if "labels" in archivo:
                    registro["n_etiquetas"] = int(len(archivo["labels"]))
        datasets.append(registro)

    ids = {r["id"] for r in entrenamientos}
    alias_a_origen = {r["id"]: r["entrenamiento_origen"] for r in alias}
    evaluaciones = []
    for ruta in sorted((ROOT / "results").glob("**/run_*.json")):
        informe = leer(ruta)
        carpeta = ruta.relative_to(ROOT / "results").parts[0]
        origen, metodo = modelo_base(carpeta, ids)
        if carpeta in alias_a_origen and alias_a_origen[carpeta]:
            origen, metodo = alias_a_origen[carpeta], "peso_declarado_en_alias"
        if carpeta.startswith(ALIAS_EXP033):
            origen, metodo = "exp033", "alias_documentado_en_engram"
        evaluaciones.append({
            "ruta": rel(ruta), "id_resultado": carpeta, "entrenamiento_origen": origen,
            "atribucion": metodo, "fuente": informe.get("source"),
            "n": informe.get("n_scored"), "hash_protocolo": informe.get("eval_config_hash"),
            "hash_flujo": informe.get("stream_hash"), "score": informe.get("score"),
            "accuracy": informe.get("accuracy_all"), "estado": informe.get("status"),
            "presupuesto_s": informe.get("model_time_s"),
        })

    protocolos = defaultdict(list)
    for e in evaluaciones:
        protocolos[(e["fuente"], e["n"], e["hash_protocolo"])].append(e["ruta"])
    salida = {
        "regla": "Solo los IDs cuyo peso declarado tiene el mismo nombre son entrenamientos; los demás son alias o reevaluaciones. Comparar score únicamente dentro del mismo conjunto, N y protocolo; para comparaciones auditadas usar las mismas filas y vista.",
        "resumen": {"entrenamientos": len(entrenamientos), "alias_con_informe_copiado": len(alias),
                    "pesos": len(pesos), "evaluaciones_harness": len(evaluaciones),
                    "datasets_y_arrays": len(datasets), "protocolos_harness": len(protocolos),
                    "evaluaciones_sin_atribucion": sum(e["entrenamiento_origen"] is None for e in evaluaciones)},
        "entrenamientos": entrenamientos, "alias": alias, "pesos": pesos,
        "datasets": datasets, "evaluaciones": evaluaciones,
        "protocolos_harness": [{"fuente": k[0], "n": k[1], "hash_protocolo": k[2],
                                 "resultados": v} for k, v in sorted(protocolos.items(), key=lambda x: str(x[0]))],
        "auditorias_agregadas": sorted(rel(r) for r in (ROOT / "experiments").glob("*.json")),
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(salida["resumen"], ensure_ascii=False))
    print(rel(SALIDA))


if __name__ == "__main__":
    main()
