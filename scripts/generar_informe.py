"""Convierte los JSON del evaluador oficial en páginas y gráficos de MkDocs"""

from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path

import graficos

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTOS = ROOT / "experiments"
RESULTADOS = ROOT / "results"
DOCS = ROOT / "docs"
PAGINAS = DOCS / "experimentos"
ACTIVOS = PAGINAS / "assets"


def leer_json(ruta: Path) -> dict:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    return json.loads(ruta.read_text(encoding="utf-8"))


def numero(valor: float | None, decimales: int = 4) -> str:
    return "—" if valor is None else f"{valor:.{decimales}f}"


def porcentaje(valor: float | None) -> str:
    return "—" if valor is None else f"{100 * valor:.2f} %"


def tarjeta(etiqueta: str, valor: str, detalle: str, color: str) -> str:
    """Presenta una métrica con un color estable y texto legible en ambos temas"""
    return (
        f'<div class="ml-kpi ml-kpi--{color}">'
        f'<span class="ml-kpi__label">{escape(etiqueta)}</span>'
        f'<strong class="ml-kpi__value">{escape(valor)}</strong>'
        f'<span class="ml-kpi__detail">{escape(detalle)}</span>'
        '</div>'
    )


def figura(archivo: str, titulo: str, descripcion: str) -> str:
    """Inserta un SVG con pie de figura y texto alternativo"""
    return (
        '<figure class="ml-figure">'
        f'<img src="assets/{escape(archivo, quote=True)}" '
        f'alt="{escape(titulo, quote=True)}" loading="lazy">'
        f'<figcaption><strong>{escape(titulo)}</strong> · {escape(descripcion)}</figcaption>'
        '</figure>'
    )


def leer_evaluacion(identificador: str) -> tuple[dict, list[dict], dict, dict]:
    """Lee los archivos que produce realmente harness.py"""
    agregado = leer_json(RESULTADOS / identificador / "model" / "aggregate.json")
    ejecuciones = [
        leer_json(RESULTADOS / identificador / "model" / f"run_{indice}.json")
        for indice in range(agregado["n_runs"])
    ]
    entrenamiento = leer_json(EXPERIMENTOS / identificador / "entrenamiento.json")
    configuracion = leer_json(EXPERIMENTOS / identificador / "configuracion.json")
    if agregado.get("project") != "model":
        raise ValueError("El agregado no corresponde a model.py")
    if any(run.get("eval_config_hash") != agregado.get("eval_config_hash") for run in ejecuciones):
        raise ValueError("Las ejecuciones tienen configuraciones oficiales distintas")
    return agregado, ejecuciones, entrenamiento, configuracion


def historico() -> list[dict]:
    """Recupera solo experimentos con todos los archivos necesarios"""
    registros = []
    if not EXPERIMENTOS.exists():
        return registros
    for carpeta in EXPERIMENTOS.iterdir():
        if not carpeta.is_dir():
            continue
        try:
            agregado, _, entrenamiento, configuracion = leer_evaluacion(carpeta.name)
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
        registros.append({
            "id": carpeta.name,
            "fecha": configuracion.get("fecha_utc", ""),
            "agregado": agregado,
            "entrenamiento": entrenamiento,
            "configuracion": configuracion,
        })
    return sorted(registros, key=lambda registro: (registro["fecha"], registro["id"]))


def clave_comparacion(registro: dict) -> tuple:
    # Solo comparamos cifras obtenidas con el mismo archivo y límites oficiales
    agregado = registro["agregado"]
    configuracion = registro["configuracion"]
    return (
        configuracion["evaluacion"]["source"],
        configuracion["particion"].get("sha256_archivo_validacion"),
        agregado["eval_config_hash"],
    )


def generar_pagina(identificador: str, agregado: dict, ejecuciones: list[dict],
                   entrenamiento: dict, configuracion: dict, registros: list[dict]) -> None:
    """Escribe el informe visual desde las métricas oficiales"""
    referencia = next(registro for registro in registros if registro["id"] == identificador)
    comparables = [registro for registro in registros if clave_comparacion(registro) == clave_comparacion(referencia)]
    anteriores = [
        registro for registro in comparables
        if registro["id"] != identificador
        and registro["fecha"] < referencia["fecha"]
        and registro["agregado"]["failed_runs"] == 0
    ]
    mejor_anterior = max(anteriores, key=lambda r: r["agregado"]["ranking_score"], default=None)

    # Sumamos matrices y recalculamos accuracy por clase con todas las muestras
    clases = int(ejecuciones[0]["n_classes"])
    columnas = ejecuciones[0]["confusion_columns"]
    matriz = [[0] * len(columnas) for _ in range(clases)]
    for run in ejecuciones:
        if run["confusion_columns"] != columnas or len(run["confusion"]) != clases:
            raise ValueError("Las matrices de confusión no tienen el mismo formato")
        for clase, fila in enumerate(run["confusion"]):
            if len(fila) != len(columnas):
                raise ValueError("La matriz de confusión está incompleta")
            for columna, valor in enumerate(fila):
                matriz[clase][columna] += valor
    aciertos_clase = [matriz[clase][clase] / sum(fila) if sum(fila) else None for clase, fila in enumerate(matriz)]

    ACTIVOS.mkdir(parents=True, exist_ok=True)
    graficas = {
        "scores": graficos.ejecuciones(ejecuciones),
        "clases": graficos.clases(matriz),
        "confusion": graficos.confusion(matriz, columnas),
        "latencias": graficos.latencias(ejecuciones),
        "entrenamiento": graficos.entrenamiento(entrenamiento),
    }
    if len(comparables) > 1:
        graficas["historial"] = graficos.historial(comparables, identificador)
    for nombre, svg in graficas.items():
        (ACTIVOS / f"{identificador}_{nombre}.svg").write_text(svg, encoding="utf-8")

    # Los hallazgos proceden de las cuentas reales, sin inferir resultados futuros
    con_muestras = [clase for clase, fila in enumerate(matriz) if sum(fila)]
    clase_debil = min(con_muestras, key=lambda clase: aciertos_clase[clase]) if con_muestras else None
    confusiones = [
        (matriz[real][predicha], real, predicha)
        for real in range(clases) for predicha in range(clases) if real != predicha
    ]
    confusion_principal = max(confusiones, default=(0, 0, 0))

    parametros = configuracion["evaluacion"]
    estado = "Correcto" if agregado["failed_runs"] == 0 else f"{agregado['failed_runs']} ejecución(es) fallidas"
    estilo_estado = "ok" if agregado["failed_runs"] == 0 else "warn"
    latencia_media = agregado["mean_latency_s"] * 1000 if agregado["mean_latency_s"] is not None else None
    texto = [
        f"# Experimento {identificador}",
        "",
        '<div class="ml-report-head">',
        '<div class="ml-report-head__eyebrow">DAVIDTORROTEAM · REGISTRO DE MODELO</div>',
        f'<div class="ml-report-head__title">{escape(entrenamiento["modelo"])} <span class="ml-pill ml-pill--{estilo_estado}">{escape(estado)}</span></div>',
        f'<div class="ml-report-head__meta">{escape(configuracion["fecha_utc"])} · {parametros["runs"]} ejecución(es) · validación 90/10</div>',
        '</div>',
        "",
        '<div class="ml-kpi-grid">',
        tarjeta("Puntuación oficial", numero(agregado["ranking_score"], 5),
                f"{agregado['ranking_metric']} de {agregado['n_runs']} ejecuciones", "violet"),
        tarjeta("Accuracy general", porcentaje(agregado["mean_accuracy_all"]),
                "Media de las ejecuciones", "teal"),
        tarjeta("Latencia media", f"{numero(latencia_media, 3)} ms",
                "Por muestra en predict", "cyan"),
        tarjeta("Muestras puntuadas", numero(agregado["mean_n_scored"], 0),
                "Media por ejecución", "orange"),
        tarjeta("Tiempo de entrenamiento", f"{numero(entrenamiento['segundos_entrenamiento_total'], 2)} s",
                "Características + ajuste", "pink"),
        tarjeta("Muestras de entrenamiento", str(entrenamiento["muestras_entrenamiento"]),
                f"{entrenamiento['caracteristicas_por_muestra']} características por muestra", "blue"),
        '</div>',
        "",
        "La puntuación oficial no es un porcentaje: combina aciertos por clase y número de muestras procesadas. El accuracy sí expresa la proporción de respuestas correctas",
        "",
        "## Lectura rápida",
        "",
        f"- **Clase más difícil en esta evaluación:** {clase_debil} con {porcentaje(aciertos_clase[clase_debil])} de accuracy" if clase_debil is not None else "- No hay muestras puntuadas por clase",
        f"- **Confusión más frecuente:** clase real {confusion_principal[1]} predicha como {confusion_principal[2]} en {confusion_principal[0]} muestras" if confusion_principal[0] else "- No se registraron confusiones entre clases",
        f"- **Alcance:** {parametros['runs']} ejecución(es), máximo de {parametros['max_scored_samples']} muestras y {parametros['time_budget_s']} s acumulados de predicción",
        "",
        "## Ejecuciones y puntuación",
        "",
        figura(f"{identificador}_scores.svg", "Puntuación y accuracy por ejecución",
               "Las dos barras usan escalas independientes"),
        "",
        "| Ejecución | Estado | Puntuación | Accuracy | Muestras | Latencia media | Final |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in ejecuciones:
        latencia = run.get("latency_s", {}).get("mean")
        texto.append(
            f"| {run['run']} | {run['status']} | {numero(run['score'], 5)} | "
            f"{porcentaje(run['accuracy_all'])} | {run['n_scored']} | "
            f"{numero(latencia * 1000 if latencia is not None else None, 3)} ms | {run.get('end_reason') or '—'} |"
        )
    texto.extend([
        "",
        "## Rendimiento por clase",
        "",
        figura(f"{identificador}_clases.svg", "Aciertos y errores por clase",
               "Cada barra se normaliza por el número de muestras reales de esa clase"),
        "",
        "| Clase | Aciertos | Total | Accuracy |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for clase, fila in enumerate(matriz):
        texto.append(f"| {clase} | {fila[clase]} | {sum(fila)} | {porcentaje(aciertos_clase[clase])} |")
    texto.extend([
        "",
        "## Matriz de confusión",
        "",
        "Las filas son las clases reales y las columnas las respuestas. `abstain` y `failed` recogen abstenciones y fallos de inferencia. El color de cada celda refleja su proporción dentro de la fila",
        "",
        figura(f"{identificador}_confusion.svg", "Matriz de confusión acumulada",
               "Número de muestras y porcentaje por clase real"),
        "",
        "## Velocidad y coste de entrenamiento",
        "",
        figura(f"{identificador}_latencias.svg", "Distribución de latencias",
               "Mediana, percentil 95 y percentil 99 de predict"),
        "",
        "| Ejecución | Mediana | p95 | p99 |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for run in ejecuciones:
        latencias = run.get("latency_s", {})
        valores = [
            numero(latencias[campo] * 1000, 3) + " ms" if campo in latencias else "—"
            for campo in ("median", "p95", "p99")
        ]
        texto.append(f"| {run['run']} | {' | '.join(valores)} |")
    texto.extend([
        "",
        figura(f"{identificador}_entrenamiento.svg", "Desglose del entrenamiento",
               "Preparación de características y ajuste del clasificador"),
        "",
        "## Comparación",
        "",
    ])
    if len(comparables) > 1:
        texto.extend([
            figura(f"{identificador}_historial.svg", "Historial comparable",
                   "Solo experimentos con el mismo archivo de validación y configuración oficial"),
            "",
        ])
    if mejor_anterior:
        diferencia = agregado["ranking_score"] - mejor_anterior["agregado"]["ranking_score"]
        texto.append(
            f"Frente al mejor experimento anterior con el mismo protocolo, "
            f"[{mejor_anterior['id']}]({mejor_anterior['id']}.md), "
            f"la diferencia de puntuación es **{diferencia:+.5f}**"
        )
    else:
        texto.append("Aún no hay un experimento anterior con el mismo protocolo de evaluación")
    if agregado["failed_runs"]:
        texto.extend(["", "!!! warning \"Ejecuciones fallidas\"", "    Consulta los JSON y logs en `results/` antes de interpretar la puntuación", ""])
    texto.extend([
        "", "## Protocolo y reproducibilidad", "",
        f"- Entrenamiento: `data/{entrenamiento['fuente']}` con semilla `{entrenamiento['semilla']}`",
        f"- Validación: `{parametros['source']}`",
        f"- Configuración oficial: `{agregado['eval_config_hash']}`",
        f"- Partición: `{configuracion['particion'].get('sha256_indices_validacion') or 'no registrada'}`",
        f"- Archivo de validación: SHA-256 `{configuracion['particion'].get('sha256_archivo_validacion') or 'no registrado'}`",
        f"- Pesos: `models/{identificador}.pkl` · SHA-256 `{configuracion['sha256']['pesos']}`",
        f"- Código de características: SHA-256 `{configuracion['sha256']['src/caracteristicas.py']}`",
        "", "Todas las cifras y los gráficos proceden de los JSON del evaluador oficial y del registro de entrenamiento", "",
    ])
    PAGINAS.mkdir(parents=True, exist_ok=True)
    (PAGINAS / f"{identificador}.md").write_text("\n".join(texto), encoding="utf-8")


def actualizar_indices(registros: list[dict]) -> None:
    """Reconstruye la tabla histórica y el panel a partir de los experimentos completos"""
    ultimo = registros[-1]
    compatibles = [r for r in registros if clave_comparacion(r) == clave_comparacion(ultimo)]
    validos = [r for r in compatibles if r["agregado"]["failed_runs"] == 0]
    mejor = max(validos, key=lambda r: r["agregado"]["ranking_score"], default=None)

    indice = [
        "# Historial de experimentos", "",
        '<div class="ml-report-head">',
        '<div class="ml-report-head__eyebrow">DAVIDTORROTEAM · EVOLUCIÓN</div>',
        '<div class="ml-report-head__title">Modelos, pruebas y decisiones</div>',
        f'<div class="ml-report-head__meta">{len(registros)} experimentos registrados · informes reproducibles</div>',
        '</div>', "",
        "Cada fila enlaza con el informe y los gráficos generados desde los JSON del evaluador oficial", "",
        "| Experimento | Fecha UTC | Muestras de entrenamiento | Ejecuciones | Puntuación | Accuracy | Estado | Protocolo |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for registro in reversed(registros):
        agregado = registro["agregado"]
        configuracion = registro["configuracion"]
        estado = "OK" if agregado["failed_runs"] == 0 else "Fallos"
        protocolo = "Actual" if clave_comparacion(registro) == clave_comparacion(ultimo) else "Distinto"
        indice.append(
            f"| [{registro['id']}]({registro['id']}.md) | {registro['fecha']} | "
            f"{registro['entrenamiento']['muestras_entrenamiento']} | {agregado['n_runs']} | "
            f"{numero(agregado['ranking_score'], 5)} | {porcentaje(agregado['mean_accuracy_all'])} | "
            f"{estado} | {protocolo} |"
        )
    indice.extend([
        "", "El protocolo `Actual` corresponde a la última ejecución. Para comparar puntuaciones deben coincidir el archivo de validación, la partición y la configuración oficial", ""
    ])
    if len(compatibles) > 1:
        ACTIVOS.mkdir(parents=True, exist_ok=True)
        (ACTIVOS / "historial_actual.svg").write_text(
            graficos.historial(compatibles, ultimo["id"]), encoding="utf-8"
        )
        indice.extend([
            "## Evolución con el protocolo actual", "",
            figura("historial_actual.svg", "Comparación de modelos compatibles",
                   "La puntuación se compara solo dentro del mismo protocolo"), "",
        ])
    (PAGINAS / "index.md").write_text("\n".join(indice), encoding="utf-8")

    panel = [
        "# Cuaderno de experimentos", "",
        '<div class="ml-report-head">',
        '<div class="ml-report-head__eyebrow">GDG UAM ML CONTEST 2026</div>',
        '<div class="ml-report-head__title">Laboratorio de modelos · DavidTorroTeam</div>',
        '<div class="ml-report-head__meta">Resultados locales, gráficos y evolución del proyecto</div>',
        '</div>', "",
        '<div class="ml-kpi-grid">',
        tarjeta("Última puntuación", numero(ultimo["agregado"]["ranking_score"], 5),
                f"Experimento {ultimo['id']}", "violet"),
        tarjeta("Último accuracy", porcentaje(ultimo["agregado"]["mean_accuracy_all"]),
                "Validación local", "teal"),
        tarjeta("Latencia media", f"{numero(ultimo['agregado']['mean_latency_s'] * 1000 if ultimo['agregado']['mean_latency_s'] is not None else None, 3)} ms",
                "Por muestra", "cyan"),
        '</div>', "",
        f"**Última ejecución:** [{ultimo['id']}](experimentos/{ultimo['id']}.md)", "",
    ]
    if mejor:
        panel.extend([
            "## Mejor resultado con el protocolo actual", "",
            f"[{mejor['id']}](experimentos/{mejor['id']}.md) · "
            f"puntuación **{numero(mejor['agregado']['ranking_score'], 5)}**", "",
        ])
    panel.extend([
        "Las puntuaciones locales proceden de la partición de validación. La evaluación final usará datos nuevos", "",
        "Consulta el [historial completo](experimentos/index.md) para revisar las ejecuciones y sus gráficos", "",
    ])
    (DOCS / "index.md").write_text("\n".join(panel), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("Identificador de experimento inválido")
    agregado, ejecuciones, entrenamiento, configuracion = leer_evaluacion(args.id)
    registros = historico()
    if not any(registro["id"] == args.id for registro in registros):
        raise ValueError(f"El experimento {args.id} no está completo")
    generar_pagina(args.id, agregado, ejecuciones, entrenamiento, configuracion, registros)
    actualizar_indices(registros)
    print(f"Informe generado en docs/experimentos/{args.id}.md")


if __name__ == "__main__":
    main()
