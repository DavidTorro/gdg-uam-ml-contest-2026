"""Gráficos SVG del cuaderno generados sin dependencias de visualización"""

from __future__ import annotations

from html import escape

COLORES_CLASE = (
    "#2563eb", "#0891b2", "#7c3aed", "#db2777",
    "#ea580c", "#16a34a", "#4f46e5", "#0d9488",
)
CORRECTO = "#0f766e"
ERROR = "#ef4444"
ABSTENCION = "#f59e0b"
FALLO = "#8b5cf6"
TEXTO = "#16332f"
SECUNDARIO = "#64748b"
REJILLA = "#e2e8f0"


def _base(ancho: int, alto: int, titulo: str, subtitulo: str) -> list[str]:
    """Crea un lienzo accesible con la misma tipografía y paleta en todos los gráficos"""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" role="img" aria-label="{escape(titulo, quote=True)}">',
        f'<title>{escape(titulo)}</title>',
        f'<desc>{escape(subtitulo)}</desc>',
        '<defs><linearGradient id="fondo" x2="1" y2="1"><stop stop-color="#ffffff"/><stop offset="1" stop-color="#f3faf9"/></linearGradient></defs>',
        f'<rect x="1" y="1" width="{ancho-2}" height="{alto-2}" rx="20" fill="url(#fondo)" stroke="{REJILLA}"/>',
        f'<text x="38" y="42" font-family="Arial,Helvetica,sans-serif" font-size="22" font-weight="700" fill="{TEXTO}">{escape(titulo)}</text>',
        f'<text x="38" y="67" font-family="Arial,Helvetica,sans-serif" font-size="13" fill="{SECUNDARIO}">{escape(subtitulo)}</text>',
    ]


def _texto(x: float, y: float, contenido: str, *, tamano: int = 13,
           color: str = TEXTO, peso: int = 400, ancla: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{ancla}" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{tamano}" '
        f'font-weight="{peso}" fill="{color}">{escape(contenido)}</text>'
    )


def _terminar(partes: list[str]) -> str:
    return "\n".join([*partes, "</svg>", ""])


def ejecuciones(runs: list[dict]) -> str:
    """Dibuja puntuación y accuracy en escalas separadas para no confundirlas"""
    ancho, alto = 940, 155 + 88 * len(runs)
    partes = _base(ancho, alto, "Resultado por ejecución", "Puntuación oficial y accuracy general en cada ejecución")
    izquierda, anchura = 185, 590
    techo = max((float(run["score"]) for run in runs), default=1) * 1.10 or 1
    partes.extend([
        f'<rect x="{izquierda}" y="93" width="12" height="12" rx="3" fill="#7c3aed"/>',
        _texto(204, 104, "Puntuación", tamano=12, color=SECUNDARIO),
        f'<rect x="312" y="93" width="12" height="12" rx="3" fill="#0d9488"/>',
        _texto(331, 104, "Accuracy", tamano=12, color=SECUNDARIO),
    ])
    for indice, run in enumerate(runs):
        y = 127 + indice * 88
        score = max(0.0, float(run["score"]))
        accuracy = max(0.0, min(1.0, float(run.get("accuracy_all") or 0)))
        partes.extend([
            _texto(38, y + 19, f"Ejecución {run['run']}", tamano=15, peso=700),
            f'<rect x="{izquierda}" y="{y}" width="{anchura}" height="23" rx="11" fill="#ece9f8"/>',
            f'<rect x="{izquierda}" y="{y}" width="{anchura * score / techo:.1f}" height="23" rx="11" fill="#7c3aed"/>',
            _texto(792, y + 17, f"{score:.4f}", tamano=14, peso=700, color="#6d28d9"),
            f'<rect x="{izquierda}" y="{y+33}" width="{anchura}" height="14" rx="7" fill="#ddf4ef"/>',
            f'<rect x="{izquierda}" y="{y+33}" width="{anchura * accuracy:.1f}" height="14" rx="7" fill="#0d9488"/>',
            _texto(792, y + 46, f"{accuracy * 100:.1f} %", tamano=13, peso=700, color="#0f766e"),
        ])
    return _terminar(partes)


def clases(matriz: list[list[int]]) -> str:
    """Muestra aciertos, errores, abstenciones y fallos dentro de cada clase real"""
    ancho, alto = 940, 170 + 54 * len(matriz)
    partes = _base(ancho, alto, "Rendimiento por clase", "Cada barra representa el 100 % de las muestras reales de esa clase")
    for x, color, nombre in ((38, CORRECTO, "Acierto por clase"), (218, ERROR, "Error"),
                             (312, ABSTENCION, "Abstención"), (462, FALLO, "Fallo")):
        partes.extend([
            f'<circle cx="{x+6}" cy="99" r="6" fill="{color}"/>',
            _texto(x+20, 103, nombre, tamano=12, color=SECUNDARIO),
        ])
    izquierda, anchura = 155, 610
    for clase, fila in enumerate(matriz):
        y = 132 + clase * 54
        total = sum(fila)
        correcto = fila[clase]
        abstencion = fila[-2] if len(fila) > len(matriz) else 0
        fallo = fila[-1] if len(fila) > len(matriz) + 1 else 0
        error = total - correcto - abstencion - fallo
        partes.extend([
            f'<circle cx="53" cy="{y+12}" r="12" fill="{COLORES_CLASE[clase % len(COLORES_CLASE)]}"/>',
            _texto(53, y+16, str(clase), tamano=12, color="#ffffff", peso=700, ancla="middle"),
            _texto(76, y+16, f"Clase {clase}", tamano=13, peso=700),
            f'<rect x="{izquierda}" y="{y}" width="{anchura}" height="24" rx="12" fill="#e7eef0"/>',
        ])
        cursor = izquierda
        for cuenta, color in ((correcto, COLORES_CLASE[clase % len(COLORES_CLASE)]),
                              (error, ERROR), (abstencion, ABSTENCION), (fallo, FALLO)):
            ancho_segmento = anchura * cuenta / total if total else 0
            if ancho_segmento > 0:
                partes.append(f'<rect x="{cursor:.1f}" y="{y}" width="{ancho_segmento:.1f}" height="24" fill="{color}"/>')
            cursor += ancho_segmento
        porcentaje = 100 * correcto / total if total else 0
        partes.extend([
            _texto(789, y+17, f"{porcentaje:.1f} %", tamano=14, peso=700),
            _texto(789, y+34, f"{correcto}/{total} aciertos", tamano=11, color=SECUNDARIO),
        ])
    partes.append(_texto(38, alto-24, "Los colores de la parte correcta identifican cada clase; rojo indica una clase equivocada", tamano=12, color=SECUNDARIO))
    return _terminar(partes)


def confusion(matriz: list[list[int]], columnas: list[str]) -> str:
    """Colorea las celdas según sean aciertos, confusiones o respuestas especiales"""
    celda = 60
    izquierda, arriba = 148, 160
    ancho, alto = izquierda + celda * len(columnas) + 110, arriba + celda * len(matriz) + 54
    partes = _base(ancho, alto, "Matriz de confusión", "Filas: clase real · Columnas: predicción · Intensidad: porcentaje dentro de la fila")
    for x, color, nombre in ((38, CORRECTO, "Acierto por clase"), (218, ERROR, "Confusión"),
                             (363, ABSTENCION, "Abstención"), (512, FALLO, "Fallo")):
        partes.extend([
            f'<rect x="{x}" y="88" width="13" height="13" rx="3" fill="{color}"/>',
            _texto(x+21, 100, nombre, tamano=12, color=SECUNDARIO),
        ])
    partes.append(_texto(38, 136, "Real ↓", tamano=12, color=SECUNDARIO, peso=700))
    for indice, columna in enumerate(columnas):
        x = izquierda + indice * celda + celda / 2
        partes.append(_texto(x, arriba-16, columna, tamano=11, color=SECUNDARIO, peso=700, ancla="middle"))
    partes.append(_texto(izquierda + celda*len(columnas)+17, arriba-16, "Acc.", tamano=11, color=SECUNDARIO, peso=700))
    for clase, fila in enumerate(matriz):
        total = sum(fila)
        y = arriba + clase * celda
        partes.extend([
            f'<circle cx="54" cy="{y+29}" r="13" fill="{COLORES_CLASE[clase % len(COLORES_CLASE)]}"/>',
            _texto(54, y+34, str(clase), tamano=12, color="#ffffff", peso=700, ancla="middle"),
            _texto(79, y+34, f"Clase {clase}", tamano=12, peso=700),
        ])
        for columna, cuenta in enumerate(fila):
            x = izquierda + columna * celda
            proporcion = cuenta / total if total else 0
            if cuenta == 0:
                partes.extend([
                    f'<rect x="{x+2}" y="{y+2}" width="{celda-4}" height="{celda-4}" rx="8" fill="#f1f5f9"/>',
                    _texto(x+celda/2, y+35, "0", tamano=12, color="#94a3b8", ancla="middle"),
                ])
                continue
            color = (COLORES_CLASE[clase % len(COLORES_CLASE)] if columna == clase else
                     ABSTENCION if columna == len(matriz) else
                     FALLO if columna == len(matriz)+1 else ERROR)
            opacidad = 0.12 + 0.88 * min(1.0, proporcion * 2.2)
            partes.extend([
                f'<rect x="{x+2}" y="{y+2}" width="{celda-4}" height="{celda-4}" rx="8" fill="{color}" fill-opacity="{opacidad:.3f}"/>',
                _texto(x+celda/2, y+29, str(cuenta), tamano=13, peso=700,
                       color="#ffffff" if opacidad > 0.75 else TEXTO, ancla="middle"),
                _texto(x+celda/2, y+45, f"{proporcion*100:.0f}%", tamano=10,
                       color="#ffffff" if opacidad > 0.75 else SECUNDARIO, ancla="middle"),
            ])
        acierto = fila[clase] / total if total else 0
        partes.append(_texto(izquierda + celda*len(columnas)+17, y+34, f"{acierto*100:.1f} %", tamano=12, peso=700))
    return _terminar(partes)


def latencias(runs: list[dict]) -> str:
    """Compara la mediana con las colas p95 y p99 de cada ejecución"""
    ancho, alto = 940, 175 + 60 * len(runs)
    partes = _base(ancho, alto, "Latencia de inferencia", "Tiempo de predict por muestra en milisegundos; percentiles calculados por ejecución")
    izquierda, anchura = 170, 600
    nombres = (("median", "p50", "#0891b2"), ("p95", "p95", "#f97316"),
               ("p99", "p99", "#db2777"))
    maximo = max((float(run.get("latency_s", {}).get("p99", 0)) * 1000 for run in runs), default=0)
    escala = max(maximo * 1.20, 0.1)
    for indice in range(5):
        x = izquierda + anchura * indice / 4
        partes.append(f'<line x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{alto-60}" stroke="{REJILLA}" stroke-dasharray="3 5"/>')
        partes.append(_texto(x, 95, f"{escala*indice/4:.2f}", tamano=11, color=SECUNDARIO, ancla="middle"))
    for indice, run in enumerate(runs):
        y = 143 + indice * 60
        lat = run.get("latency_s", {})
        partes.append(_texto(38, y+5, f"Ejecución {run['run']}", tamano=14, peso=700))
        if not lat:
            partes.append(_texto(izquierda, y+5, "Sin latencias registradas", tamano=12, color=SECUNDARIO))
            continue
        posiciones = []
        for campo, nombre, color in nombres:
            valor = float(lat[campo]) * 1000
            x = izquierda + anchura * valor / escala
            posiciones.append(x)
            partes.extend([
                f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>',
                _texto(x, y-14, nombre, tamano=11, color=color, peso=700, ancla="middle"),
            ])
        partes.append(f'<line x1="{min(posiciones):.1f}" y1="{y}" x2="{max(posiciones):.1f}" y2="{y}" stroke="#94a3b8" stroke-width="2"/>')
        # Los puntos se redibujan encima de la línea para mantener su color
        for (campo, _, color), x in zip(nombres, posiciones):
            partes.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        partes.append(_texto(792, y+5, f"p99 {float(lat['p99'])*1000:.3f} ms", tamano=12, peso=700))
    partes.append(_texto(38, alto-25, "p50 = mitad de las predicciones · p95 y p99 = colas lentas", tamano=12, color=SECUNDARIO))
    return _terminar(partes)


def entrenamiento(datos: dict) -> str:
    """Separa el tiempo de preparar características del ajuste del clasificador"""
    ancho, alto = 940, 220
    partes = _base(ancho, alto, "Tiempo de entrenamiento", "Coste local para preparar las señales y ajustar el clasificador")
    caracteristicas = max(0.0, float(datos["segundos_caracteristicas"]))
    ajuste = max(0.0, float(datos["segundos_ajuste"]))
    total = caracteristicas + ajuste
    x, y, anchura = 38, 112, 760
    primeras = anchura * caracteristicas / total if total else 0
    partes.extend([
        f'<rect x="{x}" y="{y}" width="{anchura}" height="31" rx="15" fill="#e2e8f0"/>',
        f'<rect x="{x}" y="{y}" width="{primeras:.1f}" height="31" fill="#06b6d4"/>',
        f'<rect x="{x+primeras:.1f}" y="{y}" width="{anchura-primeras:.1f}" height="31" fill="#7c3aed"/>',
        _texto(821, y+22, f"{total:.2f} s", tamano=16, peso=700),
        '<circle cx="44" cy="177" r="6" fill="#06b6d4"/>',
        _texto(60, 182, f"Características: {caracteristicas:.2f} s", tamano=13),
        '<circle cx="302" cy="177" r="6" fill="#7c3aed"/>',
        _texto(318, 182, f"Ajuste: {ajuste:.2f} s", tamano=13),
        _texto(590, 182, f"{datos['muestras_entrenamiento']:,} muestras".replace(",", "."), tamano=13, color=SECUNDARIO),
    ])
    return _terminar(partes)


def historial(registros: list[dict], actual: str) -> str:
    """Compara solo experimentos con el mismo protocolo oficial"""
    ancho, alto = 940, 150 + 54 * len(registros)
    partes = _base(ancho, alto, "Evolución de la puntuación", "Comparación válida: misma validación y configuración oficial")
    izquierda, anchura = 185, 590
    techo = max((r["agregado"]["ranking_score"] for r in registros), default=1) * 1.12 or 1
    for indice, registro in enumerate(registros):
        y = 98 + indice * 54
        identificador = registro["id"]
        score = float(registro["agregado"]["ranking_score"])
        color = "#7c3aed" if identificador == actual else "#06b6d4"
        partes.extend([
            _texto(38, y+21, identificador, tamano=14, peso=700 if identificador == actual else 400),
            f'<rect x="{izquierda}" y="{y}" width="{anchura}" height="29" rx="14" fill="#e5e7eb"/>',
            f'<rect x="{izquierda}" y="{y}" width="{anchura*score/techo:.1f}" height="29" rx="14" fill="{color}"/>',
            _texto(792, y+21, f"{score:.4f}", tamano=13, peso=700),
        ])
    partes.append(_texto(38, alto-24, "Morado: experimento actual · Cian: experimentos anteriores comparables", tamano=12, color=SECUNDARIO))
    return _terminar(partes)
