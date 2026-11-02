"""Menú local para configurar un experimento sin recordar todos los argumentos."""

from __future__ import annotations

import curses
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    ("id", "ID"),
    ("muestras", "Muestras de entrenamiento"),
    ("iteraciones", "Iteraciones"),
    ("aumento_ruido", "Copia con ruido y huecos (0=no, 1=sí)"),
    ("aumento_ritmo", "Variar ritmo de la copia (0=no, 1=sí)"),
    ("temporales", "Perfil temporal (0=no, 1=sí)"),
    ("extendidas", "Detalle estadístico y espectral (0=no, 1=sí)"),
    ("runs", "Ejecuciones de evaluación"),
    ("max_muestras", "Máximo de muestras puntuadas"),
    ("presupuesto_s", "Tiempo máximo para predecir (s)"),
)


def siguiente_id() -> str:
    """Busca el siguiente expNNN entre resultados, informes y pesos existentes."""
    numbers = []
    for folder in (ROOT / "experiments", ROOT / "results", ROOT / "models"):
        if not folder.is_dir():
            continue
        for entry in folder.iterdir():
            match = re.fullmatch(r"exp(\d+)(?:\.pkl)?", entry.name)
            if match:
                numbers.append(int(match.group(1)))
    return f"exp{max(numbers, default=0) + 1:03d}"


def comprobar(campo: str, valor: str) -> int | float | str:
    if campo == "id":
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", valor):
            raise ValueError("El ID debe usar minúsculas, números, - o _")
        for ruta in (
            ROOT / "experiments" / valor,
            ROOT / "results" / valor,
            ROOT / "models" / f"{valor}.pkl",
        ):
            if ruta.exists():
                raise ValueError(f"{valor} ya existe; elige otro ID")
        return valor

    try:
        numero = float(valor.replace(",", ".")) if campo == "presupuesto_s" else int(valor)
    except ValueError as error:
        raise ValueError("Introduce un número válido") from error
    if campo in ("aumento_ruido", "aumento_ritmo", "temporales", "extendidas"):
        if numero not in (0, 1):
            raise ValueError("Elige 0 (no) o 1 (sí)")
        return numero
    if campo == "presupuesto_s" and (not math.isfinite(numero) or numero <= 0):
        raise ValueError("El presupuesto debe ser mayor que cero")
    if campo == "muestras" and numero < 8:
        raise ValueError("Se necesitan al menos 8 muestras")
    if campo not in ("muestras", "presupuesto_s") and numero < 1:
        raise ValueError("El valor debe ser mayor que cero")
    return numero


def escribir(pantalla: curses.window, fila: int, texto: str, estilo: int = 0) -> None:
    alto, ancho = pantalla.getmaxyx()
    if fila < alto and ancho > 1:
        pantalla.addnstr(fila, 0, texto, ancho - 1, estilo)


def pedir_valor(pantalla: curses.window, nombre: str, actual: int | float | str) -> str | None:
    alto, ancho = pantalla.getmaxyx()
    fila = alto - 3
    pantalla.move(fila, 0)
    pantalla.clrtoeol()
    pantalla.move(fila + 1, 0)
    pantalla.clrtoeol()
    escribir(pantalla, fila, f"{nombre} (actual: {actual}; vacío = mantener):")
    pantalla.refresh()
    curses.echo()
    try:
        curses.curs_set(1)
        bruto = pantalla.getstr(fila + 1, 0, max(1, min(40, ancho - 1)))
    finally:
        curses.noecho()
        curses.curs_set(0)
    valor = bruto.decode("utf-8").strip()
    return None if not valor or valor == "\x1b" else valor


def pantalla_menu(pantalla: curses.window) -> dict[str, int | float | str] | None:
    valores: dict[str, int | float | str] = {
        "id": siguiente_id(),
        "muestras": 90_000,
        "iteraciones": 70,
        "aumento_ruido": 0,
        "aumento_ritmo": 0,
        "temporales": 0,
        "extendidas": 0,
        "runs": 1,
        "max_muestras": 1_000,
        "presupuesto_s": 2.0,
    }
    elegido = 0
    error = ""
    curses.curs_set(0)

    while True:
        pantalla.erase()
        alto, ancho = pantalla.getmaxyx()
        if alto < 20 or ancho < 48:
            escribir(pantalla, 0, "Amplía la terminal (mínimo 48 × 20) o pulsa Esc")
        else:
            escribir(pantalla, 0, "Nuevo experimento (↑/↓, Enter para editar; Esc para cancelar):")
            escribir(pantalla, 2, "Propuesta inicial: 90k muestras y 70 iteraciones")
            for indice, (clave, nombre) in enumerate(FIELDS):
                estilo = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else curses.A_REVERSE) if indice == elegido else 0
                prefijo = "❯" if indice == elegido else " "
                escribir(pantalla, indice + 4, f"{prefijo} {nombre}: {valores[clave]}", estilo)
            estilo = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else curses.A_REVERSE) if elegido == len(FIELDS) else 0
            escribir(pantalla, len(FIELDS) + 4, f"{'❯' if elegido == len(FIELDS) else ' '} Iniciar entrenamiento y evaluación", estilo)
            escribir(pantalla, alto - 4, error or "Los pesos y resultados anteriores no se sobrescriben")
        pantalla.refresh()
        tecla = pantalla.getch()
        if tecla in (27, 3):
            return None
        if tecla == curses.KEY_UP:
            elegido = (elegido - 1) % (len(FIELDS) + 1)
        elif tecla == curses.KEY_DOWN:
            elegido = (elegido + 1) % (len(FIELDS) + 1)
        elif tecla in (10, 13, curses.KEY_ENTER) and alto >= 20 and ancho >= 48:
            if elegido == len(FIELDS):
                try:
                    for clave, _ in FIELDS:
                        comprobar(clave, str(valores[clave]))
                    if valores["aumento_ritmo"] and not valores["aumento_ruido"]:
                        raise ValueError("Variar ritmo requiere activar la copia con ruido")
                except ValueError as fallo:
                    error = str(fallo)
                else:
                    return valores
            else:
                clave, nombre = FIELDS[elegido]
                nuevo = pedir_valor(pantalla, nombre, valores[clave])
                if nuevo is not None:
                    try:
                        valores[clave] = comprobar(clave, nuevo)
                        error = ""
                    except ValueError as fallo:
                        error = str(fallo)


def menu_entrenamiento() -> dict[str, int | float | str] | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("El menú necesita una terminal interactiva; usa --id y los demás argumentos")

    def abrir(pantalla: curses.window) -> dict[str, int | float | str] | None:
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        return pantalla_menu(pantalla)

    try:
        elegido = curses.wrapper(abrir)
    except KeyboardInterrupt:
        return None
    if elegido is not None:
        print(f"Elegido: {elegido['id']} · {elegido['muestras']} muestras · {elegido['iteraciones']} iteraciones")
    return elegido
