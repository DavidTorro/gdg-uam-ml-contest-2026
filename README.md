# GDG UAM ML Contest 2026

Proyecto de David Torró para el 1.º Machine Learning Contest de GDG on Campus UAM.

- Equipo: DavidTorroTeam
- Modalidad: individual
- Objetivo: clasificar series temporales de 4 canales y 256 valores en 8 clases

## Organización

Este repositorio contiene el modelo que se entregará: `model.py`, los pesos en `models/`, el código de características en `src/` y `requirements.txt`. El panel hermano `../ml-dashboard/` sirve para leer los resultados; no forma parte de la entrega.

Los datos descargados, las particiones y los resultados del evaluador permanecen fuera de Git. Cada experimento usa un `id` nuevo y conserva sus pesos y JSON por separado. Para comparar modelos, usa el mismo archivo de evaluación y los mismos límites; después importa `results/<id>/model` y `experiments/<id>` en el dashboard.

La partición fija usa la semilla `2026`: `data/train_90k.npz` para entrenar y `data/val_10k.npz` para validar. El script `scripts/ejecutar_experimento.py` entrena, ejecuta el evaluador oficial y guarda los JSON necesarios para el dashboard. Sin argumentos abre un menú: ↑/↓ para elegir un campo, Enter para editar o iniciar y Esc para cancelar. Propone automáticamente el siguiente `expNNN` libre y los valores 90 000 muestras y 40 iteraciones; puedes cambiarlos antes de arrancar:

```bash
.venv/bin/python scripts/ejecutar_experimento.py
```

También admite los argumentos de siempre; sustituye `exp_nuevo` por un ID libre. `--presupuesto-s` limita el tiempo de predicción de cada evaluación, no el entrenamiento. Mantén los mismos límites si quieres comparar puntuaciones dentro del protocolo A:

```bash
.venv/bin/python scripts/ejecutar_experimento.py --id exp_nuevo --muestras 90000 --iteraciones 70 --runs 1 --max-muestras 1000 --presupuesto-s 2
```

## Estado de los modelos

`model.py` carga `models/exp001.pkl` por defecto como base provisional. La variable `GDG_MODEL_PATH` permite evaluar otro peso local sin cambiar el código. Antes de la entrega elegiremos el modelo final y comprobaremos el repositorio desde un clon limpio, tal como pide el concurso.

| Modelo | Puntuación local | Muestras puntuadas | Acierto | Latencia media | `test_demo` (400 muestras) |
| --- | ---: | ---: | ---: | ---: | --- |
| exp001 · 5k · 80 it | 3,39156 | 1000 | 74,5 % | 1,57 ms | 0,09696 puntos · 37,25 % de acierto |
| exp002 · 30k · 150 it | 3,37993 | 673 | 78,6 % | 2,97 ms | 0,06791 puntos · 34,5 % de acierto |
| exp003 · 30k · 80 it | 3,71389 | 1000 | 77,8 % | 1,62 ms | — |
| exp004 · 90k · 80 it | **4,91809** | 1000 | 81,5 % | 1,73 ms | — |
| exp005 · 90k · 150 it | 3,68637 | 580 | 81,55 % | 3,44 ms | — |
| exp006 · 90k · 40 it | 4,26528 | 1000 | 78,7 % | 0,88 ms | — |
| exp007 · 90k · 50 it | 4,40807 | 1000 | 79,4 % | 1,09 ms | — |
| exp008 · 90k · 60 it | 4,59885 | 1000 | 80,2 % | 1,26 ms | — |
| exp009 · 90k · 70 it | **4,92386** | 1000 | 81,4 % | 1,46 ms | — |
| exp010 · 60k · 80 it | 4,50889 | 1000 | 79,9 % | 1,67 ms | — |

Las puntuaciones locales se midieron con una ejecución, hasta 1000 muestras y 2 s de predicción. `exp009` tiene la mayor puntuación local, pero supera a `exp004` por solo 0,00577 puntos: una prueba no basta para declarar un ganador sólido. Con 90k muestras, subir de 40 a 70 iteraciones mejoró el acierto y la puntuación; `exp009` también predijo más rápido que `exp004` (1,46 frente a 1,73 ms). Bajar a 60k muestras con 80 iteraciones (`exp010`) rindió peor que usar 90k (`exp004`).

`exp005` usa 150 iteraciones, pero tarda casi el doble por muestra que `exp004` y llega solo a 580; su porcentaje de acierto se calcula sobre menos muestras, por lo que no se puede comparar directamente con el de `exp004`. La prueba local corta tampoco muestra toda la ventaja de un modelo más rápido cuando ambos llegan al límite de 1000 muestras. El siguiente paso es comparar los mejores candidatos con más muestras puntuadas y varias ejecuciones, además de medir su generalización en `test_demo`.

Las puntuaciones de validación y `test_demo` usan conjuntos y cantidades de muestras distintas y no se comparan entre sí. En `test_demo`, exp001 y exp002 aciertan bastante menos que en la validación local; esto orienta la siguiente fase hacia la generalización a datos nuevos. La clase 6 cae al 0 % con exp001 y al 2 % con exp002 (50 muestras por clase); las clases 2 y 3 también empeoran mucho. Los resultados de `test_demo` se calcularon con el evaluador oficial en tres ejecuciones sobre el mismo conjunto de 400 muestras.
