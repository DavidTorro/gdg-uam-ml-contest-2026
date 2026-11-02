# GDG UAM ML Contest 2026

Proyecto de David Torró para el 1.º Machine Learning Contest de GDG on Campus UAM.

- Equipo: DavidTorroTeam
- Modalidad: individual
- Objetivo: clasificar series temporales de 4 canales y 256 valores en 8 clases

## Organización

Este repositorio contiene el modelo que se entregará: `model.py`, los pesos en `models/`, el código de características en `src/` y `requirements.txt`. El panel hermano `../ml-dashboard/` sirve para leer los resultados; no forma parte de la entrega.

Los datos descargados, las particiones, los resultados del evaluador y los informes locales permanecen fuera de Git. Cada experimento usa un `id` nuevo y conserva sus pesos y JSON por separado. Para comparar modelos, usa el mismo archivo de evaluación y los mismos límites; después importa `results/<id>/model` y `experiments/<id>` en el dashboard.

## Estado de los modelos

`model.py` carga `models/exp001.pkl` por defecto como base provisional. La variable `GDG_MODEL_PATH` permite evaluar otro peso local sin cambiar el código. Antes de la entrega elegiremos el modelo final y comprobaremos el repositorio desde un clon limpio, tal como pide el concurso.

| Modelo | Validación local (1000 muestras, 2 s, 1 ejecución) | `test_demo` (400 muestras) |
| --- | --- | --- |
| exp001 | 3,39156 puntos · 74,5 % de acierto | 0,09696 puntos · 37,25 % de acierto |
| exp002 | 3,37993 puntos · 78,6 % de acierto | 0,06791 puntos · 34,5 % de acierto |

Las puntuaciones de ambas columnas usan cantidades de muestras distintas y no se comparan entre sí. En `test_demo`, los dos modelos aciertan bastante menos que en la validación local; esto orienta la siguiente fase de trabajo hacia la generalización a datos nuevos. La clase 6 cae al 0 % con exp001 y al 2 % con exp002 (50 muestras por clase); las clases 2 y 3 también empeoran mucho. Los resultados de `test_demo` se calcularon con el evaluador oficial en tres ejecuciones sobre el mismo conjunto de 400 muestras.
