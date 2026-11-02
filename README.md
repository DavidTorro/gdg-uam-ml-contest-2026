# GDG UAM ML Contest 2026

Proyecto de David Torró para el 1.º Machine Learning Contest de GDG on Campus UAM.

- Equipo: DavidTorroTeam
- Modalidad: individual
- Objetivo: clasificar series temporales de 4 canales y 256 valores en 8 clases

## Organización

Este repositorio contiene el modelo que se entregará: `model.py`, los pesos en `models/`, el código de características en `src/` y `requirements.txt`. El panel hermano `../ml-dashboard/` sirve para leer los resultados; no forma parte de la entrega.

Los datos descargados, las particiones y los resultados del evaluador permanecen fuera de Git. Cada experimento usa un `id` nuevo y conserva sus pesos y JSON por separado. Para comparar modelos, usa el mismo archivo de evaluación y los mismos límites; después importa `results/<id>/model` y `experiments/<id>` en el dashboard.

La partición fija usa la semilla `2026`: `data/train_90k.npz` para entrenar y `data/val_10k.npz` para validar. El script `scripts/ejecutar_experimento.py` entrena, ejecuta el evaluador oficial y guarda los JSON necesarios para el dashboard. Sin argumentos abre un menú: ↑/↓ para elegir un campo, Enter para editar o iniciar y Esc para cancelar. Propone automáticamente el siguiente `expNNN` libre y los valores 90 000 muestras y 70 iteraciones; puedes cambiarlos antes de arrancar:

```bash
.venv/bin/python scripts/ejecutar_experimento.py
```

También admite los argumentos de siempre; sustituye `exp_nuevo` por un ID libre. `--presupuesto-s` limita el tiempo de predicción de cada evaluación, no el entrenamiento. Mantén los mismos límites si quieres comparar puntuaciones dentro del protocolo A:

```bash
.venv/bin/python scripts/ejecutar_experimento.py --id exp_nuevo --muestras 90000 --iteraciones 70 --runs 1 --max-muestras 1000 --presupuesto-s 2
```

La opción `--aumento-ruido` añade al entrenamiento una copia de cada muestra con ruido moderado y algunos huecos extra. Sirve para comprobar si el modelo aguanta mejor los cambios que observamos en `test_demo`. No modifica la extracción de características durante la predicción ni los pesos de experimentos anteriores. El menú permite activarla con `1`.

La opción `--temporales` añade 32 características: la media de ocho tramos de cada canal. Conserva parte del orden temporal de la señal y permite probar si ayuda a distinguir clases que las características globales confunden. Los pesos anteriores siguen usando las 59 características originales. No presupone que vaya a mejorar la puntuación: hay que medirlo.

La opción `--extendidas` añade 68 características: cuatro cuantiles, cinco autocorrelaciones y ocho bandas espectrales por canal. Con `--temporales` el vector tiene 159 valores. Se inspira en el mayor detalle del baseline oficial, pero mantiene las características anteriores y las 70 iteraciones para medir el coste y el efecto de este cambio.

La opción `--aumento-ritmo` aplica a la copia con ruido una variación temporal moderada (±15 %) igual en los cuatro canales. Requiere `--aumento-ruido` y permite comprobar si esa variación ayuda con el cambio entre entrenamiento y demo. Solo afecta al entrenamiento; la predicción y los pesos anteriores no cambian.

## Siguiente prueba: red convolucional

`exp013` sigue siendo la referencia mientras se prueba una arquitectura que recibe las cuatro series completas. La CNN 1D es un experimento independiente: carga pesos `.pt`, usa la misma entrada `model.py` y pasa por el mismo evaluador. Separa un 10 % de las muestras seleccionadas de `train_90k` para elegir la mejor época; no utiliza `val_10k` ni `test_demo` durante el entrenamiento. Por ello, con 90 000 muestras seleccionadas entrena directamente con 81 000 y controla con 9 000.

La red requiere PyTorch. Desde la raíz de este repositorio, ejecuta:

```bash
.venv/bin/python -m pip install -r requirements-cnn.txt
.venv/bin/python scripts/ejecutar_experimento.py --id exp015 --arquitectura cnn --muestras 90000 --epocas 12 --batch-size 256 --runs 1 --max-muestras 1000 --presupuesto-s 2
```

El entrenamiento puede usar MPS en Mac; el evaluador siempre carga la red en CPU y usa un solo hilo. La cifra de épocas aparece en el dashboard en el campo de iteraciones. La primera prueba solo decide si esta arquitectura merece más trabajo: compara la puntuación, la matriz de confusión y la latencia con `exp013` bajo el mismo protocolo. Si supera la validación, evalúala después en `test_demo` una sola vez; no uses esas 400 etiquetas para entrenar ni para ajustar repetidamente parámetros. Para importarla al panel, desde `../ml-dashboard/` ejecuta `node scripts/importar_experimento.mjs` y selecciona `exp015`.

Si se elige la CNN para la entrega, hay que añadir PyTorch a `requirements.txt` y cambiar el peso predeterminado de `model.py`. Hasta entonces, `exp013` y los demás pesos permanecen intactos.

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
| exp011 · 90k originales + 90k copias · 70 it | 4,54386 | 1000 | 80,0 % | 1,53 ms | 0,11737 puntos · 37,75 % de acierto |
| exp012 · 90k originales + 90k copias · 70 it · temporal | 4,52434 | 1000 | 80,3 % | 1,55 ms | 0,15668 puntos · 39,5 % de acierto |
| exp013 · 90k originales + 90k copias · 70 it · extendido | 5,07698 | 1000 | 82,1 % | 1,72 ms | **0,25565 puntos · 44,25 % de acierto** |
| exp014 · como exp013 + variación de ritmo | **5,39955** | 1000 | **83,0 %** | 1,70 ms | 0,18867 puntos · 42,25 % de acierto |

Las puntuaciones locales se midieron con una ejecución, hasta 1000 muestras y 2 s de predicción. `exp009` tiene la mayor puntuación local, pero supera a `exp004` por solo 0,00577 puntos: una prueba no basta para declarar un ganador sólido. Con 90k muestras, subir de 40 a 70 iteraciones mejoró el acierto y la puntuación; `exp009` también predijo más rápido que `exp004` (1,46 frente a 1,73 ms). Bajar a 60k muestras con 80 iteraciones (`exp010`) rindió peor que usar 90k (`exp004`).

`exp005` usa 150 iteraciones, pero tarda casi el doble por muestra que `exp004` y llega solo a 580; su porcentaje de acierto se calcula sobre menos muestras, por lo que no se puede comparar directamente con el de `exp004`. La prueba local corta tampoco muestra toda la ventaja de un modelo más rápido cuando ambos llegan al límite de 1000 muestras. En la comparación de 5 s y tres ejecuciones, `exp013` obtuvo **8,27503** puntos frente a 7,79577 de `exp009`: respondió menos muestras (3197 frente a 3494), pero acertó más (80,55 % frente a 78,86 %).

Las puntuaciones de validación y `test_demo` usan conjuntos y cantidades de muestras distintas y no se comparan entre sí. En `test_demo`, `exp004` acertó 34,75 %, `exp009` 34,5 %, `exp011` 37,75 %, `exp012` 39,5 %, `exp013` **44,25 %** y `exp014` 42,25 % sobre las mismas 400 muestras. La variación de ritmo de `exp014` mejoró la validación corta, pero perdió ocho aciertos frente a `exp013` en demo y tampoco reconoció ninguna muestra de la clase 6; por eso `exp013` sigue siendo el candidato principal. El baseline oficial, entrenado también con `train_90k.npz`, acertó 42 %, pero tardó 6,28 ms por muestra frente a los 1,71 ms de `exp013`. Tanto el baseline como `exp013` fallaron las 50 muestras de la clase 6; en `exp013`, 32 se confundieron con la clase 7. La auditoría encontró más valores ausentes y más energía de alta frecuencia en demo; esto orienta las hipótesis, pero no demuestra la causa de los errores. Repetir ajustes sobre las mismas 400 muestras puede sobreajustar al demo.
