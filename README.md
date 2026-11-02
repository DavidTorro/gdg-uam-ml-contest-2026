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

En el modelo de características, `--aumento-ruido` añade al entrenamiento una copia de cada muestra con ruido moderado y algunos huecos extra. En la CNN modifica los lotes durante el entrenamiento sin duplicar muestras. No cambia la predicción ni los pesos de experimentos anteriores. El menú interactivo se usa para el modelo de características; para la CNN hay que pasar `--arquitectura cnn` por argumentos.

La opción `--temporales` añade 32 características: la media de ocho tramos de cada canal. Conserva parte del orden temporal de la señal y permite probar si ayuda a distinguir clases que las características globales confunden. Los pesos anteriores siguen usando las 59 características originales. No presupone que vaya a mejorar la puntuación: hay que medirlo.

La opción `--extendidas` añade 68 características: cuatro cuantiles, cinco autocorrelaciones y ocho bandas espectrales por canal. Con `--temporales` el vector tiene 159 valores. Se inspira en el mayor detalle del baseline oficial, pero mantiene las características anteriores y las 70 iteraciones para medir el coste y el efecto de este cambio.

La opción `--aumento-ritmo` aplica a la copia con ruido una variación temporal moderada (±15 %) igual en los cuatro canales. Requiere `--aumento-ruido` y permite comprobar si esa variación ayuda con el cambio entre entrenamiento y demo. Solo afecta al entrenamiento; la predicción y los pesos anteriores no cambian.

## Red convolucional

Después de los primeros modelos de características se probó una arquitectura que recibe las cuatro series completas. La CNN 1D carga pesos `.pt`, usa la misma entrada `model.py` y pasa por el mismo evaluador. Separa un 10 % de las muestras seleccionadas de `train_90k` para elegir la mejor época; no utiliza `val_10k` ni `test_demo` durante el entrenamiento. Por ello, con 90 000 muestras seleccionadas entrena directamente con 81 000 y controla con 9 000.

La red requiere PyTorch. Desde la raíz de este repositorio, ejecuta:

```bash
.venv/bin/python -m pip install -r requirements-cnn.txt
.venv/bin/python scripts/ejecutar_experimento.py --id exp015 --arquitectura cnn --muestras 90000 --epocas 12 --batch-size 256 --runs 1 --max-muestras 1000 --presupuesto-s 2
```

El entrenamiento puede usar MPS en Mac; el evaluador carga la red en CPU y usa un solo hilo. La cifra de épocas aparece en el dashboard en el campo de iteraciones. Para importar un resultado al panel, desde `../ml-dashboard/` ejecuta `node scripts/importar_experimento.mjs` y selecciona el ID.

Antes de entregar la CNN hay que añadir PyTorch a `requirements.txt` y cambiar el peso predeterminado de `model.py`. Los pesos anteriores permanecen disponibles para comparación.

### Resultados de la CNN

Con el mismo protocolo local de 1000 muestras y 2 s de predicción, `exp015` (12 épocas) obtuvo 7,19975 puntos y 89,3 % de acierto; `exp016` (24 épocas) obtuvo 7,99723 y 91,6 %; `exp017` (60 épocas, mejor peso en la época 31) obtuvo 8,45425 y 92,8 %. En las 400 muestras de `test_demo`, los aciertos fueron 69,25 %, **74,25 %** y 71,25 %, respectivamente. La puntuación del demo fue 1,43556, **1,79811** y 1,62195. Por tanto, `exp016` es el candidato actual: más épocas mejoraron la validación local de `exp017`, pero empeoraron el demo.

La clase 6 sigue siendo el mayor fallo fuera de distribución: `exp016` acertó 4 de sus 50 muestras en el demo y confundió 35 con la clase 7. La auditoría de datos también observó más huecos y energía de alta frecuencia en el demo. `exp018` añadió ruido de distinta intensidad por canal y aproximadamente un 0,8 % de huecos interpolados en cada lote de entrenamiento. No duplicó las muestras, no tocó el control interno ni cambió la predicción. Con las mismas 24 épocas que `exp016`, bajó a 7,72282 puntos y 90,9 % en validación local. En el demo acertó 74,0 % y obtuvo 1,74010 puntos; la clase 6 subió ligeramente a 6/50, pero la puntuación total quedó por debajo de `exp016`.

`exp019` y `exp020` probaron 36 épocas sin y con ese aumento de ruido. `exp020` es el mejor **en validación local**: 8,65878 puntos y 93,5 % de acierto, frente a 8,35708 y 92,7 % de `exp019`. La clase 6 local fue mejor en `exp019` (90/96) que en `exp020` (86/96). Ninguno de los dos tiene aún prueba en `test_demo`; por tanto, `exp016` sigue siendo el mejor **medido en el demo**, no hay un ganador final entre los 20 experimentos.

El demo tiene solo 400 muestras etiquetadas. Se ha consultado varias veces para comparar modelos; seguir ajustando parámetros contra esos mismos resultados puede sobreajustar la selección al demo. Antes de otro entrenamiento, conviene revisar los 20 experimentos, el protocolo oficial y la diferencia entre validación local y demo.

## Estado de los modelos

`model.py` todavía carga `models/exp001.pkl` por defecto como base provisional. La variable `GDG_MODEL_PATH` permite evaluar otro peso local sin cambiar el código. Antes de la entrega hay que elegir el peso final, completar `requirements.txt` y comprobar el repositorio desde un clon limpio, tal como pide el concurso.

| Modelo | Puntuación local | Muestras puntuadas | Acierto | Latencia media | `test_demo` (400 muestras) |
| --- | ---: | ---: | ---: | ---: | --- |
| exp001 · 5k · 80 it | 3,39156 | 1000 | 74,5 % | 1,57 ms | 0,09696 puntos · 37,25 % de acierto |
| exp002 · 30k · 150 it | 3,37993 | 673 | 78,6 % | 2,97 ms | 0,06791 puntos · 34,5 % de acierto |
| exp003 · 30k · 80 it | 3,71389 | 1000 | 77,8 % | 1,62 ms | — |
| exp004 · 90k · 80 it | 4,91809 | 1000 | 81,5 % | 1,73 ms | — |
| exp005 · 90k · 150 it | 3,68637 | 580 | 81,55 % | 3,44 ms | — |
| exp006 · 90k · 40 it | 4,26528 | 1000 | 78,7 % | 0,88 ms | — |
| exp007 · 90k · 50 it | 4,40807 | 1000 | 79,4 % | 1,09 ms | — |
| exp008 · 90k · 60 it | 4,59885 | 1000 | 80,2 % | 1,26 ms | — |
| exp009 · 90k · 70 it | 4,92386 | 1000 | 81,4 % | 1,46 ms | — |
| exp010 · 60k · 80 it | 4,50889 | 1000 | 79,9 % | 1,67 ms | — |
| exp011 · 90k originales + 90k copias · 70 it | 4,54386 | 1000 | 80,0 % | 1,53 ms | 0,11737 puntos · 37,75 % de acierto |
| exp012 · 90k originales + 90k copias · 70 it · temporal | 4,52434 | 1000 | 80,3 % | 1,55 ms | 0,15668 puntos · 39,5 % de acierto |
| exp013 · 90k originales + 90k copias · 70 it · extendido | 5,07698 | 1000 | 82,1 % | 1,72 ms | 0,25565 puntos · 44,25 % de acierto |
| exp014 · como exp013 + variación de ritmo | 5,39955 | 1000 | 83,0 % | 1,70 ms | 0,18867 puntos · 42,25 % de acierto |
| exp015 · CNN · 12 épocas | 7,19975 | 1000 | 89,3 % | 0,132 ms | 1,43556 puntos · 69,25 % de acierto |
| exp016 · CNN · 24 épocas | 7,99723 | 1000 | 91,6 % | 0,128 ms | **1,79811 puntos · 74,25 % de acierto** |
| exp017 · CNN · 60 épocas (mejor: 31) | 8,45425 | 1000 | 92,8 % | 0,142 ms | 1,62195 puntos · 71,25 % de acierto |
| exp018 · CNN · 24 épocas · ruido | 7,72282 | 1000 | 90,9 % | 0,129 ms | 1,74010 puntos · 74,0 % de acierto |
| exp019 · CNN · 36 épocas | 8,35708 | 1000 | 92,7 % | 0,131 ms | — |
| exp020 · CNN · 36 épocas · ruido | **8,65878** | 1000 | **93,5 %** | 0,129 ms | — |

Las puntuaciones locales se midieron con una ejecución, hasta 1000 muestras y 2 s de predicción. Entre los primeros modelos de características, `exp009` superó a `exp004` por solo 0,00577 puntos: una prueba no bastaba para declarar un ganador sólido. Con 90k muestras, subir de 40 a 70 iteraciones mejoró el acierto y la puntuación; `exp009` también predijo más rápido que `exp004` (1,46 frente a 1,73 ms). Bajar a 60k muestras con 80 iteraciones (`exp010`) rindió peor que usar 90k (`exp004`).

`exp005` usa 150 iteraciones, pero tarda casi el doble por muestra que `exp004` y llega solo a 580; su porcentaje de acierto se calcula sobre menos muestras, por lo que no se puede comparar directamente con el de `exp004`. La prueba local corta tampoco muestra toda la ventaja de un modelo más rápido cuando ambos llegan al límite de 1000 muestras. En la comparación de 5 s y tres ejecuciones, `exp013` obtuvo **8,27503** puntos frente a 7,79577 de `exp009`: respondió menos muestras (3197 frente a 3494), pero acertó más (80,55 % frente a 78,86 %).

Las puntuaciones de validación y `test_demo` usan conjuntos y cantidades de muestras distintas y no se comparan entre sí. Entre los modelos de características, `exp013` fue el mejor medido en el demo: 44,25 % frente a 42,25 % de `exp014`; ambos quedaron por debajo de las CNN. El baseline oficial, entrenado también con `train_90k.npz`, acertó 42 % en el demo y tardó 6,28 ms por muestra. Tanto el baseline como `exp013` fallaron las 50 muestras de la clase 6; en `exp013`, 32 se confundieron con la clase 7. La auditoría encontró más valores ausentes y más energía de alta frecuencia en demo; esto orienta las hipótesis, pero no demuestra la causa de los errores.
