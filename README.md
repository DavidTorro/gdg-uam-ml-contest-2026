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

PyTorch solo hace falta para entrenar y para los pesos `.pt`. La entrega usa la misma red exportada a ONNX (ver [Entrega actual](#entrega-actual)); `requirements.txt` incluye ONNX Runtime 1.21.1, Numba 0.68.1 y NumPy: 2.2.6 para Python 3.10–3.11 y 2.5.4 para Python 3.12 o posterior. Los modelos antiguos `.pkl` requieren instalar scikit-learn por separado; no forman parte de la predicción por defecto.

### Resultados de la CNN

Con el mismo protocolo local de 1000 muestras y 2 s de predicción, `exp015` (12 épocas) obtuvo 7,19975 puntos y 89,3 % de acierto; `exp016` (24 épocas) obtuvo 7,99723 y 91,6 %; `exp017` (60 épocas, mejor peso en la época 31) obtuvo 8,45425 y 92,8 %. En las 400 muestras de `test_demo`, los aciertos fueron 69,25 %, **74,25 %** y 71,25 %, respectivamente. La puntuación del demo fue 1,43556, **1,79811** y 1,62195. Por tanto, `exp016` es el candidato actual: más épocas mejoraron la validación local de `exp017`, pero empeoraron el demo.

La clase 6 sigue siendo el mayor fallo fuera de distribución: `exp016` acertó 4 de sus 50 muestras en el demo y confundió 35 con la clase 7. La auditoría de datos también observó más huecos y energía de alta frecuencia en el demo. `exp018` añadió ruido de distinta intensidad por canal y aproximadamente un 0,8 % de huecos interpolados en cada lote de entrenamiento. No duplicó las muestras, no tocó el control interno ni cambió la predicción. Con las mismas 24 épocas que `exp016`, bajó a 7,72282 puntos y 90,9 % en validación local. En el demo acertó 74,0 % y obtuvo 1,74010 puntos; la clase 6 subió ligeramente a 6/50, pero la puntuación total quedó por debajo de `exp016`.

`exp019` y `exp020` probaron 36 épocas sin y con ese aumento de ruido. `exp020` es el mejor **en validación local**: 8,65878 puntos y 93,5 % de acierto, frente a 8,35708 y 92,7 % de `exp019`. La clase 6 local fue mejor en `exp019` (90/96) que en `exp020` (86/96). Ninguno de los dos tiene aún prueba en `test_demo`; por tanto, `exp016` sigue siendo el mejor **medido en el demo**, no hay un ganador final entre los 20 experimentos.

El demo tiene solo 400 muestras etiquetadas. Se ha consultado varias veces para comparar modelos; seguir ajustando parámetros contra esos mismos resultados puede sobreajustar la selección al demo. Antes de otro entrenamiento, conviene revisar los 20 experimentos, el protocolo oficial y la diferencia entre validación local y demo.

## Simulador de examen (exp021–exp039)

El fichero de train tiene trampas que el examen no tiene: `level` delata la clase (en el test no), los canales 2 y 3 de las clases 3 y 6 cambian de forma, y el demo trae el doble de huecos, más ruido rápido, picos y amplitudes más variables. Por eso la validación de `val_10k` premiaba modelos que luego fallaban en el demo.

- `src/simulador.py` transforma cada lote de entrenamiento para que se parezca al examen: ruido, picos, huecos, ganancias y un `level` nuevo independiente de la clase. La versión 2 aplica las «mezclas» de las trampas a **todas** las clases, para que la red no pueda usarlas como pista. `scripts/calibrar_simulador.py` comprueba con 55 rasgos que el resultado se parece al demo y genera `data/val_examen_v2.npz`, la validación con la que se elige la época.
- `--sin-atajos` anula en la red las medias y el `level`; `--equilibrar` hace que todas las clases pesen igual; `--elegir-epoca examen` entrena con las 90k y guarda la época con mejor nota oficial en `val_examen_v2`.
- `scripts/calibrar_sesgo.py` calcula un «empujón» para la clase 6: un valor que se suma a su salida hasta que la red la elige en 1/8 del demo. No usa las etiquetas. `scripts/ajustar_con_demo.py` (reajustar la capa final con el demo, `exp021`) se descartó porque empeoraba en validación cruzada.
- `scripts/barrido.py` entrena varias configuraciones seguidas, las evalúa en `val_examen_v2`, calcula la nota estimada a 60 s (la nota oficial crece con la raíz de las muestras respondidas, así que la velocidad cuenta) y las importa en el dashboard.

| Modelo | `val_examen_v2` | Nota 60 s estimada | `test_demo` | Con empujón |
| --- | ---: | ---: | ---: | ---: |
| exp023 · ancho 1 · 100 épocas · lr 0,001 | 21,25 | 139,8 | 2,473 | 2,880 |
| exp027 · ancho 1 · 200 épocas | 22,42 | 150,3 | 2,984 | 3,062 |
| exp032 · ancho 3 · lr 0,002 | 22,82 | 136,3 | 2,931 | 3,404 |
| **exp033 · ancho 1 · 200 épocas · lr 0,002** | 23,03 | 155,0 | 3,586 | **3,779 (84,5 %)** |
| exp034 · como exp033 · semilla 2027 | 23,15 | 158,6 | 3,266 | 3,405 |
| exp036–exp039 · 300/400 épocas, lr 0,003, semilla 2028 | 22,83–23,19 | 155–157 | — | — |
| exp040 · exp033 + ritmo variable en canal 0 | 22,63 | — | — | 3,520 (83,25 %) |

Las redes más anchas (ancho 2–4, `exp024`–`exp026`, `exp028`–`exp032`, `exp035`) aciertan parecido pero son más lentas, y pierden en la nota a 60 s. Desde `exp033` la receta está estancada: más épocas, otra tasa de aprendizaje u otra semilla dan lo mismo dentro del ruido (~±0,3). La diferencia de demo entre `exp033` y `exp034` (misma receta) muestra que parte de la ventaja de `exp033` es suerte. El error principal que queda es la confusión entre las clases 6 y 7 (15 fallos en cada sentido en el demo).

`exp040` añadió variación del ritmo del canal 0 durante el entrenamiento. Mejoró la prueba de estrés creada para ese cambio, pero bajó a 3,520 en el demo: la clase 6 subió de 64 % a 66 % y la 7 cayó de 64 % a 56 %. Se conserva `exp033`. La memoria de trabajo y los JSON de evaluación permanecen fuera de este repositorio.

## Entrega actual

`model.py` carga por defecto `models/exp033.sesgo.onnx`: la CNN `exp033` con el empujón de la clase 6 (0,9414), exportada a ONNX con `scripts/exportar_onnx.py`. `src/red_onnx.py` prepara las entradas con Numba JIT y ejecuta la red con ONNX Runtime en CPU y un hilo. La compilación ocurre en setup, fuera del tiempo de predicción. La variante NumPy permanece en `models/exp033.sesgo.npz` y puede seleccionarse con `GDG_MODEL_PATH`; `GDG_PREP_JIT=0` conserva ONNX con la preparación NumPy anterior.

**Estado al 10 de noviembre:** se mantiene `exp033.sesgo.onnx` como modelo de entrega. La puerta de calidad (`plans/gate_calidad_01.md`) y la destilación (`plans/destilacion_01.md`) fallaron sus filtros prefijados. Los afinamientos sin destilación (`plans/afinamiento_02.md`) mejoraron la auditoría simulada, pero cedieron aciertos de clase 6 y ninguno superó a `exp033` en el demo de Colab. La mezcla de pesos (`plans/sopa_01.md`), la retención de márgenes 6/7 (`plans/retencion_67_01.md`) y la red compacta (`plans/compacta_01.md`) fallaron sus límites de clase 6: −23, −20 y −50 aciertos. La consulta selectiva de exp034 (`plans/selectivo_01.md`) conservó bien las clases 6/7, pero mejoró el score simulado +0,401, por debajo del mínimo prefijado +0,5. Esas auditorías y el demo ya se han consultado repetidamente; no se retocan parámetros con ellos ni se cambia la entrega por esos candidatos. La fusión de media JIT (`plans/jit_media_01.md`) conservó 50.000 predicciones idénticas en Colab, pero fue 0,14 % más lenta de media y se descartó. La preparación JIT en buffers anterior sigue por defecto.

### Comprobación final de la entrega

Las bases piden un repositorio GitHub clonable con este `model.py`, sus pesos y módulos, y `requirements.txt`. Antes de enviar el enlace, probar desde un clon limpio usando el `harness.py` del repositorio oficial `GDG-UAM/ml-contest-materials`. La evaluación real se hace en Google Colab; el demo de 400 señales sirve para verificar el contrato, no para estimar la nota oculta. El enlace se envía por correo a los cuatro organizadores antes del **12 de noviembre de 2026 a las 23:59**. El panel `ml-dashboard` es local y no forma parte de la entrega.

### Comprobación de ONNX

`models/exp033.sesgo.onnx` contiene los mismos pesos y sesgo que la variante NumPy. `scripts/exportar_onnx.py` requiere `onnx` solo en el entorno de desarrollo; la inferencia de entrega usa `onnxruntime`. `scripts/comparar_backend_entrega.py` compara ambos motores con el harness oficial sobre el demo repetido para medir tiempo y paridad de predicciones.

En este Mac, el harness dio las mismas clases y scores en demo, pseudo-test y `val_10k` (60 400 muestras); la latencia media de ONNX fue entre un 4 % y un 11 % menor según el conjunto. En Colab Linux, David instaló las dependencias y ejecutó la comparación con el harness: 10 000 predicciones idénticas, latencia media **0,5205 ms** con NumPy frente a **0,4342 ms** con ONNX. Esto proyecta **+9,49 % de score por velocidad** a 60 s si el throughput y la distribución se mantienen; el score 18,894532 de esa prueba procede del demo repetido y no es una nota de examen. El entorno oficial puede rendir distinto. `requirements-onnx.txt` queda como alias de `requirements.txt` para reproducir la prueba anterior.

### Preparación JIT en evaluación

La preparación JIT es ahora la ruta por defecto y Numba 0.68.1 forma parte de `requirements.txt`; `requirements-numba.txt` queda como alias para repetir la prueba. En este Mac conservó las clases en **60 400 muestras** de demo, pseudo-test y `val_10k`; el harness sobre el demo repetido pasó de 0,0660 a 0,0367 ms por muestra, con score idéntico. En Colab Linux, David obtuvo **0,4014 → 0,1852 ms** y 10 000/10 000 predicciones iguales, lo que proyecta **+47,22 % de score por velocidad** a 60 s si throughput y distribución se mantienen. El score **18,894532** de esta prueba procede del demo repetido, no es una nota del examen. El instalador de Colab avisó de una incompatibilidad con `pytensor` preinstalado; ni la entrega ni el harness lo importan, y las cuatro evaluaciones acabaron correctamente. `scripts/comparar_preparacion_jit.py` automatiza la comparación.

Desde el 10 de noviembre, ONNX reutiliza por defecto buffers de entrada y salida mediante I/O Binding (`GDG_ORT_IOBINDING=0` restaura la ruta anterior). En Colab CPU, el harness verificó **10.000/10.000 predicciones idénticas** y midió **0,12840 → 0,11309 ms** por muestra en dos órdenes alternados: **−11,93 % de latencia** sobre el demo repetido. Si esa relación se mantiene durante los 60 s oficiales, proyecta aproximadamente +6,6 % de score por velocidad; no se ha medido el test oculto. El peso `exp033.sesgo.onnx` y la preparación JIT siguen siendo los mismos. Véase `plans/io_binding_01.md`.

La preparación JIT también escribe por defecto directamente en esos buffers (`GDG_JIT_BUFFERS=0` vuelve a la preparación anterior). En otra comparación Colab CPU, las **10.000 predicciones fueron idénticas** y la latencia media bajó **0,11051 → 0,10639 ms** por muestra (**−3,72 %**) en dos órdenes. La segunda pareja mejoró por poco; la cifra es de esta medición, no una garantía de throughput futuro. El peso y el resultado por muestra no cambiaron. Véase `plans/jit_buffers_01.md`.

### Resultados recientes y criterio de entrega

Con el mismo harness local y ONNX+JIT, las variantes entrenadas después de `exp043` mejoran a menudo los controles sintéticos y pierden en el demo. Cada columna compara modelos **dentro del mismo conjunto**; los scores de conjuntos con 400, 10 000 y 50 000 muestras no son comparables entre sí.

| Modelo | Demo 400 | Pseudo-test 50k | Val10k | Acierto clase 6/7 en demo |
| --- | ---: | ---: | ---: | ---: |
| **exp033, entrega** | **3,778906** | 48,561543 | 23,557647 | 64 % / 64 % |
| exp047, consistencia entre vistas | 3,287634 | **52,677401** | 24,772631 | 48 % / 72 % |
| exp048, MixUp | 3,140696 | 52,594106 | **25,087263** | 42 % / 74 % |
| exp049, alineación de dominio | 3,323500 | 51,670775 | 24,800716 | 46 % / 72 % |
| exp051, CNN multiescala | 3,703511 | 52,118131 | 24,563333 | 56 % / 74 % |

`exp051` también tarda aproximadamente **1,4 veces** lo que `exp033` en el mismo Mac; su mejora sintética no compensa la velocidad. Una adaptación profunda supervisada con el demo (`exp050`) dio 3,948911 y 85,5 % de acierto **fuera de pliegue**, solo cuatro aciertos netos más que exp033 y por debajo del filtro fijado antes de entrenarla. No se produjo un peso final. Una versión INT8 de exp033 conservó los aciertos en Mac, pero en Colab Linux cayó a 66,75 % y fue más lenta; se descartó. El demo de 400 etiquetas se ha consultado muchas veces, y pseudo-test y val10k reutilizan origen o simulador: no son una estimación fiable de la nota oculta. Por ello se mantiene exp033 ONNX+JIT mientras se investiga una validación más cercana al test.

Para exportar otro modelo y comprobar que coincide con PyTorch:

```bash
.venv/bin/python scripts/exportar_numpy.py --pesos models/expNNN.pt --sesgo models/expNNN.sesgo.json
```

## Estado de los modelos

La variable `GDG_MODEL_PATH` permite evaluar otro peso local (`.pkl`, `.pt`, `.npz` u `.onnx`) sin cambiar el código; con `.pt`, `GDG_SESGO_PATH` añade el empujón. La tabla siguiente recoge los primeros 20 experimentos con el protocolo local antiguo (1000 muestras, 2 s), que no es comparable con el de la sección anterior.

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
