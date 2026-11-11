# Afinamiento 02: reproducir la mejora sin destilación

Preparado el 9 nov 2026 antes de entrenar las nuevas semillas. David lanza entrenamientos, evaluaciones y auditoría; el asistente analiza e importa los resultados. `model.py` sigue con `exp033.sesgo.onnx`.

## Pregunta

En la auditoría anterior, `d01_control` mejoró el score simulado de `exp033` en +0,921 con 40 épocas adicionales y sin maestros. La destilación solo sumó +0,328 frente a ese control y no pasó su filtro. Comprobamos si la ganancia del **entrenamiento adicional** se reproduce con otras dos semillas, sin cambiar arquitectura, datos, épocas, LR ni elección de checkpoint. La receta exacta y los hashes están en `plans/afinamiento_02.json`.

La auditoría de 5.000 señales ya se ha consultado en rondas anteriores. Esta repetición puede medir estabilidad entre semillas en ese bloque, pero no estima de forma independiente el examen oculto. No se usará el demo para elegir épocas ni parámetros.

## Entrenar — David

Desde `gdg-uam-ml-contest-2026/`, una sola orden lanza las dos corridas secuenciales y después la auditoría. Se detiene si falla una etapa:

```bash
bash scripts/lanzar_afinamiento_02.sh
```

Los comandos individuales son:

```bash
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d02_control_27 --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2027 --runs 1 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d02_control_28 --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2028 --runs 1 --min-libre-gib 6
```

Tras terminar ambas, David ejecuta una sola vez:

```bash
.venv/bin/python scripts/auditar_afinamiento_02.py
```

El auditor verifica las fuentes y la receta de ambas corridas antes de abrir el bloque de auditoría. Produce `experiments/afinamiento_02_auditoria.json` y nunca sobrescribe un resultado anterior.

## Lectura prefijada

Considerar el afinamiento como **reproducido en este bloque** solo si las dos semillas nuevas mejoran a `exp033` por al menos +0,5 de score en la vista simulada, no bajan en la cruda y no reducen los aciertos conjuntos de las clases 6 y 7. Estos umbrales se fijan después de haber visto `d01_control`, por lo que son un filtro práctico de estabilidad, no un test independiente. Si alguna réplica falla, no ajustar la receta con esta auditoría. Si ambas cumplen, David compara candidatos ONNX con el mismo harness y presupuesto en Colab antes de cambiar la entrega; la velocidad y el demo pueden diferir del bloque simulado. `exp033` queda como respaldo hasta completar esa comprobación.

## Resultado (9 nov 2026)

David completó ambas corridas y lanzó el auditor una sola vez. El informe `experiments/afinamiento_02_auditoria.json` indica `reproducido_en_bloque=true`: las dos semillas nuevas superan los tres filtros.

| Modelo | Mejor época / 40 | Score simulado | Δ vs exp033 | Score crudo | Δ vs exp033 | Clase 6 sim. | Clase 7 sim. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp033 sesgado | — | 15,2846 | — | 16,5446 | — | 433/506 | 311/492 |
| d01_control · 2026 | 3 | 16,2057 | +0,9211 | 17,3268 | +0,7822 | 396/506 | 380/492 |
| d02_control_27 · 2027 | 22 | 16,2318 | +0,9472 | 17,4391 | +0,8946 | 399/506 | 379/492 |
| d02_control_28 · 2028 | 21 | 16,4637 | +1,1791 | 17,4115 | +0,8669 | 385/506 | 407/492 |

La mejora se repite, pero hay un intercambio claro entre clases: las dos réplicas pierden 34 y 48 aciertos de clase 6 frente a exp033 y ganan 68 y 96 de clase 7; el total 6+7 sube 34 y 48. El bloque ya había sido estudiado y el simulador fue calibrado históricamente con el demo. No usar el máximo de estas tres semillas como estimación independiente del examen ni cambiar `model.py` aún.

Se exportaron `models/d02_control_27.onnx` y `models/d02_control_28.onnx` sin sesgo adicional. `onnx.checker` pasó; en tres entradas sintéticas por modelo, PyTorch y ONNX eligieron las mismas clases, con diferencias máximas de logits de 5,73×10⁻⁶ y 5,25×10⁻⁶. Esta comprobación solo verifica la exportación. El siguiente paso es que David mida los candidatos bajo el mismo harness y presupuesto en Colab Linux, y revise el demo como comprobación de contrato, recordando que se ha consultado antes. `exp033.sesgo.onnx` sigue por defecto.

Para una primera comparación reproducible desde este Mac, David ejecuta:

```bash
.venv/bin/python scripts/comparar_afinamiento_02.py
```

El script usa el harness oficial con el demo original de 400 señales y con 25 copias del mismo demo para medir latencia con 10.000 llamadas. Repite la medición de velocidad en orden inverso y guarda `experiments/afinamiento_02_harness.json` sin sobrescribir. **La repetición no añade datos independientes** y su score no se usa. Para repetirlo en Colab, llevar los cuatro ONNX, el demo y `ml-contest-materials`, e indicar otro `--out` si ya existe el informe. David hace ambas evaluaciones; el asistente solo prepara y analiza.

## Comparación en Colab CPU (10 nov 2026)

David ejecutó el paquete `afinamiento_02_colab.zip` en Colab con Python 3.13.18 y el harness oficial. El informe descargado se conservó sin cambios en `experiments/afinamiento_02_harness.json` (SHA-256 `3423558d3237a4d8170df6498a92590c3f52e028c7f52581cf3695075d593247`). Cada ONNX usó `CPUExecutionProvider` y un hilo. El instalador avisó de una incompatibilidad con `pytensor` preinstalado, pero las doce evaluaciones del harness completaron correctamente; el modelo no importa `pytensor`.

| Modelo | Demo: aciertos / 400 | Score demo | Clase 6 / 50 | Clase 7 / 50 | Latencia media, 10k repetidas, dos órdenes |
| --- | ---: | ---: | ---: | ---: | ---: |
| **exp033 sesgado** | **338** | **3,778906** | **32** | 32 | 0,1195 ms |
| d01_control | 335 | 3,543575 | 25 | 36 | **0,1146 ms** |
| d02_control_27 | 336 | 3,610102 | 26 | 36 | 0,1195 ms |
| d02_control_28 | **338** | 3,666160 | 25 | **39** | 0,1188 ms |

El score demo de `d02_control_28` es un 3,0 % menor que el de exp033; su latencia solo mejora un 0,6 %. Si ambos mantuvieran calidad y latencia en una corriente larga con la misma distribución, la ganancia de velocidad no compensaría esa diferencia (relación de score estimada `d02_control_28 / exp033` ≈ 0,973). Esta proyección no es la nota del test oculto. Los tres afinados mejoraron el bloque sintético de auditoría, pero ninguno superó a exp033 en este demo y los tres cedieron aciertos de clase 6. **Mantener `models/exp033.sesgo.onnx` como entrega por defecto.** El demo y la auditoría ya se han consultado repetidamente; no ajustar sesgos o hiperparámetros a partir de este resultado.
