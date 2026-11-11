# JIT buffers 01: preparar directamente en los buffers de ONNX

Preparado el 10 nov 2026 **antes de la medición en Colab**. La entrega actual usa `exp033.sesgo.onnx`, I/O Binding y preparación JIT. La nueva ruta estuvo apagada durante la preparación; tras pasar el filtro se activó por defecto. `GDG_JIT_BUFFERS=0` conserva la ruta anterior. David ejecuta las evaluaciones; el asistente prepara código y analiza informes.

## Hipótesis y cambio único

La preparación JIT actual crea arrays nuevos para la serie normalizada y los 17 auxiliares. Después `RedOnnx` los copia a los buffers ya vinculados con ONNX Runtime. La variante `GDG_JIT_BUFFERS=1` hace que Numba escriba directamente en esos buffers y reutiliza un tercer array para interpolar huecos. Se compila en `Model()` durante setup, fuera de los 60 s cronometrados. El peso ONNX, la arquitectura, la normalización y el proveedor CPU son los mismos. La función JIT original se dejó intacta para que la comparación tenga un control real.

Una comprobación local sin compilación Numba, con 128 señales sintéticas de metadatos variados, huecos y canales enteros ausentes, dio **128/128 predicciones y entradas ONNX idénticas bit a bit** entre las rutas. Este Mac no tiene Numba instalado y el entorno no pudo acceder a PyPI; la paridad del código compilado y la velocidad deben comprobarse en Colab.

## Prueba fijada

El paquete `../jit_buffers_colab.zip` lleva el modelo, el harness oficial y `scripts/comparar_jit_buffers.py`. En Colab CPU, el script compara la ruta actual y la nueva en orden actual–nueva–nueva–actual, con las mismas 10.000 señales repetidas por corrida y predicciones por muestra guardadas. No usa el score del demo repetido para elegir modelo: las copias solo miden velocidad.

Paquete final generado y verificado: SHA-256 `5a66ce737c8d5e2bea9cebf1d4289c5e2d7dd8c2428f294e0815bf6328f09ca7`. Los scripts y módulos modificados pasaron análisis de sintaxis y el ZIP pasó comprobación de integridad. Se regeneró tras activar la ruta nueva por defecto; el comparador fija explícitamente ambos modos, así que la medición anterior conserva su validez.

Activar `GDG_JIT_BUFFERS=1` por defecto **solo si** las 10.000 predicciones de las cuatro corridas son idénticas, ambas corridas nuevas son más rápidas que su pareja de control y la reducción media de latencia es al menos **3 %**. Si falla o hay error de compilación/contrato, conservar la ruta actual y no retocar esta variante con el mismo benchmark. La medición es del demo repetido, no del test oculto.

## Resultado en Colab CPU (10 nov 2026)

David ejecutó las cuatro corridas. El informe descargado se conservó en `experiments/jit_buffers_harness.json` (SHA-256 `30d539aa4861e7a36cac184ac673c30517a16c9e7f1b5dba62b174978fad9e40`). Hubo **10.000/10.000 predicciones idénticas** en cada comparación y las cuatro corridas terminaron.

| Orden | Ruta | Latencia media |
| ---: | --- | ---: |
| 1 | JIT actual | 0,11108 ms |
| 2 | JIT en buffers | 0,10300 ms |
| 3 | JIT en buffers | 0,10978 ms |
| 4 | JIT actual | 0,10993 ms |

Media JIT actual **0,11051 ms**, nueva **0,10639 ms**: **−3,72 %**. Las dos parejas son más rápidas con la variante, aunque la segunda diferencia es pequeña (0,10993 frente a 0,10978 ms). Se cumple el mínimo prefijado del 3 %, por lo que `GDG_JIT_BUFFERS=1` pasó a ser el valor por defecto en `src/red_onnx.py`. Se puede desactivar con `GDG_JIT_BUFFERS=0`. Si el throughput relativo se mantiene en el examen de 60 s, la ganancia proyectada de score por velocidad es ~**1,92 %**; no es una nota del test oculto. El score 18,894532 del demo repetido no se usa como calidad ni como puntuación oficial.
