# I/O Binding 01: reutilizar buffers en la inferencia CPU

Preparado el 10 nov 2026 con la ruta nueva inicialmente apagada. Tras pasar el filtro de Colab descrito abajo, se activó por defecto; `GDG_ORT_IOBINDING=0` conserva la ruta anterior. `model.py` y el peso `exp033.sesgo.onnx` no cambiaron.

## Hipótesis y regla

La red recibe siempre `(1,4,256)` y `(1,17)` y devuelve `(1,8)`. ONNX Runtime permite vincular buffers de entrada y salida una sola vez en setup. Reutilizarlos puede reducir la asignación y el envoltorio de arrays en cada `predict`; el modelo, sus pesos y su preparación JIT no cambian. El harness llama a `predict` secuencialmente.

El cambio opcional está en `src/red_onnx.py`. Una comprobación local con `GDG_PREP_JIT=0` y 128 señales sintéticas, incluidas señales con huecos, dio **128/128 clases y logits idénticos bit a bit**. En cuatro mediciones alternadas de 4096 llamadas, la versión normal tardó 0,04648 y 0,04764 ms, y la reutilización 0,04544 y 0,04577 ms por muestra. Estas cifras son solo del Mac y sin JIT; no deciden la entrega.

David ejecutará en Colab CPU `scripts/comparar_io_binding.py` desde el paquete preparado. El script usa el harness oficial, la misma secuencia de 10.000 señales del demo repetido en cada corrida, en orden normal–reutilización–reutilización–normal, y guarda las predicciones por muestra. **La repetición mide latencia, no añade datos independientes ni estima la nota oculta.**

Paquete preparado: `../io_binding_colab.zip` (SHA-256 final `f3a9380b143cef80541b10fb2c299c8c607445df5dc1f443a9124e26e3b63822`). Contiene únicamente `exp033.sesgo.onnx`, el código de inferencia, el demo público y las copias locales del harness y contrato oficiales. El ZIP pasó comprobación de integridad y los scripts pasaron análisis de sintaxis. Se regeneró tras activar la ruta por defecto y corregir la ruta de subida en las instrucciones de Colab; la comparación ya ejecutada fijaba explícitamente ambos modos y no cambia su resultado.

Activar I/O Binding por defecto solo si las 10.000 predicciones son idénticas en las cuatro corridas, ambas corridas con reutilización son más rápidas que su pareja normal y la reducción media de latencia en Colab es al menos **3 %**. Si falla cualquiera, conservar la ruta actual. No cambiar coeficientes, pesos ni preprocesado con este resultado.

## Resultado en Colab CPU (10 nov 2026)

David ejecutó el paquete en Colab. Se conservó el informe descargado en `experiments/io_binding_harness.json` (SHA-256 `c3e23aa8d196f963b2e0003625f6a17a3a0814396787c573b9fd0fdc62d52f7e`). Los cuatro recorridos procesaron las mismas 10.000 señales repetidas y tuvieron **10.000/10.000 predicciones idénticas**. La latencia media por corrida fue:

| Orden | Ruta | Latencia media |
| ---: | --- | ---: |
| 1 | normal | 0,13285 ms |
| 2 | I/O Binding | 0,11126 ms |
| 3 | I/O Binding | 0,11491 ms |
| 4 | normal | 0,12395 ms |

Media normal **0,12840 ms**, media con reutilización **0,11309 ms**: **−11,93 %**. Ambos pares cumplen la mejora y se supera el mínimo prefijado del 3 %. Por ello se activó `GDG_ORT_IOBINDING=1` por defecto en `src/red_onnx.py`; `GDG_ORT_IOBINDING=0` conserva la ruta anterior para diagnóstico. El peso y la preparación JIT no cambian. Con calidad y flujo idénticos, esto proyecta aproximadamente **+6,6 %** de score por velocidad a 60 s si la latencia relativa se mantiene; no es una nota del examen oculto. El score 18,894532 del demo repetido no se usa como calidad ni como puntuación oficial.
