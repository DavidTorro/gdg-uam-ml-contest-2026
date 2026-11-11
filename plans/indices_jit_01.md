# Índices JIT 01: reutilizar posiciones válidas por canal

Preparado el 10 nov 2026 **antes de medir en Colab**. **Resultado: descartada y retirada del código de entrega.** La entrega sigue usando `exp033.sesgo.onnx`, I/O Binding y preparación JIT directa en buffers. David ejecuta las evaluaciones; el asistente prepara y analiza.

## Cambio e hipótesis

Para interpolar huecos, el código JIT crea un array de 256 índices dentro de cada uno de los cuatro canales en cada llamada. La variante reutiliza un único array de índices entre canales y predicciones. El modelo, los pesos, los valores interpolados, la normalización y el proveedor CPU permanecen iguales. La función ya probada se conserva como control y la variante nueva se compila en `Model()` durante setup.

Sin Numba compilado en este Mac, una comprobación técnica con 128 señales sintéticas, incluidos huecos y canales enteros ausentes, dio **128/128 entradas, logits y clases idénticos bit a bit**. El entorno local no tiene Numba y no pudo acceder a PyPI; Colab verificará compilación y paridad real.

## Comparación prefijada

El paquete `../indices_jit_colab.zip` usa el harness oficial con la misma secuencia de **50.000 señales** por corrida (125 copias de las 400 del demo). Corre en orden control–variante–variante–control, guarda predicciones por muestra y compara los dos pares de latencia. Se eligieron 50.000 en vez de 10.000 para reducir el ruido de una medición de aproximadamente un segundo. Las copias no aportan información nueva sobre la calidad del clasificador.

El paquete se generó y verificó con SHA-256 `1e830b252d5425269e1f76326d473025b56f8a3b2d2d67bdc4e87db55f405c30`; los módulos pasaron análisis de sintaxis y el ZIP pasó comprobación de integridad.

Activar `GDG_JIT_PRESENTES=1` por defecto **solo si** las 50.000 predicciones coinciden en las cuatro corridas, ambas corridas nuevas son más rápidas que su pareja de control y la reducción media de latencia es al menos **3 %**. Si falla un requisito o aparece un error de compilación o de contrato, conservar la ruta actual y cerrar esta variante sin ajustar el umbral con el resultado. No usar el score del demo repetido como estimación del examen oculto.

## Resultado en Colab CPU (10 nov 2026)

David completó las cuatro corridas. Informe conservado en `experiments/indices_jit_harness.json` (SHA-256 `e75cf28504ad626c9d4c6c4ec61de34511394ffbbf698088a71c2e92779189c9`). Las **50.000 predicciones coincidieron** en cada comparación.

| Orden | Ruta | Latencia media |
| ---: | --- | ---: |
| 1 | Control | 0,104934 ms |
| 2 | Índices reutilizados | 0,104077 ms |
| 3 | Índices reutilizados | 0,103519 ms |
| 4 | Control | 0,103146 ms |

Media control **0,104040 ms**, variante **0,103798 ms**: reducción de solo **0,23 %**. En el segundo par la variante fue más lenta. **No pasa** los filtros de ambos pares y 3 %. No se repitió ni ajustó el benchmark. Se retiró `GDG_JIT_PRESENTES` y su función opcional de `src/`; la versión probada permanece en el ZIP local `../indices_jit_colab.zip` para reproducir este resultado, pero no forma parte de la entrega. El score 42,249459 del demo repetido no se usa como calidad ni como nota del examen.
