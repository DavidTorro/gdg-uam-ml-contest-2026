# calibrador_67_01: combinar dos opiniones en casos dudosos 6/7

Preparado el 10 de noviembre de 2026 antes de ajustar el calibrador. David ejecuta el ajuste; el asistente hace la auditoría después. `model.py` y la entrega `exp033.sesgo.onnx` no cambian.

## Hipótesis y regla cerrada

La consulta selectiva de `exp051` ganó 42 aciertos de clase 7 y perdió 10 de clase 6 en 5.000 señales simuladas. Usar media 1/2–1/2 parece inclinar demasiado la decisión hacia 7. Un calibrador lineal de **solo dos variables**, margen 6−7 de `exp033` y margen 6−7 de `exp051`, puede aprender el equilibrio con una parte de validación separada de la auditoría. Esta idea surge después de consultar el bloque de auditoría con una regla distinta: incluso si funciona, la evidencia está influida por esa observación previa y no será una estimación independiente del oculto.

Se conserva el umbral **0,7819924354553223** de `models/selectivo_01.json`. `exp051` solo se consulta cuando `exp033` predice 6/7 y su margen absoluto 6−7 está por debajo del umbral. El calibrador solo puede elegir 6 o 7 en esas señales; fuera de ellas se conserva exactamente la predicción de `exp033`. No se optimiza el umbral ni la frecuencia de consulta.

## Ajuste que ejecuta David

`scripts/entrenar_calibrador_67_01.py` usa **solo** `data/val_exp052_select.npz` para ajustar una regresión logística L2 con dos variables estandarizadas, `C=0,1`, pesos de clase equilibrados y semilla 2026. Entrena solo con las señales activadas cuya etiqueta verdadera es 6 o 7. No abre `val_exp052_audit`, el demo ni otras etiquetas. Guarda el resultado en `models/calibrador_67_01.json` sin sobrescribirlo.

```bash
.venv/bin/python scripts/entrenar_calibrador_67_01.py
```

Después, el asistente ejecuta `scripts/auditar_calibrador_67_01.py` una sola vez sobre las 5.000 señales de `val_exp052_audit`, simuladas y crudas. Ambas vistas comparten señales. Exigir todos los filtros frente a `exp033`: score simulado a igual N **≥+0,5**, score crudo sin caída, pérdida máxima de **5 aciertos** en cada clase 6/7, y ganancia conjunta 6+7 de al menos **10**. Si falla, no retocar C, pesos, intercepto ni umbral según la auditoría. Si pasa, exportar `exp051` a ONNX, verificar predicciones y exigir en Colab un sobrecoste de latencia **≤8 %** y una ganancia de score proyectada a 60 s de al menos **2 %**. El bloque de auditoría ya fue consultado en experimentos previos; incluso un pase requiere cautela y no revela la nota del oculto.

## Resultado del 10 de noviembre

David ajustó el calibrador en 181 señales activadas con etiqueta 6/7 del bloque de selección (63 de clase 6 y 118 de clase 7). El asistente ejecutó `scripts/auditar_calibrador_67_01.py`, verificó que las predicciones base coinciden con la evaluación ONNX anterior y guardó `experiments/calibrador_67_01_auditoria.json`, sin abrir el demo.

| Vista, 5.000 señales | exp033 | Calibrador | Cambio |
| --- | ---: | ---: | ---: |
| Score simulado a igual N | 15,284593 | 15,715246 | **+0,430653** |
| Score crudo a igual N | 16,544563 | 16,710225 | +0,165662 |
| Aciertos clase 6 simulada | 433/506 | 434/506 | **+1** |
| Aciertos clase 7 simulada | 311/492 | 330/492 | **+19** |

El acierto conjunto 6+7 sube 20 y ninguna de las dos clases pierde, pero el score simulado **no alcanza el +0,5 prefijado**. La mejora de calidad a igual N es +2,82 %; por la raíz cuadrada de la velocidad, la cascada solo mantendría una ganancia neta si su latencia aumentase menos de **5,71 %**, y para superar el objetivo neto de +2 % necesitaría menos de **1,61 %**. No se ha medido esa latencia en Colab. **Decisión conforme al plan:** no exportar ni cambiar la entrega; no ajustar el calibrador con esta auditoría. La mejora de ambas clases es una pista de investigación, no prueba de ventaja a 60 s ni del test oculto.
