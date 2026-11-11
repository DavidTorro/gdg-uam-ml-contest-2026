# selectivo_051_01: segunda opinión multiescala solo en dudas 6/7

Preparado el 10 de noviembre de 2026 antes de evaluar esta regla. No entrena nada ni cambia `model.py`. El modelo de entrega sigue siendo `exp033.sesgo.onnx`.

## Regla fija

`exp051` es una CNN multiescala ya entrenada que comete errores diferentes de `exp033`, pero ejecutarla siempre es más lenta. Se reutiliza **exactamente** la regla de activación anterior (`models/selectivo_01.json`): si `exp033` predice 6 o 7 y la diferencia absoluta de sus logits 6–7 es menor que **0,7819924354553223**, consultar `exp051`; si no, conservar `exp033`. Para las señales consultadas, promediar las distribuciones softmax de ambas redes a partes iguales y elegir la clase mayor. El umbral se había fijado con márgenes del demo sin abrir etiquetas; no se mueve para este modelo.

`exp033` usa su sesgo histórico; `exp051` no tiene sesgo. Ambos emplean el mismo preprocesado (`sin_atajos`, con escala y grupo). El coste de `exp051` se paga solo en las señales activadas. La evaluación offline de calidad usa PyTorch para ambos pesos; si pasa, habrá que exportar `exp051` a ONNX y verificar paridad y latencia con el harness oficial en Colab antes de considerar la entrega.

## Auditoría prefijada

Ejecutar una sola vez `scripts/auditar_selectivo_051_01.py` en `val_exp052_audit` simulado y crudo, sin abrir el demo. Se compara con `exp033` sobre las mismas 5.000 señales. La auditoría previa de `exp033` sirve de control para comprobar la paridad de sus predicciones. Son dos vistas de las mismas señales, y este bloque ya ha sido estudiado muchas veces: el resultado es exploratorio y no estima el test oculto.

Pasar a pruebas de Colab solo si **todos** los filtros se cumplen:

1. Score simulado a igual N mejora por lo menos **+0,5**.
2. Score crudo a igual N no baja.
3. En simulado, clases 6 y 7 no pierden más de **5 aciertos cada una** y el total 6+7 gana al menos **10**.

Después, exigir en Colab CPU una latencia media de la cascada como máximo **8 %** mayor que la base y una ganancia proyectada de score a 60 s de al menos **2 %**, usando calidad a igual N × raíz cuadrada de la razón de latencias. Comprobar predicciones idénticas entre PyTorch y ONNX en las señales auditadas antes de cambiar `model.py`. Si falla cualquier filtro, descartar esta regla sin variar el umbral, el peso 1/2–1/2 o el presupuesto mirando la auditoría. El demo no se utiliza para rescatarla.

## Resultado del 10 de noviembre

`scripts/auditar_selectivo_051_01.py` evaluó la regla sin abrir el demo y verificó que las predicciones de `exp033` coinciden con la auditoría ONNX anterior. Guardó `experiments/selectivo_051_01_auditoria.json`.

| Vista, 5.000 señales | exp033 | Selectivo exp051 | Cambio |
| --- | ---: | ---: | ---: |
| Score simulado a igual N | 15,284593 | 16,080845 | **+0,796252** |
| Score crudo a igual N | 16,544563 | 17,022163 | **+0,477600** |
| Aciertos clase 6 simulada | 433/506 | 423/506 | **−10** |
| Aciertos clase 7 simulada | 311/492 | 353/492 | **+42** |

Se activó `exp051` en 201/5.000 señales simuladas y 162/5.000 crudas; cambió 88 y 66 predicciones. Pasan los filtros de score y de ganancia conjunta 6+7 (+32), pero **falla el límite prefijado de pérdida de clase 6** (−10, máximo permitido −5). Conforme al plan, no exportar ONNX ni medir en Colab para rescatar esta regla; no retocar umbral o pesos con la auditoría. Mantener `exp033` como entrega. La mejora agregada es una pista exploratoria de complementariedad, no evidencia independiente del oculto.
