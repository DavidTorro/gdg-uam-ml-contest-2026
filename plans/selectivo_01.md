# Consulta selectiva de una segunda CNN

Preparado el 10 nov 2026 **antes de abrir etiquetas con el candidato**. La entrega sigue siendo `model.py` con `exp033.sesgo.onnx`. David ejecuta las evaluaciones. Esta es una arquitectura de inferencia, sin entrenamiento nuevo.

## Hipótesis y regla cerrada

El ensamble de varias CNN mejoró la calidad a igual número de señales, pero ejecutar todas en cada muestra cuesta demasiado dentro de los 60 s de `predict`. `model_selectivo.py` ejecuta siempre `exp033.sesgo.onnx`. Solo si su clase ganadora es 6 o 7 **y** la diferencia absoluta entre sus logits 6 y 7 queda bajo un umbral, consulta `exp034.sesgo.onnx`. Reutiliza los buffers ya preparados para no repetir el preprocesado. En esas señales combina ambas distribuciones softmax con pesos 1/2–1/2 y elige el máximo. En las demás devuelve exactamente la predicción de exp033.

`scripts/preparar_selectivo_01.py` fijó el umbral como el cuantil 20 % del margen 6/7 entre las señales que exp033 predijo como 6/7 en el **demo sin leer etiquetas**. Umbral fijo: **0,781992** (valor completo en `models/selectivo_01.json`). Esto activó la segunda red en 20/400 señales (5 %). En una prueba de funcionamiento, ocho de las 400 predicciones cambiaron; no se consultó si eran aciertos. La elección del presupuesto no optimiza calidad con el demo. Los dos sesgos de los pesos fueron calibrados históricamente con el demo, así que existe influencia indirecta previa.

La exportación de exp034 pasó `onnx.checker`; en cuatro entradas sintéticas, PyTorch y ONNX eligieron las mismas clases (diferencia máxima de logits 3,52×10⁻⁶). La candidata devolvió clases válidas en 400 señales sin consultar etiquetas. Estas comprobaciones son de implementación, no de calidad. Los hashes de pesos, regla y datos están en `plans/selectivo_01.json`.

## Auditoría de calidad, antes de Colab

David ejecuta **una vez**, desde el repositorio:

```bash
.venv/bin/python scripts/auditar_selectivo_01.py
```

Este auditor fuerza la ruta NumPy de preparación porque Numba no está instalado en la `.venv` de este Mac. Compara `model.py` y `model_selectivo.py` sobre las mismas 5.000 señales `val_exp052_audit` en vistas simulada y cruda, con el mismo ONNX y sin límite de tiempo. La ruta JIT se comprobará en Colab solo si pasa el filtro. Guarda `experiments/selectivo_01_auditoria.json` sin sobrescribir.

Pasar a Colab **solo si se cumplen todos** los criterios frente a exp033:

1. Score simulado a igual N: **≥+0,5**.
2. Score crudo a igual N: **sin caída**.
3. Clase 6 simulada: pérdida máxima **5 aciertos**.
4. Clase 7 simulada: pérdida máxima **5 aciertos**.
5. Total 6+7 simulado: ganancia mínima **10 aciertos**.

Este bloque y el demo se han consultado muchas veces: el filtro reduce arbitrariedad, pero **no estima el examen oculto**. Si falla, descartar la candidata sin mover el umbral ni cambiar los pesos contra estos datos.

## Filtro final si pasa calidad

Comparar ambos `Model` con el harness oficial en Colab CPU, un hilo, en dos órdenes alternos sobre la misma secuencia repetida, y comprobar la ruta JIT. Exigir un sobrecoste medio de latencia **≤12 %** y que el score simulado a igual N multiplicado por `sqrt(latencia_base/latencia_selectiva)` proyecte una mejora de **al menos 2 %**. La repetición sirve para medir velocidad, no aporta etiquetas independientes. Si alguna pareja de corridas cambia mucho, repetir solo para diagnosticar el entorno, sin ajustar la regla con el demo. Cambiar la entrega únicamente tras completar calidad, velocidad y contrato en el entorno oficial.

## Resultado del 10 nov 2026

David ejecutó la auditoría una vez. Informe: `experiments/selectivo_01_auditoria.json`. La ruta NumPy activó la segunda red en 201 de las 5.000 señales simuladas y cambió 67 predicciones.

| Vista, 5.000 señales | exp033 | selectivo | Cambio |
| --- | ---: | ---: | ---: |
| Score simulado | 15,284593 | 15,686005 | **+0,401412** |
| Score crudo | 16,544563 | 16,893544 | +0,348981 |
| Clase 6 simulada | 433/506 | 432/506 | −1 |
| Clase 7 simulada | 311/492 | 328/492 | +17 |
| Clases 6+7 simuladas | 744 | 760 | +16 |

Pasan cuatro filtros, pero **no** el mínimo de +0,5 en score simulado. La mejora observada es interesante, pero el segundo modelo añade coste y no se debe saltar la regla de avance por una décima. **Descartar `model_selectivo.py` para entrega y no medirlo en Colab para rescatarlo.** Mantener `model.py` con `exp033.sesgo.onnx`. No cambiar el umbral, la fracción 20 % ni los pesos usando esta auditoría.
