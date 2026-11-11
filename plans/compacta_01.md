# Compacta 01: menos canales a partir de exp033

Preparado el 10 nov 2026 **antes de entrenar y auditar la candidata**. David ejecuta el entrenamiento y las evaluaciones. `model.py` sigue entregando `exp033.sesgo.onnx` hasta demostrar una mejora de score con el coste por predicción incluido.

## Hipótesis y construcción

En vez de volver a desplazar la frontera 6/7, reducir el coste de la CNN. `scripts/crear_compacta_01.py` toma `exp033.pt` y conserva el 75 % de los canales en cada una de sus cuatro convoluciones (12/24/36/48 en lugar de 16/32/48/64) y 48 de 64 neuronas ocultas. Los elige por el producto de magnitud de pesos entrantes y salientes, **sin leer ninguna etiqueta ni muestra de validación/demo**. Las posiciones elegidas quedan en `models/c01_pruned_init.pt`, junto con hashes de exp033 y de su sesgo histórico. El sesgo +0,9414 de clase 6 queda como buffer fijo de la red compacta y se aplica tanto al entrenar como al inferir; no se reajusta.

La red tiene **31.760 parámetros** frente a 55.656 de exp033 (−43 %). Menos parámetros no garantizan menor latencia en Colab; esa es la segunda parte de la prueba. El checkpoint inicial se creó sin entrenamiento y pasó carga estricta. En tres entradas sintéticas, PyTorch y ONNX eligieron las mismas clases; diferencia máxima de logits 2,87×10⁻⁶. Esto verifica implementación, no calidad.

## Entrenamiento único

Desde el repositorio:

```bash
bash scripts/lanzar_compacta_01.sh
```

El lanzador verifica los hashes antes de empezar. Ajusta la subred con la misma receta de 40 épocas, semilla 2028, LR 0,0002, 90.000 señales simuladas y selección de época en `val_exp052_select` que el afinamiento de control. La única intervención es la arquitectura/estado inicial podados; el sesgo histórico de exp033 se mantiene fijo. Al terminar se espera `models/c01_compacta_28.pt` y el informe de entrenamiento. Después David ejecuta **una vez**:

```bash
.venv/bin/python scripts/auditar_compacta_01.py
```

## Filtros fijados antes del resultado

La auditoría usa las 5.000 señales `val_exp052_audit` en vista simulada y cruda, comparando con `exp033.sesgo.onnx` a igual N. **Pasar a medir velocidad solo si se cumplen los cinco puntos**:

1. Score simulado: caída máxima **0,3**.
2. Score crudo: caída máxima **0,3**.
3. Clase 6 simulada: pérdida máxima **10 aciertos**.
4. Clase 7 simulada: pérdida máxima **10 aciertos**.
5. Total 6+7 simulado: pérdida máxima **5 aciertos**.

Si pasa, exportar ONNX y comparar en Colab CPU con el harness oficial, alternando orden de los dos modelos sobre la misma secuencia repetida de 10.000 señales. Exigir **≥15 % menos latencia media**, ambas parejas a favor de la compacta, y que el score de calidad simulado ajustado por `sqrt(latencia_base/latencia_compacta)` proyecte **≥5 %** de ganancia frente a exp033. La misma secuencia repetida solo mide velocidad. El demo de 400 señales, ya muy consultado, se abre solo al final para comprobar contrato y detectar una caída grave, no para ajustar pesos ni escoger épocas.

Este bloque de auditoría y su simulador se han usado muchas veces: son un filtro práctico, no prueba independiente del examen oculto. Si falla cualquier filtro, conservar exp033 y cerrar esta variante sin cambiar anchos, semillas, época o sesgo contra estos datos. El plazo de entrega deja `exp033.sesgo.onnx` como respaldo probado.

## Resultado del 10 nov 2026

David completó el entrenamiento y ejecutó la auditoría una vez: `experiments/compacta_01_auditoria.json`. La subred quedó muy cerca en score simulado, pero perdió demasiados aciertos de clase 6.

| Vista, 5.000 señales | exp033 | c01_compacta_28 | Cambio |
| --- | ---: | ---: | ---: |
| Score simulado | 15,284593 | 15,271112 | −0,013481 |
| Score crudo | 16,544563 | 16,313553 | −0,231009 |
| Clase 6 simulada | 433/506 | 383/506 | **−50** |
| Clase 7 simulada | 311/492 | 388/492 | +77 |
| Clases 6+7 simuladas | 744 | 771 | +27 |

Pasan cuatro filtros, pero clase 6 excede ampliamente la pérdida máxima de 10. **Descartar la compacta para entrega y no abrir el demo ni medir su latencia en Colab para rescatarla.** No ajustar la poda, las épocas o el sesgo contra esta auditoría; `model.py` sigue cargando `exp033.sesgo.onnx`.
