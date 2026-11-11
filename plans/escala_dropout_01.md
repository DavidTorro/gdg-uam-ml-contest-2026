# escala_dropout_01: reducir la dependencia de la amplitud en 6/7

Preparado el 10 de noviembre de 2026, antes de entrenar. David ejecuta los entrenamientos. `model.py` y `exp033.sesgo.onnx` no cambian.

## Hipótesis

La CNN normaliza cada canal por separado y recibe además cuatro auxiliares con los tamaños originales. En `train_90k`, la mediana del cociente de desviaciones canal 1 / canal 0 es **1,269** en clase 6 y **1,003** en clase 7. En las 50 señales de cada clase del demo son **1,027** y **0,994**. Esa pista de amplitud distingue las clases en train y mucho menos en el demo. El demo es pequeño y se ha consultado repetidamente: estos números motivan una prueba, no garantizan generalización.

La opción ya implementada `--prob-escala-cero 0.5` oculta las cuatro auxiliares de tamaño en el 50 % de cada lote de entrenamiento. Conserva la señal normalizada, las etiquetas, la red y la inferencia. Es una intervención menos agresiva que `corr67_01`, que cambió la forma del canal 1 y perdió ~37 % en el control de época. La probabilidad 0,5 queda fijada antes de entrenar y no se ajustará según los resultados.

Se compara con los controles `n01_b_ref_26` y `n01_b_ref_27` de igual semilla, 90.000 señales, 200 épocas, simulador v2, selección de época en `val_exp052_select`, arquitectura y equilibrio. `val_exp052_audit` simulado y crudo se abre solo después de los dos pesos; son las mismas 5.000 señales base y el bloque ya se estudió, de modo que la comparación es exploratoria, no una nota estimada del oculto.

## Entrenar

Desde la raíz del proyecto, ejecutar primero la semilla 2026 y, cuando termine, la 2027:

```bash
bash scripts/entrenar_escala_01.sh 26
bash scripts/entrenar_escala_01.sh 27
```

Los IDs `escala_26` y `escala_27` son nuevos. Los controles ya están entrenados. La semilla 2026 se comprueba antes de lanzar la segunda: si su mejor score en `val_exp052_select` cae más de 20 % frente al control 2026 (**<12,959** frente a 16,199), se detiene por futilidad sin abrir la auditoría ni entrenar la segunda. Este control no sirve para declarar una mejora, pues también elige la época.

Tras ambos pesos, el asistente ejecuta `scripts/auditar_escala_01.py`. Para pasar a una comparación de latencia y contrato en Colab, **ambas** semillas deben: ganar al menos +0,5 de score simulado a igual N, no perder score crudo, no perder más de 10 aciertos en ninguna de las clases 6/7 y ganar al menos 10 entre las dos, y no perder más de 0,3 de score simulado en ningún grupo A/B/C. Si falla, descartar sin modificar el 0,5, la época o el sesgo según la misma auditoría. Si pasa, todavía debe competir con el `exp033` de entrega en velocidad y calidad; la auditoría no estima el oculto.

## Progreso

David completó `escala_26`: 200 épocas, mejor época 152, peso e informe completos. En `val_exp052_select` obtuvo **16,300564**, frente a **16,198929** del control `n01_b_ref_26`. Los campos de receta comparables difieren solo en `prob_escala_cero` (0 → 0,5). Supera el control de futilidad; esta diferencia de +0,102 en el conjunto usado para elegir época **no** acredita mejora. La semilla 2027 y la auditoría siguen pendientes.

## Auditoría terminada

David completó también `escala_27`: 200 épocas, mejor época 161, score de selección 16,333598 frente a 16,365705 de su control. `scripts/auditar_escala_01.py` verificó los pesos, recetas, hashes de las fuentes y alineación de las predicciones de referencia antes de abrir las 5.000 señales reservadas. Guardó `experiments/escala_dropout_01_auditoria.json`. Las dos vistas son las mismas señales base.

| Semilla | Score simulado control → candidata | Score crudo control → candidata | Δ aciertos 6/7 simulado | Mayor caída de grupo simulado |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 16,3004 → 15,8787 (−0,4217) | 17,4957 → 17,3305 (−0,1652) | −16 / +1 | B: −0,387 |
| 2027 | 16,0845 → 15,9638 (−0,1207) | 16,9879 → 16,9925 (+0,0047) | −6 / +16 | A: −0,390 |

**Decisión conforme al filtro prefijado:** ambas semillas bajan en score simulado; ninguna pasa todos los filtros. Descartar `escala_26/27` para la entrega, sin retocar el 0,5 ni consultar el demo para rescatar el resultado. Mantener `models/exp033.sesgo.onnx`. La auditoría reutiliza un bloque históricamente estudiado y no estima el test oculto.
