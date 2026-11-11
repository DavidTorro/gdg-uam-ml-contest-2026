# Retención 6/7 · experimento 01

Preparado el 10 nov 2026 antes de entrenar el candidato. La entrega sigue siendo `exp033.sesgo.onnx`. David ejecuta los entrenamientos y las evaluaciones.

## Hipótesis

Los afinamientos desde `exp033` elevaron el score del simulador, pero desplazaron predicciones de clase 6 hacia la 7. Repetimos **exactamente** el entrenamiento de `d02_control_28` (90.000 señales, 40 épocas, semilla 2028, LR 0,0002, selección de época en `val_exp052_select`) con una única pérdida adicional. En cada lote, el maestro congelado `exp033.pt` con su sesgo histórico fijo marca las señales 6/7 que clasifica correctamente entre esas dos clases. La alumna paga una penalización si su margen para la clase verdadera cae por debajo del margen del maestro, limitado a 2 logits. Coeficiente fijado: **0,2**. No se cambian el simulador, las etiquetas, la arquitectura ni el sesgo de la candidata en inferencia.

El sesgo del maestro procede de una calibración histórica sobre el demo; por tanto, este entrenamiento hereda esa influencia. No se vuelve a usar el demo para elegir el coeficiente. Los archivos y hashes exactos están en `plans/retencion_67_01.json`.

## Ejecutar

Desde `gdg-uam-ml-contest-2026`, en el terminal de David:

```bash
bash scripts/lanzar_retencion_67_01.sh
```

El lanzador comprueba los hashes de las siete fuentes antes de entrenar y contiene la orden completa. Si termina, genera `models/r67_retencion_28.pt` y `experiments/r67_retencion_28/entrenamiento.json`. La evaluación incluida en el comando usa el mismo bloque de selección que el control; no es una auditoría independiente. A continuación, ejecutar **una vez**:

```bash
.venv/bin/python scripts/auditar_retencion_67.py
```

El informe se guardará en `experiments/retencion_67_01_auditoria.json`, sin sobrescribir una ejecución previa.

## Filtro fijado antes del resultado

Frente a `exp033` con su sesgo fijo, en las 5.000 señales de `val_exp052_audit`:

1. Score simulado a igual N: al menos **+0,5**.
2. Score crudo a igual N: **sin caída**.
3. Clase 6 simulada: pérdida máxima de **10 aciertos**.
4. Clases 6+7 simuladas: ganancia mínima de **15 aciertos**.

La auditoría también muestra `d02_control_28` como control de afinamiento, sin modificar los umbrales. El bloque y el simulador ya fueron consultados varias veces; estos filtros son una decisión práctica, no una estimación independiente del test oculto. Si algún filtro falla, se descarta el candidato sin ajustar el coeficiente, el margen, la época ni un sesgo a partir del resultado. Si pasa, se exporta ONNX y David comprueba latencia en Colab CPU y calidad con el harness antes de cambiar `model.py`. La nota del demo ya está contaminada por consultas previas y solo sirve como comprobación adicional.

## Resultado del 10 nov 2026

David completó el entrenamiento y ejecutó la auditoría una vez. La mejor época fue la 13. El informe inmutable está en `experiments/retencion_67_01_auditoria.json`; SHA-256 del checkpoint: `7da8068a05853230135c015098ad3867929c1e5b628eadecd00c535906ef28ed`.

| Vista, 5.000 señales | exp033 sesgado | d02_control_28 | r67_retencion_28 |
| --- | ---: | ---: | ---: |
| Score simulado | 15,284593 | 16,463715 | 16,129448 |
| Score crudo | 16,544563 | 17,411472 | 17,183973 |
| Clase 6 simulada | 433/506 | 385/506 | 413/506 |
| Clase 7 simulada | 311/492 | 407/492 | 353/492 |
| Clases 6+7 simuladas | 744 | 792 | 766 |

Frente a `exp033`, el candidato gana **+0,844855** de score simulado y **+0,639410** crudo; el total 6+7 gana **22** aciertos. Pero clase 6 pierde **20**, por encima del límite prefijado de 10. `d02_control_28` muestra que la retención recuperó 28 aciertos de clase 6 respecto al afinamiento sin ella, mientras cedió 54 de clase 7 y 0,334267 de score simulado. El cambio desplaza la frontera, pero no pasa el filtro de entrega.

**Decisión:** descartar `r67_retencion_28` para la entrega. No exportar su ONNX ni usar el demo o Colab para rescatarlo; mantener `models/exp033.sesgo.onnx` en `model.py`. No ajustar el coeficiente 0,2 ni el margen 2 usando esta auditoría ya reutilizada.
