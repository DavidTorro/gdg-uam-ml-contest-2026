# Sopa 01: combinar pesos sin añadir coste de inferencia

Preparado el 10 nov 2026 **antes de evaluar el candidato**. La entrega sigue siendo `exp033.sesgo.onnx`. David ejecuta las evaluaciones; el asistente prepara el peso y analiza los resultados.

## Hipótesis y regla fija

Los tres afinamientos `d01_control`, `d02_control_27` y `d02_control_28` mejoraron el mismo bloque de auditoría, pero en el demo cedieron aciertos de clase 6 a la 7. Comparten arquitectura e inicialización con `exp033`, por lo que sus pesos admiten una media coordenada a coordenada. La hipótesis es que una media conserva parte de la robustez de `exp033` y parte de la ganancia de clase 7 sin ejecutar dos redes por señal.

El candidato `s01_sopa` usa **1/2 de `exp033.pt` y 1/6 de cada uno de los tres afinamientos** en todos los tensores de la red. El sesgo histórico de clase 6 de `exp033` (+0,9414) recibe el mismo coeficiente 1/2, por lo que el candidato suma +0,4707 a la clase 6. Ningún coeficiente se elegirá ni retocará con el demo o el bloque de auditoría. No hay entrenamiento ni consulta de etiquetas para construir el peso.

## Crear y comprobar

Desde el repositorio:

```bash
.venv/bin/python scripts/crear_sopa_01.py
.venv/bin/python scripts/exportar_onnx.py --pesos models/s01_sopa.pt --sesgo models/s01_sopa.sesgo.json --destino models/s01_sopa.onnx
```

El creador verifica que los cuatro checkpoints tienen la misma arquitectura, las mismas claves y formas de tensores, y registra sus SHA-256 en el candidato. El exportador hace `onnx.checker` y no sobrescribe un peso existente. La comparación de logits PyTorch–ONNX en señales sintéticas comprueba solo la exportación, no la calidad.

**Preparación completada:** `models/s01_sopa.pt` (SHA-256 `fee42882a3443433be5f2438743c4f1210e405171666fd87f750a1fd1ce9d375`) y `models/s01_sopa.onnx` (SHA-256 `4e8a9d7ca980f5b625483bdb3bf94bd03cd28a598bbd16cfb5bc97074aa20e1c`) se generaron el 10 nov. ONNX usa `CPUExecutionProvider`. En 20 entradas sintéticas, PyTorch y ONNX eligieron las mismas clases y la diferencia máxima de logits fue `5,73×10⁻⁶`. No se han abierto etiquetas de auditoría ni demo para este candidato.

## Filtro antes de abrir el demo

David ejecuta una sola vez `.venv/bin/python scripts/auditar_sopa_01.py`. Compara `s01_sopa` contra `exp033` sesgado en las 5.000 señales `val_exp052_audit` simuladas y en su vista cruda; ambas vistas comparten señales base. Seguir a Colab y demo solo si se cumplen **todos** estos criterios:

1. Score simulado a igual N mejora al menos +0,5.
2. Score crudo a igual N no baja.
3. Clase 6 simulada pierde como máximo 15 aciertos de 506 frente a `exp033`.
4. El total de clases 6 y 7 simuladas gana al menos 15 aciertos frente a `exp033`.

Estos umbrales son un filtro práctico sobre un bloque **ya consultado**, no una prueba independiente del examen oculto. Si falla, se descarta `s01_sopa` sin ajustar coeficientes ni sesgo y sin usar el demo para rescatarlo. Si pasa, comparar su score demo y latencia con `exp033` en Colab CPU bajo el mismo harness. Cambiar la entrega solo si la mejora calidad–velocidad es clara y se conserva el comportamiento de ambas clases. El demo también se ha consultado repetidamente y no debe tratarse como estimador independiente.

## Resultado de auditoría (10 nov 2026)

David ejecutó `scripts/auditar_sopa_01.py` una vez. Informe: `experiments/sopa_01_auditoria.json`.

| Vista, 5.000 señales | exp033 sesgado | s01_sopa | Cambio |
| --- | ---: | ---: | ---: |
| Score simulado | 15,284593 | 16,008298 | +0,723705 |
| Score crudo | 16,544563 | 17,144850 | +0,600287 |
| Clase 6 simulada | 433/506 | 410/506 | **−23** |
| Clase 7 simulada | 311/492 | 357/492 | +46 |
| Clases 6+7 simuladas | 744 | 767 | +23 |

Pasan tres filtros, pero **falla el límite prefijado de clase 6** (−23 < −15). El score global más alto no compensa esta pérdida según la regla fijada. Se descarta `s01_sopa` como candidata de entrega; no retocar coeficientes o sesgo con este bloque ni abrir el demo para rescatarla. `model.py` sigue cargando `exp033.sesgo.onnx`.
