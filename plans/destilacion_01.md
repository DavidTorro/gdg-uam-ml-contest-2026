# Destilación 01: calidad del cuádruple en una CNN rápida

Preparado el 9 nov 2026. David ejecuta los entrenamientos, las evaluaciones y Git. El asistente prepara código y analiza informes. `model.py` sigue cargando exp033 por defecto mientras se investiga.

## Hipótesis y control

El ensamble uniforme exp033+exp034+exp051+exp053 acierta más que exp033 a igual número de señales, pero ejecutar cuatro redes por muestra puede perder en el examen de 60 s. Una CNN base inicializada desde `exp033.pt` podría aprender parte de esas decisiones con el coste de una sola red. Se comparan **dos corridas** de la misma CNN, ambas con 90k, semilla 2026, 40 épocas, LR 0,0002, simulador v2, clases equilibradas, sin atajos y elección de época en `val_exp052_select`:

- `d01_control`: 40 épocas adicionales con etiquetas reales solamente.
- `d01_kd`: misma receta e inicio, pero pérdida 50 % etiquetas reales + 50 % KL frente al promedio de cuatro maestros, a temperatura 2. Los sesgos históricos de exp033 y exp034 se conservan en sus maestros.

Los maestros se mantienen congelados y ven exactamente la misma señal simulada que la alumna en cada lote. La semilla Torch se reinicia tras cargarlos para que el control y la candidata tengan el mismo orden de lotes y simulaciones. El entrenamiento no abre `test_demo` ni `val_exp052_audit`; los sesgos de dos maestros y la elección histórica del ensamble sí tienen dependencia indirecta del demo. El archivo `plans/destilacion_01.json` fija IDs, receta y hashes de las fuentes.

## Entrenar — David

Desde `gdg-uam-ml-contest-2026/`, uno detrás de otro, sin cambiar parámetros después de ver el resultado del primero:

```bash
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d01_control --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2026 --runs 1 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d01_kd --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --peso-destilacion 0.5 --temperatura-destilacion 2 --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2026 --runs 1 --min-libre-gib 6
```

Ambas evaluaciones del wrapper usan las 5.000 señales de **elección de época**; no son la auditoría. Tras las dos corridas, David ejecuta:

```bash
.venv/bin/python scripts/auditar_destilacion_01.py
```

El auditor comprueba informes, checkpoints y hashes antes de abrir las otras 5.000 señales, crudas y simuladas. Se detiene si falta un peso o si ya existe el informe; no sobrescribe. No consulta demo.

## Decisión prefijada

Seguir con la alumna solo si, en auditoría simulada, supera al **control y a exp033 sesgado** por al menos **+0,5 de score** cada uno; en la vista cruda no baja frente al control; y en simulada no pierde más de cinco aciertos de clase 6 ni de clase 7 respecto al control. Si falla, conservar exp033 como candidato de entrega sin retocar temperatura, peso de KL, épocas o sesgo según estos datos.

Si pasa, exportar la alumna a ONNX, comprobar paridad y comparar con exp033 en Colab Linux bajo el mismo presupuesto de 60 s. Solo entonces considerar cambiar `model.py`. El demo puede servir como comprobación final única, con la advertencia de que ya ha guiado rondas anteriores. La auditoría de 5k también se ha consultado repetidamente, por lo que un resultado positivo sería una señal para avanzar, no una estimación independiente del examen oculto.

## Incidencia del primer lanzamiento (9 nov)

`d01_control` completó entrenamiento y evaluación sobre el conjunto de elección de época. `d01_kd` falló en el primer lote: las probabilidades de los maestros se habían creado con `torch.inference_mode()` y la operación KL necesita guardar ese tensor para derivar respecto a la alumna. Se cambió solo ese bloque a `torch.no_grad()`; una comprobación aislada de KL y retropropagación pasó. El directorio vacío del intento fallido se retiró; no hubo checkpoint `d01_kd`, no se abrió el conjunto de auditoría y no se cambió la receta. Relanzar únicamente `d01_kd` con el comando fijado arriba.

En el segundo intento, David interrumpió la ejecución antes de que se creara el directorio del experimento: el rastreo quedó dentro de la importación de `scikit-learn`/`SciPy` desde el SSD. La receta de 90k con elección externa no llama a `train_test_split`; se pasó esa importación a la función que la necesita, sin cambiar el cálculo de esta receta. La entrada `--help` del entrenador pasó de más de un minuto de espera a unos 2 s y se añadieron mensajes antes de cargar datos y antes de las épocas. `d01_control` conserva su resultado; el segundo intento tampoco abrió auditoría ni produjo pesos de la alumna.

## Resultado de la auditoría (9 nov 2026)

David completó `d01_control` (mejor época 3/40) y `d01_kd` (31/40) y ejecutó `scripts/auditar_destilacion_01.py` después de ambas. Los informes y checkpoints pasaron la verificación de receta antes de abrir las 5.000 señales de auditoría; informe local `experiments/destilacion_01_auditoria.json`. No se abrió el demo en esta ronda.

| Vista a igual N=5000 | exp033 sesgado | Control | Destilada | Δ destilada−control |
| --- | ---: | ---: | ---: | ---: |
| Simulada · score | 15,284593 | 16,205719 | **16,533558** | **+0,327838** |
| Simulada · acierto | 88,16 % | 89,04 % | **89,54 %** | +0,50 p.p. |
| Cruda · score | 16,544563 | 17,326751 | **17,444654** | **+0,117903** |
| Cruda · acierto | 89,74 % | 90,66 % | **90,76 %** | +0,10 p.p. |

En la vista simulada, clase 6 control→destilada **396→401/506** y clase 7 **380→378/492**. En crudo, **407→414** y **392→395**. El IC95 bootstrap pareado descriptivo del cambio de score simulado destilada−control es **[+0,059785,+0,614521]** dentro de estas mismas señales; no cubre otros dominios ni corrige las consultas históricas al bloque.

**Veredicto prefijado: no pasa.** El cambio simulado frente al control es +0,328, por debajo del +0,5 exigido. Los otros cuatro filtros sí pasan, incluido +1,249 frente a exp033 en simulado. La ganancia de seguir entrenando sin maestros es grande (+0,921 sobre exp033), así que no se atribuye a destilación toda la diferencia alumna−exp033. Mantener exp033 como candidato de entrega; no retocar temperatura, peso de KL, épocas o sesgo a partir de esta auditoría. El dashboard `/destilacion` muestra ambos resultados y filtros. Una conclusión sobre el examen oculto requeriría validación realmente nueva y medición de velocidad en Colab.

Para comprobar la viabilidad técnica de ambos pesos ya entrenados, se exportaron `models/d01_control.onnx` y `models/d01_kd.onnx` sin sesgo adicional. Ambos pasaron `onnx.checker`; en tres entradas sintéticas por modelo, PyTorch y ONNX dieron las mismas clases (diferencia máxima de logits: 3,82×10⁻⁶ y 2,87×10⁻⁶, respectivamente). Esta comprobación no es una evaluación de calidad ni altera el veredicto prefijado. `model.py` sigue apuntando a `exp033.sesgo.onnx`. Si David decide comparar los finalistas en el harness, debe ejecutar él mismo las evaluaciones bajo el mismo protocolo y medir latencia antes de considerar cualquier cambio de entrega.
