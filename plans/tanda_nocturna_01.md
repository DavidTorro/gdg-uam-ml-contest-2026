# Tanda nocturna 01: grupo, arquitectura y simulador

Preparada el 5 de noviembre de 2026 para aprovechar 7–8 horas sin ajustar con las 400 etiquetas del demo. El plan exacto de 20 corridas y hashes de las fuentes está en `plans/tanda_nocturna_01.json`. David ejecuta los entrenamientos largos con `caffeinate` y gestiona Git.

## Preguntas que separa

Hay dos arquitecturas (`base`, `dos_rutas`), dos semillas (2026, 2027) y cinco condiciones. Las 20 combinaciones comparten 90.000 señales de train, 200 épocas, lote 256, LR 0,002, equilibrio de clases, simulador v2, auxiliares de media/level anuladas y **el mismo control de época de 5.000 señales** (`val_exp052_select`). No se cambian pasos o muestras entre condiciones.

| Condición | Cambio único frente a referencia | Hipótesis |
| --- | --- | --- |
| `ref` | Ninguno | Referencia contemporánea justa para cada arquitectura y semilla. Los pesos anteriores eligieron época con controles distintos. |
| `nog` | Anula A/B/C en entrenamiento y evaluación | La identidad de grupo puede inducir fronteras frágiles, especialmente 6/7. |
| `gdrop` | Oculta A/B/C en 50 % de cada lote, pero lo deja visible en evaluación | Aprender con y sin grupo puede conservar utilidad del metadato sin depender de él siempre. |
| `mixsoft` | Probabilidades de mezcla común 1–2 y ritmo 2–3: 0,50/0,70 → 0,25/0,35 | Las mezclas v2 podrían deformar señales etiquetadas; reducirlas quizá retiene morfología. |
| `noisesoft` | Ruido 0,10–0,50 → 0,05–0,35; huecos 0,56 → 0,28; picos 0,35 → 0,20 | La suciedad calibrada con demo podría ser excesiva para aprender la frontera 6/7. |

Es un **factorial parcial**, no un barrido de 20 umbrales: cada intervención responde a un mecanismo diferente; las dos semillas revelan sensibilidad de entrenamiento. El orden hace primero las diez condiciones con semilla 2026, después las mismas diez con 2027. Los tiempos históricos son ~16 min para base y ~20 min para dos rutas por 200 épocas: estimación ~6 h de ajuste más preparación y evaluación. El lanzador deja hasta 7,5 h antes de iniciar otra corrida, sin cortar la activa, y vigila 6 GiB libres. En el SSD APFS dentro del contenedor exFAT hay 64 GiB de capacidad lógica.

## Validación y lectura fijadas

`val_exp052_select` sirve **solo para elegir época**. `val_exp052_audit` y su vista cruda están bloqueados hasta que terminen las 20 corridas; `scripts/auditar_tanda_nocturna_01.py` se niega a abrirlos antes y no consulta `test_demo`. El auditor comprueba hashes de las fuentes, pesos y recetas; guarda predicciones y score/acierto por grupo, tasa y clase. Cada cambio se compara con `ref` de **igual arquitectura y semilla** en las mismas 5.000 señales. Primero revisar score simulado promedio de ambas semillas; después crudo, A/B/C y aciertos 6/7. Una ganancia de 6 a costa de 7, o una pérdida marcada de B, invalida una lectura de mejora robusta aunque suba el agregado.

El bloque de auditoría y el simulador ya fueron estudiados históricamente; además crudo/simulado reutilizan las mismas señales. Veinte comparaciones aumentan la probabilidad de un ganador casual. **Esta tanda es exploratoria**, no una estimación independiente del examen. No elegir un único ganador por décimas ni repetir variantes ajustadas contra este bloque. Las combinaciones que muestren una señal coherente en dos semillas deberán comprobarse con otra fuente o un protocolo nuevo. La CV A/B/C anterior no se vuelve a consultar para seleccionar estas condiciones.

## Ejecutar desde el proyecto del SSD

Con la imagen APFS montada en `/Volumes/concurso_ml_ssd`, desde `gdg-uam-ml-contest-2026/`:

```bash
.venv/bin/python scripts/tanda_nocturna_01.py --listar
caffeinate -dimsu .venv/bin/python scripts/tanda_nocturna_01.py --max-horas 7.5 --min-libre-gib 6
```

El lanzador salta solo corridas **completas y con la receta correcta**. Si encuentra una carpeta parcial, peso cambiado, falta de disco o fallo del evaluador, se detiene sin borrar ni repetir; conservar la salida y revisarla. Cada corrida deja `models/n01_*.pt`, `experiments/n01_*/` y `results/n01_*/` para el dashboard. El lanzador no abre la auditoría ni el demo.

Al terminar las 20, avisar al asistente para ejecutar:

```bash
.venv/bin/python scripts/auditar_tanda_nocturna_01.py
```

Se verificaron `py_compile`, el listado de 20 IDs pendientes y dos humos de 80 señales/1 época del wrapper, uno con `gdrop` en dos rutas y otro con `nog` en base; entrenamiento y harness coincidieron en ambos. Los artefactos de humo se eliminaron. No se lanzó ninguno de los entrenamientos largos.

## Resultado de la auditoría (6 nov 2026)

David ejecutó las 20 corridas en el SSD; el lanzador verificó que todas acabaron con pesos, receta e informes coherentes. Tiempo de ajuste acumulado **5,68 h**. El auditor abrió las 5.000 señales reservadas una vez al terminar toda la tanda, sin consultar `test_demo`, y guardó `experiments/tanda_nocturna_01_auditoria.json` y `data/tanda_nocturna_01_pred.npz` en el SSD. Las dos vistas usan las mismas señales base.

| Arquitectura | Condición | Score simulado medio | Δ frente a referencia, semilla 26 / 27 | Δ medio simulado | Δ medio crudo |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Referencia | **16,192446** | — | — | — |
| Base | Sin grupo | 15,821717 | −0,646761 / −0,094697 | −0,370729 | −0,135858 |
| Base | Dropout grupo | 16,135246 | −0,364105 / +0,249705 | −0,057200 | +0,067186 |
| Base | Mezclas suaves | 15,750756 | −0,887716 / +0,004335 | −0,441690 | −0,108997 |
| Base | Ruido suave | 15,384398 | −1,210464 / −0,405633 | −0,808048 | −0,356030 |
| Dos rutas | Referencia | **15,706780** | — | — | — |
| Dos rutas | Sin grupo | 15,572106 | +0,307360 / −0,576708 | −0,134674 | +0,100735 |
| Dos rutas | Dropout grupo | 15,744258 | +0,468933 / −0,393977 | +0,037478 | +0,219801 |
| Dos rutas | Mezclas suaves | 15,609522 | −0,127367 / −0,067149 | −0,097258 | +0,135118 |
| Dos rutas | Ruido suave | 15,149976 | −0,418512 / −0,695096 | −0,556804 | −0,089581 |

**Ningún cambio mejora el score simulado de ambas semillas en ambas arquitecturas.** Reducir ruido, huecos y picos pierde en las cuatro comparaciones simuladas; reducir las mezclas tampoco ofrece señal estable. El dropout de grupo es casi neutro en media pero muy sensible a semilla. En B recupera aciertos de clase 7 frente a su referencia en las cuatro comparaciones: base **+8/+13**, dos rutas **+9/+13**, y reduce errores 7→6 en B en 8–10 casos cada vez. Sin embargo, a menudo cede clase 6: por ejemplo, base semilla 2026 pierde 23 aciertos de clase 6 en todo el bloque al ganar solo uno de clase 7. Es una pista sobre la dependencia del metadato de grupo, no una mejora de la frontera 6/7.

Las referencias de 90k muestran base **16,192446** frente a dos rutas **15,706780** de score simulado medio; base gana en A y C con ambas semillas. La CV anterior, con ~60k y grupo reservado, favorecía dos rutas en A y C. Los protocolos, datos de entrenamiento y uso de grupo son distintos, así que esta inversión **no identifica por sí sola** la causa. La lectura útil es que la ventaja OOF de dos rutas no se transporta automáticamente al ajuste con 90k.

El bloque de auditoría ya había sido consultado históricamente y 20 variantes multiplican las comparaciones. No elegir el mayor score de la tabla como nuevo modelo ni probar un pequeño ajuste basado en este resultado; necesitamos otra fuente de validación o un mecanismo verificable en train. El demo permanece sin consultar en esta ronda.
