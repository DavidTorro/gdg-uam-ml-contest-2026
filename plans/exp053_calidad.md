# exp053: dos rutas para mejorar la representación y la diversidad

Preparado y evaluado el 5 de noviembre de 2026. **Exp053 mejora la referencia de calidad al combinarse**, aunque su resultado individual es peor. El objetivo actual es maximizar calidad predictiva, sin tomar decisiones de entrega o velocidad en cada ronda.

## Evidencia que motiva la prueba

La media uniforme de probabilidades de exp033, exp034 y exp051, con sus sesgos históricos, mejora los cuatro conjuntos examinados frente a exp033: demo **3,778906 → 4,032922** (84,50 → 85,75 %), auditoría simulada **15,284593 → 16,897500** (88,16 → 90,10 %), val10k **23,557647 → 25,326568** y pseudo **48,561543 → 53,765617**. En demo, clases 6/7 pasan de **32/32 a 34/35 de 50** y la AUC 6-vs-7 de **0,760 a 0,819**. El IC95 bootstrap pareado del delta de score demo es **[−0,146,+0,686]**: mejora observada, incertidumbre amplia y selección histórica sobre el mismo demo. El informe reproducible es `experiments/ensamble_033_034_051.json` y el script `scripts/evaluar_ensamble_calidad.py`.

Exp052 mostró que repetir ejemplos de clase 6 con pistas menos espurias desplazó la decisión hacia la clase 7: en auditoría, la AUC 6-vs-7 bajó **0,901 → 0,892**. Un clasificador logístico regularizado sobre los tres márgenes 6−7 del ensamble, entrenado fuera de pliegue con el demo, cambió tres predicciones y empeoró **4,032922 → 3,981209**. Esto orienta la siguiente prueba hacia **nueva representación complementaria**, no hacia otro umbral o remuestreo de clase 6.

## Intervención única

`src/red_dos_rutas.py` añade dos rutas paralelas: la conjunta replica la mezcla temprana de canales de exp033; la segunda mantiene filtros por canal hasta el resumen temporal. La cabeza recibe ambos resúmenes y los 17 auxiliares. Tiene **134.312 parámetros** frente a 55.656 de exp033. Exp044, que usaba solo convoluciones separadas por canal, cayó mucho; esta red conserva una ruta conjunta completa para no obligar a depender solo de formas por canal. No modifica el simulador v2 ni los datos de train. La hipótesis es que la ruta por canal capture información complementaria sin perder relaciones útiles entre señales.

El control para elegir época será `data/val_exp052_select.npz` (5.000 señales); `data/val_exp052_audit.npz` (otras 5.000 señales base) queda fuera de la elección de época. Estos datos ya se han consultado en investigaciones previas, así que la reserva reduce una fuga directa de checkpoint pero no constituye un test totalmente nuevo. El demo se evalúa una sola vez con la regla de ensamble fijada antes de entrenar exp053. Pseudo deriva de val10k y no es una confirmación independiente.

## Entrenar

Desde `gdg-uam-ml-contest-2026`:

```bash
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id exp053 --arquitectura cnn --red dos_rutas --muestras 90000 --epocas 200 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.002 --semilla 2026 --runs 1 --min-libre-gib 6
```

Se comprobó la forma de salida, gradientes y un humo completo del wrapper con 80 muestras y una época; los artefactos de humo fueron retirados. Los resultados del wrapper sobre `val_exp052_select` son control de funcionamiento y elección de época, no validación independiente.

## Evaluar al terminar

```bash
.venv/bin/python scripts/auditar_exp053_calidad.py
```

El script compara exp053 individual con el ensamble triple fijo y con el cuádruple **uniforme** exp033+exp034+exp051+exp053; no aprende pesos con el demo. Reporta score al mismo número de ejemplos, acierto por clase e intervalos pareados en auditoría y demo, más val10k y pseudo. Un buen resultado sería una mejora del cuádruple que se sostenga en auditoría y demo, particularmente sin intercambiar de nuevo clases 6/7. Como la arquitectura se diseñó tras muchas consultas al demo, incluso ese resultado requeriría cautela antes de generalizar al examen oculto.

## Resultado

David completó **200 épocas**, mejor época **176**, **1.182 s** de entrenamiento, control de elección de época en `val_exp052_select` **16,411252**. La configuración confirma 90.000 muestras, simulador v2, `--red dos_rutas`, sin selección de índices y sin sesgo adicional. El informe `experiments/exp053_auditoria_calidad.json` compara la regla fija de 1/3 por modelo con la de 1/4 por modelo:

| Conjunto | Triple → cuádruple: score | Acierto | Clases 6/7: triple → cuádruple | Exp053 solo: score |
| --- | ---: | ---: | ---: | ---: |
| Demo 400 | **4,032922 → 4,263524** | 85,75 % → 87,00 % | 34/35 → 33/38 de 50 | 3,371718 |
| Auditoría simulada 5k | **16,897500 → 17,204961** | 90,10 % → 90,48 % | 421/368 → 419/377 | 15,464010 |
| Auditoría cruda 5k | **17,913563 → 18,008472** | 91,52 % → 91,64 % | 428/384 → 428/386 | 16,515543 |
| Val10k | **25,326568 → 25,600132** | 91,47 % → 91,69 % | 862/773 → 856/793 | 23,726042 |
| Pseudo 50k | **53,765617 → 54,651028** | 89,63 % → 90,00 % | 5174/4669 → 5132/4828 | 49,765645 |

IC95 bootstrap pareado del delta de score cuádruple−triple: demo **[−0,063,+0,548]** y auditoría simulada **[−0,006,+0,631]**. Ambas incluyen cero por poco o mucho, así que la mejora incremental es prometedora pero todavía incierta. El modelo individual exp053 aporta diversidad de errores; la media uniforme mejora en los cinco controles, con una ligera cesión de clase 6 a la 7. No ajustar pesos del ensamble con las mismas 400 etiquetas demo.

Como comprobación exploratoria posterior, se añadió exp043 con el mismo peso a los cuatro modelos **solo en las dos vistas de auditoría**. El score simulado bajó **17,204961 → 17,103269**, aunque el crudo subió **18,008472 → 18,198524**. No se miró el demo de ese quinto ensamble; no hay evidencia consistente para ampliarlo así.

El oráculo retrospectivo que elige, usando la etiqueta verdadera, cualquier predicción correcta de los cuatro modelos alcanzaría **373/400** en demo, frente a **348/400** del cuádruple: 27 ejemplos fallan en todos y 25 podrían corregirse en teoría por una regla de combinación perfecta. En auditoría simulada los números son **4786/5000** y **4524/5000**. Un stacker multinomial logístico regularizado (`C=0,05`) sobre las 32 probabilidades, evaluado fuera de pliegue en las 400 muestras demo, bajó a **3,986215 / 85,5 %** frente a **4,263524 / 87,0 %** del promedio fijo; arregló 4 fallos y rompió 10 aciertos. No ajustar otro voto supervisado sobre el mismo demo: el techo oracular no demuestra que pueda aprenderse una regla generalizable con 400 etiquetas.
