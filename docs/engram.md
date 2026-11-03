# Memoria del concurso GDG UAM ML 2026

Actualizado el 3 de noviembre de 2026. Datos de evaluación locales; `data/`, `results/` y `experiments/` están excluidos de Git. David hace los commits y ejecuta los entrenamientos largos. El modelo de entrega sigue intacto: el único cambio de inferencia probado en esta ronda se revirtió.

## Estado y mejor modelo

- Rama `feat/analisis-datos`, árbol limpio al iniciar la auditoría; último commit `8469b8b`.
- Entrega actual: `model.py` carga `models/exp033.sesgo.npz`. CNN exp033 con sesgo +0,9414 a la clase 6, exportada a NumPy. Dependencias de entrega: numpy y scikit-learn; inferencia de este peso solo usa NumPy.
- `test_demo` (400, 50 por clase): score **3,778906**, accuracy **84,5 %**, clases 6 y 7 **32/50 cada una**, con 15 confusiones 6→7 y 15 de 7→6. La última evaluación de esta ronda midió p50 **0,0893 ms** y p95 **0,1004 ms**; la medida histórica `results/demo_exp033_numpy` fue p50 0,1102 y p95 0,2912 ms. La latencia depende del entorno y carga de cada ejecución.
- En `val_examen_v2`, exp033 sin sesgo obtuvo 23,032341 y 89,1 % (10 000 ejemplos). No comparar scores brutos entre conjuntos de tamaños distintos.
- El detalle de `exp001`–`exp039` está en `README.md` y `results/<id>/model/run_*.json`. `results/leaderboard.csv` no existe en la raíz: cada evaluación tiene el suyo.

## Qué se probó y descartó antes

- HistGradientBoosting con características (`exp001`–`exp014`): el mejor en demo fue exp013, 0,255648 y 44,25 %, clase 6 a 0/50. Baseline oficial: 0,195745 y 42 %.
- CNN inicial (`exp015`–`exp020`): exp016 llegó a 1,798111 y 74,25 % en demo, pero clase 6 a 4/50. Más épocas o ruido mejoraron a veces `val_10k` sin mejorar demo; no usar `val_10k` como único criterio.
- Reajuste de la capa final con demo (`exp021`): descartado por empeorar en validación cruzada. Simulador v1 (`exp022`) dejó huellas de clase; v2 aplica mezclas de trampas a todas las clases.
- CNN con simulador v2, sin atajos de media/`level`, clases equilibradas y época elegida en validación simulada (`exp023`–`exp039`): exp033 es el mejor medido en demo. Exp034 con otra semilla obtuvo 3,404803 con sesgo; redes más anchas pierden velocidad sin ganancia suficiente. Exp036–039, con más épocas/otro LR, se estancaron cerca de 23 en `val_examen_v2`.
- Exportación NumPy: mismas clases que PyTorch en 400 ejemplos demo y 3000 de validación, ~25 % más rápida. No requiere PyTorch en la entrega.

## Hipótesis auditadas en esta ronda

| Hipótesis | Evidencia y veredicto |
| --- | --- |
| La clase es función de `(group, sample_rate)` | **Falsa**: las ocho clases aparecen en las seis combinaciones. Regla de mayoría por combinación: 15,99 % en train, 18,5 % en demo. |
| `level` es un atajo útil que generaliza | **Falsa como atajo estable**: medias por clase 3/5 de aprox. +1/−1 en train; en demo aprox. −0,33/+0,06. Un HistGradientBoostingClassifier de solo `(group, sample_rate, level)` (70 iteraciones, 15 hojas, semilla 2026) acertó 20,73 % en train, 20,15 % en `val_10k` y **12,5 %** en demo. La CNN actual anula `level`. No se ha descartado una interacción débil de metadatos con la forma. |
| `sample_index` del demo continúa train | **Falsa**: demo usa índices 0–502, 400 únicos, no monótonos; coinciden numéricamente con train pero 350/400 etiquetas difieren. Son índices locales de dos flujos. |
| Un afín por canal o un z-score basta para igualar train y demo | **Falsa como explicación completa**: con 400 train equilibrados y 400 demo, un clasificador de origen con validación cruzada de 5 pliegues logra AUC 0,849 en rasgos brutos y 0,833 en rasgos de forma tras z-score por canal. NaN 0,776 % vs 1,380 %, energía alta mediana 0,0686 vs 0,1196. Es evidencia de desplazamiento residual; no identifica una causa única. |
| Pseudo-test simulado equivale al test real | **No demostrado**: AUC de origen pseudo vs demo 0,575 con rasgos brutos y 0,577 tras z-score, mejor parecido global. El simulador se calibró con el demo y la clase 6 sigue siendo demasiado fácil en el pseudo; no usarlo solo para seleccionar cambios de 6↔7. |

El diagnóstico reproducible está en `scripts/diagnosticar_desplazamiento.py` y los números completos en `experiments/auditoria_desplazamiento.json`. Solo hay 400 ejemplos de demo; las AUC son descriptivas y pueden variar con otra selección de train.

## Pseudo-test y evaluaciones de esta ronda

`scripts/crear_pseudo_test.py` generó `data/pseudo_test.npz`: 50 000 ejemplos, 6250 por clase, mediante `simular_examen(mezclas=False)` sobre 9931 señales únicas de `data/val_10k.npz`, semilla 20261103. El informe de procedencia está en `experiments/pseudo_test.json`. Reutiliza señales y el simulador se calibró con el demo: no equivale a 50 000 muestras independientes. `data/val.npz` no existe; se usó `data/val_10k.npz`.

| Conjunto y resultado | N | Score | Accuracy | Peor clase | p50 / p95 |
| --- | ---: | ---: | ---: | --- | --- |
| `results/auditoria_base_pseudo` | 50 000 | 48,561543 | 87,45 % | 7: 64,06 % | 0,0924 / 0,1163 ms |
| `results/auditoria_base_demo` | 400 | 3,778906 | 84,50 % | 6 y 7: 64,00 % | 0,0893 / 0,1004 ms |
| `results/auditoria_base_val10k` | 10 000 | 23,557647 | 89,86 % | 7: 70,82 % | 0,0857 / 0,1022 ms |

Accuracy por clase (0–7):

- Pseudo: 77,84 %, 99,81 %, 96,11 %, 86,38 %, 94,98 %, 95,42 %, **84,99 %**, **64,06 %**. Confusiones 6→7: 724; 7→6: 1740.
- Demo: 78 %, 100 %, 94 %, 82 %, 98 %, 96 %, **64 %**, **64 %**. Confusiones 6→7: 15; 7→6: 15.
- Val 10k: 79,05 %, 99,93 %, 98,49 %, 87,74 %, 95,91 %, 96,79 %, **86,66 %**, **70,82 %**. Confusiones 6→7: 109; 7→6: 231.

No hay un cambio de modelo que haya superado el 3,778906 del demo en esta ronda, así que se conserva el peso de entrega. La diferencia de puntuación entre pseudo y demo se debe también al número de muestras; comparar accuracy por clase y matrices, no los scores brutos.

## Siguiente paso

Revisar específicamente el desplazamiento condicional de las clases 6 y 7 en los canales y la posición de los huecos, con selección de cambios basada en validación simulada y una comprobación prudente en demo. Cualquier cambio de inferencia debe medirse con el harness en pseudo, demo y `val_10k`, guardando resultados separados; revertirlo si no supera el score 3,778906 del demo (desempate por latencia). Evitar múltiples ajustes guiados por las 50 etiquetas de cada clase del demo.

### Seguimiento: origen del error 6↔7 y prueba descartada

Sobre muestras de clase 6, las medianas train / pseudo / demo son: correlación entre canales 1 y 2 **0,472 / 0,408 / 0,148**; desviación del canal 1 respecto al 0 **1,276 / 1,317 / 1,027**; canal 2 respecto al 0 **0,895 / 0,983 / 0,583**. En clase 7, las mismas medianas de correlación son 0,148 / 0,128 / 0,169 y de canal 2 respecto al 0, 0,566 / 0,574 / 0,611. El pseudo retiene una pista de entrenamiento de la clase 6 que el demo casi no tiene. Estas medianas son descriptivas (50 ejemplos demo por clase), no una prueba causal de lo que la CNN usa.

Se probó **un solo cambio de inferencia** en `src/red_numpy.py`: anular los cuatro auxiliares de tamaño de canal (posiciones 4:8) para pesos con `sin_atajos=True`. Resultados antes → después:

| Conjunto | Score | Accuracy | Clase 6 | Clase 7 | p50 / p95 después |
| --- | --- | --- | --- | --- | --- |
| Pseudo 50k | 48,561543 → 45,474296 | 87,45 % → 86,386 % | 84,99 % → 83,60 % | 64,06 % → 54,94 % | 0,0961 / 0,1259 ms |
| Demo 400 | 3,778906 → 3,567071 | 84,5 % → 83,5 % | 64 % → 60 % | 64 % → 60 % | 0,0998 / 0,1301 ms |
| Val 10k | 23,557647 → 22,649515 | 89,86 % → 89,48 % | 86,66 % → 85,87 % | 70,82 % → 63,53 % | 0,0842 / 0,0965 ms |

**Descartado y revertido.** JSON en `results/prueba_sin_escala_{pseudo,demo,val10k}/model/run_0.json`. La ablación de un modelo ya entrenado no demuestra que entrenar desde cero sin estos auxiliares vaya a fallar.

Una prueba corta con `simular_examen(mezclas=True)` sobre 900 ejemplos de validación por clase tampoco arregló el pseudo: en clase 6 la correlación 1–2 cambió de 0,410 a 0,372 (demo 0,148) y la razón de desviaciones canal 2/0 de 0,994 a 1,116 (demo 0,583). No se generó un nuevo pseudo-test con esas mezclas.

**Próximo experimento útil:** diseñar una validación de estrés de 6↔7 que quite la componente común espuria y las diferencias de tamaño de canal de la clase 6, sin usarla como estimador de nota oficial. Solo después merece la pena entrenar una CNN que aprenda a ignorar estas pistas; usar el demo una sola vez como comprobación final del candidato.

### Seguimiento: validación de estrés y experimento preparado

`scripts/crear_validacion_estres.py` creó tres variantes de `val_examen_v2` que modifican **solo la clase 6** usando estadísticas de la clase 7 de la propia validación (sin leer demo). Quitan por separado la correlación entre canales 1–2, la diferencia de escala de canales 1–2 respecto al 0, o ambas. Los valores medidos cambiaron hasta los de referencia, pero la accuracy de la clase 6 apenas se movió: **85,47 % original; 84,58 % sin correlación; 85,28 % sin escala; 84,19 % sin ambas**. Scores respectivos **21,87671; 21,77934; 21,85514; 21,73582**. Resultados en `results/estres_6_{original,correlacion,escala,ambas}/model/run_0.json`. **Veredicto:** esos rasgos separan train y demo, pero no parecen ser la pista principal que usa exp033; no entrenar aún una corrección específica para ellos. Son intervenciones artificiales, no pruebas del test real.

`scripts/diagnosticar_canales_67.py` ocultó un canal normalizado cada vez. En demo, ocultar el canal 0 llevó el acierto de la clase 7 de 64 % a 0 %; ocultar el canal 1 llevó el de la clase 6 de 64 % a 0 %. Ocultar el 2 o el 3 casi no cambió 6/7. Estos enmascaramientos crean entradas fuera de distribución; orientan la búsqueda hacia los canales 0 y 1, sin probar causalidad. JSON en `experiments/diagnostico_canales_67.json`.

En un análisis de 20 rasgos normalizados, la separación 6↔7 por la energía de alta frecuencia del **canal 0** (d de Cohen) fue **+0,67 en pseudo**, pero **−0,38 en demo**; rugosidad **+0,68 → −0,05**; autocorrelación de retardo 1 **−0,65 → +0,30**. El cambio de signo con solo 50 ejemplos demo por clase exige prudencia, pero explica una hipótesis concreta para el siguiente entrenamiento.

Se añadió al simulador una perturbación **opcional** del ritmo del canal 0, igual para todas las clases, con probabilidad 0 por defecto (preserva v2 y exp033). Con `prob_ritmo_0=0.7, ritmo_0_max=2.4`, la separación alta frecuencia 6↔7 en 1800 muestras simuladas bajó de **d=0,642 a d=0,065**. `scripts/crear_val_ritmo0.py` generó `data/val_estres_ritmo0.npz` (10k, base `val_10k`), que no representa el test: exp033 bajó de **21,87671 / 88,28 %** en `val_examen_v2` a **10,77294 / 72,47 %** en este estrés. Por clase: 6, 85,47 % → 87,55 %; 7, 65,35 % → 49,14 %. JSON en `results/estres_ritmo0_base/model/run_0.json`. Esta dureza aconseja juzgar el entrenamiento nuevo también en demo y no elegirlo solo por el estrés.

Quedó preparado **un único experimento** en `plans/ritmo_canal0.json`: misma receta que exp033 (ancho 1, 200 épocas, lr 0,002, semilla 2026) y solo ese ajuste del simulador. `scripts/barrido.py --plan plans/ritmo_canal0.json --solo-mostrar` identifica el próximo ID libre como **exp040**, unos **18 min**. David lanza el entrenamiento largo. Después, evaluar exp040 en `val_estres_ritmo0`, pseudo, demo y `val_10k` con el harness, comparar score del demo con 3,778906 y conservar el peso actual si no supera esa marca. Hasta entonces `model.py` y sus pesos siguen intactos.

### Resultado de exp040 — descartado para la entrega

David entrenó exp040 con el plan anterior: mejor época 172, 1756 s de entrenamiento. `val_examen_v2` sin sesgo: **22,625807**, 88,77 %, frente a exp033 sin sesgo **23,032341**, 89,1 %. El sesgo no supervisado para la clase 6 fue **+0,9007**, calculado sobre las predicciones del demo sin leer etiquetas. `scripts/exportar_numpy.py` verificó **100 % de clases iguales** frente a PyTorch en 400 muestras demo y 3000 de `val_examen_v2`; se conserva `models/exp040.sesgo.npz` solo como artefacto histórico. `model.py` continúa apuntando a exp033.

| Fuente | Score exp033 → exp040 | Accuracy exp033 → exp040 | Clase 6 | Clase 7 | p50/p95 exp040 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demo 400 | **3,778906 → 3,520105** | 84,5 % → 83,25 % | 64 % → 66 % | 64 % → 56 % | 0,1042 / 0,1397 ms |
| Pseudo 50k | 48,561543 → 47,677435 | 87,45 % → 87,034 % | 84,99 % → 82,96 % | 64,06 % → 63,30 % | 0,0968 / 0,1268 ms |
| Val 10k | 23,557647 → 22,866864 | 89,86 % → 89,39 % | 86,66 % → 85,28 % | 70,82 % → 66,77 % | 0,0963 / 0,1353 ms |
| Estrés ritmo 0 | 10,772943 → **19,642214** | 72,47 % → 85,75 % | 87,65 % → 81,92 % | 49,14 % → 61,50 % | 0,0978 / 0,1243 ms |

JSON en `results/exp040_sesgo_{demo,pseudo,val10k,estres}/model/run_0.json`. La variante aprende resistencia a la perturbación artificial del canal 0, pero empeora el demo, la validación ordinaria y el pseudo. **Veredicto: descartar exp040 como candidato a entrega; conservar exp033 por score y latencia.** No se ha modificado el peso por defecto. El siguiente intento debe buscar una señal 6↔7 que se conserve en el demo o mejorar la evaluación del desplazamiento, evitando más ajustes al mismo conjunto de 400 etiquetas.

### Optimización ganadora de inferencia NumPy

Se sustituyó `numpy.lib.stride_tricks.as_strided` por una vista `np.ndarray(..., buffer=..., strides=...)` en `src/red_numpy.py`. La forma, límites y pasos de la ventana son los mismos; la construcción evita comprobaciones repetidas cuatro veces por muestra. Frente al código de `HEAD`, los logits fueron **idénticos bit a bit** en 400 ejemplos demo y 3000 de `val_10k` (0 clases distintas). No cambia `model.py` ni los pesos `exp033`.

| Fuente | Score antes → después | Accuracy | Media ms antes → después | p50 ms antes → después | p95 ms antes → después |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pseudo 50k | 48,561543 → 48,561543 | 87,45 % | **0,0958 → 0,0872** | 0,0924 → 0,0867 | 0,1163 → 0,0984 |
| Demo 400 | 3,778906 → 3,778906 | 84,5 % | 0,0899 → 0,0902 | 0,0893 → 0,0888 | 0,1004 → 0,1061 |
| Val 10k | 23,557647 → 23,557647 | 89,86 % | **0,0877 → 0,0822** | 0,0857 → 0,0823 | 0,1022 → 0,0944 |

JSON posteriores en `results/opt_ventanas_{pseudo,demo,val10k}/model/run_0.json`; anteriores en `results/auditoria_base_{pseudo,demo,val10k}/model/run_0.json`. La muestra demo es demasiado corta para una conclusión estable de latencia. Una comparación alternada en el mismo proceso sobre 400 ejemplos repetidos 10 veces dio, en cuatro pares, 0,0874→0,0815; 0,0830→0,0766; 0,0839→0,0807; 0,0878→0,0839 ms. **Conservar el cambio** por identidad de predicciones y mejora repetida de latencia en datos grandes.
