# CV por grupos A/B/C: comprobar generalización antes de otra arquitectura

Preparada el 5 de noviembre de 2026. Objetivo: medir si la ventaja de la red de dos rutas frente a la CNN base resiste un cambio de grupo de origen, con reglas fijadas antes de ver los resultados de estos nuevos pesos. Es una **prueba de estrés estructural**, no una estimación independiente del examen oculto.

## Diseño congelado

Hay tres pliegues. En el pliegue A se entrenan las señales de `train_90k` de B y C, se elige época con `val_examen_v2` de B y C, y se audita solo en las señales A de `val_10k`; análogamente para B y C. `sample_index` de train y val son disjuntos. La auditoría se calcula sobre las mismas señales base en dos vistas, cruda y simulada; **no son dos pruebas independientes**. El entrenamiento oculta las tres auxiliares que identifican A/B/C (`--sin-grupo`) para que el grupo reservado no aparezca como una categoría one-hot jamás entrenada. Se conservan la tasa de muestreo y las demás auxiliares de la receta.

| Grupo reservado | Train de los otros dos | Selección de época | Auditoría | Clase 6/7 en auditoría |
| --- | ---: | ---: | ---: | ---: |
| A | 60.053 | 6.610 | 3.390 | 343 / 312 |
| B | 59.894 | 6.663 | 3.337 | 338 / 334 |
| C | 60.053 | 6.727 | 3.273 | 331 / 341 |

`scripts/preparar_cv_grupos.py` ya generó los índices de entrenamiento y auditoría, los tres controles de época, y `experiments/cv_grupos_v1_preparacion.json` con SHA256, tamaños y clases. El script falla si se intenta regenerar encima. No abre `test_demo`. Los ficheros de `data/` y `experiments/` son locales y están ignorados por Git.

La comparación requiere **reentrenar ambas arquitecturas en los mismos pliegues**. Los pesos exp033 y exp053 originales vieron los tres grupos, así que evaluarlos en un grupo reservado no sería una CV válida. Las seis ejecuciones se fijan a 200 épocas, semilla 2026, LR 0,002, batch 256, simulador v2, `--sin-atajos`, `--sin-grupo`, equilibrio de clases y elección por score en el control del pliegue. No se modifica `model.py` ni se mide latencia.

## Entrenar — David ejecuta con `caffeinate`

Desde `gdg-uam-ml-contest-2026/`, **un solo comando ejecuta los seis entrenamientos de forma secuencial** con `caffeinate`:

```bash
.venv/bin/python scripts/entrenar_cv_grupos.py
```

El lanzador comprueba los hashes del manifiesto antes de empezar, salta pesos ya completos y verificados, y se detiene ante un entrenamiento parcial sin borrar archivos. Se puede consultar el estado sin entrenar con `.venv/bin/python scripts/entrenar_cv_grupos.py --listar`. Si una corrida queda parcial, revisar su carpeta y peso antes de volver a lanzar; **no** se repite automáticamente. Al terminar, avisar para que el asistente ejecute la auditoría pareada.

Los comandos individuales equivalentes quedan aquí como referencia por si se prefiere lanzar un pliegue concreto. Cada ID es único.

```bash
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_base_a --red base --indices-entrenamiento data/cv_grupos_v1_sin_a_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_a_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_base_b --red base --indices-entrenamiento data/cv_grupos_v1_sin_b_train.npy --muestras 59894 --archivo-examen data/cv_grupos_v1_sin_b_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_base_c --red base --indices-entrenamiento data/cv_grupos_v1_sin_c_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_c_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_dos_a --red dos_rutas --indices-entrenamiento data/cv_grupos_v1_sin_a_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_a_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_dos_b --red dos_rutas --indices-entrenamiento data/cv_grupos_v1_sin_b_train.npy --muestras 59894 --archivo-examen data/cv_grupos_v1_sin_b_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id cvg_dos_c --red dos_rutas --indices-entrenamiento data/cv_grupos_v1_sin_c_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_c_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
```

## Auditar después de los seis pesos

```bash
.venv/bin/python scripts/auditar_cv_grupos.py --nombre base --ids cvg_base_a cvg_base_b cvg_base_c
.venv/bin/python scripts/auditar_cv_grupos.py --nombre dos_rutas --ids cvg_dos_a cvg_dos_b cvg_dos_c --referencia base
```

El auditor comprueba hashes, selección de época, 200 épocas, arquitectura y `sin_grupo`; guarda predicciones fuera de pliegue y reporta score a igual número de muestras, accuracy, confusión y aciertos por clase en cada grupo y en las 10.000 señales reunidas. La segunda ejecución añade deltas pareados e IC95 por remuestreo dentro de cada combinación de grupo y clase. Ese IC describe variación de ejemplos **dentro de estos tres grupos**; no mide incertidumbre sobre grupos nuevos.

Lectura fijada: usar como métrica primaria el cambio de score **OOF simulado** de dos rutas frente a base, acompañado de los tres cambios por grupo. Revisar score OOF crudo y clases 6 y 7 para detectar ganancias por intercambio de aciertos. Una ventaja global que proceda de un único grupo, o que aumente 7 a costa de 6, será evidencia débil. No elegir épocas, pesos de ensamble ni una arquitectura nueva a partir de `test_demo` durante esta ronda. Tampoco repetir barridos sobre estos mismos tres pliegues; si el resultado es ambiguo, reconocer la limitación y buscar una fuente de validación adicional.

## Alcance y limitaciones

`val_10k` pertenece a la misma colección de 100.000 señales que `train_90k`; el grupo reservado se mantiene fuera del ajuste y de la elección de época de **estos nuevos pesos**, pero la colección y el simulador se han estudiado antes. El simulador v2 fue calibrado históricamente con train y demo. Por ello esta CV añade un cambio estructural controlado, no un examen externo limpio. Las etiquetas del demo siguen sin intervenir en esta prueba.

Se verificó una época de humo con 80 señales y el control del pliegue A: el peso y el informe registraron `sin_grupo=true` y el archivo de selección correcto. Los artefactos de humo se retiraron. David ejecuta los entrenamientos largos y gestiona Git.

## Resultado de las seis ejecuciones (5 nov 2026)

David completó los seis entrenamientos. `scripts/entrenar_cv_grupos.py --listar` verificó pesos, recetas y hashes; `scripts/auditar_cv_grupos.py` generó `experiments/cv_grupos_v1_base.json` y `experiments/cv_grupos_v1_dos_rutas.json`, con predicciones OOF locales. No se consultó `test_demo` para esta comparación.

| Vista OOF, 10.000 señales | CNN base | Dos rutas | Cambio | IC95 pareado del cambio |
| --- | ---: | ---: | ---: | ---: |
| Simulada, score | 13,996515 | **14,682496** | **+0,685980** | [+0,182375, +1,213249] |
| Simulada, acierto | 77,60 % | **78,66 %** | +1,06 p.p. | — |
| Cruda, score | 15,611416 | **16,317635** | **+0,706219** | [+0,195706, +1,232063] |
| Cruda, acierto | 79,96 % | **80,96 %** | +1,00 p.p. | — |

| Grupo reservado | Score simulado base → dos rutas | Cambio simulado | Cambio crudo | Acierto simulado base → dos rutas |
| --- | ---: | ---: | ---: | ---: |
| A | 5,748853 → 6,403208 | +0,654356 | +0,667500 | 70,94 % → 73,04 % |
| B | 12,357832 → 11,622825 | **−0,735007** | −0,159373 | 87,62 % → 86,34 % |
| C | 6,802430 → 7,634143 | +0,831713 | +0,579814 | 74,27 % → 76,66 % |

En la vista simulada, dos rutas cambia los aciertos de clase 6 de **678/1012 a 677/1012** y los de clase 7 de **684/987 a 683/987**: no resuelve la confusión principal. En la vista cruda también bajan ligeramente, 6 de 702 a 696 y 7 de 721 a 718. El aumento de aciertos simulado procede sobre todo de las clases 0 (+32), 2 (+33) y 3 (+27). B pierde 30 aciertos de clase 7 (272/334 → 242/334) mientras gana 11 de clase 6 (243/338 → 254/338); sus errores 7→6 suben de 32 a 56. Por tasa de muestreo, las mayores ganancias de acierto simulado se concentran en A/tasa 1 (+4,88 p.p.) y C/tasa 2 (+4,82 p.p.); B baja en ambas tasas.

**Lectura:** dos rutas aporta una mejora agregada en esta prueba estructural fijada de antemano, con IC descriptivo positivo dentro de estos grupos. No es una mejora uniforme y no corrige 6/7. El score absoluto de estos pesos entrenados con ~60k ejemplos no se compara con el ensamble de modelos de 90k. Los tres grupos pertenecen a la colección ya estudiada; el simulador también se calibró históricamente. El IC remuestrea ejemplos dentro de A/B/C y no da cobertura para grupos nuevos. Crudo y simulado son dos vistas de las mismas señales. No convertir estos tres pliegues en un barrido de nuevas variantes; investigar con train la causa del retroceso B y buscar validación adicional antes de concluir que una nueva arquitectura generaliza al examen.
