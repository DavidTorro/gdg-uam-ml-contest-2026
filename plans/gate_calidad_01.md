# Gate de calidad 01: una prueba de representación condicionada

Preparado el 6 nov 2026. David ejecuta los entrenamientos largos con `caffeinate` y gestiona Git. Esta ronda no lee etiquetas ni señales del demo durante ajuste, elección de época o auditoría.

## Hipótesis falsable

Las rutas conjunta y por canal cometen errores complementarios, pero la concatenación fija de `exp053` no aprende a elegir entre ellas cuando cambia la calidad de la señal. `RedDosRutasGate` da una cabeza a cada ruta y mezcla sus logits con una puerta que ve solo fracción de huecos, rugosidad normalizada y tasa de muestreo. Ambas cabezas reciben pérdida supervisada adicional fija de 0,25 cada una para que la puerta no inutilice ninguna. La puerta empieza exactamente a 0,5. No recibe etiqueta, clase predicha ni logits como entrada. El resultado refutaría la hipótesis si no supera a las referencias contemporáneas en las pruebas prefijadas, o si mejora el agregado a costa de B o de 6/7.

La arquitectura tiene más capacidad que la base y cambia también la forma de supervisión; una mejora no aislaría cuál de esos dos elementos causa el efecto. La rugosidad puede codificar clase y sufrir desplazamiento: es el riesgo central de la puerta, por eso se fijan controles por clase, grupo y tasa y no se ajustará su entrada después de mirar estos resultados.

## Datos y controles

`plans/gate_calidad_01.json` fija los cinco IDs y SHA256 de 18 fuentes. Dos pesos entrenan con 90k, semillas 2026/2027, mismo simulador v2, 200 épocas, lote 256, LR 0,002, equilibrio de clases, `--sin-atajos` y elección de época en `val_exp052_select` que las referencias nocturnas. Los otros tres reservan A/B/C exactamente como `cv_grupos_v1`, con `--sin-grupo`, mismos índices de ~60k, mismos controles de época y semilla 2026. Las referencias base/dos rutas ya existen: no se reentrena ninguna. La única intervención nueva es la red con puerta y sus dos pérdidas auxiliares fijas.

Las cinco ejecuciones deben terminar antes de abrir `val_exp052_audit` o `val_10k` con los nuevos pesos. La comparación de 90k usa `scripts/auditar_gate_90k.py`, que comprueba recetas y hashes y compara en las mismas 5k señales simuladas y crudas contra la base y dos rutas de igual semilla. La CV usa `scripts/auditar_cv_grupos.py --nombre gate_calidad_01 --referencia base`, con predicciones fuera de grupo sobre las mismas 10k señales. Crudo y simulado son dos vistas de las mismas señales, no dos tests independientes. El bloque audit y los pliegues ya se consultaron históricamente; este protocolo reduce fuga de checkpoint y selección posterior, pero **no crea prueba externa**.

## Criterios fijados antes del resultado

Métrica primaria: media de score simulado a N=5000 de las dos semillas de 90k. Para seguir investigando la puerta exigir **≥16,69**, es decir ≥+0,50 sobre la media base 16,192; además ninguna semilla debe quedar por debajo de su base y la media cruda debe ser al menos la de la base. En la misma auditoría, clase 6 y clase 7 deben perder menos de 10 aciertos cada una en cada semilla respecto a la base, y el score de B no debe caer más de 0,30 en ninguna semilla. La CV fuera de grupo es control estructural: score OOF simulado debe superar a base por ≥0,40, sin caída de score en B, y el score crudo OOF no debe empeorar. Revisar por tasa 1/2 y grupo×tasa para localizar concentraciones; no usar esos desgloses para retocar la puerta.

Si falla cualquiera de estos filtros, cerrar esta arquitectura sin ajuste de puerta ni consulta al demo. Si pasa, la señal merece una fuente objetivo nueva o una validación externa antes de afirmar mejora en examen; no seleccionar una semilla por su décima superior. Ningún score a N=5000 o N=10000 se compara directamente con el demo 400 ni con proyecciones de 60 s.

## Entrenar (solo David)

Desde `gdg-uam-ml-contest-2026/`, tras comprobar que los IDs están libres:

```bash
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id gq01_26 --arquitectura cnn --red dos_rutas_gate --muestras 90000 --epocas 200 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.002 --semilla 2026 --runs 1 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id gq01_27 --arquitectura cnn --red dos_rutas_gate --muestras 90000 --epocas 200 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.002 --semilla 2027 --runs 1 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id gq01_a --red dos_rutas_gate --indices-entrenamiento data/cv_grupos_v1_sin_a_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_a_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id gq01_b --red dos_rutas_gate --indices-entrenamiento data/cv_grupos_v1_sin_b_train.npy --muestras 59894 --archivo-examen data/cv_grupos_v1_sin_b_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py --id gq01_c --red dos_rutas_gate --indices-entrenamiento data/cv_grupos_v1_sin_c_train.npy --muestras 60053 --archivo-examen data/cv_grupos_v1_sin_c_select.npz --epocas 200 --batch-size 256 --simular-examen --sin-atajos --sin-grupo --equilibrar --elegir-epoca examen --ancho 1 --lr 0.002 --semilla 2026 --min-libre-gib 6
```

Estimación: unos **100–130 min** acumulados en este Mac/SSD, por comparación con las corridas de 200 épocas nocturnas y CV, más 5–10 min de auditoría. La puerta añade dos cabezas; el tiempo real puede variar. No ejecutar los cinco en paralelo con MPS.

Después de las cinco corridas:

```bash
.venv/bin/python scripts/auditar_gate_90k.py
.venv/bin/python scripts/auditar_cv_grupos.py --nombre gate_calidad_01 --ids gq01_a gq01_b gq01_c --referencia base
```

## Resultado de la auditoría (9 nov 2026)

David completó los cinco entrenamientos. Las épocas elegidas fueron 190 y 197 para los pesos de 90k, y 133/179/158 para A/B/C. `scripts/auditar_gate_90k.py` comprobó recetas y hashes antes de abrir el bloque de 5.000 señales y guardó `experiments/gate_calidad_01_auditoria.json`. La auditoría fuera de grupo guardó `experiments/cv_grupos_v1_gate_calidad_01.json`. No se consultó el demo.

| Comparación de 90k, score simulado a N=5000 | Base | Puerta | Cambio |
| --- | ---: | ---: | ---: |
| Semilla 2026 | 16,300410 | 16,244315 | −0,056095 |
| Semilla 2027 | 16,084483 | 16,292185 | +0,207702 |
| Media | **16,192446** | **16,268250** | **+0,075803** |

La media queda bajo el mínimo prefijado **16,69** y la semilla 2026 retrocede. En esa semilla la puerta pierde **14 aciertos de clase 6 y 10 de clase 7** frente a la base, incumpliendo también el límite de menos de 10 por clase. El score crudo medio sube 17,241781 → 17,423537; en B el cambio simulado es −0,177/+0,319 por semilla, dentro del límite. Estos controles favorables no reparan los fallos del criterio principal.

En CV fuera de grupo, la puerta mejora el score OOF simulado **13,996515 → 14,832015** (+0,835500) y el crudo **15,611416 → 16,293113** (+0,681697). B mejora +0,465688 en simulado, pero A baja −0,193761. La clase 7 agregada pierde **49 aciertos** (684/987 → 635/987), aunque la clase 6 gana 7 (678/1012 → 685/1012). El IC95 pareado descriptivo del cambio simulado es [+0,334680, +1,309207]; remuestrea señales dentro de estos grupos ya estudiados, no grupos nuevos.

**Decisión conforme al plan: cerrar esta arquitectura sin ajustar la puerta ni consultar el demo.** La mejora de CV es una pista de representación para investigación posterior, no justifica sustituir el modelo de entrega. Los pesos de 90k tampoco han demostrado ganancia suficiente para compensar una inferencia previsiblemente más costosa; no se midió esa latencia. El dashboard `/gate` muestra ambas auditorías y todos los filtros prefijados.
