# Media fusionada JIT · prueba 01

Preparado el 10 nov 2026 antes de medir la variante en Colab CPU. El peso y la red siguen siendo `exp033.sesgo.onnx`. El cambio experimental está **apagado** por defecto (`GDG_JIT_MEDIA=0`). David ejecuta la evaluación; el asistente prepara el paquete y analiza el informe.

## Hipótesis

`src/preparacion_numba.py` interpola cada canal y después lo recorre otra vez para sumar la media. `src/preparacion_numba_media.py` suma el mismo valor completo en el mismo orden dentro del bucle de interpolación. Conserva los cálculos de desviación, normalización y auxiliares. No cambia pesos, red, etiquetas ni sesgos. La reducción de un recorrido podría disminuir el coste por predicción, aunque Numba puede compilar los dos bucles de forma distinta.

La variante es optativa en `src/red_onnx.py`. Con `GDG_JIT_MEDIA=1` usa la media fusionada; con `0` usa la ruta estable. Requiere JIT y buffers activados como en la entrega actual.

En este Mac no está instalado Numba en `.venv`. Se comprobó en Python la identidad exacta de los tres buffers para 40 señales sintéticas con huecos, canales vacíos y grupos distintos, y se validó la sintaxis. **La paridad compilada y la velocidad solo quedan verificadas si pasa la prueba de Colab.**

## Paquete y ejecución

Paquete: `../jit_media_colab.zip` (SHA-256 `d3ea32830ffe2f7be8734a739d5bf866853dac6e894a7c29836d12340b5ebef6`). Contiene el harness y la API oficiales, el modelo actual, las dependencias y el comparador. En un cuaderno Colab con **CPU**, subir el ZIP y ejecutar:

```python
from google.colab import files
from zipfile import ZipFile
uploaded = files.upload()
ZipFile(next(iter(uploaded))).extractall('/content')
%cd /content/jit_media_colab/gdg-uam-ml-contest-2026
!python -m pip install -q -r requirements.txt
!python scripts/comparar_jit_media.py
files.download('experiments/jit_media_harness.json')
```

El comparador usa las mismas 50.000 señales (125 copias del demo de 400) cuatro veces, en orden control–fusión–fusión–control. Exige mismo `stream_hash`, índices y 50.000 predicciones idénticas antes de guardar un resultado. Las copias miden latencia y **no son datos nuevos de calidad**.

## Decisión fijada antes del resultado

Activar `GDG_JIT_MEDIA=1` por defecto **solo si**:

1. Las 50.000 predicciones de las cuatro corridas coinciden exactamente.
2. La fusión es más rápida que su control en **ambas parejas**.
3. La reducción media de latencia es de **al menos 3 %**.

Si falla cualquier punto, quitar la variante experimental de la ruta de entrega y mantener la preparación actual. No retocar el umbral ni el código de la variante con el mismo benchmark. El score del demo repetido no se usa para elegir nada; el resultado de latencia es específico de ese entorno Colab.

## Resultado del 10 nov 2026

David ejecutó el paquete en Colab CPU. El informe descargado se conserva en `experiments/jit_media_harness.json` (SHA-256 `114099004f094ace95acf094ac2646cca3451ba4cfb6164838e6abac5472e79d`). Las cuatro corridas procesaron **50.000 predicciones idénticas**:

| Orden | Control | Media fusionada |
| --- | ---: | ---: |
| Primer par | 0,106644 ms | 0,105741 ms |
| Segundo par | 0,106090 ms | 0,107302 ms |
| Media | 0,106367 ms | 0,106521 ms |

La reducción relativa fue **−0,145 %** y el segundo par fue más lento. `pasa: false`: fallan ambos requisitos de velocidad, aunque la identidad de predicciones sí se cumple.

**Decisión:** descartar la media fusionada y retirar `GDG_JIT_MEDIA` de `src/red_onnx.py`. El archivo experimental y el ZIP quedan como registro de la prueba; la entrega mantiene el preprocesado JIT en buffers anterior. No repetir este benchmark para ajustar el código de la misma variante.
