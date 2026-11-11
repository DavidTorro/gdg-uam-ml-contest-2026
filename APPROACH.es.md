# Cómo construimos esta entrega

[Read in English](APPROACH.md) · [Volver al README](README.es.md)

El concurso pide clasificar una señal de cuatro canales y 256 valores en una de ocho clases. El evaluador oficial llama a `predict` una vez por muestra y puntúa tanto las predicciones como el número de muestras procesadas en 60 segundos. Por eso trabajamos en la capacidad de generalizar y en la velocidad de inferencia en CPU.

## De los datos al modelo

Primero comparamos los datos de entrenamiento con el demo público. Algunas pistas del entrenamiento, como las medias de los canales y el metadato `level`, estaban muy asociadas a ciertas clases, pero no se mantenían de forma fiable en el demo. Eliminamos esas pistas y construimos un simulador parecido al examen: añade ruido, picos, valores ausentes, cambios de amplitud y otras alteraciones a todas las clases. Equilibramos las clases al entrenar y elegimos la época con una validación simulada.

Probamos modelos basados en características y varias configuraciones de CNN 1D. El modelo elegido, `exp033`, es una CNN pequeña entrenada con PyTorch. Interpola los valores ausentes, normaliza cada canal y combina la señal con unas pocas medidas auxiliares. Calibramos un pequeño ajuste de la salida de la clase 6 con el demo público *sin consultar sus etiquetas*, para acercar la frecuencia de predicciones al reparto equilibrado esperado. Sí usamos las etiquetas del demo para medir resultados, pero no para entrenar la red ni calcular ese ajuste.

## Preparación de la entrega

Exportamos la red elegida y el ajuste de la clase 6 a `models/exp033.sesgo.onnx`. La entrega usa ONNX Runtime en un hilo de CPU y no necesita PyTorch para predecir. Compilamos el preprocesado con Numba durante la inicialización y después usamos I/O Binding de ONNX Runtime y buffers reutilizables para reducir el trabajo de cada llamada a `predict`. Comprobamos que estas optimizaciones conservaran las predicciones en las comparaciones realizadas.

En el **demo público de 400 muestras**, la entrega acierta **338 (84,5 %)** y obtiene **3,778906 puntos**. Una ejecución completa de 60 segundos en Colab, repitiendo esas mismas 400 muestras, procesó 407.260 predicciones y obtuvo 120,576 puntos. La repetición mide velocidad sostenida; no es una prueba independiente de calidad. También evaluamos otras ideas de modelo, pero mantuvimos `exp033` porque no aportaron pruebas suficientemente sólidas de una mejor combinación de calidad y velocidad.

El [README](README.es.md) y el [cuaderno de comprobación](colab.es.ipynb) explican cómo repetir la prueba del demo desde un entorno nuevo.
