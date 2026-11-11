<div align="center">

![Banner](./assets/banner.svg)
<br>
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch (training)">
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/-Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white" alt="Numba">
  <img src="https://img.shields.io/badge/-ONNX%20Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab (benchmark)">
  <br>
  <a href="https://github.com/DavidTorro">
    <img src="https://img.shields.io/badge/author-David%20Torr%C3%B3-181717?style=for-the-badge&logo=github&logoColor=white" alt="Author: David Torró">
  </a>
</div>

# GDG UAM ML Contest 2026

[Read in English](README.md)

Mi participación individual en el [GDG UAM ML Contest 2026](https://github.com/GDG-UAM/ml-contest-materials). El objetivo es clasificar una serie temporal de cuatro canales y 256 valores en una de ocho clases. El evaluador oficial llama a `predict` para cada muestra por separado y puntúa tanto la calidad como la velocidad.

## Modelo

`model.py` carga por defecto `models/exp033.sesgo.onnx`. Es una CNN 1D exportada a ONNX, con el ajuste de la salida de la clase 6 incorporado en el propio peso. La inferencia usa ONNX Runtime en un hilo de CPU. Numba compila la preparación durante la inicialización; los buffers reutilizables y el I/O Binding de ONNX Runtime reducen el coste por muestra. La ruta de inferencia predeterminada no necesita PyTorch ni los datos de entrenamiento.

## Resultados en el demo

| Evaluación | Resultado |
| --- | ---: |
| `test_demo.npz` oficial (400 muestras distintas) | 338/400 aciertos · 84,5 % · 3,779 puntos |
| Prueba de velocidad en Colab (las mismas 400 muestras repetidas hasta sumar 60 s de `predict`) | 120,576 puntos |

Son comprobaciones con el demo público, **no la nota del test oculto ni una estimación de ella**. La repetición del demo permite medir la ruta de inferencia con un presupuesto más largo, pero no aporta ejemplos de test independientes.

## Archivos de entrega

| Ruta | Función |
| --- | --- |
| `model.py` | Punto de entrada `Model` del evaluador |
| `models/exp033.sesgo.onnx` | CNN entrenada y ajuste de salida incorporado |
| `src/red_onnx.py` | Inferencia con ONNX Runtime e I/O Binding |
| `src/preparacion_numba.py` | Preparación JIT en buffers reutilizables |
| `src/red_numpy.py` | Preparación NumPy auxiliar que importa el módulo ONNX |
| `src/caracteristicas.py`, `src/__init__.py` | Importaciones necesarias del punto de entrada |
| `requirements.txt` | Dependencias de inferencia fijadas |

El harness oficial aporta `competition_api.py`; este repositorio no lo incluye.

## Prueba desde un clon limpio

Ejecuta estos comandos en un entorno limpio con Python 3.10 o posterior:

```bash
git clone https://github.com/DavidTorro/gdg-uam-ml-contest-2026.git submission-check
git clone https://github.com/GDG-UAM/ml-contest-materials.git
python -m pip install -r submission-check/requirements.txt
curl -L --create-dirs -o ml-contest-materials/data/test_demo.npz \
  https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
cd ml-contest-materials
python harness.py evaluate --model ../submission-check/model.py \
  --source file:data/test_demo.npz --runs 1
```

La versión completa del trabajo, con scripts de entrenamiento y experimentos, sigue disponible en el [commit `ffb3255`](https://github.com/DavidTorro/gdg-uam-ml-contest-2026/tree/ffb3255).
