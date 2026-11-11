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

[Leer en español](README.es.md)

My individual entry for the [GDG UAM ML Contest 2026](https://github.com/GDG-UAM/ml-contest-materials). The task is to classify a four-channel, 256-step time series into one of eight classes. The official harness calls `predict` on one sample at a time and scores both accuracy and throughput.

## Model

`model.py` loads `models/exp033.sesgo.onnx` by default. It is a 1D CNN exported to ONNX, including the class-6 logit adjustment in the weight file. Inference uses ONNX Runtime on one CPU thread. Numba compiles preprocessing during setup; reusable buffers and ONNX Runtime I/O Binding reduce per-sample overhead. The default inference path does not require PyTorch or training data.

## Demo results

| Evaluation | Result |
| --- | ---: |
| Official `test_demo.npz` (400 distinct samples) | 338/400 correct · 84.5% accuracy · 3.779 score |
| Colab throughput check (the same 400 demo samples repeated until 60 s of `predict`) | 120.576 score |

These are checks on the public demo, **not a hidden-test score or an estimate of it**. The repeated-demo run measures the inference path under a longer time budget; it does not add independent test examples.

## Submission files

| Path | Purpose |
| --- | --- |
| `model.py` | Official `Model` entry point |
| `models/exp033.sesgo.onnx` | Trained CNN and embedded logit adjustment |
| `src/red_onnx.py` | ONNX Runtime inference and I/O Binding |
| `src/preparacion_numba.py` | JIT preprocessing into reusable buffers |
| `src/red_numpy.py` | NumPy preprocessing fallback used by the ONNX module |
| `src/caracteristicas.py`, `src/__init__.py` | Imports required by the entry point |
| `requirements.txt` | Pinned inference dependencies |

The official harness provides `competition_api.py`; it is not part of this repository.

## Test from a fresh clone

Run these commands in a clean Python 3.10+ environment:

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

The complete development snapshot, including training scripts and experiments, remains available in [commit `ffb3255`](https://github.com/DavidTorro/gdg-uam-ml-contest-2026/tree/ffb3255).
