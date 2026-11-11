"""Empaqueta la comparación de preparación JIT en buffers para Colab CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "ml-contest-materials"
FOLDER = "jit_buffers_colab"
PROJECT_FILES = (
    "model.py", "requirements.txt", "data/test_demo.npz",
    "models/exp033.sesgo.onnx",
    "src/__init__.py", "src/caracteristicas.py", "src/red_numpy.py",
    "src/red_onnx.py", "src/preparacion_numba.py",
    "scripts/comparar_jit_buffers.py",
)
OFFICIAL_FILES = ("harness.py", "competition_api.py")
INSTRUCTIONS = """# Comparación de preparación JIT en buffers en Colab CPU

Sube `jit_buffers_colab.zip` a un cuaderno con entorno CPU y ejecuta:

```python
from google.colab import files
from zipfile import ZipFile
uploaded = files.upload()
ZipFile(next(iter(uploaded))).extractall('/content')
%cd /content/jit_buffers_colab/gdg-uam-ml-contest-2026
!python -m pip install -q -r requirements.txt
!python scripts/comparar_jit_buffers.py
files.download('experiments/jit_buffers_harness.json')
```

El script ejecuta el harness oficial cuatro veces. Comprueba identidad de las
10.000 predicciones y compara latencias en dos órdenes. El demo repetido no
añade ejemplos independientes ni estima la nota del test oculto.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "jit_buffers_colab.zip")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f"Ya existe {output}; indica otra ruta con --out")
    entries = [
        (ROOT / name, f"{FOLDER}/{ROOT.name}/{name}") for name in PROJECT_FILES
    ] + [
        (OFFICIAL / name, f"{FOLDER}/{OFFICIAL.name}/{name}")
        for name in OFFICIAL_FILES
    ]
    missing = [str(source) for source, _ in entries if not source.is_file()]
    if missing:
        parser.error("Faltan archivos:\n" + "\n".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
            for source, target in entries:
                archive.write(source, target)
            archive.writestr(f"{FOLDER}/README_COLAB.md", INSTRUCTIONS)
        with ZipFile(output) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"ZIP dañado: {corrupt}")
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(f"Paquete listo: {output} ({output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
