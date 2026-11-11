"""Empaqueta los cuatro finalistas y el harness oficial para medirlos en Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / "ml-contest-materials"
FOLDER = "afinamiento_02_colab"

PROJECT_FILES = (
    "model.py",
    "requirements.txt",
    "data/test_demo.npz",
    "models/exp033.sesgo.onnx",
    "models/d01_control.onnx",
    "models/d02_control_27.onnx",
    "models/d02_control_28.onnx",
    "src/__init__.py",
    "src/caracteristicas.py",
    "src/red_numpy.py",
    "src/red_onnx.py",
    "src/preparacion_numba.py",
    "scripts/comparar_afinamiento_02.py",
)
OFFICIAL_FILES = ("harness.py", "competition_api.py")

INSTRUCTIONS = """# Comparación de finalistas en Colab CPU

Sube `afinamiento_02_colab.zip` a Colab, elige un entorno de ejecución CPU y
ejecuta estas celdas:

```python
!unzip -q /content/afinamiento_02_colab.zip -d /content
%cd /content/afinamiento_02_colab/gdg-uam-ml-contest-2026
!python -m pip install -q -r requirements.txt
!python scripts/comparar_afinamiento_02.py
```

```python
from google.colab import files
files.download('experiments/afinamiento_02_harness.json')
```

La prueba usa el harness oficial. El demo original compara aciertos; las 25
copias del demo sirven solo para medir velocidad en dos órdenes. Ni el score
del demo repetido ni esta prueba estiman la nota del test oculto. La inferencia
ONNX usa `CPUExecutionProvider` y un hilo. El entrenamiento puede usar GPU.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "afinamiento_02_colab.zip")
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f"Ya existe {output}; indica otra ruta con --out")

    entries = [
        (ROOT / name, f"{FOLDER}/{ROOT.name}/{name}") for name in PROJECT_FILES
    ] + [
        (OFFICIAL / name, f"{FOLDER}/{OFFICIAL.name}/{name}") for name in OFFICIAL_FILES
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
