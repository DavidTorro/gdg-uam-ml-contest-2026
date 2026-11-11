#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

plan = json.loads(Path("plans/retencion_67_01.json").read_text(encoding="utf-8"))
for name, expected in plan["sha256_fuentes"].items():
    with Path(name).open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Cambió la fuente: {name}")
print("Fuentes verificadas; iniciando r67_retencion_28", flush=True)
PY

caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py \
  --id r67_retencion_28 --arquitectura cnn --red base \
  --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 \
  --batch-size 256 --simular-examen --sin-atajos --equilibrar \
  --elegir-epoca examen --archivo-examen data/val_exp052_select.npz \
  --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2028 \
  --peso-retencion-67 0.2 --runs 1 --min-libre-gib 6
