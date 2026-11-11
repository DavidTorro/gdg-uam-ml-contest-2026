#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "1/3 Entrenamiento d02_control_27"
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d02_control_27 --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2027 --runs 1 --min-libre-gib 6

echo "2/3 Entrenamiento d02_control_28"
caffeinate -dimsu .venv/bin/python scripts/ejecutar_experimento.py --id d02_control_28 --arquitectura cnn --red base --inicializar-pesos models/exp033.pt --muestras 90000 --epocas 40 --batch-size 256 --simular-examen --sin-atajos --equilibrar --elegir-epoca examen --archivo-examen data/val_exp052_select.npz --validacion val_exp052_select --ancho 1 --lr 0.0002 --semilla 2028 --runs 1 --min-libre-gib 6

echo "3/3 Auditoría de las dos réplicas"
.venv/bin/python scripts/auditar_afinamiento_02.py
