#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  26) seed=2026; experiment=corr67_26 ;;
  27) seed=2027; experiment=corr67_27 ;;
  *) echo 'Uso: bash scripts/entrenar_corr67_01.sh 26|27' >&2; exit 2 ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec caffeinate -dimsu .venv/bin/python scripts/entrenar_cnn.py \
  --id "$experiment" --muestras 90000 --epocas 200 --batch-size 256 \
  --simular-examen --sin-atajos --equilibrar --elegir-epoca examen \
  --archivo-examen data/val_exp052_select.npz --ancho 1 --red base \
  --lr 0.002 --semilla "$seed" --igualar-correlacion-67 --min-libre-gib 6
