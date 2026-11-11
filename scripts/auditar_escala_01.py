"""Audita las dos semillas de dropout de escala frente a sus controles."""

from __future__ import annotations

from pathlib import Path

from auditar_corr67_01 import run_audit

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {2026: ("n01_b_ref_26", "escala_26"),
         2027: ("n01_b_ref_27", "escala_27")}


if __name__ == "__main__":
    run_audit("escala_dropout_01", SEEDS, "prob_escala_cero", 0.5,
              ROOT / "experiments/escala_dropout_01_auditoria.json",
              "plans/escala_dropout_01.md")
