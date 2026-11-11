"""Entrena y evalúa un experimento para importarlo en el dashboard"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT.parent / "ml-contest-materials" / "harness.py"


def sha256(ruta: Path) -> str:
    # El hash identifica los archivos exactos usados en esta ejecución
    with ruta.open("rb") as archivo:
        return hashlib.file_digest(archivo, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", help="Identificador único, por ejemplo exp001")
    parser.add_argument("--muestras", type=int, default=5_000)
    parser.add_argument("--indices-entrenamiento", default=None,
                        help="CNN: archivo .npy de remuestreo de train_90k")
    parser.add_argument("--iteraciones", type=int, default=80)
    parser.add_argument("--arquitectura", choices=("histgb", "cnn"), default="histgb")
    parser.add_argument("--epocas", type=int, default=12, help="Épocas de la CNN")
    parser.add_argument("--batch-size", type=int, default=256, help="Tamaño de lote de la CNN")
    parser.add_argument("--aumento-ruido", action="store_true", help="HGB: copias ruidosas; CNN: ruido y huecos por lote")
    parser.add_argument("--aumento-ritmo", action="store_true", help="Varía el ritmo de las copias ruidosas")
    parser.add_argument("--temporales", action="store_true", help="Añade características de tramos temporales")
    parser.add_argument("--extendidas", action="store_true", help="Añade características estadísticas y espectrales detalladas")
    parser.add_argument("--simular-examen", action="store_true", help="CNN: pasa cada época por el simulador de examen")
    parser.add_argument("--sin-atajos", action="store_true", help="CNN: anula medias y level en las auxiliares")
    parser.add_argument("--sin-grupo", action="store_true", help="CNN: anula el indicador A/B/C en entrenamiento y evaluación")
    parser.add_argument("--prob-grupo-cero", type=float, default=0.0,
                        help="CNN: oculta el indicador A/B/C por muestra durante el entrenamiento")
    parser.add_argument("--sin-escala", action="store_true", help="CNN: anula tamaños de canal en las auxiliares")
    parser.add_argument("--prob-escala-cero", type=float, default=0.0,
                        help="CNN: probabilidad de ocultar tamaños de canal durante entrenamiento")
    parser.add_argument("--equilibrar", action="store_true", help="CNN: todas las clases pesan igual")
    parser.add_argument("--peso-clase-6", type=float, default=1.0,
                        help="CNN: multiplica el peso de la clase 6 sobre el equilibrio")
    parser.add_argument("--igualar-correlacion-67", action="store_true",
                        help="CNN: iguala correlación 0–1 de clases 6/7 durante entrenamiento")
    parser.add_argument("--peso-consistencia-vistas", type=float, default=0.0,
                        help="CNN: consistencia entre dos simulaciones independientes de cada muestra")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="CNN: mezcla señales y etiquetas simuladas")
    parser.add_argument("--peso-alineacion-demo", type=float, default=0.0,
                        help="CNN: alinea representaciones de train con demo sin leer sus etiquetas")
    parser.add_argument("--inicializar-pesos", default=None, help="CNN: checkpoint .pt inicial")
    parser.add_argument("--peso-destilacion", type=float, default=0.0,
                        help="CNN: KL frente al cuádruple fijo exp033+034+051+053")
    parser.add_argument("--peso-retencion-67", type=float, default=0.0,
                        help="CNN: conserva márgenes correctos 6/7 de exp033")
    parser.add_argument("--temperatura-destilacion", type=float, default=2.0)
    parser.add_argument("--ancho", type=int, default=1, help="CNN: multiplica los filtros de la red")
    parser.add_argument("--red", choices=("base", "compacta", "multiescala", "dos_rutas", "dos_rutas_gate"), default="base",
                        help="CNN: arquitectura temporal; dos_rutas mezcla patrones conjuntos y por canal")
    parser.add_argument("--fusion-tardia", action="store_true",
                        help="CNN: procesa cada canal por separado hasta el resumen final")
    parser.add_argument("--elegir-epoca", choices=("control", "examen"), default="control", help="CNN: cómo elegir la mejor época")
    parser.add_argument("--archivo-examen", default="data/val_examen_v2.npz",
                        help="CNN: archivo para elegir la época cuando --elegir-epoca examen")
    parser.add_argument("--lr", type=float, default=0.001, help="CNN: tasa de aprendizaje")
    parser.add_argument("--semilla", type=int, default=2026, help="CNN: semilla del entrenamiento")
    parser.add_argument("--min-libre-gib", type=float, default=0.0,
                        help="CNN: corta el entrenamiento si el disco baja de este umbral")
    parser.add_argument("--ajustes-simulador", default="{}", help="CNN: JSON con ajustes del simulador")
    parser.add_argument("--validacion", choices=("val_10k", "val_examen", "val_examen_v2", "val_exp052_select"), default="val_10k",
                        help="Archivo de la evaluación oficial; los val_examen usan todo el archivo y 60 s")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-muestras", type=int, default=1_000)
    parser.add_argument("--presupuesto-s", type=float, default=2.0)
    args = parser.parse_args()

    if args.id is None:
        if len(sys.argv) > 1:
            parser.error("Indica --id o ejecuta el script sin argumentos para abrir el menú")
        from menu_entrenamiento import menu_entrenamiento

        try:
            seleccion = menu_entrenamiento()
        except ValueError as error:
            parser.error(str(error))
        if seleccion is None:
            print("Entrenamiento cancelado")
            return
        for campo, valor in seleccion.items():
            setattr(args, campo, valor)

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("--id debe usar letras minúsculas, números, guiones o guiones bajos")
    if (args.muestras < 8 or args.iteraciones < 1 or args.epocas < 1 or args.batch_size < 1
            or args.runs < 1 or args.max_muestras < 1 or args.presupuesto_s <= 0):
        parser.error("--muestras debe ser al menos 8; los demás valores numéricos deben ser positivos")
    if args.arquitectura == "cnn" and args.muestras < 80:
        parser.error("La CNN requiere al menos 80 muestras para separar control interno")
    if not 0.0 <= args.prob_escala_cero <= 1.0:
        parser.error("--prob-escala-cero debe estar entre 0 y 1")
    if not 0.0 <= args.prob_grupo_cero <= 1.0 or (args.sin_grupo and args.prob_grupo_cero):
        parser.error("--prob-grupo-cero debe estar entre 0 y 1 y no se combina con --sin-grupo")
    if args.sin_escala and args.prob_escala_cero:
        parser.error("--sin-escala y --prob-escala-cero no se pueden combinar")
    if args.peso_clase_6 <= 0 or (args.peso_clase_6 != 1.0 and not args.equilibrar):
        parser.error("--peso-clase-6 debe ser positivo y requiere --equilibrar si cambia de 1")
    if args.peso_consistencia_vistas < 0 or not math.isfinite(args.peso_consistencia_vistas):
        parser.error("--peso-consistencia-vistas debe ser finito y no negativo")
    if args.min_libre_gib < 0 or not math.isfinite(args.min_libre_gib):
        parser.error("--min-libre-gib debe ser finito y no negativo")
    if args.peso_consistencia_vistas and (args.arquitectura != "cnn" or not args.simular_examen):
        parser.error("--peso-consistencia-vistas requiere CNN y --simular-examen")
    if any(valor < 0 or not math.isfinite(valor) for valor in (args.mixup_alpha, args.peso_alineacion_demo)):
        parser.error("--mixup-alpha y --peso-alineacion-demo deben ser finitos y no negativos")
    if (args.mixup_alpha or args.peso_alineacion_demo) and (args.arquitectura != "cnn" or not args.simular_examen):
        parser.error("mixup y alineación demo requieren CNN y --simular-examen")
    if sum(bool(valor) for valor in (args.peso_consistencia_vistas, args.mixup_alpha, args.peso_alineacion_demo)) > 1:
        parser.error("Elige solo una de consistencia, mixup o alineación demo")
    if not 0.0 <= args.peso_destilacion < 1.0 or not math.isfinite(args.peso_destilacion):
        parser.error("--peso-destilacion debe estar entre 0 y 1")
    if args.peso_retencion_67 < 0 or not math.isfinite(args.peso_retencion_67):
        parser.error("--peso-retencion-67 debe ser finito y no negativo")
    if args.temperatura_destilacion <= 0 or not math.isfinite(args.temperatura_destilacion):
        parser.error("--temperatura-destilacion debe ser positiva")
    if args.arquitectura == "cnn" and (args.aumento_ritmo or args.temporales or args.extendidas):
        parser.error("--aumento-ritmo, --temporales y --extendidas solo se usan con histgb")
    if args.red != "base" and (args.arquitectura != "cnn" or args.fusion_tardia):
        parser.error("--red multiescala/dos_rutas requiere CNN y no admite --fusion-tardia")
    if args.red == "compacta" and args.ancho != 1:
        parser.error("La red compacta solo admite --ancho 1")
    if args.arquitectura != "cnn" and (args.indices_entrenamiento or args.simular_examen or args.sin_atajos or args.sin_grupo or args.prob_grupo_cero or args.sin_escala or args.prob_escala_cero or args.equilibrar or args.peso_clase_6 != 1.0 or args.igualar_correlacion_67 or args.peso_consistencia_vistas or args.mixup_alpha or args.peso_alineacion_demo or args.inicializar_pesos or args.peso_destilacion or args.peso_retencion_67 or args.min_libre_gib or args.ancho != 1 or args.fusion_tardia
                                       or args.elegir_epoca != "control"):
        parser.error("Las opciones de simulador, auxiliares, pesos, ancho, fusión y elección de época solo se usan con cnn")
    if args.aumento_ritmo and not args.aumento_ruido:
        parser.error("--aumento-ritmo requiere --aumento-ruido")

    # Cada experimento conserva sus pesos y JSON oficiales propios
    carpeta = ROOT / "experiments" / args.id
    salida = ROOT / "results" / args.id
    extension = ".pt" if args.arquitectura == "cnn" else ".pkl"
    pesos = ROOT / "models" / f"{args.id}{extension}"
    for ruta in (carpeta, salida, pesos):
        if ruta.exists():
            parser.error(f"Ya existe {ruta}; elige otro --id para no sobrescribir el historial")

    particion = ROOT / "experiments/particion_90_10_2026.json"
    validacion = ROOT / f"data/{args.validacion}.npz"
    if args.validacion.startswith("val_examen") or args.validacion == "val_exp052_select":
        # Mismo protocolo que results/examen_*: un archivo entero con el límite de 60 s del concurso
        args.max_muestras, args.presupuesto_s = 10_000_000, 60.0
    for ruta in (HARNESS, ROOT / "data/train_90k.npz", validacion, particion):
        if not ruta.is_file():
            parser.error(f"Falta {ruta}")

    # Se usa el mismo intérprete Python que inició este script
    # El entrenamiento solo lee train_90k y guarda pesos independientes
    script_entrenamiento = "entrenar_cnn.py" if args.arquitectura == "cnn" else "entrenar_modelo.py"
    entrenamiento = [
        sys.executable, str(ROOT / "scripts" / script_entrenamiento),
        "--id", args.id, "--muestras", str(args.muestras),
    ]
    if args.arquitectura == "cnn":
        entrenamiento += ["--epocas", str(args.epocas), "--batch-size", str(args.batch_size),
                          "--ancho", str(args.ancho), "--red", args.red,
                          "--elegir-epoca", args.elegir_epoca,
                          "--archivo-examen", args.archivo_examen,
                          "--lr", str(args.lr), "--semilla", str(args.semilla),
                          "--min-libre-gib", str(args.min_libre_gib),
                          "--prob-escala-cero", str(args.prob_escala_cero),
                          "--prob-grupo-cero", str(args.prob_grupo_cero),
                          "--peso-clase-6", str(args.peso_clase_6),
                          "--peso-consistencia-vistas", str(args.peso_consistencia_vistas),
                          "--mixup-alpha", str(args.mixup_alpha),
                          "--peso-alineacion-demo", str(args.peso_alineacion_demo),
                          "--peso-destilacion", str(args.peso_destilacion),
                          "--peso-retencion-67", str(args.peso_retencion_67),
                          "--temperatura-destilacion", str(args.temperatura_destilacion),
                          "--ajustes-simulador", args.ajustes_simulador]
        if args.inicializar_pesos:
            entrenamiento += ["--inicializar-pesos", args.inicializar_pesos]
        if args.indices_entrenamiento:
            entrenamiento += ["--indices-entrenamiento", args.indices_entrenamiento]
        for activo, opcion in ((args.aumento_ruido, "--aumento-ruido"), (args.simular_examen, "--simular-examen"),
                               (args.sin_grupo, "--sin-grupo"),
                               (args.sin_atajos, "--sin-atajos"), (args.sin_escala, "--sin-escala"),
                               (args.equilibrar, "--equilibrar"), (args.fusion_tardia, "--fusion-tardia"),
                               (args.igualar_correlacion_67, "--igualar-correlacion-67")):
            if activo:
                entrenamiento.append(opcion)
    else:
        entrenamiento += ["--iteraciones", str(args.iteraciones)]
        if args.aumento_ruido:
            entrenamiento.append("--aumento-ruido")
        if args.aumento_ritmo:
            entrenamiento.append("--aumento-ritmo")
        if args.temporales:
            entrenamiento.append("--temporales")
        if args.extendidas:
            entrenamiento.append("--extendidas")
    print("\n1/2 Entrenamiento", flush=True)
    subprocess.run(entrenamiento, cwd=ROOT, check=True)

    # Guardamos la configuración antes de evaluar para diagnosticar fallos
    datos_particion = json.loads(particion.read_text(encoding="utf-8"))
    configuracion = {
        "id": args.id,
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrenamiento": {
            "arquitectura": args.arquitectura,
            "muestras": args.muestras,
            "indices_entrenamiento": args.indices_entrenamiento,
            "sha256_indices_entrenamiento": sha256(ROOT / args.indices_entrenamiento) if args.indices_entrenamiento else None,
            "iteraciones": args.epocas if args.arquitectura == "cnn" else args.iteraciones,
            "batch_size": args.batch_size if args.arquitectura == "cnn" else None,
            "aumento_ruido": args.aumento_ruido,
            "aumento_ritmo": args.aumento_ritmo,
            "temporales": args.temporales,
            "extendidas": args.extendidas,
            "simular_examen": args.simular_examen,
            "sin_atajos": args.sin_atajos,
            "sin_grupo": args.sin_grupo,
            "prob_grupo_cero": args.prob_grupo_cero,
            "sin_escala": args.sin_escala,
            "prob_escala_cero": args.prob_escala_cero,
            "peso_clase_6": args.peso_clase_6,
            "igualar_correlacion_67": args.igualar_correlacion_67,
            "peso_consistencia_vistas": args.peso_consistencia_vistas,
            "mixup_alpha": args.mixup_alpha,
            "peso_alineacion_demo": args.peso_alineacion_demo,
            "inicializar_pesos": args.inicializar_pesos,
            "peso_destilacion": args.peso_destilacion,
            "peso_retencion_67": args.peso_retencion_67,
            "temperatura_destilacion": args.temperatura_destilacion,
            "equilibrar": args.equilibrar,
            "ancho": args.ancho,
            "red": args.red,
            "fusion_tardia": args.fusion_tardia,
            "elegir_epoca": args.elegir_epoca,
            "archivo_examen": args.archivo_examen if args.arquitectura == "cnn" else None,
            "lr": args.lr,
            "semilla": args.semilla,
            "min_libre_gib": args.min_libre_gib,
            "ajustes_simulador": json.loads(args.ajustes_simulador),
        },
        "evaluacion": {
            "source": f"file:data/{args.validacion}.npz",
            "runs": args.runs,
            "max_scored_samples": args.max_muestras,
            "time_budget_s": args.presupuesto_s,
        },
        "particion": {
            "sha256_indices_validacion": datos_particion.get("sha256_indices_validacion"),
            "semilla": datos_particion.get("semilla"),
            "sha256_archivo_validacion": sha256(validacion),
        },
        "sha256": {
            "model.py": sha256(ROOT / "model.py"),
            "src/caracteristicas.py" if args.arquitectura == "histgb" else "src/red_convolucional.py":
                sha256(ROOT / "src" / ("caracteristicas.py" if args.arquitectura == "histgb" else "red_convolucional.py")),
            **({"src/red_multiescala.py": sha256(ROOT / "src/red_multiescala.py")}
               if args.red == "multiescala" else {}),
            **({"src/red_compacta.py": sha256(ROOT / "src/red_compacta.py")}
               if args.red == "compacta" else {}),
            **({"src/red_dos_rutas.py": sha256(ROOT / "src/red_dos_rutas.py")}
               if args.red == "dos_rutas" else {}),
            **({"src/red_dos_rutas_gate.py": sha256(ROOT / "src/red_dos_rutas_gate.py")}
               if args.red == "dos_rutas_gate" else {}),
            f"scripts/{script_entrenamiento}": sha256(ROOT / "scripts" / script_entrenamiento),
            "src/simulador.py": sha256(ROOT / "src/simulador.py"),
            **({args.inicializar_pesos: sha256(ROOT / args.inicializar_pesos)}
               if args.inicializar_pesos else {}),
            **({"models/exp033.sesgo.json": sha256(ROOT / "models/exp033.sesgo.json")}
               if args.peso_retencion_67 else {}),
            **({f"models/{nombre}.pt": sha256(ROOT / f"models/{nombre}.pt")
                for nombre in ("exp033", "exp034", "exp051", "exp053")}
               if args.peso_destilacion else {}),
            **({f"models/{nombre}.sesgo.json": sha256(ROOT / f"models/{nombre}.sesgo.json")
                for nombre in ("exp033", "exp034")}
               if args.peso_destilacion else {}),
            **({f"src/{nombre}.py": sha256(ROOT / f"src/{nombre}.py")
                for nombre in ("red_multiescala", "red_dos_rutas")}
               if args.peso_destilacion else {}),
            "harness.py": sha256(HARNESS),
            "pesos": sha256(pesos),
        },
    }
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # El evaluador recibe solo validación y carga los pesos de este experimento
    # Los límites indicados aquí son de prueba rápida, no de evaluación final
    entorno = os.environ.copy()
    entorno["GDG_MODEL_PATH"] = f"models/{args.id}{extension}"
    evaluacion = [
        sys.executable, str(HARNESS), "evaluate",
        "--model", "model.py",
        "--source", f"file:data/{args.validacion}.npz",
        "--runs", str(args.runs),
        "--out", str(salida),
        "--set", f"max_scored_samples={args.max_muestras}",
        "--set", f"time_budget_s={args.presupuesto_s}",
    ]
    print("\n2/2 Evaluación oficial sobre validación", flush=True)
    subprocess.run(evaluacion, cwd=ROOT, env=entorno, check=True)
    print(
        f"\nExperimento {args.id} terminado. Importa results/{args.id}/model "
        f"y experiments/{args.id} en el dashboard"
    )


if __name__ == "__main__":
    main()
