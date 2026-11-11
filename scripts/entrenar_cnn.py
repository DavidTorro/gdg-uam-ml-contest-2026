"""Entrena una CNN 1D sin utilizar la validación oficial ni el test demo.

Sin opciones nuevas reproduce exp015 a exp020: separa un 10 % de control interno y elige
la época de menor pérdida en él. Desde exp022 se puede entrenar como para el examen:
--simular-examen pasa cada época por src/simulador.py, --sin-atajos anula las auxiliares
que delatan la clase solo en train, --equilibrar hace que todas las clases pesen igual y
--elegir-epoca examen elige la época con la mejor nota en el archivo de --archivo-examen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "ml-contest-materials"))
sys.path.insert(0, str(ROOT))

from src.red_convolucional import RedConvolucional, preparar_lote
from src.red_compacta import RedCompacta
from src.red_dos_rutas import RedDosRutas
from src.red_dos_rutas_gate import RedDosRutasGate
from src.red_multiescala import RedMultiescala
from src.simulador import PARAMETROS, simular_examen


def train_test_split(*args, **kwargs):
    """Carga scikit-learn solo si esta corrida necesita crear una partición."""
    from sklearn.model_selection import train_test_split as split

    return split(*args, **kwargs)


def aumentar_senales(serie: torch.Tensor, auxiliares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simula ruido rápido y huecos extra únicamente en los lotes de entrenamiento."""
    cantidad, canales, _ = serie.shape
    activo = (torch.rand((cantidad, 1, 1), device=serie.device) < 0.7).to(serie.dtype)
    intensidad = (0.08 + 0.22 * torch.rand((cantidad, canales, 1), device=serie.device)) * activo
    aumentada = serie + torch.randn_like(serie) * intensidad

    # Sustituimos pocos puntos por la media de sus vecinos, como una
    # aproximación de la interpolación que se hace con los huecos reales.
    huecos = torch.rand(serie.shape, device=serie.device) < 0.008
    izquierda = torch.cat((aumentada[:, :, :1], aumentada[:, :, :-1]), dim=2)
    derecha = torch.cat((aumentada[:, :, 1:], aumentada[:, :, -1:]), dim=2)
    aumentada = torch.where(huecos, (izquierda + derecha) * 0.5, aumentada)
    auxiliares = auxiliares.clone()
    auxiliares[:, 8:12] = torch.clamp(
        auxiliares[:, 8:12] + huecos.float().mean(dim=2), max=1.0
    )
    return aumentada.clamp(-6.0, 6.0), auxiliares


def igualar_correlacion_67(serie: torch.Tensor, etiqueta: torch.Tensor) -> torch.Tensor:
    """Da a 6 y 7 la misma distribución de correlación 0–1 durante entrenamiento.

    Conserva el canal 0 y la parte del canal 1 ortogonal a él. La correlación
    objetivo se sortea igual para ambas clases; las otras seis no se modifican.
    """
    mascara = (etiqueta == 6) | (etiqueta == 7)
    if not bool(mascara.any()):
        return serie
    salida = serie.clone()
    canal_0 = serie[mascara, 0]
    canal_1 = serie[mascara, 1]
    canal_0 = canal_0 - canal_0.mean(dim=1, keepdim=True)
    canal_0 = canal_0 / (canal_0.square().mean(dim=1, keepdim=True).sqrt() + 1e-6)
    canal_1 = canal_1 - canal_1.mean(dim=1, keepdim=True)
    residuo = canal_1 - (canal_0 * canal_1).mean(dim=1, keepdim=True) * canal_0
    residuo = residuo / (residuo.square().mean(dim=1, keepdim=True).sqrt() + 1e-6)
    objetivo = 0.55 + 0.20 * torch.rand((len(canal_0), 1), device=serie.device)
    salida[mascara, 1] = (objetivo * canal_0 + (1 - objetivo.square()).sqrt() * residuo).clamp(-6, 6)
    return salida


def divergencia_vistas(logits_1: torch.Tensor, logits_2: torch.Tensor,
                       etiquetas: torch.Tensor, pesos_clase: torch.Tensor | None) -> torch.Tensor:
    """Jensen-Shannon simétrica entre dos simulaciones de la misma muestra, a T=2."""
    temperatura = 2.0
    log_1 = F.log_softmax(logits_1 / temperatura, dim=1)
    log_2 = F.log_softmax(logits_2 / temperatura, dim=1)
    prob_1, prob_2 = log_1.exp(), log_2.exp()
    log_media = ((prob_1 + prob_2) * 0.5).clamp_min(1e-12).log()
    por_muestra = 0.5 * (
        (prob_1 * (log_1 - log_media)).sum(dim=1)
        + (prob_2 * (log_2 - log_media)).sum(dim=1)
    ) * temperatura ** 2
    if pesos_clase is None:
        return por_muestra.mean()
    pesos = pesos_clase[etiquetas]
    return (por_muestra * pesos).sum() / pesos.sum()


def perdida_coral(origen: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    """Alinea medias y covarianzas de las representaciones sin usar etiquetas demo."""
    media_origen = origen.mean(dim=0)
    media_objetivo = objetivo.mean(dim=0)
    centrado_origen = origen - media_origen
    centrado_objetivo = objetivo - media_objetivo
    cov_origen = centrado_origen.T @ centrado_origen / max(len(origen) - 1, 1)
    cov_objetivo = centrado_objetivo.T @ centrado_objetivo / max(len(objetivo) - 1, 1)
    return (media_origen - media_objetivo).square().mean() + 0.25 * (cov_origen - cov_objetivo).square().mean()


def perdida_retencion_67(logits: torch.Tensor, logits_maestro: torch.Tensor,
                         etiquetas: torch.Tensor) -> torch.Tensor:
    """Conserva el margen 6/7 del maestro cuando acertaba la etiqueta real.

    El margen objetivo se limita a 2 logits para no perseguir la confianza extrema
    del maestro. Los ejemplos que el maestro confundía no reciben esta pérdida.
    """
    mascara = (etiquetas == 6) | (etiquetas == 7)
    if not bool(mascara.any()):
        return logits.sum() * 0.0
    signo = torch.where(etiquetas[mascara] == 6, 1.0, -1.0)
    margen_maestro = signo * (logits_maestro[mascara, 6] - logits_maestro[mascara, 7])
    aciertos = margen_maestro > 0
    if not bool(aciertos.any()):
        return logits.sum() * 0.0
    margen_alumno = signo * (logits[mascara, 6] - logits[mascara, 7])
    objetivo = margen_maestro[aciertos].detach().clamp(max=2.0)
    return F.relu(objetivo - margen_alumno[aciertos]).mean()


def nota_oficial(confusion: np.ndarray) -> float:
    """Fórmula del concurso sobre una matriz de confusión, sin el efecto de la velocidad"""
    n = confusion.sum(axis=1)
    c = np.diag(confusion)
    t = np.sqrt(n) * (c - 0.33 * (n - c)) / np.maximum(n, 1) * (c / np.maximum(n, 1)) ** 2
    return float(np.exp(np.log(np.maximum(0.01, t)).mean()))


def comprobar_disco(min_libre_gib: float) -> None:
    if min_libre_gib and shutil.disk_usage(ROOT).free < min_libre_gib * 1024 ** 3:
        raise RuntimeError(f"Espacio libre por debajo de {min_libre_gib:g} GiB; se corta el entrenamiento")


def preparar(datos: dict, indices: np.ndarray, sin_atajos: bool, sin_escala: bool = False,
             sin_grupo: bool = False,
             series: np.ndarray | None = None,
             nivel: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepara señales y auxiliares en bloques para no disparar la memoria"""
    partes_s, partes_a = [], []
    for inicio in range(0, len(indices), 20_000):
        bloque = indices[inicio:inicio + 20_000]
        s, a = preparar_lote(
            datos["series"][bloque] if series is None else series[inicio:inicio + 20_000],
            datos["meta__sample_rate"][bloque],
            datos["meta__level"][bloque] if nivel is None else nivel[inicio:inicio + 20_000],
            datos["meta__group"][bloque], sin_atajos=sin_atajos, sin_escala=sin_escala,
        )
        if sin_grupo:
            a[:, 14:17] = 0.0
        partes_s.append(s)
        partes_a.append(a)
    return torch.from_numpy(np.concatenate(partes_s)), torch.from_numpy(np.concatenate(partes_a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True)
    parser.add_argument("--muestras", type=int, default=90_000)
    parser.add_argument("--indices-entrenamiento", default=None,
                        help="Archivo .npy de índices de train_90k; admite repeticiones para remuestreo")
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aumento-ruido", action="store_true", help="Añade ruido y huecos nuevos a los lotes de entrenamiento")
    parser.add_argument("--simular-examen", action="store_true", help="Pasa cada época por el simulador de examen")
    parser.add_argument("--sin-atajos", action="store_true", help="Anula las medias y el level en las auxiliares")
    parser.add_argument("--sin-grupo", action="store_true",
                        help="Anula las tres auxiliares de grupo; útil al reservar un grupo entero")
    parser.add_argument("--prob-grupo-cero", type=float, default=0.0,
                        help="Probabilidad de ocultar el indicador de grupo por muestra solo al entrenar")
    parser.add_argument("--sin-escala", action="store_true", help="Anula los tamaños de canal en las auxiliares")
    parser.add_argument("--prob-escala-cero", type=float, default=0.0,
                        help="Probabilidad de ocultar los tamaños de canal por muestra solo durante el entrenamiento")
    parser.add_argument("--equilibrar", action="store_true", help="Todas las clases pesan igual en la pérdida")
    parser.add_argument("--peso-clase-6", type=float, default=1.0,
                        help="Multiplicador adicional de la clase 6 sobre los pesos equilibrados")
    parser.add_argument("--igualar-correlacion-67", action="store_true",
                        help="Iguala la correlación de canales 0–1 entre clases 6 y 7 solo al entrenar")
    parser.add_argument("--peso-consistencia-vistas", type=float, default=0.0,
                        help="Fuerza predicciones parecidas para dos simulaciones de cada muestra (0 desactiva)")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mezcla señales y etiquetas simuladas; 0 desactiva")
    parser.add_argument("--peso-alineacion-demo", type=float, default=0.0,
                        help="Alinea representaciones de train simulado con demo sin usar sus etiquetas; 0 desactiva")
    parser.add_argument("--inicializar-pesos", default=None,
                        help="Checkpoint .pt del que parte esta corrida; no se modifica")
    parser.add_argument("--peso-destilacion", type=float, default=0.0,
                        help="Peso de KL frente al cuádruple fijo exp033+034+051+053; 0 desactiva")
    parser.add_argument("--peso-retencion-67", type=float, default=0.0,
                        help="Conserva márgenes correctos 6/7 de exp033 con su sesgo fijo; 0 desactiva")
    parser.add_argument("--temperatura-destilacion", type=float, default=2.0)
    parser.add_argument("--elegir-epoca", choices=("control", "examen"), default="control",
                        help="control: menor pérdida en el 10 %% interno; examen: mejor nota en --archivo-examen")
    parser.add_argument("--archivo-examen", default="data/val_examen_v2.npz",
                        help="Examen de prácticas para elegir la época; val_examen.npz es el de exp022, con huella")
    parser.add_argument("--ancho", type=int, default=1, help="Multiplica los filtros de la red")
    parser.add_argument("--red", choices=("base", "compacta", "multiescala", "dos_rutas", "dos_rutas_gate"), default="base",
                        help="Arquitectura CNN; dos_rutas combina mezcla temprana y formas por canal")
    parser.add_argument("--fusion-tardia", action="store_true",
                        help="Convoluciones independientes por canal; fusiona solo tras resumirlos")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--semilla", type=int, default=2026)
    parser.add_argument("--min-libre-gib", type=float, default=0.0,
                        help="Corta el entrenamiento si el disco baja de este umbral (0 desactiva)")
    parser.add_argument("--ajustes-simulador", default="{}",
                        help='JSON con valores de PARAMETROS que cambian, por ejemplo {"prob_ritmo_23": 0.5}')
    args = parser.parse_args()
    try:
        ajustes = json.loads(args.ajustes_simulador)
    except json.JSONDecodeError as error:
        parser.error(f"--ajustes-simulador no es JSON válido: {error}")
    desconocidos = sorted(set(ajustes) - set(PARAMETROS))
    if not isinstance(ajustes, dict) or desconocidos:
        parser.error(f"--ajustes-simulador tiene claves desconocidas: {desconocidos}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", args.id):
        parser.error("--id debe usar letras minúsculas, números, guiones o guiones bajos")
    if args.epocas < 1 or args.batch_size < 1 or args.ancho < 1:
        parser.error("--epocas, --batch-size y --ancho deben ser positivos")
    if args.min_libre_gib < 0 or not np.isfinite(args.min_libre_gib):
        parser.error("--min-libre-gib debe ser finito y no negativo")
    if args.red != "base" and args.fusion_tardia:
        parser.error("--fusion-tardia solo existe para la red base")
    if args.red == "dos_rutas_gate" and any((args.peso_consistencia_vistas, args.mixup_alpha,
                                              args.peso_alineacion_demo)):
        parser.error("dos_rutas_gate no se combina con otras pérdidas experimentales")
    if not 0.0 <= args.prob_escala_cero <= 1.0:
        parser.error("--prob-escala-cero debe estar entre 0 y 1")
    if not 0.0 <= args.prob_grupo_cero <= 1.0 or (args.sin_grupo and args.prob_grupo_cero):
        parser.error("--prob-grupo-cero debe estar entre 0 y 1 y no se combina con --sin-grupo")
    if args.sin_escala and args.prob_escala_cero:
        parser.error("--sin-escala y --prob-escala-cero no se pueden combinar")
    if args.peso_clase_6 <= 0 or (args.peso_clase_6 != 1.0 and not args.equilibrar):
        parser.error("--peso-clase-6 debe ser positivo y requiere --equilibrar si cambia de 1")
    if args.peso_consistencia_vistas < 0 or not np.isfinite(args.peso_consistencia_vistas):
        parser.error("--peso-consistencia-vistas debe ser finito y no negativo")
    if args.peso_consistencia_vistas and not args.simular_examen:
        parser.error("--peso-consistencia-vistas requiere --simular-examen")
    if any(valor < 0 or not np.isfinite(valor) for valor in (args.mixup_alpha, args.peso_alineacion_demo)):
        parser.error("--mixup-alpha y --peso-alineacion-demo deben ser finitos y no negativos")
    if (args.mixup_alpha or args.peso_alineacion_demo) and not args.simular_examen:
        parser.error("mixup y alineación demo requieren --simular-examen")
    if sum(bool(valor) for valor in (args.peso_consistencia_vistas, args.mixup_alpha, args.peso_alineacion_demo)) > 1:
        parser.error("Elige solo una de consistencia, mixup o alineación demo para atribuir el resultado")
    if not 0.0 <= args.peso_destilacion < 1.0 or not np.isfinite(args.peso_destilacion):
        parser.error("--peso-destilacion debe estar entre 0 (incluido) y 1 (excluido)")
    if args.peso_retencion_67 < 0 or not np.isfinite(args.peso_retencion_67):
        parser.error("--peso-retencion-67 debe ser finito y no negativo")
    if args.temperatura_destilacion <= 0 or not np.isfinite(args.temperatura_destilacion):
        parser.error("--temperatura-destilacion debe ser positiva")
    if args.peso_destilacion and (args.red != "base" or not args.inicializar_pesos
            or not args.simular_examen or not args.sin_atajos or args.sin_grupo or args.sin_escala
            or args.peso_consistencia_vistas or args.mixup_alpha or args.peso_alineacion_demo
            or args.aumento_ruido or args.igualar_correlacion_67 or args.prob_grupo_cero
            or args.prob_escala_cero):
        parser.error("La destilación fija requiere CNN base inicializada, simulador, sin atajos y sin otras pérdidas o perturbaciones")
    if args.peso_retencion_67 and (args.red != "base" or args.inicializar_pesos != "models/exp033.pt"
            or not args.simular_examen or not args.sin_atajos or not args.equilibrar
            or args.peso_destilacion or args.peso_consistencia_vistas or args.mixup_alpha
            or args.peso_alineacion_demo or args.aumento_ruido or args.igualar_correlacion_67
            or args.sin_grupo or args.sin_escala or args.prob_grupo_cero or args.prob_escala_cero):
        parser.error("La retención 6/7 requiere la receta base de afinamiento exp033 sin otras intervenciones")

    fuente = ROOT / "data/train_90k.npz"
    examen = ROOT / args.archivo_examen
    if args.elegir_epoca == "examen" and not examen.is_file():
        parser.error(f"Falta {args.archivo_examen}: créalo con scripts/calibrar_simulador.py --guardar-val")
    print(f"Preparando {args.id}: cargando datos y red...", flush=True)
    datos = dict(np.load(fuente))
    total = len(datos["labels"])
    if not 80 <= args.muestras <= total:
        parser.error(f"--muestras debe estar entre 80 y {total}")
    destino = ROOT / "models" / f"{args.id}.pt"
    carpeta = ROOT / "experiments" / args.id
    informe = carpeta / "entrenamiento.json"
    if destino.exists() or informe.exists():
        parser.error(f"El experimento {args.id} ya existe; usa otro --id")

    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    rng = np.random.default_rng(args.semilla)
    ruta_indices = ROOT / args.indices_entrenamiento if args.indices_entrenamiento else None
    if ruta_indices is not None:
        if ruta_indices.suffix != ".npy" or not ruta_indices.resolve().is_relative_to(ROOT):
            parser.error("--indices-entrenamiento debe ser un .npy dentro del repositorio")
        if not ruta_indices.is_file():
            parser.error(f"Falta {ruta_indices}")
        sha_indices = hashlib.sha256(ruta_indices.read_bytes()).hexdigest()
        indices = np.load(ruta_indices, allow_pickle=False)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < args.muestras:
            parser.error("--indices-entrenamiento debe ser un vector entero con al menos --muestras filas")
        if np.any(indices >= total) or np.any(indices < 0):
            parser.error("--indices-entrenamiento contiene índices fuera de train_90k")
    else:
        indices = np.arange(total)
        sha_indices = None
    if args.muestras < len(indices):
        indices, _ = train_test_split(indices, train_size=args.muestras,
                                      stratify=datos["labels"][indices], random_state=2026)
    indices = np.sort(indices)
    etiquetas_todas = np.asarray(datos["labels"][indices], dtype=np.int64)

    # Con control interno se reserva un 10 %; eligiendo con val_examen se entrena con todo
    if args.elegir_epoca == "control":
        entrenamiento_pos, control_pos = train_test_split(
            np.arange(len(indices)), test_size=0.1, stratify=etiquetas_todas, random_state=2027
        )
    else:
        entrenamiento_pos, control_pos = np.arange(len(indices)), np.array([], dtype=np.int64)
    entrenamiento_idx = indices[np.sort(entrenamiento_pos)]
    etiquetas = np.asarray(datos["labels"][entrenamiento_idx], dtype=np.int64)

    inicio_total = perf_counter()
    comprobar_disco(args.min_libre_gib)
    if args.elegir_epoca == "control":
        control_idx = indices[np.sort(control_pos)]
        control_s, control_a = preparar(datos, control_idx, args.sin_atajos, args.sin_escala, args.sin_grupo)
        control_y = torch.from_numpy(np.asarray(datos["labels"][control_idx], dtype=np.int64))
        nombre_control = "control interno"
    else:
        datos_examen = dict(np.load(examen))
        todos = np.arange(len(datos_examen["labels"]))
        control_s, control_a = preparar(datos_examen, todos, args.sin_atajos, args.sin_escala, args.sin_grupo)
        control_y = torch.from_numpy(np.asarray(datos_examen["labels"], dtype=np.int64))
        nombre_control = "examen de prácticas"
    fijas = None if args.simular_examen else preparar(datos, entrenamiento_idx, args.sin_atajos, args.sin_escala, args.sin_grupo)
    objetivo_s = objetivo_a = None
    if args.peso_alineacion_demo:
        ruta_demo = ROOT / "data/test_demo.npz"
        if not ruta_demo.is_file():
            parser.error(f"Falta {ruta_demo}")
        datos_demo = dict(np.load(ruta_demo))
        datos_demo.pop("labels", None)
        indices_demo = np.arange(len(datos_demo["series"]))
        objetivo_s, objetivo_a = preparar(datos_demo, indices_demo, args.sin_atajos, args.sin_escala, args.sin_grupo)
    segundos_preparacion = perf_counter() - inicio_total

    dispositivo = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if objetivo_s is not None:
        objetivo_s = objetivo_s.to(dispositivo)
        objetivo_a = objetivo_a.to(dispositivo)
    red = (RedMultiescala(ancho=args.ancho) if args.red == "multiescala"
           else RedDosRutas(ancho=args.ancho) if args.red == "dos_rutas"
           else RedDosRutasGate(ancho=args.ancho) if args.red == "dos_rutas_gate"
           else RedCompacta() if args.red == "compacta"
           else RedConvolucional(ancho=args.ancho, fusion_tardia=args.fusion_tardia)).to(dispositivo)
    if args.red == "compacta" and args.ancho != 1:
        parser.error("La red compacta solo admite --ancho 1")
    sha_inicial = None
    if args.inicializar_pesos:
        ruta_inicial = ROOT / args.inicializar_pesos
        if ruta_inicial.suffix != ".pt" or not ruta_inicial.resolve().is_relative_to(ROOT) or not ruta_inicial.is_file():
            parser.error("--inicializar-pesos debe señalar un .pt existente dentro del repositorio")
        inicial = torch.load(ruta_inicial, map_location="cpu", weights_only=True)
        if inicial.get("formato") != "gdg-cnn-v1":
            parser.error("El checkpoint inicial no tiene el formato CNN esperado")
        for campo, esperado in (("red", args.red), ("ancho", args.ancho),
                                ("sin_atajos", args.sin_atajos), ("sin_escala", args.sin_escala),
                                ("sin_grupo", args.sin_grupo)):
            valor = inicial.get(campo, "base" if campo == "red" else False)
            if valor != esperado:
                parser.error(f"El checkpoint inicial tiene {campo}={valor!r}; esperado {esperado!r}")
        red.load_state_dict(inicial["state_dict"])
        sha_inicial = hashlib.sha256(ruta_inicial.read_bytes()).hexdigest()
    profesores = []
    sha_profesores = {}
    maestro_retencion = sesgo_retencion = None
    sha_sesgo_retencion = None
    if args.peso_retencion_67:
        ruta_sesgo = ROOT / "models/exp033.sesgo.json"
        if not ruta_sesgo.is_file():
            parser.error(f"Falta {ruta_sesgo}")
        sesgo_info = json.loads(ruta_sesgo.read_text(encoding="utf-8"))
        if (sesgo_info.get("formato") != "gdg-sesgo-v1"
                or sesgo_info.get("pesos") != "models/exp033.pt"
                or len(sesgo_info.get("sesgo", [])) != 8):
            parser.error("El sesgo de exp033 no tiene el formato esperado")
        sesgo_retencion = torch.as_tensor(sesgo_info["sesgo"], dtype=torch.float32, device=dispositivo)
        maestro_retencion = RedConvolucional(ancho=args.ancho).to(dispositivo)
        maestro_retencion.load_state_dict(inicial["state_dict"])
        maestro_retencion.eval().requires_grad_(False)
        sha_sesgo_retencion = hashlib.sha256(ruta_sesgo.read_bytes()).hexdigest()
    if args.peso_destilacion:
        from evaluar_ensamble_calidad import cargar

        for nombre in ("exp033", "exp034", "exp051", "exp053"):
            maestro, artefacto, sesgo = cargar(nombre)
            if not artefacto.get("sin_atajos") or artefacto.get("sin_escala") or artefacto.get("sin_grupo"):
                parser.error(f"{nombre}: preparación distinta de la fijada para destilar")
            profesores.append((maestro.to(dispositivo).eval(), torch.as_tensor(sesgo, device=dispositivo)))
            for ruta in (ROOT / f"models/{nombre}.pt", ROOT / f"models/{nombre}.sesgo.json"):
                if ruta.is_file():
                    sha_profesores[str(ruta.relative_to(ROOT))] = hashlib.sha256(ruta.read_bytes()).hexdigest()
    # Cargar los maestros construye capas y consumiría números aleatorios solo
    # en la candidata. Reiniciar mantiene iguales simulaciones y lotes del control.
    torch.manual_seed(args.semilla)
    optimizador = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=0.01)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)
    conteo = np.bincount(etiquetas, minlength=8)
    pesos_clase = torch.tensor(len(etiquetas) / (8 * conteo), dtype=torch.float32, device=dispositivo) if args.equilibrar else None
    if pesos_clase is not None:
        pesos_clase[6] *= args.peso_clase_6
    criterio = nn.CrossEntropyLoss(weight=pesos_clase)
    criterio_control = nn.CrossEntropyLoss()
    mejor = {"epoca": 0, "perdida": float("inf"), "nota": -1.0}
    mejores_pesos = None
    historial = []
    segundos_simulador = 0.0
    inicio_ajuste = perf_counter()
    carpeta.mkdir(parents=True, exist_ok=True)
    progreso = carpeta / "progreso.json"
    print(f"{args.id}: datos preparados en {segundos_preparacion:.1f} s; iniciando {args.epocas} épocas", flush=True)

    for epoca in range(1, args.epocas + 1):
        comprobar_disco(args.min_libre_gib)
        inicio_epoca = perf_counter()
        if args.simular_examen:
            # Cada época ve una versión distinta y sucia de las mismas muestras
            antes = perf_counter()
            series, niveles = [], []
            series_2, niveles_2 = [], []
            for inicio in range(0, len(entrenamiento_idx), 10_000):
                bloque = entrenamiento_idx[inicio:inicio + 10_000]
                s, nv = simular_examen(datos["series"][bloque], rng, parametros=ajustes)
                series.append(s)
                niveles.append(nv)
                if args.peso_consistencia_vistas:
                    s_2, nv_2 = simular_examen(datos["series"][bloque], rng, parametros=ajustes)
                    series_2.append(s_2)
                    niveles_2.append(nv_2)
            serie_e, aux_e = preparar(datos, entrenamiento_idx, args.sin_atajos, args.sin_escala, args.sin_grupo,
                                      series=np.concatenate(series), nivel=np.concatenate(niveles))
            if args.peso_consistencia_vistas:
                serie_e_2, aux_e_2 = preparar(datos, entrenamiento_idx, args.sin_atajos, args.sin_escala, args.sin_grupo,
                                              series=np.concatenate(series_2), nivel=np.concatenate(niveles_2))
            del series, niveles, series_2, niveles_2
            segundos_simulador += perf_counter() - antes
        else:
            serie_e, aux_e = fijas
        if args.peso_consistencia_vistas:
            conjunto = TensorDataset(serie_e, aux_e, serie_e_2, aux_e_2, torch.from_numpy(etiquetas))
        else:
            conjunto = TensorDataset(serie_e, aux_e, torch.from_numpy(etiquetas))
        cargador = DataLoader(conjunto,
                              batch_size=args.batch_size, shuffle=True, num_workers=0)

        red.train()
        perdida_entrenamiento = 0.0
        suma_supervisada = 0.0
        suma_consistencia = 0.0
        suma_alineacion = 0.0
        suma_destilacion = 0.0
        suma_retencion_67 = 0.0
        for numero_lote, lote in enumerate(cargador):
            if numero_lote % 25 == 0:
                comprobar_disco(args.min_libre_gib)
            if args.peso_consistencia_vistas:
                serie, auxiliar, serie_2, auxiliar_2, etiqueta = lote
                serie_2 = serie_2.to(dispositivo)
                auxiliar_2 = auxiliar_2.to(dispositivo)
            else:
                serie, auxiliar, etiqueta = lote
            serie = serie.to(dispositivo)
            auxiliar = auxiliar.to(dispositivo)
            etiqueta = etiqueta.to(dispositivo)
            if args.prob_escala_cero:
                auxiliar = auxiliar.clone()
                ocultar = torch.rand(len(auxiliar), device=dispositivo) < args.prob_escala_cero
                auxiliar[ocultar, 4:8] = 0.0
                if args.peso_consistencia_vistas:
                    auxiliar_2 = auxiliar_2.clone()
                    auxiliar_2[ocultar, 4:8] = 0.0
            if args.prob_grupo_cero:
                auxiliar = auxiliar.clone()
                ocultar_grupo = torch.rand(len(auxiliar), device=dispositivo) < args.prob_grupo_cero
                auxiliar[ocultar_grupo, 14:17] = 0.0
                if args.peso_consistencia_vistas:
                    auxiliar_2 = auxiliar_2.clone()
                    auxiliar_2[ocultar_grupo, 14:17] = 0.0
            if args.aumento_ruido:
                serie, auxiliar = aumentar_senales(serie, auxiliar)
                if args.peso_consistencia_vistas:
                    serie_2, auxiliar_2 = aumentar_senales(serie_2, auxiliar_2)
            if args.igualar_correlacion_67:
                serie = igualar_correlacion_67(serie, etiqueta)
                if args.peso_consistencia_vistas:
                    serie_2 = igualar_correlacion_67(serie_2, etiqueta)
            optimizador.zero_grad(set_to_none=True)
            if args.mixup_alpha:
                proporcion = float(rng.beta(args.mixup_alpha, args.mixup_alpha))
                permutacion = torch.randperm(len(etiqueta), device=dispositivo)
                mezclada = proporcion * serie + (1.0 - proporcion) * serie[permutacion]
                aux_mezclada = proporcion * auxiliar + (1.0 - proporcion) * auxiliar[permutacion]
                logits = red(mezclada, aux_mezclada)
                perdida_1 = F.cross_entropy(logits, etiqueta, weight=pesos_clase, reduction="none")
                perdida_2 = F.cross_entropy(logits, etiqueta[permutacion], weight=pesos_clase, reduction="none")
                if pesos_clase is None:
                    normalizador = len(etiqueta)
                else:
                    normalizador = (proporcion * pesos_clase[etiqueta]
                                   + (1.0 - proporcion) * pesos_clase[etiqueta[permutacion]]).sum()
                perdida = (proporcion * perdida_1 + (1.0 - proporcion) * perdida_2).sum() / normalizador
            elif args.peso_alineacion_demo:
                indices_objetivo = torch.randint(len(objetivo_s), (len(etiqueta),), device=dispositivo)
                logits, oculto_origen = red(serie, auxiliar, devolver_oculto=True)
                _, oculto_objetivo = red(objetivo_s[indices_objetivo], objetivo_a[indices_objetivo],
                                         devolver_oculto=True)
                supervisada = criterio(logits, etiqueta)
                alineacion = perdida_coral(oculto_origen, oculto_objetivo)
                perdida = supervisada + args.peso_alineacion_demo * alineacion
                suma_supervisada += supervisada.item() * len(etiqueta)
                suma_alineacion += alineacion.item() * len(etiqueta)
            elif args.red == "dos_rutas_gate":
                logits, logits_conjunta, logits_canal, _ = red(serie, auxiliar, devolver_componentes=True)
                perdida = (criterio(logits, etiqueta)
                           + 0.25 * (criterio(logits_conjunta, etiqueta) + criterio(logits_canal, etiqueta)))
            else:
                logits = red(serie, auxiliar)
            if args.peso_consistencia_vistas:
                logits_2 = red(serie_2, auxiliar_2)
                supervisada = 0.5 * (criterio(logits, etiqueta) + criterio(logits_2, etiqueta))
                consistencia = divergencia_vistas(logits, logits_2, etiqueta, pesos_clase)
                perdida = supervisada + args.peso_consistencia_vistas * consistencia
                suma_supervisada += supervisada.item() * len(etiqueta)
                suma_consistencia += consistencia.item() * len(etiqueta)
            elif not args.mixup_alpha and not args.peso_alineacion_demo and args.red != "dos_rutas_gate":
                perdida = criterio(logits, etiqueta)
            if args.peso_destilacion:
                temperatura = args.temperatura_destilacion
                # KL necesita guardar el objetivo para derivar respecto al alumno.
                # no_grad congela maestros sin crear tensores de inference_mode.
                with torch.no_grad():
                    probabilidades = torch.stack([
                        F.softmax((maestro(serie, auxiliar) + sesgo) / temperatura, dim=1)
                        for maestro, sesgo in profesores
                    ]).mean(dim=0)
                destilacion = F.kl_div(F.log_softmax(logits / temperatura, dim=1),
                                       probabilidades, reduction="batchmean") * temperatura ** 2
                perdida = (1.0 - args.peso_destilacion) * perdida + args.peso_destilacion * destilacion
                suma_destilacion += destilacion.item() * len(etiqueta)
            if args.peso_retencion_67:
                with torch.no_grad():
                    logits_maestro = maestro_retencion(serie, auxiliar) + sesgo_retencion
                retencion = perdida_retencion_67(logits, logits_maestro, etiqueta)
                perdida = perdida + args.peso_retencion_67 * retencion
                suma_retencion_67 += retencion.item() * len(etiqueta)
            perdida.backward()
            optimizador.step()
            perdida_entrenamiento += perdida.item() * len(etiqueta)
        plan.step()

        red.eval()
        perdida_control = 0.0
        confusion = np.zeros((8, 8), dtype=np.int64)
        with torch.inference_mode():
            for inicio in range(0, len(control_y), 2048):
                fin = inicio + 2048
                y = control_y[inicio:fin].to(dispositivo)
                salida = red(control_s[inicio:fin].to(dispositivo), control_a[inicio:fin].to(dispositivo))
                perdida_control += criterio_control(salida, y).item() * len(y)
                np.add.at(confusion, (y.cpu().numpy(), salida.argmax(dim=1).cpu().numpy()), 1)
        perdida_control /= len(control_y)
        acierto_clase = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
        fila = {
            "epoca": epoca,
            "perdida_entrenamiento": perdida_entrenamiento / len(etiquetas),
            "perdida_supervisada": suma_supervisada / len(etiquetas) if (args.peso_consistencia_vistas or args.peso_alineacion_demo) else None,
            "divergencia_vistas": suma_consistencia / len(etiquetas) if args.peso_consistencia_vistas else None,
            "perdida_alineacion_demo": suma_alineacion / len(etiquetas) if args.peso_alineacion_demo else None,
            "perdida_destilacion": suma_destilacion / len(etiquetas) if args.peso_destilacion else None,
            "perdida_retencion_67": suma_retencion_67 / len(etiquetas) if args.peso_retencion_67 else None,
            "perdida_control": perdida_control,
            "acierto_control": float(np.trace(confusion) / confusion.sum()),
            "nota_control": nota_oficial(confusion),
            "acierto_por_clase": [round(float(v), 4) for v in acierto_clase],
            "segundos": perf_counter() - inicio_epoca,
        }
        historial.append(fila)
        mejora = (fila["nota_control"] > mejor["nota"]) if args.elegir_epoca == "examen" else (perdida_control < mejor["perdida"])
        if mejora:
            mejor = {"epoca": epoca, "perdida": perdida_control, "nota": fila["nota_control"]}
            mejores_pesos = {nombre: tensor.detach().cpu().clone() for nombre, tensor in red.state_dict().items()}
        print(
            f"Época {epoca}/{args.epocas}: pérdida {fila['perdida_entrenamiento']:.4f} · {nombre_control}: "
            f"acierto {fila['acierto_control']:.2%}, nota {fila['nota_control']:.3f}, clase 6 {acierto_clase[6]:.0%}"
            f"{' ← mejor' if mejora else ''} ({fila['segundos']:.0f} s)",
            flush=True,
        )
        # Estado en directo para seguir el entrenamiento desde fuera
        progreso.write_text(json.dumps({
            "id": args.id, "epoca": epoca, "epocas": args.epocas, "mejor_epoca": mejor["epoca"],
            "elegir_epoca": args.elegir_epoca, "historial": historial,
            "segundos_transcurridos": perf_counter() - inicio_total,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.simular_examen:
            del cargador, conjunto, serie_e, aux_e
            if args.peso_consistencia_vistas:
                del serie_e_2, aux_e_2

    segundos_ajuste = perf_counter() - inicio_ajuste
    destino.parent.mkdir(exist_ok=True)
    torch.save({
        "formato": "gdg-cnn-v1", "state_dict": mejores_pesos,
        "red": args.red,
        "ancho": args.ancho, "batch_size": args.batch_size,
        "sin_atajos": args.sin_atajos, "sin_escala": args.sin_escala,
        "sin_grupo": args.sin_grupo,
        "prob_grupo_cero": args.prob_grupo_cero,
        "fusion_tardia": args.fusion_tardia,
        "prob_escala_cero": args.prob_escala_cero,
        "peso_clase_6": args.peso_clase_6,
        "igualar_correlacion_67": args.igualar_correlacion_67,
        "peso_consistencia_vistas": args.peso_consistencia_vistas,
        "mixup_alpha": args.mixup_alpha,
        "peso_alineacion_demo": args.peso_alineacion_demo,
        "inicializar_pesos": args.inicializar_pesos,
        "sha256_inicializar_pesos": sha_inicial,
        "peso_destilacion": args.peso_destilacion,
        "peso_retencion_67": args.peso_retencion_67,
        "sha256_sesgo_retencion": sha_sesgo_retencion,
        "temperatura_destilacion": args.temperatura_destilacion,
        "sha256_profesores": sha_profesores,
        "min_libre_gib": args.min_libre_gib,
    }, destino)
    resumen = {
        "modelo": ("Red multiescala 1D" if args.red == "multiescala"
                   else "Red compacta 1D" if args.red == "compacta"
                   else "Red de dos rutas con puerta de calidad" if args.red == "dos_rutas_gate"
                   else "Red de dos rutas 1D" if args.red == "dos_rutas" else "Red convolucional 1D")
                  + (" · simulador de examen" if args.simular_examen else "")
                  + (" · destilación cuádruple" if args.peso_destilacion else ""
                     if not args.inicializar_pesos else " · reajuste desde checkpoint"),
        "id": args.id,
        "fuente": fuente.name,
        "indices_entrenamiento": str(ruta_indices.relative_to(ROOT)) if ruta_indices else None,
        "sha256_indices_entrenamiento": sha_indices,
        "muestras_seleccionadas": int(len(indices)),
        "muestras_entrenamiento": int(len(entrenamiento_idx)),
        "muestras_control_interno": int(len(control_pos)),
        "aumento_ruido": args.aumento_ruido,
        "simular_examen": args.simular_examen,
        "parametros_simulador": {**PARAMETROS, **ajustes} if args.simular_examen else None,
        "sin_atajos": args.sin_atajos,
        "sin_escala": args.sin_escala,
        "sin_grupo": args.sin_grupo,
        "prob_grupo_cero": args.prob_grupo_cero,
        "prob_escala_cero": args.prob_escala_cero,
        "peso_clase_6": args.peso_clase_6,
        "igualar_correlacion_67": args.igualar_correlacion_67,
        "peso_consistencia_vistas": args.peso_consistencia_vistas,
        "mixup_alpha": args.mixup_alpha,
        "peso_alineacion_demo": args.peso_alineacion_demo,
        "inicializar_pesos": args.inicializar_pesos,
        "sha256_inicializar_pesos": sha_inicial,
        "peso_destilacion": args.peso_destilacion,
        "peso_retencion_67": args.peso_retencion_67,
        "sha256_sesgo_retencion": sha_sesgo_retencion,
        "temperatura_destilacion": args.temperatura_destilacion,
        "sha256_profesores": sha_profesores,
        "min_libre_gib": args.min_libre_gib,
        "equilibrar": args.equilibrar,
        "elegir_epoca": args.elegir_epoca,
        "archivo_examen": args.archivo_examen if args.elegir_epoca == "examen" else None,
        "ancho": args.ancho,
        "batch_size": args.batch_size,
        "red": args.red,
        "fusion_tardia": args.fusion_tardia,
        "lr": args.lr,
        "clases_entrenamiento": conteo.tolist(),
        "caracteristicas_por_muestra": 4 * 256 + 17,
        "iteraciones": args.epocas,
        "mejor_epoca": mejor["epoca"],
        "perdida_control_interno": mejor["perdida"],
        "nota_control_mejor": mejor["nota"],
        "historial": historial,
        "semilla": args.semilla,
        "dispositivo_entrenamiento": str(dispositivo),
        "segundos_caracteristicas": segundos_preparacion + segundos_simulador,
        "segundos_ajuste": segundos_ajuste - segundos_simulador,
        "segundos_entrenamiento_total": segundos_preparacion + segundos_ajuste,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pesos": str(destino.relative_to(ROOT)),
    }
    informe.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nMejor época: {mejor['epoca']} · nota en {nombre_control} {mejor['nota']:.3f} · "
          f"{resumen['segundos_entrenamiento_total'] / 60:.1f} min · {destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
