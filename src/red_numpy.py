"""La misma CNN de red_convolucional.py, pero calculada solo con NumPy para predecir más rápido.

Da las mismas clases que PyTorch (diferencias de ~1e-5 en la salida) y tarda ~35 % menos
por muestra, porque se ahorra el coste fijo de llamar a PyTorch en cada capa. Además la
entrega no necesita PyTorch. Los pesos salen de scripts/exportar_numpy.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

from competition_api import Sample

TIEMPOS = np.arange(256)
# Las cuatro convoluciones de RedConvolucional: (tamaño de ventana, relleno); todas con paso 2
CONVOLUCIONES = [(9, 4), (7, 3), (5, 2), (3, 1)]


def preparar_rapida(sample: Sample, sin_atajos: bool) -> tuple[np.ndarray, np.ndarray]:
    """Igual que preparar_muestra, pero sin trabajo extra cuando un canal no tiene huecos"""
    serie = np.asarray(sample.series, dtype=np.float32)
    if serie.shape != (4, 256):
        raise ValueError(f"Se esperaba una señal (4, 256), recibida {serie.shape}")
    ausentes = ~np.isfinite(serie)
    if ausentes.any():
        completa = serie.copy()
        for canal in np.flatnonzero(ausentes.any(axis=1)):
            presentes = ~ausentes[canal]
            if presentes.any():
                completa[canal] = np.interp(TIEMPOS, TIEMPOS[presentes], serie[canal, presentes])
            else:
                completa[canal] = 0.0
        faltan = ausentes.mean(axis=1)
    else:
        completa = serie
        faltan = np.zeros(4)

    medias = completa.mean(axis=1)
    desviaciones = completa.std(axis=1)
    normalizada = (completa - medias[:, None]) / (desviaciones[:, None] + 1e-6)
    np.clip(normalizada, -6.0, 6.0, out=normalizada)

    # Mismo orden de auxiliares que preparar_muestra
    tasa = float(sample.meta.get("sample_rate", 0.0))
    grupo = str(sample.meta.get("group", ""))
    auxiliares = np.zeros(17, dtype=np.float32)
    auxiliares[4:8] = np.log1p(desviaciones)
    auxiliares[8:12] = faltan
    auxiliares[12] = np.sign(tasa) * np.log1p(abs(tasa))
    if grupo in ("A", "B", "C"):
        auxiliares[14 + "ABC".index(grupo)] = 1.0
    if not sin_atajos:
        nivel = float(sample.meta.get("level", 0.0))
        auxiliares[0:4] = np.arcsinh(medias)
        auxiliares[13] = np.sign(nivel) * np.log1p(abs(nivel))
    auxiliares = np.nan_to_num(auxiliares, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return normalizada.astype(np.float32, copy=False), auxiliares


class RedNumpy:
    """Calcula la salida de RedConvolucional con los pesos exportados a .npz"""

    def __init__(self, ruta: Path) -> None:
        with np.load(ruta, allow_pickle=False) as datos:
            if str(datos["formato"]) != "gdg-numpy-v1":
                raise ValueError(f"Formato de pesos NumPy desconocido: {ruta}")
            pesos = {clave: datos[clave].astype(np.float32) for clave in datos.files
                     if clave.startswith(("convoluciones.", "salida."))}
            self.sin_atajos = bool(datos["sin_atajos"])
            self.sesgo = datos["sesgo"].astype(np.float32)

        # Cada convolución pasa a una multiplicación de matrices sobre ventanas (posición, ventana·canal)
        self.capas = []
        for numero, (ventana, relleno) in zip((0, 2, 4, 6), CONVOLUCIONES):
            w = pesos[f"convoluciones.{numero}.weight"]
            salidas, canales, _ = w.shape
            matriz = np.ascontiguousarray(w.transpose(2, 1, 0).reshape(ventana * canales, salidas))
            self.capas.append((matriz, pesos[f"convoluciones.{numero}.bias"], ventana, relleno))
        self.filtros = salidas

        # PyTorch aplana el resumen como canal·4 + trozo; aquí lo tenemos como trozo·filtros + canal
        w1 = pesos["salida.0.weight"]
        orden = np.array([c * 4 + p for p in range(4) for c in range(salidas)])
        self.w_media = np.ascontiguousarray(w1[:, orden].T)
        self.w_maximo = np.ascontiguousarray(w1[:, 4 * salidas + orden].T)
        self.w_auxiliares = np.ascontiguousarray(w1[:, 8 * salidas:].T)
        self.b1 = pesos["salida.0.bias"]
        self.w2 = np.ascontiguousarray(pesos["salida.2.weight"].T)
        # El sesgo de clase se suma junto al de la última capa
        self.b2 = pesos["salida.2.bias"] + self.sesgo

    def salida(self, serie: np.ndarray, auxiliares: np.ndarray) -> np.ndarray:
        x = serie.T
        for matriz, sesgo, ventana, relleno in self.capas:
            largo, canales = x.shape
            con_relleno = np.zeros((largo + 2 * relleno, canales), dtype=np.float32)
            con_relleno[relleno:relleno + largo] = x
            posiciones = (largo + 2 * relleno - ventana) // 2 + 1
            paso_fila, paso_canal = con_relleno.strides
            ventanas = as_strided(con_relleno, (posiciones, ventana, canales),
                                  (2 * paso_fila, paso_fila, paso_canal)).reshape(posiciones, ventana * canales)
            x = ventanas @ matriz
            x += sesgo
            np.maximum(x, 0.0, out=x)

        # Media y máximo en 4 trozos de 4 posiciones, como AdaptiveAvgPool1d(4) y AdaptiveMaxPool1d(4)
        trozos = x.reshape(4, -1, self.filtros)
        oculto = (trozos.mean(axis=1).ravel() @ self.w_media + trozos.max(axis=1).ravel() @ self.w_maximo
                  + auxiliares @ self.w_auxiliares + self.b1)
        np.maximum(oculto, 0.0, out=oculto)
        return oculto @ self.w2 + self.b2

    def predecir(self, sample: Sample) -> int:
        serie, auxiliares = preparar_rapida(sample, self.sin_atajos)
        return int(self.salida(serie, auxiliares).argmax())
