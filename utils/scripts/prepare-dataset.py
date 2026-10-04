"""Prepara el dataset crudo para Teachable Machine.

1. Recorta cada imagen a cuadrado de 512 px. En vez de tomar siempre el centro,
   desliza la ventana cuadrada a lo largo del lado largo y se queda con la que
   tiene más detalle (bordes y color), con una leve preferencia por el centro;
   así una jarra alta no pierde la boca.
2. Separa 120 de entrenamiento y 30 de prueba por clase. Las fotos de una misma
   serie o ejemplar (columna `grupo` de renombrado.csv, más las tomas casi
   idénticas detectadas por dHash) van completas a un solo lado.

Salida: dataset/prepared/{train,test}/<clase>/ y dataset/prepared/particion.csv.
Uso: python prepare-dataset.py  (el dataset crudo está en ../dataset/raw)
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

RAIZ = Path(__file__).resolve().parents[3] / "dataset"   # ../dataset respecto al repo
RAW = RAIZ / "raw"
OUT = RAIZ / "prepared"
CLASES = ["Dionaea", "Drosera", "Sarracenia", "Nepenthes", "Darlingtonia",
          "Heliamphora", "Pinguicula", "No carnivora"]
LADO = 512
N_PRUEBA = 30
SEMILLA = 42
UMBRAL_DHASH = 10   # bits distintos como máximo para considerar dos fotos la misma toma


def dhash(im, n=8):
    px = np.asarray(im.convert("L").resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return int("".join("1" if a > b else "0" for a, b in zip(px[:, :-1].ravel(), px[:, 1:].ravel())), 2)


def recortes_manuales():
    """Posición (0 = inicio, 1 = final del lado largo) fijada a mano para las
    pocas imágenes en las que el recorte automático no encuadra la trampa."""
    ruta = RAW / "_fuentes" / "recortes_manuales.csv"
    if not ruta.exists():
        return {}
    with ruta.open(encoding="utf-8-sig") as f:
        return {(r["clase"], r["archivo"]): float(r["posicion"]) for r in csv.DictReader(f)}


def recorte_cuadrado(im, posicion=None):
    """Devuelve el recorte cuadrado con más detalle a lo largo del lado largo."""
    w, h = im.size
    lado = min(w, h)
    if w == h:
        return im
    if posicion is not None:
        off = int(round(posicion * (max(w, h) - lado)))
        return im.crop((off, 0, off + lado, lado) if w > h else (0, off, lado, off + lado))
    chica = im.copy()
    chica.thumbnail((400, 400))
    escala = chica.size[0] / w
    bordes = np.asarray(chica.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sat = np.asarray(chica.convert("HSV"), dtype=np.float32)[..., 1]
    mapa = bordes / (bordes.max() + 1e-6) + 0.5 * sat / 255
    lado_c = int(round(lado * escala))
    largo_c = chica.size[0] if w > h else chica.size[1]
    mejor, mejor_pos = -1.0, 0
    for pos in np.linspace(0, largo_c - lado_c, 9):
        pos = int(round(pos))
        ventana = mapa[:, pos:pos + lado_c] if w > h else mapa[pos:pos + lado_c, :]
        centrado = 1 - 0.15 * abs(pos - (largo_c - lado_c) / 2) / max(1, (largo_c - lado_c) / 2)
        puntaje = ventana.mean() * centrado
        if puntaje > mejor:
            mejor, mejor_pos = puntaje, pos
    off = int(round(mejor_pos / escala))
    off = max(0, min(off, (w if w > h else h) - lado))
    caja = (off, 0, off + lado, lado) if w > h else (0, off, lado, off + lado)
    return im.crop(caja)


def grupos_por_clase(clase, archivos):
    """Grupo = serie del CSV de renombrado, unido con tomas casi idénticas (dHash)."""
    with (RAW / "_fuentes" / "renombrado.csv").open(encoding="utf-8-sig") as f:
        # en iNaturalist el "grupo" es el taxón; ahí cada observación cuenta aparte
        base = {r["nuevo"]: r["grupo"] for r in csv.DictReader(f)
                if r["clase"] == clase and r["fuente"] != "inaturalist"}
    padre = {a.name: a.name for a in archivos}

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a, b):
        padre[raiz(a)] = raiz(b)

    por_serie = defaultdict(list)
    hashes = {}
    for a in archivos:
        por_serie[base.get(a.name, a.name)].append(a.name)
        with Image.open(a) as im:
            im.draft("RGB", (320, 320))
            hashes[a.name] = dhash(ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (256, 256)))
    for miembros in por_serie.values():
        for m in miembros[1:]:
            unir(miembros[0], m)
    nombres = list(hashes)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= UMBRAL_DHASH:
                unir(a, b)
    grupos = defaultdict(list)
    for n in nombres:
        grupos[raiz(n)].append(n)
    return list(grupos.values())


def solo_entrenamiento(clase):
    """Imágenes cuyo grupo no puede ir a prueba (duplicados que contarían doble)."""
    ruta = RAW / "_fuentes" / "solo_entrenamiento.csv"
    if not ruta.exists():
        return set()
    with ruta.open(encoding="utf-8-sig") as f:
        return {r["archivo"] for r in csv.DictReader(f) if r["clase"] == clase}


def repartir(grupos, rng, excluidas=frozenset()):
    """Elige grupos al azar para prueba hasta sumar exactamente N_PRUEBA."""
    candidatos = [g for g in grupos if not excluidas & set(g)]
    for _ in range(1000):
        orden = candidatos[:]
        rng.shuffle(orden)
        prueba, total = [], 0
        for g in orden:
            if total + len(g) <= N_PRUEBA:
                prueba.extend(g)
                total += len(g)
            if total == N_PRUEBA:
                return set(prueba)
    raise RuntimeError("no se pudo completar el segmento de prueba")


def main():
    rng = random.Random(SEMILLA)
    manuales = recortes_manuales()
    filas = []
    for clase in CLASES:
        archivos = sorted(f for f in (RAW / clase).iterdir() if f.is_file())
        grupos = grupos_por_clase(clase, archivos)
        en_prueba = repartir(grupos, rng, solo_entrenamiento(clase))
        n_series = sum(1 for g in grupos if len(g) > 1)
        for particion in ("train", "test"):
            (OUT / particion / clase).mkdir(parents=True, exist_ok=True)
        for a in archivos:
            particion = "test" if a.name in en_prueba else "train"
            with Image.open(a) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = recorte_cuadrado(im, manuales.get((clase, a.name))).resize((LADO, LADO), Image.LANCZOS)
                im.save(OUT / particion / clase / a.name, "JPEG", quality=90, optimize=True)
            filas.append({"clase": clase, "archivo": a.name, "particion": particion})
        print(f"{clase}: {len(archivos) - len(en_prueba)} train / {len(en_prueba)} test, "
              f"{len(grupos)} grupos ({n_series} series de más de una foto)")
    with (OUT / "particion.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "archivo", "particion"])
        w.writeheader()
        w.writerows(filas)


if __name__ == "__main__":
    main()
