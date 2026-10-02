"""Renombra las imágenes del dataset crudo a <clase>_<nnn>.jpg.

Las fotos propias van primero y en orden alfabético, de modo que las series de
un mismo ejemplar ("DM A2 1", "DM A2 2"...) quedan consecutivas; después las de
la comunidad y las de subastas; en "No carnivora", las de iNaturalist agrupadas
por taxón. Se guarda la correspondencia en renombrado.csv (nombre nuevo,
original, fuente y grupo de ejemplar) para separar entrenamiento/prueba sin
partir una serie. Las extensiones .jpeg/.jfif pasan a .jpg (el contenido ya es JPEG).
Uso: python rename-dataset.py
"""
import csv
import re
import unicodedata
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[3] / "dataset" / "raw"   # ../dataset/raw respecto al repo
FUENTES = RAIZ / "_fuentes"
CLASES = ["Dionaea", "Drosera", "Sarracenia", "Nepenthes", "Darlingtonia",
          "Heliamphora", "Pinguicula", "No carnivora"]
ORDEN_FUENTE = {"propia": 0, "comunidad": 1, "subastas": 2, "inaturalist": 3}


def leer(nombre, col):
    with (FUENTES / nombre).open(encoding="utf-8-sig") as f:
        return {r["archivo"] for r in csv.DictReader(f) if r.get(col, "x")}


PROPIAS = leer("fotos_propias_clasificacion.csv", "carpeta_destino")
SUBASTAS = leer("subastas_clasificacion.csv", "carpeta_destino")
INAT = leer("no_carnivora_atribuciones.csv", "archivo")


def normalizar(s):
    s = unicodedata.normalize("NFD", s.lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).strip()


def fuente(nombre):
    if nombre in PROPIAS:
        return "propia"
    if nombre in SUBASTAS:
        return "subastas"
    if nombre in INAT:
        return "inaturalist"
    stem = Path(nombre).stem
    if re.fullmatch(r"\d+_\d+_\d+_n(?: \(\d+\))?", stem):      # descarga de Facebook
        return "comunidad"
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{23}", stem):  # WhatsApp
        return "comunidad"
    if re.fullmatch(r"[a-z0-9]{20}", stem):                   # subastas sin registrar
        return "subastas"
    return "propia"                                           # nombre descriptivo o IMG_


def grupo(nombre, fte):
    stem = Path(nombre).stem
    if fte == "inaturalist":
        return stem.rsplit("_", 1)[0]                         # taxón
    if fte == "propia":
        return normalizar(re.sub(r"\s*\d+$", "", stem))       # serie sin el número final
    return stem                                               # cada foto es su propio grupo


def main():
    filas = []
    for clase in CLASES:
        carpeta = RAIZ / clase
        archivos = [f for f in carpeta.iterdir() if f.is_file()]
        prefijo = normalizar(clase).replace(" ", "_")
        info = []
        for f in archivos:
            fte = fuente(f.name)
            info.append((ORDEN_FUENTE[fte], normalizar(f.stem), f, fte, grupo(f.name, fte)))
        info.sort(key=lambda x: (x[0], x[1]))
        temporales = []
        for i, (_, _, f, fte, grp) in enumerate(info, 1):
            with Image.open(f) as im:
                assert im.format == "JPEG", (f, im.format)
            nuevo = f"{prefijo}_{i:03d}.jpg"
            tmp = carpeta / f"__tmp_{nuevo}"
            f.rename(tmp)
            temporales.append((tmp, carpeta / nuevo))
            filas.append({"clase": clase, "nuevo": nuevo, "original": f.name,
                          "fuente": fte, "grupo": grp})
        for tmp, final in temporales:
            tmp.rename(final)
        print(f"{clase}: {len(temporales)} renombradas")

    with (FUENTES / "renombrado.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "nuevo", "original", "fuente", "grupo"])
        w.writeheader()
        w.writerows(filas)

    # añade el nombre nuevo a las atribuciones de iNaturalist
    nuevo_de = {r["original"]: r["nuevo"] for r in filas if r["clase"] == "No carnivora"}
    p = FUENTES / "no_carnivora_atribuciones.csv"
    with p.open(encoding="utf-8") as f:
        atr = list(csv.DictReader(f))
    for r in atr:
        r["archivo_dataset"] = nuevo_de.get(r["archivo"], "")
    with p.open("w", newline="", encoding="utf-8") as f:
        campos = ["archivo_dataset"] + [c for c in atr[0] if c != "archivo_dataset"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(atr)


if __name__ == "__main__":
    main()
