"""Descarga imágenes para la clase "No carnivora" desde iNaturalist.

Solo toma fotos con licencia Creative Commons que permite obras derivadas
(se excluyen las ND) y deja la atribución de cada una en un CSV.
Uso: python download-non-carnivorous.py
"""
import csv
import time
from pathlib import Path

import requests

API = "https://api.inaturalist.org/v1"
LICENCIAS = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
RAIZ = Path(__file__).resolve().parents[3] / "dataset" / "raw"   # ../dataset/raw respecto al repo
DESTINO = RAIZ / "No carnivora"
CSV_ATRIBUCIONES = RAIZ / "_fuentes" / "no_carnivora_atribuciones.csv"

# (taxón, cantidad). Se priorizan plantas que se confunden con carnívoras
# (rosetas, jarras, capuchas) y las que suelen compartir colección con ellas.
TAXONES = [
    ("Echeveria", 28), ("Sempervivum", 12), ("Haworthiopsis", 10), ("Aloe", 10),
    ("Crassula ovata", 8), ("Kalanchoe", 8), ("Sedum", 8), ("Lithops", 6),
    ("Phalaenopsis", 15), ("Orchidaceae", 10),
    ("Tillandsia", 14), ("Guzmania", 10), ("Neoregelia", 8),
    ("Zantedeschia", 14), ("Anthurium", 10), ("Spathiphyllum", 8),
    ("Arisaema", 12), ("Aristolochia", 10),
    ("Monstera deliciosa", 8), ("Epipremnum aureum", 8),
    ("Chlorophytum comosum", 6), ("Dracaena trifasciata", 8),
    ("Nephrolepis", 10), ("Sphagnum", 14), ("Bryophyta", 8),
    ("Mammillaria", 10), ("Opuntia", 8),
    ("Poaceae", 10), ("Rosa", 8), ("Hibiscus", 8), ("Taraxacum", 6),
    ("Viola", 8), ("Streptocarpus ionanthus", 8),
]

# Descartadas en la revisión visual: duplicadas, sin planta reconocible o con
# una carnívora en el encuadre (el Sphagnum cultivado suele ser su sustrato).
EXCLUIDAS = {
    "echeveria_404893653.jpg", "phalaenopsis_404800491.jpg",
    "crassula-ovata_405003437.jpg", "crassula-ovata_404648332.jpg",
    "zantedeschia_404726465.jpg", "sphagnum_374012314.jpg",
    "sphagnum_370112804.jpg",
}

sesion = requests.Session()
sesion.headers["User-Agent"] = "proyecto-academico-vision-por-computador/1.0"


def buscar_taxon(nombre):
    r = sesion.get(f"{API}/taxa", params={"q": nombre, "per_page": 10}, timeout=30)
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"].lower() == nombre.lower() and t.get("is_active", True):
            return t["id"]
    return None


def observaciones(taxon_id, cultivada):
    params = {
        "taxon_id": taxon_id, "photos": "true", "photo_license": LICENCIAS,
        "captive": str(cultivada).lower(), "per_page": 80,
        "order_by": "created_at", "order": "desc",
    }
    if not cultivada:
        params["quality_grade"] = "research"
    r = sesion.get(f"{API}/observations", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


def elegir(taxon_id, cantidad):
    """Mitad cultivadas (en maceta) y mitad silvestres, un usuario por foto."""
    usuarios, elegidas = set(), []
    for cultivada, cupo in ((True, cantidad // 2), (False, cantidad)):
        for obs in observaciones(taxon_id, cultivada):
            if len(elegidas) >= cupo:
                break
            usuario = obs["user"]["login"]
            foto = next((f for f in obs["photos"] if f.get("license_code")
                         and "nd" not in f["license_code"]), None)
            if foto is None or usuario in usuarios:
                continue
            usuarios.add(usuario)
            elegidas.append((obs, foto, cultivada))
        time.sleep(1)
    return elegidas


def descargar(url, intentos=4):
    for intento in range(intentos):
        try:
            r = sesion.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 * (intento + 1))
    return None


def main():
    DESTINO.mkdir(exist_ok=True)
    filas = []
    for nombre, cantidad in TAXONES:
        taxon_id = buscar_taxon(nombre)
        time.sleep(1)
        if taxon_id is None:
            print(f"[sin taxón] {nombre}")
            continue
        elegidas = elegir(taxon_id, cantidad)
        prefijo = nombre.lower().replace(" ", "-")
        for obs, foto, cultivada in elegidas:
            url = foto["url"].replace("/square.", "/medium.")
            archivo = f"{prefijo}_{obs['id']}.jpg"
            if archivo in EXCLUIDAS:
                continue
            if not (DESTINO / archivo).exists():
                contenido = descargar(url)
                if contenido is None:
                    continue
                (DESTINO / archivo).write_bytes(contenido)
                time.sleep(0.2)
            filas.append({
                "archivo": archivo, "taxon": nombre,
                "cultivada": "si" if cultivada else "no",
                "observacion": f"https://www.inaturalist.org/observations/{obs['id']}",
                "autor": obs["user"]["login"], "licencia": foto["license_code"],
                "atribucion": foto.get("attribution", ""), "url_foto": url,
            })
        print(f"{nombre}: {len(elegidas)}/{cantidad}")

    with CSV_ATRIBUCIONES.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"Total: {len(filas)} imágenes")


if __name__ == "__main__":
    main()
