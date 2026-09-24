# Uso: python split-construction.py
# Elimina duplicados por contenido (dHash), aparta un segmento de prueba que no
# comparte ninguna imagen con el de entrenamiento y toma una muestra balanceada
# de cada clase, guardada como JPEG RGB con el nombre de la clase en español.
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset" / "datasets" / "techsash" / "waste-classification-data" / "versions" / "1" / "DATASET"
DST = ROOT / "segments"
CLASSES = {"O": "Organico", "R": "Reciclable"}   # carpeta original -> clase
N_TRAIN, N_TEST, SEED = 500, 250, 42


def dhash(path):
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L").resize((9, 8)), dtype=int)
    return "".join("1" if bit else "0" for bit in (gray[:, 1:] > gray[:, :-1]).flatten())


random.seed(SEED)
summary = {}
for code, name in CLASSES.items():
    pools = {}
    for split in ("TRAIN", "TEST"):
        unique = {}
        files = sorted((SRC / split / code).iterdir())
        for path in files:
            unique.setdefault(dhash(path), path)
        pools[split] = {"total": len(files), "unique": unique}

    train_hashes = set(pools["TRAIN"]["unique"])
    # Una imagen de prueba solo sirve si no aparece en ninguna del entrenamiento
    test_clean = {h: p for h, p in pools["TEST"]["unique"].items() if h not in train_hashes}
    overlap = len(pools["TEST"]["unique"]) - len(test_clean)

    train_sel = random.sample(sorted(pools["TRAIN"]["unique"].values()), N_TRAIN)
    test_sel = random.sample(sorted(test_clean.values()), N_TEST)

    for segment, selected in (("train", train_sel), ("test", test_sel)):
        out = DST / segment / name
        out.mkdir(parents=True, exist_ok=True)
        for path in selected:
            with Image.open(path) as img:
                img.convert("RGB").save(out / f"{path.stem}.jpg", "JPEG", quality=95)

    summary[name] = {
        "train_total": pools["TRAIN"]["total"], "train_unique": len(train_hashes),
        "test_total": pools["TEST"]["total"], "test_unique": len(pools["TEST"]["unique"]),
        "test_overlap_with_train": overlap, "train_selected": N_TRAIN, "test_selected": N_TEST,
    }
    print(f"{name:10s} TRAIN {pools['TRAIN']['total']:5d} -> unicas {len(train_hashes):5d} | "
          f"TEST {pools['TEST']['total']:4d} -> unicas {len(pools['TEST']['unique']):4d}, "
          f"repetidas en TRAIN {overlap:3d} | seleccion train={N_TRAIN} test={N_TEST}")

(DST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
