# Uso: python dataset-inspection.py
# Recorre el dataset descargado con kagglehub y describe su estructura,
# formatos, modos de color y tamaños de imagen, sobre una muestra por clase.
import random
from collections import Counter
from pathlib import Path

from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "dataset" / "datasets" / "techsash" / "waste-classification-data" / "versions" / "1" / "DATASET"
SAMPLE, SEED = 300, 42
random.seed(SEED)

for split in ("TRAIN", "TEST"):
    for class_dir in sorted((SRC / split).iterdir()):
        files = sorted(class_dir.iterdir())
        sample = random.sample(files, min(SAMPLE, len(files)))
        formats, modes, widths, heights = Counter(), Counter(), [], []
        for path in sample:
            with Image.open(path) as img:
                formats[img.format] += 1
                modes[img.mode] += 1
                widths.append(img.width)
                heights.append(img.height)
        print(f"{split}/{class_dir.name}: {len(files):5d} archivos | "
              f"formatos {dict(formats)} | modos {dict(modes)} | "
              f"ancho {min(widths)}-{max(widths)} | alto {min(heights)}-{max(heights)}")
