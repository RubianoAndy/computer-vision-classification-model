# Uso: python error-gallery.py <predicciones.csv> <nombre del modelo>
# Reúne en una figura los errores con mayor confianza del modelo (hasta 16)
# y lista los pares de clases que más se confunden.
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

csv_path, model_name = sys.argv[1], sys.argv[2]
ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT.parent / "dataset" / "prepared" / "test"
OUT = ROOT.parent / "computer vision-classification-model-report" / "assets" / "images" / "results"

df = pd.read_csv(csv_path)
labels = list(df.columns[2:])
df["pred"] = df[labels].idxmax(axis=1)
df["conf"] = df[labels].max(axis=1)
errors = df[df.pred != df["true"]].sort_values("conf", ascending=False)

# Galería: los 16 errores con mayor confianza, en 2 filas de 8
top = errors.head(16)
fig, axes = plt.subplots(2, 8, figsize=(7.4, 2.6))
for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
    with Image.open(TEST / r["true"] / r.file) as img:
        ax.imshow(img.resize((224, 224)))
    ax.set_title(f"{r['true']}\n→ {r.pred} {r.conf * 100:.0f} %".replace("No carnivora", "No carn."),
                 fontsize=5.5, pad=2)
    ax.axis("off")
for ax in axes.ravel()[len(top):]:
    ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.08, hspace=0.5)
fig.savefig(OUT / f"{model_name}-errors.png", dpi=300)

pares = Counter(zip(errors["true"], errors["pred"]))
print("pares real → predicha más frecuentes:")
for (t, p), n in pares.most_common(10):
    print(f"  {t} → {p}: {n}")
print(errors[["file", "true", "pred", "conf"]].head(16).to_string(index=False))
