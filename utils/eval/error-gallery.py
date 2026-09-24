# Uso: python error-gallery.py <predicciones.csv> <nombre del modelo>
# Reúne en una sola figura los errores más seguros del modelo y calcula el
# umbral que equilibra los dos tipos de error sobre la curva ROC.
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

csv_path, model_name = sys.argv[1], sys.argv[2]
ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "segments" / "test"
OUT = ROOT.parent / "computer vision-classification-model-report" / "assets" / "images" / "results"
folder = lambda label: label.replace("á", "a")

df = pd.read_csv(csv_path)
labels = list(df.columns[2:])
df["pred"] = df[labels].idxmax(axis=1)
df["conf"] = df[labels].max(axis=1)
errors = df[df.pred != df["true"]].sort_values("conf", ascending=False)

# Galería: los 12 errores con mayor confianza, seis por clase real
fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.9))
for row, label in enumerate(labels):
    subset = errors[errors["true"] == label].head(6)
    for col, (_, r) in enumerate(subset.iterrows()):
        ax = axes[row, col]
        with Image.open(TEST / folder(label) / r.file) as img:
            side = min(img.size)
            left, top = (img.width - side) // 2, (img.height - side) // 2
            ax.imshow(img.crop((left, top, left + side, top + side)).resize((224, 224)))
        ax.set_title(f"{r.pred} {r.conf * 100:.0f} %".replace(".", ","), fontsize=7, pad=2)
        ax.axis("off")
    axes[row, 0].text(-0.15, 0.5, f"Real:\n{label}", transform=axes[row, 0].transAxes,
                      ha="right", va="center", fontsize=8)
fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.02, wspace=0.08, hspace=0.35)
fig.savefig(OUT / f"{model_name}-errors.png", dpi=300)

# Umbral alternativo: el punto de la curva ROC más cercano a la esquina (0, 1)
pos = labels[0]
fpr, tpr, thr = roc_curve(df["true"] == pos, df[pos])
k = int(np.argmin(np.hypot(fpr, 1 - tpr)))
pred_alt = np.where(df[pos] >= thr[k], labels[0], labels[1])
print(f"umbral por defecto 0,5: TVP={tpr[np.argmin(np.abs(thr - 0.5))]:.3f} "
      f"TFP={fpr[np.argmin(np.abs(thr - 0.5))]:.3f}")
print(f"umbral alternativo {thr[k]:.3f}: TVP={tpr[k]:.3f} TFP={fpr[k]:.3f} "
      f"exactitud={np.mean(pred_alt == df['true']):.3f}")
print(errors[["file", "true", "pred", "conf"]].head(12).to_string(index=False))
