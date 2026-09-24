# Uso: python model-evaluation.py <predicciones.csv> <nombre del modelo>
# Calcula exactitud, precisión, exhaustividad, F1, curva ROC, AUC y pérdida
# logarítmica a partir del CSV de probabilidades y dibuja las figuras del informe.
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_curve)

csv_path, model_name = sys.argv[1], sys.argv[2]
OUT = Path(__file__).resolve().parents[2] / "computer vision-classification-model-report" / "assets" / "images" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED, BOOTSTRAP = 42, 2000

df = pd.read_csv(csv_path)
labels = list(df.columns[2:])
probs = df[labels].to_numpy()
probs = probs / probs.sum(axis=1, keepdims=True)  # corrige el redondeo del CSV
y_true = df["true"].map(labels.index).to_numpy()
y_pred = probs.argmax(axis=1)

prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred)

# Curva ROC de cada clase contra el resto; con dos clases ambas son simétricas
roc = {}
for i, label in enumerate(labels):
    fpr, tpr, thr = roc_curve(y_true == i, probs[:, i])
    roc[label] = (fpr, tpr, auc(fpr, tpr), thr)

# Intervalos de confianza del 95 % por bootstrap sobre las mismas imágenes
rng = np.random.default_rng(SEED)
boot = {"accuracy": [], "f1_macro": [], "auc": []}
pos = 0
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(y_true), len(y_true))
    yt, yp, pp = y_true[idx], y_pred[idx], probs[idx, pos]
    boot["accuracy"].append(accuracy_score(yt, yp))
    boot["f1_macro"].append(precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2])
    if len(np.unique(yt)) == 2:
        f, t, _ = roc_curve(yt == pos, pp)
        boot["auc"].append(auc(f, t))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

errors_mask = y_pred != y_true
metrics = {
    "model": model_name,
    "n": int(len(df)),
    "accuracy": accuracy_score(y_true, y_pred),
    "log_loss": log_loss(y_true, probs, labels=range(len(labels))),
    "mean_confidence": float(probs.max(axis=1).mean()),
    "mean_confidence_errors": float(probs.max(axis=1)[errors_mask].mean()) if errors_mask.any() else None,
    "macro": dict(zip(["precision", "recall", "f1"], macro[:3])),
    "auc": roc[labels[pos]][2],
    "ci95": ci,
    "per_class": {
        label: {"precision": prec[i], "recall": rec[i], "f1": f1[i],
                "support": int(support[i]), "auc": roc[label][2]}
        for i, label in enumerate(labels)
    },
    "confusion_matrix": cm.tolist(),
    "errors": [
        {"file": r.file, "true": r.true, "pred": labels[p], "prob": float(probs[k, p])}
        for k, (r, p) in enumerate(zip(df.itertuples(), y_pred)) if labels[p] != r.true
    ],
}
Path(csv_path).with_name(f"{model_name}-metrics.json").write_text(
    json.dumps(metrics, indent=2, default=float, ensure_ascii=False), encoding="utf-8")

# ---------- Gráficos ----------
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2e8b57", "#2a78d6", "#eb6834"]
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})
comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

# Matriz de confusión
fig, ax = plt.subplots(figsize=(3.6, 3.3))
greens = LinearSegmentedColormap.from_list("greens", ["#f1f8f3", "#8fcaa6", "#2e8b57", "#12442a"])
ax.imshow(cm, cmap=greens, vmin=0, vmax=cm.max())
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() * 0.5 else INK)
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(OUT / f"{model_name}-confusion-matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Curva ROC de la clase positiva, con el umbral de 0,5 marcado
fig, ax = plt.subplots(figsize=(4.2, 3.9))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (2, 3)), label="Azar (AUC = 0,500)")
fpr, tpr, area, thr = roc[labels[pos]]
ax.plot(fpr, tpr, color=SERIES[0], lw=2, label=f"{labels[pos]} vs. resto (AUC = {area:.3f})".replace(".", ","))
k = int(np.argmin(np.abs(thr - 0.5)))
ax.plot(fpr[k], tpr[k], "o", color=SERIES[2], ms=7, markeredgecolor="white", label="Umbral 0,5")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0.0, 1.02)
ax.grid(color=GRID, lw=0.6)
ax.xaxis.set_major_formatter(comma)
ax.yaxis.set_major_formatter(comma)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.savefig(OUT / f"{model_name}-roc-curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Distribución de la probabilidad asignada a la clase positiva, por clase real
fig, ax = plt.subplots(figsize=(4.6, 3.0))
bins = np.linspace(0, 1, 21)
for i, label in enumerate(labels):
    ax.hist(probs[y_true == i, pos], bins=bins, alpha=0.75, color=SERIES[i], label=f"Real: {label}")
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.set_xlabel(f"Probabilidad asignada a {labels[pos]}")
ax.set_ylabel("Imágenes")
ax.xaxis.set_major_formatter(comma)
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.savefig(OUT / f"{model_name}-probability-histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "errors")},
                 indent=2, default=float, ensure_ascii=False))
print(cm)
print(f"errores: {int(errors_mask.sum())}")
