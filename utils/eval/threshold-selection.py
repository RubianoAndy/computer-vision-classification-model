# Uso: python threshold-selection.py <validacion.csv> <prueba.csv> <nombre del modelo>
# Elige el umbral de decisión sobre el segmento de validación (nunca sobre el
# de prueba) y evalúa el modelo en prueba con el umbral por defecto y con el
# ajustado, con intervalos por bootstrap y las figuras del informe.
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)

val_path, test_path, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path(__file__).resolve().parents[3] / "computer vision-classification-model-report" / "assets" / "images" / "results"
SEED, BOOTSTRAP = 42, 2000


def load(path):
    df = pd.read_csv(path)
    labels = list(df.columns[2:])
    probs = df[labels].to_numpy()
    probs = probs / probs.sum(axis=1, keepdims=True)
    return df, labels, probs


def predict_with(probs, labels, thr, pos=0):
    # Clase positiva solo si supera el umbral; si no, la otra clase
    return np.where(probs[:, pos] >= thr, pos, 1 - pos)


def summarize(y_true, y_pred, labels):
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro": dict(zip(["precision", "recall", "f1"], macro[:3])),
        "per_class": {l: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "support": int(support[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "errors": int((y_true != y_pred).sum()),
    }


# ---------- Elección del umbral sobre validación ----------
val, labels, pv = load(val_path)
yv = val["true"].map(labels.index).to_numpy()
fpr, tpr, thr = roc_curve(yv == 0, pv[:, 0])
k = int(np.argmin(np.hypot(fpr, 1 - tpr)))          # punto más cercano a la esquina (0, 1)
t_adj = float(thr[k])
grid = np.linspace(0.5, 0.999, 500)
acc_val = [accuracy_score(yv, predict_with(pv, labels, t)) for t in grid]

# ---------- Evaluación en prueba con ambos umbrales ----------
test, _, pt = load(test_path)
yt = test["true"].map(labels.index).to_numpy()
acc_test = [accuracy_score(yt, predict_with(pt, labels, t)) for t in grid]
results = {"default": summarize(yt, predict_with(pt, labels, 0.5), labels),
           "adjusted": summarize(yt, predict_with(pt, labels, t_adj), labels)}

# Intervalos de confianza pareados: la misma remuestra para los dos umbrales
rng = np.random.default_rng(SEED)
diffs, acc_d, acc_a = [], [], []
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(yt), len(yt))
    a_d = accuracy_score(yt[idx], predict_with(pt[idx], labels, 0.5))
    a_a = accuracy_score(yt[idx], predict_with(pt[idx], labels, t_adj))
    acc_d.append(a_d), acc_a.append(a_a), diffs.append(a_a - a_d)
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
results["threshold"] = t_adj
results["validation"] = {"n": int(len(yv)), "tpr": float(tpr[k]), "fpr": float(fpr[k]),
                         "accuracy_default": accuracy_score(yv, predict_with(pv, labels, 0.5)),
                         "accuracy_adjusted": accuracy_score(yv, predict_with(pv, labels, t_adj))}
results["ci95"] = {"accuracy_default": ci(acc_d), "accuracy_adjusted": ci(acc_a), "difference": ci(diffs)}
Path(test_path).with_name(f"{model_name}-threshold-metrics.json").write_text(
    json.dumps(results, indent=2, default=float, ensure_ascii=False), encoding="utf-8")

# ---------- Figuras ----------
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})
comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

# Exactitud según el umbral, en validación y en prueba
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(grid, acc_val, color="#2a78d6", lw=2, label=f"Validación (n = {len(yv)})")
ax.plot(grid, acc_test, color="#2e8b57", lw=2, ls="--", label=f"Prueba (n = {len(yt)})")
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.axvline(t_adj, color="#eb6834", lw=1.2)
ax.text(t_adj - 0.01, min(acc_val) + 0.005, f"umbral ajustado {t_adj:.3f}".replace(".", ","),
        ha="right", va="bottom", fontsize=8, color="#eb6834")
ax.set_xlabel(f"Umbral sobre la probabilidad de {labels[0]}")
ax.set_ylabel("Exactitud")
ax.xaxis.set_major_formatter(comma)
ax.yaxis.set_major_formatter(comma)
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(OUT / f"{model_name}-threshold-curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Matriz de confusión en prueba con el umbral ajustado
cm = np.array(results["adjusted"]["confusion_matrix"])
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
fig.savefig(OUT / f"{model_name}-threshold-confusion-matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps(results, indent=2, default=float, ensure_ascii=False))
