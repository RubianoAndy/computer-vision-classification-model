# Uso: python model-evaluation.py <predicciones.csv> <nombre del modelo>
# Evaluación multiclase (8 géneros) a partir del CSV de probabilidades:
# exactitud, precisión, exhaustividad y F1 por clase y macro, curvas ROC
# uno-contra-el-resto con AUC, pérdida logarítmica, intervalos bootstrap
# y el efecto de un umbral de confianza ("no estoy seguro"). Dibuja las
# figuras del informe en assets/images/results del repositorio del informe.
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_curve, top_k_accuracy_score)

csv_path, model_name = sys.argv[1], sys.argv[2]
OUT = Path(__file__).resolve().parents[3] / "computer vision-classification-model-report" / "assets" / "images" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED, BOOTSTRAP = 42, 2000
UMBRALES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]

df = pd.read_csv(csv_path)
labels = list(df.columns[2:])
K = len(labels)
probs = df[labels].to_numpy()
probs = probs / probs.sum(axis=1, keepdims=True)  # corrige el redondeo del CSV
y_true = df["true"].map(labels.index).to_numpy()
y_pred = probs.argmax(axis=1)
conf = probs.max(axis=1)

prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred)

# Curva ROC de cada clase contra el resto
roc = {}
for i, label in enumerate(labels):
    fpr, tpr, thr = roc_curve(y_true == i, probs[:, i])
    roc[label] = (fpr, tpr, auc(fpr, tpr))
auc_macro = float(np.mean([roc[l][2] for l in labels]))

# Intervalos de confianza del 95 % por bootstrap sobre las mismas imágenes
rng = np.random.default_rng(SEED)
boot = {"accuracy": [], "f1_macro": [], "auc_macro": []}
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(y_true), len(y_true))
    yt, yp, pp = y_true[idx], y_pred[idx], probs[idx]
    boot["accuracy"].append(accuracy_score(yt, yp))
    boot["f1_macro"].append(precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)[2])
    aucs = []
    for i in range(K):
        if 0 < (yt == i).sum() < len(yt):
            f, t, _ = roc_curve(yt == i, pp[:, i])
            aucs.append(auc(f, t))
    boot["auc_macro"].append(np.mean(aucs))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

# Umbral de confianza: si la probabilidad máxima no lo alcanza, la app responde
# "no estoy seguro". Se mide cuántas imágenes responde y qué exactitud tiene en ellas.
umbral = []
for t in UMBRALES:
    m = conf >= t
    umbral.append({
        "umbral": t,
        "cobertura": float(m.mean()),
        "exactitud_respondidas": float(accuracy_score(y_true[m], y_pred[m])) if m.any() else None,
        "errores_respondidos": int((y_pred[m] != y_true[m]).sum()),
        "errores_evitados": int((y_pred[~m] != y_true[~m]).sum()),
        "aciertos_perdidos": int((y_pred[~m] == y_true[~m]).sum()),
    })

errors_mask = y_pred != y_true
metrics = {
    "model": model_name,
    "n": int(len(df)),
    "accuracy": accuracy_score(y_true, y_pred),
    "top2_accuracy": float(top_k_accuracy_score(y_true, probs, k=2, labels=range(K))),
    "log_loss": log_loss(y_true, probs, labels=range(K)),
    "mean_confidence": float(conf.mean()),
    "mean_confidence_errors": float(conf[errors_mask].mean()) if errors_mask.any() else None,
    "mean_confidence_hits": float(conf[~errors_mask].mean()),
    "macro": dict(zip(["precision", "recall", "f1"], macro[:3])),
    "auc_macro": auc_macro,
    "ci95": ci,
    "per_class": {
        label: {"precision": prec[i], "recall": rec[i], "f1": f1[i],
                "support": int(support[i]), "auc": roc[label][2]}
        for i, label in enumerate(labels)
    },
    "confusion_matrix": cm.tolist(),
    "confidence_threshold": umbral,
    "errors": [
        {"file": r.file, "true": r.true, "pred": labels[p], "prob": float(probs[k, p]),
         "second": labels[int(np.argsort(probs[k])[-2])]}
        for k, (r, p) in enumerate(zip(df.itertuples(), y_pred)) if labels[p] != r.true
    ],
}
Path(csv_path).with_name(f"{model_name}-metrics.json").write_text(
    json.dumps(metrics, indent=2, default=float, ensure_ascii=False), encoding="utf-8")

# ---------- Gráficos ----------
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SERIES = ["#2e8b57", "#2a78d6", "#eb6834", "#8e44ad", "#c0392b", "#16a085", "#d4a017", "#5d6d7e"]
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
    "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
})
comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
short = [l.replace("No carnivora", "No carn.") for l in labels]

# Matriz de confusión
fig, ax = plt.subplots(figsize=(5.2, 4.8))
greens = LinearSegmentedColormap.from_list("greens", ["#ffffff", "#8fcaa6", "#2e8b57", "#12442a"])
ax.imshow(cm, cmap=greens, vmin=0, vmax=cm.max())
for i in range(K):
    for j in range(K):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > cm.max() * 0.5 else INK)
ax.set_xticks(range(K), short, rotation=45, ha="right")
ax.set_yticks(range(K), short)
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(OUT / f"{model_name}-confusion-matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Curvas ROC uno contra el resto
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (2, 3)), label="Azar (AUC = 0,500)")
for i, label in enumerate(labels):
    fpr, tpr, area = roc[label]
    ax.plot(fpr, tpr, color=SERIES[i], lw=1.6, label=f"{short[i]} (AUC = {area:.3f})".replace(".", ","))
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_xlim(-0.02, 1.0)
ax.set_ylim(0.0, 1.02)
ax.grid(color=GRID, lw=0.6)
ax.xaxis.set_major_formatter(comma)
ax.yaxis.set_major_formatter(comma)
ax.legend(loc="lower right", frameon=False, fontsize=7)
fig.savefig(OUT / f"{model_name}-roc-curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Precisión, exhaustividad y F1 por clase
fig, ax = plt.subplots(figsize=(5.4, 3.0))
x = np.arange(K)
for k, (name, vals) in enumerate([("Precisión", prec), ("Exhaustividad", rec), ("F1", f1)]):
    ax.bar(x + (k - 1) * 0.27, vals, width=0.27, color=SERIES[k], label=name)
ax.set_xticks(x, short, rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(comma)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT / f"{model_name}-per-class.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Confianza de la predicción: aciertos frente a errores
fig, ax = plt.subplots(figsize=(4.6, 3.0))
bins = np.linspace(0, 1, 21)
ax.hist(conf[~errors_mask], bins=bins, alpha=0.75, color=SERIES[0], label="Aciertos")
ax.hist(conf[errors_mask], bins=bins, alpha=0.75, color=SERIES[2], label="Errores")
ax.set_xlabel("Probabilidad de la clase predicha")
ax.set_ylabel("Imágenes")
ax.xaxis.set_major_formatter(comma)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.savefig(OUT / f"{model_name}-confidence-histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Cobertura y exactitud según el umbral de confianza
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ts = np.linspace(0, 0.99, 100)
cov = [(conf >= t).mean() for t in ts]
acc = [accuracy_score(y_true[conf >= t], y_pred[conf >= t]) if (conf >= t).any() else np.nan for t in ts]
ax.plot(ts, acc, color=SERIES[0], lw=2, label="Exactitud en las respondidas")
ax.plot(ts, cov, color=SERIES[1], lw=2, label="Cobertura (imágenes respondidas)")
ax.set_xlabel("Umbral de confianza")
ax.xaxis.set_major_formatter(comma)
ax.yaxis.set_major_formatter(comma)
ax.set_ylim(0, 1.02)
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(OUT / f"{model_name}-threshold-curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "errors", "confidence_threshold")},
                 indent=2, default=float, ensure_ascii=False))
print(pd.DataFrame(cm, index=short, columns=short))
print(pd.DataFrame(umbral).to_string(index=False))
print(f"errores: {int(errors_mask.sum())}")
