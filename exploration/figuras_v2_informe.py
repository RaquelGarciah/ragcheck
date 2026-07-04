"""Figuras y datos que faltan para la v2 del capítulo (marcados % FIGURA/DATO CC).

Genera eda_global y pr_operacion_por_tarea, imprime la precisión a recall=0,80 por
tarea (para los % DATO CC), y saca candidatos reales de par (fuente, respuesta) de
Data2txt (label 0 y 1) para la tabla de ejemplo. Modelo ganador: xgboost 11 vars.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")
CB = sns.color_palette("colorblind")
TCOL = {"Data2txt": CB[0], "QA": CB[1], "Summary": CB[2]}
CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]
WIN = {"max_depth": 6, "learning_rate": 0.05}


def eda_global(tr, te):
    """Panel (a) longitud de respuesta global; (b) nº de ejemplos por tarea train/test."""
    length = tr["response"].str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    sns.histplot(length[length < length.quantile(0.99)], bins=40, ax=axes[0], color=CB[0])
    axes[0].set(xlabel="longitud de la respuesta (tokens)", ylabel="nº respuestas",
                title="(a) Distribución de longitudes (train)")
    rows = []
    for split, df in [("train", tr), ("test", te)]:
        for k in TASKS:
            rows.append({"split": split, "tarea": k, "n": int((df["task_type"] == k).sum())})
    sns.barplot(pd.DataFrame(rows), x="tarea", y="n", hue="split", ax=axes[1], order=list(TASKS))
    axes[1].set(xlabel="", ylabel="nº de ejemplos", title="(b) Ejemplos por tarea")
    for c in axes[1].containers:
        axes[1].bar_label(c, fontsize=7)
    fig.tight_layout()
    savefig(fig, "eda_global")


def pr_operacion(yte, task_te, pte):
    """Curvas PR por tarea con línea base y marca en recall=0,80; devuelve prec@0,80."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    prec80 = {}
    for k in TASKS:
        m = task_te == k
        pr, rc, _ = precision_recall_curve(yte[m], pte[m])
        ax.plot(rc, pr, color=TCOL[k], lw=2, label=k)
        ax.axhline(yte[m].mean(), color=TCOL[k], ls=":", lw=1)
        prec80[k] = float(pr[rc >= 0.80].max())
    ax.axvline(0.80, color="grey", ls="--", lw=1.2)
    ax.text(0.805, 0.05, "recall = 0,80", fontsize=8, color="grey", rotation=90)
    for k in TASKS:
        ax.plot(0.80, prec80[k], "o", color=TCOL[k], ms=7, mec="black")
    ax.set(xlabel="recall", ylabel="precisión", xlim=(0, 1), ylim=(0, 1),
           title="Curvas precisión-recall por tarea (test, modelo ganador)")
    ax.legend(title="tarea", loc="upper right", fontsize=9)
    savefig(fig, "pr_operacion_por_tarea")
    return prec80


def main():
    set_style()
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr = extract_features(tr)[CHOSEN]
    Xte = extract_features(te)[CHOSEN]
    ytr, yte = tr["label"].values, te["label"].values
    task_te = te["task_type"].values

    eda_global(tr, te)
    pte = build_xgboost().set_params(**WIN).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    prec80 = pr_operacion(yte, task_te, pte)

    print("=== DATO CC: precisión a recall=0,80 por tarea ===")
    for k in TASKS:
        print(f"   {k:9s} precisión@recall0.80 = {prec80[k]*100:.1f}%")

    print("\n=== Candidatos de par real Data2txt (train) ===")
    d = tr[tr["task_type"] == "Data2txt"].copy()
    d["rlen"] = d["response"].str.split().str.len()
    d["slen"] = d["source"].str.split().str.len()
    # Cortos, para que quepan en la tabla.
    for lab in (0, 1):
        cand = d[(d["label"] == lab) & (d["rlen"].between(20, 45)) & (d["slen"] < 80)].head(3)
        print(f"\n--- label={lab} ---")
        for _, r in cand.iterrows():
            print(f"[id={r['id']}] SOURCE: {r['source'][:260]}")
            print(f"          RESP:   {r['response'][:260]}")
            if lab == 1:
                print(f"          SPANS:  {r['hallucination_labels']}")


if __name__ == "__main__":
    main()
