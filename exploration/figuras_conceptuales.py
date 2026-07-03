"""Figuras conceptuales para el capítulo (estilo Sergio): esquemas y descriptivo del
conjunto de variables elegido.

Genera: concept_matriz_confusion, concept_pipeline, concept_groupkfold,
feat_sel_separabilidad, feat_sel_correlacion. Todas con `plotting` (PNG+PDF 300 dpi).
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import seaborn as sns  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402

CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]
CB = sns.color_palette("colorblind")


def matriz_confusion():
    """Esquema conceptual VP/VN/FP/FN + errores tipo I/II."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cells = [((1, 1), "VP\n(alucina bien detectada)", CB[2]),
             ((2, 1), "FP\n(falsa alarma)\nError tipo I", CB[3]),
             ((1, 0), "FN\n(alucinación no vista)\nError tipo II", CB[3]),
             ((2, 0), "VN\n(limpia bien detectada)", CB[2])]
    for (cx, cy), t, col in cells:
        ax.add_patch(mpatches.Rectangle((cx, cy), 1, 1, fc=col, ec="black", alpha=0.35))
        ax.text(cx + 0.5, cy + 0.5, t, ha="center", va="center", fontsize=8)
    ax.text(1.5, 2.15, "Predicción", ha="center", fontsize=10, weight="bold")
    ax.text(2.0, 2.0, "alucina", ha="center", fontsize=9)
    ax.text(1.0, 2.0, "limpia", ha="center", fontsize=9)
    ax.text(0.55, 1.5, "alucina", va="center", rotation=90, fontsize=9)
    ax.text(0.55, 0.5, "limpia", va="center", rotation=90, fontsize=9)
    ax.text(0.30, 1.0, "Realidad", va="center", rotation=90, fontsize=10, weight="bold")
    ax.set(xlim=(0.2, 3.1), ylim=(-0.1, 2.3))
    ax.axis("off")
    savefig(fig, "concept_matriz_confusion")


def pipeline():
    """Esquema del flujo entrenamiento→validación→test."""
    fig, ax = plt.subplots(figsize=(9, 2.6))
    steps = ["RAGTruth\n(train/test\noficial)", "Ingeniería de\ncaracterísticas",
             "Selección\nRFE (11 vars)", "GridSearch +\nGroupKFold\n(train)",
             "Modelo\nentrenado", "Evaluación\n(test oficial)"]
    x = 0.0
    for i, s in enumerate(steps):
        col = CB[0] if i < 5 else CB[2]
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.2), 1.25, 0.9,
                     boxstyle="round,pad=0.03", fc=col, ec="black", alpha=0.35, lw=1.2))
        ax.text(x + 0.625, 0.65, s, ha="center", va="center", fontsize=7.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.42, 0.65), xytext=(x + 1.25, 0.65),
                        arrowprops=dict(arrowstyle="->", lw=1.4))
        x += 1.5
    ax.set(xlim=(-0.1, x), ylim=(0, 1.3))
    ax.axis("off")
    ax.set_title("Flujo de trabajo: del dato en crudo a la evaluación honesta", fontsize=10)
    savefig(fig, "concept_pipeline")


def groupkfold():
    """Esquema de GroupKFold por `source` (ningún documento en train y test a la vez)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    n_groups, n_folds = 10, 5
    rng = np.random.RandomState(SEED)
    group_fold = rng.randint(0, n_folds, n_groups)
    for k in range(n_folds):
        for gi in range(n_groups):
            test = group_fold[gi] == k
            col = CB[3] if test else CB[0]
            ax.add_patch(mpatches.Rectangle((gi, n_folds - 1 - k), 0.9, 0.9,
                         fc=col, ec="white", alpha=0.7))
    ax.set(xlim=(0, n_groups), ylim=(0, n_folds))
    ax.set_xticks(np.arange(n_groups) + 0.45)
    ax.set_xticklabels([f"doc {i+1}" for i in range(n_groups)], fontsize=7, rotation=45)
    ax.set_yticks(np.arange(n_folds) + 0.45)
    ax.set_yticklabels([f"fold {n_folds - i}" for i in range(n_folds)], fontsize=8)
    ax.set_title("GroupKFold por documento: cada `source` cae entero en un solo fold",
                 fontsize=10)
    handles = [mpatches.Patch(color=CB[0], alpha=0.7, label="entrenamiento"),
               mpatches.Patch(color=CB[3], alpha=0.7, label="validación")]
    ax.legend(handles=handles, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "concept_groupkfold")


def descriptivo_subconjunto():
    """Separabilidad por clase y correlación de Spearman, SOBRE las 11 elegidas."""
    df = load_ragtruth("train")
    X = extract_features(df)[CHOSEN]
    clase = df["label"].map({0: "limpio", 1: "alucina"}).values

    cont = [c for c in CHOSEN if not c.startswith("task_")]
    long = X[cont].assign(clase=clase).melt(id_vars="clase", var_name="feature",
                                             value_name="valor")
    g = sns.displot(long, x="valor", hue="clase", col="feature", col_wrap=4,
                    kind="kde", common_norm=False, fill=True, height=2.2,
                    facet_kws={"sharex": False, "sharey": False})
    g.set_titles("{col_name}")
    savefig(g.figure, "feat_sel_separabilidad")

    fig, ax = plt.subplots(figsize=(7, 6))
    corr = X.assign(label=df["label"].values).corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                annot_kws={"size": 6}, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlación de Spearman — 11 variables elegidas + objetivo")
    savefig(fig, "feat_sel_correlacion")


def main():
    set_style()
    matriz_confusion()
    pipeline()
    groupkfold()
    descriptivo_subconjunto()
    print("Figuras conceptuales y de subconjunto escritas.")


if __name__ == "__main__":
    main()
