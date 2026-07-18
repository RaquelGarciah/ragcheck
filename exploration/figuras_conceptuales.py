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
    ax.set_title("Flujo de trabajo", fontsize=10)
    savefig(fig, "concept_pipeline")


def groupkfold():
    """Esquema de GroupKFold por `context` (ningún documento en train y test a la vez)."""
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
    ax.set_title("GroupKFold por documento: cada `context` cae entero en un solo fold",
                 fontsize=10)
    handles = [mpatches.Patch(color=CB[0], alpha=0.7, label="entrenamiento"),
               mpatches.Patch(color=CB[3], alpha=0.7, label="validación")]
    ax.legend(handles=handles, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, "concept_groupkfold")


def gridsearch():
    """Esquema de la búsqueda en rejilla: F1 de CV por combinación, se elige el máximo."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    # Superficie ilustrativa (esquema, no resultados) con un máximo claro.
    surf = np.array([[0.55, 0.61, 0.64, 0.62],
                     [0.61, 0.67, 0.70, 0.68],
                     [0.63, 0.70, 0.73, 0.71],
                     [0.62, 0.69, 0.72, 0.70]])
    im = ax.imshow(surf, cmap="Blues", aspect="auto", origin="lower")
    bi = np.unravel_index(np.argmax(surf), surf.shape)
    ax.add_patch(mpatches.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1,
                 fill=False, ec=CB[3], lw=3))
    ax.annotate("mejor combinación\n(mayor F1 medio)", xy=(bi[1], bi[0]),
                xytext=(bi[1] - 1.7, bi[0] + 0.9), fontsize=8, color=CB[3],
                arrowprops=dict(arrowstyle="->", color=CB[3], lw=1.4))
    ax.set_xticks(range(4)); ax.set_xticklabels(["3", "4", "5", "6"])
    ax.set_yticks(range(4)); ax.set_yticklabels(["0,01", "0,05", "0,1", "0,2"])
    ax.set_xlabel("hiperparámetro B (p. ej. profundidad)")
    ax.set_ylabel("hiperparámetro A (p. ej. tasa de aprendizaje)")
    ax.set_title("Grid search: cada celda es el F1 medio de 6 folds", fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("F1 de validación cruzada", fontsize=8)
    savefig(fig, "concept_gridsearch")


def umbral():
    """Esquema del punto de decisión: umbral sobre el score y un corte por tarea."""
    x = np.linspace(0, 1, 400)
    def pdf(mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    limpia, alucina = pdf(0.32, 0.14), pdf(0.68, 0.14)
    thr = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    ax.plot(x, limpia, color=CB[0], lw=2)
    ax.plot(x, alucina, color=CB[3], lw=2)
    ax.fill_between(x, alucina, where=x < thr, color=CB[3], alpha=0.30)  # FN
    ax.fill_between(x, limpia, where=x >= thr, color=CB[0], alpha=0.30)  # FP
    ax.axvline(thr, color="black", ls="--", lw=1.5)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, "umbral", fontsize=9)
    ax.text(0.16, max(limpia) * 0.5, "limpia", color=CB[0], fontsize=9, ha="center")
    ax.text(0.84, max(alucina) * 0.5, "alucina", color=CB[3], fontsize=9, ha="center")
    ax.text(0.30, max(alucina) * 0.12, "FN", color=CB[3], fontsize=8, weight="bold")
    ax.text(0.62, max(limpia) * 0.12, "FP", color=CB[0], fontsize=8, weight="bold")
    ax.annotate("", xy=(0.28, -0.6), xytext=(0.72, -0.6),
                arrowprops=dict(arrowstyle="<->", lw=1.2), annotation_clip=False)
    ax.text(0.30, -1.35, "más cobertura", fontsize=8, clip_on=False)
    ax.text(0.58, -1.35, "más precisión", fontsize=8, clip_on=False)
    ax.set(xlim=(0, 1), xlabel="score del modelo", yticks=[],
           title="(a) El umbral separa limpia de alucina")

    ax = axes[1]
    cuts = {"Data2txt": 0.43, "QA": 0.30, "Summary": 0.27}
    prev = {"Data2txt": "69%", "QA": "18%", "Summary": "23%"}
    tcol = {"Data2txt": CB[0], "QA": CB[1], "Summary": CB[2]}
    for i, (k, t) in enumerate(cuts.items()):
        y = 3 - i
        ax.hlines(y, 0, 1, color="0.8", lw=6)
        ax.plot(t, y, "v", color=tcol[k], ms=12)
        ax.text(-0.02, y, k, ha="right", va="center", fontsize=9, color=tcol[k])
        ax.text(t, y + 0.30, f"{str(t).replace('.', ',')}", ha="center", fontsize=8,
                color=tcol[k])
        ax.text(1.02, y, f"prev. {prev[k]}", ha="left", va="center", fontsize=8,
                color="0.4")
    ax.axvline(0.45, color="black", ls="--", lw=1)
    ax.text(0.45, 3.75, "umbral global", ha="center", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0.5, 4.0), yticks=[], xlabel="score",
           title="(b) Un corte por tarea")
    fig.tight_layout()
    savefig(fig, "concept_umbral")


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
    gridsearch()
    umbral()
    descriptivo_subconjunto()
    print("Figuras conceptuales y de subconjunto escritas.")


if __name__ == "__main__":
    main()
