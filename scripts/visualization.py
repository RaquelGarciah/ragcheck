"""Análisis descriptivo del dataset RAGTruth (Nivel 1).

Genera las figuras del capítulo de datos: balance de clases, tasa de
alucinación por tarea, longitud de la respuesta por clase, reuso de documentos,
matriz de correlación de features y separabilidad por clase. Todas las figuras
van a outputs/figures/.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import FEATURES, extract_features  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402

LABELS = {0: "limpio", 1: "alucina"}


def main() -> None:
    set_style()
    df = load_ragtruth("train")
    df["clase"] = df["label"].map(LABELS)
    df["len_resp"] = df["response"].str.split().str.len()
    X = extract_features(df)

    # 1. Balance global de clases.
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.countplot(df, x="clase", order=["limpio", "alucina"], ax=ax)
    ax.set(title="Balance de clases (train)", xlabel="", ylabel="nº respuestas")
    savefig(fig, "eda_balance_clases")

    # 2. Tasa de alucinación por tipo de tarea.
    rate = df.groupby("task_type")["label"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=rate.values, y=rate.index, ax=ax)
    ax.set(title="Tasa de alucinación por tarea", xlabel="fracción que alucina", ylabel="")
    for i, v in enumerate(rate.values):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    savefig(fig, "eda_tasa_por_tarea")

    # 3. Longitud de la respuesta por clase.
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(df, x="clase", y="len_resp", order=["limpio", "alucina"], ax=ax)
    ax.set(title="Longitud de la respuesta por clase", xlabel="", ylabel="tokens")
    ax.set_ylim(0, df["len_resp"].quantile(0.99))
    savefig(fig, "eda_longitud_por_clase")

    # 4. Reuso de documentos (justifica GroupKFold).
    reuse = df["source"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(reuse.values, bins=30, ax=ax)
    ax.set(title="Respuestas por documento", xlabel="nº respuestas por `source`", ylabel="documentos")
    savefig(fig, "eda_reuso_documentos")

    # 5. Correlación entre features y con la etiqueta.
    corr = X.assign(label=df["label"].values).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlación de features y etiqueta")
    savefig(fig, "eda_correlacion")

    # 6. Separabilidad por clase de cada feature (solo las continuas; las
    # one-hot de tarea son binarias y su KDE no informa).
    cont = [c for c in X.columns if not c.startswith("task_")]
    long = X[cont].assign(clase=df["clase"].values).melt(
        id_vars="clase", var_name="feature", value_name="valor"
    )
    g = sns.displot(
        long, x="valor", hue="clase", col="feature", col_wrap=3,
        kind="kde", common_norm=False, fill=True, height=2.4,
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set_titles("{col_name}")
    savefig(g.figure, "eda_separabilidad_features")

    print("Figuras EDA (Nivel 1) escritas en outputs/figures/eda_*.")
    print("balance global:", round(df["label"].mean(), 3))
    print(rate.round(3).to_dict())


if __name__ == "__main__":
    main()
