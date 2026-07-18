# -*- coding: utf-8 -*-
"""Boxplot de las características continuas para visualizar los valores atípicos.

Las variables se estandarizan (z-score) para poder compararlas en un mismo eje
pese a sus escalas distintas. Junto a cada nombre se anota el porcentaje de
atípicos según la regla del rango intercuartílico de Tukey. Sirve de apoyo a la
decisión de conservarlos (fase de modificación del capítulo de resultados).
"""
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "/Users/Raquel/Desktop/ALUCINACIONES_RAG/src")
from ragcheck.config import SEED
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features

random.seed(SEED)
np.random.seed(SEED)

FIG_DIR = "/Users/Raquel/Desktop/ALUCINACIONES_RAG-informe/outputs/figures"


def iqr_outlier_frac(v: np.ndarray) -> float:
    """Fracción de valores fuera de [Q1-1,5·RIC, Q3+1,5·RIC] (Tukey, 1977)."""
    q1, q3 = np.percentile(v, [25, 75])
    ric = q3 - q1
    return float(np.mean((v < q1 - 1.5 * ric) | (v > q3 + 1.5 * ric)))


def main() -> None:
    df = load_ragtruth("train")
    X = extract_features(df)
    drop = [c for c in X.columns if c.startswith("task_")] + [
        "label", "id", "source", "response",
    ]
    cont = [c for c in X.columns if c not in drop and pd.api.types.is_numeric_dtype(X[c])]

    # % de atípicos por variable y orden descendente
    frac = {c: iqr_outlier_frac(X[c].astype(float).values) for c in cont}
    cont = sorted(cont, key=lambda c: frac[c], reverse=True)

    # estandarización z para comparar en un eje común
    Z = (X[cont] - X[cont].mean()) / X[cont].std(ddof=0)
    long = Z.melt(var_name="feature", value_name="z")

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.boxplot(
        data=long, y="feature", x="z", order=cont, orient="h",
        color="#4C72B0", fliersize=1.6, linewidth=0.9,
        flierprops=dict(marker="o", markerfacecolor="#C44E52",
                        markeredgecolor="none", alpha=0.35), ax=ax,
    )
    ax.set_xlabel("valor estandarizado (z)")
    ax.set_ylabel("")
    ax.axvline(0, color="0.6", lw=0.8, zorder=0)
    # etiqueta con el % de atípicos a la derecha del nombre
    ax.set_yticks(range(len(cont)))
    ax.set_yticklabels([f"{c}  ({frac[c]*100:.1f}%)" for c in cont])
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/eda_outliers_boxplot.png")
    fig.savefig(f"{FIG_DIR}/eda_outliers_boxplot.pdf")
    plt.close(fig)

    print("media %atípicos:", round(np.mean(list(frac.values())) * 100, 1))
    print("guardado en", FIG_DIR)


if __name__ == "__main__":
    main()
