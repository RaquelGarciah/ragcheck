# -*- coding: utf-8 -*-
"""Regenera las curvas ROC y precisión-cobertura conjuntas sobre el test oficial
(11 variables RFE) para los cuatro modelos, cada uno con su mejor configuración
de las tablas de rejilla. La logística lleva estandarizado (corrección). Las
leyendas muestran el AUC-ROC / AUC-PR, que deben coincidir con la tabla comparativa.
"""
import sys, warnings, random
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/Raquel/Desktop/ALUCINACIONES_RAG/src")
from ragcheck.config import SEED
from ragcheck.data import load_ragtruth
from ragcheck.features import extract_features
from ragcheck.models import (build_logreg, build_knn, build_random_forest,
                             build_xgboost)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score)
random.seed(SEED); np.random.seed(SEED)

FIG_DIR = "/Users/Raquel/Desktop/ALUCINACIONES_RAG-informe/outputs/figures"
CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]

# Mejor configuración de cada modelo (según las tablas de rejilla del informe).
MODELS = [
    ("Reg.\\ logística", build_logreg().set_params(
        logisticregression__C=0.01, logisticregression__penalty="l1",
        logisticregression__class_weight="balanced")),
    ("KNN", build_knn().set_params(
        kneighborsclassifier__n_neighbors=25, kneighborsclassifier__p=1,
        kneighborsclassifier__weights="distance")),
    ("Bosque aleatorio", build_random_forest().set_params(
        max_depth=20, max_features="sqrt", min_samples_leaf=4)),
    ("XGBoost", build_xgboost().set_params(
        max_depth=6, learning_rate=0.05, subsample=0.8)),
]


def main():
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr = extract_features(tr)[CHOSEN]; Xte = extract_features(te)[CHOSEN]
    ytr, yte = tr["label"].values, te["label"].values

    probs = []
    for name, m in MODELS:
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        probs.append((name, p, roc_auc_score(yte, p), average_precision_score(yte, p)))
        print(f"{name:20s} AUC-ROC={roc_auc_score(yte,p):.3f}  AUC-PR={average_precision_score(yte,p):.3f}")

    sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
    plt.rcParams["figure.dpi"] = 300; plt.rcParams["savefig.bbox"] = "tight"

    # ROC
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for name, p, auc, _ in probs:
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name.replace(chr(92)+' ',' ')} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos")
    ax.legend(loc="lower right", title="AUC-ROC")
    fig.savefig(f"{FIG_DIR}/roc_test_rfe.png"); fig.savefig(f"{FIG_DIR}/roc_test_rfe.pdf")
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for name, p, _, ap in probs:
        prec, rec, _ = precision_recall_curve(yte, p)
        ax.plot(rec, prec, lw=2, label=f"{name.replace(chr(92)+' ',' ')} ({ap:.3f})")
    base = yte.mean()
    ax.axhline(base, color="0.6", ls="--", lw=1)
    ax.set(xlabel="cobertura", ylabel="precisión")
    ax.legend(loc="upper right", title="AUC-PR")
    fig.savefig(f"{FIG_DIR}/pr_test_rfe.png"); fig.savefig(f"{FIG_DIR}/pr_test_rfe.pdf")
    plt.close(fig)
    print("figuras guardadas en", FIG_DIR)


if __name__ == "__main__":
    main()
