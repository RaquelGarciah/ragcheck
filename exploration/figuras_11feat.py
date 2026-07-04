"""Recomputa las figuras dependientes del modelo sobre el conjunto RFE de 11 variables
(coherencia con el resto del capítulo) y vuelca las cifras a citar.

Regenera: desb_barrido_umbral, desb_estrategias_umbral, desb_fn_recuperados,
eval_confusion_por_tarea, eval_roc_pr_por_tarea, eval_calibracion, lit_comparacion_f1.
Modelo: xgboost (ganador) sobre las 11 variables.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")
CB = sns.color_palette("colorblind")
TASK_COLOR = {"Data2txt": CB[0], "QA": CB[1], "Summary": CB[2]}
CHOSEN = ["containment", "task_QA", "task_Summary", "task_Data2txt", "num_context",
          "answer_len", "jaccard", "sent_cont_min", "novel_bigram", "num_overlap",
          "sent_sim_min"]
# Hiperparámetros del ganador (grid search ampliado sobre las 11 vars, grid_tables.py).
WIN = {"max_depth": 6, "learning_rate": 0.05}


def win_xgb():
    return build_xgboost().set_params(**WIN)


def best_fbeta(y, p, beta):
    pr, rc, thr = precision_recall_curve(y, p)
    b2 = beta * beta
    fb = (1 + b2) * pr * rc / (b2 * pr + rc + 1e-12)
    return float(thr[int(np.argmax(fb[:-1]))])


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"prec": prec, "recall": rec, "f1": f1, "acc": (tp + tn) / len(y),
            "bal_acc": (rec + spec) / 2, "FN": fn, "FP": fp, "TN": tn, "TP": tp}


def main():
    set_style()
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr)[CHOSEN], extract_features(te)[CHOSEN]
    ytr, yte = tr["label"].values, te["label"].values
    g, task_tr, task_te = tr["source"].values, tr["task_type"].values, te["task_type"].values

    ptr = cross_validate(win_xgb(), Xtr, ytr, g)["y_prob"]
    pte = win_xgb().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    t_glob = ev.best_threshold(ytr, ptr)
    t_task = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}
    t_f2 = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 2) for k in TASKS}

    # --- barrido de umbral (OOF) ---
    grid = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ("precision", "recall", "F1", "bal_acc"):
        ys = []
        for t in grid:
            m = ev.threshold_metrics(ytr, ptr, t)
            ys.append(m[name.lower()] if name != "bal_acc"
                      else ev.balanced_metrics(ytr, ptr, t)["balanced_accuracy"])
        ax.plot(grid, ys, lw=2, label=name)
    ax.axvline(t_glob, color="grey", ls="--", lw=1)
    ax.set(xlabel="umbral de decisión", ylabel="valor", ylim=(0, 1),
           title="Barrido de umbral (XGBoost, 11 vars, OOF)")
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    savefig(fig, "desb_barrido_umbral")

    # --- estrategias por tarea (test) ---
    def predict(thr):
        pred = np.zeros(len(yte), int)
        for k in TASKS:
            m = task_te == k
            pred[m] = (pte[m] >= (thr if np.isscalar(thr) else thr[k])).astype(int)
        return pred
    strat = {"Youden global": t_glob, "por tarea (F1)": t_task, "por tarea (F2)": t_f2}
    rows = []
    for sname, thr in strat.items():
        pred = predict(thr)
        for k in TASKS:
            m = task_te == k; mm = metrics(yte[m], pred[m])
            for mk, ml in [("recall", "recall"), ("prec", "precisión"), ("f1", "F1")]:
                rows.append({"estrategia": sname, "tarea": k, "métrica": ml, "valor": mm[mk]})
    d = pd.DataFrame(rows)
    gg = sns.catplot(d, x="tarea", y="valor", hue="estrategia", col="métrica",
                     kind="bar", order=list(TASKS), height=3.4, aspect=0.9)
    gg.set_axis_labels("", "valor"); gg.set_titles("{col_name}")
    gg.fig.suptitle("Estrategias de umbral por tarea (test, 11 vars)", y=1.03)
    savefig(gg.fig, "desb_estrategias_umbral")

    # --- FN por estrategia ---
    fn_g = metrics(yte, predict(t_glob))["FN"]
    fn_t = metrics(yte, predict(t_task))["FN"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Youden global", "por tarea (F1)"], [fn_g, fn_t],
                  color=[CB[3], CB[2]])
    ax.bar_label(bars, fmt="%d", fontsize=11)
    ax.set(ylabel="falsos negativos (test)", title="Falsos negativos por estrategia")
    savefig(fig, "desb_fn_recuperados")

    # --- confusión por tarea ---
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for j, k in enumerate(TASKS):
        m = task_te == k
        for i, (lab, thr) in enumerate([("umbral global", t_glob), ("umbral por tarea", t_task[k])]):
            mm = metrics(yte[m], (pte[m] >= thr).astype(int))
            cm = np.array([[mm["TN"], mm["FP"]], [mm["FN"], mm["TP"]]])
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=axes[i, j],
                        xticklabels=["limpio", "alucina"], yticklabels=["limpio", "alucina"])
            axes[i, j].set_title(f"{k} — {lab}", fontsize=9)
            if j == 0: axes[i, j].set_ylabel("verdad")
            if i == 1: axes[i, j].set_xlabel("predicho")
    fig.suptitle("Matrices de confusión por tarea (XGBoost, 11 vars, test)", y=1.01)
    fig.tight_layout(); savefig(fig, "eval_confusion_por_tarea")

    # --- ROC/PR por tarea ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for k in TASKS:
        m = task_te == k
        fpr, tpr, _ = roc_curve(yte[m], pte[m])
        axes[0].plot(fpr, tpr, color=TASK_COLOR[k], lw=2,
                     label=f"{k} ({roc_auc_score(yte[m], pte[m]):.2f})")
        pr, rc, _ = precision_recall_curve(yte[m], pte[m])
        axes[1].plot(rc, pr, color=TASK_COLOR[k], lw=2,
                     label=f"{k} ({average_precision_score(yte[m], pte[m]):.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="RFP", ylabel="RVP", title="ROC por tarea"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="recall", ylabel="precisión", title="PR por tarea"); axes[1].legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "eval_roc_pr_por_tarea")

    # --- calibración ---
    iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfecta")
    for p, nm in [(pte, "crudo"), (iso.predict(pte), "isotónica")]:
        cal = ev.calibration(yte, p); r = cal["reliability"]
        ax.plot(r["confidence"], r["accuracy"], "o-",
                label=f"{nm} (ECE={cal['ece']:.3f}, Brier={cal['brier']:.3f})")
    ax.set(xlabel="probabilidad predicha", ylabel="frecuencia observada",
           title="Diagrama de fiabilidad (test, 11 vars)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8); savefig(fig, "eval_calibracion")

    # --- literatura F1 por tarea ---
    ours = {k: metrics(yte[task_te == k], (pte[task_te == k] >= t_task[k]).astype(int))["f1"] * 100
            for k in TASKS}
    lit = {"GPT-4-turbo (juez)": {"QA": 45.6, "Data2txt": 78.3, "Summary": 47.6},
           "Luna": {"QA": 51.3, "Data2txt": 75.9, "Summary": 52.5},
           "LettuceDetect-large": {"QA": 70.2, "Data2txt": 88.5, "Summary": 59.7},
           "Llama-2-13B (fine-tuned)": {"QA": 68.2, "Data2txt": 88.1, "Summary": 59.1},
           "RAG-HAT (SOTA)": {"QA": 74.8, "Data2txt": 91.6, "Summary": 67.6},
           "Este TFG (clásico)": ours}
    rows = [{"método": m, "tarea": k, "F1": v[k]} for m, v in lit.items() for k in TASKS]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(pd.DataFrame(rows), x="tarea", y="F1", hue="método", ax=ax, order=list(TASKS))
    ax.set(xlabel="", ylabel="F1 nivel respuesta (%)",
           title="F1 por tarea: detector clásico (11 vars) frente a la literatura")
    ax.legend(title="", fontsize=7, ncol=2)
    savefig(fig, "lit_comparacion_f1")

    # --- volcado de cifras ---
    print("umbral Youden global =", round(t_glob, 3))
    print("umbral por tarea F1 =", {k: round(v, 3) for k, v in t_task.items()})
    for sname, thr in strat.items():
        pred = predict(thr); gm = metrics(yte, pred)
        macro = np.mean([metrics(yte[task_te == k], pred[task_te == k])["f1"] for k in TASKS])
        print(f"\n[{sname}] F1pooled={gm['f1']:.3f} F1macro={macro:.3f} prec={gm['prec']:.3f} "
              f"rec={gm['recall']:.3f} acc={gm['acc']:.3f} FN={gm['FN']} FP={gm['FP']}")
        for k in TASKS:
            m = task_te == k; km = metrics(yte[m], pred[m])
            print(f"   {k:9s} F1={km['f1']:.3f} prec={km['prec']:.3f} rec={km['recall']:.3f} "
                  f"acc={km['acc']:.3f} FN={km['FN']}")
    print("\nAUC/AUC-PR por tarea (test):")
    for k in TASKS:
        m = task_te == k
        print(f"   {k:9s} AUC={roc_auc_score(yte[m], pte[m]):.3f} "
              f"AUC-PR={average_precision_score(yte[m], pte[m]):.3f} prev={yte[m].mean():.3f}")
    print("\nNuestro F1 por tarea (umbral por tarea):", {k: round(v, 1) for k, v in ours.items()})
    print("Accuracy global (Youden) =", round(metrics(yte, predict(t_glob))["acc"], 3))


if __name__ == "__main__":
    main()
