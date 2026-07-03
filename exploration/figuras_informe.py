"""Figuras nuevas para el informe de resultados (caso de estudio).

Borrador (NO citable como código, pero produce figuras citables). Genera las
figuras que faltaban para el capítulo de resultados: descriptivo del desbalance,
estrategias de decisión, remuestreo/SMOTE, calibración, desagregado por tarea y
comparación con la literatura. Reusa `ragcheck.plotting` (tema colorblind, PNG+PDF
300 dpi) y el protocolo honesto (OOF GroupKFold en train, evaluación en test oficial).

Al final imprime un volcado de cifras consolidado: la fuente única para las tablas
del report y del .tex.
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.plotting import savefig, set_style  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")
# Colores por tarea, coherentes en todas las figuras (paleta colorblind).
_CB = sns.color_palette("colorblind")
TASK_COLOR = {"Data2txt": _CB[0], "QA": _CB[1], "Summary": _CB[2]}


# --------------------------------------------------------------------------- #
# Utilidades numéricas
# --------------------------------------------------------------------------- #
def saerens(p, pi_ref, iters=500):
    """Corrige el posterior por prior-shift; el prior test se estima por EM."""
    pi = pi_ref
    for _ in range(iters):
        num = (pi / pi_ref) * p
        post = num / (num + ((1 - pi) / (1 - pi_ref)) * (1 - p))
        new = float(post.mean())
        if abs(new - pi) < 1e-10:
            break
        pi = new
    return pi


def best_fbeta(y, p, beta):
    """Umbral que maximiza F-beta sobre la curva precision-recall."""
    prec, rec, thr = precision_recall_curve(y, p)
    b2 = beta * beta
    fb = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
    return float(thr[int(np.argmax(fb[:-1]))])


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"prec": prec, "recall": rec, "f1": f1, "acc": (tp + tn) / len(y),
            "bal_acc": (rec + spec) / 2, "FN": fn, "FP": fp, "TP": tp, "TN": tn}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def fig_prevalencia(tr, te):
    """Prevalencia de alucinación por tarea en train vs test (prior-shift)."""
    rows = []
    for split, df in [("train", tr), ("test", te)]:
        for k in TASKS:
            m = df["task_type"] == k
            rows.append({"split": split, "tarea": k, "prev": df.loc[m, "label"].mean()})
        rows.append({"split": split, "tarea": "GLOBAL", "prev": df["label"].mean()})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(d, x="tarea", y="prev", hue="split", ax=ax,
                order=["GLOBAL", *TASKS])
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set(xlabel="", ylabel="prevalencia de alucinación", ylim=(0, 1),
           title="Prevalencia por tarea: train frente a test")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    ax.legend(title="")
    savefig(fig, "desb_prevalencia_tarea")


def fig_barrido_umbral(y, p):
    """Precision/recall/F1/bal_acc al mover el umbral (xgboost, OOF de train)."""
    grid = np.linspace(0.05, 0.95, 91)
    curves = {"precision": [], "recall": [], "F1": [], "bal_acc": []}
    for t in grid:
        m = ev.threshold_metrics(y, p, t)
        curves["precision"].append(m["precision"])
        curves["recall"].append(m["recall"])
        curves["F1"].append(m["f1"])
        curves["bal_acc"].append(ev.balanced_metrics(y, p, t)["balanced_accuracy"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, v in curves.items():
        ax.plot(grid, v, label=name, lw=2)
    t_y = ev.best_threshold(y, p)
    ax.axvline(t_y, color="grey", ls="--", lw=1)
    ax.text(t_y + 0.01, 0.05, f"Youden = {t_y:.2f}", fontsize=8, color="grey")
    ax.set(xlabel="umbral de decisión", ylabel="valor de la métrica",
           title="Barrido de umbral (XGBoost, OOF de train)", ylim=(0, 1))
    ax.legend(loc="lower center", ncol=4, fontsize=8)
    savefig(fig, "desb_barrido_umbral")


def fig_estrategias(yte, task_te, pte, ptr, ytr, task_tr):
    """recall/precision/F1 por tarea: Youden global vs por-tarea F1 vs F2."""
    t_glob = ev.best_threshold(ytr, ptr)
    strategies = {"Youden global": {k: t_glob for k in TASKS}}
    strategies["por tarea (F1)"] = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}
    strategies["por tarea (F2)"] = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 2) for k in TASKS}

    rows = []
    for sname, thr in strategies.items():
        for k in TASKS:
            m = task_te == k
            mm = metrics(yte[m], (pte[m] >= thr[k]).astype(int))
            for metric in ("recall", "prec", "f1"):
                rows.append({"estrategia": sname, "tarea": k,
                             "métrica": {"recall": "recall", "prec": "precisión", "f1": "F1"}[metric],
                             "valor": mm[metric]})
    d = pd.DataFrame(rows)
    g = sns.catplot(d, x="tarea", y="valor", hue="estrategia", col="métrica",
                    kind="bar", order=TASKS, height=3.5, aspect=0.9)
    g.set_axis_labels("", "valor")
    g.set_titles("{col_name}")
    g.fig.suptitle("Estrategias de umbral por tarea (test oficial)", y=1.03)
    g.savefig  # noqa
    savefig(g.fig, "desb_estrategias_umbral")


def fig_frontera(Xtr, ytr, task_tr, Xte, yte, task_te):
    """AUC-PR por tarea: base vs over/under/SMOTE/coste-sensible (frontera)."""
    from imblearn.over_sampling import SMOTE

    def train_probs(X, y):
        return build_xgboost().fit(X, y).predict_proba(Xte)[:, 1]

    def resample(mode, rng):
        keep = []
        for k in TASKS:
            pos = np.where((task_tr == k) & (ytr == 1))[0]
            neg = np.where((task_tr == k) & (ytr == 0))[0]
            if mode == "over":
                n = max(len(pos), len(neg))
                keep += [rng.choice(pos, n, True), rng.choice(neg, n, True)]
            else:
                n = min(len(pos), len(neg))
                keep += [rng.choice(pos, n, False), rng.choice(neg, n, False)]
        idx = np.concatenate(keep)
        return Xtr.iloc[idx], ytr[idx]

    rng = np.random.RandomState(SEED)
    probs = {"base": train_probs(Xtr, ytr)}
    Xo, yo = resample("over", rng)
    probs["oversample"] = train_probs(Xo, yo)
    Xu, yu = resample("under", rng)
    probs["undersample"] = train_probs(Xu, yu)
    Xs, ys = [], []
    for k in TASKS:
        m = task_tr == k
        Xk, yk = SMOTE(random_state=SEED).fit_resample(Xtr[m], ytr[m])
        Xs.append(Xk); ys.append(yk)
    probs["SMOTE"] = train_probs(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
    npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
    probs["coste-sensible"] = build_xgboost().set_params(
        scale_pos_weight=nneg / npos).fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    rows = []
    for sname, p in probs.items():
        for k in TASKS:
            m = task_te == k
            rows.append({"estrategia": sname, "tarea": k,
                         "AUC-PR": average_precision_score(yte[m], p[m])})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.barplot(d, x="tarea", y="AUC-PR", hue="estrategia", ax=ax, order=TASKS)
    ax.set(xlabel="", title="La frontera (AUC-PR) no se mueve con el remuestreo")
    ax.legend(title="", fontsize=8, ncol=2)
    savefig(fig, "desb_frontera_remuestreo")
    return d


def fig_saerens(pte, task_te, yte, pi_ref):
    """Prior estimado por EM (Saerens) frente al real, por tarea: el colapso."""
    rows = []
    for k in TASKS:
        m = task_te == k
        rows.append({"tarea": k, "tipo": "EM (Saerens)", "prior": saerens(pte[m], pi_ref)})
        rows.append({"tarea": k, "tipo": "real", "prior": yte[m].mean()})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(d, x="tarea", y="prior", hue="tipo", ax=ax, order=TASKS)
    ax.set(xlabel="", ylabel="prevalencia (prior)", ylim=(0, 1.05),
           title="Saerens colapsa: el EM estima priors 0/1")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    ax.legend(title="")
    savefig(fig, "desb_saerens_colapso")


def fig_fn(yte, task_te, pte, ptr, ytr, task_tr, smote_pte):
    """Falsos negativos totales por estrategia de decisión (la gráfica clave)."""
    t_glob = ev.best_threshold(ytr, ptr)
    t_task = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}

    def fn_total(p, thr):
        pred = np.zeros(len(yte), int)
        for k in TASKS:
            m = task_te == k
            t = thr if np.isscalar(thr) else thr[k]
            pred[m] = (p[m] >= t).astype(int)
        return metrics(yte, pred)["FN"]

    data = {
        "Youden global": fn_total(pte, t_glob),
        "por tarea (F1)": fn_total(pte, t_task),
        "SMOTE + Youden": fn_total(smote_pte, t_glob),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [_CB[3], _CB[2], _CB[7]]
    bars = ax.bar(list(data), list(data.values()), color=colors)
    ax.bar_label(bars, fmt="%d", fontsize=10)
    ax.set(ylabel="falsos negativos (test oficial)",
           title="Falsos negativos recuperados por estrategia")
    savefig(fig, "desb_fn_recuperados")


def fig_confusion(yte, task_te, pte, ptr, ytr, task_tr):
    """Matrices de confusión por tarea: umbral global vs por tarea."""
    t_glob = ev.best_threshold(ytr, ptr)
    t_task = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for j, k in enumerate(TASKS):
        m = task_te == k
        for i, (label, thr) in enumerate([("umbral global", t_glob), ("umbral por tarea", t_task[k])]):
            pred = (pte[m] >= thr).astype(int)
            cm = np.array([[metrics(yte[m], pred)[x] for x in ("TN", "FP")],
                           [metrics(yte[m], pred)[x] for x in ("FN", "TP")]])
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues",
                        ax=axes[i, j], xticklabels=["limpio", "alucina"],
                        yticklabels=["limpio", "alucina"])
            axes[i, j].set(title=f"{k} — {label}")
            if j == 0:
                axes[i, j].set_ylabel("verdad")
            if i == 1:
                axes[i, j].set_xlabel("predicho")
    fig.suptitle("Matrices de confusión por tarea (XGBoost, test oficial)", y=1.01)
    fig.tight_layout()
    savefig(fig, "eval_confusion_por_tarea")


def fig_roc_pr_tarea(yte, task_te, pte):
    """ROC y PR desagregadas por tarea."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for k in TASKS:
        m = task_te == k
        fpr, tpr, _ = roc_curve(yte[m], pte[m])
        auc = roc_auc_score(yte[m], pte[m])
        axes[0].plot(fpr, tpr, label=f"{k} ({auc:.2f})", color=TASK_COLOR[k], lw=2)
        prec, rec, _ = precision_recall_curve(yte[m], pte[m])
        ap = average_precision_score(yte[m], pte[m])
        axes[1].plot(rec, prec, label=f"{k} ({ap:.2f})", color=TASK_COLOR[k], lw=2)
        axes[1].axhline(yte[m].mean(), color=TASK_COLOR[k], ls=":", lw=1)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos",
                title="Curvas ROC por tarea")
    axes[0].legend(title="AUC", fontsize=8)
    axes[1].set(xlabel="recall", ylabel="precisión", title="Curvas precisión-recall por tarea")
    axes[1].legend(title="AUC-PR", fontsize=8)
    fig.tight_layout()
    savefig(fig, "eval_roc_pr_por_tarea")


def fig_calibracion(yte, pte, ptr, ytr):
    """Reliability diagram: xgboost crudo vs calibrado por isotónica."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
    pte_c = iso.predict(pte)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfecta")
    for p, name in [(pte, "crudo"), (pte_c, "isotónica")]:
        cal = ev.calibration(yte, p)
        r = cal["reliability"]
        ax.plot(r["confidence"], r["accuracy"], "o-",
                label=f"{name} (ECE={cal['ece']:.3f}, Brier={cal['brier']:.3f})")
    ax.set(xlabel="probabilidad predicha", ylabel="frecuencia observada",
           title="Diagrama de fiabilidad (test oficial)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    savefig(fig, "eval_calibracion")


def fig_literatura(yte, task_te, pte, ptr, ytr, task_tr):
    """F1 por tarea: nuestro detector frente a la literatura sobre RAGTruth."""
    t_task = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}
    ours = {}
    for k in TASKS:
        m = task_te == k
        ours[k] = metrics(yte[m], (pte[m] >= t_task[k]).astype(int))["f1"] * 100
    # F1 nivel respuesta verificado (EXPLORACION §2 / literatura_RAGTRUTH §11).
    lit = {
        "GPT-4-turbo (juez)": {"QA": 45.6, "Data2txt": 78.3, "Summary": 47.6},
        "Luna": {"QA": 51.3, "Data2txt": 75.9, "Summary": 52.5},
        "LettuceDetect-large": {"QA": 70.2, "Data2txt": 88.5, "Summary": 59.7},
        "Llama-2-13B (fine-tuned)": {"QA": 68.2, "Data2txt": 88.1, "Summary": 59.1},
        "RAG-HAT (SOTA)": {"QA": 74.8, "Data2txt": 91.6, "Summary": 67.6},
        "Este TFG (clásico)": {k: ours[k] for k in TASKS},
    }
    rows = [{"método": mth, "tarea": k, "F1": v[k]} for mth, v in lit.items() for k in TASKS]
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(d, x="tarea", y="F1", hue="método", ax=ax, order=TASKS)
    ax.set(xlabel="", ylabel="F1 nivel respuesta (%)",
           title="F1 por tarea: nuestro detector clásico frente a la literatura")
    ax.legend(title="", fontsize=7, ncol=2)
    savefig(fig, "lit_comparacion_f1")
    return ours


def fig_importancia_acumulada():
    """Importancia por permutación acumulada (valores de EXPLORACION §7.1)."""
    perm = {
        "containment": 0.161, "answer_len": 0.020, "jaccard": 0.019, "num_context": 0.012,
        "task_QA": 0.010, "num_overlap": 0.010, "sent_sim_min": 0.008, "sent_cont_min": 0.006,
        "sent_sim_mean": 0.005, "neg_diff": 0.004, "tfidf_cos": 0.004, "resto (7)": 0.021,
    }
    names = list(perm)
    vals = np.array(list(perm.values()))
    cum = np.cumsum(vals) / vals.sum()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(names, vals, color=_CB[0], label="caída de AUC")
    ax2 = ax.twinx()
    ax2.plot(names, cum, "o-", color=_CB[3], label="acumulado")
    ax2.axhline(0.8, color="grey", ls="--", lw=1)
    ax.set(ylabel="caída de AUC por permutación", title="Importancia de features acumulada")
    ax2.set(ylabel="fracción acumulada", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=60)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    savefig(fig, "feat_importancia_acumulada")


# --------------------------------------------------------------------------- #
def main():
    set_style()
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr), extract_features(te)
    ytr, yte = tr["label"].values, te["label"].values
    gtr = tr["source"].values
    task_tr, task_te = tr["task_type"].values, te["task_type"].values
    pi_ref = float(ytr.mean())

    ptr = cross_validate(build_xgboost(), Xtr, ytr, gtr)["y_prob"]
    pte = build_xgboost().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    # SMOTE para la figura de FN (misma lógica que smote_check).
    from imblearn.over_sampling import SMOTE
    Xs, ys = [], []
    for k in TASKS:
        m = task_tr == k
        Xk, yk = SMOTE(random_state=SEED).fit_resample(Xtr[m], ytr[m])
        Xs.append(Xk); ys.append(yk)
    smote_pte = build_xgboost().fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys)).predict_proba(Xte)[:, 1]

    print(">> generando figuras...")
    fig_prevalencia(tr, te)
    fig_barrido_umbral(ytr, ptr)
    fig_estrategias(yte, task_te, pte, ptr, ytr, task_tr)
    frontera = fig_frontera(Xtr, ytr, task_tr, Xte, yte, task_te)
    fig_saerens(pte, task_te, yte, pi_ref)
    fig_fn(yte, task_te, pte, ptr, ytr, task_tr, smote_pte)
    fig_confusion(yte, task_te, pte, ptr, ytr, task_tr)
    fig_roc_pr_tarea(yte, task_te, pte)
    fig_calibracion(yte, pte, ptr, ytr)
    ours = fig_literatura(yte, task_te, pte, ptr, ytr, task_tr)
    fig_importancia_acumulada()

    # ------- Volcado de cifras (fuente única para report y .tex) ------- #
    print("\n" + "=" * 70 + "\nVOLCADO DE CIFRAS CONSOLIDADO\n" + "=" * 70)
    t_glob = ev.best_threshold(ytr, ptr)
    t_task = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 1) for k in TASKS}
    t_f2 = {k: best_fbeta(ytr[task_tr == k], ptr[task_tr == k], 2) for k in TASKS}
    print(f"prior ref train={pi_ref:.3f} | prev test global={yte.mean():.3f}")
    print(f"umbral Youden global={t_glob:.3f}")
    print(f"umbral por tarea F1={ {k: round(v,3) for k,v in t_task.items()} }")
    print(f"umbral por tarea F2={ {k: round(v,3) for k,v in t_f2.items()} }")

    for sname, thr in [("GLOBAL YOUDEN", t_glob), ("POR TAREA F1", t_task), ("POR TAREA F2", t_f2)]:
        pred = np.zeros(len(yte), int)
        for k in TASKS:
            m = task_te == k
            t = thr if np.isscalar(thr) else thr[k]
            pred[m] = (pte[m] >= t).astype(int)
        gm = metrics(yte, pred)
        macro = np.mean([metrics(yte[task_te == k], pred[task_te == k])["f1"] for k in TASKS])
        print(f"\n[{sname}] GLOBAL: F1(pooled)={gm['f1']:.3f} F1(macro)={macro:.3f} "
              f"prec={gm['prec']:.3f} rec={gm['recall']:.3f} acc={gm['acc']:.3f} FN={gm['FN']} FP={gm['FP']}")
        for k in TASKS:
            m = task_te == k
            km = metrics(yte[m], pred[m])
            print(f"    {k:9s} F1={km['f1']:.3f} prec={km['prec']:.3f} rec={km['recall']:.3f} "
                  f"acc={km['acc']:.3f} FN={km['FN']} FP={km['FP']}")

    print("\nAUC/AUC-PR por tarea (xgboost base, test):")
    for k in TASKS:
        m = task_te == k
        print(f"    {k:9s} AUC={roc_auc_score(yte[m], pte[m]):.3f} AUC-PR={average_precision_score(yte[m], pte[m]):.3f}")
    print("\nFrontera (AUC-PR) por estrategia de remuestreo:")
    print(frontera.pivot(index="estrategia", columns="tarea", values="AUC-PR").round(3).to_string())
    print("\nNuestro F1 por tarea (umbral por tarea F1):", {k: round(v, 1) for k, v in ours.items()})


if __name__ == "__main__":
    main()
