"""SMOTE por tarea: ¿la interpolación sintética mueve la frontera? Todas las métricas.

Borrador (NO citable). Cierra la pregunta del remuestreo con SMOTE (Chawla et al.,
2002), la técnica canónica: en vez de replicar positivos, los interpola con sus
vecinos. Se aplica SOLO en train, por tarea, hasta 50/50; evaluación en el test
oficial intacto. Se reportan TODAS las métricas: frontera (AUC, AUC-PR, invariantes
al umbral) y punto de operación (precision/recall/F1/acc/bal_acc, FN/FP) a dos
umbrales comparables (0,5 y Youden global fijado en el OOF de train).

Predicción de researcher (ya visto con over/under/coste-sensible): la frontera no
se mueve —SMOTE no crea señal, solo redistribuye el punto de operación—.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")


def _smote_per_task(X: pd.DataFrame, y: np.ndarray, task: np.ndarray) -> tuple:
    """Aplica SMOTE a 50/50 dentro de cada tarea y concatena. Solo en train."""
    Xs, ys = [], []
    for k in TASKS:
        m = task == k
        Xk, yk = SMOTE(random_state=SEED).fit_resample(X[m], y[m])
        Xs.append(Xk)
        ys.append(yk)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def _metrics(y, y_pred) -> dict:
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"prec": round(prec, 3), "recall": round(rec, 3), "F1": round(f1, 3),
            "acc": round((tp + tn) / len(y), 3), "bal_acc": round((rec + spec) / 2, 3),
            "FN": fn, "FP": fp}


def _predict(pte, task_te, thr) -> np.ndarray:
    """Predicción binaria con umbral escalar o dict por tarea."""
    if np.isscalar(thr):
        return (pte >= thr).astype(int)
    pred = np.zeros(len(pte), int)
    for k in TASKS:
        m = task_te == k
        pred[m] = (pte[m] >= thr[k]).astype(int)
    return pred


def _full(name, yte, task_te, pte, thr) -> pd.DataFrame:
    """AUC, AUC-PR y métricas a un umbral, por tarea y global."""
    pred = _predict(pte, task_te, thr)
    rows = []
    for k in ("GLOBAL", *TASKS):
        m = np.ones(len(yte), bool) if k == "GLOBAL" else (task_te == k)
        t = thr if np.isscalar(thr) else (thr[k] if k != "GLOBAL" else float("nan"))
        rows.append({"modelo": name, "ámbito": k, "prev": round(yte[m].mean(), 3),
                     "AUC": round(roc_auc_score(yte[m], pte[m]), 3),
                     "AUC-PR": round(average_precision_score(yte[m], pte[m]), 3),
                     "thr": round(float(t), 3),
                     **_metrics(yte[m], pred[m])})
    return pd.DataFrame(rows).set_index(["modelo", "ámbito"])


def main() -> None:
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr), extract_features(te)
    ytr, yte = tr["label"].values, te["label"].values
    gtr, task_tr, task_te = tr["context"].values, tr["task_type"].values, te["task_type"].values

    # Umbrales honestos: Youden global y max-F1 por tarea desde el OOF de train.
    ptr = cross_validate(build_xgboost(), Xtr, ytr, gtr)["y_prob"]
    t_glob = ev.best_threshold(ytr, ptr)
    from sklearn.metrics import precision_recall_curve
    def _maxf1(y, p):
        pr, rc, th = precision_recall_curve(y, p)
        f1 = 2 * pr * rc / (pr + rc + 1e-12)
        return float(th[int(np.argmax(f1[:-1]))])
    t_task = {k: _maxf1(ytr[task_tr == k], ptr[task_tr == k]) for k in TASKS}

    # Modelos: base y SMOTE por tarea.
    pte_base = build_xgboost().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    Xsm, ysm = _smote_per_task(Xtr, ytr, task_tr)
    pte_sm = build_xgboost().fit(Xsm, ysm).predict_proba(Xte)[:, 1]
    print(f"train base n={len(ytr)} (prev {ytr.mean():.3f}) | train SMOTE n={len(ysm)} (prev {ysm.mean():.3f})")
    print(f"umbral Youden global={t_glob:.3f} | max-F1 por tarea={ {k: round(v,3) for k,v in t_task.items()} }\n")

    for thr, tag in [(t_glob, "umbral Youden global"), (t_task, "umbral max-F1 por tarea")]:
        print(f"{'='*78}\n### {tag}\n{'='*78}")
        out = pd.concat([_full("base", yte, task_te, pte_base, thr),
                        _full("SMOTE", yte, task_te, pte_sm, thr)])
        print(out.to_string(), "\n")


if __name__ == "__main__":
    main()
