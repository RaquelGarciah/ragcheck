"""Desbalance por tarea: decisión bajo prevalencia heterogénea y prior-shift.

Borrador (NO citable). El diagnóstico (EXPLORACION_resultados §5, §8) mostró que
los falsos negativos de QA no son falta de señal (AUC 0,82) sino un umbral global
demasiado alto para su baja prevalencia (0,18). Aquí se ataca la DECISIÓN y el
APRENDIZAJE bajo desbalance, todo con protocolo honesto:

  Fase 1 — decisión (mismo modelo, sin reentrenar):
    · global_youden : umbral único por Youden en el OOF de train (baseline).
    · pertask_F1    : un umbral por tarea que maximiza F1 en el OOF de train.
    · pertask_F2    : idem con F2 (pesa 4x el recall: penaliza el falso negativo).
    · saerens       : corrección de prior (Saerens et al., 2002) sobre posteriores
                      CALIBRADOS; el prior del test se estima por EM sin etiquetas.

  Fase 2 — aprendizaje coste-sensible:
    · xgboost con scale_pos_weight = neg/pos. Se mide si MUEVE la frontera
      (AUC-PR por tarea) o solo desliza el punto de operación.

Calibración: isotónica ajustada sobre las OOF de train (no toca el test).
Umbrales y priors de referencia salen SOLO de train; evaluación en el test oficial.
"""

import random

import numpy as np
import pandas as pd

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)

from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from ragcheck import evaluate as ev  # noqa: E402
from ragcheck.data import load_ragtruth  # noqa: E402
from ragcheck.features import extract_features  # noqa: E402
from ragcheck.models import build_xgboost  # noqa: E402
from ragcheck.training import cross_validate  # noqa: E402

TASKS = ("Data2txt", "QA", "Summary")


def saerens(p: np.ndarray, pi_ref: float, iters: int = 500) -> tuple[np.ndarray, float]:
    """Corrige el posterior por desplazamiento de prior (Saerens et al., 2002).

    `p` son posteriores calibrados bajo el prior de referencia `pi_ref`. El prior
    del test se estima por EM sobre los scores, sin etiquetas. Devuelve
    (posteriores corregidos, prior estimado).
    """
    pi = pi_ref
    for _ in range(iters):
        num = (pi / pi_ref) * p
        post = num / (num + ((1 - pi) / (1 - pi_ref)) * (1 - p))
        new = float(post.mean())
        if abs(new - pi) < 1e-10:
            pi = new
            break
        pi = new
    num = (pi / pi_ref) * p
    return num / (num + ((1 - pi) / (1 - pi_ref)) * (1 - p)), pi


def best_fbeta(y: np.ndarray, p: np.ndarray, beta: float) -> float:
    """Umbral que maximiza F-beta sobre la curva precision-recall (train OOF)."""
    prec, rec, thr = precision_recall_curve(y, p)
    b2 = beta * beta
    fb = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
    return float(thr[int(np.argmax(fb[:-1]))])


def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y == 1) & (y_pred == 1)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    tn = int(((y == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"prec": prec, "recall": rec, "F1": f1, "acc": (tp + tn) / len(y),
            "bal_acc": (rec + spec) / 2, "FN": fn, "FP": fp}


def _table(y, task, pred) -> pd.DataFrame:
    rows = []
    for scope in ("GLOBAL", *TASKS):
        m = np.ones(len(y), bool) if scope == "GLOBAL" else (task == scope)
        rows.append({"ámbito": scope, "prev": round(y[m].mean(), 3), **_metrics(y[m], pred[m])})
    return pd.DataFrame(rows).set_index("ámbito")


def main() -> None:
    tr, te = load_ragtruth("train"), load_ragtruth("test")
    Xtr, Xte = extract_features(tr), extract_features(te)
    ytr, yte = tr["label"].values, te["label"].values
    gtr = tr["context"].values
    task_tr, task_te = tr["task_type"].values, te["task_type"].values
    pi_ref = float(ytr.mean())

    # OOF honesto en train para fijar umbrales y calibrar; ajuste final para test.
    ptr = cross_validate(build_xgboost(), Xtr, ytr, gtr)["y_prob"]
    pte = build_xgboost().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    # Calibración isotónica aprendida SOLO de las OOF de train.
    iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
    ptr_c, pte_c = iso.predict(ptr), iso.predict(pte)

    print(f"prior ref (train)={pi_ref:.3f} | prev test global={yte.mean():.3f}\n")

    # --- Fase 1: estrategias de decisión ---
    strat = {}
    t = ev.best_threshold(ytr, ptr)
    strat["1) GLOBAL YOUDEN (baseline)"] = (pte >= t).astype(int)

    for beta, tag in [(1.0, "2) PERTASK max-F1"), (2.0, "3) PERTASK max-F2 (anti-FN)")]:
        pred = np.zeros(len(yte), int)
        thrs = {}
        for k in TASKS:
            tk = best_fbeta(ytr[task_tr == k], ptr[task_tr == k], beta)
            thrs[k] = round(tk, 3)
            pred[task_te == k] = (pte[task_te == k] >= tk).astype(int)
        strat[tag] = pred
        print(f"{tag} umbrales/tarea: {thrs}")

    pred_s = np.zeros(len(yte), int)
    pis = {}
    for k in TASKS:
        m = task_te == k
        post, pi = saerens(pte_c[m], pi_ref)
        pis[k] = round(pi, 3)
        pred_s[m] = (post >= 0.5).astype(int)
    strat["4) SAERENS calibrado (prior EM + 0.5)"] = pred_s
    print(f"Saerens prior EM/tarea: {pis} | real: "
          f"{ {k: round(yte[task_te==k].mean(),3) for k in TASKS} }\n")

    for name, pred in strat.items():
        print(f"{'='*66}\n{name}\n{'='*66}")
        print(_table(yte, task_te, pred).round(3).to_string(), "\n")

    # --- Fase 2: coste-sensible, ¿mueve la frontera (AUC-PR) o solo desliza? ---
    npos, nneg = int(ytr.sum()), int((1 - ytr).sum())
    wpte = build_xgboost().set_params(scale_pos_weight=nneg / npos).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print(f"{'='*66}\nFASE 2 — coste-sensible: AUC-PR por tarea (frontera)\n{'='*66}")
    rows = []
    for k in ("GLOBAL", *TASKS):
        m = np.ones(len(yte), bool) if k == "GLOBAL" else (task_te == k)
        rows.append({"ámbito": k,
                     "AUC-PR base": round(average_precision_score(yte[m], pte[m]), 3),
                     "AUC-PR coste-sens": round(average_precision_score(yte[m], wpte[m]), 3)})
    print(pd.DataFrame(rows).set_index("ámbito").to_string())


if __name__ == "__main__":
    main()
