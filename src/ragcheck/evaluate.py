"""Métricas de evaluación, organizadas en las secciones A–F de la memoria.

Todas las funciones devuelven números o arrays; el dibujo de figuras vive en
`plotting`. Positiva = alucinación.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from ragcheck.config import N_BOOTSTRAP, SEED


def discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Sección A: AUC-ROC y AUC-PR."""
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
    }


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Sección B: precision, recall, F1, accuracy y specificity al umbral dado."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def balanced_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Sección C: balanced accuracy (robusta al desbalance)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {"balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Sección D: Brier, ECE y datos del reliability diagram.

    ECE = error de calibración esperado, media de |confianza - acierto|
    ponderada por el tamaño de cada bin (Naeini et al., 2015).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    conf, acc, weight = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf.append(y_prob[mask].mean())
        acc.append(y_true[mask].mean())
        weight.append(mask.mean())
    conf, acc, weight = np.array(conf), np.array(acc), np.array(weight)
    ece = float(np.sum(weight * np.abs(conf - acc)))
    return {
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece,
        "reliability": {"confidence": conf, "accuracy": acc, "weight": weight},
    }


def confusion(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> np.ndarray:
    """Sección E: matriz de confusión 2x2 al umbral dado (filas = verdad)."""
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric=roc_auc_score,
    n: int = N_BOOTSTRAP,
) -> tuple[float, float, float]:
    """Sección F: intervalo de confianza al 95% por bootstrap de una métrica.

    Devuelve (valor sobre la muestra completa, límite inferior, superior).
    """
    point = metric(y_true, y_prob)
    rng = np.random.RandomState(SEED)
    size = len(y_true)
    vals = []
    for _ in range(n):
        s = rng.randint(0, size, size)
        if len(np.unique(y_true[s])) < 2:  # bootstrap sin ambas clases
            continue
        vals.append(metric(y_true[s], y_prob[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Umbral óptimo por el índice de Youden (maximiza sensibilidad+especificidad)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Agrega las secciones A–F en un único diccionario plano.

    El umbral se elige por Youden sobre estas mismas probabilidades.
    """
    thr = best_threshold(y_true, y_prob)
    point, lo, hi = bootstrap_ci(y_true, y_prob)
    cal = calibration(y_true, y_prob)
    return {
        **discrimination(y_true, y_prob),
        "auc_ci": (lo, hi),
        **threshold_metrics(y_true, y_prob, thr),
        **balanced_metrics(y_true, y_prob, thr),
        "brier": cal["brier"],
        "ece": cal["ece"],
        "threshold": thr,
    }
