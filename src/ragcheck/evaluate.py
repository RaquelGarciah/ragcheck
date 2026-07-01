"""Métricas de evaluación, organizadas en las secciones A–F de la memoria.

Todas las funciones devuelven números o arrays; el dibujo de figuras vive en
`plotting`. Positiva = alucinación.
"""

import numpy as np


def discrimination(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Sección A: AUC-ROC y AUC-PR."""
    raise NotImplementedError  # Fase 0


def threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Sección B: precision, recall, F1, accuracy y specificity al umbral dado."""
    raise NotImplementedError  # Fase 0


def balanced_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Sección C: balanced accuracy (robusta al desbalance)."""
    raise NotImplementedError  # Fase 0


def calibration(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Sección D: Brier, ECE y datos del reliability diagram."""
    raise NotImplementedError  # Fase 0


def confusion(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> np.ndarray:
    """Sección E: matriz de confusión 2x2 al umbral dado."""
    raise NotImplementedError  # Fase 0


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, metric, n: int
) -> tuple[float, float, float]:
    """Sección F: intervalo de confianza por bootstrap de una métrica.

    Devuelve (valor, límite inferior, límite superior).
    """
    raise NotImplementedError  # Fase 0


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Umbral óptimo elegido en validación (p. ej. índice de Youden)."""
    raise NotImplementedError  # Fase 0
