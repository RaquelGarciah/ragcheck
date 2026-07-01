"""Wrappers finos de los cinco clasificadores clásicos, con semilla fija.

Cada `build_*` devuelve un estimador de scikit-learn/xgboost sin entrenar,
con los hiperparámetros de `config`.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ragcheck.config import (
    KNN_PARAMS,
    LOGREG_PARAMS,
    RANDOM_FOREST_PARAMS,
    SVM_PARAMS,
    XGBOOST_PARAMS,
)


def build_logreg() -> LogisticRegression:
    """Regresión logística (Cox, 1958). Interpretable por sus coeficientes."""
    return LogisticRegression(**LOGREG_PARAMS)


def build_xgboost() -> XGBClassifier:
    """Gradient boosting sobre árboles (Chen y Guestrin, 2016)."""
    return XGBClassifier(**XGBOOST_PARAMS)


def build_random_forest() -> RandomForestClassifier:
    """Bosque aleatorio (Breiman, 2001)."""
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)


def build_svm() -> SVC:
    """Máquina de vectores soporte con núcleo RBF (Cortes y Vapnik, 1995)."""
    return SVC(**SVM_PARAMS)


def build_knn() -> KNeighborsClassifier:
    """K vecinos más cercanos (Cover y Hart, 1967)."""
    return KNeighborsClassifier(**KNN_PARAMS)


# Registro nombre -> constructor, para iterar en los scripts de comparación.
BUILDERS = {
    "logreg": build_logreg,
    "xgboost": build_xgboost,
    "random_forest": build_random_forest,
    "svm": build_svm,
    "knn": build_knn,
}
