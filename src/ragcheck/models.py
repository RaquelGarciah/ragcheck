"""Wrappers finos de los cinco clasificadores clásicos, con semilla fija.

Cada `build_*` devuelve un estimador de scikit-learn/xgboost sin entrenar,
con los hiperparámetros de `config`.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ragcheck.config import (
    KNN_PARAMS,
    LOGREG_PARAMS,
    RANDOM_FOREST_PARAMS,
    SVM_PARAMS,
    XGBOOST_PARAMS,
)


def build_logreg() -> Pipeline:
    """Regresión logística (Cox, 1958) con estandarizado previo.

    El escalado es necesario porque la regularización penaliza el tamaño del
    coeficiente, que depende de la escala de la variable: sin estandarizar,
    `answer_len` (cientos de tokens) se penaliza mucho menos que las features en
    [0, 1], y el modelo cambiaría con las unidades. Con `StandardScaler` el
    castigo compara los coeficientes en igualdad.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", **LOGREG_PARAMS),
    )


def build_xgboost() -> XGBClassifier:
    """Gradient boosting sobre árboles (Chen y Guestrin, 2016)."""
    return XGBClassifier(**XGBOOST_PARAMS)


def build_random_forest() -> RandomForestClassifier:
    """Bosque aleatorio (Breiman, 2001)."""
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)


def build_svm() -> Pipeline:
    """SVM de núcleo RBF (Cortes y Vapnik, 1995) con estandarizado previo.

    El escalado es imprescindible: sin él, `answer_len` domina el núcleo RBF
    frente a las features en [0, 1].
    """
    return make_pipeline(StandardScaler(), SVC(**SVM_PARAMS))


def build_knn() -> Pipeline:
    """K vecinos más cercanos (Cover y Hart, 1967) con estandarizado previo.

    El escalado es imprescindible: la distancia euclídea la dominaría
    `answer_len` sin normalizar.
    """
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**KNN_PARAMS))


# Registro nombre -> constructor, para iterar en los scripts de comparación.
BUILDERS = {
    "logreg": build_logreg,
    "xgboost": build_xgboost,
    "random_forest": build_random_forest,
    "svm": build_svm,
    "knn": build_knn,
}
