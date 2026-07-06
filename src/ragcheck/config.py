"""Configuración única del proyecto: semilla, rutas e hiperparámetros.

Todo lo que otro módulo necesite parametrizar se lee de aquí, nunca se
codifica a mano en los scripts.
"""

from pathlib import Path

SEED = 42

# Raíz del repositorio (config.py está en src/ragcheck/).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Dataset en HuggingFace Hub.
DATASET = "wandb/RAGTruth-processed"

# Hiperparámetros por modelo (semilla 42 en todos). Los defaults de LogReg y
# XGBoost reproducen la línea base de la guía; el resto son puntos de partida
# razonables que los scripts pueden refinar con búsqueda en validación.
LOGREG_PARAMS = {"max_iter": 2000, "random_state": SEED}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2,
    "eval_metric": "logloss",
    "random_state": SEED,
}

RANDOM_FOREST_PARAMS = {"n_estimators": 300, "random_state": SEED, "n_jobs": -1}

SVM_PARAMS = {"kernel": "rbf", "probability": True, "random_state": SEED}

KNN_PARAMS = {"n_neighbors": 15}

# Evaluación.
N_SPLITS = 5  # folds de GroupKFold por `context`
N_BOOTSTRAP = 1000  # remuestreos para intervalos de confianza

# Modelo que sirve la inferencia por defecto (el mejor de la comparativa).
DEFAULT_MODEL = "xgboost"
