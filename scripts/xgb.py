"""Entrena y evalúa el modelo XGBoost regularizado sobre RAGTruth.

Carga los datos, extrae features, valida con GroupKFold por `source`, calcula
las métricas A–F y guarda informe y figuras (incluida la ganancia por variable).
El fichero se llama `xgb.py` (no `xgboost.py`) para no ensombrecer al paquete
`xgboost` al ejecutar el script.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # Fase 0: load_ragtruth -> extract_features -> cross_validate(build_xgboost())
    # -> métricas A–F -> informe en outputs/reports/ y figuras en outputs/figures/.
    raise NotImplementedError  # Fase 0


if __name__ == "__main__":
    main()
