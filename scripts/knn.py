"""Entrena y evalúa el clasificador K vecinos más cercanos sobre RAGTruth.

Carga los datos, extrae features, valida con GroupKFold por `source`, calcula
las métricas A–F y guarda informe y figuras.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # Fase 0: load_ragtruth -> extract_features -> cross_validate(build_knn())
    # -> métricas A–F -> informe en outputs/reports/ y figuras en outputs/figures/.
    raise NotImplementedError  # Fase 0


if __name__ == "__main__":
    main()
