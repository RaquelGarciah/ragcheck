"""Análisis descriptivo del dataset RAGTruth (Nivel 1).

Genera las figuras del capítulo de datos: balance de clases, tasa de
alucinación por tarea y por modelo generador, longitudes de respuesta y
contexto, reuso de documentos (justifica GroupKFold), matriz de correlación de
features y separabilidad por clase. Todas las figuras van a outputs/figures/.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # Fase 0: load_ragtruth -> extract_features -> figuras seaborn del Nivel 1.
    raise NotImplementedError  # Fase 0


if __name__ == "__main__":
    main()
