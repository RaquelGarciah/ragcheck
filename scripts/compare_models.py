"""Comparativa de los cinco modelos: reproduce la línea base del TFG.

Entrena y evalúa los cinco clasificadores con la misma partición, consolida
las métricas A–F en una tabla comparativa (media ± IC), superpone las curvas
ROC y PR, y añade los baselines honestos (azar, clase mayoritaria, una sola
feature). Genera outputs/reports/e0_baseline.md.
"""

import random

import numpy as np

from ragcheck.config import SEED
from ragcheck.models import BUILDERS

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # Fase 0: para cada nombre en BUILDERS, cross_validate y métricas A–F;
    # consolidar tabla + figuras comparativas + baselines.
    _ = BUILDERS
    raise NotImplementedError  # Fase 0


if __name__ == "__main__":
    main()
