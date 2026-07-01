"""Análisis descriptivo de los spans alucinados (Nivel 2).

Parsea los spans anotados y genera las figuras que caracterizan el fenómeno:
posición relativa en la respuesta, longitud del span, categoría gramatical
(POS) y tipo de entidad (NER con spacy), número de spans por respuesta,
cobertura, comparación por tarea y fracción de números inventados ausentes de
la fuente. Todas las figuras van a outputs/figures/.
"""

import random

import numpy as np

from ragcheck.config import SEED

random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # Fase 0: load_ragtruth -> span_distributions -> figuras seaborn del Nivel 2.
    raise NotImplementedError  # Fase 0


if __name__ == "__main__":
    main()
