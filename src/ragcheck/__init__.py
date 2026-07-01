"""ragcheck — detección de alucinaciones en RAG con ML clásico.

API pública de inferencia rápida:

    from ragcheck import score
    score(response, source)  # -> probabilidad de alucinación en [0, 1]
"""

from ragcheck.inference import load_model, score

__version__ = "0.1.0"
__all__ = ["score", "load_model", "__version__"]
