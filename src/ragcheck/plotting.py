"""Estilo común de figuras (seaborn) y helpers de dibujo.

Centraliza tema, paleta, tamaños y exportación para que todas las figuras del
proyecto salgan homogéneas y con acabado de publicación.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = "colorblind"


def set_style() -> None:
    """Fija el tema seaborn del proyecto (llamar una vez por script)."""
    sns.set_theme(context="paper", style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def savefig(fig: plt.Figure, name: str) -> Path:
    """Guarda la figura en `outputs/figures/<name>` en PNG y PDF a 300 dpi."""
    raise NotImplementedError  # Fase 0
