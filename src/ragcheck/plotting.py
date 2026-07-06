"""Estilo común de figuras (seaborn) y helpers de dibujo.

Centraliza tema, paleta, tamaños y exportación para que todas las figuras del
proyecto salgan homogéneas y con acabado de publicación.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PALETTE = "colorblind"

# Colores estables reutilizados en todo el capítulo.
TASKS = ("Data2txt", "QA", "Summary")
_CB = sns.color_palette(PALETTE)
TASK_COLOR = {"Data2txt": _CB[0], "QA": _CB[1], "Summary": _CB[2]}
CLASS_COLOR = {"alucina": _CB[3], "limpia": "0.82"}


def set_style() -> None:
    """Fija el tema seaborn del proyecto (llamar una vez por script)."""
    sns.set_theme(context="paper", style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def coma(x: float, dec: int = 2) -> str:
    """Formatea un número con coma decimal (convención española)."""
    return f"{x:.{dec}f}".replace(".", ",")


def eje_coma(ax, axis: str = "y", dec: int = 1) -> None:
    """Pone coma decimal en el eje indicado (`"x"` o `"y"`) del `ax`."""
    fmt = mticker.FuncFormatter(lambda v, _: coma(v, dec))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def savefig(fig: plt.Figure, name: str) -> Path:
    """Guarda la figura en `outputs/figures/<name>` en PNG y PDF a 300 dpi.

    Devuelve la ruta del PNG. `name` va sin extensión.
    """
    from ragcheck.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(png)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    return png
