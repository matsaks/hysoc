"""Shared matplotlib style for thesis figures.

Call ``apply_latex_style()`` at the top of ``main()`` to match the LaTeX thesis
fonts and sizing. Figures use 9 pt body / 8 pt legend type with 1:1 scaling
between the matplotlib ``figsize`` width and the LaTeX ``\\includegraphics``
width (5.2 in at ``0.8\\textwidth``; 6.5 in at ``\\textwidth`` for wide grids).
Use ``recommended_figsize`` for a default that respects the rule.
"""

DEFAULT_INCLUDE_WIDTH_INCHES = 5.2  # 0.8 * 6.5 in textwidth
FULL_TEXTWIDTH_INCHES = 6.5


def apply_latex_style(use_latex: bool = True) -> None:
    import matplotlib as mpl

    rc: dict = {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.7,
    }
    if use_latex:
        rc.update({
            "text.usetex": True,
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{lmodern}",
        })
    else:
        rc.update({
            "font.serif": ["CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    mpl.rcParams.update(rc)


def recommended_figsize(
    n_cols: int,
    n_rows: int,
    *,
    full_width: bool = False,
    panel_aspect: float = 1.0,
) -> tuple[float, float]:
    """Return a ``(width, height)`` figsize aligned with the LaTeX include width.

    Width is fixed (5.2 in default, 6.5 in when ``full_width``) so matplotlib pt
    and on-page pt match 1:1; height = width / n_cols * panel_aspect * n_rows.
    """
    width = FULL_TEXTWIDTH_INCHES if full_width else DEFAULT_INCLUDE_WIDTH_INCHES
    height = (width / n_cols) * panel_aspect * n_rows
    return (width, height)
