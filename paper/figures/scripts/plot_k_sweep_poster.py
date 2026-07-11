#!/usr/bin/env python3
"""Versiones POSTER de las figuras del barrido de k (Modulo A).

A diferencia de `plot_k_sweep.py` (paleta purpura vieja, fuentes ~10pt pensadas
para el informe), estas usan el estilo de la casa `figstyle.py` (indigo UdeSA,
Palatino) y **fuentes enormes**, porque en el poster van chicas (lado a lado en
una columna) y tienen que leerse a distancia.

Mismos datos verificados que `plot_k_sweep.py`.

Uso (desde la raiz del repo):
    python3 paper/figures/scripts/plot_k_sweep_poster.py \
        --out-dir Poster_NLP/paperfigures
"""
import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

# Palatino (TeX Gyre Pagella) vive en el arbol de TeX Live y matplotlib no lo
# escanea por defecto; lo registramos por path para que las figuras casen con
# el \usepackage{palatino} del poster.
_PAGELLA_DIR = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre")
_PAGELLA_NAME = None
if _PAGELLA_DIR.exists():
    for _f in _PAGELLA_DIR.glob("texgyrepagella-*.otf"):
        fm.fontManager.addfont(str(_f))
    _PAGELLA_NAME = fm.FontProperties(
        fname=str(_PAGELLA_DIR / "texgyrepagella-regular.otf")).get_name()


def _use_pagella():
    """Fuerza Palatino (Pagella) en texto y mathtext si esta disponible."""
    if _PAGELLA_NAME:
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": [_PAGELLA_NAME, "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": _PAGELLA_NAME,
            "mathtext.it": f"{_PAGELLA_NAME}:italic",
            "mathtext.bf": f"{_PAGELLA_NAME}:bold",
        })

KS = list(range(1, 9))
ACC_FULL = [0.4077, 0.4267, 0.4787, 0.5785, 0.6370, 0.6395, 0.6366, 0.6308]  # n=7473
ACC_200 = [0.3700, 0.4450, 0.4950, 0.5450, 0.6450, 0.6650, 0.6500, 0.6550]   # n=200
KSTAR_FULL = [4939, 1243, 396, 446, 319, 75, 32, 23]  # suma = 7473
N_FULL = sum(KSTAR_FULL)

# Fuentes de POSTER: masivas. La figura se muestra a ~12 cm en A0 y se lee de lejos.
FS_LABEL = 30
FS_TICK = 26
FS_ANNOT = 25
FS_LEGEND = 24


def _style(ax):
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(figstyle.GRID)
        ax.spines[spine].set_linewidth(1.6)
    ax.tick_params(colors=figstyle.TEXT, labelsize=FS_TICK, width=1.6, length=6)


def plot_kstar(out_path: Path) -> None:
    figstyle.set_style("poster")
    _use_pagella()
    pcts = [c / N_FULL * 100 for c in KSTAR_FULL]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(KS, pcts, color=figstyle.BLUE, width=0.74, zorder=2)
    for k, pct in zip(KS, pcts):
        ax.text(k, pct + 1.6, f"{pct:.0f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=figstyle.TEXT)
    ax.set_xlabel(r"$k^{*}$ óptimo por instancia", fontsize=FS_LABEL,
                  color=figstyle.TEXT)
    ax.set_ylabel("Ejemplos (%)", fontsize=FS_LABEL, color=figstyle.TEXT)
    ax.set_xticks(KS)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 25, 50, 75])
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[poster] escrito {out_path}")


def plot_accuracy(out_path: Path) -> None:
    figstyle.set_style("poster")
    _use_pagella()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(KS, [a * 100 for a in ACC_FULL], color=figstyle.BLUE, marker="o",
            markersize=9, linewidth=3.4, zorder=3,
            label=r"Train ($n{=}7473$)")
    ax.plot(KS, [a * 100 for a in ACC_200], color=figstyle.ACCENT, marker="D",
            markersize=8, linewidth=2.6, linestyle="--", zorder=2,
            label=r"Piloto ($n{=}200$)")
    ax.set_xlabel(r"$k$ (pasos forzados)", fontsize=FS_LABEL, color=figstyle.TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=FS_LABEL, color=figstyle.TEXT)
    ax.set_xticks(KS)
    ax.set_ylim(33, 72)
    ax.set_yticks([40, 50, 60, 70])
    ax.legend(loc="lower right", frameon=False, fontsize=FS_LEGEND,
              handlelength=1.6, borderpad=0.2, labelspacing=0.25)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[poster] escrito {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=Path("Poster_NLP/paperfigures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_kstar(args.out_dir / "k_sweep_kstar.png")
    plot_accuracy(args.out_dir / "k_sweep_accuracy.png")
