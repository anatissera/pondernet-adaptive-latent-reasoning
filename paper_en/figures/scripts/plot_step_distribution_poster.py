#!/usr/bin/env python3
"""Version POSTER del histograma de n_i (distribucion de pasos en GSM8k-Aug).

Como `plot_k_sweep_poster.py`: estilo de la casa (`figstyle.py`, indigo UdeSA +
frambuesa, Palatino/Pagella) pero con **fuentes enormes** para que se lea al
tamaño chico del poster. Calcula n_i = max(0, cot.count('<<') - 1) desde los
datos reales.

Uso (desde la raiz del repo):
    python3 paper/figures/scripts/plot_step_distribution_poster.py \
        --data data/gsm8k_aug/subsamples/train100k.jsonl \
        --out Poster_NLP/paperfigures/step_distribution.png
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

_PAGELLA_DIR = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre")
_PAGELLA_NAME = None
if _PAGELLA_DIR.exists():
    for _f in _PAGELLA_DIR.glob("texgyrepagella-*.otf"):
        fm.fontManager.addfont(str(_f))
    _PAGELLA_NAME = fm.FontProperties(
        fname=str(_PAGELLA_DIR / "texgyrepagella-regular.otf")).get_name()


def _use_pagella():
    if _PAGELLA_NAME:
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": [_PAGELLA_NAME, "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": _PAGELLA_NAME,
            "mathtext.it": f"{_PAGELLA_NAME}:italic",
            "mathtext.bf": f"{_PAGELLA_NAME}:bold",
        })


FS_LABEL = 30
FS_TICK = 26
FS_LEGEND = 30


def load_counts(data_path: Path):
    counts = []
    with data_path.open() as f:
        for line in f:
            cot = json.loads(line).get("cot", "")
            counts.append(max(0, cot.count("<<") - 1))
    return counts


def plot(counts, out_path: Path, x_max: int = 8):
    n = len(counts)
    freq = Counter(counts)
    mean_n = sum(counts) / n
    xs = list(range(0, x_max + 1))
    ys = [freq.get(x, 0) / n for x in xs]
    tail = sum(c for x, c in freq.items() if x > x_max)

    figstyle.set_style("poster")
    _use_pagella()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(xs, ys, color=figstyle.BLUE, width=0.74, zorder=2)
    ax.axvline(mean_n, color=figstyle.ACCENT, linestyle="--", linewidth=3.0,
               zorder=3, label=fr"$\mu = {mean_n:.2f}$")
    ax.legend(loc="upper right", frameon=False, fontsize=FS_LEGEND,
              labelcolor=figstyle.ACCENT, handlelength=1.4, borderpad=0.2)
    ax.set_xlabel(r"$n_i$ (instance steps)", fontsize=FS_LABEL,
                  color=figstyle.TEXT)
    ax.set_ylabel("Fraction", fontsize=FS_LABEL, color=figstyle.TEXT)
    ax.set_xticks(xs)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=figstyle.GRID, linewidth=1.0, alpha=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(figstyle.GRID)
        ax.spines[spine].set_linewidth(1.6)
    ax.tick_params(colors=figstyle.TEXT, labelsize=FS_TICK, width=1.6, length=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[poster] n={n}  mean n_i={mean_n:.4f}  tail(>{x_max})={tail}  -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("data/gsm8k_aug/subsamples/train100k.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("Poster_NLP/paperfigures/step_distribution.png"))
    p.add_argument("--x-max", type=int, default=8)
    args = p.parse_args()
    plot(load_counts(args.data), args.out, x_max=args.x_max)
