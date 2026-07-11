#!/usr/bin/env python3
"""
Figura de la frontera accuracy--cómputo (Figura~\\ref{fig:frontera} del paper).

Grafica en matplotlib (estilo por defecto, a color) los valores REALES leídos de
paper/results_test/frontier_test.json. Cada configuración adaptativa (M0/M1/M2) es
un barrido sobre los umbrales de parada; el baseline de K fijo es una línea de
referencia horizontal.

Uso (desde la raíz del repo):
    .venv/bin/python paper/figures/scripts/plot_frontier.py \
        [--frontier paper/results_test/frontier_test.json] \
        [--out paper/figures/frontier_test.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle
from figstyle import GRID, TEXT

# Umbrales de parada, en orden creciente (definen el barrido de cada curva).
THRESHOLDS = ["0.3", "0.4", "0.5", "0.8"]

# Paleta de valentino: purpura como color protagonista (M2), azules tenues como
# secundarios (M0/M1) y rojo para la referencia de K fijo.
PURPLE = "#6a4fa3"       # M2 (configuracion protagonista)
BLUE = "#4d84c0"         # M0 (azul secundario)
BLUE_LIGHT = "#6aa8dd"   # M1 (azul secundario mas claro)
RED = "#e34948"          # baseline de K fijo (referencia)

# Configuraciones: (clave en json, etiqueta leyenda, marcador, color).
MODELS = [
    ("M0", "M0", "o", BLUE),
    ("M1", "M1", "s", BLUE_LIGHT),
    ("M2", "M2", "^", PURPLE),
]


def curve(d, model):
    """pasos, accuracy ordenados por umbral de parada."""
    steps = [d[model][t]["steps"] for t in THRESHOLDS]
    acc = [d[model][t]["acc"] for t in THRESHOLDS]
    return steps, acc


def plot(d, out_path, profile="paper"):
    figstyle.set_style(profile)
    FS = figstyle.sizes(profile)
    base_s = d["baseline"]["test"]["steps"]
    base_a = d["baseline"]["test"]["acc"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for model, label, marker, color in MODELS:
        xs, ys = curve(d, model)
        ax.plot(xs, ys, color=color, marker=marker, markersize=7.5, linewidth=2.2,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3, label=label)

    # baseline de K fijo: un único punto de referencia (en K=base_s pasos)
    ax.plot([base_s], [base_a], marker="D", markersize=9, color=RED,
            markeredgecolor="white", markeredgewidth=1.0, linestyle="none",
            zorder=5, label=r"Baseline $K$ fijo")

    ax.set_xlabel("Pasos latentes promedio", fontsize=FS["label"], color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=FS["label"], color=TEXT)
    ax.set_xlim(1.6, 7.2)
    ax.set_ylim(37.8, 40.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=FS["tick"])
    ax.legend(loc="lower right", frameon=False, fontsize=FS["legend"])

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"escrito {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="paper/results_test/frontier_test.json")
    ap.add_argument("--out", default="paper/figures/frontier_test.png")
    ap.add_argument("--profile", choices=["paper", "poster"], default="paper")
    args = ap.parse_args()
    with open(args.frontier) as f:
        d = json.load(f)
    plot(d, args.out, profile=args.profile)


if __name__ == "__main__":
    main()
