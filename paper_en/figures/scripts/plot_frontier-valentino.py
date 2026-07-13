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

# Umbrales de parada, en orden creciente (definen el barrido de cada curva).
THRESHOLDS = ["0.3", "0.4", "0.5", "0.8"]

# Paleta: purpura como color principal, azules tenues como secundarios.
PURPLE = "#6a4fa3"       # M2 (configuracion protagonista)
BLUE = "#4d84c0"         # M0 (azul secundario)
BLUE_LIGHT = "#6aa8dd"   # M1 (azul secundario mas claro)
RED = "#e34948"          # baseline de K fijo (referencia)

# Configuraciones: (clave en json, etiqueta leyenda, color).
MODELS = [
    ("M0", "M0", BLUE),
    ("M1", "M1", BLUE_LIGHT),
    ("M2", "M2", PURPLE),
]


def curve(d, model):
    """pasos, accuracy ordenados por umbral de parada."""
    steps = [d[model][t]["steps"] for t in THRESHOLDS]
    acc = [d[model][t]["acc"] for t in THRESHOLDS]
    return steps, acc


def plot(d, out_path):
    base_s = d["baseline"]["test"]["steps"]
    base_a = d["baseline"]["test"]["acc"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for model, label, color in MODELS:
        xs, ys = curve(d, model)
        ax.plot(xs, ys, color=color, linewidth=2, marker="D", markersize=4,
                label=label)

    # baseline de K fijo: un único punto en K=6 pasos (referencia)
    ax.plot([base_s], [base_a], marker="D", markersize=9, color=RED,
            linestyle="none", label=r"Fixed-$K$ baseline", zorder=5)

    ax.set_xlabel("Average latent steps")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(1.6, 7.2)
    ax.set_ylim(37.8, 40.8)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"escrito {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="paper/results_test/frontier_test.json")
    ap.add_argument("--out", default="paper/figures/frontier_test.png")
    args = ap.parse_args()
    with open(args.frontier) as f:
        d = json.load(f)
    plot(d, args.out)


if __name__ == "__main__":
    main()
