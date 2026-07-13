#!/usr/bin/env python3
"""
Figura HERO para el poster: la frontera exactitud-computo.

Un solo mensaje, legible de lejos: el baseline de K fijo gasta 6 tokens latentes
en TODA entrada; el modelo adaptativo alcanza la MISMA exactitud (~39-40%) barriendo
una frontera hacia la izquierda -- hasta la mitad de los tokens. Se grafica una sola
curva (M2, los 4 umbrales de parada), y una nota aclara que cada punto es el MISMO
modelo con distinto presupuesto, para que la curva no se lea como modelos distintos.

Fuente: paper/results_test/frontier_test.json

Uso (desde la raiz del repo):
    .venv/bin/python paper/figures/scripts/plot_poster_frontier.py \
        [--frontier paper/results_test/frontier_test.json] [--out paper/figures/poster_frontier.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# paleta de la casa: purpura como color principal, azul tenue como secundario
PURPLE = "#6a4fa3"  # protagonista (modelo adaptativo)
BLUE = "#4d84c0"    # azul secundario tenue
RED = "#e34948"     # baseline de K fijo (referencia)
GRAY = "#8a897f"
LIGHT = "#b9c9e0"   # azul tenue para las otras configuraciones
GRID = "#c3c2b7"
TEXT = "#52514e"


def curve(d, model):
    pts = sorted(((v["steps"], v["acc"]) for v in d[model].values()))
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(d, out_path):
    base_s = d["baseline"]["test"]["steps"]
    base_a = d["baseline"]["test"]["acc"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    # banda de paridad (exactitud esencialmente plana) - muy tenue, sin linea
    ax.axhspan(base_a - 0.55, base_a + 0.65, color=PURPLE, alpha=0.06, zorder=0)

    # protagonista: la frontera adaptativa (M2), linea continua sin marcadores
    xs, ys = curve(d, "M2")
    ax.plot(xs, ys, color=PURPLE, linewidth=3.6, zorder=4)

    # baseline: un unico punto
    ax.plot([base_s], [base_a], marker="D", markersize=16, color=BLUE, zorder=5)

    # titular
    ax.text(4.3, 40.95, "Mitad del cómputo, misma accuracy",
            ha="center", va="top", fontsize=18, color=TEXT, fontweight="bold")

    # callouts directos de los dos puntos comparables (a la misma exactitud)
    knee_s, knee_a = 3.011, 39.88  # M2 @ umbral 0.5
    ax.annotate("Modelo adaptativo", xy=(knee_s, knee_a),
                xytext=(2.02, 40.35), ha="left", va="center",
                fontsize=14, color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.5))
    ax.annotate("Baseline $K$ fijo\n6 tokens", xy=(base_s, base_a),
                xytext=(5.65, 38.7), ha="center", va="center",
                fontsize=14, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.5))

    # flecha del ahorro entre los dos puntos comparables
    ax.annotate("", xy=(knee_s + 0.15, base_a), xytext=(base_s - 0.28, base_a),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.4,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.text((knee_s + base_s) / 2, base_a + 0.1, "$-$50% cómputo",
            ha="center", va="bottom", fontsize=14, color=TEXT, fontweight="bold")

    ax.set_xlabel("Tokens latentes por problema (cómputo)", fontsize=17, color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=17, color=TEXT)
    ax.set_xlim(1.6, 7.0)
    ax.set_ylim(37.8, 41.0)
    ax.tick_params(colors=TEXT, labelsize=14)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[poster] frontier -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frontier", type=Path, default=Path("paper/results_test/frontier_test.json"))
    p.add_argument("--out", type=Path, default=Path("paper/figures/poster_frontier.png"))
    args = p.parse_args()
    d = json.loads(args.frontier.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(d, args.out)
