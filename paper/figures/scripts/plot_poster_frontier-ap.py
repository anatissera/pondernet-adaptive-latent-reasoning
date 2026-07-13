#!/usr/bin/env python3
"""
Figura HERO para el poster: la frontera accuracy-computo.

Un solo mensaje, legible de lejos: el baseline de K fijo gasta 6 tokens latentes
en TODA entrada; el modelo adaptativo alcanza el MISMO accuracy (~39-40%) barriendo
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

import figstyle
from figstyle import BLUE, ACCENT as RED, GRID, TEXT


def curve(d, model):
    pts = sorted(((v["steps"], v["acc"]) for v in d[model].values()))
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(d, out_path, profile="poster"):
    figstyle.set_style(profile)
    base_s = d["baseline"]["test"]["steps"]
    base_a = d["baseline"]["test"]["acc"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    # banda de paridad (accuracy esencialmente plano) - muy tenue, sin linea
    ax.axhspan(base_a - 0.55, base_a + 0.65, color=figstyle.BAND, alpha=0.7, zorder=0)

    # protagonista: la frontera adaptativa (M2)
    xs, ys = curve(d, "M2")
    ax.plot(xs, ys, color=BLUE, linewidth=3.0, zorder=4, solid_capstyle="round")
    ax.plot(xs, ys, color=BLUE, marker="o", markersize=11, linestyle="none",
            markeredgecolor="white", markeredgewidth=1.6, zorder=5)

    # baseline: un unico punto
    ax.plot([base_s], [base_a], marker="D", markersize=14, color=RED,
            markeredgecolor="white", markeredgewidth=1.6, zorder=6)

    # callouts directos de los dos puntos comparables (al mismo accuracy)
    knee_s, knee_a = 3.011, 39.88  # M2 @ umbral 0.5
    ax.annotate("Modelo adaptativo\n3 pasos", xy=(knee_s, knee_a),
                xytext=(2.02, 40.55), ha="left", va="center",
                fontsize=23, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.3))
    ax.annotate("Baseline $K$ fijo\n6 pasos", xy=(base_s, base_a),
                xytext=(5.55, 38.55), ha="center", va="center",
                fontsize=23, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.3))

    # flecha del ahorro entre los dos puntos comparables
    ax.annotate("", xy=(knee_s + 0.15, base_a), xytext=(base_s - 0.28, base_a),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.9,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text((knee_s + base_s) / 2, base_a + 0.13, "$-$50% cómputo",
            ha="center", va="bottom", fontsize=23, color=TEXT)

    ax.set_xlabel("Pasos latentes por problema (cómputo)", fontsize=21, color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=21, color=TEXT)
    ax.set_xlim(1.6, 6.6)
    ax.set_ylim(37.8, 40.9)
    ax.tick_params(colors=TEXT, labelsize=18)
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
    p.add_argument("--profile", choices=["paper", "poster"], default="poster")
    args = p.parse_args()
    d = json.loads(args.frontier.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(d, args.out, profile=args.profile)
