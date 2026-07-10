#!/usr/bin/env python3
"""
Figuras del barrido oráculo de k sobre CODI (Módulo A, Sección~\\ref{sec:ksweep}).

Genera dos figuras a partir de los resultados de Option-A/scripts/run_k_sweep.py
(backend CODI, GSM8k train, k_max=8, max_new_tokens=128):

  1. k_sweep_accuracy.png  — accuracy agregada forzando k pasos latentes para
     todas las instancias (train completo n=7473 + submuestra piloto n=200).
  2. k_sweep_kstar.png     — distribución del presupuesto mínimo óptimo k* por
     instancia (train completo n=7473).

Los valores están volcados de las corridas en la máquina remota (simcot-t4):
  Option-A/results/k_sweep_train_200_codi.jsonl   (n=200,  ~8 min)
  Option-A/results/k_sweep_train_codi.jsonl        (n=7473, ~283 min)

Uso (desde la raíz del repo):
    python3 paper/figures/scripts/plot_k_sweep.py \
        [--out-dir paper/figures]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# Paleta de la casa (commit 3b0fdc9): purpura principal, azul tenue secundario.
PURPLE = "#6a4fa3"  # serie protagonista (train completo)
BLUE = "#4d84c0"    # serie secundaria (submuestra piloto)
GRID = "#c3c2b7"
TEXT = "#52514e"

KS = list(range(1, 9))

# Accuracy agregada por k (fracción de respuestas correctas).
ACC_FULL = [0.4077, 0.4267, 0.4787, 0.5785, 0.6370, 0.6395, 0.6366, 0.6308]  # n=7473
ACC_200 = [0.3700, 0.4450, 0.4950, 0.5450, 0.6450, 0.6650, 0.6500, 0.6550]   # n=200

# Distribución de k* (menor k que alcanza el mejor puntaje), train completo.
KSTAR_FULL = [4939, 1243, 396, 446, 319, 75, 32, 23]  # suma = 7473
N_FULL = sum(KSTAR_FULL)


def style_axes(ax):
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)


def plot_accuracy(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    ax.plot(KS, [a * 100 for a in ACC_FULL], color=PURPLE, marker="D",
            markersize=4, linewidth=2, zorder=3,
            label=r"Train completo ($n{=}7473$)")
    ax.plot(KS, [a * 100 for a in ACC_200], color=BLUE, marker="D",
            markersize=4, linewidth=1.6, linestyle="--", zorder=2,
            label=r"Submuestra piloto ($n{=}200$)")

    ax.set_xlabel(r"$k$  (pasos latentes forzados)", fontsize=10, color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=TEXT)
    ax.set_xticks(KS)
    ax.set_ylim(33, 70)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_k_sweep] escrito {out_path}")


def plot_kstar(out_path: Path) -> None:
    pcts = [c / N_FULL * 100 for c in KSTAR_FULL]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    ax.bar(KS, pcts, color=PURPLE, width=0.72, zorder=2)
    for k, pct in zip(KS, pcts):
        ax.text(k, pct + 1.2, f"{pct:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)

    ax.set_xlabel(r"$k^*$  (presupuesto mínimo óptimo de la instancia)",
                  fontsize=10, color=TEXT)
    ax.set_ylabel("Ejemplos (%)", fontsize=10, color=TEXT)
    ax.set_xticks(KS)
    ax.set_ylim(0, 74)

    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot_k_sweep] escrito {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(args.out_dir / "k_sweep_accuracy.png")
    plot_kstar(args.out_dir / "k_sweep_kstar.png")
