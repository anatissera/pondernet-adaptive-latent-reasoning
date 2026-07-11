#!/usr/bin/env python3
"""
Genera las figuras de resultados del modelo adaptativo para el paper, a partir de
los `instance_results.json` crudos del conjunto de test held-out (n=1319).

Cuatro figuras, todas en el estilo de `plot_dataset_step_distribution.py`:

  A. accuracy_by_difficulty.png  -- exactitud por dificultad (cot_steps del dataset):
                                    modelo adaptativo vs baseline de K fijo. Barras agrupadas.
  B. steps_by_difficulty.png     -- pasos latentes promedio vs dificultad: el adaptativo
                                    escala con la dificultad; el baseline es plano en K=6.
  B2. steps_accuracy_by_difficulty.png -- accuracy y tokens usados vs dificultad en un
                                    mismo grafico con doble eje Y (dos lineas continuas).
  C. joint_matrix.png            -- histograma 2D dificultad (cot_steps) x pasos usados,
                                    normalizado por fila; visualiza el Spearman.
  D. steps_used_distribution.png -- distribucion de pasos usados del modelo adaptativo.

El modelo adaptativo protagonista es el punto de minimo computo a paridad (M2 @ umbral
0.5 en el repo; en las figuras se rotula solo como "modelo adaptativo", sin exponer los
hiperparametros al lector). El baseline es SIM-CoT/CODI de K fijo (K=6, 39.5%).

Uso (desde la raiz del repo):
    .venv/bin/python paper/figures/scripts/plot_model_results.py \
        [--results-dir paper/results_test] [--out-dir paper/figures]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# --- paleta de la casa: purpura como color principal, azul tenue como secundario ---
PURPLE = "#6a4fa3"  # adaptativo (nuestro metodo, protagonista) / eje accuracy
BLUE = "#4d84c0"    # serie/eje secundario (azul tenue, p.ej. tokens usados)
RED = "#e34948"     # linea de referencia (baseline constante / medias)
GRAY = "#8a897f"    # baseline como serie real (barras)
GRID = "#c3c2b7"
TEXT = "#52514e"

ADAPT_LABEL = "Modelo adaptativo"
BASE_LABEL = "Baseline $K$ fijo"


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def style_axes(ax):
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=9)


def difficulty_bins(rows, max_bin=5):
    """Agrupa por cot_steps con cola '5+'. Descarta cot_steps None."""
    acc = defaultdict(list)
    steps = defaultdict(list)
    for r in rows:
        c = r.get("cot_steps")
        if c is None:
            continue
        b = min(c, max_bin)
        acc[b].append(1.0 if r["correct"] else 0.0)
        steps[b].append(r["steps_used"])
    return acc, steps


def bin_labels(max_bin=5):
    xs = list(range(0, max_bin + 1))
    labels = [str(x) for x in xs[:-1]] + [f"{max_bin}+"]
    return xs, labels


# ---------------------------------------------------------------- Figura A
def plot_accuracy_by_difficulty(adapt, base, out_path):
    xs, labels = bin_labels()
    acc_a, _ = difficulty_bins(adapt)
    acc_b, _ = difficulty_bins(base)
    ns = [len(acc_a[x]) for x in xs]
    ya = [100 * np.mean(acc_a[x]) if acc_a[x] else 0 for x in xs]
    yb = [100 * np.mean(acc_b[x]) if acc_b[x] else 0 for x in xs]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    w = 0.38
    idx = np.arange(len(xs))
    ax.bar(idx - w / 2, ya, width=w, color=PURPLE, zorder=2, label=ADAPT_LABEL)
    ax.bar(idx + w / 2, yb, width=w, color=BLUE, zorder=2, label=BASE_LABEL)

    for i, (a, b) in enumerate(zip(ya, yb)):
        ax.text(i - w / 2, a + 1.0, f"{a:.0f}", ha="center", va="bottom",
                fontsize=6.5, color=PURPLE)
        ax.text(i + w / 2, b + 1.0, f"{b:.0f}", ha="center", va="bottom",
                fontsize=6.5, color=TEXT)

    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Pasos de razonamiento del dataset (dificultad)", fontsize=10, color=TEXT)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=TEXT)
    ax.set_ylim(0, 72)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[A] accuracy_by_difficulty -> {out_path}")


# ---------------------------------------------------------------- Figura B
def plot_steps_by_difficulty(adapt, base, out_path, k_fixed=6):
    xs, labels = bin_labels()
    _, steps_a = difficulty_bins(adapt)
    ya = [np.mean(steps_a[x]) if steps_a[x] else np.nan for x in xs]

    # Spearman sobre datos crudos (cot_steps no-None), sin agrupar la cola
    cot = [r["cot_steps"] for r in adapt if r.get("cot_steps") is not None]
    su = [r["steps_used"] for r in adapt if r.get("cot_steps") is not None]
    rho = spearmanr(cot, su).correlation

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    idx = np.arange(len(xs))
    ax.plot(idx, ya, color=PURPLE, linewidth=2,
            zorder=3, label=ADAPT_LABEL)
    ax.axhline(k_fixed, color=BLUE, linestyle="--", linewidth=1.6, zorder=2,
               label=f"{BASE_LABEL} ($K{{=}}{k_fixed}$)")

    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Pasos de razonamiento del dataset (dificultad)", fontsize=10, color=TEXT)
    ax.set_ylabel("Tokens latentes usados (prom.)", fontsize=10, color=TEXT)
    ax.set_ylim(0, k_fixed + 1.2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[B] steps_by_difficulty -> {out_path}  (Spearman={rho:+.3f})")


# --------------------------------------------------- Figura B2 (eje dual)
def plot_steps_and_accuracy_by_difficulty(adapt, out_path):
    """Accuracy y tokens usados vs dificultad, en un mismo grafico con doble eje Y
    (accuracy a la izquierda, tokens a la derecha). Dos lineas continuas."""
    xs, labels = bin_labels()
    acc_a, steps_a = difficulty_bins(adapt)
    acc = [100 * np.mean(acc_a[x]) if acc_a[x] else np.nan for x in xs]
    stp = [np.mean(steps_a[x]) if steps_a[x] else np.nan for x in xs]
    idx = np.arange(len(xs))

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(idx, acc, color=PURPLE, linewidth=2.2,
                   zorder=3, label="Accuracy (%)")
    l2, = ax2.plot(idx, stp, color=BLUE, linewidth=2.2,
                   zorder=3, label="Tokens latentes usados")

    ax1.set_xlabel(r"Pasos de razonamiento del dataset (dificultad)", fontsize=10, color=TEXT)
    ax1.set_ylabel("Accuracy (%)", fontsize=10, color=PURPLE)
    ax2.set_ylabel("Tokens latentes usados (prom.)", fontsize=10, color=BLUE)

    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 70)
    ax2.set_ylim(0, 6)
    ax1.tick_params(axis="x", colors=TEXT, labelsize=9)
    ax1.tick_params(axis="y", colors=PURPLE, labelsize=9)
    ax2.tick_params(axis="y", colors=BLUE, labelsize=9)

    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["left"].set_color(PURPLE)
    ax2.spines["right"].set_color(BLUE)
    ax1.spines["bottom"].set_color(GRID)

    ax1.legend(handles=[l1, l2], loc="upper center", frameon=False, fontsize=9,
               ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[B2] steps_accuracy_by_difficulty -> {out_path}")


# ---------------------------------------------------------------- Figura C
def plot_joint_matrix(adapt, out_path, max_cot=7, max_steps=8):
    cot_vals = list(range(0, max_cot + 1))
    step_vals = list(range(1, max_steps + 1))
    M = np.zeros((len(cot_vals), len(step_vals)))
    for r in adapt:
        c, s = r.get("cot_steps"), r["steps_used"]
        if c is None or c > max_cot or not (1 <= s <= max_steps):
            continue
        M[c, s - 1] += 1
    row_tot = M.sum(axis=1, keepdims=True)
    row_tot[row_tot == 0] = 1
    Mn = M / row_tot  # P(pasos usados | dificultad)

    cot = [r["cot_steps"] for r in adapt if r.get("cot_steps") is not None]
    su = [r["steps_used"] for r in adapt if r.get("cot_steps") is not None]
    rho = spearmanr(cot, su).correlation

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    im = ax.imshow(Mn, aspect="auto", cmap="Purples", origin="lower",
                   vmin=0, vmax=Mn.max())

    ax.set_xticks(range(len(step_vals)))
    ax.set_xticklabels(step_vals)
    ax.set_yticks(range(len(cot_vals)))
    ax.set_yticklabels(cot_vals)
    ax.set_xlabel("Tokens latentes usados por el modelo", fontsize=10, color=TEXT)
    ax.set_ylabel(r"Pasos del dataset (dificultad)", fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    ax.text(0.97, 0.05, rf"Spearman $= {rho:+.3f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Fracción por fila  $P(\\mathrm{pasos}\\mid\\mathrm{dificultad})$",
                   fontsize=8.5, color=TEXT)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.outline.set_edgecolor(GRID)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[C] joint_matrix -> {out_path}")


# ---------------------------------------------------------------- Figura D
def plot_steps_used_distribution(adapt, out_path, k_fixed=6):
    steps = [r["steps_used"] for r in adapt]
    n = len(steps)
    mean_s = np.mean(steps)
    xs = list(range(1, max(max(steps), k_fixed) + 1))
    ys = [steps.count(x) / n for x in xs]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(xs, ys, color=PURPLE, width=0.72, zorder=2)

    ax.set_xlabel("Tokens latentes usados", fontsize=10, color=TEXT)
    ax.set_ylabel("Fracción del test", fontsize=10, color=TEXT)
    ax.set_xticks(xs)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[D] steps_used_distribution -> {out_path}  (media={mean_s:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("paper/results_test"))
    p.add_argument("--adapt", default="M2/thr0.5",
                   help="ruta relativa al run adaptativo protagonista")
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = p.parse_args()

    adapt = load(args.results_dir / args.adapt / "instance_results.json")
    base = load(args.results_dir / "baseline" / "instance_results.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_difficulty(adapt, base, args.out_dir / "accuracy_by_difficulty.png")
    plot_steps_by_difficulty(adapt, base, args.out_dir / "steps_by_difficulty.png")
    plot_steps_and_accuracy_by_difficulty(adapt, args.out_dir / "steps_accuracy_by_difficulty.png")
    plot_joint_matrix(adapt, args.out_dir / "joint_matrix.png")
    plot_steps_used_distribution(adapt, args.out_dir / "steps_used_distribution.png")
