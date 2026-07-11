#!/usr/bin/env python3
"""
Plot the distribution of ground-truth reasoning-step counts (n_i) in GSM8k-Aug.

n_i is the same statistic the adaptive prior anchors to (Section 3.2 of the
paper): the number of calculator annotations in the CoT minus one,
n_i = max(0, cot.count('<<') - 1).

Usage:
    python scripts/plot_dataset_step_distribution.py \
        [--data data/gsm8k_aug/subsamples/train100k.jsonl] \
        [--out paper/figures/step_distribution.png]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle
from figstyle import BLUE, ACCENT, GRID, TEXT


def load_step_counts(data_path: Path) -> list[int]:
    counts = []
    with data_path.open() as f:
        for line in f:
            example = json.loads(line)
            cot = example.get("cot", "")
            counts.append(max(0, cot.count("<<") - 1))
    return counts


def plot(counts: list[int], out_path: Path, K_max: int = 12, profile: str = "paper") -> None:
    n = len(counts)
    freq = Counter(counts)
    xs = list(range(0, max(freq) + 1))
    ys = [freq.get(x, 0) / n for x in xs]
    mean_n = sum(counts) / n

    over_kmax = sum(c for x, c in freq.items() if x > K_max)

    figstyle.set_style(profile)
    FS = figstyle.sizes(profile)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    ax.bar(xs, ys, color=BLUE, width=0.72, zorder=2)
    ax.axvline(mean_n, color=ACCENT, linestyle="--", linewidth=1.6, zorder=3,
               label=f"$\\mu = {mean_n:.2f}$")
    ax.legend(loc="upper right", frameon=False, fontsize=FS["legend"], labelcolor=ACCENT)

    ax.set_xlabel(r"$n_i$  (pasos de razonamiento de la instancia)", fontsize=FS["label"], color=TEXT)
    ax.set_ylabel("Fracción del dataset", fontsize=FS["label"], color=TEXT)
    ax.set_xticks(xs)

    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=FS["tick"])

    if over_kmax:
        ax.text(0.99, 0.95, f"{over_kmax} ejemplos con $n_i > {K_max}$ (fuera de rango, no graficados)",
                 transform=ax.transAxes, ha="right", va="top", fontsize=FS["tick"] * 0.72, color=TEXT)
        ax.set_xlim(-0.6, K_max + 0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[plot_dataset_step_distribution] n={n}  mean n_i={mean_n:.2f}  -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k_aug/subsamples/train100k.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/step_distribution.png"))
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--profile", choices=["paper", "poster"], default="paper")
    args = parser.parse_args()

    counts = load_step_counts(args.data)
    plot(counts, args.out, K_max=args.k_max, profile=args.profile)
