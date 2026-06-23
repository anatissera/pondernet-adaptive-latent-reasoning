#!/usr/bin/env python3
"""
Post-process a single PonderNet eval results directory.

Reads instance_results.json (must already have cot_steps populated),
writes cot_steps_matrix.png into the same dir.

Usage:
    python scripts/postprocess_eval.py <results_dir>

<results_dir> is a threshold-level directory, e.g.
    results/05-.../perinstance-g0.05-b1.5-ep5/ep4/thr0.8/
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr


def run(results_dir: Path) -> None:
    ir_path = results_dir / "instance_results.json"
    if not ir_path.exists():
        raise FileNotFoundError(ir_path)

    records = json.loads(ir_path.read_text())

    # Read threshold from summary.json if available, else infer from dir name
    summ_path = results_dir / "summary.json"
    if summ_path.exists():
        thr = json.loads(summ_path.read_text()).get("threshold", 0.0)
    else:
        name = results_dir.name  # e.g. "thr0.8"
        thr  = float(name.replace("thr", "")) if name.startswith("thr") else 0.0

    steps    = [r["steps_used"] for r in records]
    cot_vals = [r.get("cot_steps") for r in records]

    valid = [(s, c) for s, c in zip(steps, cot_vals) if c is not None]
    if not valid:
        print(f"[postprocess] no cot_steps in {ir_path} — skipping", flush=True)
        return

    vs, vc = zip(*valid)
    r_val, p_val = spearmanr(vs, vc)

    all_cot_steps    = sorted(set(vc))
    all_latent_steps = sorted(set(vs))

    # per-bin stats
    bins = defaultdict(lambda: {"latent": [], "correct": [], "dist": defaultdict(int)})
    for rec in records:
        cs = rec.get("cot_steps")
        if cs is None:
            continue
        ls = rec["steps_used"]
        bins[cs]["latent"].append(ls)
        bins[cs]["correct"].append(rec["correct"])
        bins[cs]["dist"][ls] += 1

    by_cot_steps = [
        {
            "cot_steps":                cs,
            "n":                        len(b["latent"]),
            "avg_latent_steps":         round(float(np.mean(b["latent"])), 4),
            "std_latent_steps":         round(float(np.std(b["latent"])), 4),
            "accuracy_pct":             round(100 * sum(b["correct"]) / len(b["correct"]), 2),
            "latent_step_distribution": {str(ls): b["dist"].get(ls, 0)
                                         for ls in all_latent_steps},
        }
        for cs, b in sorted(bins.items())
    ]

    crosstab_counts = [
        [bins[cs]["dist"].get(ls, 0) if cs in bins else 0 for ls in all_latent_steps]
        for cs in all_cot_steps
    ]

    print(f"[postprocess] spearman r={r_val:.4f}  p={p_val:.3g}", flush=True)

    # --- plot ---
    counts = np.array(crosstab_counts, dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    normed   = np.where(row_sums > 0, counts / row_sums, 0.0)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.45},
    )
    fig.suptitle(
        f"CoT steps vs latent steps  |  thr={thr}  |  Spearman r = {r_val:+.3f}",
        fontsize=10, y=0.99,
    )

    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for ri in range(normed.shape[0]):
        for ci in range(normed.shape[1]):
            n = int(counts[ri, ci])
            if n == 0:
                continue
            pct   = normed[ri, ci]
            color = "white" if pct > 0.55 else "black"
            ax.text(ci, ri, f"{n}\n({100*pct:.0f}%)",
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(all_latent_steps)))
    ax.set_xticklabels(all_latent_steps)
    ax.set_yticks(range(len(all_cot_steps)))
    ax.set_yticklabels(all_cot_steps)
    ax.set_xlabel("Latent steps used", fontsize=10)
    ax.set_ylabel("CoT steps  n_i  (dataset)", fontsize=10)
    ax.set_title("P(latent steps | CoT steps)  — row-normalised", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ns         = [b["n"] for b in by_cot_steps]
    avg_s      = [b["avg_latent_steps"] for b in by_cot_steps]
    bar_colors = plt.cm.Blues(np.array(ns) / max(ns))
    bars = ax2.bar(all_cot_steps, avg_s, color=bar_colors,
                   edgecolor="grey", linewidth=0.6, zorder=2)
    ax2.set_xticks(all_cot_steps)
    ax2.set_xlabel("CoT steps  n_i  (dataset)", fontsize=10)
    ax2.set_ylabel("Avg latent steps", fontsize=10)
    ax2.set_ylim(0, max(all_latent_steps) + 0.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.set_title("Avg latent steps per CoT-step bin", fontsize=9)
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.08,
                 f"n={n}", ha="center", va="bottom", fontsize=7)

    out = results_dir / "cot_steps_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[postprocess] wrote cot_steps_matrix.png", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(Path(sys.argv[1]))
