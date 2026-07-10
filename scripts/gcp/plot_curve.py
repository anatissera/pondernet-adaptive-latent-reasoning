#!/usr/bin/env python3
"""Plot the exp-10 from-scratch accuracy-vs-epoch curve from the eval bucket.

Fetches every results/epNN/summary.json the L4 worker has published and renders
two stacked panels (accuracy, avg latent steps) sharing the epoch axis. Rerun it
any time; it always reflects whatever has been evaluated so far.

    python3 scripts/gcp/plot_curve.py [out.png]
"""
import json, subprocess, sys, re

BUCKET = "gs://alr-exp10-ckpts-244544686610"
PROJECT = ["--project", "adaptative-latent-reasoning"]
BASELINE = 39.5   # SIM-CoT GPT-2 warm-start baseline (AGENTS.md)
K_MAX = 12

# palette: dataviz reference instance (categorical slot 1, validated)
BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def fetch():
    ls = subprocess.run(["gcloud", "storage", "ls", f"{BUCKET}/results/*/summary.json", *PROJECT],
                        capture_output=True, text=True).stdout.split()
    rows = []
    for url in ls:
        ep = int(re.search(r"/ep(\d+)/", url).group(1))
        d = json.loads(subprocess.run(["gcloud", "storage", "cat", url, *PROJECT],
                                      capture_output=True, text=True).stdout)
        if d.get("accuracy_pct") is not None:
            rows.append((ep, d["accuracy_pct"], d["avg_steps_used"]))
    return sorted(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = fetch()
    if not rows:
        sys.exit("no results published yet")
    eps, acc, steps = zip(*rows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.6]},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)

    ax1.plot(eps, acc, color=BLUE, lw=2, marker="o", ms=5,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax1.axhline(BASELINE, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax1.annotate(f"SIM-CoT warm-start baseline  {BASELINE}%", (0.99, BASELINE),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom",
                 fontsize=8.5, color=MUTED)
    ax1.annotate(f"{acc[-1]:.1f}%", (eps[-1], acc[-1]), textcoords="offset points",
                 xytext=(8, 2), fontsize=10, color=INK, fontweight="bold")
    ax1.set_ylim(0, 45)
    ax1.set_ylabel("accuracy on GSM8K test (%)", fontsize=10, color=INK)
    ax1.set_title(f"exp-10 from-scratch (GPT-2, Run-C recipe) — epoch {eps[-1]} of 40",
                  fontsize=12, color=INK, loc="left", pad=12)

    ax2.plot(eps, steps, color=BLUE, lw=2, marker="o", ms=5,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax2.axhline(K_MAX, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax2.annotate(f"K_max = {K_MAX}", (0.99, K_MAX), xycoords=("axes fraction", "data"),
                 ha="right", va="top", fontsize=8.5, color=MUTED)
    ax2.annotate(f"{steps[-1]:.2f}", (eps[-1], steps[-1]), textcoords="offset points",
                 xytext=(8, 2), fontsize=10, color=INK, fontweight="bold")
    ax2.set_ylim(0, 13)
    ax2.set_ylabel("avg latent steps", fontsize=10, color=INK)
    ax2.set_xlabel("epoch", fontsize=10, color=INK)
    ax2.set_xticks(range(0, 41, 5))
    ax2.set_xlim(0, 41)

    out = sys.argv[1] if len(sys.argv) > 1 else "results-curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(rows)} epochs)")
    for e, a, s in rows:
        print(f"  ep{e:02d}  acc={a:5.2f}%  steps={s:.3f}")


if __name__ == "__main__":
    main()
