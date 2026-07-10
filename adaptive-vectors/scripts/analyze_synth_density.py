#!/usr/bin/env python3
"""Aggregate synthetic-density eval logs into headline results.

For each test level L0..L3: build the fixed-c uniform curve, place adaptive/random/oracle
points against it (gap = acc - uniform_interp at matched budget), and compute the
correlation between MLP-allocated vectors and the ground-truth per-step depth.
"""
import json, os, re, statistics

RES = "../results/optionb-synth"
DATA = "../data/synth_density"
LEVELS = ["L0", "L1", "L2", "L3"]


def interp_uniform(cur, budget):
    """Linear interpolation of the (vectors, acc) fixed-c curve at `budget` vectors."""
    cur = sorted(cur)
    if budget <= cur[0][0]:
        return cur[0][1]
    if budget >= cur[-1][0]:
        return cur[-1][1]
    for (x0, y0), (x1, y1) in zip(cur, cur[1:]):
        if x0 <= budget <= x1:
            t = (budget - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return cur[-1][1]


def pearson(xs, ys):
    """Pearson correlation; returns 0.0 if either series is constant."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def level_variance(depths_rows):
    """Population variance of all per-step depths across instances."""
    flat = [d for row in depths_rows for d in row]
    return statistics.pvariance(flat)


def _parse_log(logpath):
    """Return (accuracy_pct, avg_total_vectors) from a test.py log, or (None, None)."""
    try:
        txt = open(logpath, encoding="utf-8", errors="ignore").read()
    except FileNotFoundError:
        return None, None
    ma = re.findall(r"GSM8K test accuracy:\s*([0-9.]+)%", txt)
    mv = re.findall(r"avg TOTAL vectors/instance:\s*([0-9.]+)", txt)
    acc = float(ma[-1]) if ma else None
    vec = float(mv[0]) if mv else None
    return acc, vec


def main():
    summary = {}
    for lvl in LEVELS:
        data_path = os.path.join(DATA, f"{lvl}.json")
        if not os.path.exists(data_path):
            print(f"[analyze] missing {data_path}, skipping {lvl}")
            continue
        depths_rows = [ex["depths"] for ex in json.load(open(data_path))]
        var = level_variance(depths_rows)

        # Fixed-c curve: (mean_vectors, accuracy) for c=1..4
        cur = []
        for c in (1, 2, 3, 4):
            log = os.path.join(RES, lvl, f"fixed_c{c}.log")
            acc, vec = _parse_log(log)
            if acc is not None and vec is not None:
                cur.append((vec, acc))

        # Other configs: adaptive eps, random, oracle
        rows = {}
        for tag in ["eps005", "eps015", "eps030", "random", "oracle"]:
            log = os.path.join(RES, lvl, f"{tag}.log")
            acc, vec = _parse_log(log)
            if acc is not None and vec is not None:
                gap = acc - interp_uniform(cur, vec) if cur else None
                rows[tag] = {"acc": acc, "vec": vec, "gap": gap}

        # Vectors-vs-depth correlation from the adaptive (eps015) detail JSON
        corr = None
        det = os.path.join(RES, lvl, "eps015", "gsm8k_optionb_detail.json")
        if os.path.exists(det):
            detail = json.load(open(det))
            vps = detail.get("vectors_per_step", [])
            xs, ys = [], []
            for n_row, d_row in zip(vps, depths_rows):
                for n, d in zip(n_row, d_row):
                    xs.append(float(n))
                    ys.append(float(d))
            if xs:
                corr = pearson(xs, ys)

        summary[lvl] = {
            "variance": round(var, 3),
            "fixed_curve": [(round(v, 2), round(a, 2)) for v, a in cur],
            "configs": {k: {kk: round(vv, 3) if vv is not None else None for kk, vv in v.items()} for k, v in rows.items()},
            "vec_depth_corr": round(corr, 4) if corr is not None else None,
        }

        print(f"\n== {lvl}  (depth var={var:.2f}) ==")
        if cur:
            print("  fixed c-curve:", [(round(v, 1), round(a, 1)) for v, a in cur])
        for tag, vals in rows.items():
            gap_str = f"{vals['gap']:+.2f}" if vals["gap"] is not None else "n/a"
            print(f"  {tag:8s}  acc={vals['acc']:.1f}%  vecs={vals['vec']:.1f}  gap_vs_uniform={gap_str}pt")
        print(f"  vectors-vs-depth corr: {corr:.3f}" if corr is not None else "  corr: n/a")

    out = os.path.join(RES, "summary.json")
    os.makedirs(RES, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n[analyze] summary → {out}")
    print("[analyze] headline: gap_vs_uniform of best adaptive eps should grow L0→L3 with depth variance")


if __name__ == "__main__":
    main()
