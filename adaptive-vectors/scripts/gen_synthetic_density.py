#!/usr/bin/env python3
"""Synthetic controlled-density arithmetic dataset for Option-B.

Each instance is a K-step chain of <<expr=result>> blocks (same format as GSM8K-Aug).
Per-step DEPTH = number of binary ops in that block is the controlled difficulty knob.
Numbers stay positive bounded integers; division only when exact. See the design spec:
Option-B/docs/superpowers/specs/2026-06-23-synthetic-density-dataset-design.md
"""
import random
from typing import List, Tuple
import argparse, json, os

MAX_VAL = 1000


def apply_op(v: int, rng: random.Random) -> Tuple[str, int]:
    """One left-associative op that keeps the running value a positive bounded int."""
    choices = []
    if v < MAX_VAL:
        choices.append("+")
    if v > 2:
        choices.append("-")
    if v * 2 <= MAX_VAL:
        choices.append("*")
    divisors = [d for d in (2, 3, 4, 5, 6) if v % d == 0 and v // d >= 1]
    if divisors:
        choices.append("/")
    op = rng.choice(choices)
    if op == "+":
        a = rng.randint(1, min(12, MAX_VAL - v)); return f"+{a}", v + a
    if op == "-":
        a = rng.randint(1, v - 1); return f"-{a}", v - a
    if op == "*":
        hi = min(4, MAX_VAL // v)
        a = rng.randint(2, max(2, hi)); return f"*{a}", v * a
    a = rng.choice(divisors); return f"/{a}", v // a


def gen_ops(v0: int, depth: int, rng: random.Random) -> Tuple[List[str], int]:
    v = v0; ops: List[str] = []
    for _ in range(depth):
        tok, v = apply_op(v, rng)
        ops.append(tok)
    return ops, v


def render(base: str, ops: List[str]) -> str:
    e = base
    for tok in ops:
        e = f"({e}{tok})"
    return e


def gen_instance(K: int, depths: List[int], rng: random.Random) -> dict:
    seed = rng.randint(2, 20)
    prev_val = seed
    comp_expr = str(seed)          # composed expression for the question
    cot_steps: List[str] = []
    for k in range(K):
        ops, new_val = gen_ops(prev_val, depths[k], rng)
        num_expr = render(str(prev_val), ops)   # cot uses the numeric prev result
        comp_expr = render(comp_expr, ops)      # question uses the composed expr
        cot_steps.append(f"<<{num_expr}={new_val}>>")
        prev_val = new_val
    return {
        "question": f"Calculate: {comp_expr}",
        "cot": " ".join(cot_steps),
        "answer": str(prev_val),
        "depths": list(depths),
    }


LEVELS = {
    "warmup": [(2, 1.0)],
    "train":  [(1, .25), (2, .25), (3, .25), (4, .25)],
    "L0":     [(2, .5), (3, .5)],
    "L1":     [(1, .1), (2, .4), (3, .4), (4, .1)],
    "L2":     [(1, .25), (2, .25), (3, .25), (4, .25)],
    "L3":     [(1, .5), (4, .5)],
}


def sample_depths(level: str, K: int, rng: random.Random) -> List[int]:
    pairs = LEVELS[level]
    vals = [d for d, _ in pairs]
    wts = [w for _, w in pairs]
    return [rng.choices(vals, weights=wts, k=1)[0] for _ in range(K)]


def build_split(level: str, n: int, K: int, seed: int, exclude: set):
    rng = random.Random(seed)
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < n * 100:
        guard += 1
        depths = sample_depths(level, K, rng)
        inst = gen_instance(K, depths, rng)
        q = inst["question"]
        if q in seen or q in exclude:
            continue
        seen.add(q); rows.append(inst)
    if len(rows) < n:
        raise RuntimeError(f"build_split({level!r}) produced only {len(rows)}/{n} unique rows (collision-bound); raise the guard or reduce n")
    return rows, seen


def _write(path: str, rows: List[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../data/synth_density")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_warmup", type=int, default=5000)
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--n_test", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    warm, qw = build_split("warmup", a.n_warmup, a.K, a.seed, set())
    train, qt = build_split("train", a.n_train, a.K, a.seed + 1, qw)
    used = qw | qt
    _write(os.path.join(a.out_dir, "warmup.jsonl"), warm)
    _write(os.path.join(a.out_dir, "train.jsonl"), train)
    for i, lvl in enumerate(["L0", "L1", "L2", "L3"]):
        rows, q = build_split(lvl, a.n_test, a.K, a.seed + 10 + i, used)
        used |= q
        _write(os.path.join(a.out_dir, f"{lvl}.jsonl"), rows)
        print(f"{lvl}: {len(rows)} rows")
    print(f"warmup={len(warm)} train={len(train)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
