#!/usr/bin/env python3
"""Synthetic controlled-density arithmetic dataset for Option-B.

Each instance is a K-step chain of <<expr=result>> blocks (same format as GSM8K-Aug).
Per-step DEPTH = number of binary ops in that block is the controlled difficulty knob.
Numbers stay positive bounded integers; division only when exact. See the design spec:
Option-B/docs/superpowers/specs/2026-06-23-synthetic-density-dataset-design.md
"""
import random
from typing import List, Tuple

MAX_VAL = 1000


def apply_op(v: int, rng: random.Random) -> Tuple[str, int]:
    """One left-associative op that keeps the running value a positive bounded int."""
    choices = ["+"]
    if v > 2:
        choices.append("-")
    if v * 2 <= MAX_VAL:
        choices.append("*")
    divisors = [d for d in (2, 3, 4, 5, 6) if v % d == 0 and v // d >= 1]
    if divisors:
        choices.append("/")
    op = rng.choice(choices)
    if op == "+":
        a = rng.randint(1, 12); return f"+{a}", v + a
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
