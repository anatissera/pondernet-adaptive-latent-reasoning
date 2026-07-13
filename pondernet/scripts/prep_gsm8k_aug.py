#!/usr/bin/env python3
"""Materialize the GSM8k-Aug training split to data/gsm8k_aug/train.jsonl.

The training data is gitignored, so a fresh clone has the code but no data. This
script downloads the full GSM8k-Aug training set (385,620 examples) from the HF hub
and writes it as line-delimited JSON in the exact place the training wrapper expects.

Run from the repo root or from pondernet/:
    python scripts/prep_gsm8k_aug.py            # -> ../data/gsm8k_aug/train.jsonl
    python scripts/prep_gsm8k_aug.py --out /abs/path/train.jsonl

Schema written per line: {"question": str, "cot": str, "answer": str}
(This is what train.py's `icot` loader reads; --data_name icot.)

Eval data: test.jsonl (1319 ex, this experiment's ONLY eval set) is TRACKED IN GIT
at data/gsm8k_aug/test.jsonl - a fresh clone already has it. It is the GSM8k-Aug
HF test split, so `--split test` can regenerate it if needed.

There is NO validation.jsonl for this experiment (do not fetch or add one): the
team's 500-ex validation split (used in exps 01-07) was sampled from train.jsonl,
and this experiment trains on the ENTIRE train set, so those 500 examples are
inside its training data. See docs/experiments/10-simcot-pondernet-fromscratch/
experiment.md -> "Evaluation protocol" for the full reasoning.
"""
import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    # Default output: <repo>/data/gsm8k_aug/train.jsonl, resolved relative to this file
    # so it works whether you launch from repo root or from pondernet/.
    here = os.path.dirname(os.path.abspath(__file__))            # .../pondernet/scripts
    repo_root = os.path.dirname(os.path.dirname(here))           # repo root

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="HF split to materialize (default: train)")
    ap.add_argument("--out", default=None,
                    help="output .jsonl path (default: <repo>/data/gsm8k_aug/<split>.jsonl)")
    ap.add_argument("--dataset", default="zen-E/GSM8k-Aug", help="HF dataset id")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(repo_root, "data", "gsm8k_aug", f"{args.split}.jsonl")

    print(f"[prep] downloading {args.dataset} ({args.split} split) ...")
    ds = load_dataset(args.dataset)[args.split]
    print("[prep] FIELDS:", list(ds.features.keys()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")
    print(f"[prep] wrote {len(ds)} examples to {args.out}")
    print("[prep] sample:", json.dumps(ds[0])[:300])


if __name__ == "__main__":
    main()
