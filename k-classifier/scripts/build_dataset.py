#!/usr/bin/env python3
"""Build the (X, y) training dataset for the Option-A k* classifier.

Joins the precomputed sentence embeddings (id -> embedding) with the sweep
labels (id -> k_star) on the example id, then makes an internal stratified
train/val split (default 80/20). The GSM8K test set is NOT touched here: this
split is carved out of the training sweep only, so the test set stays reserved
for the final end-to-end evaluation.

Reports the k* distribution in each split.

Usage (from repo root):
    python3 k-classifier/scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


OPTION_A_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = OPTION_A_DIR / "cache" / "embeddings_minilm_train_full.npz"
DEFAULT_SWEEP = OPTION_A_DIR / "results" / "k_sweep_train_full_codi.jsonl"
DEFAULT_OUTPUT = OPTION_A_DIR / "cache" / "dataset_train_full.npz"


def load_labels(path: Path) -> dict[str, int]:
    """Map example_id -> k_star from the sweep JSONL."""

    labels: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[str(row["example_id"])] = int(row["k_star"])
    return labels


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Map example_id -> embedding vector from the .npz cache."""

    with np.load(path, allow_pickle=True) as data:
        ids = [str(i) for i in data["ids"].tolist()]
        embeddings = data["embeddings"]
    return {i: embeddings[idx] for idx, i in enumerate(ids)}


def distribution(y: np.ndarray) -> str:
    """Human-readable count + percentage of each label."""

    counts = Counter(int(v) for v in y)
    total = len(y)
    parts = [f"k={k}: {counts[k]} ({100 * counts[k] / total:.1f}%)"
             for k in sorted(counts)]
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Fraction held out for internal validation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = load_labels(args.sweep)
    embeddings = load_embeddings(args.embeddings)
    print(f"Labels: {len(labels)} | Embeddings: {len(embeddings)}")

    # Join on id: keep only examples present in both sources.
    common = sorted(set(labels) & set(embeddings))
    missing_emb = set(labels) - set(embeddings)
    if missing_emb:
        print(f"WARNING: {len(missing_emb)} labelled ids lack an embedding "
              f"(skipped). Re-run precompute_embeddings.py if unexpected.")
    if not common:
        raise SystemExit("No ids in common between labels and embeddings.")

    X = np.stack([embeddings[i] for i in common]).astype(np.float32)
    y = np.array([labels[i] for i in common], dtype=np.int64)
    ids = np.array(common)
    print(f"Joined dataset: X={X.shape}, y={y.shape}")
    print(f"Full k* distribution: {distribution(y)}")

    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, ids,
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=y,
    )

    print(f"\nTrain split (n={len(y_train)}): {distribution(y_train)}")
    print(f"Val   split (n={len(y_val)}): {distribution(y_val)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        X_train=X_train, y_train=y_train, ids_train=ids_train,
        X_val=X_val, y_val=y_val, ids_val=ids_val,
    )
    print(f"\nWrote dataset -> {args.output}")


if __name__ == "__main__":
    main()
