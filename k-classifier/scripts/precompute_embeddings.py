#!/usr/bin/env python3
"""Precompute frozen sentence embeddings for the Option-A k* classifier.

Reads the `input` field of each example in the k sweep, encodes it once with a
frozen all-MiniLM-L6-v2 sentence encoder (384 dims, CPU), and caches
(example_id, embedding) pairs to a .npz file. The encoder is never fine-tuned;
this is a one-shot, reproducible feature extraction step.

The cache is reused as-is: if the output already exists and covers every id in
the sweep, the script is a no-op (pass --force to recompute from scratch).

Usage (from repo root):
    python3 k-classifier/scripts/precompute_embeddings.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OPTION_A_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP = OPTION_A_DIR / "results" / "k_sweep_train_full_codi.jsonl"
DEFAULT_OUTPUT = OPTION_A_DIR / "cache" / "embeddings_minilm_train_full.npz"
ENCODER_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def read_sweep_inputs(path: Path) -> tuple[list[str], list[str]]:
    """Return (ids, texts) from the sweep JSONL, preserving file order."""

    ids: list[str] = []
    texts: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["example_id"]))
            texts.append(str(row["input"]))
    if not ids:
        raise SystemExit(f"No examples found in {path}")
    return ids, texts


def load_cached_ids(path: Path) -> set[str]:
    """Return the set of ids already present in an existing cache, or empty."""

    if not path.exists():
        return set()
    with np.load(path, allow_pickle=True) as data:
        return set(str(i) for i in data["ids"].tolist())


def encode(texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts with the frozen encoder on CPU -> (N, 384) float32."""

    # Imported lazily so the script fails fast with a clear message if the
    # dependency is missing, and so `--help` works without the heavy import.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER_NAME, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP,
                        help="Sweep JSONL with example_id + input fields.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination .npz cache (ids + embeddings).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if a complete cache already exists.")
    args = parser.parse_args()

    ids, texts = read_sweep_inputs(args.sweep)
    print(f"Sweep: {len(ids)} examples from {args.sweep}")

    if not args.force:
        cached = load_cached_ids(args.output)
        if cached and set(ids).issubset(cached):
            print(f"Cache hit: {args.output} already covers all ids. "
                  f"Nothing to do (use --force to recompute).")
            return
        if cached:
            print(f"Cache exists but is incomplete "
                  f"({len(cached)}/{len(ids)} ids); recomputing in full.")

    print(f"Encoding with frozen {ENCODER_NAME} on CPU...")
    embeddings = encode(texts, args.batch_size)
    if embeddings.shape != (len(ids), EMBED_DIM):
        raise SystemExit(f"Unexpected embedding shape {embeddings.shape}, "
                         f"expected {(len(ids), EMBED_DIM)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=np.array(ids), embeddings=embeddings)
    print(f"Wrote {embeddings.shape[0]} embeddings "
          f"({embeddings.shape[1]} dims) -> {args.output}")


if __name__ == "__main__":
    main()
