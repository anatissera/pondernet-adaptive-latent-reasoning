#!/usr/bin/env python3
"""Train simple classical classifiers to predict k* from sentence embeddings.

First-version baseline, deliberately simple (professor's request: "start
simple"). Trains logistic regression and random forest (both scikit-learn) to
predict k* from the frozen MiniLM embedding, handling the heavy class imbalance
with class_weight='balanced'. A trivial "always predict the majority class"
baseline (k=1) is included as the mandatory reference point.

The question this answers: does the classifier learn anything beyond "always
predict k=1"? We report accuracy AND macro-F1 (accuracy alone is misleading at
~66% k=1 base rate), the confusion matrix, and the explicit comparison against
the majority-class baseline.

No optimization here yet: no MLP, no target transforms. Just the simple classics.

Usage (from repo root):
    python3 k-classifier/scripts/train_classifier.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OPTION_A_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = OPTION_A_DIR / "cache" / "dataset_train_full.npz"
DEFAULT_OUTPUT = OPTION_A_DIR / "results" / "classifier_results.json"


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray,
             labels: list[int]) -> dict:
    """Compute accuracy, macro-F1, and confusion matrix for one model."""

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", labels=labels,
                   zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n=== {name} ===")
    print(f"accuracy : {acc:.4f}")
    print(f"f1_macro : {f1m:.4f}")
    print("confusion matrix (rows=true, cols=pred), label order "
          f"{labels}:")
    print(cm)
    print(classification_report(y_true, y_pred, labels=labels,
                                zero_division=0))
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=True) as data:
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

    labels = sorted(set(int(v) for v in np.concatenate([y_train, y_val])))
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | labels: {labels}")

    results: dict[str, dict] = {}

    # --- Mandatory reference: always predict the training majority class. ---
    majority = Counter(int(v) for v in y_train).most_common(1)[0][0]
    y_majority = np.full_like(y_val, fill_value=majority)
    results["majority_baseline"] = evaluate(
        f"Majority baseline (always k={majority})", y_val, y_majority, labels)

    # --- Logistic regression (scaled features, balanced classes). ---
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
        ),
    )
    logreg.fit(X_train, y_train)
    results["logistic_regression"] = evaluate(
        "Logistic regression (class_weight=balanced)",
        y_val, logreg.predict(X_val), labels)

    # --- Random forest (balanced classes). ---
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(
        "Random forest (class_weight=balanced)",
        y_val, rf.predict(X_val), labels)

    # --- Explicit comparison against the majority baseline. ---
    base = results["majority_baseline"]
    print("\n=== Comparison vs majority baseline ===")
    print(f"{'model':<24}{'accuracy':>10}{'f1_macro':>10}"
          f"{'d_acc':>9}{'d_f1':>9}")
    for name in ("majority_baseline", "logistic_regression", "random_forest"):
        r = results[name]
        print(f"{name:<24}{r['accuracy']:>10.4f}{r['f1_macro']:>10.4f}"
              f"{r['accuracy'] - base['accuracy']:>+9.4f}"
              f"{r['f1_macro'] - base['f1_macro']:>+9.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "majority_class": int(majority),
        "labels": labels,
        "results": results,
    }
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote metrics -> {args.output}")


if __name__ == "__main__":
    main()
