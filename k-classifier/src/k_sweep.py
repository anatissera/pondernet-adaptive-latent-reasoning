"""Shared k-sweep pipeline for Coconut/SIM-CoT and CODI backends."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from metrics import evaluate
from model_runner import run_model


def load_examples(path: str | Path, n_examples: int | None = None) -> list[dict]:
    """Load examples from JSONL records with id, input, and gold fields."""

    examples: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            missing = {"id", "input", "gold"} - set(example)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing required fields: {sorted(missing)}"
                )
            examples.append(example)
            if n_examples is not None and len(examples) >= n_examples:
                break
    return examples


def evaluate_all_k(
    example: dict,
    model: Any,
    tokenizer: Any,
    k_max: int,
    generation_config: dict | None = None,
) -> dict:
    """Run all k values for one example and compute k_star."""

    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")

    generation_config = dict(generation_config or {})
    prompt = example["input"]
    gold = example["gold"]
    predictions: dict[str, str] = {}
    scores: dict[str, float] = {}

    for k in range(1, k_max + 1):
        prediction = run_model(model, tokenizer, prompt, k, generation_config)
        score = evaluate(prediction, gold)
        predictions[str(k)] = prediction
        scores[str(k)] = score

    k_star = compute_k_star(scores)
    return format_result_row(
        example_id=example["id"],
        prompt=prompt,
        gold=gold,
        predictions=predictions,
        scores=scores,
        k_star=k_star,
    )


def compute_k_star(scores: dict[str, float]) -> int:
    """Return the smallest k that reaches the maximum observed score."""

    if not scores:
        raise ValueError("scores must not be empty")
    best_score = max(scores.values())
    return min(int(k) for k, score in scores.items() if score == best_score)


def format_result_row(
    example_id: str,
    prompt: str,
    gold: str,
    predictions: dict[str, str],
    scores: dict[str, float],
    k_star: int,
) -> dict:
    """Create the nested and flat output row expected by downstream analysis."""

    row: dict[str, Any] = {
        "example_id": example_id,
        "input": prompt,
        "gold_answer": gold,
        "predictions": predictions,
        "scores": scores,
        "k_star": k_star,
    }
    for k in sorted((int(key) for key in predictions), key=int):
        row[f"prediction_k{k}"] = predictions[str(k)]
    for k in sorted((int(key) for key in scores), key=int):
        row[f"score_k{k}"] = scores[str(k)]
    return row


def run_k_sweep(
    examples: Iterable[dict],
    model: Any,
    tokenizer: Any,
    k_max: int,
    generation_config: dict,
) -> list[dict]:
    """Evaluate all examples over k=1..k_max."""

    return [
        evaluate_all_k(example, model, tokenizer, k_max, generation_config)
        for example in examples
    ]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    """Write result rows as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_results(rows: list[dict], k_max: int) -> dict:
    """Compute aggregate k-sweep diagnostics."""

    if not rows:
        return {
            "accuracy_by_k": {},
            "average_score_by_k": {},
            "distribution_of_k_star": {},
            "percentage_of_examples_where_outputs_change_across_k": 0.0,
        }

    accuracy_by_k: dict[str, float] = {}
    average_score_by_k: dict[str, float] = {}
    for k in range(1, k_max + 1):
        key = str(k)
        values = [float(row["scores"][key]) for row in rows if key in row["scores"]]
        average = sum(values) / len(values) if values else 0.0
        average_score_by_k[key] = average
        accuracy_by_k[key] = average

    k_star_counts = Counter(int(row["k_star"]) for row in rows)
    changed_count = sum(1 for row in rows if outputs_changed(row["predictions"]))

    return {
        "accuracy_by_k": accuracy_by_k,
        "average_score_by_k": average_score_by_k,
        "distribution_of_k_star": dict(sorted(k_star_counts.items())),
        "percentage_of_examples_where_outputs_change_across_k": (
            100.0 * changed_count / len(rows)
        ),
    }


def outputs_changed(predictions: dict[str, str]) -> bool:
    """Return True if at least two k values produced different outputs."""

    normalized = {str(value).strip() for value in predictions.values()}
    return len(normalized) > 1


def write_summary_csv(summary: dict, path: str | Path) -> None:
    """Write summary diagnostics in a simple long CSV format."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "k", "value"])
        writer.writeheader()

        for metric in ("accuracy_by_k", "average_score_by_k"):
            for k, value in summary[metric].items():
                writer.writerow({"metric": metric, "k": k, "value": value})

        for k, count in summary["distribution_of_k_star"].items():
            writer.writerow(
                {"metric": "distribution_of_k_star", "k": k, "value": count}
            )

        writer.writerow(
            {
                "metric": "percentage_of_examples_where_outputs_change_across_k",
                "k": "",
                "value": summary[
                    "percentage_of_examples_where_outputs_change_across_k"
                ],
            }
        )


def print_summary(summary: dict, k_max: int) -> None:
    """Print the required console diagnostics."""

    print("Accuracy by k:")
    for k in range(1, k_max + 1):
        value = summary["accuracy_by_k"].get(str(k), 0.0)
        print(f"k={k}: {value:.4f}")

    print("k_star distribution:")
    for k in range(1, k_max + 1):
        value = summary["distribution_of_k_star"].get(k, 0)
        print(f"k={k}: {value}")

    changed = summary["percentage_of_examples_where_outputs_change_across_k"]
    print(f"Outputs changed across k: {changed:.2f}%")
