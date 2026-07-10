#!/usr/bin/env python3
"""Build a multi-output k-classifier dataset from k-sweep JSONL results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    k_values = list(range(args.k_min, args.k_max + 1))
    if not k_values:
        raise ValueError("--k-max must be >= --k-min")

    stats = build_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        k_values=k_values,
    )
    print_summary(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Option-A k-sweep JSONL rows into k-classifier labels."
    )
    parser.add_argument("--input", required=True, help="Input sweep JSONL path.")
    parser.add_argument("--output", required=True, help="Output classifier JSONL path.")
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=8)
    return parser.parse_args()


def build_dataset(input_path: Path, output_path: Path, k_values: list[int]) -> dict:
    """Convert sweep rows into binary multi-output classifier rows."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "processed": 0,
        "written": 0,
        "skipped": 0,
        "all_correct": 0,
        "all_wrong": 0,
        "mixed": 0,
        "oracle_correct": 0,
        "num_correct_sum": 0,
        "min_correct_k": Counter(),
    }

    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue

                stats["processed"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    warn(f"{input_path}:{line_number} invalid JSON: {exc}")
                    stats["skipped"] += 1
                    continue

                try:
                    output_row = convert_row(row, k_values)
                except ValueError as exc:
                    warn(f"{input_path}:{line_number} skipped: {exc}")
                    stats["skipped"] += 1
                    continue

                target.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                update_stats(stats, output_row)

    return stats


def convert_row(row: dict[str, Any], k_values: list[int]) -> dict[str, Any]:
    """Create one classifier example from one sweep row."""

    example_id = first_present(row, ("example_id", "id"))
    prompt = first_present(row, ("input", "prompt", "question"))
    answer = first_present(row, ("gold_answer", "answer", "gold", "target"))
    if example_id is None:
        raise ValueError("missing id field; expected example_id or id")
    if prompt is None:
        raise ValueError("missing prompt field; expected input, prompt, or question")
    if answer is None:
        raise ValueError("missing answer field; expected gold_answer, answer, or gold")

    scores = extract_scores(row, k_values)
    missing_k = [k for k in k_values if k not in scores]
    if missing_k:
        raise ValueError(f"missing scores for k values: {missing_k}")

    labels = [1 if scores[k] > 0 else 0 for k in k_values]
    correct_k = [k for k, label in zip(k_values, labels) if label == 1]
    all_correct = len(correct_k) == len(k_values)
    all_wrong = not correct_k
    mixed = not all_correct and not all_wrong

    return {
        "id": str(example_id),
        "prompt": str(prompt),
        "answer": str(answer),
        "k_values": k_values,
        "labels": labels,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "mixed": mixed,
        "min_correct_k": min(correct_k) if correct_k else None,
        "max_correct_k": max(correct_k) if correct_k else None,
        "num_correct_k": len(correct_k),
    }


def extract_scores(row: dict[str, Any], k_values: list[int]) -> dict[int, float]:
    """Extract numeric scores for requested k values from supported sweep shapes."""

    extracted: dict[int, float] = {}
    for k in k_values:
        value = find_score_for_k(row, k)
        if value is not None:
            extracted[k] = value
    return extracted


def find_score_for_k(row: dict[str, Any], k: int) -> float | None:
    nested_scores = row.get("scores")
    if isinstance(nested_scores, dict):
        score = get_mapping_value(nested_scores, k)
        if score is not None:
            return coerce_score(score, f"scores[{k}]")

    flat_key = f"score_k{k}"
    if flat_key in row:
        return coerce_score(row[flat_key], flat_key)

    for field_name in ("k_results", "per_k", "results"):
        score = extract_from_per_k_container(row.get(field_name), k)
        if score is not None:
            return score

    return None


def extract_from_per_k_container(container: Any, k: int) -> float | None:
    """Read scores from common per-k container variants."""

    if isinstance(container, dict):
        value = get_mapping_value(container, k)
        if value is not None:
            return coerce_score(value, f"per-k[{k}]")
        for item in container.values():
            if isinstance(item, dict) and int_or_none(item.get("k")) == k:
                return score_from_result_item(item, k)

    if isinstance(container, list):
        for item in container:
            if isinstance(item, dict) and int_or_none(item.get("k")) == k:
                return score_from_result_item(item, k)

    return None


def score_from_result_item(item: dict[str, Any], k: int) -> float | None:
    for key in ("score", "correct", "is_correct", "accuracy", "acc", "value"):
        if key in item:
            return coerce_score(item[key], f"per-k[{k}].{key}")
    return None


def coerce_score(value: Any, field_name: str) -> float:
    """Convert a supported score value into a float."""

    if isinstance(value, dict):
        score = score_from_result_item(value, int(value.get("k", -1)))
        if score is not None:
            return score
        raise ValueError(f"{field_name} has no score/correct field")

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "correct", "yes"}:
            return 1.0
        if normalized in {"false", "incorrect", "wrong", "no"}:
            return 0.0
        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not numeric: {value!r}") from exc

    raise ValueError(f"{field_name} has unsupported score type: {type(value).__name__}")


def get_mapping_value(mapping: dict[Any, Any], k: int) -> Any:
    for key in (str(k), k):
        if key in mapping:
            return mapping[key]
    return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["written"] += 1
    stats["all_correct"] += int(row["all_correct"])
    stats["all_wrong"] += int(row["all_wrong"])
    stats["mixed"] += int(row["mixed"])
    stats["oracle_correct"] += int(not row["all_wrong"])
    stats["num_correct_sum"] += int(row["num_correct_k"])
    if row["min_correct_k"] is not None:
        stats["min_correct_k"][row["min_correct_k"]] += 1


def print_summary(stats: dict[str, Any]) -> None:
    written = stats["written"]
    oracle_accuracy = stats["oracle_correct"] / written if written else 0.0
    avg_num_correct = stats["num_correct_sum"] / written if written else 0.0

    print("k-classifier dataset summary")
    print(f"total examples processed: {stats['processed']}")
    print(f"total examples written: {written}")
    print(f"total examples skipped: {stats['skipped']}")
    print(f"all_correct: {stats['all_correct']}")
    print(f"all_wrong: {stats['all_wrong']}")
    print(f"mixed: {stats['mixed']}")
    print(f"oracle accuracy: {oracle_accuracy:.6f}")
    print("distribution of min_correct_k:")
    if stats["min_correct_k"]:
        for k, count in sorted(stats["min_correct_k"].items()):
            print(f"  k={k}: {count}")
    else:
        print("  (none)")
    print(f"average num_correct_k: {avg_num_correct:.6f}")


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
