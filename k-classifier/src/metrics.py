"""Metrics for the Option-A k sweep."""

from __future__ import annotations

import re


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    """Lowercase, trim, and collapse duplicate whitespace."""

    return re.sub(r"\s+", " ", str(text).lower().strip())


def extract_number(text: str) -> str | None:
    """Return the last numeric value in text, normalized as a string."""

    matches = _NUMBER_RE.findall(str(text).replace(",", ""))
    if not matches:
        return None
    raw = matches[-1]
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return str(int(number))
    return str(number)


def evaluate(prediction: str, gold: str) -> float:
    """Exact match with light normalization and numeric answer support."""

    pred_number = extract_number(prediction)
    gold_number = extract_number(gold)
    if pred_number is not None and gold_number is not None:
        return 1.0 if pred_number == gold_number else 0.0

    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0
