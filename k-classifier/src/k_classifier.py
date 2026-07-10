"""Dataset, model, and policy helpers for adaptive k classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel


class KClassifierDataset(Dataset):
    """JSONL dataset for multi-output k correctness labels."""

    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 256) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rows = self._load_rows(self.path)
        if not self.rows:
            raise ValueError(f"No examples found in {self.path}")

        first_k_values = self.rows[0].get("k_values")
        if not isinstance(first_k_values, list) or not first_k_values:
            raise ValueError(f"{self.path} rows must include non-empty k_values")
        self.k_values = [int(k) for k in first_k_values]
        self.num_k = len(self.k_values)

        for index, row in enumerate(self.rows):
            labels = row.get("labels")
            k_values = row.get("k_values")
            if not isinstance(labels, list) or len(labels) != self.num_k:
                raise ValueError(
                    f"{self.path} row {index + 1} labels must have length {self.num_k}"
                )
            if [int(k) for k in k_values] != self.k_values:
                raise ValueError(
                    f"{self.path} row {index + 1} k_values differ from first row"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["labels"], dtype=torch.float),
            "metadata": {
                "id": row.get("id"),
                "k_values": row.get("k_values"),
                "all_correct": bool(row.get("all_correct", False)),
                "all_wrong": bool(row.get("all_wrong", False)),
                "mixed": bool(row.get("mixed", False)),
            },
        }
        return item

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = {"id", "prompt", "k_values", "labels"} - set(row)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} missing fields: {sorted(missing)}"
                    )
                rows.append(row)
        return rows


class BertKClassifier(nn.Module):
    """BERT-style encoder with one correctness logit per k."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_k: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_k),
        )

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def select_k_argmax(probs: Sequence[float], k_values: Sequence[int]) -> int:
    """Select the k with the highest predicted probability."""

    probs_list, k_list = validate_policy_inputs(probs, k_values)
    best_index = max(range(len(probs_list)), key=lambda index: probs_list[index])
    return k_list[best_index]


def select_k_threshold(
    probs: Sequence[float],
    k_values: Sequence[int],
    threshold: float = 0.7,
    fallback: str = "argmax",
    fallback_k: int | None = None,
) -> int:
    """Select the smallest k above threshold, then apply a fallback policy."""

    probs_list, k_list = validate_policy_inputs(probs, k_values)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    for probability, k in zip(probs_list, k_list):
        if probability >= threshold:
            return k

    if fallback == "argmax":
        return select_k_argmax(probs_list, k_list)
    if fallback == "fixed":
        if fallback_k is None:
            raise ValueError("fallback_k is required when fallback='fixed'")
        if fallback_k not in k_list:
            raise ValueError(f"fallback_k={fallback_k} is not in k_values={k_list}")
        return int(fallback_k)

    raise ValueError("fallback must be one of {'argmax', 'fixed'}")


def validate_policy_inputs(
    probs: Sequence[float],
    k_values: Sequence[int],
) -> tuple[list[float], list[int]]:
    probs_list = [float(value) for value in probs]
    k_list = [int(value) for value in k_values]
    if not probs_list:
        raise ValueError("probs must not be empty")
    if len(probs_list) != len(k_list):
        raise ValueError(
            f"probs and k_values must have the same length, got "
            f"{len(probs_list)} and {len(k_list)}"
        )
    return probs_list, k_list
