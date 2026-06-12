#!/usr/bin/env python3
"""Train a multi-output adaptive k classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OPTION_A_DIR = SCRIPT_DIR.parent
SRC_DIR = OPTION_A_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

np = None
torch = None
nn = None
AdamW = None
DataLoader = None
Subset = None
tqdm = None
AutoTokenizer = None
BertKClassifier = None
KClassifierDataset = None
select_k_threshold = None


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = KClassifierDataset(args.data, tokenizer, max_length=args.max_length)
    if args.fallback_k not in dataset.k_values:
        raise ValueError(
            f"--fallback-k={args.fallback_k} is not in dataset k_values={dataset.k_values}"
        )
    train_subset, val_subset = split_dataset(dataset, args.val_ratio, args.seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BertKClassifier(
        model_name=args.model_name,
        num_k=dataset.num_k,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        val_loss, val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            k_values=dataset.k_values,
            threshold=args.threshold,
            fallback_k=args.fallback_k,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"policy_accuracy={val_metrics['all']['policy_accuracy']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            torch.save(model.state_dict(), output_dir / "model.pt")

    tokenizer.save_pretrained(output_dir)
    config = {
        "model_name": args.model_name,
        "k_values": dataset.k_values,
        "threshold": args.threshold,
        "fallback_k": args.fallback_k,
        "max_length": args.max_length,
        "freeze_encoder": args.freeze_encoder,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "metrics.json",
        {
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_metrics,
            "history": history,
            "train_size": len(train_subset),
            "val_size": len(val_subset),
        },
    )
    print(f"saved best checkpoint to {output_dir / 'model.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive k classifier.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--fallback-k", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    """Import ML dependencies after argparse so --help remains lightweight."""

    global np
    global torch
    global nn
    global AdamW
    global DataLoader
    global Subset
    global tqdm
    global AutoTokenizer
    global BertKClassifier
    global KClassifierDataset
    global select_k_threshold

    import numpy as np_module
    import torch as torch_module
    from torch import nn as nn_module
    from torch.optim import AdamW as AdamW_class
    from torch.utils.data import DataLoader as DataLoader_class
    from torch.utils.data import Subset as Subset_class
    from tqdm import tqdm as tqdm_function
    from transformers import AutoTokenizer as AutoTokenizer_class

    from k_classifier import (
        BertKClassifier as BertKClassifier_class,
        KClassifierDataset as KClassifierDataset_class,
        select_k_threshold as select_k_threshold_function,
    )

    np = np_module
    torch = torch_module
    nn = nn_module
    AdamW = AdamW_class
    DataLoader = DataLoader_class
    Subset = Subset_class
    tqdm = tqdm_function
    AutoTokenizer = AutoTokenizer_class
    BertKClassifier = BertKClassifier_class
    KClassifierDataset = KClassifierDataset_class
    select_k_threshold = select_k_threshold_function


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_dataset(
    dataset: KClassifierDataset,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {val_ratio}")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_ratio)))
    train_size = len(indices) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small to create a non-empty train split")

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_one_epoch(
    model: BertKClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_examples


def evaluate_model(
    model: BertKClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    k_values: list[int],
    threshold: float,
    fallback_k: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            logits_batches.append(logits.cpu())
            label_batches.append(labels.cpu())
            metadata.extend(normalize_metadata(batch["metadata"]))

    if total_examples == 0:
        raise ValueError("Validation split is empty")

    logits_all = torch.cat(logits_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    probs_all = torch.sigmoid(logits_all)
    metrics = compute_metric_groups(
        probs=probs_all,
        labels=labels_all,
        metadata=metadata,
        k_values=k_values,
        threshold=threshold,
        fallback_k=fallback_k,
    )
    return total_loss / total_examples, metrics


def normalize_metadata(metadata_batch: Any) -> list[dict[str, Any]]:
    """Convert DataLoader-collated metadata into per-example dicts."""

    if isinstance(metadata_batch, list):
        return metadata_batch
    if not isinstance(metadata_batch, dict):
        raise TypeError(f"Unexpected metadata batch type: {type(metadata_batch).__name__}")

    batch_size = len(next(iter(metadata_batch.values())))
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        row: dict[str, Any] = {}
        for key, values in metadata_batch.items():
            if isinstance(values, torch.Tensor):
                row[key] = values[index].item()
            elif key == "k_values" and isinstance(values, list):
                row[key] = [
                    int(value[index].item()) if isinstance(value, torch.Tensor) else int(value)
                    for value in values
                ]
            else:
                row[key] = values[index]
        rows.append(row)
    return rows


def compute_metric_groups(
    probs: torch.Tensor,
    labels: torch.Tensor,
    metadata: list[dict[str, Any]],
    k_values: list[int],
    threshold: float,
    fallback_k: int,
) -> dict[str, Any]:
    masks = {
        "all": torch.ones(labels.shape[0], dtype=torch.bool),
        "mixed_only": torch.tensor([bool(row["mixed"]) for row in metadata]),
        "excluding_all_wrong": torch.tensor(
            [not bool(row["all_wrong"]) for row in metadata]
        ),
        "excluding_all_correct": torch.tensor(
            [not bool(row["all_correct"]) for row in metadata]
        ),
    }
    return {
        name: compute_metrics_for_mask(
            probs=probs,
            labels=labels,
            mask=mask,
            k_values=k_values,
            threshold=threshold,
            fallback_k=fallback_k,
        )
        for name, mask in masks.items()
    }


def compute_metrics_for_mask(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    k_values: list[int],
    threshold: float,
    fallback_k: int,
) -> dict[str, Any]:
    count = int(mask.sum().item())
    if count == 0:
        return {"count": 0}

    selected_probs = probs[mask]
    selected_labels = labels[mask]
    binary_predictions = (selected_probs >= 0.5).float()

    exact_vector_match = (
        binary_predictions.eq(selected_labels).all(dim=1).float().mean().item()
    )
    per_k_binary_accuracy = (
        binary_predictions.eq(selected_labels).float().mean(dim=0).tolist()
    )
    policy_hits: list[float] = []
    selected_k_values: list[int] = []
    for prob_row, label_row in zip(selected_probs.tolist(), selected_labels.tolist()):
        selected_k = select_k_threshold(
            prob_row,
            k_values,
            threshold=threshold,
            fallback="fixed",
            fallback_k=fallback_k,
        )
        selected_index = k_values.index(selected_k)
        policy_hits.append(float(label_row[selected_index] > 0.5))
        selected_k_values.append(selected_k)

    fixed_baselines = {}
    for fixed_k in (1, 6, 8):
        fixed_baselines[f"fixed_k_{fixed_k}"] = fixed_k_accuracy(
            labels=selected_labels,
            k_values=k_values,
            fixed_k=fixed_k,
        )

    oracle_accuracy = selected_labels.max(dim=1).values.float().mean().item()
    return {
        "count": count,
        "exact_vector_match": exact_vector_match,
        "per_k_binary_accuracy": dict(zip(map(str, k_values), per_k_binary_accuracy)),
        "policy_accuracy": float(np.mean(policy_hits)),
        "average_selected_k": float(np.mean(selected_k_values)),
        "fixed_baselines": fixed_baselines,
        "oracle_accuracy": oracle_accuracy,
    }


def fixed_k_accuracy(labels: torch.Tensor, k_values: list[int], fixed_k: int) -> float | None:
    if fixed_k not in k_values:
        return None
    index = k_values.index(fixed_k)
    return float(labels[:, index].float().mean().item())


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
