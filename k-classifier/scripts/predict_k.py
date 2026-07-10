#!/usr/bin/env python3
"""Predict adaptive k values with a trained k classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OPTION_A_DIR = SCRIPT_DIR.parent
SRC_DIR = OPTION_A_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

torch = None
AutoTokenizer = None
BertKClassifier = None
select_k_argmax = None
select_k_threshold = None


def main() -> None:
    args = parse_args()
    if bool(args.prompt) == bool(args.input_jsonl):
        raise ValueError("Provide exactly one of --prompt or --input-jsonl")

    load_runtime_dependencies()
    checkpoint = Path(args.checkpoint)
    config = load_json(checkpoint / "config.json")
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = load_model(checkpoint, config, device)

    if args.prompt:
        result = predict_one(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            example_id=None,
            config=config,
            device=device,
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        if not args.output_jsonl:
            raise ValueError("--output-jsonl is required with --input-jsonl")
        predict_jsonl(
            model=model,
            tokenizer=tokenizer,
            input_path=Path(args.input_jsonl),
            output_path=Path(args.output_jsonl),
            config=config,
            device=device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict adaptive k for prompts.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    """Import ML dependencies after argparse so --help remains lightweight."""

    global torch
    global AutoTokenizer
    global BertKClassifier
    global select_k_argmax
    global select_k_threshold

    import torch as torch_module
    from transformers import AutoTokenizer as AutoTokenizer_class

    from k_classifier import (
        BertKClassifier as BertKClassifier_class,
        select_k_argmax as select_k_argmax_function,
        select_k_threshold as select_k_threshold_function,
    )

    torch = torch_module
    AutoTokenizer = AutoTokenizer_class
    BertKClassifier = BertKClassifier_class
    select_k_argmax = select_k_argmax_function
    select_k_threshold = select_k_threshold_function


def load_model(checkpoint: Path, config: dict[str, Any], device: torch.device) -> BertKClassifier:
    model = BertKClassifier(
        model_name=config["model_name"],
        num_k=len(config["k_values"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config.get("dropout", 0.1)),
        freeze_encoder=bool(config.get("freeze_encoder", False)),
    )
    state_dict = torch.load(checkpoint / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_jsonl(
    model: BertKClassifier,
    tokenizer: Any,
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("input") or row.get("question")
                if prompt is None:
                    raise ValueError(
                        f"{input_path}:{line_number} missing prompt/input/question"
                    )
                example_id = row.get("id") or row.get("example_id")
                result = predict_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=str(prompt),
                    example_id=example_id,
                    config=config,
                    device=device,
                )
                target.write(json.dumps(result, ensure_ascii=False) + "\n")


def predict_one(
    model: BertKClassifier,
    tokenizer: Any,
    prompt: str,
    example_id: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        max_length=int(config["max_length"]),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        probs = torch.sigmoid(logits)[0].cpu().tolist()

    k_values = [int(k) for k in config["k_values"]]
    selected_argmax = select_k_argmax(probs, k_values)
    selected_threshold = select_k_threshold(
        probs,
        k_values,
        threshold=float(config["threshold"]),
        fallback="fixed",
        fallback_k=int(config["fallback_k"]),
    )
    return {
        "id": example_id,
        "prompt": prompt,
        "probs": probs,
        "k_values": k_values,
        "selected_k_argmax": selected_argmax,
        "selected_k_threshold": selected_threshold,
    }


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
