#!/usr/bin/env python3
"""Smoke test real Option-A model loaders without writing experiment results."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
OPTION_A_DIR = SCRIPT_DIR.parent
SRC_DIR = OPTION_A_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(OPTION_A_DIR))

from model_runner import run_model  # noqa: E402


DEFAULT_PROMPT = "What is 2 + 2?"


class SmokeStageError(RuntimeError):
    """Wraps failures with a human-readable smoke-test stage."""

    def __init__(self, stage: str, backend: str, original: BaseException):
        self.stage = stage
        self.backend = backend
        self.original = original
        super().__init__(f"[{backend}] {stage} failed: {original}")


def main() -> int:
    args = parse_args()
    backend = args.backend
    loader = get_loader(backend)

    try:
        prompt = load_prompt(args.data)
    except Exception as exc:
        return report_failure(SmokeStageError("prompt load", backend, exc), args.verbose)

    try:
        model, tokenizer = load_backend(loader, backend)
    except SmokeStageError as exc:
        return report_failure(exc, args.verbose)

    generation_config = build_generation_config(args)
    try:
        prediction = run_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            k=1,
            generation_config=generation_config,
        )
    except Exception as exc:
        return report_failure(SmokeStageError("k=1 inference", backend, exc), args.verbose)

    print(f"Backend: {backend}")
    print(f"Loader: src.model_loaders:{loader.__name__}")
    print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(
        "Tokenizer class: "
        f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    )
    print(f"Prompt: {prompt}")
    print("Generated text:")
    print(prediction)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Option-A model loaders.")
    parser.add_argument("--backend", choices=["codi", "coconut"], required=True)
    parser.add_argument(
        "--data",
        default=str(OPTION_A_DIR / "data" / "examples.jsonl"),
        help="JSONL file used only to pick the first prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--c-thought", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_loader(backend: str) -> Callable[[], tuple[Any, Any]]:
    stage = "loader import"
    try:
        from src.model_loaders import load_codi, load_coconut
    except Exception as exc:
        raise SmokeStageError(stage, backend, exc) from exc

    return load_codi if backend == "codi" else load_coconut


def load_backend(loader: Callable[[], tuple[Any, Any]], backend: str) -> tuple[Any, Any]:
    try:
        loaded = loader()
    except Exception as exc:
        raise SmokeStageError("model load", backend, exc) from exc

    if not (isinstance(loaded, tuple) and len(loaded) == 2):
        raise SmokeStageError(
            "model load",
            backend,
            TypeError("loader must return a (model, tokenizer) tuple"),
        )
    model, tokenizer = loaded
    if model is None or tokenizer is None:
        raise SmokeStageError(
            "model load",
            backend,
            TypeError("loader returned None for model or tokenizer"),
        )
    return model, tokenizer


def load_prompt(data_path: str) -> str:
    path = Path(data_path)
    if not path.exists():
        return DEFAULT_PROMPT

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = str(record.get("input", "")).strip()
            if prompt:
                return prompt
    return DEFAULT_PROMPT


def build_generation_config(args: argparse.Namespace) -> dict:
    config = {
        "backend": args.backend,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "do_sample": False,
    }
    if args.backend == "coconut":
        config.update(
            {
                "c_thought": args.c_thought,
                "latent_token": "<|latent|>",
                "start_latent_token": "<|start-latent|>",
                "end_latent_token": "<|end-latent|>",
                "synced_gpus": False,
            }
        )
    else:
        config.update(
            {
                "remove_eos": True,
                "greedy": True,
                "top_k": 0,
            }
        )
    return config


def report_failure(error: SmokeStageError, verbose: bool) -> int:
    print(f"ERROR stage={error.stage} backend={error.backend}", file=sys.stderr)
    print(f"{error.original.__class__.__name__}: {error.original}", file=sys.stderr)
    if verbose:
        traceback.print_exception(error.original, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
