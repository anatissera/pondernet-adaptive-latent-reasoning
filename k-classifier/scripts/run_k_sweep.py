#!/usr/bin/env python3
"""CLI for the Option-A multi-backend k sweep."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OPTION_A_DIR = SCRIPT_DIR.parent
SRC_DIR = OPTION_A_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(OPTION_A_DIR))

from k_sweep import (  # noqa: E402
    load_examples,
    print_summary,
    run_k_sweep,
    summarize_results,
    write_jsonl,
    write_summary_csv,
)


def main() -> None:
    args = parse_args()
    generation_config = build_generation_config(args)

    model, tokenizer = load_model_and_tokenizer(args.model_loader)
    examples = load_examples(args.data, args.n_examples)
    rows = run_k_sweep(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        k_max=args.k_max,
        generation_config=generation_config,
    )

    write_jsonl(rows, args.output)
    summary = summarize_results(rows, args.k_max)
    summary_path = Path(args.summary) if args.summary else default_summary_path(args.output)
    write_summary_csv(summary, summary_path)
    print_summary(summary, args.k_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Option-A k sweep.")
    parser.add_argument("--backend", choices=["coconut", "codi"], required=True)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--data", default=str(OPTION_A_DIR / "data" / "examples.jsonl"))
    parser.add_argument(
        "--output",
        default=str(OPTION_A_DIR / "results" / "k_sweep_results.jsonl"),
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path for summary CSV. Defaults next to --output.",
    )
    parser.add_argument(
        "--model-loader",
        required=True,
        help=(
            "Dotted callable path that returns (model, tokenizer), e.g. "
            "my_package.loaders:load_coconut"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--c-thought", type=int, default=1)
    parser.add_argument("--latent-token", default="<|latent|>")
    parser.add_argument("--start-latent-token", default="<|start-latent|>")
    parser.add_argument("--end-latent-token", default="<|end-latent|>")
    parser.add_argument("--synced-gpus", action="store_true")

    parser.add_argument("--remove-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-prj", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def build_generation_config(args: argparse.Namespace) -> dict:
    config = {
        "backend": args.backend,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": False,
    }

    if args.backend == "coconut":
        config.update(
            {
                "c_thought": args.c_thought,
                "latent_token": args.latent_token,
                "start_latent_token": args.start_latent_token,
                "end_latent_token": args.end_latent_token,
                "synced_gpus": args.synced_gpus,
            }
        )
    elif args.backend == "codi":
        config.update(
            {
                "remove_eos": args.remove_eos,
                "greedy": args.greedy,
                "top_k": args.top_k,
            }
        )
        if args.use_prj is not None:
            config["use_prj"] = args.use_prj

    return config


def load_model_and_tokenizer(loader_path: str) -> tuple[Any, Any]:
    loader = import_dotted_callable(loader_path)
    loaded = loader()

    if isinstance(loaded, tuple) and len(loaded) == 2:
        return loaded
    if hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
        return loaded.model, loaded.tokenizer

    raise TypeError(
        "--model-loader must return (model, tokenizer) or an object with "
        ".model and .tokenizer"
    )


def import_dotted_callable(path: str):
    if ":" in path:
        module_name, function_name = path.split(":", 1)
    else:
        module_name, function_name = path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    if not callable(loader):
        raise TypeError(f"{path} is not callable")
    return loader


def default_summary_path(output: str) -> Path:
    return Path(output).with_name("k_sweep_summary.csv")


if __name__ == "__main__":
    main()
