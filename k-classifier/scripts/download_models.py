#!/usr/bin/env python3
"""Download the local model files required by Option-A loaders."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OPTION_A_DIR = SCRIPT_DIR.parent
DEFAULT_MODELS_DIR = OPTION_A_DIR / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    local_name: str


MODEL_SPECS = {
    "gpt2": ModelSpec(
        name="GPT-2 base",
        repo_id="openai-community/gpt2",
        local_name="gpt2",
    ),
    "coconut": ModelSpec(
        name="SIM-CoT GPT-2 Coconut",
        repo_id="internlm/SIM_COT-GPT2-Coconut",
        local_name="SIM_COT-GPT2-Coconut",
    ),
    "codi": ModelSpec(
        name="SIM-CoT GPT-2 CODI",
        repo_id="internlm/SIM_COT-GPT2-CODI",
        local_name="SIM_COT-GPT2-CODI",
    ),
}


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. "
            "Install k-classifier/requirements.txt first.",
            file=sys.stderr,
        )
        return 1

    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    selected = list(MODEL_SPECS) if args.model == "all" else [args.model]
    for key in selected:
        spec = MODEL_SPECS[key]
        destination = models_dir / spec.local_name
        print(f"Downloading {spec.name}")
        print(f"  repo: {spec.repo_id}")
        print(f"  dest: {destination}")
        snapshot_download(
            repo_id=spec.repo_id,
            revision=args.revision,
            local_dir=str(destination),
            force_download=args.force,
        )
        print(f"Done: {destination}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the GPT-2, Coconut, and CODI model files for Option-A."
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODEL_SPECS.keys()],
        default="all",
        help="Which model to download. Default: all.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory where model folders will be created.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision to download. Default: main.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force Hugging Face Hub to re-download files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
