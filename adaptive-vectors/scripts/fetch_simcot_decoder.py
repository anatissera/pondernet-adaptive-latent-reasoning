"""Download internlm/SIM_COT-GPT2-CODI and extract its `decoder.*` weights
into a standalone GPT-2 checkpoint loadable via --decoder_path.

Usage:
    python scripts/fetch_simcot_decoder.py [--out ../models/pretrained/simcot-gpt2-decoder]
"""
import argparse
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import GPT2Config, GPT2LMHeadModel

SOURCE_REPO = "internlm/SIM_COT-GPT2-CODI"
PREFIX = "decoder."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../models/pretrained/simcot-gpt2-decoder")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    print(f"Downloading {SOURCE_REPO}...")
    repo_dir = snapshot_download(repo_id=SOURCE_REPO, cache_dir=args.cache_dir)

    sft_path = os.path.join(repo_dir, "model-00001-of-00001.safetensors")
    decoder_state = {}
    with safe_open(sft_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(PREFIX):
                decoder_state[key[len(PREFIX):]] = f.get_tensor(key)

    print(f"Extracted {len(decoder_state)} decoder tensors")

    vocab_size, n_embd = decoder_state["transformer.wte.weight"].shape
    n_positions = decoder_state["transformer.wpe.weight"].shape[0]
    n_layer = max(
        int(k.split(".")[2]) for k in decoder_state if k.startswith("transformer.h.")
    ) + 1
    n_head = 12  # standard gpt2 (124M); n_embd=768 confirms this size

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    print(f"Inferred config: vocab_size={vocab_size}, n_embd={n_embd}, "
          f"n_layer={n_layer}, n_positions={n_positions}")

    model = GPT2LMHeadModel(config)
    missing, unexpected = model.load_state_dict(decoder_state, strict=False)
    print(f"missing keys: {missing}")
    print(f"unexpected keys: {unexpected}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    print(f"Saved standalone decoder checkpoint to {args.out}")


if __name__ == "__main__":
    main()
