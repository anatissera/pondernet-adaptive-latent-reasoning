#!/usr/bin/env python3
"""
Cache frozen CODI backbone features for PonderNet halting-head training.

Runs one forward pass (no_grad) over the training set and saves:
  - hidden_states : (N, K, dim)  latent h_k after projection at each step
  - step_losses   : (N, K)       per-step answer CE loss

Since the backbone is frozen these tensors are deterministic given the input,
so they only need to be computed once.  Any classifier can then be trained on
the cached features without re-running transformer inference.

Run from pondernet/:
    python cache_features.py \
        --model_name_or_path gpt2 \
        --data_name icot \
        --max_train_samples 15000 \
        --num_latent 6 \
        --output_dir ./features \
        --pondernet True \
        --use_prj True --prj_dim 768 \
        --bf16 \
        --use_lora True --lora_r 128 --lora_alpha 32 --lora_init \
        --remove_eos True \
        --model_max_length 384 --max_token_num 700 \
        --use_decoder True \
        --per_device_train_batch_size 32   # only used to set batch size below
"""
import os
import logging
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from peft import LoraConfig, TaskType

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments, freeze_model

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Tokenization helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    tokenized = [
        tokenizer(text, return_tensors="pt", padding="longest",
                  max_length=256, truncation=True, return_attention_mask=False)
        for text in strings
    ]
    return {"input_ids": [t.input_ids[0] for t in tokenized]}


def _get_answer_token_position(tokens, answer_prompts, tokenizer):
    try:
        match = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
        return match + len(answer_prompts[0])
    except Exception:
        return len(tokens) - 1


def _preprocess(questions, cots, answers, tokenizer, bot_id, eot_id, training_args) -> Dict:
    sources_id  = _tokenize_fn(questions, tokenizer)["input_ids"]
    cot_id      = _tokenize_fn(cots,      tokenizer)["input_ids"]
    answers_id  = _tokenize_fn(answers,   tokenizer)["input_ids"]

    if not training_args.remove_eos:
        sources_id = [torch.tensor(x.tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
        cot_id     = [torch.tensor(x.tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
    answers_id = [torch.tensor(x.tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

    if cot_id[0][0] == tokenizer.bos_token_id:
        cot_id     = [x[1:] for x in cot_id]
        answers_id = [x[1:] for x in answers_id]

    ref_input_ids = [torch.cat([x, y, z]).long() for x, y, z in zip(sources_id, cot_id, answers_id)]
    ref_labels = []
    for seq, src in zip(ref_input_ids, sources_id):
        lbl = seq.clone()
        lbl[:len(src)] = IGNORE_INDEX
        ref_labels.append(lbl)

    sources_id = [torch.tensor(x.tolist() + [bot_id], dtype=torch.long) for x in sources_id]
    if training_args.remove_eos:
        answers_id = [torch.tensor([eot_id] + x.tolist(), dtype=torch.long) for x in answers_id]
    else:
        answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.tolist(), dtype=torch.long) for x in answers_id]

    answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")),
                      torch.tensor(tokenizer.encode("The next step result is:"))]
    if answer_prompts[0][0] == tokenizer.bos_token_id:
        answer_prompts = [p[1:] for p in answer_prompts]

    ref_answer_position  = [_get_answer_token_position(x, answer_prompts, tokenizer) for x in ref_input_ids]
    model_answer_position = [_get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

    return dict(
        encoder_input_ids=sources_id,
        decoder_input_ids=answers_id,
        ref_input_ids=ref_input_ids,
        labels=answers_id,
        ref_answer_position=ref_answer_position,
        model_answer_position=model_answer_position,
        ref_eos_position=[len(x) - 1 for x in ref_input_ids],
        model_eos_position=[len(x) - 1 for x in answers_id],
        ref_labels=ref_labels,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


class FeatureDataset(Dataset):
    def __init__(self, data_args, training_args, tokenizer, bot_id, eot_id):
        logging.warning("Loading and tokenising dataset …")
        if data_args.data_path:
            import json
            with open(data_args.data_path) as f:
                raw = json.load(f)
        else:
            raw = list(load_dataset("zen-E/GSM8k-Aug")["train"])
        if data_args.max_train_samples is not None:
            raw = raw[:data_args.max_train_samples]

        questions, cots, answers = [], [], []
        for ex in tqdm(raw):
            if 'cot' not in ex:
                ex['cot'] = ' '.join(ex['steps'])
            tok_len = len(tokenizer.encode(ex["question"] + ex["cot"] + ex["answer"]))
            if tok_len > training_args.max_token_num:
                continue
            cot = ex['cot'].split(" ")
            if not training_args.include_last_cot:
                cot = cot[:-1]
            answer = ex['answer'].split(' ')[-1].replace("####", "")
            if not answer[0].isdigit():
                continue
            questions.append(ex['question'])
            cots.append(" ".join(cot))
            answers.append(f"The answer is: {answer}")

        print(f"{len(questions)} examples after filtering")
        self.data = _preprocess(questions, cots, answers, tokenizer, bot_id, eot_id, training_args)
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.data["encoder_input_ids"])

    def __getitem__(self, i):
        return {k: self.data[k][i] for k in self.keys}


def collate_fn(instances, tokenizer):
    pad = tokenizer.pad_token_id

    enc = [inst["encoder_input_ids"] for inst in instances]
    enc = torch.nn.utils.rnn.pad_sequence(
        [x.flip(0) for x in enc], batch_first=True, padding_value=pad).flip(1)

    dec   = torch.nn.utils.rnn.pad_sequence([inst["decoder_input_ids"] for inst in instances], batch_first=True, padding_value=pad)
    ref   = torch.nn.utils.rnn.pad_sequence([inst["ref_input_ids"]     for inst in instances], batch_first=True, padding_value=pad)
    lbl   = torch.nn.utils.rnn.pad_sequence([inst["labels"]            for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
    rlbl  = torch.nn.utils.rnn.pad_sequence([inst["ref_labels"]        for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)

    return dict(
        encoder_input_ids=enc,
        decoder_input_ids=dec,
        ref_input_ids=ref,
        labels=lbl,
        encoder_attention_mask=enc.ne(pad),
        ref_attention_mask=ref.ne(pad),
        ref_answer_position=torch.tensor([inst["ref_answer_position"]   for inst in instances], dtype=torch.long),
        model_answer_position=torch.tensor([inst["model_answer_position"] for inst in instances], dtype=torch.long),
        ref_labels=rlbl,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    out_path = os.path.join(training_args.output_dir, "features.pt")

    # Build LoRA config (mirrors train.py)
    lora_config = None
    if model_args.use_lora:
        name = model_args.model_name_or_path.lower()
        if any(n in name for n in ["llama", "mistral", "falcon", "qwen"]):
            targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif "phi" in name:
            targets = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        else:
            targets = ["c_attn", "c_proj", "c_fc"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1, target_modules=targets, init_lora_weights=True,
        )

    model = CODI(model_args, training_args, lora_config)
    freeze_model(model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, token=model_args.token,
        cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length,
        padding_side="right", use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    dataset = FeatureDataset(data_args, training_args, tokenizer, model.bot_id, model.eot_id)
    loader  = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2,
        pin_memory=True,
    )

    all_hidden, all_losses = [], []
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=training_args.bf16):
                out = model(**batch)
            all_hidden.append(out["pondernet_hidden_states"].cpu())   # (B, K, dim)
            all_losses.append(out["pondernet_step_losses"].cpu())     # (B, K)

    hidden_states = torch.cat(all_hidden, dim=0)  # (N, K, dim)
    step_losses   = torch.cat(all_losses, dim=0)  # (N, K)

    torch.save({"hidden_states": hidden_states, "step_losses": step_losses}, out_path)
    print(f"Saved {hidden_states.shape[0]} examples → {out_path}")
    print(f"  hidden_states : {tuple(hidden_states.shape)}  dtype={hidden_states.dtype}")
    print(f"  step_losses   : {tuple(step_losses.shape)}    dtype={step_losses.dtype}")
    size_mb = (hidden_states.nbytes + step_losses.nbytes) / 1e6
    print(f"  Total size    : {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
