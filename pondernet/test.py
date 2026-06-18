#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Evaluation entrypoint for the PonderNet adaptive-halting latent-CoT model (CODI backbone)."""

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def save_jsonl_line(filepath, data):
    """Append a single dict as a JSON line to a JSONL file.

    Args:
        filepath (str): Path to the target JSONL file.
        data (dict): Data to write; must be JSON-serializable.
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    with open(filepath, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")

def read_json(file_path):
    """Read a JSON or JSONL file and return the parsed object (list for JSONL)."""
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in file if line.strip()]
        return json.load(file)


def write_json(data, file_path):
    """Write a Python object to file_path as pretty-printed JSON."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def _slice_past_key_values(past_key_values, idx):
    """Return a copy of `past_key_values` keeping only the batch rows in `idx`.

    `idx` is a 1-D LongTensor of batch indices (on the cache's device). Handles both
    the legacy tuple-of-tuples layout and transformers' Cache objects (via the legacy
    round-trip). Each kept tensor is `index_select`-ed on the batch dim, so the result
    is an independent copy — decoding from it does not mutate the caller's cache.

    Used by PonderNet eval to decode each example's answer from the KV prefix at *its
    own* halt step instead of the batch's shared termination prefix.
    """
    if past_key_values is None:
        return None
    # transformers Cache object (e.g. DynamicCache): round-trip through legacy form
    if hasattr(past_key_values, "to_legacy_cache") and hasattr(type(past_key_values), "from_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        sliced = tuple(tuple(t.index_select(0, idx) for t in layer) for layer in legacy)
        return type(past_key_values).from_legacy_cache(sliced)
    # legacy tuple-of-tuples
    return tuple(tuple(t.index_select(0, idx) for t in layer) for layer in past_key_values)

def evaluation(model_args, data_args, training_args):
    import os
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError
    model = CODI(model_args, training_args, lora_config)
    #if "llama" in model_args.model_name_or_path:
    #    model.codi.resize_token_embeddings(128261)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    
    # new_state_dict = { k.replace("coconut", "codi"): v for k, v in state_dict.items() }
    # torch.save(new_state_dict, "/scratch/prj/inf_multimodal_qa/scratch_tmp/transfer/pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    
    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)
    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm-hard" == data_args.data_name:
        if not data_args.data_path:
            raise ValueError("gsm-hard requires --data_path pointing to gsm8k_hard_format.json")
        test_set = read_json(data_args.data_path)
    elif "multi-arith" == data_args.data_name:
        if not data_args.data_path:
            raise ValueError("multi-arith requires --data_path pointing to multiarith_format.json")
        test_set = read_json(data_args.data_path)
    elif "svamp" == data_args.data_name:
        if not data_args.data_path:
            raise ValueError("svamp requires --data_path pointing to svamp_format.json")
        test_set = read_json(data_args.data_path)
    elif "commonsense" == data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset['validation']
    elif "gsm8k" == data_args.data_name:
        if data_args.data_path:
            test_set = read_json(data_args.data_path)
        else:
            test_set = list(load_dataset("gsm8k", "main")["test"])
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            if example == "True":
                ans = True
            else:
                ans = False
            answer.append(ans)
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split('####')[-1]
        else:
            ans = example
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature":0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    def generate_answers(past_key_values, attn):
        """Autoregressively decode answers for `attn.size(0)` sequences from `past_key_values`.

        `attn` is the attention mask (B, L) covering the cache so far (1 = real token,
        0 = left-pad); it is extended by the number of tokens appended each step so the
        model keeps masking padded positions (with batch>1 the question is left-padded,
        and an unmasked cache would let later steps attend to pad keys). Returns a list
        of B token-id lists. The cache and mask are extended locally, so the caller's
        copies are untouched.
        """
        bs = attn.size(0)
        if training_args.remove_eos:
            eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
        else:
            eot_emb = model.get_embd(model.codi, model.model_name)(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
        output = eot_emb.expand(bs, -1, -1)

        finished = torch.zeros(bs, dtype=torch.bool, device="cuda")  # Track EOS for each sequence
        pred_tokens = [[] for _ in range(bs)]
        for _ in range(gen_kwargs["max_new_tokens"]):
            # position ids for the token(s) appended this step = running count of real
            # tokens (attn.sum) .. +n_new-1, then extend the mask to cover them.
            n_new = output.size(1)
            base = attn.sum(dim=1, keepdim=True).long()
            pos_ids = base + torch.arange(n_new, device=attn.device).unsqueeze(0)
            attn = torch.cat([attn, torch.ones((bs, n_new), dtype=attn.dtype, device=attn.device)], dim=1)
            out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=attn,
                    position_ids=pos_ids,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

            # implement the sampling process
            if training_args.greedy:
                next_token_ids = torch.argmax(logits, dim=-1)
            else:
                logits /= gen_kwargs["temperature"]
                if gen_kwargs["top_k"] > 1:
                    top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    logits[logits < min_top_k_value] = -float("inf")

                if gen_kwargs["top_p"] < 1.0:
                    sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                    if sorted_indices_to_remove.any():
                        sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                        sorted_indices_to_remove[:, 0] = False

                    for b in range(logits.size(0)):
                        logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Handle EOS for each sequence
            for b in range(bs):
                if not finished[b]:
                    pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True

            # Break if all sequences have finished
            if finished.all():
                break

            output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)
        return pred_tokens

    ans_pred_list = []
    ans_pred_list_accu_at_n_passes = []
    attention_map_weights = []
    attention_to_latents_against_len_sum = []
    attention_to_latents_against_len_count = []
    gating_probs_sums = None
    len_cot = []
    steps_used_list = []   # latent steps used per instance (pondernet mode only)
    cot_steps_list = []    # ground-truth CoT step count per instance (None if dataset lacks 'cot')
    model.eval()
    attn_to_latent_list = []
    if model_args.soft_weight:
        embedding_matrix = model.codi.get_base_model().model.embed_tokens.weight.data.to(model.codi.device)
        vocab_center = embedding_matrix.mean(dim=0)
    
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            # encode the question
            past_key_values = None
            # Position ids derived from the mask so left-padding does not shift the real
            # tokens' positions (else batch>1 would see different positions than bs=1).
            # No-op at bs=1 (no padding => 0..L-1). Continuation positions are taken as
            # attn.sum() (the running count of real tokens) at each appended step.
            enc_pos = (batch["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"], position_ids=enc_pos)
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            if model_args.soft_weight:
                soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                soft_embeds = (soft_probs.squeeze(dim=1).unsqueeze(dim=-1) * embedding_matrix.unsqueeze(dim=0)).sum(dim=0)

                soft_probs_expanded = soft_probs.squeeze(dim=1)
                soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)  # Shape: [128, 2048]
                if training_args.use_prj:
                    soft_embeds = model.prj(soft_embeds)
                latent_embd = latent_embd + model_args.soft_weight * soft_embeds
            
            # Running attention mask over the cache (1 = real token, 0 = left-pad),
            # grown by one column per appended latent. Passing it on every step keeps
            # the model from attending to left-padding KVs when batch_size > 1.
            attn = batch["attention_mask"]

            if model.pondernet:
                # --- PonderNet adaptive halting inference (faithful per-example) ---
                # Each example's answer is decoded from the KV prefix at *its own* halt
                # step. When a row crosses the cumulative-halt threshold we slice the
                # shared cache down to that row and decode immediately, so batched eval
                # matches the batch_size=1 result. (The previous loop decoded every row
                # from the batch's termination prefix, inflating the latent steps that
                # early-halting examples were actually evaluated with.)
                k_max = model.max_latent_steps
                threshold = model.pondernet_inf_threshold
                not_halted = torch.ones(batch_size, dtype=torch.float32, device=device)
                done = torch.zeros(batch_size, dtype=torch.bool, device=device)
                steps_used = [k_max] * batch_size      # default: used all steps
                pred_tokens = [None] * batch_size      # answer per row, filled at halt

                for i in range(k_max):
                    pos_ids = attn.sum(dim=1, keepdim=True).long()   # position of the latent appended now
                    attn = torch.cat([attn, torch.ones((batch_size, 1), dtype=attn.dtype, device=attn.device)], dim=1)
                    outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=attn, position_ids=pos_ids)
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

                    lambda_k = model._halting_lambda(latent_embd)          # (B,)
                    not_halted = not_halted * (1.0 - lambda_k.float())     # survivor prob

                    # Rows that cross the halt threshold this step (and, on the final
                    # step, any still-running row, capped at k_max) get their answer
                    # decoded now from the current per-row prefix.
                    crossed = (not_halted < (1.0 - threshold)) & (~done)
                    last_step = (i == k_max - 1)
                    finalize = crossed | (last_step & ~done)
                    if finalize.any():
                        idx = finalize.nonzero(as_tuple=False).flatten()
                        sub_pkv = _slice_past_key_values(past_key_values, idx)
                        sub_preds = generate_answers(sub_pkv, attn.index_select(0, idx))
                        for j, b in enumerate(idx.tolist()):
                            pred_tokens[b] = sub_preds[j]
                            steps_used[b] = i + 1
                            done[b] = True

                    if bool(done.all()):
                        break

                # Safety net: any row that never finalized (should not happen) → empty.
                pred_tokens = [pt if pt is not None else [] for pt in pred_tokens]
            else:
                # --- Original fixed-step inference ---
                steps_used = [training_args.inf_latent_iterations] * batch_size
                inf_latent_iterations = training_args.inf_latent_iterations
                for i in range(inf_latent_iterations):
                    pos_ids = attn.sum(dim=1, keepdim=True).long()   # position of the latent appended now
                    attn = torch.cat([attn, torch.ones((batch_size, 1), dtype=attn.dtype, device=attn.device)], dim=1)
                    outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=attn, position_ids=pos_ids)
                    past_key_values = outputs.past_key_values
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

                    if model_args.soft_weight:
                        soft_probs = F.softmax(model.codi.lm_head(latent_embd).to(model.codi.device), dim=-1)
                        soft_probs_expanded = soft_probs.squeeze(dim=1)
                        soft_embeds = (soft_probs_expanded.unsqueeze(dim=2) * embedding_matrix).sum(dim=1)
                        if training_args.use_prj:
                            soft_embeds = model.prj(soft_embeds)
                        latent_embd = latent_embd + model_args.soft_weight * soft_embeds

                # Fixed-K: every row shares the same prefix, so one batched decode is exact.
                pred_tokens = generate_answers(past_key_values, attn)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                steps_used_list.append(steps_used[mini_step])
                _q = step * data_args.batch_size + mini_step
                _rec = test_set[_q]
                _cot = (_rec.get('cot', '') if isinstance(_rec, dict) else getattr(_rec, 'cot', ''))
                cot_steps_list.append(_cot.count('<<') if _cot else None)
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    q_idx = step * data_args.batch_size + mini_step
                    print(f"Question {q_idx} Starts...")
                    print(f"Q: {question[q_idx]}")
                    print(decoded_pred)
                    print(f"Question {q_idx} Ends")
                    if model.pondernet:
                        print(f"Latent steps used: {steps_used[mini_step]}/{model.max_latent_steps}")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[q_idx]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))
    os.makedirs(data_args.results_dir, exist_ok=True)
    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot)/len(len_cot)}")
    if model.pondernet and steps_used_list:
        from collections import defaultdict
        avg_steps = sum(steps_used_list) / len(steps_used_list)
        print(f"[PonderNet] average latent steps used: {avg_steps:.2f} / {model.max_latent_steps}  "
              f"(threshold={model.pondernet_inf_threshold})")

        # Accuracy-vs-budget table
        step_correct = defaultdict(list)
        for pred, gold, su in zip(ans_pred_list, answer, steps_used_list):
            step_correct[su].append(pred == gold)
        print("\n[PonderNet] Accuracy vs latent budget:")
        print(f"{'Steps':>6} | {'N':>6} | {'Acc (%)':>8}")
        print("-" * 26)
        for k in sorted(step_correct):
            n = len(step_correct[k])
            acc_k = 100.0 * sum(step_correct[k]) / n
            print(f"{k:>6} | {n:>6} | {acc_k:>7.1f}%")

        # instance_results.json — one object per datapoint
        per_example_path = os.path.join(data_args.results_dir, "instance_results.json")
        write_json([
            {
                "steps_used":  su,
                "correct":     p == g,
                "cot_steps":   cs,
                "pred_answer": p,
                "gt_answer":   g,
            }
            for su, p, g, cs in zip(steps_used_list, ans_pred_list, answer, cot_steps_list)
        ], per_example_path)
        print(f"[PonderNet] Instance results saved to {per_example_path}")

        # summary.json — run-level metrics + breakdowns
        by_steps = [
            {"steps_used": k, "n": len(v), "accuracy_pct": round(100.0 * sum(v) / len(v), 2)}
            for k, v in sorted(step_correct.items())
        ]
        cot_buckets = defaultdict(lambda: {"n": 0, "correct": 0, "steps_sum": 0})
        for su, p, g, cs in zip(steps_used_list, ans_pred_list, answer, cot_steps_list):
            key = cs  # int or None
            cot_buckets[key]["n"] += 1
            cot_buckets[key]["correct"] += int(p == g)
            cot_buckets[key]["steps_sum"] += su
        by_cot = [
            {
                "cot_steps":      k,
                "n":              v["n"],
                "accuracy_pct":   round(100.0 * v["correct"] / v["n"], 2),
                "avg_steps_used": round(v["steps_sum"] / v["n"], 2),
            }
            for k, v in sorted(cot_buckets.items(), key=lambda x: (x[0] is None, x[0]))
        ]
        summary = {
            "accuracy_pct":     round(100 * accuracy, 2),
            "total_examples":   len(ans_pred_list),
            "avg_steps_used":   round(avg_steps, 3),
            "max_latent_steps": model.max_latent_steps,
            "threshold":        model.pondernet_inf_threshold,
            "by_steps_used":    by_steps,
            "by_cot_steps":     by_cot,
        }
        summary_path = os.path.join(data_args.results_dir, "summary.json")
        write_json(summary, summary_path)
        print(f"[PonderNet] Run summary saved to {summary_path}")
    else:
        predictions_path = os.path.join(data_args.results_dir, "predictions.json")
        write_json({"ans": ans_pred_list}, predictions_path)
    if model_args.save_ablation:
        ablation_path = os.path.join(data_args.results_dir, f"{data_args.data_name}.jsonl")
        save_jsonl_line(ablation_path, {'model_name': model_args.ckpt_dir, 'data_name': data_args.data_name, 'soft_weight': model_args.soft_weight, 'acc.': accuracy})
    return 100*accuracy

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        if "commonsense" in data_args.data_name:
            pred = sentence.split("The answer is:")[-1].strip()
            if pred[0] not in "ABCDE":
                raise ValueError
            return pred[0]
        elif "strategy" in data_args.data_name or "prontoqa" in data_args.data_name.lower():
            if "True" in sentence:
                return True
            elif "False" in sentence:
                return False
            else:
                raise ValueError
        return float('inf')

    # use the last number as the answer
    pred_answer = float(pred[-1])

    return pred_answer


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1

    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Greedy decoding is deterministic, so multi-pass averaging is redundant — run once.
    num_passes = 1 if training_args.greedy else training_args.inf_num_iterations
    accu_list = []
    for i in range(num_passes):
        set_seed(training_args.seed + i)
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    label = "greedy (1 pass)" if training_args.greedy else f"{num_passes} sampling passes"
    print(f"Average accuracy over {label}: {sum(accu_list) / len(accu_list)}")
