# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
"""Training entrypoint for the PonderNet adaptive-halting latent-CoT model (CODI backbone)."""
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)


def _to_scalar(x):
    """Convert Tensor/number/None to python float (mean-reduced if needed)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # detach, cast to float, mean-reduce if multi-element, then .item()
        return x.detach().float().mean().item()
    # already a number
    return float(x)
def read_json(file_path):
    """Read a JSON document from file_path and return the parsed object."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class EpochProgressCallback(TrainerCallback):
    """Prints a clear separator marking the start of each epoch, alongside
    the default tqdm bar (which only tracks overall step progress)."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            print(f"\n=== Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} ===")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        loss_for_log = loss.detach() if isinstance(loss, torch.Tensor) else loss
        if step % self.args.logging_steps == 0:
            logs = {
                "loss": _to_scalar(loss_for_log),
                "ce_loss": _to_scalar(outputs.get("ce_loss")),
                "distill_loss": _to_scalar(outputs.get("distill_loss")),
                "ref_ce_loss": _to_scalar(outputs.get("ref_ce_loss")),
            }
            if not hasattr(self, "is_global_zero") or self.is_global_zero:
                self.log(logs)
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        # if step % self.args.logging_steps == 0:
        #     self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], "ref_ce_loss": outputs["ref_ce_loss"],})

        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,#training_args.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    # use the last number as the answer
    pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2", "gsm-cot"]):
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

    # model_name_or_path is always a plain GPT-2 (the backbone scaffold) -- never the
    # SIM-CoT CODI checkpoint, which would load as a bare GPT2LMHeadModel and random-init
    # the backbone. Two warm-start recipes are available (see pondernet/README.md
    # "Warm-start strategy"): decoder-only via --decoder_path (inside CODI.__init__), and
    # full-model via --simcot_ckpt (below). With --simcot_ckpt unset the run is decoder-only
    # (only --decoder_path warm-starts); set it to additionally warm-start the backbone +
    # LoRA + prj. NOTE: the training script enables --simcot_ckpt by default, so its
    # out-of-the-box default is full-model -- clear SIMCOT_CKPT there for decoder-only.
    model = CODI(model_args, training_args, lora_config)

    # Optional full-model warm-start: load the full CODI state dict (backbone + LoRA +
    # decoder + prj) from a SIM-CoT CODI checkpoint into the assembled module. This MUST be a
    # load_state_dict into the wrapper -- NOT model_name_or_path pointed at the checkpoint
    # (that random-inits the backbone). Skipped entirely when --simcot_ckpt is unset.
    if model_args.simcot_ckpt:
        sd = load_file(model_args.simcot_ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Every checkpoint tensor must map onto a model param: a non-empty `unexpected` means
        # a namespace mismatch (the exact failure mode that silently produced a random model).
        if unexpected:
            raise RuntimeError(
                f"SIM-CoT warm-start: {len(unexpected)} checkpoint tensors did not map onto the "
                f"model (namespace mismatch). First few: {unexpected[:8]}"
            )
        # The only params allowed to be newly-initialized are the new PonderNet halt head.
        unexpected_missing = [k for k in missing if not k.startswith("halt_head")]
        if unexpected_missing:
            raise RuntimeError(
                f"SIM-CoT warm-start: {len(unexpected_missing)} model params were NOT loaded from "
                f"the checkpoint (expected only halt_head.*). First few: {unexpected_missing[:8]}"
            )
        # Sentinel check: the SIM-CoT latent-reasoning weights actually landed.
        for sentinel in ("codi.base_model.model.transformer.wte.weight",
                         "decoder.lm_head.weight", "prj.1.weight"):
            if sentinel in missing:
                raise RuntimeError(f"SIM-CoT warm-start: core weight '{sentinel}' was not loaded.")
        logging.warning(
            f"Warm-started {len(sd)} tensors from SIM-CoT checkpoint {model_args.simcot_ckpt}. "
            f"Newly-initialized (PonderNet halt head): {missing}"
        )

    if training_args.pondernet:
        freeze_model(model)
        for name, p in model.named_parameters():
            if 'halt_head' in name or 'lora_' in name:
                p.requires_grad = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception as e:
            raise ValueError(
                f"Could not locate answer prompt {answer_prompts[0].tolist()} in token sequence "
                f"{tokens.tolist()}"
            ) from e

    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")
            
            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            token_nums = []
            if data_args.data_path:
                raw_data = read_json(data_args.data_path)
            elif raw_data is None:
                raw_data = list(load_dataset("zen-E/GSM8k-Aug")["train"])
            if data_args.max_train_samples is not None:
                raw_data = raw_data[:data_args.max_train_samples]
            for num_iter, example in tqdm(enumerate(raw_data)):
                if 'cot' not in example: 
                    example['cot'] = example['steps']
                    example['cot'] = ' '.join(example['cot'])
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name: # icot-full (GSM8k-Aug-NL)
                    # bad data
                    if example["answer"] is None or example["response"] is None:
                        continue
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]
                    answer = f"The answer is: {example['answer'].split(' ')[-1]}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(". ".join(cot)+".\n")
                    answers.append(answer)
                elif "icot" in self.data_name: # icot (GSM8k-Aug)
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    if not training_args.include_last_cot:
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.max_latent_steps):
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in data_args.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")

            self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))
        
            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
          
            return dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")
        if "icot" in data_args.data_name:
            if data_args.data_path:
                raw_data = read_json(data_args.data_path)
            else:
                raw_data = list(load_dataset("zen-E/GSM8k-Aug")["train"])
            if data_args.max_train_samples:
                raw_data = raw_data[:data_args.max_train_samples]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=raw_data, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_args.data_name:
            if not data_args.data_path:
                raise ValueError("prontoqa requires --data_path pointing to prontoqa_train.json")
            with open(data_args.data_path) as f:
                dataset = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.add_callback(EpochProgressCallback())

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.warning(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
