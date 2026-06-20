"""CODI model definition with PonderNet adaptive-halting extension and data-argument dataclasses."""
import logging
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput
import copy
from torch.amp import autocast
from typing import List, Sequence, Iterable, Union, Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    use_decoder: bool = field(default=False, metadata={"help": 'use decoder'})
    decoder_path: str = field(default=None)
    soft_weight: float = field(default=None, metadata={"help": "soft weight"})
    save_ablation: bool = field(
        default=False,
        metadata={"help": "Save ablation results. Only True when specified in command line."},
    )
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})
    simcot_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a SIM-CoT CODI .safetensors to warm-start the full CODI model "
                          "(backbone+LoRA+decoder+prj) from. Loaded via load_state_dict(strict=False) "
                          "AFTER the model is assembled; only the PonderNet halt head stays newly-initialized. "
                          "Do NOT point model_name_or_path at this file — it is a CODI wrapper, not a plain GPT-2."},
    )

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})
    data_path: str = field(default="", metadata={"help": "Local path to training/eval data JSON. If empty, loads from HuggingFace."})
    results_dir: str = field(default="../results", metadata={"help": "Directory for evaluation output JSON files (repo root, outside the module)."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate training set to this many samples."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="", metadata={"help": "Deprecated; unused."})
    max_latent_steps: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    explain_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the explain loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})
    # --- PonderNet adaptive halting (Option C) ---
    pondernet: bool = field(default=False, metadata={"help": "Enable PonderNet-style adaptive latent halting. Off = original SIM-CoT path."})
    pondernet_halt_bias_init: float = field(default=-2.0, metadata={"help": "Initial bias of the halting head so early lambda_k is small (model uses more steps early in training)."})
    pondernet_beta: float = field(default=1.0, metadata={"help": "Weight of auxiliary decoder loss L_step in PonderNet mode."})
    pondernet_gamma: float = field(default=0.01, metadata={"help": "Weight of KL-geometric regularizer in PonderNet mode."})
    pondernet_geom_mean: float = field(default=3.0, metadata={"help": "Mean number of steps for the geometric prior used in KL_geom (controls compute pressure)."})
    pondernet_adaptive_prior: bool = field(default=False, metadata={"help": "Per-instance geometric prior: geom_mean_i = affine remap of the example's real reasoning-step count (#expr-1 from get_steps). Off = global pondernet_geom_mean (exp <=04). Requires use_decoder."})
    pondernet_prior_scale: float = field(default=1.0, metadata={"help": "Adaptive prior slope alpha in geom_mean_i = alpha*n_i + beta."})
    pondernet_prior_offset: float = field(default=1.5, metadata={"help": "Adaptive prior offset beta in geom_mean_i = alpha*n_i + beta. Must be >1 to avoid the degenerate g=1 point-mass prior; doubles as the z0/answer latent overhead."})
    pondernet_inf_threshold: float = field(default=0.5, metadata={"help": "Cumulative halting probability threshold for inference early-stopping. Stop when sum_k p_k > threshold."})
    pondernet_trunc_k: bool = field(default=False, metadata={"help": "Per-instance truncated-K training (exp-06): stop the latent loop at K_i = max(1, n_i) steps per example instead of always running max_latent_steps. Requires pondernet=True and use_decoder=True."})

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    logging.info(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

def get_steps(
    ref_input_ids: Union[torch.Tensor, Sequence[Sequence[int]]],
    latent_num: int = 2,
    start_ids: Iterable[int] = (2501, 1134),
    end_id: int = 2511,
    eot_id: int = 128009,
    pad_id: int = 128256,
    stop_ids: Iterable[int] = (128009, 128256),
    trim_at_first_stop: bool = True,
) -> List[List[List[int]]]:
    if isinstance(ref_input_ids, torch.Tensor):
        assert ref_input_ids.dim() == 2, "ref_input_ids must be a 2-D tensor [B, T]"
        batch = ref_input_ids
        B, T = batch.size()
        as_lists = batch.detach().cpu().tolist()
    else:
        as_lists = ref_input_ids
        B = len(as_lists)

    start_set = set(start_ids)
    stop_set = set(stop_ids)

    result: List[List[List[int]]] = []
    for b in range(B):
        seq: List[int] = list(as_lists[b])
        if trim_at_first_stop:
            for k, tok in enumerate(seq):
                if tok in stop_set:
                    seq = seq[:k]
                    break

        steps_for_sample: List[List[int]] = []
        i = 0
        n = len(seq)
        while i < n:
            tok = seq[i]
            if tok in start_set:
                j = i + 1
                end_pos: Optional[int] = None
                while j < n:
                    if seq[j] == end_id:
                        end_pos = j
                        break
                    if seq[j] in stop_set:
                        break
                    j += 1
                if end_pos is not None:
                    steps_for_sample.append(seq[i:end_pos + 1] + [eot_id])
                    i = end_pos + 1
                    continue
            i += 1
        # >
        max_steps = latent_num
        if len(steps_for_sample) > max_steps:
            kept = steps_for_sample[:max_steps - 1]
            merged: List[int] = []
            for s in steps_for_sample[max_steps - 1:]:
                if len(s) > 0 and s[-1] == eot_id:
                    merged.extend(s[:-1])
                else:
                    merged.extend(s)
            merged.append(eot_id)
            kept.append(merged)
            steps_for_sample = kept
        # <
        elif len(steps_for_sample) < max_steps:
            while len(steps_for_sample) < max_steps:
                steps_for_sample.append([pad_id])

        result.append(steps_for_sample)

    return result

def pad_steps(
    step_list,
    pad_id: int = 128256
):
    max_len = max(len(step) for steps in step_list for step in steps)
    # maximum number of steps
    S_max = max(len(steps) for steps in step_list)
    # globally longest step length
    L_max = max(len(step) for steps in step_list for step in steps)
    
    result: List[List[List[int]]] = []
    for steps in step_list:
        padded_steps: List[List[int]] = []
        for step in steps:
            cur = list(step)
            pad_len = L_max - len(cur)
            if pad_len > 0:
                cur = cur + [pad_id] * pad_len
            padded_steps.append(cur)
        while len(padded_steps) < S_max:
            padded_steps.append([pad_id] * L_max)
        result.append(padded_steps)

    return result

def dedup_trailing_pads(step_token_ids, pad_id=128256):
    if not step_token_ids:
        return []

    max_len = len(step_token_ids[0])

    while max_len > 1:
        if all(row[max_len - 2] == pad_id for row in step_token_ids):
            max_len -= 1
        else:
            break

    return [row[:max_len] for row in step_token_ids]


def truncated_geometric_prior(geom_mean, K, dtype=torch.float32, device=None):
    """Truncated-geometric halting prior q over steps 0..K-1.

    g = 1/geom_mean; q_k = (1-g)^k * g for k < K-1, q_{K-1} = (1-g)^{K-1} (absorbing
    boundary, matching the halting-distribution convention), renormalized to sum 1.

    geom_mean: float (global prior, returns shape (K,)) OR tensor (B,) of per-example
    means (returns shape (B, K), one prior per row).
    """
    k_idx = torch.arange(K, dtype=dtype, device=device)  # (K,)
    if torch.is_tensor(geom_mean):
        g = (1.0 / geom_mean.to(dtype)).unsqueeze(1)             # (B, 1)
        q = (1.0 - g) ** k_idx.unsqueeze(0) * g                   # (B, K)
        q[:, -1] = (1.0 - g.squeeze(1)) ** (K - 1)               # absorbing boundary
        q = q / q.sum(dim=1, keepdim=True)
    else:
        g = 1.0 / geom_mean
        q = (1.0 - g) ** k_idx * g                                # (K,)
        q[-1] = (1.0 - g) ** (K - 1)
        q = q / q.sum()
    return q


def kl_to_truncated_geometric(p_k, geom_mean):
    """Per-row KL(p_k || q) where q is the truncated-geometric prior. Returns (B,).

    geom_mean: float (global) or tensor (B,) (per-example). p_k: (B, K), rows sum to 1.
    """
    K = p_k.size(1)
    q = truncated_geometric_prior(geom_mean, K, dtype=p_k.dtype, device=p_k.device)
    log_q = torch.log(q.clamp_min(1e-8))
    if log_q.dim() == 1:
        log_q = log_q.unsqueeze(0)                                # (1, K) broadcast
    log_p = torch.log(p_k.clamp_min(1e-8))
    return (p_k * (log_p - log_q)).sum(dim=1)                     # (B,)


def count_real_steps(steps_list, pad_id):
    """Per-example count of real reasoning steps from get_steps output.

    steps_list: list (B) of per-example step lists; padded slots are the [pad_id]
    sentinel. Returns a python list (B) of counts (number of non-pad slots). The
    caller clamps to a valid prior-mean range (>= 1).
    """
    return [sum(1 for step in ex if step != [pad_id]) for ex in steps_list]


def adaptive_prior_mean(n_real, scale, offset, K, dtype=torch.float32, device=None):
    """Affine remap of per-example reasoning-step counts to geometric-prior means.

    geom_mean_i = clamp(scale * n_i + offset, offset, K). The offset (>= ~1.5) keeps the
    prior off the degenerate g=1 point mass that geom_mean=n_i hits for the ~53% of
    examples with n_i <= 1, while preserving a monotonic per-instance signal. n_real:
    list/tensor (B,) of non-pad reasoning-step counts. Returns means (B,).
    """
    n = torch.as_tensor(n_real, dtype=dtype, device=device)
    return (scale * n + offset).clamp(min=offset, max=float(K))


def build_active_mask(K_i_list, K_batch_max, device=None):
    """Boolean mask (B, K_batch_max) where active[i, k] = (k < K_i_list[i]).

    K_i_list: list (B,) of per-example step counts (clamped to [1, max_latent_steps]).
    K_batch_max: int, max(K_i_list) — the loop iteration count for this batch.
    """
    k_idx = torch.arange(K_batch_max, device=device).unsqueeze(0)   # (1, K)
    Ki_t = torch.tensor(K_i_list, device=device).unsqueeze(1)        # (B, 1)
    return k_idx < Ki_t                                               # (B, K)


class LowRankProjector(nn.Module):
    def __init__(self, input_dim, output_dim, rank=64):
        super(LowRankProjector, self).__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.U), self.V)

class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM 
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )
        if model_args.use_decoder:
            if model_args.decoder_path:
                self.decoder = model_wrapper_class.from_pretrained(
                    model_args.decoder_path,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    use_flash_attention_2=False,
                    resume_download=True,
                )
                if self.codi.lm_head.in_features == self.decoder.lm_head.in_features:
                    self.pj_in = nn.Identity()
                else:
                    self.pj_in = nn.Linear(self.codi.lm_head.in_features, self.decoder.lm_head.in_features)
                # self.pj_out = nn.Linear(self.decoder.lm_head.out_features, self.codi.lm_head.out_features)
                input_dim = self.decoder.lm_head.out_features
                output_dim = self.codi.lm_head.out_features
                if input_dim == output_dim:
                    self.pj_out = nn.Identity()
                else:
                    self.pj_out = LowRankProjector(input_dim, output_dim, rank=input_dim // 4)
            else:
                self.decoder = model_wrapper_class.from_pretrained(
                        self.model_name,
                        torch_dtype=(
                            torch.float16 if training_args.bf16 is False else torch.bfloat16
                        ),
                        use_flash_attention_2=False,
                        resume_download=True,
                    )
        

        # saved_weights = torch.load(
        #     '/fs-computility/mllm/shared/weixilin/coconut/ckpts/gsm_cot/gsm-cot/checkpoint_13', map_location=torch.device(self.codi.device)
        # )
        # self.codi.load_state_dict(saved_weights, strict=False)
        

        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.max_latent_steps = training_args.max_latent_steps
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
                
        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        
        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # Explain Loss
        self.explain_loss_factor = training_args.explain_loss_factor
        # general 
        self.fix_attn_mask = training_args.fix_attn_mask

        # --- PonderNet adaptive halting (Option C) ---
        self.pondernet = getattr(training_args, "pondernet", False)
        if self.pondernet:
            # Halting head: latent hidden state h_k -> scalar logit -> lambda_k in [0,1]
            self.halt_head = nn.Linear(self.dim, 1)
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, training_args.pondernet_halt_bias_init)
            self.pondernet_beta = training_args.pondernet_beta
            self.pondernet_gamma = training_args.pondernet_gamma
            self.pondernet_geom_mean = training_args.pondernet_geom_mean
            self.pondernet_adaptive_prior = getattr(training_args, "pondernet_adaptive_prior", False)
            self.pondernet_prior_scale = getattr(training_args, "pondernet_prior_scale", 1.0)
            self.pondernet_prior_offset = getattr(training_args, "pondernet_prior_offset", 1.5)
            self.pondernet_inf_threshold = training_args.pondernet_inf_threshold
            self.pondernet_trunc_k = getattr(training_args, "pondernet_trunc_k", False)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.codi.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if self.model_args.use_decoder:
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.codi.gradient_checkpointing_disable()
        if self.model_args.use_decoder:
            self.decoder.gradient_checkpointing_disable()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name.lower():
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name.lower():
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def _halting_distribution(self, lambdas_list):
        """Compute PonderNet halting distribution from per-step conditional lambdas.

        lambdas_list: list of K tensors each (B,) - conditional halt prob at step k.
        Returns p: (B, K) - probability of halting exactly at step k.
          For k < K: p_k = lambda_k * prod_{j<k}(1 - lambda_j)
          For k = K: p_K = prod_{j<K}(1 - lambda_j)  [absorbing boundary, lambda_K = 1]
        Rows sum to 1 by construction (no renormalization needed beyond fp precision).
        """
        lambdas = torch.stack(lambdas_list, dim=1)  # (B, K)
        one_minus = 1.0 - lambdas

        # exclusive cumprod of (1-lambda): p_prior[k] = prod_{j<k}(1-lambda_j)
        ones_col = torch.ones(lambdas.size(0), 1, dtype=lambdas.dtype, device=lambdas.device)
        p_prior = torch.cat([ones_col, torch.cumprod(one_minus[:, :-1], dim=1)], dim=1)  # (B, K)

        # p_k = lambda_k * p_prior_k for k < K; p_K = p_prior_K (absorbing)
        p = lambdas * p_prior                       # (B, K) — first K-1 entries correct
        p = torch.cat([p[:, :-1], p_prior[:, -1:]], dim=1)  # last entry = p_prior_K

        # clamp negatives from fp error, renorm for exact sum-1
        p = p.clamp_min(0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return p  # (B, K)

    def _kl_geom(self, p_k, geom_mean=None):
        """KL(p_k || q) to a truncated-geometric prior; scalar, mean over batch.

        p_k: (B, K) halting distribution (rows sum to 1).
        geom_mean: optional per-example means, tensor (B,). Defaults to the global
        scalar ``self.pondernet_geom_mean`` (exp <=04 behavior). A per-example tensor
        builds one prior per row (exp 05 adaptive-prior path).
        """
        if geom_mean is None:
            geom_mean = self.pondernet_geom_mean
        return kl_to_truncated_geometric(p_k, geom_mean).mean()

    def _halting_lambda(self, hidden):
        """hidden: (B, 1, dim) latent state h_k -> lambda_k: (B,) in (0,1)."""
        logit = self.halt_head(hidden.squeeze(1))
        return torch.sigmoid(logit).squeeze(-1)

    def _answer_logits_and_loss(self, decoder_input_ids, labels, past_key_values, attention_mask=None):
        """Decode the answer after the CURRENT latent prefix.
        Returns (outputs, logits, per_example_ce[(B,)]). Does NOT mutate past_key_values
        (GPT-2 legacy tuple cache returns a fresh cache; input tuple is left intact)."""
        embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True,
                                past_key_values=past_key_values, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        tok_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)).float(),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none",
        ).view(shift_labels.size(0), -1)
        valid = (shift_labels != -100).float()
        per_example_ce = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return outputs, logits, per_example_ce

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None
        
        # Encode the question
        past_key_values = None
        outputs = self.codi(input_ids=encoder_input_ids, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=encoder_attention_mask)
        past_key_values = outputs.past_key_values
        latent_hidden = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) # as the next input
        
        if self.model_args.use_decoder:
            step_idx = 0
            explain_loss_total = 0.0
            effective_steps_cnt = 0
            if 'llama' in self.model_args.model_name_or_path.lower():
                steps_list = get_steps(ref_input_ids, self.max_latent_steps+1)
                steps_pad_list = pad_steps(steps_list)
                # print()
                # steps_list = pad_steps(steps_list)
            elif 'gpt' in self.model_args.model_name_or_path.lower():
                steps_list = get_steps(ref_input_ids, self.max_latent_steps+1, start_ids=(16791, 9959), end_id=4211, 
                                       eot_id=self.tokenizer.eos_token_id, pad_id=self.tokenizer.pad_token_id, 
                                       stop_ids=(self.tokenizer.eos_token_id, self.tokenizer.pad_token_id))
                steps_pad_list = pad_steps(steps_list, pad_id=self.tokenizer.pad_token_id)

            else:
                raise ValueError("no implementaion")
        
        if self.use_prj:
            with autocast('cuda', dtype=torch.bfloat16, enabled=True):
                latent_hidden = self.prj(latent_hidden)
            # latent_hidden = self.prj(latent_hidden)


        if self.model_args.use_decoder:
            
            bz = len(steps_pad_list)
            step_token_ids = []
            for bz_idx in range(bz):
                step_token_ids.append(steps_pad_list[bz_idx][step_idx])
                step_token_ids = dedup_trailing_pads(step_token_ids, pad_id=self.tokenizer.pad_token_id)
            indices = torch.tensor(step_token_ids, dtype=torch.long, device=self.codi.device)
            explain_embds = self.get_embd(self.codi, self.model_name)(indices)
            explain_embds = torch.concat([latent_hidden, explain_embds], dim=1)
            
            prefix = torch.full((bz, 1), -570, dtype=indices.dtype, device=indices.device)
            indices_with_prefix = torch.cat([prefix, indices], dim=1)
            explain_attention_mask = (indices_with_prefix != self.tokenizer.pad_token_id)
            

            explain_labels = indices_with_prefix.clone()
            explain_labels = explain_labels.masked_fill(
                (explain_labels == -570) | (explain_labels == self.tokenizer.pad_token_id),
                -100
            )
            step_idx += 1

            if self.model_args.decoder_path:
                explain_embds = self.pj_in(explain_embds)


            if (explain_labels != -100).sum() == 0:
                explain_loss_total += 0.0
            else:
                
                with autocast('cuda', dtype=torch.bfloat16):
                    explain_outputs = self.decoder(
                        inputs_embeds=explain_embds,
                        attention_mask=explain_attention_mask,
                        output_hidden_states=True
                    )

                
                # explain_outputs = self.decoder(
                #         inputs_embeds=explain_embds,
                #         attention_mask=explain_attention_mask,
                #         output_hidden_states=True
                #     )
                explain_logits = explain_outputs.logits

                if self.model_args.decoder_path:
                    explain_logits = self.pj_out(explain_logits)

                shift_explain_logits = explain_logits[..., :-1, :].contiguous()
                shift_explain_logits = shift_explain_logits.view(-1, shift_explain_logits.size(-1))

                shift_explain_labels = explain_labels[..., 1:].contiguous()
                shift_explain_labels = shift_explain_labels.view(-1)
                        
                if (shift_explain_labels != -100).sum() == 0:
                    explain_loss = torch.tensor(0.0, device=shift_explain_logits.device)
                else:    
                    explain_loss = self.loss_fct(shift_explain_logits, shift_explain_labels)
                    effective_steps_cnt += 1
                explain_loss_total += explain_loss
            # print(step_idx, explain_loss, explain_loss_total)
            # print()

        len_pred_loss = 0
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((encoder_attention_mask.size(0), self.max_latent_steps), device=ref_labels.device)

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0

        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask) 
        
        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs] 
        ref_input_ids = [ref_input_ids] 

        # Process the position tensor
        # Normalise the position definition 
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower(): # there is one more token standing for " " 
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
       
        # For DEBUG: Print the probability of the teacher task to predict the correct answer
        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(zip(ref_input_ids, ref_outputs_list)):
                # evalutae the reference model
                if len(ref_outputs_list) > 1:
                    pos = ref_answer_position[i]
                else:
                    pos = ref_answer_position
                ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
                input_positions = (pos-1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(1, probe_positions_positions).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                print(f'stage{i}: mean of the prob of the target token: {ref_probs_of_target.mean()}')
        
        # the model answer position is the position of the eot token to predict the first token of the response
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position -1
      
        max_latent_steps = self.max_latent_steps

        # Exp-06: per-instance truncated-K — shorten the loop per batch
        active_mask = None
        if self.pondernet and getattr(self, "pondernet_trunc_k", False) and self.model_args.use_decoder:
            n_real = count_real_steps(steps_list, self.tokenizer.pad_token_id)
            K_i_list = [max(1, min(n, self.max_latent_steps)) for n in n_real]
            K_batch_max = max(K_i_list)
            active_mask = build_active_mask(K_i_list, K_batch_max, device=self.codi.device)
            max_latent_steps = K_batch_max

        pondernet_lambdas, pondernet_step_losses = [], []
        if self.max_latent_steps != 0:
            for i in range(max_latent_steps):
                # Implicit CoT generation
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.codi(inputs_embeds=latent_hidden, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                # outputs = self.codi(inputs_embeds=latent_hidden, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_hidden = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    with autocast('cuda', dtype=torch.bfloat16, enabled=True):
                        latent_hidden = self.prj(latent_hidden)
                    # latent_hidden = self.prj(latent_hidden)

                if self.pondernet:
                    # lambda_k from this latent state; candidate answer after k = i+1 steps
                    pondernet_lambdas.append(self._halting_lambda(latent_hidden))           # (B,)
                    _, _, ans_ce_k = self._answer_logits_and_loss(
                        decoder_input_ids, labels, past_key_values, attention_mask=None)  # (B,)
                    pondernet_step_losses.append(ans_ce_k)

                if self.model_args.use_decoder:
                    bz = len(steps_pad_list)
                    step_token_ids = []
                    for bz_idx in range(bz):
                        step_token_ids.append(steps_pad_list[bz_idx][step_idx])
                        step_token_ids = dedup_trailing_pads(step_token_ids, pad_id=self.tokenizer.pad_token_id)
                    indices = torch.tensor(step_token_ids, dtype=torch.long, device=self.codi.device)
                    explain_embds = self.get_embd(self.codi, self.model_name)(indices)
                    explain_embds = torch.concat([latent_hidden, explain_embds], dim=1)
                    
                    prefix = torch.full((bz, 1), -570, dtype=indices.dtype, device=indices.device)
                    indices_with_prefix = torch.cat([prefix, indices], dim=1)
                    explain_attention_mask = (indices_with_prefix != self.tokenizer.pad_token_id)
                    

                    explain_labels = indices_with_prefix.clone()
                    explain_labels = explain_labels.masked_fill(
                        (explain_labels == -570) | (explain_labels == self.tokenizer.pad_token_id),
                        -100
                    )

                    if self.model_args.decoder_path:
                        explain_embds = self.pj_in(explain_embds)

                    step_idx += 1
                    if (explain_labels != -100).sum() == 0:
                        explain_loss_total += 0.0
                    else:
                        with autocast('cuda', dtype=torch.bfloat16):
                            explain_outputs = self.decoder(
                                inputs_embeds=explain_embds,
                                attention_mask=explain_attention_mask,
                                output_hidden_states=True
                            )
                        # explain_outputs = self.decoder(
                        #     inputs_embeds=explain_embds,
                        #     attention_mask=explain_attention_mask,
                        #     output_hidden_states=True
                        # )
                        explain_logits = explain_outputs.logits

                        if self.model_args.decoder_path:
                            explain_logits = self.pj_out(explain_logits)

                        shift_explain_logits = explain_logits[..., :-1, :].contiguous()
                        shift_explain_logits = shift_explain_logits.view(-1, shift_explain_logits.size(-1))

                        shift_explain_labels = explain_labels[..., 1:].contiguous()
                        shift_explain_labels = shift_explain_labels.view(-1)
                        if (shift_explain_labels != -100).sum() == 0:
                            explain_loss = torch.tensor(0.0, device=shift_explain_logits.device)
                        else:    
                            explain_loss = self.loss_fct(shift_explain_logits, shift_explain_labels)
                            effective_steps_cnt += 1
                        
                        explain_loss_total += explain_loss
                    # print(step_idx, explain_loss, explain_loss_total)
                    # print()

                

                # Calculate the distillation loss
                if i == max_latent_steps - 1: # the last latent embedding
                    # Decode the final answer in natural language
                    embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
                  
                    if dynamic_mask is not None: # Prevent attending the paddings
                        decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
                        dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1)
                        dynamic_mask = dynamic_mask.bool()
                    # Student task's output

                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
                    # outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
                    # Teacher task's output
                    ref_outputs = ref_outputs_list[0]
                    
                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs.hidden_states)):
                        ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                        out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

                        distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                        
                        if self.distill_loss_div_std:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= ref_selected.std()
                            distill_loss_tmp /= ref_selected.std()
                        distill_loss += distill_loss_tmp
                    
                    distill_loss /= len(outputs.hidden_states)
                    
                    if self.print_loss:
                        print(f'latent{i}: distill_loss={distill_loss}')

                    distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == max_latent_steps - 1:
                        logits = outputs.logits
                        effective_logits = logits[:, :-1, :]
                        effective_logits = effective_logits.reshape(-1, logits.size(-1))
                        target_ids = labels[:, 1:].reshape(-1)                        
                        ce_loss = self.loss_fct(effective_logits, target_ids)
                        ce_loss_total += ce_loss

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor 

        # Weigh the distillation loss
        distill_loss *= self.distill_loss_factor
        distill_loss_total *= self.distill_loss_factor
        if self.model_args.use_decoder:
            explain_loss_total *= self.explain_loss_factor
            explain_loss_total /= max(1.0, effective_steps_cnt)

        if self.print_loss:
            if self.model_args.use_decoder:
                print(f'loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}, explain_loss={explain_loss_total}')    
            else:
                print(f'loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}')

        loss = ce_loss_total + distill_loss_total + ref_ce_loss

        if self.model_args.use_decoder:
            explain_loss_total = torch.as_tensor(explain_loss_total, device=loss.device, dtype=loss.dtype)
            loss += explain_loss_total
        if ce_loss_total != 0:
            ce_loss_total = ce_loss_total.detach()
        if distill_loss_total != 0:
            distill_loss_total = distill_loss_total.detach()
        if ref_ce_loss != 0:
            ref_ce_loss = ref_ce_loss.detach()
        if self.model_args.use_decoder:
            if explain_loss_total != 0:
                explain_loss_total = explain_loss_total.detach()
        # print(f"{ce_loss_total=}, {distill_loss_total=}, {ref_ce_loss=}, {explain_loss_total}")

        if self.pondernet and len(pondernet_lambdas) > 0:
            # --- Phase 4: PonderNet objective ---
            pondernet_p = self._halting_distribution(pondernet_lambdas)  # (B, K) in-graph

            # L_pondernet = E_p[L_ans^(k)] = sum_k p_k * L_ans^(k), averaged over batch
            step_losses_tensor = torch.stack(pondernet_step_losses, dim=1)  # (B, K) in-graph

            # Exp-06: per-instance mask — zero out halting mass and step losses beyond K_i
            if active_mask is not None:
                pondernet_p = pondernet_p * active_mask.float()
                pondernet_p = pondernet_p / pondernet_p.sum(dim=1, keepdim=True).clamp_min(1e-8)
                step_losses_tensor = step_losses_tensor * active_mask.float()

            l_pondernet = (pondernet_p * step_losses_tensor).sum(dim=1).mean()

            # KL_geom regularizer — penalises distributions far from geometric prior.
            # Adaptive prior (exp 05): per-example mean = affine remap of the example's real
            # reasoning-step count, geom_mean_i = alpha*n_i + beta clamped to [beta, K]; the
            # offset keeps the prior off the degenerate g=1 point mass. Else global scalar.
            if self.pondernet_adaptive_prior:
                assert self.model_args.use_decoder, \
                    "pondernet_adaptive_prior requires use_decoder (needs steps_list)"
                K = pondernet_p.size(1)
                n_real = count_real_steps(steps_list, self.tokenizer.pad_token_id)
                geom_mean_i = adaptive_prior_mean(
                    n_real, self.pondernet_prior_scale, self.pondernet_prior_offset, K,
                    dtype=pondernet_p.dtype, device=pondernet_p.device,
                )
                kl_geom = self._kl_geom(pondernet_p, geom_mean=geom_mean_i)
            else:
                kl_geom = self._kl_geom(pondernet_p)

            # Assemble total loss:
            #   L_pondernet  replaces ce_loss_total (the fixed-K answer CE)
            #   beta * L_step keeps the aux-decoder reconstruction loss
            #   gamma * KL_geom adds the compute-efficiency pressure
            #   distill and ref_ce are preserved from original CODI
            loss = l_pondernet + distill_loss_total + ref_ce_loss
            if self.model_args.use_decoder:
                loss = loss + self.pondernet_beta * explain_loss_total
            loss = loss + self.pondernet_gamma * kl_geom

            if self.print_loss:
                print(f"l_pondernet={l_pondernet.item():.4f}  kl_geom={kl_geom.item():.4f}  "
                      f"p_mean_step={(pondernet_p * torch.arange(1, pondernet_p.size(1)+1, dtype=pondernet_p.dtype, device=pondernet_p.device)).sum(dim=1).mean().item():.2f}")

            ret = {"loss": loss, "logits": logits,
                   "ce_loss": l_pondernet.detach(), "distill_loss": distill_loss_total,
                   "ref_ce_loss": ref_ce_loss, "kl_geom": kl_geom.detach()}
            if self.model_args.use_decoder:
                ret['explain_loss'] = explain_loss_total
            ret['pondernet_p'] = pondernet_p
            ret['pondernet_lambdas'] = torch.stack(pondernet_lambdas, dim=1).detach()
            ret['pondernet_step_losses'] = step_losses_tensor.detach()
        else:
            ret = {"loss": loss, "logits": logits, "ce_loss": ce_loss_total, "distill_loss": distill_loss_total, "ref_ce_loss": ref_ce_loss}
            if self.model_args.use_decoder:
                ret['explain_loss'] = explain_loss_total
        return ret

