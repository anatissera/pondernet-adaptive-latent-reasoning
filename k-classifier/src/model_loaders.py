"""Local model loaders for Option-A.

The checkpoints downloaded for SIM-CoT are not plain Hugging Face model
directories: their config.json files are empty and the original code expects a
GPT-2 base model plus a project-specific wrapper. These loaders reconstruct
those wrappers and then load the local checkpoint weights.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from contextlib import contextmanager
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPTION_A_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = OPTION_A_DIR.parent
MODELS_DIR = OPTION_A_DIR / "models"
BASE_GPT2_DIR = MODELS_DIR / "gpt2"
COCONUT_DIR = MODELS_DIR / "SIM_COT-GPT2-Coconut"
CODI_DIR = MODELS_DIR / "SIM_COT-GPT2-CODI"


class ModelLoadError(RuntimeError):
    """Raised when a local checkpoint cannot be loaded."""


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any


def load_coconut():
    """Load the local GPT-2 Coconut/SIM-CoT checkpoint.

    Returns:
        (model, tokenizer), compatible with Option-A's coconut backend.
    """

    torch = _require("torch", "Coconut loading requires PyTorch.")
    transformers = _require("transformers", "Coconut loading requires transformers.")
    _require_path(BASE_GPT2_DIR / "config.json", "GPT-2 base model")
    checkpoint_path = _require_path(COCONUT_DIR / "checkpoint_28", "Coconut checkpoint")

    _add_to_syspath(REPO_ROOT / "baselines" / "Coconut")
    try:
        from coconut import Coconut, CoconutGPT_Same_Word_Embedding
        from utils import Config
    except Exception as exc:
        raise ModelLoadError(
            "Could not import Coconut classes from baselines/Coconut."
        ) from exc

    device = _device(torch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(BASE_GPT2_DIR),
        local_files_only=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    state_dict = _torch_load(torch, checkpoint_path, device)
    mode = _infer_coconut_mode(state_dict)

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(BASE_GPT2_DIR),
        local_files_only=True,
    )
    # The Coconut/SIM-CoT base model consumes the added latent tokens at
    # inference, so it must be resized before loading checkpoint weights that
    # include those tokens. The auxiliary explainable model is not used at
    # inference and the checkpoint stores it with the original GPT-2 vocab.
    base_model.resize_token_embeddings(len(tokenizer))

    if mode == "simcot":
        explainable_model = transformers.AutoModelForCausalLM.from_pretrained(
            str(BASE_GPT2_DIR),
            local_files_only=True,
        )
        config = Config(
            {
                "c_thought": int(os.environ.get("OPTION_A_C_THOUGHT", "2")),
                "max_latent_stage": int(os.environ.get("OPTION_A_MAX_LATENT_STAGE", "6")),
                "training_method": "full",
                "mode": "coconutgpt_same_word_embedding",
                "explain_mode": "v1_aug",
                "w_prompt": False,
            }
        )
        model = CoconutGPT_Same_Word_Embedding(
            base_model,
            explainable_model,
            tokenizer,
            latent_id,
            start_id,
            end_id,
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<<"),
            config.c_thought,
            config,
        )
    else:
        model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    load_result = model.load_state_dict(state_dict, strict=False)
    _initialize_coconut_special_embeddings(base_model, tokenizer, [latent_id, start_id, end_id])
    model.to(device)
    model.eval()
    print(
        "Loaded Coconut checkpoint "
        f"mode={mode}, missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    return model, tokenizer


def load_codi():
    """Load the local GPT-2 CODI checkpoint.

    Returns:
        (model, tokenizer), compatible with Option-A's codi backend.
    """

    torch = _require("torch", "CODI loading requires PyTorch.")
    transformers = _require("transformers", "CODI loading requires transformers.")
    peft = _require("peft", "CODI loading requires peft.")
    safetensors_torch = _require(
        "safetensors.torch", "CODI loading requires safetensors."
    )
    _require_path(BASE_GPT2_DIR / "config.json", "GPT-2 base model")
    checkpoint_path = _require_path(
        CODI_DIR / "model-00001-of-00001.safetensors", "CODI checkpoint"
    )

    device = _device(torch)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(BASE_GPT2_DIR),
        local_files_only=True,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(BASE_GPT2_DIR),
        local_files_only=True,
        torch_dtype=dtype,
    )

    ori_vocab_size = int(base_model.config.vocab_size)
    pad_token_id = ori_vocab_size
    bot_id = ori_vocab_size + 1
    eot_id = ori_vocab_size + 2
    base_model.resize_token_embeddings(ori_vocab_size + 3)

    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
    codi = peft.get_peft_model(base_model, lora_config)

    class OptionACodiModel(torch.nn.Module):
        """Minimal CODI inference wrapper matching the saved checkpoint keys."""

        def __init__(self):
            super().__init__()
            self.codi = codi
            self.model_name = str(BASE_GPT2_DIR)
            self.pad_token_id = pad_token_id
            self.bot_id = bot_id
            self.eot_id = eot_id
            self.dim = int(base_model.config.hidden_size)
            self.num_latent = 6
            self.use_prj = True
            self.prj_no_ln = False
            self.prj = torch.nn.Sequential(
                torch.nn.Dropout(0.0),
                torch.nn.Linear(self.dim, self.dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.dim, self.dim),
            )
            self.prj.add_module("ln", torch.nn.LayerNorm(self.dim))

        def get_embd(self, model, model_name):
            model_name = str(model_name).lower()
            if "gpt2" not in model_name:
                raise NotImplementedError(
                    f"Option-A CODI loader only supports GPT-2 checkpoints, got {model_name!r}"
                )
            try:
                return model.get_base_model().transformer.wte
            except Exception:
                return model.transformer.wte

    model = OptionACodiModel()
    state_dict = safetensors_torch.load_file(str(checkpoint_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    _validate_codi_load_result(load_result)
    if hasattr(model.codi, "tie_weights"):
        model.codi.tie_weights()

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id

    model.to(device=device, dtype=dtype)
    model.eval()
    print(
        "Loaded CODI checkpoint "
        f"missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}, "
        f"dtype={dtype}"
    )
    return model, tokenizer


def _validate_codi_load_result(load_result: Any) -> None:
    """Fail if the inference-critical CODI weights did not load."""

    critical_prefixes = ("codi.", "prj.")
    missing = [
        key
        for key in load_result.missing_keys
        if key.startswith(critical_prefixes)
        and ".lora_A." not in key
        and ".lora_B." not in key
    ]
    if missing:
        preview = ", ".join(missing[:8])
        raise ModelLoadError(
            "CODI checkpoint did not load required inference weights. "
            f"Missing examples: {preview}"
        )

    unexpected = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith(("decoder.", "loss_fct.", "distill_loss_fct."))
    ]
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise ModelLoadError(
            "CODI checkpoint contains unexpected non-decoder weights. "
            f"Unexpected examples: {preview}"
        )


@contextmanager
def _filter_unsupported_from_pretrained_kwargs(transformers: Any, unsupported: set[str]):
    """Temporarily drop kwargs that older GPT-2 loaders do not accept.

    The upstream CODI class passes use_flash_attention_2=False into
    AutoModelForCausalLM.from_pretrained. Some Transformers/GPT-2 combinations
    forward unknown kwargs into GPT2LMHeadModel.__init__, where this fails. This
    keeps the original architecture intact and only removes unsupported kwargs.
    """

    original = transformers.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        for key in unsupported:
            kwargs.pop(key, None)
        return original(*args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original


def _require(module_name: str, message: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelLoadError(
            f"{message} Install the SIM-CoT dependencies before running a real sweep."
        ) from exc


def _require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise ModelLoadError(f"Missing {label}: {path}")
    return path



def _load_module_from_path(module_name: str, path: Path):
    if not path.exists():
        raise ModelLoadError(f"Missing module file: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelLoadError(f"Could not load module spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def _add_to_syspath(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _device(torch):
    requested = os.environ.get("OPTION_A_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_available(torch) -> bool:
    return bool(torch.cuda.is_available())


def _torch_load(torch, path: Path, device: Any):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_coconut_mode(state_dict: dict) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("expainable_llm.") for key in keys):
        return "simcot"
    if any(key.startswith("base_causallm.") for key in keys):
        return "coconut"
    return "base"


def _initialize_coconut_special_embeddings(base_model: Any, tokenizer: Any, token_ids: list[int]) -> None:
    embeddings = base_model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    if target_id is None or target_id < 0:
        target_id = tokenizer.eos_token_id
    for token_id in token_ids:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        lm_head = getattr(base_model, "lm_head", None)
        if lm_head is not None:
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
