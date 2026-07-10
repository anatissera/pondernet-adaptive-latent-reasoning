"""Model execution helpers for k sweeps.

This module keeps the public runner small while making the backend-specific
meaning of k explicit:
- coconut: k latent stages -> k * c_thought <|latent|> tokens in the prompt.
- codi: k latent iterations in the hidden-state loop before answer decoding.
"""

from __future__ import annotations

from typing import Any


DEFAULT_START_LATENT_TOKEN = "<|start-latent|>"
DEFAULT_LATENT_TOKEN = "<|latent|>"
DEFAULT_END_LATENT_TOKEN = "<|end-latent|>"


class ModelRunnerError(RuntimeError):
    """Raised when a backend cannot be executed with the provided objects."""


def run_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    generation_config: dict,
) -> str:
    """Run a model with exactly the requested latent budget.

    Args:
        model: A loaded Coconut/SIM-CoT or CODI-compatible model wrapper.
        tokenizer: The matching Hugging Face tokenizer.
        prompt: Input question or prompt.
        k: Latent stages for Coconut/SIM-CoT, latent iterations for CODI.
        generation_config: Must include ``backend`` with value ``coconut`` or
            ``codi``. Other keys are backend-specific generation options.
    """

    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not generation_config:
        raise ValueError("generation_config is required")

    backend = generation_config.get("backend")
    if backend == "coconut":
        return run_coconut(model, tokenizer, prompt, k, generation_config)
    if backend == "codi":
        return run_codi(model, tokenizer, prompt, k, generation_config)

    raise ValueError(
        "generation_config['backend'] must be one of {'coconut', 'codi'}"
    )


def run_coconut(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    generation_config: dict,
) -> str:
    """Run Coconut/SIM-CoT by inserting k stages of latent tokens."""

    torch = _require_torch()
    _require_attrs(model, ["generate"], "coconut model")
    _require_attrs(tokenizer, ["encode", "decode", "convert_tokens_to_ids"], "tokenizer")

    c_thought = int(generation_config.get("c_thought", 1))
    if c_thought < 1:
        raise ValueError(f"c_thought must be >= 1, got {c_thought}")

    start_token = generation_config.get(
        "start_latent_token", DEFAULT_START_LATENT_TOKEN
    )
    latent_token = generation_config.get("latent_token", DEFAULT_LATENT_TOKEN)
    end_token = generation_config.get("end_latent_token", DEFAULT_END_LATENT_TOKEN)

    input_ids = build_coconut_latent_input_ids(
        tokenizer=tokenizer,
        prompt=prompt,
        k=k,
        c_thought=c_thought,
        start_latent_token=start_token,
        latent_token=latent_token,
        end_latent_token=end_token,
    )

    device = _infer_device(model, generation_config)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, device=device)

    generate_kwargs = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask,
        "max_new_tokens": int(generation_config.get("max_new_tokens", 128)),
        "synced_gpus": bool(generation_config.get("synced_gpus", False)),
    }

    with torch.no_grad():
        _set_eval(model)
        output_ids = model.generate(**generate_kwargs)

    generated_ids = output_ids[0][input_tensor.shape[-1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _clean_coconut_prediction(
        decoded,
        latent_tokens=[start_token, latent_token, end_token],
    )


def build_coconut_latent_input_ids(
    tokenizer: Any,
    prompt: str,
    k: int,
    c_thought: int,
    start_latent_token: str = DEFAULT_START_LATENT_TOKEN,
    latent_token: str = DEFAULT_LATENT_TOKEN,
    end_latent_token: str = DEFAULT_END_LATENT_TOKEN,
) -> list[int]:
    """Build the Coconut/SIM-CoT prompt whose latent token count realizes k."""

    question_ids = tokenizer.encode(prompt + "\n", add_special_tokens=True)
    start_id = _token_to_id(tokenizer, start_latent_token)
    latent_id = _token_to_id(tokenizer, latent_token)
    end_id = _token_to_id(tokenizer, end_latent_token)

    n_latent_tokens = k * c_thought
    return question_ids + [start_id] + [latent_id] * n_latent_tokens + [end_id]


def _clean_coconut_prediction(text: str, latent_tokens: list[str]) -> str:
    """Remove latent markers if the model emits them in the generated suffix."""

    for token in latent_tokens:
        text = text.replace(token, "")
    return " ".join(text.split()).strip()


def run_codi(
    model: Any,
    tokenizer: Any,
    prompt: str,
    k: int,
    generation_config: dict,
) -> str:
    """Run CODI by executing k latent iterations before decoding the answer."""

    torch = _require_torch()
    _require_attrs(model, ["codi", "bot_id", "eot_id", "get_embd"], "CODI model")
    _require_attrs(tokenizer, ["__call__", "decode"], "tokenizer")

    codi = model.codi
    _require_attrs(codi, ["__call__"], "model.codi")

    device = _infer_device(model, generation_config)
    remove_eos = bool(generation_config.get("remove_eos", True))
    max_new_tokens = int(generation_config.get("max_new_tokens", 128))
    greedy = bool(generation_config.get("greedy", True))

    batch = tokenizer([prompt], return_tensors="pt", padding="longest")
    batch = _move_batch_to_device(batch, device)

    if remove_eos:
        bot_ids = [int(model.bot_id)]
    else:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ModelRunnerError("CODI remove_eos=False requires tokenizer.eos_token_id")
        bot_ids = [int(eos_id), int(model.bot_id)]

    bot_tensor = torch.tensor([bot_ids], dtype=torch.long, device=device)
    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
    batch["attention_mask"] = torch.cat(
        (batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1
    )

    with torch.no_grad():
        _set_eval(model)
        outputs = codi(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=batch["attention_mask"],
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        latent_embd = _maybe_project_codi_latent(model, latent_embd, generation_config)

        for _ in range(k):
            outputs = codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            latent_embd = _maybe_project_codi_latent(
                model, latent_embd, generation_config
            )

        eot_ids = [int(model.eot_id)]
        if not remove_eos:
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                eot_ids.append(int(eos_id))

        embedding = model.get_embd(codi, getattr(model, "model_name", ""))
        output_embd = embedding(torch.tensor(eot_ids, dtype=torch.long, device=device))
        output_embd = output_embd.unsqueeze(0)

        generated_ids: list[int] = []
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            out = codi(
                inputs_embeds=output_embd,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            vocab_size = getattr(getattr(codi, "config", None), "vocab_size", None)
            logits = out.logits[:, -1, : vocab_size - 1 if vocab_size else None]

            if greedy:
                next_token = torch.argmax(logits, dim=-1)
            else:
                next_token = _sample_next_token(logits, generation_config)

            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            generated_ids.append(token_id)
            output_embd = embedding(next_token).unsqueeze(1).to(device)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _token_to_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        raise ModelRunnerError(
            f"Tokenizer does not know required token {token!r}. "
            "Load the tokenizer with the latent special tokens added."
        )
    return int(token_id)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelRunnerError(
            "PyTorch is required for real Coconut/CODI inference."
        ) from exc
    return torch


def _require_attrs(obj: Any, attrs: list[str], label: str) -> None:
    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if missing:
        raise ModelRunnerError(f"{label} is missing required attributes: {missing}")


def _infer_device(model: Any, generation_config: dict):
    torch = _require_torch()
    device_name = generation_config.get("device")
    if device_name:
        return torch.device(device_name)

    try:
        return next(model.parameters()).device
    except Exception:
        pass

    codi = getattr(model, "codi", None)
    codi_device = getattr(codi, "device", None)
    if codi_device is not None:
        return torch.device(codi_device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_eval(model: Any) -> None:
    if hasattr(model, "eval"):
        model.eval()


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    for key, value in list(batch.items()):
        if hasattr(value, "to"):
            batch[key] = value.to(device)
    return batch


def _maybe_project_codi_latent(model: Any, latent_embd: Any, generation_config: dict):
    use_prj = bool(generation_config.get("use_prj", hasattr(model, "prj")))
    if use_prj and hasattr(model, "prj"):
        return model.prj(latent_embd)
    return latent_embd


def _sample_next_token(logits: Any, generation_config: dict):
    torch = _require_torch()
    temperature = float(generation_config.get("temperature", 1.0))
    top_k = int(generation_config.get("top_k", 0))
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k > 0:
        top_values, _ = torch.topk(logits, top_k, dim=-1)
        cutoff = top_values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, -float("inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _strip_prompt_prefix(text: str, prompt: str) -> str:
    normalized_prompt = prompt.strip()
    if normalized_prompt and text.strip().startswith(normalized_prompt):
        return text.strip()[len(normalized_prompt) :]
    return text
