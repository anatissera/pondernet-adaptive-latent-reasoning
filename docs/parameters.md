# PonderNet Latent-CoT Training: Parameter Reference

This document covers every CLI flag exposed by `pondernet/src/model.py`, the two
warm-start recipes and their known trap, and the module/loss names that are frozen
by checkpoint convention.

---

## (a) CLI-flag reference

### ModelArguments

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--model_name_or_path` | `"mistralai/Mistral-7B-Instruct-v0.2"` | Path or HF hub ID of the **plain** base language model used as the CODI backbone. Must NOT be a SIM-CoT CODI checkpoint (see Warm-start trap below). |
| `--separate_decoder_name` | `""` | Unused name field for a separate decoder model; kept for compatibility. |
| `--lora_r` | `128` | LoRA rank applied to the backbone via PEFT. |
| `--lora_dropout` | `0.05` | Dropout probability inside each LoRA adapter layer. |
| `--full_precision` | `True` | Load backbone in full precision (fp16/bf16). When `False`, loads in 4-bit NF4 via BitsAndBytes. |
| `--use_decoder` | `False` | Attach an auxiliary decoder module for per-step reconstruction (`explain_loss`). |
| `--decoder_path` | `None` | Path to a standalone GPT-2 checkpoint to use as the auxiliary decoder (used with `--use_decoder`). Decoder-only warm-start: the auxiliary decoder is re-loaded from the same path as the backbone (`model_name_or_path`). |
| `--soft_weight` | `None` | Multiplier for soft/distillation loss (legacy; mostly unused in PonderNet runs). |
| `--save_ablation` | `False` | Save ablation evaluation results. Only activated when explicitly passed on the command line. |
| `--train` | `True` | Whether the model is being initialised for training (`True`) or inference (`False`). |
| `--lora_init` | `False` | When `True`, use zero/Gaussian LoRA init. When `False`, load adapters from LoftQ in HF hub. |
| `--token` | `None` | HuggingFace access token for private model repositories (e.g., `meta-llama`). |
| `--adapter_name_or_path` | `None` | Path to a saved LoRA adapter; used when resuming from a checkpoint or during evaluation. |
| `--lora_alpha` | `16` | LoRA alpha (scaling). Not required for LoftQ; used for QLoRA. |
| `--ckpt_dir` | `None` | Checkpoint directory used during inference mode. |
| `--simcot_ckpt` | `None` | Path to a SIM-CoT CODI `.safetensors` for **full-model warm-start** (backbone + LoRA + decoder + `prj`). Loaded via `load_state_dict(strict=False)` after the model is assembled; only the PonderNet halt head is freshly initialised. Do NOT point `--model_name_or_path` at this file. |

---

### DataArguments

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--data_name` | `None` | HuggingFace dataset name / local path identifier for training data. |
| `--debug_data` | `False` | Enable a tiny debug dataset to quickly verify the training pipeline end-to-end. |
| `--batch_size` | `1` | Batch size for inference (not the trainer batch size; see `--per_device_train_batch_size`). |
| `--data_path` | `""` | Local path to a training/eval JSON file. If empty, data is loaded from HuggingFace. |
| `--results_dir` | `"../results"` | Directory for evaluation output JSON files (relative to repo root, outside the module). |
| `--max_train_samples` | `None` | If set, truncate the training set to this many samples (useful for fast ablations). |

---

### TrainingArguments

The following fields extend `transformers.TrainingArguments` with CODI/PonderNet-specific
controls. Standard HuggingFace training flags (learning rate, epochs, etc.) are also
available but not listed here.

| Flag | Default | What it controls |
|------|---------|-----------------|
| `--cache_dir` | `None` | HuggingFace model/tokenizer cache directory. |
| `--optim` | `"adamw_torch"` | Optimizer identifier passed to the HF Trainer. |
| `--model_max_length` | `28000` | Maximum sequence length; sequences are right-padded and may be truncated. |
| `--restore_from` | `""` | Path to a `.safetensors` checkpoint to restore from for fine-tuning (full state dict). |
| `--per_device_train_batch_size` | `1` | Training batch size per GPU. |
| `--per_device_eval_batch_size` | `1` | Evaluation batch size per GPU. |
| `--expt_name` | `"default"` | Experiment name, used to label output directories and logs. |
| `--icot_train_path` | `""` | Deprecated; unused. |
| `--max_latent_steps` | `5` | Hard upper bound K_max on latent steps (adaptive count ≤ this; renamed from the legacy `--num_latent`). |
| `--use_lora` | `True` | Whether to wrap the backbone with LoRA adapters via PEFT. |
| `--greedy` | `False` | Use greedy decoding during inference (vs. sampling). |
| `--exp_mode` | `False` | Use a partial data subset for debugging purposes. |
| `--exp_data_num` | `10000` | Number of samples used when `--exp_mode` is enabled. |
| `--use_prj` | `False` | Attach a projection module (`prj`) after the backbone hidden states before feeding the next latent step. |
| `--prj_dim` | `2048` | Hidden dimension of the projection module's intermediate layer. |
| `--prj_dropout` | `0.0` | Dropout ratio inside the projection module. |
| `--prj_no_ln` | `False` | Remove the LayerNorm at the end of the projection module. |
| `--distill_loss_div_std` | `False` | Normalise the distillation loss by the standard deviation of the teacher hidden states. |
| `--distill_loss_type` | `"smooth_l1"` | Distillation loss function: `"smooth_l1"` (SmoothL1) or `"l2"` (MSE). |
| `--distill_loss_factor` | `1.0` | Scalar multiplier applied to the distillation loss. |
| `--explain_loss_factor` | `1.0` | Scalar multiplier applied to the auxiliary decoder reconstruction loss (`explain_loss`). |
| `--ref_loss_factor` | `1.0` | Scalar multiplier applied to the reference/teacher CE loss. |
| `--inf_latent_iterations` | `1` | Number of latent iterations during inference (legacy). |
| `--inf_num_iterations` | `5` | Number of full forward passes to run during inference. |
| `--remove_eos` | `False` | Do not append `<eos>` as a delimiter when splitting question/answer. |
| `--print_ref_model_stats` | `False` | Print probability statistics for the teacher model's target token predictions. |
| `--include_last_cot` | `False` | Include the final CoT step in the training data. |
| `--fix_attn_mask` | `False` | Enable a corrected attention mask that prevents attending to latent padding positions. |
| `--log_full` | `False` | Log all individual loss components to the training output. |
| `--print_loss` | `True` | Print loss values to stdout during training steps. |
| `--max_token_num` | `1000` | Discard examples longer than this many tokens to avoid OOM. |
| `--pondernet` | `False` | Enable PonderNet-style adaptive latent halting. When `False`, runs the original fixed-K SIM-CoT path. |
| `--pondernet_halt_bias_init` | `-2.0` | Initial bias of the halting head `halt_head`; a negative value keeps early λ_k small so the model uses more steps at the start of training. |
| `--pondernet_beta` | `1.0` | Weight of the auxiliary decoder reconstruction loss (`explain_loss` / L_step) in PonderNet mode. |
| `--pondernet_gamma` | `0.01` | Weight of the KL-geometric regulariser (`kl_geom`) in PonderNet mode; penalises using more steps than the geometric prior. |
| `--pondernet_geom_mean` | `3.0` | Mean number of steps for the truncated geometric prior used in KL_geom. Larger value relaxes the compute-efficiency pressure. |
| `--pondernet_inf_threshold` | `0.5` | Cumulative halting probability threshold for inference early-stopping: halt when Σ_k p_k > this value. |

---

## (b) Warm-start recipes

### Full-model warm-start (default in train script)

```bash
--simcot_ckpt <path/to/simcot_codi.safetensors>
```

After the full `CODI` wrapper is assembled (backbone + LoRA adapters + `decoder` +
`prj`), `train.py` calls:

```python
missing, unexpected = model.load_state_dict(sd, strict=False)
```

This loads the entire SIM-CoT checkpoint into the already-assembled module. The
**only** parameters left newly-initialised are the PonderNet halt head (`halt_head.*`).
Everything else — backbone weights, LoRA adapters, decoder, projection — comes from
the checkpoint.

### Decoder-only warm-start

```bash
--decoder_path <path/to/standalone_gpt2>   # and set SIMCOT_CKPT="" in the train script
```

Set `SIMCOT_CKPT=""` (the train-script environment variable) so that the
`simcot_ckpt` branch is skipped entirely. The backbone is loaded cold from
`--model_name_or_path`; only the auxiliary decoder is warm-started from the
standalone GPT-2 checkpoint at `--decoder_path`.

### The trap: never point `--model_name_or_path` at a SIM-CoT checkpoint

`--model_name_or_path` (equivalently the `GPT2_PATH` env var used by the train
script) must be a **plain GPT-2 (or other base LM) checkpoint**, never a saved
SIM-CoT `CODI` wrapper.

A SIM-CoT checkpoint stores its backbone weights under keys like
`codi.base_model.model.transformer.*`. When HuggingFace's `AutoModelForCausalLM`
loads it as a bare `GPT2LMHeadModel`, it cannot match those namespaced keys and
**silently random-initialises the entire backbone** — producing a model that trains
from scratch while appearing to warm-start. This is exactly the failure mode that
caused `simcot_joint_ep40` to train to garbage (see `memory/project_simcot_warmstart_fix.md`).

### Sentinel checks in `train.py` (lines ~184–209)

After the `load_state_dict` call, `train.py` enforces three invariants and raises
`RuntimeError` if any is violated:

1. **No unexpected keys** — every tensor in the checkpoint must map onto a model
   parameter. A non-empty `unexpected` list means a namespace mismatch (the same
   failure mode as the silent random-init trap described above).
2. **Only `halt_head.*` missing** — the only parameters allowed to be absent from
   the checkpoint (i.e. newly-initialised) are those whose name starts with
   `halt_head`. Any other missing key triggers an error.
3. **Core sentinels present** — explicitly checks that
   `codi.base_model.model.transformer.wte.weight`, `decoder.lm_head.weight`, and
   `prj.1.weight` were all loaded (i.e., are not in `missing`), confirming that
   the backbone, decoder, and projector weights actually landed.

A success warning logs how many tensors were loaded and lists the newly-initialised
halt-head parameters.

---

## (c) Glossary — names kept as-is (checkpoint-bound)

The following `nn.Module` attribute names appear directly in saved `state_dict` keys.
They are deliberately not renamed so that existing checkpoints remain loadable.

| Name | `state_dict` prefix | What it is |
|------|---------------------|------------|
| `codi` | `codi.*` | The LoRA-wrapped backbone module that runs the latent recurrence loop. At each step it consumes the current latent embedding and produces the next one. Its keys include both the frozen base-model weights (`codi.base_model.model.*`) and the trainable LoRA adapter weights (`codi.base_model.model.*.lora_A.*`, etc.). |
| `decoder` | `decoder.*` | Auxiliary GPT-2 that receives each latent embedding concatenated with a CoT step and is trained to reconstruct that step via cross-entropy — the `explain_loss` / L_step signal. Only present when `--use_decoder` is set. |
| `prj` | `prj.*` | Optional projection module applied to the latent embedding after each backbone pass before feeding it back in. Architecture: `Dropout → Linear(dim, prj_dim) → GELU → Linear(prj_dim, dim) → LayerNorm`. Enabled with `--use_prj`. |
| `halt_head` | `halt_head.*` | `nn.Linear(hidden_dim, 1)` that maps each latent hidden state h_k to a scalar halting logit; `sigmoid` of this logit gives the conditional halting probability λ_k at step k. This is the **only** component that is freshly initialised during a full-model warm-start. Its bias is initialised to `--pondernet_halt_bias_init` (default −2.0). |
| `pj_in` | `pj_in.*` | Input projection that projects the auxiliary decoder's input token embeddings (in hidden-size space) from backbone hidden size to decoder hidden size. When sizes match it is an `nn.Identity`; otherwise it is `nn.Linear(backbone_hidden, decoder_hidden)`. Only present when `--decoder_path` is set. |
| `pj_out` | `pj_out.*` | Output projection from the auxiliary decoder's vocabulary logits back to the backbone's vocabulary space. When sizes match it is an `nn.Identity`; otherwise it is a low-rank projector `LowRankProjector(decoder_vocab, backbone_vocab, rank=decoder_vocab//4)`. Only present when `--decoder_path` is set. |

### Loss terms

| Name | Description |
|------|-------------|
| `l_pondernet` | Expected answer cross-entropy under the learned halting distribution: Σ_k p_k · L_ans^(k), averaged over the batch. In PonderNet mode this replaces the fixed-K `ce_loss_total`. |
| `kl_geom` | KL divergence from the halting distribution p_k to a truncated geometric prior with mean `--pondernet_geom_mean`. Weighted by `--pondernet_gamma`; penalises using more latent steps than necessary. |
| `explain_loss` | Auxiliary decoder reconstruction cross-entropy (L_step): how well `decoder` can predict each CoT step from the corresponding latent embedding. Weighted by `--explain_loss_factor` (standard mode) or `--pondernet_beta` (PonderNet mode). |
| `distill_loss` | Hidden-state distillation loss between student (latent path) and teacher (reference CoT path) at the answer position, averaged over all transformer layers. Weighted by `--distill_loss_factor`. |
| `ref_ce_loss` | Cross-entropy loss for the reference/teacher task (the model predicting the full CoT sequence). Weighted by `--ref_loss_factor`. |
