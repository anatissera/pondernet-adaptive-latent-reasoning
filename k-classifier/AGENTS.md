# AGENTS.md

## What this project is
Research on **adaptive latent reasoning**. Final goal (Option A): train a lightweight
classifier that, given a prompt, predicts how many latent reasoning steps `k` the model
should use before answering
(input -> classifier -> k -> the model reasons with that k -> answer).

Current stage: **k sweeps** (no classifier yet). The same model is run over the same
examples varying k, to confirm that changing the latent budget changes the
answers/accuracy, and `k_star` is computed (the smallest k reaching each example's best
score).

## Repo structure
- `k-classifier/` is **the real work**. Everything that runs lives here.
  - `src/model_loaders.py` rebuilds the CODI and Coconut wrappers and loads the local
    checkpoints (the SIM-CoT config.json files ship empty).
  - `src/model_runner.py`: `run_model` dispatches by backend (coconut/codi).
  - `src/k_sweep.py` sweeps k=1..k_max, computes k_star, aggregates metrics.
  - `src/metrics.py`: numeric exact match (last number in the output).
  - `scripts/run_k_sweep.py`: experiment CLI.
  - `scripts/smoke_model_load.py`: verifies loading + 1 generation (k=1).
  - `scripts/prepare_gsm8k.py`: downloads GSM8K and converts it to {id,input,gold}.
  - `requirements.txt`: the **real dependencies** (do not use the root pyproject.toml).
- `baselines/`: vendored upstream repos (CODI, Coconut). **Only the model classes from
  `baselines/Coconut/` (coconut.py, utils.py) are imported.** Everything else (scripts,
  test.py, train.py, args/*.yaml) has paths hardcoded for the original author's cluster
  and is NOT run directly.
- Root `README.md`, `main.py`, `pyproject.toml` (name=tpnlp): `uv` scaffolding, ignore
  for this experiment. Note: the pyproject asks for `python >= 3.12` but the VM runs
  3.10, so the pyproject does not represent the real environment.

## Execution environment (VM)
- We run on a **GCP VM with a Tesla T4 GPU (15 GB)**, a Deep Learning image with
  **PyTorch already installed system-wide** (`torch 2.9.1+cu129`, CUDA 12.9).
- The Python binary is **`python3`** (3.10.12). **There is no `python`**; always use
  `python3`.
- The VM is **Spot** (it can shut itself down) and is also **shut down manually at the
  end of each session** to save credit. Do not assume it stays alive between sessions;
  nothing in RAM is persistent.

## Python environment / dependencies
Source of truth for deps: `k-classifier/requirements.txt`
(torch 2.5.1 / transformers 4.46.2 / peft 0.15.2 / accelerate 1.7.0 /
datasets 3.1.0 / safetensors 0.5.3 / numpy 2.1.3).

**Version tension (read before installing):**
- The system ships **torch 2.9.1+cu129** (the right build for the T4), but
  `requirements.txt` pins **torch 2.5.1**. **Do not blindly downgrade torch:** a
  `pip install -r requirements.txt` would drag in torch 2.5.1 and could break the CUDA
  build that already works on the GPU.
- `transformers`, `peft`, and `safetensors` are NOT installed system-wide (verified),
  so that stack still needs installing.

**How the venv is created (method verified on the VM):** an isolated
**`.venv-option-a`** venv with `--system-site-packages` to **reuse the system's
torch 2.9.1** and install the rest WITHOUT reinstalling torch.

> **The venv is NOT versioned, but it DOES live on the persistent disk** (`/dev/sda1`),
> so **it survives a normal stop and a Spot preemption**: after restarting the VM just
> `source .venv-option-a/bin/activate`; no need to recreate it. It is only lost if the
> **VM is deleted/recreated** (new disk); in that case run `scripts/setup_session.sh`
> (see below). See *"Persistence between sessions"* further down.

**Careful: `python3-venv` is not installed** on the VM, so `python3 -m venv` fails for
lack of `ensurepip` and would ask for `sudo apt install python3.10-venv`. To avoid
sudo, the venv is created **`--without-pip`** and the **system pip** is reused (visible
thanks to `--system-site-packages`). pip still installs inside the venv because it is
invoked by the venv's `python3`. This is automated in
**`k-classifier/scripts/setup_session.sh`** (idempotent), but the manual detail is:

```bash
# 1) create the venv without pip, reusing the system torch + pip
python3 -m venv --system-site-packages --without-pip .venv-option-a
source .venv-option-a/bin/activate

# 2) install ALL of requirements.txt EXCEPT the torch==2.5.1 line
#    (do NOT use `pip install -r requirements.txt`: it would drag in torch 2.5.1)
python3 -m pip install \
  transformers==4.46.2 safetensors==0.5.3 peft==0.15.2 accelerate==1.7.0 \
  datasets==3.1.0 numpy==2.1.3 tqdm==4.67.0 PyYAML==6.0.2 Pillow==11.3.0
```

Notes:
- During installation you will see *"Can't uninstall X -- outside environment"*
  warnings (system numpy/Pillow/PyYAML/etc.). They are **benign**: the venv installs
  its own version that shadows the system one on the path; nothing outside the venv is
  touched.
- `transformers`/`peft`/`accelerate` do not pin torch, so they accept the 2.9.1 already
  present. Verified: after installing, `torch.__version__ == 2.9.1+cu129` and
  `torch.cuda.is_available() == True`.

- **Verify torch 2.5.1 is really needed before pinning it.** The Option-A code uses
  APIs stable between torch 2.5 and 2.9 (`AutoModelForCausalLM`,
  `resize_token_embeddings`, `past_key_values`, `safetensors.load_file`, `torch.load`,
  peft LoRA). The real risk is the **transformers** version (4.46.2 is what the
  Coconut/CODI code expects; 4.52+ deprecates the legacy cache `coconut.py` uses), not
  torch. Plan: run the smoke test with the system torch 2.9.1; **only** pin torch 2.5.1
  if something actually breaks.
- Do **not** install `baselines/CODI/requirements.txt` on top (it asks for torch 2.7.1 /
  transformers 4.52.4 and would break compatibility). Option-A reimplements the CODI
  wrapper against transformers 4.46.2, so it needs neither CODI's code nor its deps.
  `baselines/Coconut/requirements.txt` is aligned (torch 2.5.1 / transformers 4.46.2).

## Meaning of k (important)
- **Coconut**: k = latent stages; `k * c_thought` `<|latent|>` tokens are inserted in
  the prompt (between `<|start-latent|>` and `<|end-latent|>`) and `model.generate` is
  called. `c_thought` defaults to 1 in the CLI, 2 in the smoke test / loader (env
  `OPTION_A_C_THOUGHT`).
- **CODI**: k = latent iterations; a manual loop reinjects the last hidden state k
  times (through the `prj` projection) before decoding the answer token by token. No
  special text tokens (uses bot/eot = vocab_size+1/+2).

Both backends enter through the same `run_model(...)` and differ by
`generation_config["backend"]`.

## How to run
1. Data: `python3 k-classifier/scripts/prepare_gsm8k.py` (needs internet/HF).
2. Smoke: `python3 k-classifier/scripts/smoke_model_load.py --backend {codi|coconut}`.
3. Sweep:
   ```bash
   python3 k-classifier/scripts/run_k_sweep.py --backend codi --k-max 8 --n-examples 100 \
     --data k-classifier/data/gsm8k_test_100.jsonl \
     --output k-classifier/results/<name>.jsonl \
     --model-loader src.model_loaders:load_codi
   ```
   (for Coconut: `--model-loader src.model_loaders:load_coconut --c-thought 2`).

## Checkpoints (NOT versioned; `k-classifier/models/` is empty in git)
Downloaded from Hugging Face with `python3 k-classifier/scripts/download_models.py`
(`--model all` by default; or `gpt2|coconut|codi` for one). The script uses
`huggingface_hub.snapshot_download` and leaves each model under `k-classifier/models/`:
- `k-classifier/models/gpt2/`                 <- `openai-community/gpt2`        (GPT-2 base)
- `k-classifier/models/SIM_COT-GPT2-Coconut/` <- `internlm/SIM_COT-GPT2-Coconut` (Coconut)
- `k-classifier/models/SIM_COT-GPT2-CODI/`    <- `internlm/SIM_COT-GPT2-CODI`    (CODI)

`k-classifier/models/` is gitignored but lives on the persistent disk, so the checkpoints
**survive a normal stop** (no need to re-download each session). They only need
re-downloading on a **new VM/disk**. Without them, both loaders raise `ModelLoadError`
at startup.

## Persistence between sessions (what survives a VM shutdown)
The VM is **Spot/preemptible**, but the whole repo (including `.venv-option-a/`,
`k-classifier/models/`, and generated data) lives on the **persistent boot disk**
(`/dev/sda1`, `PERSISTENT-BALANCED`). There is no ephemeral local SSD.
- **Manual shutdown / stop / Spot preemption**: the VM stops but the disk is kept;
  **venv, checkpoints, and data are still there** on restart. Only RAM/processes are
  volatile (hence long sweeps run in `tmux`).
- **Deleting/recreating the VM (new disk)**: everything unversioned is lost (venv +
  `models/` + data). Only then re-bootstrap:
  `bash k-classifier/scripts/setup_session.sh` (recreates venv + installs deps) and
  `python3 k-classifier/scripts/download_models.py` (re-downloads checkpoints).

## Data
- Canonical JSONL format: `{"id", "input", "gold"}`.
- Versioned: only `k-classifier/data/examples.jsonl` (2 test rows).
- `gsm8k_test_100.jsonl` is generated with prepare_gsm8k.py; not in git.

## Git / collaboration
- Repo **shared by 5 people**. Option A work happens on the **`option-a-k-classifier`**
  branch.
- **Do not push directly to `main`.** Changes go to `option-a-k-classifier` or branches
  derived from it.
- Before pushing, sync with the remote branch (several authors touch the same files).

## Known gotchas
- **Limited compute budget (~USD 50 total).** Run the minimum necessary; no extra
  exploratory sweeps.
- **Long sweeps go in `tmux`** to survive Cloud Shell disconnects and the VM's Spot
  nature. Do not launch long runs in the foreground of an interactive session.
- `k-classifier/models/` empty -> both loaders fail until the checkpoints are populated.
- `k-classifier/data/gsm8k_test_100.jsonl` does not exist on disk; it must be generated.
- Broken symlink: `baselines/Coconut/ckpts/gsm_cot` -> cluster path (`/mnt/...`).
- `baselines/CODI/test.py` has an active `pdb.set_trace()` (around line 321) and
  hardcoded `/mnt/...` paths; do not run it directly.
- The versioned `results/*.jsonl|csv` are empty stubs; the numbers in the Option-A
  README's pilot experiment are not reproducible from what is committed.
- Dead helpers in model_loaders.py (`_filter_unsupported_from_pretrained_kwargs`,
  `_load_module_from_path`) are not wired up.

## Conventions
- New code and wrappers: keep the `k-classifier/src` style (type hints, small functions,
  custom errors like `ModelLoadError` / `ModelRunnerError`).
- User-facing text / repo docs: English.
