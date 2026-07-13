#!/usr/bin/env python3
"""
Profile PonderNet GPU utilisation vs per-device batch size.

Run from pondernet/:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python scripts/profile_batch_size.py
"""
import os, sys, time, subprocess, threading, gc

# Must be set before any CUDA init if not already in env.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch
from peft import LoraConfig, TaskType
from src.model import CODI, ModelArguments, TrainingArguments

SEQ_LEN = 256
K       = 6
WARMUP  = 3
MEASURE = 10

# Physical GPU id (nvidia-smi index) - change if your 3090 is at index 0.
PHYS_GPU_ID = os.environ.get("PHYS_GPU_ID", "1")

# ── GPU poller ────────────────────────────────────────────────────────────────
_stop   = threading.Event()
_samps  = []

def _poll(iv=0.2):
    while not _stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 f"--id={PHYS_GPU_ID}",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True)
            u, m = r.stdout.strip().split(", ")
            _samps.append((int(u), int(m)))
        except Exception:
            pass
        time.sleep(iv)

def start_poll():
    _samps.clear(); _stop.clear()
    threading.Thread(target=_poll, daemon=True).start()

def stop_poll():
    _stop.set()
    if not _samps:
        return 0, 0, 0
    utils = [s[0] for s in _samps]
    mems  = [s[1] for s in _samps]
    return max(utils), max(mems), round(sum(utils) / len(utils))

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    ma = ModelArguments(
        model_name_or_path="../models/pretrained/gpt2",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        use_decoder=False,
        decoder_path=None,
        simcot_ckpt=None,   # random weights are fine for profiling
        train=True,
    )

    ta = TrainingArguments(
        output_dir="/tmp/_profile_run",
        pondernet=True,
        max_latent_steps=K,
        pondernet_gamma=0.001,
        pondernet_beta=1.0,
        pondernet_halt_bias_init=-2.0,
        pondernet_geom_mean=5.0,
        use_lora=True,
        use_prj=True,
        prj_dim=768,
        prj_dropout=0.0,
        prj_no_ln=False,
        bf16=True,
        gradient_checkpointing=True,
        print_loss=False,
        max_token_num=SEQ_LEN,
        no_cuda=False,
        local_rank=-1,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=ma.lora_r,
        lora_alpha=ma.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )

    model = CODI(ma, ta, lora_cfg)

    # Move to CUDA; keep Float32 params (Trainer does model.to(bf16) but the inner
    # autocast blocks in model.py were written assuming Float32 outside them).
    # We wrap the entire forward in an outer autocast below, which is equivalent.
    model = model.to(device="cuda")
    if hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
    model.train()
    return model

# ── Synthetic batch matching the real data collator output ───────────────────
# Typical GSM8K lengths (post-tokenisation):
#   encoder (question + eot):  ~55 tokens
#   decoder (eot + eos + ans): ~90 tokens
#   ref     (q + cot + ans):   ~200 tokens
ENC_LEN = 55
DEC_LEN = 90
REF_LEN = 200

def make_batch(bs):
    dev = "cuda"
    enc = torch.randint(100, 50256, (bs, ENC_LEN), device=dev)
    dec = torch.randint(100, 50256, (bs, DEC_LEN), device=dev)
    ref = torch.randint(100, 50256, (bs, REF_LEN), device=dev)
    # ref_labels: question part is -100, rest are token ids
    ref_lbl = ref.clone()
    ref_lbl[:, :ENC_LEN] = -100

    # Positions must be valid indices (forward subtracts 1, then gathers):
    #   model_answer_position should land in [0, DEC_LEN) after -1  → init to DEC_LEN-2
    #   ref_answer_position   should land in [0, REF_LEN) after -1  → init to REF_LEN-20
    ref_ans_pos = torch.full((bs,), REF_LEN - 20, dtype=torch.long, device=dev)
    mod_ans_pos = torch.full((bs,), DEC_LEN - 2,  dtype=torch.long, device=dev)

    return dict(
        encoder_input_ids=enc,
        decoder_input_ids=dec,
        ref_input_ids=ref,
        labels=dec,
        ref_labels=ref_lbl,
        ref_answer_position=ref_ans_pos,
        model_answer_position=mod_ans_pos,
    )

# ── One forward+backward step ─────────────────────────────────────────────────
def step(model, bs):
    batch = make_batch(bs)
    # Outer autocast mirrors what HF Trainer does with bf16=True: heavy matmuls
    # run in BF16 while the model stays in Float32 (mixed precision).
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    out["loss"].backward()
    model.zero_grad(set_to_none=True)

# ── Benchmark one batch size ──────────────────────────────────────────────────
def bench(model, bs):
    for _ in range(WARMUP):
        try:
            step(model, bs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_poll()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        try:
            step(model, bs)
        except torch.cuda.OutOfMemoryError:
            stop_poll()
            torch.cuda.empty_cache()
            return None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / MEASURE
    u_max, mem_smi, u_avg = stop_poll()
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    tput = bs / elapsed  # samples/sec

    return dict(bs=bs, secs=elapsed, tput=tput,
                u_avg=u_avg, u_max=u_max, vram_gb=vram_gb)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...", flush=True)
    model = build_model()
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {torch.cuda.get_device_name(0)}  ({total_vram:.0f} GB VRAM)")
    print(f"Seq : {SEQ_LEN} tokens   K={K} latent steps   BF16 + grad-ckpt\n")

    hdr = f"{'BS':>5}  {'s/step':>7}  {'samp/s':>8}  {'util_avg%':>9}  {'util_max%':>9}  {'VRAM_GB':>9}"
    sep = "-" * len(hdr)
    print(hdr); print(sep)

    best_bs = 16
    for bs in [16, 32, 48, 64, 96, 128, 192, 256, 320]:
        r = bench(model, bs)
        if r is None:
            print(f"{bs:>5}  OOM")
            break
        flag = "  ← ~90% util" if r["u_avg"] >= 85 else ""
        print(f"{bs:>5}  {r['secs']:>7.2f}  {r['tput']:>8.1f}  "
              f"{r['u_avg']:>9}  {r['u_max']:>9}  {r['vram_gb']:>9.2f}{flag}",
              flush=True)
        if r["u_avg"] >= 85:
            best_bs = bs

    print(sep)
    print(f"\nRecommended --per_device_train_batch_size: {best_bs}")
    # Suggest accumulation steps to keep effective batch ~256
    eff_target = 256
    accum = max(1, round(eff_target / best_bs))
    print(f"  gradient_accumulation_steps={accum}  →  effective batch = {best_bs * accum}")
