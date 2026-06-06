"""Phase 2+3+4 smoke test for PonderNet adaptive halting (Option C).

Self-contained: builds a tiny GPT-2 CODI with --pondernet on a SYNTHETIC batch
(no dependency on the cluster-hardcoded GSM8K-Aug paths) and overfits it for a
few steps. Verifies:
  (1) the original CODI objective still optimizes (loss decreases on a fixed batch),
  (2) the halting head produces lambda_k in (0,1) with shape (B, K),
  (3) per-prefix answer losses have shape (B, K) and are finite,
  (4) with --pondernet OFF the forward path is unchanged (no pondernet_* keys),
  (5) halting distribution p_k has shape (B, K), non-negative, sums to 1,
  (6) p_k is in-graph (gradients flow from it to the halt head),
  (7) L_pondernet (ce_loss key) is finite,
  (8) kl_geom is non-negative and finite,
  (9) halt_head.weight receives non-zero gradients after backward.

Run from SIM-CoT/CODI/:
    python smoke_pondernet.py                 # uses HF "gpt2"
    python smoke_pondernet.py --model /path/to/local/gpt2
"""
import argparse
import torch
from peft import LoraConfig, TaskType
from src.model import CODI, ModelArguments, TrainingArguments


def build_model(model_name, pondernet, device):
    model_args = ModelArguments(
        model_name_or_path=model_name,
        full_precision=True,
        use_decoder=False,   # isolate the new code; L_step decoder tested in later phases
        train=True,
    )
    ta = TrainingArguments(
        output_dir="/tmp/codi_smoke",
        bf16=True,
        num_latent=4,
        use_lora=True,
        use_prj=True,
        prj_dim=768,
        print_loss=False,
        print_ref_model_stats=False,
        remove_eos=True,
        pondernet=pondernet,
        pondernet_halt_bias_init=-2.0,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
    model = CODI(model_args, ta, lora).to(device)
    return model


def make_batch(model, device, B=6, Lq=8, La=5, Lr=12):
    """Construct a valid synthetic batch matching CODI.forward's expectations."""
    V = model.codi.config.vocab_size - 3  # normal-token range (last 3 ids are pad/bot/eot)
    g = torch.Generator().manual_seed(0)
    rand = lambda *s: torch.randint(0, V, s, generator=g)

    encoder_input_ids = torch.cat(
        [rand(B, Lq - 1), torch.full((B, 1), model.bot_id)], dim=1
    )
    decoder_input_ids = torch.cat(
        [torch.full((B, 1), model.eot_id), rand(B, La - 1)], dim=1
    )
    labels = decoder_input_ids.clone()

    ref_input_ids = rand(B, Lr)
    ref_labels = ref_input_ids.clone()
    ref_labels[:, : Lr // 2] = -100  # mask the "question" half

    batch = dict(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        ref_input_ids=ref_input_ids,
        labels=labels,
        encoder_attention_mask=torch.ones(B, Lq, dtype=torch.long),
        ref_attention_mask=torch.ones(B, Lr, dtype=torch.long),
        ref_labels=ref_labels,
        ref_answer_position=torch.full((B,), Lr - 2, dtype=torch.long),
        model_answer_position=torch.full((B,), La - 1, dtype=torch.long),
    )
    return {k: v.to(device) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ---- 1) PonderNet ON: overfit a fixed batch, inspect halting diagnostics ----
    model = build_model(args.model, pondernet=True, device=device)
    model.train()
    batch = make_batch(model, device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    last = None
    for step in range(args.steps):
        opt.zero_grad()
        out = model(**batch)
        loss = out["loss"]
        loss.backward()
        opt.step()
        losses.append(loss.item())
        last = out
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    lam = last["pondernet_lambdas"]          # (B, K)
    sl = last["pondernet_step_losses"]       # (B, K)
    K = model.num_latent
    B = batch["encoder_input_ids"].size(0)

    print("\n--- checks ---")
    ok = True

    drop = losses[0] - losses[-1]
    c1 = drop > 0
    print(f"[{'PASS' if c1 else 'FAIL'}] loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f} (drop={drop:.4f})")
    ok &= c1

    c2 = tuple(lam.shape) == (B, K)
    print(f"[{'PASS' if c2 else 'FAIL'}] lambda shape == ({B},{K}): got {tuple(lam.shape)}")
    ok &= c2

    c3 = bool(((lam > 0) & (lam < 1)).all())
    print(f"[{'PASS' if c3 else 'FAIL'}] lambda in (0,1): min={lam.min().item():.4f} max={lam.max().item():.4f}")
    ok &= c3

    c4 = tuple(sl.shape) == (B, K) and bool(torch.isfinite(sl).all())
    print(f"[{'PASS' if c4 else 'FAIL'}] step_losses shape ({B},{K}) & finite: got {tuple(sl.shape)}")
    ok &= c4

    p = last["pondernet_p"]  # (B, K) — in-graph

    c5 = tuple(p.shape) == (B, K)
    print(f"[{'PASS' if c5 else 'FAIL'}] p_k shape == ({B},{K}): got {tuple(p.shape)}")
    ok &= c5

    row_sums = p.sum(dim=1)
    c6 = bool((p >= 0).all()) and bool(((row_sums - 1.0).abs() < 1e-5).all())
    print(f"[{'PASS' if c6 else 'FAIL'}] p_k >= 0 and sums to 1: min={p.min().item():.6f} row_sums={row_sums.tolist()}")
    ok &= c6

    c7 = p.requires_grad
    print(f"[{'PASS' if c7 else 'FAIL'}] p_k requires_grad (in-graph for Phase 4)")
    ok &= c7

    l_pond = last["ce_loss"]  # L_pondernet stored under ce_loss key in pondernet mode
    c8 = bool(torch.isfinite(torch.as_tensor(l_pond)))
    print(f"[{'PASS' if c8 else 'FAIL'}] L_pondernet finite: {float(l_pond):.4f}")
    ok &= c8

    kl = last["kl_geom"]
    c9 = bool(torch.isfinite(torch.as_tensor(kl))) and float(kl) >= 0.0
    print(f"[{'PASS' if c9 else 'FAIL'}] kl_geom >= 0 and finite: {float(kl):.6f}")
    ok &= c9

    # Check halt_head gets gradient after backward on the last step
    opt.zero_grad()
    out_grad = model(**batch)
    out_grad["loss"].backward()
    halt_grad = model.halt_head.weight.grad
    c10 = halt_grad is not None and bool((halt_grad.abs() > 0).any())
    print(f"[{'PASS' if c10 else 'FAIL'}] halt_head.weight has non-zero grad (L_pondernet backprops through p_k)")
    ok &= c10

    # ---- 2) PonderNet OFF: original path must not expose pondernet_* keys ----
    model_off = build_model(args.model, pondernet=False, device=device)
    model_off.train()
    out_off = model_off(**make_batch(model_off, device))
    c8 = ("pondernet_lambdas" not in out_off) and ("pondernet_p" not in out_off) and (not hasattr(model_off, "halt_head"))
    print(f"[{'PASS' if c8 else 'FAIL'}] pondernet OFF: no halting head / diagnostics")
    ok &= c8
    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ABOVE ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
