"""Tests for per-instance truncated-K training helpers (exp-06)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model import build_active_mask


def test_active_mask_shape():
    mask = build_active_mask([2, 4, 1], K_batch_max=4, device="cpu")
    assert mask.shape == (3, 4)
    assert mask.dtype == torch.bool


def test_active_mask_values():
    mask = build_active_mask([2, 4, 1], K_batch_max=4, device="cpu")
    expected = torch.tensor([
        [True, True, False, False],
        [True, True, True,  True ],
        [True, False, False, False],
    ])
    assert torch.equal(mask, expected)


def test_active_mask_all_same_Ki():
    mask = build_active_mask([3, 3, 3], K_batch_max=3, device="cpu")
    assert mask.all()


def test_active_mask_Ki_equals_K_batch_max():
    mask = build_active_mask([4, 2], K_batch_max=4, device="cpu")
    # example 0: all active; example 1: first 2 only
    assert mask[0].all()
    assert mask[1, :2].all()
    assert not mask[1, 2:].any()


def test_masked_halting_dist_sums_to_one():
    """After masking + renorm, each row of p_k still sums to 1."""
    torch.manual_seed(0)
    B, K = 3, 6
    # Simulate a raw halting distribution (all rows sum to 1)
    p_raw = torch.rand(B, K)
    p_raw = p_raw / p_raw.sum(dim=1, keepdim=True)

    active = build_active_mask([2, 4, 1], K_batch_max=K, device="cpu")
    p_masked = p_raw * active.float()
    p_masked = p_masked / p_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)

    assert torch.allclose(p_masked.sum(dim=1), torch.ones(B), atol=1e-6)


def test_masked_halting_dist_zero_beyond_Ki():
    """p_k[i, k] == 0 for all k >= K_i[i] after masking."""
    torch.manual_seed(1)
    B, K = 3, 6
    p_raw = torch.rand(B, K)
    p_raw = p_raw / p_raw.sum(dim=1, keepdim=True)

    K_i_list = [2, 4, 1]
    active = build_active_mask(K_i_list, K_batch_max=K, device="cpu")
    p_masked = p_raw * active.float()
    p_masked = p_masked / p_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)

    for i, Ki in enumerate(K_i_list):
        assert torch.allclose(p_masked[i, Ki:], torch.zeros(K - Ki), atol=1e-8), \
            f"example {i}: mass beyond K_i={Ki} should be zero"


def test_step_losses_masked_beyond_Ki():
    """Step losses beyond K_i[i] are zeroed out by the active mask."""
    B, K = 3, 6
    losses = torch.ones(B, K)
    K_i_list = [2, 4, 1]
    active = build_active_mask(K_i_list, K_batch_max=K, device="cpu")
    masked_losses = losses * active.float()

    for i, Ki in enumerate(K_i_list):
        assert masked_losses[i, :Ki].eq(1.0).all()
        assert masked_losses[i, Ki:].eq(0.0).all()
