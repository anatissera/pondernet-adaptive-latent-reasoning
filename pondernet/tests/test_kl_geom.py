"""Unit tests for the (per-instance) truncated-geometric halting prior.

Experiment 05 (simcot-pondernet-adaptive-prior): the halting prior's mean becomes
per-example (geom_mean_i = #expr-1) instead of a single global scalar. These tests pin
the pure prior/KL helpers so the per-example path is correct and the global path is
unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.model import (
    truncated_geometric_prior,
    kl_to_truncated_geometric,
    count_real_steps,
    adaptive_prior_mean,
)


def _reference_global_q(geom_mean, K):
    """Replicates the ORIGINAL inline global geometric prior (regression anchor)."""
    g = 1.0 / geom_mean
    k_idx = torch.arange(K, dtype=torch.float32)
    q = (1.0 - g) ** k_idx * g
    q[-1] = (1.0 - g) ** (K - 1)
    return q / q.sum()


def test_scalar_prior_matches_original_formula():
    K = 6
    q = truncated_geometric_prior(3.0, K)
    assert q.shape == (K,)
    assert torch.allclose(q, _reference_global_q(3.0, K), atol=1e-6)


def test_per_example_prior_shape_and_normalization():
    means = torch.tensor([2.0, 4.0, 6.0])
    K = 6
    q = truncated_geometric_prior(means, K)
    assert q.shape == (3, K)
    assert torch.allclose(q.sum(dim=1), torch.ones(3), atol=1e-6)


def test_per_example_rows_match_scalar_build():
    means = [2.0, 4.0, 6.0]
    K = 6
    q = truncated_geometric_prior(torch.tensor(means), K)
    for i, m in enumerate(means):
        assert torch.allclose(q[i], truncated_geometric_prior(m, K), atol=1e-6)


def test_smaller_mean_puts_more_mass_early():
    K = 6
    q = truncated_geometric_prior(torch.tensor([2.0, 5.0]), K)
    # mean=2 concentrates earlier than mean=5 -> larger first-step probability
    assert q[0, 0] > q[1, 0]


def test_kl_zero_when_p_equals_per_row_prior():
    means = torch.tensor([2.0, 5.0])
    K = 6
    q = truncated_geometric_prior(means, K)
    kl = kl_to_truncated_geometric(q, means)  # p == its own prior -> 0
    assert kl.shape == (2,)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-5)


PAD = 128256


def test_count_real_steps_counts_non_pad_entries():
    # get_steps pads each example to a fixed slot count with [pad_id] sentinels;
    # the real-step count is the number of non-pad slots.
    steps_list = [
        [[1, 2], [3, 4], [PAD], [PAD]],   # 2 real steps
        [[5], [6], [7], [PAD]],            # 3 real steps
        [[8], [9], [10], [11]],            # 4 real steps
    ]
    assert count_real_steps(steps_list, pad_id=PAD) == [2, 3, 4]


def test_count_real_steps_handles_all_pad():
    # a degenerate example with no reasoning steps -> 0 (caller clamps to >=1)
    steps_list = [[[PAD], [PAD], [PAD]]]
    assert count_real_steps(steps_list, pad_id=PAD) == [0]


def test_count_real_steps_all_real():
    steps_list = [[[1], [2], [3]]]
    assert count_real_steps(steps_list, pad_id=PAD) == [3]


def test_adaptive_prior_mean_affine_values():
    # geom_mean = scale*n_i + offset, default scale=1, offset=1.5
    m = adaptive_prior_mean([0, 1, 2, 3], scale=1.0, offset=1.5, K=6)
    assert torch.allclose(m, torch.tensor([1.5, 2.5, 3.5, 4.5]), atol=1e-6)


def test_adaptive_prior_mean_clamps_to_K():
    # large n_i clamps at K; everything stays within [offset, K]
    m = adaptive_prior_mean([5, 8, 20], scale=1.0, offset=1.5, K=6)
    assert torch.allclose(m, torch.tensor([6.0, 6.0, 6.0]), atol=1e-6)


def test_adaptive_prior_mean_never_degenerate():
    # the whole point: even n_i=0 yields mean > 1 -> non-degenerate prior (no zero collapse)
    m = adaptive_prior_mean([0, 1], scale=1.0, offset=1.5, K=6)
    assert (m > 1.0).all()
    q0 = truncated_geometric_prior(m[0].item(), 6)
    assert (q0 > 0).all()  # identity map (mean=1) would give [1,0,0,0,0,0]


def test_adaptive_prior_mean_monotonic():
    m = adaptive_prior_mean([0, 1, 2, 3, 4, 5, 6], scale=1.0, offset=1.5, K=6)
    assert torch.all(m[1:] >= m[:-1])


def test_kl_global_scalar_path_matches_reference():
    torch.manual_seed(0)
    p = torch.rand(4, 6)
    p = p / p.sum(dim=1, keepdim=True)
    kl = kl_to_truncated_geometric(p, 3.0)
    q = _reference_global_q(3.0, 6)
    log_q = torch.log(q.clamp_min(1e-8)).unsqueeze(0)
    expected = (p * (torch.log(p.clamp_min(1e-8)) - log_q)).sum(dim=1)
    assert kl.shape == (4,)
    assert torch.allclose(kl, expected, atol=1e-6)
