# Latent-step classifier design (Option A)

## Goal
Given a prompt x, predict how many latent steps k the model should use before
answering. Input: r(x), a representation of the prompt. Output: an estimate of k.

## Input representation r(x)
- **Decision:** frozen all-MiniLM-L6-v2 encoder (sentence-transformer, 384 dims).
  Not fine-tuned; embeddings are precomputed once.
- **Why:** cheap (runs on CPU), good sentence embeddings with no training, and
  independent of the SIM-CoT backbone (can be computed without the CODI/Coconut
  checkpoints).
- **Alternatives set aside for now:** hand-crafted features (a future baseline), and
  backbone hidden states (more faithful but more expensive and coupled; left as
  future work).

## Problem formulation
- **Decision:** regression over k, with a threshold/rounding step afterwards.
- **Why NOT multiclass:** the k* distribution is heavily imbalanced (in the CODI
  n=100 sweep: 79 examples with k*=1, 10 with k*=2, an almost empty long tail). A
  multiclass model would learn to always predict "1" and be right ~79% of the time
  without learning any useful signal. Also, k is ordinal, not categorical: k=2 sits
  "between" 1 and 3, a structure multiclass ignores and regression respects.
- **To explore:** how to map the continuous prediction to an integer k (simple
  rounding vs. a calibrated threshold), and which loss function (MSE vs. losses that
  penalize underestimating k, which is costlier than overestimating because it leaves
  answers wrong).

## Labels
- k*(x) = the smallest k that reaches the example's best score (from the sweeps).
- Embeddings r(x) and k* labels are joined by the example `id`.

## Open questions
- Handling the imbalance (reweighting? transforming k?).
- Evaluation metric for the classifier (not just error in k, but the final system
  accuracy using the predicted k vs. fixed-k baselines).
- MLP size and architecture.

## Empirical evidence (CODI sweep over the full train set, n=7473, k_max=8)

### Accuracy by k
k=1: 0.408 | k=2: 0.427 | k=3: 0.479 | k=4: 0.579
k=5: 0.637 | k=6: 0.640 | k=7: 0.637 | k=8: 0.631

Accuracy climbs steeply up to k of about 5-6 (+23 points over k=1) and then
saturates: k=7 and k=8 do not improve (they even dip slightly). This confirms the
project premise (an optimal latent budget exists below the maximum) and suggests
that predicting k>6 adds nothing.

### k* distribution (smallest k reaching the best score)
k=1: 4939 (66%) | k=2: 1243 | k=3: 396 | k=4: 446
k=5: 319 | k=6: 75 | k=7: 32 | k=8: 23

Heavily imbalanced toward k=1 (66%), with a very thin tail at high k.

### Design implications
- Plain MSE would bias the regressor toward k of about 1 (where the mass is). The
  imbalance must be handled: reweighting, target transform, or an asymmetric loss.
- Underestimating k is costlier than overestimating: predicting less than needed
  leaves the answer wrong, predicting more only wastes compute. The loss should
  reflect this asymmetry.
- Classes k=6,7,8 (75/32/23 examples) are nearly unlearnable for lack of data. Since
  accuracy saturates at k of about 5-6, consider capping the predicted range (e.g. k
  in {1..5}) instead of {1..8}.
- Outputs change with k in 97.4% of cases, so k has a real effect and learning to
  assign it is a meaningful problem.
