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

## v1 results: simple classic classifier (request: start simple)

A deliberately simple first version (scikit-learn classic ML), before any MLP or
target transformation. Note: the rest of this doc frames the problem as
**regression**; this v1 attacks it as **multiclass** (k* in {1..8}) to get a cheap
starting point and a hard baseline. The negative result below in fact reinforces the
imbalance concern already noted.

### Pipeline (3 reproducible scripts, in `k-classifier/scripts/`)
1. `precompute_embeddings.py`: encodes the sweep's `input` field with the frozen
   all-MiniLM-L6-v2 encoder (384 dims, CPU) and caches (id, embedding) in
   `cache/embeddings_minilm_train_full.npz`. Does not recompute if the cache covers
   all ids (~2 min for n=7473 on CPU).
2. `build_dataset.py`: joins embeddings (id -> vector) with the sweep's k* labels
   (id -> k_star) by `example_id`, and makes an internal stratified 80/20 split.
   **The GSM8K test set stays reserved**: this split comes only from the train sweep.
3. `train_classifier.py`: trains a majority baseline + logistic regression + random
   forest (both `class_weight='balanced'`) and reports accuracy, F1-macro, confusion
   matrix, and the comparison against the baseline. Raw metrics in
   `results/classifier_results.json`.

### Data
- n=7473 examples (full CODI train sweep). Split: train n=5978, val n=1495.
- The stratified split preserves the k* distribution in both partitions
  (k=1 ~66.1% in train and val).

### Metrics (internal validation, n=1495)

| Model                             | Accuracy | F1-macro | dAcc vs base | dF1 vs base |
|-----------------------------------|----------|----------|--------------|-------------|
| Majority baseline (k=1)           | 0.6609   | 0.0995   | --           | --          |
| Logistic regression (balanced)    | 0.2100   | 0.1112   | -0.4508      | +0.0118     |
| Random forest (balanced)          | 0.6609   | 0.0995   | +0.0000      | +0.0000     |

Confusion matrices (summary):
- **Random forest**: all the mass falls in the k=1 column (predicts k=1 for all 1495
  val examples). Identical to the baseline despite `class_weight='balanced'`.
- **Logistic regression**: spreads predictions across classes, but with ~0.01-0.09
  precision on the minority ones and k=1 precision/recall dropping to 0.64/0.19.

### Does the classifier learn anything beyond "always predict k=1"?
**Practically no.**
- The RF literally collapses to the baseline (same accuracy and F1-macro).
- The balanced LR raises F1-macro by just +0.012 over the baseline while sinking
  accuracy to 0.21: it is guessing, not discriminating a useful signal.

Conclusion: the frozen MiniLM embedding + simple classic ML **does not capture k\*
signal above the base rate**. An informative negative result. Before moving to the
MLP, attack the cause (the x -> k* signal may be weak in a sentence embedding, and/or
the imbalance dominates): try more faithful representations (backbone hidden state,
Option 3), cap the range to k in {1..5} where there is data, or reframe as ordinal
regression, not just swap models.
