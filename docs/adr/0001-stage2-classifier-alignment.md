# ADR-0001: Stage 2 Classifier Alignment (Pairwise Discrepancy)

## Status

Accepted

## Date

2026-04-14

## Context

MDFAN Stage 1 aligns source/target feature distributions via GRL-based adversarial training and MMD.
In multi-source settings, source-specific classifiers can still learn slightly different decision boundaries.
This can hurt target performance even when feature alignment is good.

We want an optional Stage 2 loss term that encourages source-specific classifiers to agree on *target* predictions.

Options considered:

- Pairwise discrepancy on probabilities (L1/L2)
- Symmetric KL divergence on probabilities
- Maximum Classifier Discrepancy (MCD) (requires an explicit discrepancy maximization/minimization schedule)

## Decision

Implement an optional classifier alignment loss as a simple pairwise discrepancy between the softmax outputs of all source-specific classifiers on target images.

- Default alignment loss type: **L1** (stable, easy to tune)
- Also implemented: **L2** and **symmetric KL**
  - KL uses clamped/renormalized probabilities to avoid `-inf`/NaNs when probabilities hit 0.
- The loss is computed inside `MDFAN.forward_train()` only when requested, and the training script weights it using `--lambda_align` (default `0.0` keeps behavior unchanged).

MCD and other more complex Stage 2 strategies are deferred.

## Consequences

- Adds a low-overhead, differentiable Stage 2 option without changing default training behavior.
- Provides a safe KL implementation, but L1 remains the recommended default for initial experiments.
- Future work can introduce MCD/pseudo-labeling if needed, but will require additional scheduling/ablation support.
