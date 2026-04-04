# exp_101 Design: BCE One-vs-Rest + Target-Only Supervised Contrastive Loss

## Goal

Push the `exp_099/100` direction one step further without changing the inference stack.

`exp_099` showed that changing the training task from 3-class softmax to one-vs-rest is the first training-side change that actually altered test decisions. But the gain was still tiny:

- `exp_099`: only `id=1540`
- `exp_100`: only `id=1455` and `id=1540`

This suggests the task formulation is closer to the real bottleneck, but still not strong enough to reshape the embedding geometry for the weak-evidence target cases.

## Proposed Change

Keep the `exp_099` one-vs-rest formulation and add a target-only supervised contrastive auxiliary loss.

- Main loss:
  - `BCE one-vs-rest`
  - two logits:
    - `is_class1`
    - `is_class2`
- Auxiliary loss:
  - target-only `SupCon`
  - only `class1/class2` form positive pairs
  - `other` is not forced into a single cluster
  - `other` still acts as implicit negative context

## Why This Is Better Than 3-Class Contrastive Training

The point is not to cluster all labels.

That would recreate the same bad assumption as 3-class softmax:

- treating heterogeneous `other` as a single coherent class

Instead, this auxiliary objective only strengthens the geometry that the downstream open-set inference actually uses:

- tighter `class1`
- tighter `class2`
- cleaner separation between the two targets and the rest

## Controlled Variables

Everything else stays fixed relative to `exp_099`:

- same input protocol: `exp_093`
- same backbone and finetuning depth
- same `50` epochs
- same `val_loss` early stopping
- same `prototype + neighborhood_aware + fixed055` inference

So the experiment isolates one question:

> If the one-vs-rest task is already more correct, does explicitly tightening the target embedding geometry rescue more target-like test cases?

## Initial Hyperparameters

- `supcon_weight = 0.1`
- `supcon_temperature = 0.1`

The initial choice is intentionally conservative:

- strong enough to shape embedding geometry
- weak enough not to destabilize the already-working BCE objective

## Expected Readout

Success would look like:

- more changes than `exp_099`
- especially movement on user-identified weak-evidence target cases

Failure would look like:

- still only a couple of marginal `0 -> 2` releases
- or no additional changes at all

That would imply the bottleneck is deeper than task formulation plus geometric auxiliary loss.
