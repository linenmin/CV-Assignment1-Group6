# exp_099 Design: BCE One-vs-Rest Training on Pad18 Rescue Inputs

## Goal

Test whether the current bottleneck comes from the **training task definition** rather than from thresholding or inference calibration.

The current stable inference stack already behaves like a target-only open-set recognizer:

- prototypes are built only for `class1` and `class2`
- `other` is produced by rejection, not by an `other` prototype

But training still treats the task as a unified 3-class classification problem. That is a structural mismatch when `other` is actually a heterogeneous mixture of:

- Michael/Sarah-like faces
- other identities
- weak-quality / fallback / abnormal images

## Proposed Change

Keep the current inference stack fixed and change only the training objective.

- Input protocol:
  - same as `exp_096` / `exp_097`
  - `exp_093_multiface_selected_pad18_rescue_faces_112`
- Backbone / finetuning:
  - same as `exp_096`
  - ViT AdaFace, unfreeze last 2 blocks
- Loss:
  - new `bce_ovr`
  - train two logits only:
    - `is_class1`
    - `is_class2`
  - `other` acts as shared negative for both heads
- Inference:
  - unchanged
  - still use `prototype + neighborhood_aware + threshold=0.55`

## Why This Is The Right Isolation

This is the minimal experiment that directly tests the main hypothesis:

> The remaining failure cases are caused by forcing a heterogeneous `other` set into a single training class.

It avoids leaderboard-specific rules and keeps the report story clean:

- same preprocessing
- same backbone
- same finetuning depth
- same inference stack
- only the target formulation changes

## Expected Outcomes

If the hypothesis is right:

- embeddings for weak `class1/class2` positives should move closer to the correct target manifolds
- some of the known `0` false rejects may cross the existing open-set threshold without changing post-processing

If the submission remains unchanged:

- that would indicate the main bottleneck is deeper than simple task reformulation and may require a different representation or a more explicit hard-negative / open-set training objective
