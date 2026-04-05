# Findings & Decisions — CV Assignment 1

## Core Submission Constraint
- the final graded artifact is the Kaggle notebook
- this means the notebook must not only explain the work, it must also run the final submission pipeline
- earlier experiments may be documented rather than re-run, but the final pipeline must be executable

## Current Notebook Strategy

### What remains in the notebook body
- the classical feature pipeline remains as the course baseline
- the DL section is added after Section 4.5 as the competition-oriented improvement stage
- the final notebook now treats the work as a two-stage story:
  - classical baseline
  - competition-stage deep learning upgrade

### What must not be done
- do not rewrite `0.2/0.3` as if the whole notebook always used the final `multiface` preprocessing
- do not turn the DL section into a long log of failed experiments
- do not present internal experiment labels as if they were standard CV terminology without explanation

## Main Technical Findings

### 1. The strongest gain did not come from a more exotic loss
The key performance jump in the DL part came from better face selection and cleaner training inputs, not from switching to more complicated losses or fusion tricks.

Main evidence:

| Experiment | Summary | Public score |
|-----------|---------|-------------:|
| `exp_061` | ViT deep baseline with earlier heuristic face crop + open-set scoring | `0.92731` |
| `exp_088` | stronger multi-face-aware selected crop + same general inference family | `0.98403` |

### 2. The notebook should present `exp_061 -> exp_088 -> exp_109` as the primary chain
This is the shortest result chain that still explains the final system clearly.

Recommended roles:
- `exp_061`: first deep feature baseline
- `exp_088`: main system gain
- `exp_109`: final narrow corrupted-image fix

### 3. `exp_110` is not a main result
`exp_110` is useful only as a negative control:
- `exp_109 = 0.98568`
- `exp_110 = 0.98348`

Conclusion:
- keep `exp_110` in one sentence
- do not put it in the main result table

### 4. The old preprocessing description had a factual mismatch
Earlier notebook text said that when multiple faces are detected, the pipeline simply takes the first one. This did not match the actual HAAR preprocessing code, which uses a heuristic score based on area, centrality, sharpness, and filtering logic.

Resolution:
- the notebook wording was corrected

### 5. The DL section must explicitly connect back to Section 0.2.1
The data audit in Section 0.2.1 already identified:
- multi-face proximity
- corruption
- in-the-wild quality issues

Therefore the DL section should not introduce these issues as if they appear for the first time. The correct narrative is:
- these problems were already observed in the data audit
- the classical pipeline addressed them in a baseline way
- the final DL system revisited the same stage with a stronger preprocessing strategy

## Notebook Writing Decisions

| Decision | Reason |
|----------|--------|
| rename `4.6` to emphasize the transition from classical features to a deep baseline | reduces the abrupt jump from Section 4.5 |
| rewrite `4.7` as a preprocessing-centered system improvement | makes the main gain easier to understand |
| reserve a figure slot in `4.7` for weak crop vs selected crop examples | the claim about unreliable crops needs visual support |
| replace internal wording like `neighborhood-aware` with clearer phrasing in the report body | reduces unexplained internal jargon |

## Open Evidence Still Needed
- representative crop comparison figure for Section 4.7
- final runnable Kaggle inference cells
- final Kaggle dataset payload definition

## Classical Fresh Rerun Decision

### The fresh rerun is now the authoritative reference for Sections 4.3-4.5
A full local rerun of the classical model-selection chain was completed after fixing two blockers:
- invalid PCA dimensions inside `GridSearchCV`
- fragile data-path discovery when running from `Part1/`

This rerun should now be treated as the authoritative reference for the classical results in the notebook body.

### Fresh classical results

| Pipeline | Fresh CV |
|----------|---------:|
| `[A]` HOG+PCA+SVM | `0.8625` |
| `[B]` LBP+SVM | `0.8000` |
| `[C]` HOG+LBP+PCA+SVM | `0.8625` |
| `[D]` PixelPCA+SVM | `0.7625` |
| `[E]` HOG+PCA+SVM (aug, leaky) | `0.9594` |
| `[F]` HOG+LBP+PCA+SVM (aug, leaky) | `0.9437` |
| `[G]` LBP+SVM (aug, leaky) | `0.8281` |
| `[H]` HOG+PCA+SVM (aug-aware, honest) | `0.9125` |
| `[I]` HOG+LBP+PCA+SVM (aug-aware, honest) | `0.9000` |

### Final classical-selection interpretation
- the highest leaky score is still `[E]`, but it is `0.9594`, not `0.9781`
- the best honest model is `[H]`, not a generic `[H/I]` label
- the honest model should be described using:
  - `pca__n_components = 80`
  - `C = 100`
  - `gamma = scale`

### Important writing implication
The final train-set report from the fresh rerun is:
- accuracy = `1.0000`
- support = `80`
- all classes have `1.00` precision / recall / F1

This must be described carefully:
- it is a training-set fit result
- it is not the primary evidence for generalisation
- the honest generalisation estimate still comes from augmentation-aware CV, namely `0.9125`

### Required notebook text updates
- replace stale `[E]` values such as `0.9781` and `0.9601`
- replace stale honest-best value `0.9121`
- remove the old narrative tied to the stale `69`-sample report
- rewrite the final model analysis so it does not over-interpret the train accuracy

## Saved Notebook Execution Note

### The saved notebook outputs differ slightly from the earlier standalone rerun
After re-executing the actual notebook and saving outputs through `cell 60`, the saved notebook now shows:
- `[F] = 0.9531` instead of the earlier standalone `0.9437`
- `[H] = 0.9125`
- `[I] = 0.9125`

This means the saved notebook should now be read as showing an honest-score tie between the two augmentation-aware pipelines, even though the detailed search output still prints `[H]` as the best honest model line.

### Practical implication
- the notebook is now saved with fresh outputs
- the final write-up should describe the honest result as `0.9125`, with `[H]` and `[I]` tied in the saved notebook outputs
