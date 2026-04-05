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
