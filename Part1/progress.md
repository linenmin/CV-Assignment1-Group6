# Progress Log — CV Assignment 1

## Session: 2026-04-05

### Completed in this session

**1. Final notebook structure was redefined**
- confirmed that the final graded artifact is the Kaggle notebook
- decided to keep `Part1/ga1_Group_6.ipynb` as the main submission notebook
- merged the teammate branch into `DL`, so the current branch now contains both `Part1/` and `DeepLearning/`

**2. DeepLearning story was inserted into the notebook**
- added a new DL block after Section 4.5
- added:
  - `4.6 From Classical Features to a Deep Feature Baseline`
  - `4.7 Revisiting Preprocessing: Multi-Face-Aware Training Inputs`
  - `4.8 Final Submission Patch for Corrupted Placeholder Images`
  - `5. Publishing Best Results`
- rewrote `6. Discussion` into a short closing section

**3. Narrative consistency was tightened**
- corrected the earlier preprocessing description so it matches the actual HAAR heuristic face-selection logic
- added a bridge sentence in `0.3` to make clear that the early preprocessing is the classical baseline, while the DL part revisits this stage with a stronger pipeline
- rewrote the DL section to avoid overusing internal experiment vocabulary
- reduced the main result chain to:
  - `exp_061 = 0.92731`
  - `exp_088 = 0.98403`
  - `exp_109 = 0.98568`
- kept `exp_110 = 0.98348` only as a negative control in prose, not as a main table entry

**4. Execution entry points were added**
- inserted Kaggle runtime setup code cells after Section 5
- added config / checkpoint handles for the final DL pipeline
- notebook now has a visible execution entry for the final system, but the final inference cells are not complete yet

## Current State

### Notebook status
- [x] Classical part is present and coherent
- [x] DL result narrative is present and much cleaner than before
- [x] The preprocessing story now reads as a two-stage design, not a contradiction
- [ ] `4.7` still needs representative crop comparison figures
- [ ] final Kaggle inference cells still need to be completed
- [ ] notebook has not yet been validated end-to-end on Kaggle

### Main result chain currently presented

| Stage | Experiment | Public score | Role in notebook |
|------|------------|-------------:|------------------|
| Deep baseline | `exp_061` | `0.92731` | first deep feature baseline |
| Main system gain | `exp_088` | `0.98403` | final preprocessing and inference core |
| Final patch | `exp_109` | `0.98568` | narrow corrupted-placeholder fix |

## Next Steps
- add the crop comparison figure in `4.7`
- complete the final inference code cells in Section 5
- define the Kaggle dataset payload for weights, configs, and minimal helper code
- run the notebook in Kaggle and confirm it exports the final submission correctly
