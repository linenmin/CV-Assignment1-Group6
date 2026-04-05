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

## Session: 2026-04-05 (review pass on Sections <= 4.5)

### Completed in this session

**1. A strict read-through of the classical notebook part was completed**
- reviewed the assignment PDF against the notebook content up to and including `4.5`
- checked both the markdown narrative and the visible executed outputs
- cross-checked several claims against the actual helper code in `src/`

**2. A first set of consistency mismatches was identified**
- confirmed that the current notebook is not fully self-consistent yet
- the main issue is no longer model choice, but alignment between:
  - markdown claims
  - visible notebook outputs
  - actual code behavior

### Mismatches currently confirmed

**A. Multiple CV values are reported for the same leaky augmented pipeline**
- Section `4.5` prose says pipeline `[E]` reached `0.9781`
- visible notebook output shows `[E] = 0.9625`
- later final-selection prose says the overall leaky best is `0.9601`
- this creates the impression that the notebook text and outputs were written from different runs

**B. Final training report support does not match the stated training-set size**
- throughout the notebook, the classical training set is described as `80` images
- however, the final printed classification report under model selection shows support summing to `69`
- this needs to be explained explicitly, or the report must be regenerated so the visible output matches the narrative

**C. Preprocessing / filtering story is stronger than the visible evidence**
- the notebook text discusses blur checks, eye occlusion checks, human review, and possible removal of bad samples
- but the actual notebook cells before `4.5` do not show:
  - a quality audit table
  - how many samples were flagged
  - whether any samples were finally removed
- the report therefore reads as if filtering happened, while the visible experiment path still mostly uses `train_X` / `train_y`

**D. PCA section promises outputs that are not visibly delivered**
- the prose says the PCA block includes reconstruction quality at different `k`
- the visible cells currently show eigenfaces and explained variance, but not the promised progressive reconstruction result
- the assignment also asks for visualisation in the first two principal components, while the notebook currently relies on `t-SNE` for PCA features

**E. Some explanatory text does not fully match the implemented details**
- the HAAR description is closer to the real heuristic than before, but the notebook still does not expose the concrete strict vs lenient detection outcomes
- some quality-filtering language implies actual deletion logic, while the visible pipeline does not clearly demonstrate that deletion happened

### Current State After Review

### What is now clear
- the notebook already contains many of the right ideas
- the most urgent remaining work is to make the classical part internally consistent and evidence-backed
- before polishing language, the notebook needs one clean pass to align text, outputs, and code

### Highest-priority repair targets
- [ ] unify the reported CV values for `[E]` and remove stale numbers
- [ ] explain or fix why the final printed support is `69` instead of the expected full classical training size
- [ ] either show the quality-audit / filtering evidence explicitly or weaken the wording so it matches what is actually shown
- [ ] add the missing PCA deliverables or rewrite that section so it does not overclaim

## Session: 2026-04-05 (fresh rerun for Sections 4.3-4.5)

### Completed in this session

**1. The local rerun blockers were fixed before rerunning**
- `load_data()` was made robust to the current local directory layout and future Kaggle dataset packaging
- `GridSearchCV` was guarded against invalid `pca__n_components` values under `5-fold CV`, so impossible settings such as `80` or `100` on a `64`-sample train fold are now removed before search

**2. A fresh end-to-end rerun of the classical model-selection block was completed**
- reran the classical selection chain corresponding to Sections `4.3`, `4.4`, and `4.5`
- the rerun used the current `gpu_env`
- total runtime was about `683 s` (`11.4 min`), which is consistent with the large number of CV fits in the full search space

### Fresh rerun results now treated as the source of truth

**Section 4.3**
- `[A] HOG+PCA+SVM = 0.8625`
- `[B] LBP+SVM = 0.8000`
- `[C] HOG+LBP+PCA+SVM = 0.8625`
- `[D] PixelPCA+SVM = 0.7625`

**Section 4.4**
- `[E] HOG+PCA+SVM (aug) = 0.9594`
- `[F] HOG+LBP+PCA+SVM (aug) = 0.9437`
- `[G] LBP+SVM (aug) = 0.8281`

**Section 4.5**
- `[H] HOG+PCA+SVM (aug-aware) = 0.9125`
- `[I] HOG+LBP+PCA+SVM (aug-aware) = 0.9000`
- best honest model is now clearly `[H]`, not a tied `[H/I]` description

**Final classical selection summary**
- overall leaky best remains `[E]`, but with fresh `CV = 0.9594`
- final honest selection is `[H]` with `CV = 0.9125`
- final train-set report on the original classical training images now shows:
  - accuracy = `1.0000`
  - support total = `80`
  - all three classes have precision / recall / F1 of `1.00`

### What this fresh rerun resolved

**A. The old `69`-sample report is confirmed stale**
- the fresh rerun uses the expected full `80` original training images in the final report
- the earlier `support = 69` output should no longer be treated as valid evidence

**B. The old `[E] = 0.9781` claim is confirmed stale**
- the fresh rerun gives `[E] = 0.9594`
- older notebook prose that still cites `0.9781` is now out of date

**C. The honest-best result is stable but needs numeric synchronization**
- previous prose used `0.9121`
- fresh rerun gives `0.9125`
- the qualitative conclusion is unchanged, but the notebook text should use the fresh value consistently

### Immediate follow-up actions
- [ ] update `progress.md` and `findings.md` so they point to the fresh rerun values
- [ ] rewrite the affected notebook markdown in `4.5` and final model analysis so it matches the fresh rerun
- [ ] later decide whether to re-execute and save the notebook outputs as well, so visible stdout matches the corrected markdown

## Session: 2026-04-05 (saved notebook execution to cell 60)

### Completed in this session

**1. The notebook was re-executed and saved through the classical selection block**
- executed the real notebook sequentially from the start through `cell 60`
- saved after every small batch of executed code cells to avoid losing progress
- created a safety backup before execution:
  - `ga1_Group_6.pre-exec-backup.ipynb`

**2. One missing import was fixed to make the notebook executable**
- execution first failed at the HOG `t-SNE` cell because `TSNE` was used without import
- added `from sklearn.manifold import TSNE` to the notebook cell
- reran from the beginning after this fix

**3. The saved notebook outputs are now fresh**
- `cell 57` now shows:
  - `[E] = 0.9594`
  - `[F] = 0.9531`
  - `[G] = 0.8281`
- `cell 59` now shows:
  - `[H] = 0.9125`
  - `[I] = 0.9125`
  - `Best honest model : [H]`
- `cell 60` now shows:
  - support total = `80`
  - train accuracy = `1.0000`
  - all classes with precision / recall / F1 = `1.00`

### Important observation
- the saved notebook outputs differ slightly from the earlier standalone rerun
- the most important difference is that the saved notebook now shows an honest-score tie between `[H]` and `[I]` at `0.9125`
- therefore the notebook narrative must describe the saved artifact, not the earlier standalone rerun

## Session: 2026-04-05 (submission-path refactor for Kaggle)

### Completed in this session

**1. Hyperparameter search was moved out of the default runtime path**
- `4.3`, `4.4`, and `4.5` now report the offline search results directly in the notebook
- the notebook keeps the selected CV scores and best-parameter dictionaries as report evidence
- the expensive `GridSearchCV` / augmentation-aware search is no longer rerun during the normal submission path

**2. The runnable path now fits only the final selected model once**
- `cell 59` now reuses the chosen honest configuration for `[H]`
- it augments the full training set once and fits the final estimator once
- `cell 60` then performs train-set evaluation and test prediction from that fitted model

**3. Unnecessary augmented feature extraction was removed**
- the notebook no longer recomputes `hog_aug`, `lbp_aug`, or `combined_aug` in the default path
- `cell 55` now only materialises the augmented dataset once and explicitly states that offline-search features are not recomputed

**4. The saved notebook artifact is now aligned with the intended submission design**
- `cell 60` now reports the honest winner as:
  - `[H] HOG+PCA+SVM (aug-aware, honest)`
- the earlier confusing `[H/I]` summary label is gone

### Verification

- re-executed `ga1_Group_6.ipynb` from the start through `cell 60`
- wall-clock runtime for the saved execution was about `43 s`, not `11+ min`
- saved output confirms:
  - offline results are printed in `56/57/59`
  - final selection is `[H]`
  - train accuracy remains `1.0000`
  - support total remains `80`
  - `submission_improved.csv` is produced in the runnable path

### Small follow-up polish

- added a short markdown note at the start of `4.3 / 4.4`
- this note now explicitly explains that the searches were executed offline, but the best CV scores and hyperparameters are still preserved in the notebook as report evidence
- this keeps the contribution of the original model-selection work visible while preserving a fast Kaggle execution path
