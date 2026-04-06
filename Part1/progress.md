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

## Session: 2026-04-05 (PCA deliverables aligned with assignment)

### Completed in this session

**1. Added the two missing PCA visualisation deliverables**
- added a reconstruction plot that shows one face rebuilt with progressively more eigenfaces
- added a direct `PC1 / PC2` scatter plot for the PCA feature space
- kept `t-SNE` as a supplementary visualisation instead of using it as a replacement

**2. Extended the reusable plotting helpers in `src/visualization.py`**
- added `plot_reconstruction_progression(...)`
- added `plot_pca_component_scatter(...)`
- both helpers now return figure handles so they are easier to validate and reuse

**3. Updated the notebook narrative around PCA**
- `1.2.1` now explicitly matches the reconstruction requirement from the assignment
- `1.2.2` now explains that `PC1 / PC2` is the primary PCA-space visualisation
- `1.2.3` discussion was tightened so it refers to the new reconstruction and `PC1 / PC2` plots rather than over-relying on `t-SNE`

### Verification

- added and ran lightweight visualization tests for the two new helpers
- reran `ga1_Group_6.ipynb` through `cell 60`
- saved notebook output now includes:
  - three PCA visuals in `cell 34` (eigenfaces, explained variance, progressive reconstruction)
  - two PCA-space visuals in `cell 36` (`PC1 / PC2` scatter + `t-SNE`)
- the fast submission path still executes successfully in about `40 s`

## Session: 2026-04-05 (reporting clean-up for Sections 4.3-4.8)

### Completed in this session

**1. Removed implementation-facing debug output from the classical search report**
- `cell 56` no longer prints `removed invalid pca__n_components`
- the saved notebook now shows only the final offline search results that matter for grading

**2. Tightened the tie explanation for `[H]` vs `[I]`**
- the previous wording implied that some hidden detailed output broke the tie in favour of `[H]`
- this was replaced with a simpler and defensible explanation: both honest CV scores are equal, so `[H]` is kept because it is the simpler pipeline and there is no evidence that adding LBP helps

**3. Added the missing context for `exp_061`**
- `4.7` now defines `exp_061` before using it as the baseline reference
- this makes the transition from the first deep baseline to the multi-face input revision readable without requiring the reader to infer what `exp_061` was

**4. Clarified why the 3-image placeholder patch maps to class 2**
- `4.8` now states that the three overridden test IDs match a known class-2 `IMAGE NOT FOUND` placeholder pattern seen in labeled training data
- it also states explicitly that this was not treated as a general rule for all corrupted images

### Verification

- reran `ga1_Group_6.ipynb` through `cell 60`
- confirmed that the saved `cell 56` output no longer contains the invalid-grid debug lines
- confirmed that the updated markdown for `4.5`, `4.7`, and `4.8` is present in the saved notebook

## Session: 2026-04-05 (full notebook polish + Kaggle submission)

### Completed in this session

**1. Full notebook review and style overhaul**
- all medium and small issues from the professor review pass were addressed
- Section 1.2 (PCA): replaced 7-section textbook prose with a flowchart (`figures/pca_pipeline.png`) + 5 compact paragraphs
- Section 3.0 / 3.1: compressed from ~450 words to ~100 words total
- Section 4.0: added theoretical expected CV baseline (~0.34) as reference
- Section 4.1: renamed to "Baseline: Pixel-PCA + SVM" to distinguish from HOG-based pipelines
- Section 4.2: renamed to "Feature Preparation" to accurately reflect its role
- Section 4.6: added forward reference for `exp_088` so it is not undefined when first cited
- Section 6 Discussion: expanded from 3 paragraphs to 5, covering HOG vs PCA qualitative comparison, why classical performance plateaued, why DL improved, and what we would do with more time
- Cell 0: removed template-style language from Section 0.1
- Cell 12: fixed blank HAAR parameter names (`scaleFactor`, `minNeighbors`)
- Cell 28/30: fixed t-SNE description typos and softened overconfident "clear separations" claim
- Cell 1: TODO red text still present — name fields not yet filled in (blocked on teammates confirming names)

**2. Kaggle dataset packaged and submitted**
- packaged `src/`, `figures/`, `dl_package/` into `group6-dl-final.zip` (871 MB)
- uploaded to Kaggle as dataset `enminlin/group6-dl-final`
- actual Kaggle dataset path confirmed as `/kaggle/input/datasets/enminlin/group6-dl-final/`
- Cell 0 and Cell 67 (`DL_ROOT`) updated to use the confirmed path
- Cell 69 (`Load data`): added symlink `data/ → DL_ROOT/data/` to resolve relative image paths in CSVs
- DL dependencies installed via Kaggle Dependency Manager: `lightning`, `albumentations`, `timm`, `torchmetrics`, `fvcore`
- notebook ran successfully end-to-end on Kaggle; `submission.csv` produced

### Current State
- [x] Notebook runs end-to-end on Kaggle
- [x] submission.csv produced and submitted to competition
- [x] All severe and medium review issues addressed
- [x] Cell 1 student names filled in (done manually)
- [x] Second review pass completed (see below)

## Session: 2026-04-06 (second professor review pass)

### Completed in this session

**1. Second full review against assignment PDF**
- re-read entire notebook as a strict grader
- identified 8 remaining issues across severe/medium/small categories
- items from original template (";)", "Let's plot", accuracy disclaimer) confirmed acceptable and left unchanged

**2. Issues resolved**

- **Cell 5**: BGR→RGB long explanation compressed to one sentence; detail moved to start of Section 0.3
- **Cell 10**: "Dataset Analysis & Strategy Formulation" (3 verbose paragraphs) replaced with 2-sentence factual summary; avoids duplication with Section 0.2.1
- **Cell 12**: Added BGR→RGB bridging sentence at start of Section 0.3
- **Cell 19**: Removed "flagged / human review" language that had no visible evidence; replaced with reference to the visible blurry crops in the face plots and the decision to retain all samples
- **Cell 20**: Removed fictitious semi-automated flagging pipeline description; rewritten to describe the design rationale for each quality metric and the actual decision made (no deletion)
- **Cell 30**: Fixed grammar error in first HOG property bullet ("that undergoes" → correct phrasing)
- **Cell 37**: Removed "& Academic Reflection" from Section 1.2.3 title
- **Cells 54/57**: Split merged "4.3 & 4.4" header into two independent sections with their own titles and motivations; new 4.4 header explains the leaky protocol and why it is not used for final selection

### Current State
- [x] Notebook runs end-to-end on Kaggle
- [x] submission.csv produced and submitted
- [x] All severe, medium, and small review issues addressed across two review passes
- [x] Student names filled in
- [x] progress.md up to date
