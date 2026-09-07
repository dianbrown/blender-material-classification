# ITRI626 rubric audit

**Audit scope:** repository and final report artifacts inspected on 30 August
2026 against `ITRI626_Project_New.md`. This is an evidence audit, not a
guarantee of an examiner's mark. The completed `report.docx`/`report.pdf` are
now included; slides are still absent.

## Executive verdict

The technical work and the completed report are a strong, unusually
reproducible proof-of-concept and credible GitHub-showcase material after a
small repository hygiene pass. The remaining material weaknesses are source
asset provenance/licensing, unresolved student/group/date fields, absent
slides, and portability/legal hygiene in tracked metadata. The report itself
appropriately says not to publish on Kaggle yet: `DATASET.md` and the report
state that licenses and attribution have not been normalized and that generated
renders may inherit source restrictions.

**Post-report written-report estimate (30 August 2026): 18.1/20** based on the
current 14-page PDF, 23 embedded figures, 9 numbered substantive tables, exact setup
tables, cleaning audit, discussion, appendices and references. This is an
estimate with uncertainty (roughly ±1 mark), not a promise of full marks. The
report's own placeholders and unresolved licensing/access evidence can still
cost marks if submitted unchanged.

**Oral artifact readiness:** approximately **7.8/8** for content that can be
demonstrated, with the remaining 2 marks (delivery, participation and live
questions) not assessable from files. A prepared group could therefore be in
the **7.8–9.8/10** range; this is not a prediction of delivery marks.

**GitHub verdict:** showcase-worthy as a research/teaching benchmark once
placeholders are filled, absolute local paths are made portable, the course
brief is handled appropriately, and a standard code licence decision is made.
**Kaggle verdict:** NO-GO until per-asset permission and attribution are
documented and verified.

## Evidence inventory

The strongest evidence is distributed across the following files:

- Rubric and submission contract: `ITRI626_Project_New.md`
- Overview, method, results, limitations and reproduction: `README.md`
- Dataset contract, split policy and licensing warning: `DATASET.md`
- Intended use, limitations and ResNet-50 metrics: `MODEL_CARD.md`
- Authoritative material/render records: `metadata/material_manifest.csv`,
  `metadata/material_manifest_split.csv`, `metadata/render_manifest.csv`
- Counts and generation QA: `reports/dataset_generation/final_dataset_report.md`,
  `reports/dataset_generation/material_audit_report.md`,
  `reports/results/dataset_summary.csv`,
  `reports/results/dataset_validation.json`
- Model comparison and selection: `reports/results/model_comparison.csv`,
  `reports/results/model_comparison.md`,
  `reports/results/recommended_model.json`
- Required test metrics: `reports/results/{custom_cnn,resnet50,convnext_tiny,vit_b16}/test/metrics.json`
- Grouped CV: `reports/results/cross_validation_results.csv` and
  `reports/results/cross_validation_results_aggregate.csv`
- Lock and environment: `reports/results/methodology_lock.json`,
  `reports/results/environment.json`
- Reusable implementation: `src/material_classifier/`, `scripts/`, and
  `configs/*.yaml`
- Executed notebooks: `notebooks/01_dataset_eda.ipynb` through
  `notebooks/04_error_analysis.ipynb`
- Figure catalogue: `reports/REPORT_ASSETS.md` and `reports/figures/`
- Automated checks: `tests/` (42 tests passed during this audit) and Ruff
  (`.venv/Scripts/ruff.exe check src scripts tests`: all checks passed).

## Post-report artifact verification (30 August 2026)

The final artifacts were inspected without retraining, inference, full-test
reruns, commits or pushes:

- `report.pdf`: **14 pages**, readable extracted text, contents and appendices.
- `report.docx`: **23 embedded inline figures** and 18 Word table objects,
  including **9 numbered substantive tables** (Tables 1–9; the remaining
  objects are contents/layout tables).
- Section 2 includes the exact class definitions, a before-versus-after table
  (3,150 rendered/accepted; zero removed/corrupt/failed/uncertain), automated
  QA scope and the explicit absence of a claimed manual relabelling review.
- Sections 3–4 and Tables 4–6 match the resolved YAML settings: 224 input,
  ImageNet normalization, train-only crop/flip/ColorJitter, AdamW, smoothing
  0.10, warmup/cosine, AMP, clipping 1.0, 40-epoch ceiling, patience 8,
  model-specific batch sizes/LRs, seed 42, RTX 4060 and grouped CV/test lock.
- Sections 5–8 contain comparison, required metrics, error analysis,
  limitations, a no-results-claimed Optuna proposal and conditional Kaggle
  guidance. The report embeds figures 1–23 and points to
  `reports/REPORT_ASSETS.md` for the larger catalogue.
- The reference section contains architecture, optimization, calibration,
  explainability, UMAP, PyTorch, scikit-learn and Blender sources. The exported
  PDF now displays references numbered 1–15. RenderSynth and individual PBR
  asset source URLs or licences are still not supplied.

## Mark estimate against the written-report rubric (20 marks)

| Criterion | Max | Evidence-backed estimate | What is demonstrated | Missing or weak evidence / mark risk |
|---|---:|---:|---|---|
| Problem definition and motivation | 2 | **2.0** | The 14-page report's Sections 1 and 9 define the seven-way task, research question, scope, practical value and class-selection rationale; `README.md` and `MODEL_CARD.md` corroborate it. | Group/student/date fields remain blank in the report cover. This is mainly a submission-completeness risk rather than a problem-definition gap. |
| Dataset selection and preparation | 4 | **3.0** | Report Section 2 and Tables 1–3 now give class definitions/reasons, 3,150 512×512 RGB images, 105 materials, exact before/after counts (3,150/3,150; zero removed), automated QA, material-level grouping, split counts, ethics and a candid licensing warning. Supporting evidence is in `DATASET.md`, `reports/dataset_generation/final_dataset_report.md`, `reports/dataset_generation/material_audit_report.md`, `reports/results/dataset_validation.json` and `reports/figures/dataset/`. | Source URLs, access links, licence/usage conditions and attribution remain unresolved; no explicit human label-review log exists (the report correctly does not claim one). Absolute `C:/Users/...` paths remain in tracked manifests/results. This is still the largest rubric vulnerability. |
| Model selection and methodology | 4 | **3.8** | Report Section 3 and Tables 4–5 document four meaningful families, pretrained weights, staged freezing/unfreezing, exact model roles and seven-class heads. Configs and implementation are in `configs/*.yaml` and `src/material_classifier/models.py`; references now cover the architectures and methods. | The CustomCNN has no grouped-CV result; this is disclosed as N/A. The report should retain that caveat and not imply equal CV coverage. |
| Experimental setup and reproducibility | 3 | **3.0** | Report Section 3.2/4 and Table 6 give exact augmentation and rationale, train-only application, preprocessing, optimizer, schedule, seed, hardware, fairness controls, grouped CV and test lock. Appendix A gives commands; configs, environment and lock hashes provide machine evidence. | A clean-machine reproduction has not been demonstrated in this audit. A scheduler-order warning remains in a failure-path unit test, not the headline experiment. |
| Results and evaluation | 4 | **3.9** | Report Sections 5–6, Tables 7–8 and Figures 6–23 provide the required comparison, TEST metrics, per-class metrics, confusion matrices, curves, CV, ROC/PR, calibration, errors, embeddings, explainability and efficiency. The PDF explicitly distinguishes validation/CV from locked TEST evidence. | The PDF's main detailed per-class table is ResNet-50; full machine-readable metrics for all four models remain in each model's `test/metrics.json`. Keep TEST labels and class order visible when editing. |
| Discussion, error analysis and conclusion | 2 | **1.7** | Report Sections 5.2, 6–9 explain metal/ceramic errors, validation-to-TEST shift, CV/test ranking reversal, efficiency trade-offs, limitations, Kaggle conditions and a no-results-claimed Optuna Version-2 plan. Figures 15–17 and 23 support the narrative. | The proposed improvements are not new evidence; preserve that wording. More explicit causal validation would require future experiments. |
| Report quality and references | 1 | **0.7** | `report.pdf` is 14 pages with 23 embedded figures, 9 numbered substantive tables, contents, appendices, captions and a reference section containing academic/official sources numbered 1–15. `report.docx` is the editable source. | Cover still has group/student/date placeholders; no slides are present. Check the lecturer's preferred citation style, and do not republish the untracked course brief publicly without permission. |
| **Total** | **20** | **18.1** | Strong, now submission-shaped evidence with a candid account of limitations. | Full marks remain examiner-dependent; unresolved provenance, placeholders and packaging prevent a defensible 20/20 claim. |

### Why the TEST result is not itself a full-mark failure

The locked TEST comparison is methodologically useful and honest:

| Model | TEST accuracy | TEST macro-F1 | Grouped-CV macro-F1 |
|---|---:|---:|---:|
| CustomCNN | 0.498 | 0.490 | N/A (not run) |
| ResNet-50 | **0.698** | **0.683** | 0.717 ± 0.109 |
| ConvNeXt-Tiny | 0.612 | 0.569 | **0.787 ± 0.113** |
| ViT-B/16 | 0.600 | 0.562 | 0.745 ± 0.103 |

The validation-to-TEST gap is material (ResNet validation accuracy 0.843 vs
TEST 0.698; train accuracy is approximately 1.0). This supports a discussion
of unseen-material/domain shift and overfitting rather than a claim of
production readiness. The equal accuracy and balanced accuracy are expected
because the locked TEST split is class-balanced. ResNet-50 is defensible as
the final recommendation under the artifact's composite rule, but the report
must acknowledge that ConvNeXt has the higher development-CV mean and that
five folds have substantial SD.

## Oral presentation readiness (10 marks)

| Criterion | Max | Artifact readiness | Evidence and presentation advice |
|---|---:|---:|---|
| Problem and project overview | 2 | **2.0/2** | The report's Sections 1 and 9, plus `README.md`, give the problem, seven classes, motivation and scope. Open with why synthetic material recognition is useful and what it does not claim. |
| Dataset and methodology | 2 | **1.9/2** | The report's Sections 2–4, Tables 1–6 and Figures 1–9 cover dataset QA, grouped split, four models, augmentation and transfer learning. Be prepared to answer the missing licence/provenance question honestly. |
| Results and visual communication | 2 | **2.0/2** | The report embeds 23 selected figures; dashboard, normalized confusion matrix, training curves, per-class and error figures are available. Use a small number of legible figures rather than showing every asset. |
| Critical understanding and discussion | 2 | **1.9/2** | Report Sections 5.2–9 directly discuss metal/ceramic failures, CV/test discrepancy, calibration, limitations and future work. Speakers still need to rehearse concise explanations. |
| Delivery, participation and questions | 2 | **Not assessable** | No file can establish timing, equal participation, confidence or live answers. All three members should understand every stage, as required by the rubric. |

**Content subtotal:** approximately **7.8/8**. **Overall oral estimate:**
7.8–9.8/10 depending on delivery and questions.

## High-priority remediation checklist

1. **Complete submission fields and slides.** Replace the group-member,
   student-number and submission-date placeholders in `report.docx`/`report.pdf`
   and create the required presentation slides. Recheck citation style against
   the lecturer's preferred format (reference numbering is now 1–15).
2. **Resolve provenance before any public dataset release.** For every one of
   the 105 source materials, add source URL/provider, asset name, licence,
   attribution requirement, access date and redistribution permission to a
   registry. Check RenderSynth and Blender terms too. Do not upload source
   textures or derived renders to Kaggle until every licence permits it.
3. **Document data preparation honestly.** The report now has Table 2 with raw
   and final counts, but retain the explicit statement that there was no human
   relabelling review. If a manual review is later performed, add its protocol
   and log; do not imply one occurred.
4. **Make repository metadata portable and legally clear.** Replace tracked
   `C:/Users/...` paths in manifests/results with relative paths or a documented
   placeholder convention, decide on a standard code `LICENSE`, and do not
   publicly republish the course brief `ITRI626_Project_New.md` without
   permission.
5. **Add citations.** Cite the source asset pages/licences, Blender/EEVEE and
   RenderSynth, ImageNet, ResNet, ConvNeXt, ViT, PyTorch/torchvision and any
   Grad-CAM/UMAP methods used. Ensure claims in the report have nearby
   citations rather than relying only on README links.
6. **Explain augmentation and fairness.** State why crop/flip/color jitter
   are physically plausible for the rendered sphere, confirm validation/test
   receive no augmentation, and explain why architecture-specific batch sizes
   and discriminative LRs are reasonable.
7. **Prepare slides and rehearsal.** Show one dataset contact sheet, split
   diagram, architecture/method slide, main comparison table, recommended
   confusion matrix, 2–3 errors, limitations and conclusion. Rehearse answers
   for every member.

## Medium- and low-priority improvements

### Medium priority

- Add a compact raw-vs-final cleaning table and class-definition rationale to
  `DATASET.md` and the report.
- Explain the CV/test ranking reversal and report mean ± SD conservatively;
  do not select or tune on the sealed TEST set.
- Add confidence intervals or a clearly labelled descriptive uncertainty note
  for the single 420-image TEST set. Keep the existing five-fold SD.
- Include a small real-world or separately licensed external render set only
  as a new, clearly labelled generalization experiment; it must not replace or
  alter the locked TEST result.
- Consider a later development-only experiment for targeted metal/ceramic
  augmentation or hyperparameter search. Any chosen configuration requires a
  fresh untouched test set for a new headline claim.

### Low priority

- Add a lightweight inference demo/GitHub screenshot after licensing and
  report work are complete.
- Add a dataset version identifier and checksums for a future release archive.
- Make the scheduler warning-free in the synthetic OOM/failure-path test,
  while preserving the passing behavior and sealed artifacts.

## Kaggle publication decision

**Not publishable as-is.** The repository's own `DATASET.md` and Section 8 of
`report.pdf` say “licenses and attribution have not yet been normalized” and
“generated renders inherit any applicable restrictions.” There is no resolved
public dataset access URL. A Kaggle upload would be responsible only after a
licence audit confirms redistribution of the derived PNG renders, all required
attribution is included, and the upload excludes source files/checkpoints that
are not redistributable. If any asset forbids derivatives or public hosting,
remove it or release only a manifest and regeneration instructions. The
step-by-step `kaggle.md` guide is useful future-work documentation, not proof
of permission or publication. The Kaggle description should state that this is
a synthetic Blender EEVEE benchmark, give the grouped split policy, generation
seed, classes, limitations and the actual applicable licence.

## Likely examiner questions and defensible answers

**Why split by `material_id` instead of randomly by image?** All 30 renders of
one base material share texture identity. A random image split would leak
near-duplicates across sets; `metadata/material_manifest_split.csv` and the
verification checks keep each material in exactly one split.

**Why use synthetic renders?** They provide controlled, reproducible variation
in camera, lighting and PBR maps, making a material-recognition benchmark
possible without claiming that synthetic performance transfers to photographs.

**Why are there four models?** CustomCNN is a from-scratch capacity baseline;
ResNet-50, ConvNeXt-Tiny and ViT-B/16 represent residual CNN, modern CNN and
transformer families. The rubric requires at least three families.

**Why recommend ResNet when ConvNeXt has higher grouped-CV F1?** ResNet has
the strongest locked TEST accuracy and macro-F1 and a smaller/faster model
than ConvNeXt/ViT. The CV result is reported transparently (ConvNeXt mean
higher, with SD); n=5 is small, so this is a cautious engineering choice,
not proof that ResNet is universally superior.

**Why is ROC-AUC high while accuracy/F1 is moderate?** ROC-AUC measures ranking
of class probabilities across thresholds; the reported hard-label metrics use
the argmax decision. Calibration and confusion matrices show that good ranking
does not eliminate class-boundary errors, especially metal and ceramic.

**What is the main failure mode?** ResNet-50 TEST F1 is strongest for concrete
(0.974), wood (0.846), fabric/leather (about 0.789), and weakest for metal
(0.373) and ceramic (0.420). Similar reflectance/texture cues and unseen base
materials are plausible causes.

**Was the test set used for tuning?** The lock artifact records the protocol;
grouped CV uses development data only and the sealed TEST cache is evaluated
once per model. Future Optuna/augmentation work must use development data and
a new untouched test set for any revised headline result.

**What are the ethical and licensing issues?** No human or sensitive records
are used, but public redistribution still depends on the PBR providers'
licences and attribution terms. Those terms are currently incomplete, so the
dataset is not yet suitable for Kaggle publication.

**Is this a real-world material detector?** No. It is a synthetic Blender/EEVEE
benchmark and research/teaching artifact. Synthetic-to-real generalisation has
not been established, as stated in `MODEL_CARD.md`.

## Final honest verdict

The repository and completed report now have the technical depth expected of a
high-quality ITRI626 project: leakage-safe material-level splits, four model
families, staged transfer learning, grouped CV, sealed TEST evaluation,
extensive metrics and visual analysis, reusable code, tests, references and
reproducibility metadata. It is **GitHub-showcase worthy as a research
benchmark**, once the cover fields, portability, course-brief handling and
code-licence choice are cleaned up. Keep the README and model card's
limitations visible.

It is not yet a full-mark guarantee or a public Kaggle dataset. The remaining
submission actions are slides, names/date, reference-format checking and
metadata hygiene; the remaining public-release blocker is per-asset source
URL/licence/access permission. These changes do not require touching sealed
TEST artifacts. Accuracy tuning can be a subsequent development-only
experiment, but must not overwrite or reinterpret the locked headline result.
