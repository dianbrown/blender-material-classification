# Blender Material Classification

Seven-class classification of synthetic Blender PBR renders with source-family
separation, locked model selection and traceable measured evidence. **V3 is
complete: the preselected primary achieved 80.16% clean holdout accuracy.**

[Final PDF](reports/v3/report.pdf) ? [Report Markdown](reports/v3/report.md) ?
[Executed results notebook](notebooks/05_v3_results.ipynb) ?
[Model card](MODEL_CARD.md) ? [Dataset card](DATASET.md)

## Completed results

All four final models were locked before prediction on the same **630 fresh
images, 21 source groups, 90 images per class**. The primary was selected by
its development Macro F1, not its holdout ranking.

| Locked pipeline | Fresh accuracy | Macro F1 | Role |
| --- | ---: | ---: | --- |
| Frozen DINOv2 + ConvNeXt fusion | **80.16%** | **0.7952** | Preselected primary |
| Frozen ConvNeXt control | 75.56% | 0.7566 | Predeclared control |
| Fixed ResNet-50 | 77.62% | 0.7665 | Fixed assessment baseline |
| Fine-tuned DINOv2 | **86.83%** | **0.8631** | Exploratory comparison; highest observed score |

Primary 95% source-group bootstrap accuracy CI: **66.03-93.65%**. DINOv2 CI:
**76.98-95.87%**. DINOv2 minus primary is +6.67 percentage points (95% paired CI
**-2.06 to 16.19**). All three saved fresh paired accuracy intervals include
zero, so superiority is not established. The **80% target is point accuracy**,
not a lower confidence bound or a guarantee across conditions. Primary JPEG
quality-30 accuracy falls to **59.05%**.

Sources: [fresh outputs](reports/results/v3/fresh),
[model choice](reports/results/v3/model_choice.json),
[checkpoint lock](reports/results/v3/final_model_lock.json), and
[final execution record](reports/results/v3/final_execution_summary.json).
The lock at 09:51:05 UTC preceded first fresh prediction at 09:54:06 UTC on
27 October 2026. No test-winner substitution occurred.

## Quickstart: saved checkpoints

Use Python 3.13 and the recorded V3 environment. Frozen inference also requires
`timm`, `safetensors`, `huggingface-hub`, `joblib` and `threadpoolctl` beyond
base project dependencies. Exact executed versions are in
[environment.json](reports/results/v3/environment.json) and cache/checkpoint
metadata; the older dependency lock does not describe every V3 dependency.
PDF rebuilding needs ReportLab. Notebook setup uses `requirements-notebook.txt`.

Checkpoints and pretrained caches are local artifacts, **not included in Git**.
Obtain trusted files and verify hashes in `final_model_lock.json`. Frozen
inference needs adjacent metadata and pinned pretrained caches. Neural `.pt`
exports have standalone weights and adjacent `model.json`. Do not load untrusted
joblib files. There is no public checkpoint download URL yet.

From the repository root, replace the example image path and use a new output:

```powershell
python scripts/v3/predict.py --artifact artifacts/v3/run_final/model.joblib --data-root C:/path/to/images --image sample.png --output artifacts/v3/demo_primary --device cpu --offline
python scripts/v3/predict_finetuned.py --artifact artifacts/v3/finetune/candidates/finetuned/model.pt --data-root C:/path/to/images --image sample.png --output artifacts/v3/demo_dino --device cpu
```

The second command uses the exploratory comparison; it does not replace the
primary. `--offline` requires populated pinned caches. Ordered outputs are wood,
metal, stone, concrete, ceramic, fabric and leather. Probabilities are not
calibrated guarantees. For existing results, read saved prediction CSVs rather
than rerunning the now-scored holdout.

| Model | Local artifact | Predictor |
| --- | --- | --- |
| Primary | `artifacts/v3/run_final/model.joblib` | `scripts/v3/predict.py` |
| Control | `artifacts/v3/run_final/control_model.joblib` | `scripts/v3/predict.py` |
| ResNet | `artifacts/v3/resnet-final/final/model.pt` | `scripts/v3/predict_resnet.py` (manifest input) |
| DINOv2 comparison | `artifacts/v3/finetune/candidates/finetuned/model.pt` | `scripts/v3/predict_finetuned.py` |

## Data and protocol

The study contains **3,780 images / 126 material IDs / 122 known source families**:
3,150 development images / 101 families and 630 fresh test images / 21 groups.
Thirty views per material are dependent. Old train/val/test directories remain
lineage; all 3,150 original images are development data in V3.

Five outer and three inner source-group folds assess frozen representation/C
selection. Initial nested accuracy is 79.3968%, Macro F1 0.795485, 95% accuracy
CI 72.65-85.57%. Matched control and fixed ResNet OOF accuracies are 68.8571%
and 71.5556%. These are separate from the fresh endpoint. Canonical
`artifacts/v3/features_final` and `artifacts/v3/run_final` exactly reproduce
earlier frozen feature arrays and OOF predictions.

The selected recipe combines DINOv2 small register CLS/patch mean/std at 224
and 336 pixels with ConvNeXt-Tiny pooled features at 224, C=1. Registers are
excluded from patch pooling. Scalers and logistic heads fit inside each training
partition. Development Macro F1 0.8035245993 exceeded the bounded exploratory
DINOv2 follow-up's 0.7724822183. DINOv2 searched two learning rates in six fits
and used five final epochs. These are selection scores, not independent tests.

Primary corruption tests were predeclared. DINOv2 tests on the same five
corruptions were **post hoc**, after clean and primary-corruption scores; see
[amendment](reports/results/v3/exploratory_robustness_amendment.json). No model
changed. Pretraining, resolutions and adaptation/search budgets differ, so
pipeline comparisons do not isolate architecture effects.

## Rebuild the report without training

The builder checks explicit evidence, predictions and hashes. Copied text is
UTF-8/LF before hashing; weights are referenced, not copied. Use **new output
directories**. Original root `report.pdf` and `report.docx` remain preserved.

```powershell
python scripts/v3/assemble_report.py --spec reports/results/v3/report_inputs/assembly.json --output-dir reports/results/v3/report_evidence_rebuild
python scripts/v3/build_report.py --results-dir reports/results/v3/report_evidence_rebuild --output-dir reports/v3_rebuild
python -m pytest tests/test_v3_report.py tests/test_v3_report_assembly.py -q
```

The spec records explicit local paths; relocate them for another checkout.
[Report methods](reports/V3_REPORT_METHODS.md) documents the contract.
The [manifest](reports/results/v3/report_evidence_final/report_manifest.json)
and `derived/source_inventory.json` preserve original and normalized hashes.
Actual neural histories, images, confusions, calibration, corruptions and
benchmarks are included. No logistic epoch curves are invented. Historical V1
results remain context only, not direct improvement estimates.

## Limitations and release

Only three fresh families per class, restricted ceramic/concrete semantics,
and Blender 5.1.1's softer shadows limit generalization. Physical ancestry is
incomplete; public texture overlap with backbone pretraining has not been
audited. No arbitrary real-world validation is claimed. JPEG compression is a
major limitation.

All 316 original and 63 fresh source maps have verified official CC0 provenance.
The project authors renders/code and attributes third-party maps. The complete
local audit ZIP is verified; copied PNG metadata was sanitized with identical
decoded pixels. **Dataset license unset; `release_approved=false`; no upload.**
See [study release details](reports/V3_STUDY_DATASET_RELEASE.md) and
[DATASET.md](DATASET.md). There is no public dataset URL yet.

The legacy `scripts/update_docs.py` emits V1 content and must not overwrite
these V3 cards. Future tuning on this now-scored holdout requires disclosure
and a new independent assessment for confirmatory claims.
