You are the lead ML engineer and research-project orchestrator for this repository.

You may create and delegate work to luna (high) sub-agents as useful while you (sol) review the work. Continue autonomously through repository cleanup, dataset verification, implementation, training, evaluation, visualisation, documentation, and final reporting unless a genuine blocking error requires human input.

Do NOT stop after writing a plan. Implement and execute the project.

==========================================================
PROJECT
==========================================================

This is an honours-level university deep-learning image-classification project.

Task:
Classify synthetic PBR material renders into seven classes:

- wood
- metal
- stone
- concrete
- ceramic
- fabric
- leather

The dataset was procedurally generated in Blender using RenderSynth.

Dataset statistics:

- 7 classes
- 15 independent base materials per class
- 105 base materials total
- 30 renders per base material
- 3,150 images total
- 512 × 512 PNG RGB
- Blender EEVEE
- deterministic generation seed: 62642

Existing fixed split:

TRAIN
- 77 base materials
- 2,310 images
- 330 images/class

VALIDATION
- 14 base materials
- 420 images
- 60 images/class

TEST
- 14 base materials
- 420 images
- 60 images/class

The split is by BASE MATERIAL ID, not image.

The same material must NEVER occur across different splits.

Existing manifests/reports contain this information. Inspect them rather than recreating or guessing the split.

The project should compare these main architectures:

1. Custom CNN baseline — trained from scratch
2. ResNet-50 — ImageNet pretrained
3. ConvNeXt-Tiny — ImageNet pretrained
4. ViT-B/16 — ImageNet pretrained

The three headline transfer-learning architectures represent the main family comparison:

- classical residual CNN
- modern convolutional architecture
- vision transformer

The custom CNN is an additional baseline.

Do not add dozens of unrelated architectures just to inflate the experiment count.

==========================================================
CRITICAL METHODOLOGY RULES
==========================================================

1. NEVER randomly repartition individual images.

2. `material_id` is the grouping identity.

All 30 renders from a given material must always remain together.

3. The existing TEST split is locked.

Do not use test images for:

- hyperparameter tuning
- early stopping
- model selection
- augmentation selection
- threshold selection
- debugging model performance

Only evaluate TEST after model choices and hyperparameters are locked.

4. Ordinary KFold / StratifiedKFold over images is forbidden.

For cross-validation use:

`StratifiedGroupKFold`

where:

- y = material class
- groups = material_id

5. Cross-validation should operate only on development data:

existing TRAIN + VALIDATION

= 91 base materials
= 2,730 images

TEST remains excluded.

6. Main headline experiment must still use the original:

TRAIN / VAL / TEST

so all models have directly comparable training curves and validation behaviour.

7. Data augmentation must be applied ONLY to training images.

8. Keep the comparison fair.

All headline models use:

- identical fixed split
- identical evaluation dataset
- identical input resolution
- equivalent augmentation philosophy
- equivalent evaluation code
- same random seeds where applicable

Architecture-specific optimizer/lr values are allowed when justified.

==========================================================
PHASE 1 — AUDIT AND REFACTOR THE REPOSITORY
==========================================================

First inspect the entire repository.

Do NOT delete useful project provenance.

The current repository contains things such as:

- backups/
- blender/
- dataset/
- dataset_reports/
- material_audit/
- materials/
- notebooks/
- pilot_dataset/
- renders/
- scripts/
- validation_renders/
- manifests
- RenderSynth configurations
- Blender-generation reports

Refactor it into a clean, professional ML research repository.

Aim for approximately:

/
├── README.md
├── DATASET.md
├── MODEL_CARD.md                 # generated/updated after training
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── configs/
│   ├── data.yaml
│   ├── custom_cnn.yaml
│   ├── resnet50.yaml
│   ├── convnext_tiny.yaml
│   └── vit_b16.yaml
│
├── src/
│   └── material_classifier/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── augmentations.py
│       ├── train.py
│       ├── evaluate.py
│       ├── cross_validate.py
│       ├── metrics.py
│       ├── plotting.py
│       ├── explainability.py
│       ├── inference.py
│       └── utils.py
│
├── scripts/
│   ├── verify_dataset.py
│   ├── run_eda.py
│   ├── train_model.py
│   ├── train_all.py
│   ├── run_group_cv.py
│   ├── evaluate_all.py
│   ├── generate_visuals.py
│   ├── generate_report.py
│   └── package_dataset.py
│
├── blender/
│   ├── README.md
│   ├── scripts/
│   └── configs/
│
├── metadata/
│   ├── material_manifest.csv
│   ├── material_manifest.json
│   ├── material_manifest_split.csv
│   ├── material_manifest_split.json
│   ├── render_manifest.csv
│   ├── render_manifest.json
│   └── dataset_counts.csv
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_training_results.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_error_analysis.ipynb
│
├── reports/
│   ├── dataset_generation/
│   ├── figures/
│   │   ├── dataset/
│   │   ├── training/
│   │   ├── confusion_matrices/
│   │   ├── per_class/
│   │   ├── cv/
│   │   ├── explainability/
│   │   ├── errors/
│   │   └── comparison/
│   ├── tables/
│   └── results/
│
├── tests/
│
├── dataset/                       # LOCAL ONLY — gitignored
├── materials/                     # LOCAL ONLY — gitignored
└── artifacts/                     # checkpoints/logs — gitignored

Adapt intelligently to the existing repository rather than blindly recreating everything.

Use moves rather than deleting provenance files.

Keep the repository root clean.

==========================================================
GIT AND .GITIGNORE
==========================================================

Create a thorough professional `.gitignore`.

Ignore at minimum:

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# environments
.venv/
venv/
env/
.env

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# raw/generated large data
dataset/
materials/
renders/
pilot_dataset/
validation_renders/
material_audit/
backups/

# ML artifacts
artifacts/
runs/
checkpoints/
tensorboard/
wandb/
*.pt
*.pth
*.ckpt
*.onnx
*.safetensors

# Blender binary assets
*.blend
*.blend1
*.blend2

# temporary
*.tmp
*.temp
*.log

Do NOT ignore:

- source code
- configs
- manifests
- small reports
- CSV result summaries
- contact sheets useful for the report
- publication-quality figures
- README documentation
- RenderSynth JSON config
- Blender Python scripts

If the repository is already under Git:

- preserve history where practical
- use `git mv` when appropriate
- do NOT push
- do NOT create destructive commits without permission

At the end show:

`git status`

and summarize what should be committed.

==========================================================
DATASET AVAILABILITY / GITHUB
==========================================================

Do NOT put the 3,150 image dataset or raw PBR textures into Git.

Create `DATASET.md`.

It should explain:

- dataset purpose
- seven classes
- 105 base materials
- 30 renders/material
- 3,150 images
- Blender + RenderSynth generation
- 512×512 RGB PNG
- leakage-safe grouping
- train/val/test counts
- how the dataset can be regenerated
- expected local folder path
- licensing/source information for PBR assets
- current public dataset URL:

`Not yet publicly released`

Add a clearly marked future placeholder:

`Kaggle dataset: TBD`

Create a dataset packaging script that can later produce a Kaggle-ready ZIP and checksum, but DO NOT upload anything.

==========================================================
ENVIRONMENT
==========================================================

Use Python and PyTorch.

Prefer torchvision implementations when suitable:

- torchvision ResNet50
- torchvision ConvNeXt-Tiny
- torchvision ViT-B/16

Use ImageNet pretrained weights for the three transfer-learning models.

Recommended dependencies:

- torch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- pillow
- pyyaml
- tqdm
- tensorboard
- torchinfo
- pytest
- ruff

For explainability optionally add:

- grad-cam / pytorch-grad-cam

For embedding visualisation:

- umap-learn if installation succeeds
- otherwise use sklearn t-SNE/PCA

Do not introduce huge frameworks unnecessarily.

Verify GPU before training:

- CUDA available
- GPU name
- VRAM
- CUDA version
- PyTorch version

Write environment details into:

`reports/results/environment.json`

Use the NVIDIA RTX 4060 if available.

Use mixed precision training.

==========================================================
REPRODUCIBILITY
==========================================================

Use a project-level seed:

42

Seed:

- Python random
- NumPy
- PyTorch CPU
- PyTorch CUDA
- DataLoader workers

Record:

- seed
- Python version
- PyTorch version
- torchvision version
- CUDA version
- GPU
- config
- git commit hash if available
- start/end timestamps

Use deterministic behaviour where reasonably possible.

Avoid sacrificing huge amounts of performance for absolute bit-level determinism, but make experiments reproducible enough for academic work.

==========================================================
PHASE 2 — DATASET VERIFICATION
==========================================================

Before training, write and execute:

`scripts/verify_dataset.py`

Validate:

- 3,150 images total
- 2,310 train
- 420 val
- 420 test
- exactly 450 images/class
- train = 330/class
- val = 60/class
- test = 60/class
- 105 unique material_ids
- 30 renders/material
- no material_id crosses splits
- no missing files
- no corrupt files
- all images are 512×512 RGB
- all manifest paths resolve
- no unexpected class folders
- no duplicate manifest entries

Fail loudly if leakage is detected.

Generate:

`reports/results/dataset_validation.json`

==========================================================
PHASE 3 — EDA
==========================================================

Create a full dataset EDA stage before training.

Generate:

1. class distribution bar chart
2. split distribution chart
3. class × split stacked chart
4. representative image grid for every class
5. representative images from train / val / test
6. base-material count per class
7. renders-per-material verification plot
8. image dimension summary
9. RGB channel mean/std
10. luminance distribution
11. representative material diversity contact sheet
12. class-specific contact sheets

Preserve the existing Blender-generation contact sheets where useful.

Produce:

`reports/results/dataset_summary.csv`

and publication-quality PNG figures.

==========================================================
INPUT PREPROCESSING
==========================================================

Headline comparison input size:

224 × 224

All headline models must use the same input size.

Validation/test preprocessing:

- Resize approximately 256
- CenterCrop 224
- ToTensor
- ImageNet normalization

Training augmentation should be deliberately conservative because MATERIAL TEXTURE is the signal.

Recommended:

- RandomResizedCrop 224
  scale roughly 0.88–1.0
  conservative aspect-ratio range

- RandomHorizontalFlip(p=0.5)

- very mild ColorJitter:
  brightness about ±0.10
  contrast about ±0.10
  saturation about ±0.05
  hue about ±0.02

- ImageNet normalization

Do NOT use by default:

- vertical flipping
- 90-degree whole-image rotation
- aggressive blur
- aggressive RandomErasing
- MixUp
- CutMix

because these may create physically unrealistic material/lighting compositions.

If an augmentation is added later, document and justify it.

==========================================================
PHASE 4 — MODELS
==========================================================

Implement exactly these primary experiments:

A. CustomCNN
B. ResNet50
C. ConvNeXt-Tiny
D. ViT-B/16

----------------------------------------------------------
A — CUSTOM CNN
----------------------------------------------------------

Create a sensible modern small CNN baseline.

Requirements:

- 4–5 convolutional stages
- BatchNorm
- ReLU or GELU
- MaxPool or strided conv
- AdaptiveAvgPool
- Dropout
- linear classifier

Avoid making it absurdly large.

Train from scratch.

Purpose:
demonstrate how much transfer learning contributes.

----------------------------------------------------------
B — RESNET50
----------------------------------------------------------

Use pretrained ImageNet weights.

Replace classifier for 7 classes.

----------------------------------------------------------
C — CONVNEXT-TINY
----------------------------------------------------------

Use pretrained ImageNet weights.

Replace classifier for 7 classes.

----------------------------------------------------------
D — VIT-B/16
----------------------------------------------------------

Use pretrained ImageNet weights.

Replace classifier for 7 classes.

==========================================================
TRANSFER-LEARNING STRATEGY
==========================================================

For pretrained models use staged fine-tuning.

Stage 1:
- freeze backbone
- train classification head for ~3 epochs

Stage 2:
- unfreeze backbone
- fine-tune whole model

Use discriminative learning rates where sensible:

backbone LR < classifier-head LR

Do not force identical LRs on architectures when inappropriate.

Document exact choices.

==========================================================
TRAINING STANDARD
==========================================================

Use:

- CrossEntropyLoss
- label smoothing around 0.1
- AdamW
- learning-rate warmup
- cosine decay or similarly defensible scheduler
- mixed precision / AMP
- gradient clipping around 1.0
- early stopping
- checkpoint best validation Macro F1
- validation loss as secondary/tie-breaking signal

Maximum epochs:
approximately 40

Early-stopping patience:
approximately 7–8 epochs

Do not stop a run after only a few epochs unless convergence clearly occurs.

Log every epoch:

- train loss
- train accuracy
- train macro F1
- val loss
- val accuracy
- val macro F1
- learning rate
- epoch time

Save logs as CSV and TensorBoard.

==========================================================
BATCH SIZE
==========================================================

Automatically determine a safe batch size per architecture based on GPU memory.

Expected approximate starting points on RTX 4060:

- CustomCNN: 32–64
- ResNet50: 32
- ConvNeXt-Tiny: 16–32
- ViT-B/16: 8–16

Use AMP.

If CUDA OOM occurs:

1. reduce batch size
2. use gradient accumulation if required
3. record the final effective batch size

Do not silently change methodology.

==========================================================
MODEL CHECKPOINTS
==========================================================

Save locally under:

`artifacts/checkpoints/<model>/`

Save:

- best checkpoint
- last checkpoint
- config
- class_to_idx
- input preprocessing
- normalization
- best epoch
- validation metrics
- training history

Checkpoint should contain enough information to load the model later for deployment.

Do NOT commit weights to Git.

==========================================================
PHASE 5 — GROUPED 5-FOLD CROSS-VALIDATION
==========================================================

This is an ADDITIONAL robustness experiment.

Do not replace the headline train/val/test experiment.

Use:

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`

Development data only:

existing TRAIN + VAL

Groups:

`material_id`

Target:

material class

TEST data must never enter these folds.

Run grouped CV for:

- ResNet50
- ConvNeXt-Tiny
- ViT-B/16

Custom CNN CV is optional if runtime is reasonable.

Use the same locked training configuration as the headline experiment.

For each fold save:

- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- best epoch
- training time

Generate aggregate:

mean ± standard deviation.

Create:

`reports/results/cross_validation_results.csv`

Visuals:

- CV accuracy boxplot
- CV macro-F1 boxplot
- model mean ± std plot
- fold-by-model heatmap

IMPORTANT:

CV grouping must be by material_id.

Write an explicit automated assertion that no material_id occurs in both training and validation for any fold.

==========================================================
MODEL SELECTION / TEST PROTOCOL
==========================================================

Finish:

- architecture implementation
- training configuration
- development experiments
- grouped CV

BEFORE final test evaluation.

Then run the final headline model training on the ORIGINAL:

TRAIN
with
VALIDATION

using the locked configuration.

Select the best checkpoint based only on validation Macro F1.

Then evaluate TEST exactly once for each headline model.

Do not tune after seeing test results.

==========================================================
PHASE 6 — REQUIRED TEST METRICS
==========================================================

For every model calculate:

CORE:

- test loss
- accuracy
- balanced accuracy
- precision macro
- recall macro
- F1 macro
- precision weighted
- recall weighted
- F1 weighted

PER CLASS:

- precision
- recall
- F1
- support

CONFUSION MATRICES:

- raw counts
- row-normalized percentages

PROBABILITY METRICS:

- one-vs-rest ROC-AUC macro
- one-vs-rest ROC-AUC weighted
- per-class ROC-AUC where possible
- log loss

OPTIONAL BUT VALUABLE:

- Expected Calibration Error
- reliability diagram
- confidence distribution

==========================================================
PHASE 7 — EFFICIENCY METRICS
==========================================================

For every model report:

- total parameters
- trainable parameters
- model checkpoint size MB
- total training time
- best epoch
- GPU peak VRAM if practical
- inference latency batch=1
- inference throughput images/sec
- inference latency batch=32 if memory permits

Benchmark inference after warm-up and synchronize CUDA timing correctly.

==========================================================
PHASE 8 — VISUALISATIONS
==========================================================

Generate professional publication-quality visuals.

Use consistent styling throughout.

DATASET:

- class distribution
- split distribution
- sample grids

TRAINING:

For each model:
- train vs val loss
- train vs val accuracy
- train vs val Macro F1
- learning-rate curve

Also create combined plots comparing models.

CONFUSION:

For each model:
- raw confusion matrix
- normalized confusion matrix

PER CLASS:

- precision bar chart
- recall bar chart
- F1 bar chart
- grouped F1 comparison across models

ROC / PR:

- macro ROC comparison
- per-class ROC plots
- precision-recall plots where useful

CV:

- grouped CV accuracy distribution
- grouped CV Macro F1 distribution
- mean ± std comparison

EFFICIENCY:

- accuracy vs parameter count
- Macro F1 vs model size
- Macro F1 vs inference latency
- training time comparison

EMBEDDINGS:

For each headline model extract penultimate-layer test embeddings.

Generate:

- PCA
- UMAP if available
- otherwise t-SNE

Colour points by material class.

ERROR ANALYSIS:

Generate galleries of:

- high-confidence correct predictions
- high-confidence incorrect predictions
- lowest-confidence predictions
- most confused class pairs
- several examples from each major confusion pair

Show:

- true label
- predicted label
- confidence

MATERIAL-LEVEL ANALYSIS:

Because the dataset has base-material identities:

calculate test accuracy per material_id.

Generate:

- per-material accuracy chart
- worst-performing held-out materials
- best-performing held-out materials

This is extremely valuable for this project.

==========================================================
EXPLAINABILITY
==========================================================

Add explainability where practical.

For:

- ResNet50
- ConvNeXt-Tiny
- ViT-B/16

Prefer Grad-CAM / compatible CAM technique.

For ViT use the correct reshape transform or another technically appropriate method.

Generate a curated set of:

- correct wood example
- correct metal example
- correct stone example
- correct concrete example
- correct ceramic example
- correct fabric example
- correct leather example

plus several misclassified examples.

Do not produce hundreds of useless heatmaps.

Create a concise explainability gallery.

==========================================================
PHASE 9 — COMPARISON DASHBOARD
==========================================================

Create a polished final visual inspired by a professional benchmark table.

Generate:

`reports/figures/comparison/model_comparison_dashboard.png`

Include columns similar to:

Family
Model
Pretrained?
Parameters
Best epoch
Val accuracy
Test accuracy
Test Macro F1
Test Weighted F1
CV Macro F1 mean ± std
Train accuracy
Generalisation gap
Inference ms/image
Model size MB
Training time

Create matching:

- CSV
- Markdown table
- PNG dashboard

Make the visual polished and readable.

Do not simply copy another person's screenshot.

Create an original design.

Also create a ranking figure for:

1. test Macro F1
2. test accuracy
3. efficiency

==========================================================
OVERFITTING ANALYSIS
==========================================================

Calculate:

`train_accuracy - validation_accuracy`

and:

`train_macro_f1 - validation_macro_f1`

for each model.

Also inspect validation loss trajectory.

Create a documented overfitting indicator rather than inventing an arbitrary unexplained flag.

For example flag potential overfitting if:

- generalisation gap is large
AND
- validation loss rises while training loss continues falling

Explain the rule in the report.

==========================================================
STATISTICAL COMPARISON
==========================================================

Use the 5 CV folds to report variability.

Do not pretend one single accuracy number proves superiority.

Report:

mean ± std.

If appropriate and statistically defensible, perform paired comparisons between model fold scores.

Use conservative interpretation because n=5 folds is small.

Do not make exaggerated statistical claims.

==========================================================
PHASE 10 — FINAL MODEL DECISION
==========================================================

Recommend a final model based primarily on:

1. test Macro F1
2. test accuracy
3. grouped CV stability
4. per-class consistency
5. model size
6. inference speed

Do not automatically select the largest network.

Explain the accuracy/efficiency trade-off.

==========================================================
PHASE 11 — MODEL CARD
==========================================================

Create/update:

`MODEL_CARD.md`

For the recommended model include:

- architecture
- classes
- preprocessing
- training data
- synthetic dataset origin
- performance
- per-class behaviour
- limitations
- known domain gap
- intended use
- inappropriate use
- checkpoint location
- inference example

IMPORTANT limitation:

The model is trained on synthetic Blender PBR material renders.

Do NOT claim proven arbitrary real-world material recognition.

Explicitly state that synthetic-to-real generalisation has not yet been established.

==========================================================
PHASE 12 — README
==========================================================

Create a polished main README.

Include:

1. project title
2. short overview
3. motivation / design-engineering context
4. example dataset/contact-sheet image
5. material classes
6. dataset statistics
7. procedural Blender generation pipeline
8. leakage-safe split explanation
9. models compared
10. training methodology
11. grouped CV methodology
12. headline results table
13. best confusion matrix
14. training curve figure
15. model-comparison dashboard
16. repository structure
17. setup instructions
18. dataset placement instructions
19. training commands
20. evaluation commands
21. reproducibility
22. limitations
23. future work:
    - synthetic-to-real evaluation
    - Hugging Face deployment
    - interactive web demo
    - Kaggle dataset release

Keep README concise enough to be usable but visually professional.

==========================================================
PHASE 13 — NOTEBOOKS
==========================================================

Create clean notebooks with visible outputs.

Do NOT put all project logic only in notebooks.

The real logic belongs in `src/`.

Notebooks should call library functions.

Required notebooks:

01_dataset_eda.ipynb
- dataset QA
- samples
- class distribution

02_training_results.ipynb
- histories
- curves
- core metrics

03_model_comparison.ipynb
- final comparison
- CV
- efficiency

04_error_analysis.ipynb
- misclassifications
- per-class results
- explainability
- embeddings

Ensure notebooks can run from repository root.

==========================================================
PHASE 14 — TESTS
==========================================================

Create automated tests for at least:

- dataset class mapping
- dataset length
- split counts
- no material leakage
- grouped CV leakage prevention
- model output shape = [batch, 7]
- metrics function
- checkpoint save/load
- deterministic seed helper

Run:

pytest

before finalizing.

==========================================================
CODE QUALITY
==========================================================

Use:

- type hints
- docstrings
- pathlib rather than string paths
- logging rather than excessive prints
- clear exceptions
- modular functions
- config files rather than hard-coded constants

Add Ruff configuration.

Run linting.

Do not overengineer the project.

==========================================================
MAIN COMMANDS
==========================================================

Create simple commands so the whole workflow can be reproduced.

Examples:

Dataset QA:

python scripts/verify_dataset.py

EDA:

python scripts/run_eda.py

Train a model:

python scripts/train_model.py --config configs/resnet50.yaml

Train all headline models:

python scripts/train_all.py

Grouped CV:

python scripts/run_group_cv.py --folds 5

Evaluation:

python scripts/evaluate_all.py

Visualisations:

python scripts/generate_visuals.py

Final report generation:

python scripts/generate_report.py

If helpful create:

python scripts/run_pipeline.py

that executes all non-destructive stages in order.

==========================================================
EXECUTION ORDER
==========================================================

Execute in this exact logical order:

1. inventory existing repository
2. safely refactor repository
3. create environment/config files
4. verify CUDA
5. verify dataset
6. run tests
7. generate EDA
8. implement models
9. smoke-test each model for 1 epoch
10. fix errors
11. train CustomCNN
12. train ResNet50
13. train ConvNeXt-Tiny
14. train ViT-B/16
15. run grouped 5-fold CV on the three headline transfer models
16. lock methodology
17. evaluate held-out TEST
18. benchmark inference
19. create confusion matrices
20. create per-class metrics
21. create ROC/PR/calibration figures
22. create embeddings
23. create error-analysis galleries
24. create explainability visuals
25. create comparison dashboard
26. determine recommended model
27. save all checkpoints/metadata
28. generate MODEL_CARD.md
29. generate DATASET.md
30. update README.md
31. execute notebooks so outputs are visible if practical
32. run pytest
33. run lint
34. show final git status
35. provide final execution summary

==========================================================
RESUMABILITY
==========================================================

Training can take hours.

Every run must be resumable.

If a completed checkpoint + valid metrics exist:

- do not retrain unnecessarily

unless:

`--force`

is supplied.

Maintain run-state files.

If Codex/sub-agent execution is interrupted, restarting should continue from the next incomplete stage.

==========================================================
FAILURE HANDLING
==========================================================

If CUDA OOM:
- reduce batch size
- retry
- record adjustment

If a single architecture fails:
- debug that model
- do not delete results from completed models

If UMAP fails:
- fall back to t-SNE or PCA

If Grad-CAM library fails:
- implement a simpler compatible saliency approach
- do not block the entire project

If a notebook cannot execute:
- preserve the underlying figures/results
- report the failure

Do not silently skip important assignment outputs.

==========================================================
FINAL DELIVERABLES
==========================================================

When everything completes, I expect:

SOURCE / INFRASTRUCTURE
- clean professional repo
- .gitignore
- requirements
- pyproject
- configs
- tests
- reusable source package

DATA
- verified dataset metadata
- no leakage
- dataset card
- regeneration instructions

MODELS
- CustomCNN
- ResNet50
- ConvNeXt-Tiny
- ViT-B/16

TRAINING
- best checkpoints
- training logs
- validation histories
- grouped 5-fold CV

EVALUATION
- accuracy
- balanced accuracy
- precision
- recall
- F1
- macro F1
- weighted F1
- per-class metrics
- confusion matrices
- ROC-AUC
- efficiency metrics

VISUALS
- dataset figures
- curves
- confusion matrices
- per-class charts
- CV figures
- embeddings
- misclassification galleries
- explainability
- efficiency plots
- polished final model-comparison dashboard

DOCUMENTATION
- README.md
- DATASET.md
- MODEL_CARD.md
- notebooks with visible outputs
- final results tables

==========================================================
FINAL RESPONSE
==========================================================

When all work is complete, give me:

1. repository structure after cleanup
2. environment/GPU summary
3. dataset verification result
4. exact training configuration for each model
5. training time for each model
6. best epoch for each model
7. validation metrics
8. grouped CV mean ± std
9. held-out test metrics
10. per-class F1
11. model sizes
12. inference speeds
13. final model recommendation
14. all important generated figure paths
15. checkpoint paths
16. README / DATASET / MODEL_CARD paths
17. pytest/lint results
18. git status summary
19. any limitations or issues encountered

Do not begin deployment or Hugging Face web-app work yet.

Stop after the complete training/evaluation/repository/documentation phase is finished.