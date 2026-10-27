# Material Studio review and validation

## Native interface

The final interface follows the user's monochrome reference: a 160 px sidebar,
one image workspace, compact controls, subtle acrylic, optional prediction
details, a grayscale study matrix and a filtered sample gallery.

Actual desktop checks covered live DINOv2 predictions, model changes clearing
prior results, model/sample menus, seven-class details, the Study matrix,
sample filtering, About, keyboard classification, Escape and graceful exit.
The earlier desktop-lock capture failure was resolved after the user unlocked
the desktop. Screenshots are retained in artifacts/material-studio/ui-qa;
the public screenshot is [the actual app](../docs/images/material-studio.png).

Independent reviews identified and verified fixes for the incorrect ConvNeXt
label, recoverable image errors discarding the worker/cache, clipped captions,
and hidden disclosure state consuming Escape during inference. The final fix
checks current-page visibility and prediction presence; stale responses still
require the active request ID. The final scoped review at 5a27bc2 reports no
remaining P1/P2 findings. Rust formatting, release build and 19 regular tests
pass. The opt-in real-model valid-invalid-valid test was separately run before
the UI-only redesign; it is not counted among the regular tests.

Release executable SHA-256:
65531935d8c785358f3d85c5382f37b3a970b716e6be5b8a7daf4ee868bacbdb.
Official GPUI and Rust 1.99 remain pinned. VCRUNTIME140.dll accompanies the
portable distribution; shaders are embedded.

## Inference and runtime

Bundled HF_HOME and HF_HUB_CACHE override inherited settings before loader
imports. A subprocess regression verifies this with conflicting host settings.
Actual source-worker smoke completed 16 cold/warm requests across all four
models on CUDA and CPU. CPU probabilities exactly match the reviewed APIs.
Evidence: reports/results/demo/inference_smoke.json. An earlier packaged
runtime passed eight CPU predictions from an external working directory with
restricted PATH and conflicting cache settings. These are integration checks,
not new study accuracy or throughput estimates.

The full Python suite passed 566 tests, with two Windows symlink-permission
skips and one existing warning in the mocked legacy OOM test. Demo/report Ruff
checks pass. The final monochrome package passed all four models cold/warm on
CPU and CUDA (16 predictions), with empty stderr and exact repeated probabilities.
Its native executable launched outside the checkout and completed live inference.
The later opacity-only update in 515b650 exposes more Acrylic blur. It built
successfully; the refresh verified all 22,797 previous payload entries and
updated the executable plus its manifest record. Models/runtime remain identical.
Latest executable SHA-256:
35fac15c99305572a1a33baf55c2acd1ce495ef5fab155b685861019cb4e2cc2.
See reports/results/demo/application_validation.json and docs/DEMO_PACKAGING.md.
Earlier distribution folders are superseded candidates.

## Academic submission

The assignment-aligned report has 14 main pages, one reference page and four
appendices. All 19 pages were rendered and visually inspected; eight report
evidence/layout tests pass. Student names and numbers remain blank. Independent
review verified metrics, folds, corruptions, metadata and real learning curves
without a report P1/P2 finding.

The 14-slide editable presentation and PDF were visually inspected. Slide 11
carries the provisional DINOv2 classroom recommendation while retaining fusion
as the predeclared study primary. Slide 13, notes and guides match Classify /
Study / Samples / About and optional Details. The latest render comparison
confirms the other 13 pages are unchanged.

Missing epoch-wise ResNet validation histories and unequal tuning budgets remain
disclosed. No missing measurements were fabricated. Dataset publication,
student details and oral delivery remain with the owner.
