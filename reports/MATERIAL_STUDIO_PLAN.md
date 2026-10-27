# Material Studio and final academic submission

User-authorized continuation, 27 October 2026. Completed study merged through
PR #1 to main at 64adc07; old v3 branch removed. Implementation branch:
`feat/material-studio`. Kaggle publication stays with the owner.

## Deliverables and phases

1. Native Rust GPUI desktop application: responsive Zed/Zen-inspired dark glass
   styling, local image selection, real model inference, class probability bars,
   representative samples, measured model comparison and confusion matrices.
   Keep rendering independent from model loading/inference. No remote inference.
2. Persistent Python inference worker using reviewed saved model loaders and exact
   transforms. JSON Lines on stdin/stdout; diagnostics on stderr. Startup handshake,
   bounded requests, structured errors and model caching. Four existing model IDs:
   frozen_selected, frozen_control, resnet_fixed, dinov2_finetuned. Default demo
   choice may be DINOv2; the study's preselected primary remains frozen_selected.
3. Build and run a Windows release executable. Package a documented local demo
   folder with samples and necessary runtime/model resources; verify outside the
   source directory. Measure startup/warm inference and inspect the actual UI.
4. Rewrite the academic report to the supplied ITRI626 brief: 10–15 main pages,
   title/group fields blank, abstract, introduction, dataset/preparation/ethics,
   methodology, experimental settings, results, discussion/error analysis,
   conclusion, references. Exclude repository-version history, ZIP details,
   line endings, agent workflow and internal file hashes from the academic prose.
   Preserve source-group independence, real limitations and honest model roles.
   Include all three families, exact split/fold counts, preprocessing, complete
   hyperparameters, per-class metrics, confusion matrices, actual learning curves,
   and correct/incorrect prediction examples. Never fabricate missing histories.
5. Public-facing README, classroom demonstration instructions, assessment coverage,
   and presentation material. Author/member fields remain blank per user.
6. Independent code/method review, relevant Python/Rust tests, actual executable
   smoke/visual checks, full-page PDF QA, feature PR with evidence.

## Integration contract

Rust crate: `app/`; Python package: `src/material_classifier/demo/`; entry point:
`scripts/demo/worker.py`. Protocol version 1, one request/response per JSON line.
Requests carry string `id`, `op` in `health`, `predict`, `shutdown`; predict carries
`model` and absolute `image_path`. Responses preserve `id`, `ok`; successful predict
includes model, class_names, probabilities, predicted_class, inference_ms, device.
Failures include `error` object with code/message. Initial worker event:
`{"event":"ready","protocol_version":1}`. No model load blocks UI startup.
App launches worker with explicit root and chooses configured Python or repository
environment; release packaging must not bake developer absolute paths into code.

Static demo assets/results manifest: `app/assets/study.json`, compiled/read without
Python for immediate navigation. Sample paths resolve against resources root;
saved study metrics are labeled study results, never live inference measurements.
Model worker accepts `--root ROOT --device auto|cuda|cpu`. CPU is valid fallback.
Only one active prediction runs at a time; UI exposes loading/error states.

## Reference and scope

The user-linked private AI-Vision-Models repository has Python learning exercises
and a static HTML site, no GPUI crate on main. Asked whether another reference was
intended; native Rust GPUI work can proceed. Do not copy private reference content
into this future-public repository. Use official GPUI APIs and preserve licenses.
No new training or changes to sealed benchmark artifacts are required.

## Superseding visual direction

The user rejected the teal, panel-heavy interface and supplied MonoCode
(https://github.com/hardbeat920/monocode) plus a screenshot. Match its restrained
dark monochrome character: subtle translucency, no gradients or colored accents,
quiet borders, compact navigation and far less visible explanatory text.
Default classification should be one image workspace with compact controls and
an inline result. Model choice is a dropdown; detailed probabilities and study
information appear only when requested. Preserve the native GPUI implementation,
all reviewed inference behavior and access to study results/sample browsing.
The previously built packages remain candidates until this design is verified.
