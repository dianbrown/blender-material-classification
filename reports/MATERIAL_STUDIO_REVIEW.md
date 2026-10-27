# Material Studio review and validation

## Inference service

Independent worker review found one P2: inherited HF_HOME or HF_HUB_CACHE could
override the packaged offline weights. Fixed in18c79da: when a bundled cache
exists, explicitly set both variables before importing the ML libraries. The
new subprocess regression uses three conflicting inherited cache settings and
successfully resolves a dummy bundled snapshot without network access.
Worker/report focused tests:34passed. Asset tests:2passed separately.

Actual demonstration smoke:16cold/warm requests across all four models on CUDA
and CPU, using one development image. CPU probabilities exactly match the
reviewed prediction APIs. Source:reports/results/demo/inference_smoke.json.
These are integration checks, not new study accuracy or throughput estimates.

## Native application

Official pinned GPUI, Rust1.99.0; formatting, checking,11unit tests and release
build passed in the implementation worktree. Real Rust-to-Python DINO smoke
passed after a Windows verbatim-path normalization defect was caught and fixed.
Evidence:artifacts/material-studio/rust-worker-smoke-fixed.json. The earlier
failed smoke is retained. Release requires VCRUNTIME140.dll beside the executable;
other recorded imports are Windows system libraries. Shaders are embedded.

Native visual/interaction checks remain pending. Window opens and responds, but
Computer Use activation returned GetCursorPos access denied0x80070005; refreshed
capture was black. The user was asked to unlock the desktop. This is not recorded
as a visual pass or diagnosed as an application rendering defect.

## Academic submission

New report:14main pages,1reference page,4appendices; all19rendered and inspected
by its author,8evidence/layout tests pass. Group/student fields blank. A separate
review verified aggregate/per-class metrics,23fold allocations, corruption
results, metadata and actual learning curves. No report P1/P2finding.

Presentation:14editable slides, author-rendered and visually inspected. Review
found one P2:slide11/notes did not explicitly carry the report's provisional DINO
classroom recommendation. Author correction is underway; fusion must remain the
predeclared study primary. Presentation-builder formatting corrected98f5365;
AST and literal values were preserved. Final changed slide QA remains pending.

Full native Standards and Spec reviews are running on010ffdb. Portable-package
assembly, runtime smoke and final integrated test/CI results remain pending.
