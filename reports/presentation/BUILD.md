# Build and review

The deck contains 14 slides, with native editable text, six native tables, three native charts and three embedded chart workbooks. Images are verified copies of actual renders selected for the existing report. The author field and author metadata are blank.

From the repository root, with Python 3, `lxml`, Microsoft PowerPoint and Windows PowerShell available:

```powershell
python scripts/report/build_presentation.py --data-root C:/path/to/holdout_dataset
python scripts/report/build_presentation.py --verify-only --skill-dir C:/path/to/presentations/skill
```

Verification also uses `PyMuPDF`. If the render dataset is absent, the build can reuse the checked-in image assets, which are checked against the original report selection records. No model fitting, new prediction or GPU execution occurs. All metrics and charts consume saved records under `reports/results/v3`.

PowerPoint creates text and tables through COM. Native DrawingML chart parts and their embedded workbooks preserve the exact recorded values. The final file is reopened by PowerPoint and exported to PDF and 1600 x 900 PNGs. This route is used because the skill's workspace dependency loader and `@oai/artifact-tool` runtime are unavailable on this host, and hidden-session COM chart creation failed. The skill's independent package, geometry, chart-data and chart-title checks still run. No Artifact Tool import check is claimed.

`deck.json` is the generated content and layout specification. `SPEAKING_GUIDE.md` contains timed notes and source paths, also embedded in the PowerPoint speaker notes. `renders/` contains final PowerPoint renders. `qa/validation.json` records checks. The local base deck and duplicate PDF render images are ignored.

The remaining integration step is to rehearse the completed Material Studio app on the classroom device using `docs/CLASSROOM_DEMO.md`. The presentation build does not verify the app runtime.
