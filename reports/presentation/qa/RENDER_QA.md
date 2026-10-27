# Presentation validation record

Validated on 27 October 2026.

- All 14 slides were opened and rendered by Microsoft PowerPoint to 1600 x 900 PNGs and inspected individually at full size.
- All 14 PDF pages were independently rendered with PyMuPDF and inspected individually. A spacing issue on page 3 was corrected and the slide and PDF were rendered and inspected again.
- Final text-overflow check: zero findings. Package integrity and geometry checks: zero findings. Required tables are native PowerPoint tables.
- Both learning-curve charts and the corruption chart are native editable charts. The chart-data gate verifies their category/value caches against exact embedded workbook cells. The three chart titles pass the title gate.
- All three chart objects were resized in a separate PowerPoint copy, saved, and reopened successfully. The delivered file was unchanged by this editability test.
- Dataset illustrations and prediction examples match the original report selection records. Example labels match saved predictions. Four reported accuracies match the raw evaluations. No new model fitting or prediction was performed.
- Fourteen slide notes include sources and narrative guidance. The speaking schedule totals 600 seconds, as a rehearsal choice rather than a prescribed duration. The group-member field and author metadata are empty.

The delivered deck has consistent titles and footers, readable tables, labelled axes and legends, preserved image proportions, and no observed clipping or accidental overlap. Blank title-page space is reserved for the presenters' names.

The skill's Artifact Tool import step could not run because its runtime and workspace dependency loader are unavailable on this host. Native PowerPoint opening, rendering and edit/save/reopen checks were used, together with the skill's independent package, geometry and chart validators. Google Slides and LibreOffice compatibility were not tested.

Material Studio itself was not tested during this task. The parent must verify the actual app navigation, live inference, loading and stale-result behaviour on the classroom device before rehearsal sign-off. The guide provides a stored-results fallback and prohibits presenting cached predictions as live inference.

## Classroom recommendation correction

Slide 11, its embedded notes and the speaking guide now provisionally recommend fine-tuned DINOv2 for classroom use with comparable renders, retain fusion as the predeclared study primary, and require independent confirmation before broader deployment. PowerPoint COM rebuilt and exported the deck successfully. The updated slide and its PDF page were inspected at full size and have no observed clipping or overlap.

Compared with commit `89044ad`, only slide 11 changes in the content specification, PowerPoint PNGs and PDF page pixels. The other 13 PDF pages are pixel-identical at 1600 x 900, and their PowerPoint PNGs are byte-identical. The slide 11 metrics table, all chart XML and all embedded workbook contents are unchanged. Author fields remain blank. Standard validation and Ruff checks pass. `recommendation_update.json` records these comparisons. No Material Studio UI interaction occurred.

## Minimal app navigation update

The demo is slide 13 in this deck. Its labels and notes now follow Classify, Study, Samples and About, with the compact toolbar, concise class/confidence result and full probabilities under Details described in the notes and classroom guide. PowerPoint COM rebuilt the deck successfully. Slide 13 and its PDF page were inspected at full size with no observed clipping or overlap.

Compared with commit `b27ceac`, only slide 13 changes in the content specification, PowerPoint PNGs and PDF pixels. The other 13 pages are unchanged. Metrics tables, chart XML, workbook contents and the slide 11 recommendation are unchanged. Standard validation and Ruff checks pass. `demo_navigation_update.json` records the comparisons. The redesigned Material Studio app was not controlled or visually verified, and the guides retain that limitation.
