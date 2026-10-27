# Assignment coverage and remaining items

The current student-facing deliverables are the [academic PDF](project/report.pdf) and [editable Markdown](project/report.md). The report has 14 main pages, one reference page and four appendix pages. Prior reports remain unchanged as internal records; they are not the recommended final submission.

This assessment is against all seven pages of the supplied ITRI626 practical-project brief. It is an evidence map, not a claim that every rubric item is fully satisfied or a predicted mark.

| Requirement | Evidence in the final report | Status / limitation |
| --- | --- | --- |
| Title, members, student numbers, abstract | Page 1 | Title and abstract complete. Three short name/number fields intentionally blank; students must fill them. |
| Problem, value, target classes | Section 1 | Covered; visual appearance classification is distinguished from physical composition. |
| Dataset source, links, licence and ethics | Sections 2-2.1 | Public source catalogues and CC0 terms cited. Rendered-dataset licence and public release remain unset/unapproved; no public dataset download is claimed. |
| Per-class counts before/after cleaning | Table 1 | 450/450 development and 90/90 test images per class. No invented removals or relabelling. Source exclusions are distinguished from deleting rendered images. |
| Labels, corrupt files, duplicates, imbalance | Section 2 | Recorded integrity checks and label-consistency scope stated. No independent physical-composition ground truth claimed. |
| Representative images for every class | Figure 1 | Seven actual test renders; deterministic path-based selection. |
| Three different model families and transfer learning | Section 3, Table 2 | ConvNeXt, ResNet, DINOv2. Fusion is an extension. Frozen feature classifiers are fitted models. |
| Same split / fair comparison | Section 4, Table 3, Appendix C | Same 3,150-image final development pool and 630-image test set. ResNet/frozen outer folds match; DINO/frozen selection folds match. Tuning procedures and budgets differ and are disclosed. This is not a fully identical tuning-split comparison. |
| Exact split sizes, class distributions, seed and leakage control | Sections 2, 4; Tables 1, 3, C1, C2 | Persisted group plans; all 15 nested inner partitions included. Appendix class vectors specify exact validation counts and exact training complements. Machine-readable fold counts accompany the report. |
| Preprocessing and training-only augmentation | Section 4 | Model-specific size, interpolation, crop, RGB, normalisation and augmentation parameters stated. |
| Hyperparameters, versions and hardware | Section 4.1 | Optimiser, LR, schedule, decay, smoothing, batches/accumulation, epochs, patience, class weighting, seeds, pretrained variants, library versions and GPU memory stated. Original renderer version is incompletely recorded. |
| Accuracy, precision, recall, macro F1 | Table 5 | All four saved pipelines on the common test population; paired uncertainty disclosed. |
| Per-class precision/recall/F1 and confusion matrices | Appendices A and B | All three main families and fusion; 28 class rows and four confusion matrices. |
| Training and validation loss; validation F1 curve | Figures 3-5 | DINOv2 has actual train/validation loss and F1. ResNet has actual training loss/accuracy only, with one final outer evaluation per fold. Required epoch-wise ResNet validation curves are missing. Frozen heads have search curves, no neural epochs and no saved solver-iteration loss histories; this is not presented as satisfying all requested learning curves. |
| Correct and incorrect examples with interpretation | Figure 6, Section 6 | Three correct and three incorrect primary predictions, with saved probabilities. Selection and identities retained in project/data/examples.json. |
| Limitations, recommendation and conclusion | Sections 6-7 | Fusion remains predeclared primary. DINOv2 is a provisional classroom recommendation from observed accuracy/latency, without superiority or app-validation claims. |
| References and concise structure | 14 main pages; References | Academic structure follows the brief; provider/model/software references included. |
| Executed notebook with visible outputs | [Existing results notebook](../notebooks/05_v3_results.ipynb) | Preserved, not rewritten by this task. Notebook exists with outputs; final submission consistency should be checked by the submission owner. |
| Source scripts and split files | Existing source tree; persisted split plans; new report builder | Preserved. Report assembly does no training or inference. |
| README with setup, data access and reproduction steps | Existing README, outside this task's edit scope | Owner must update final-report links to project/report.pdf and confirm all setup/data-access instructions. The report body intentionally excludes shell commands. |
| Slides and oral presentation | Not supplied by this task | Pending. All three members must participate and answer questions. No claim that the 10-mark presentation component is complete. |
| Topic approval, group details, deadline and submission destination | Not established by the saved experiments | Students/submission owner must verify with the lecturer/eFundi. |
| Dataset/weights access where large files are omitted | Provider links and existing reproduction documentation | Source assets are accessible. Final access instructions for rendered data/weights and dataset licence/release decisions remain owner responsibilities. |
| Native classroom application | Parallel work, not report evidence | No application verification claimed; report results do not depend on it. |

The material assessment gaps are the missing ResNet validation curves, the inapplicable/unrecorded epoch histories for frozen heads, unequal tuning procedures, pending presentation, blank student details and outstanding submission/release administration. Completing the prose does not remove these gaps.
