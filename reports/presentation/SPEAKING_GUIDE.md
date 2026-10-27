# Speaking guide

A 14-slide rehearsal plan of approximately 10 minutes. The lecturer has not assigned a duration in the supplied brief. Adapt the pace when it is announced. Group-member fields are intentionally blank. All members must participate and be ready to answer questions about the full project.

Suggested unnamed hand-offs: slides 1-5, slides 6-10, slides 11-14. Adjust these for equal speaking time and actual group size.

## 1. Recognising materials in Blender renders (20 seconds)

Introduce the problem as visual appearance classification in a controlled rendering setting. The practical aim is to assist material browsing or organisation. Leave the group-member fields blank until the presenters supply their names. This rehearsal plan is about ten minutes, not a lecturer-assigned duration.

Sources:
- ITRI626 practical project brief, sections 2, 10 and 12.2
- reports/v3/report.md, Abstract and section 1

## 2. The problem and seven classes (40 seconds)

An image receives one of seven appearance labels. This could help search a rendered asset library. The categories are wood, metal, stone, concrete, ceramic, fabric and leather. Each picture is an actual held-out render selected by path, without filtering by correctness. The same sphere makes texture and reflectance informative, but this does not identify physical composition. In the development collection, ceramic includes mosaic or terrazzo-like appearances and concrete includes plaster or painted walls.

Sources:
- reports/results/v3/report_inputs/gallery_selection.json
- reports/v3/report.md, sections 1-2

## 3. Dataset provenance and group separation (50 seconds)

The study has 3,780 balanced RGB images at 512 by 512 pixels, with 30 rendered views per material. Development contains 450 images per class and the fresh holdout 90 per class. Source textures come from Poly Haven and ambientCG under CC0. They include scanned and procedural assets, so we do not describe all sources as independent physical scans. Labels follow audited asset identity and appearance. Known generator relatives are merged into groups. Thirty views of a texture remain correlated. No known development-to-holdout family or exact-map reuse remains, but unknown physical ancestry cannot be ruled out. The rendered-dataset release licence is still an owner decision.

Sources:
- reports/results/v3/data_audit.json
- metadata/v3_holdout/source_identity_audit.json
- reports/results/v3/report_inputs/study_counts.csv
- https://polyhaven.com/license
- https://docs.ambientcg.com/license/

## 4. Three model families, four pipelines (50 seconds)

ConvNeXt is a modern convolutional model, ResNet is a residual convolutional network, and DINOv2 uses a vision transformer. All use pretrained weights. A frozen ConvNeXt control and fixed ResNet baseline provide family comparisons. The primary fuses DINOv2 and ConvNeXt features and fits a scaled, class-balanced logistic classifier. DINO pools CLS, patch mean and patch standard deviation at two resolutions, excluding register tokens from patch pooling. The partially fine-tuned DINO model is exploratory. These are complete pipeline comparisons because pretraining data, resolution and compute budgets differ.

Sources:
- reports/results/v3/frozen_initial/protocol.json
- configs/v3/finetune.json
- reports/results/v3/resnet_baseline/config.json
- reports/v3/report.md, section 5
- https://arxiv.org/abs/2304.07193
- https://arxiv.org/abs/2201.03545
- https://arxiv.org/abs/1512.03385

## 5. Experimental setup (55 seconds)

The frozen selection uses five outer source-group folds and three inner grouped folds, seed 20261027. Inner folds select five representations and C in 0.01, 0.1, 1 and 10 by Macro F1. The scaler and classifier are fitted inside each training partition. All original images are development data because the historical test had already been consulted. The neural comparison uses the same development pool and the same final holdout. DINO searches two learning rates in a bounded three-fold exploratory follow-up. Preprocessing converts to RGB, uses model-specific resize and centre crop and pretrained normalisation. Neural training alone adds small random crops, horizontal flips and mild colour jitter. ResNet uses AdamW with a cosine schedule for ten fixed epochs. DINO uses AdamW and development early stopping, then a fixed five-epoch final fit. Both have effective batch size 32, label-smoothed cross entropy and weight decay 0.01. Development Macro F1 selected the frozen primary before any fresh scores. Four fixed models then received one clean evaluation each.

Sources:
- reports/results/v3/frozen_initial/protocol.json
- configs/v3/finetune.json
- reports/results/v3/resnet_baseline/config.json
- reports/results/v3/model_choice.json
- reports/results/v3/final_model_lock.json

## 6. Actual learning curves (50 seconds)

These are the complete recorded histories for all three development folds at the selected DINO backbone learning rate of 1e-5. Solid curves show training and dashed curves development validation. No curve is a fresh-test result. The training fit becomes strong while validation varies by source fold. Validation chooses the learning rate and epoch count, so it is optimistic for that selection. The six-fit search also includes 3e-6, whose mean best development Macro F1 was 0.76250 versus 0.77248 at 1e-5. Best epochs 6, 5 and 3 imply five final epochs. ResNet has actual training-only logs for its fixed schedule and no epoch-wise validation curve. The frozen logistic head has regularisation-search diagnostics, not neural learning epochs.

Sources:
- reports/results/v3/finetune_development/lr_1e-05_fold_0/progress.json
- reports/results/v3/finetune_development/lr_1e-05_fold_1/progress.json
- reports/results/v3/finetune_development/lr_1e-05_fold_2/progress.json
- reports/results/v3/finetune_development/selection.json
- reports/results/v3/resnet_baseline/final_history.csv

## 7. Clean test results on unseen sources (55 seconds)

The primary achieves 505 correct images out of 630, or 80.16%, with Macro F1 0.7952. Its 95% source-group bootstrap interval is 66.03 to 93.65%. The point target of 80% is met but the lower bound is below 80%. Fine-tuned DINOv2 records 86.83% and Macro F1 0.8631. It stays an exploratory comparison. Its paired accuracy advantage over the primary is 6.67 percentage points with interval minus 2.06 to plus 16.19, which includes zero. Thus the ranking does not establish superiority. The control interval is 61.59 to 88.73%, ResNet 63.65 to 90.48%, and DINOv2 76.98 to 95.87%. Intervals use 10,000 class-stratified cluster draws over 21 sources, conditional on these saved predictions.

Sources:
- reports/results/v3/fresh/frozen_selected/evaluation.json
- reports/results/v3/fresh/frozen_control/evaluation.json
- reports/results/v3/fresh/resnet_fixed/evaluation.json
- reports/results/v3/fresh/dinov2_finetuned/evaluation.json
- reports/results/v3/report_inputs/paired_fresh.csv

## 8. Primary errors and per-class performance (55 seconds)

Read the matrix by rows: true class first, prediction second. Every row has 90 images. Ceramic is often called stone, with 30 of its 90 images following that error. Leather is called concrete in 22 cases. Wood, metal and fabric reach perfect recall on the three sampled sources per class, which is not a population guarantee. Ceramic precision is 1.000 but recall is only 0.578, illustrating that missing many examples can coexist with few false positive labels. All values are saved point metrics, not newly scored results.

Sources:
- reports/results/v3/fresh/frozen_selected/image_confusion_matrix.csv
- reports/results/v3/fresh/frozen_selected/image_per_class.csv

## 9. Correct and incorrect predictions (40 seconds)

These four examples use the saved primary predictions. The correct wood and metal examples are from the unfiltered class gallery. The ceramic and leather failures come from the report error gallery. They are selected illustrations rather than a representative performance estimate. Point to visual ambiguity in texture and reflectance as a plausible explanation, not a causal attribution. Model probabilities can be high for wrong answers and are not calibrated certainty. Show the predicted class and confidence for each.

Sources:
- reports/results/v3/report_inputs/gallery_selection.json
- reports/results/v3/fresh/frozen_selected/predictions.csv

## 10. JPEG compression exposes a weakness (40 seconds)

These saved corruption results use the same 630 images and five deterministic transforms. The primary drops from 80.16% clean accuracy to 59.05% at JPEG quality 30, a 21.11 percentage point loss. DINOv2 falls from 86.83% to 75.40%. Primary corruption tests were predeclared, whereas the DINOv2 corruption comparison was applied after observing clean and primary-corruption scores and is explicitly post hoc. Compression is the largest degradation here, but these tests do not substitute for new scenes or real-camera data.

Sources:
- reports/results/v3/robustness/primary/comparison.csv
- reports/results/v3/robustness/dinov2_exploratory/comparison.csv
- reports/results/v3/exploratory_robustness_amendment.json

## 11. Efficiency and model recommendation (45 seconds)

On the recorded RTX 4070 Laptop GPU, ResNet has the lowest warm batch-one median latency. The primary is slower because it uses multiple backbone passes. Required storage includes the backbones, not just the small classifier head. Timing starts from predecoded RGB and includes preprocessing and probabilities, but excludes disk reading and decoding. These are CUDA measurements, not classroom CPU measurements. Provisionally recommend fine-tuned DINOv2 for classroom use with comparable renders; retain fusion as the predeclared study primary; independently confirm before broader deployment. DINOv2 has higher observed accuracy and lower measured latency than the primary, but the paired uncertainty does not establish superiority. Choose ResNet only if the speed trade-off is acceptable.

Sources:
- reports/results/v3/benchmarks/frozen_selected/benchmark.json
- reports/results/v3/benchmarks/frozen_control/benchmark.json
- reports/results/v3/benchmarks/resnet_fixed/benchmark.json
- reports/results/v3/benchmarks/dinov2_finetuned/benchmark.json

## 12. Limitations and next experiment (40 seconds)

Only 21 new groups, three per class, give wide uncertainty despite 630 images. Asset provenance is audited but some common physical ancestry and backbone pretraining overlap may remain unknown. Sources combine scanned and procedural textures, and restricted class definitions and renderer differences affect transfer. No genuine photograph test has been performed. Pipeline comparison also combines architecture, resolution, pretraining and compute changes. The next experiment should be frozen before data collection, add more independent sources in ceramic, concrete, stone and leather, vary scene and camera, include real photographs, and test compression systematically. Reusing this now-scored holdout for tuning would require another fresh assessment.

Sources:
- reports/v3/report.md, sections 3, 5 and 16
- reports/results/v3/provenance_verification.json

## 13. Material Studio demo and conclusion (50 seconds)

Use the separate classroom guide. Navigation is Classify, Study, Samples and About. Classify shows one image with a compact bottom toolbar: Open image, Samples, the model menu and Classify. Choose an image or sample and model, start classification, and let the loading state complete. Explain the concise class and confidence result, then open Details for the full probability distribution. Only call this a live prediction after the actual runtime has completed inference for that input. In Study, compare stored model metrics and confusions, then browse the class gallery in Samples. About is available in navigation. The redesigned minimal app has not yet been visually verified here. If runtime inference is unavailable, explicitly announce the offline fallback and use stored results or slide 9 examples. Conclude that source-aware evaluation supports the primary point target, but uncertainty and compression weakness limit broad claims.

Sources:
- docs/CLASSROOM_DEMO.md
- reports/results/v3/fresh/frozen_selected/evaluation.json

## 14. Sources (10 seconds)

Experimental claims come from the saved evaluations, prediction tables, training histories and audit records cited in each slide note. These papers explain the model families and evaluation principles. Provider pages establish the source-texture licence, not a licence decision for the rendered collection. All presenters should be able to discuss the full study. Invite questions.

Sources:
- ITRI626 practical project brief, sections 10 and 12.2
- reports/v3/report.md, References

## Questions to rehearse

- Why group by source? Multiple views of one texture share information and cannot be treated as independent samples.
- Why retain the primary at 80.16%? Development evidence determined the model before the fresh test. Picking from the test ranking would change the meaning of that assessment.
- Is DINOv2 better? Its observed accuracy is higher, but the paired interval includes zero and the follow-up is exploratory.
- Why does an 80% target coexist with a 66.03% lower bound? A point target is not a guaranteed population lower bound.
- Is the dataset entirely scanned? No. The provider collection mixes scanned and procedural sources and known relatives are grouped.
- What does model probability mean? A model score for the available classes, not a validated guarantee that a prediction is correct.
- Why can validation curves look optimistic? They were used to select learning rate and epoch count.
- Can it classify real photos? This study has not validated that use.
- Who publishes the dataset? The user retains the Kaggle upload and rendered-dataset licensing decision. No upload is performed by this presentation workflow.
