# Independent protocol review and resolutions

Review round 1 completed before scored experiments on 2026-10-27.
No model scores were available during the review.

1. **Source independence.** Freeze evidence-backed families before folds.
   Resolve three ambientCG generator overlaps. Conservatively co-group
   wood_table and wood_table_worn pending provenance clarification; this is a
   precaution, not a claim of byte identity. Validate class-pure global groups.
2. **Endpoint and weighting.** Primary target is >=80% observed single-image
   accuracy, weighted over rendered images. Larger families therefore contribute
   more. Add class-balanced equal-family accuracy as a secondary measure,
   separately from multiview voting. Fitting uses declared per-image examples
   with class-balanced logistic loss; no implicit family reweighting.
3. **Folds.** Persist five outer folds and three inner folds within each outer
   training set. Assert all classes on both sides, disjoint families, complete
   image coverage and identical folds for candidates. Use a fixed seed and fail
   on invalid partitions; never select seeds by model performance.
4. **Selection.** Freeze five candidate representations including concatenation,
   exact pretrained revisions/weight hashes, deterministic preprocessing, token
   pooling formulas, C grid, tie rule and solver convergence behavior. Mean
   inner-fold macro-F1 is the selection criterion. All-data final selection
   uses the same grouped inner selector, never the outer leaderboard. No learned
   probability calibration is included.
5. **Uncertainty.** Use class-stratified source-family bootstrap, retaining all
   renders and repeated family draws. OOF intervals resample fixed predictions:
   they are approximate and exclude refitting and selection variability. Fresh
   holdout has only 21--35 independent groups despite many rendered views.
   Achieving an 80% point estimate does not imply a population lower bound of80%.
6. **Rendering and taxonomy.** Freeze EEVEE version, color management, UV scale,
   normal convention, shader rules, camera/light/framing envelope and30 views per
   source. Validate on old development materials first. Clearer ceramic/concrete
   eligibility makes fresh holdout a restricted endpoint; describe any domain
   change rather than implying identical distributions. Exclusions and source
   replacements use data quality only.
7. **Identity.** Bind render resume to scene/source/config hashes; bind features
   to ordered image content, weights, preprocessing and descriptor settings;
   bind predictions to class order, labels, families, folds and fitting config.
   Reject duplicates, missing joins, nonfinite features and invalid probabilities.

These resolutions were sent to implementation owners before scored runs. A
later implementation review must verify their execution, not merely this plan.

Sources: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
[cross-validation variance limitations](https://www.jmlr.org/papers/v5/grandvalet04a.html).
