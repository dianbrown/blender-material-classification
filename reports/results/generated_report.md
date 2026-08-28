# Generated evaluation report

This report is generated exclusively from cached validation/TEST metrics, run metadata, histories and development grouped-CV results. It does not evaluate a dataset.

## Overfitting indicator

A model is marked `potential_overfitting` only when the selected-epoch train-accuracy minus validation-accuracy exceeds 0.10 **and** validation loss rises above an earlier minimum while training continues. This is a screening indicator, not a statistical conclusion.

## Statistical caution

Grouped-CV values are reported as mean +/- sample standard deviation. Paired fold comparisons are only defensible when every compared model has the same five fold IDs; n=5 is small and should be interpreted descriptively.

## Artifacts

- Comparison CSV: `reports\results\model_comparison.csv`
- Comparison Markdown: `reports\results\model_comparison.md`
- Dashboard: `reports\figures\comparison\model_comparison_dashboard.png`
- Rankings: `reports\figures\comparison\model_rankings.png`

### Descriptive paired fold differences

| Model A | Model B | Folds | Mean F1 difference (A−B) | SD |
|---|---|---:|---:|---:
| convnext_tiny | resnet50 | 5 | 0.070 | 0.049 |
| convnext_tiny | vit_b16 | 5 | 0.043 | 0.016 |
| resnet50 | vit_b16 | 5 | -0.028 | 0.049 |
