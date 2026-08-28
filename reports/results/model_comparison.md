# Model comparison

Values are copied from completed artifacts; `—` means the source metric was unavailable.

| family | model | pretrained | parameters | best_epoch | val_accuracy | test_accuracy | test_macro_f1 | test_weighted_f1 | cv_macro_f1_mean | cv_macro_f1_std | train_accuracy | train_macro_f1 | accuracy_gap | macro_f1_gap | inference_ms_per_image | model_size_mb | training_time_seconds | overfitting_indicator |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline CNN | custom_cnn | no | 3389287 | 3 | 0.614 | 0.498 | 0.490 | 0.490 | — | — | 0.668 | 0.657 | 0.054 | 0.045 | 1.377 | 38.853 | 396.277 | not_flagged |
| residual CNN | resnet50 | yes | 23522375 | 7 | 0.843 | 0.698 | 0.683 | 0.683 | 0.717 | 0.109 | 1.000 | 1.000 | 0.157 | 0.167 | 6.443 | 269.631 | 557.305 | potential_overfitting |
| modern CNN | convnext_tiny | yes | 27825511 | 15 | 0.857 | 0.612 | 0.569 | 0.569 | 0.787 | 0.113 | 1.000 | 1.000 | 0.143 | 0.152 | 6.489 | 318.655 | 904.606 | not_flagged |
| vision transformer | vit_b16 | yes | 85804039 | 19 | 0.831 | 0.600 | 0.562 | 0.562 | 0.745 | 0.103 | 0.999 | 0.999 | 0.168 | 0.175 | 7.297 | 982.137 | 1511.295 | not_flagged |
