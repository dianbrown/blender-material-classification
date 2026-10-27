# Bounded development fine-tuning follow-up

The initial frozen selection procedure achieved 79.3968% nested OOF accuracy
and 0.795485 macro-F1, against 68.8571% for its matched ConvNeXt control. Preserve
those results and their original protocol. The fresh-source holdout has not
been evaluated.

To pursue the user's >=80% target, preregister a finite exploratory follow-up:
DINOv2 small at 336 pixels, CLS plus mean patch features, two head-only epochs
then last-two-block fine-tuning, backbone learning rates 3e-6 and 1e-5, maximum
ten epochs. Use the same three all-development grouped folds as the frozen
final selector, with checkpoint selection and patience three inside development
validation. This score is a development selection score, not an unbiased nested
estimate. This follow-up was motivated by the initial OOF outcome; disclose
that adaptive research history.

Choose learning rate by mean development macro-F1, tie toward smaller learning
rate; choose final full-data epoch budget by rounded median best epoch across
the selected learning rate's folds, with at least two epochs. Compare with the
frozen selector on identical development folds for the final recommendation.
Lock every final model before the fresh holdout is evaluated. Preserve both
models, all histories and the original 79.4% endpoint. No further grid expansion
is included in this amendment.

The matched ResNet baseline was separately declared before any V3 score was
read, to fulfill the assignment's third-architecture requirement. It remains a
comparison baseline rather than a retrospectively substituted primary model.
