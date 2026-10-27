# Material Studio classroom demo

This is an expected workflow for the native GPUI app under integration. It has not yet been verified against the final application. Complete the rehearsal checklist below before claiming that live inference works.

## Before class

- Open `reports/presentation/material_classification.pptx` and keep the PDF available as a slide fallback. Fill in the deliberately blank group-member field only with the presenters' actual names.
- Confirm the lecturer's duration. The guide budgets about ten minutes, including a short demo, but the brief does not prescribe ten minutes.
- Agree speaking hand-offs. Every member must participate and understand the full study.
- Launch the final Material Studio build using the parent project's verified run instructions. This guide does not assume an executable name or invent a launch command.
- Confirm the required local checkpoints, preprocessing metadata and sample images are available without a network connection. Use the machine's supported device. The published GPU timings do not predict classroom CPU latency.
- Rehearse one complete image prediction per supported model. Verify that changing the input or model cannot leave an old result appearing as a new prediction. Confirm probabilities match the displayed model and image.
- Check that the loading state is visible while inference runs, and that failures clearly report unavailable runtime or model files.
- Confirm Results and Dataset work with their local stored data. Open the slide PDF independently in case the application cannot start.

## Approximately 50-second demo

1. **Classify:** choose a built-in sample or an image, select a model, and start classification. Point out the loading state. After genuine runtime completion, explain the predicted class and probability distribution. Say that these are model scores and can be confidently wrong.
2. **Results:** show the four stored clean-test results. Identify the frozen primary at 80.16% and the exploratory DINOv2 comparison at 86.83%. Show the primary confusion matrix, especially ceramic predicted as stone.
3. **Dataset:** browse examples across the seven classes. Explain that related render views share a source group and must stay together during splitting.
4. Return to the conclusion: the point target is met on this clean test, but wide uncertainty and JPEG sensitivity limit the claim.

If inference takes longer than the available time, move to Results and identify it explicitly as the stored evaluation. Resume the live prediction only when it actually finishes. Do not describe a stored or cached prediction as live inference.

## Offline fallback

Say: "The live runtime is unavailable on this machine. I will show the stored experimental results and saved example predictions."

Use the app's Results and Dataset views if available. Otherwise use slides 7-11 and the saved correct/incorrect examples on slide 9. These are measured stored results, not newly computed predictions. The PDF contains all of the evidence needed to complete the explanation without the application.

## Integration sign-off for the parent

- [ ] Build and open the final GPUI application.
- [ ] Confirm actual Classify, Results and Dataset navigation and revise the guide if labels differ.
- [ ] Run fresh inference for an image and a built-in sample on the classroom device.
- [ ] Check model switching, probabilities, loading, failure messages and stale-result handling.
- [ ] Check stored comparison metrics and confusions against the slide values.
- [ ] Rehearse the offline fallback and complete presenter hand-offs.

The user retains the Kaggle upload and dataset-release decision. No upload is part of this demo or presentation build.
