# Material Studio classroom demo

The verified native GPUI app uses **Classify / Study / Samples / About** navigation. Native visual checks and packaged CPU/CUDA inference passed on the development machine. Rehearse on the classroom machine before presenting.

## Before class

- Open `reports/presentation/material_classification.pptx` and keep the PDF available as a slide fallback. Fill in the deliberately blank group-member field only with the presenters' actual names.
- Confirm the lecturer's duration. The guide budgets about ten minutes, including a short demo, but the brief does not prescribe ten minutes.
- Agree speaking hand-offs. Every member must participate and understand the full study.
- Open `dist/MaterialStudio-release/MaterialStudio.exe`. Keep the complete portable folder together; no separate Python installation is needed.
- Confirm the required local checkpoints, preprocessing metadata and sample images are available without a network connection. Use the machine's supported device. The published GPU timings do not predict classroom CPU latency.
- Rehearse one complete image prediction per supported model. Verify that changing the input or model cannot leave an old result appearing as a new prediction. Confirm the concise class/confidence result and the full probabilities under Details match the displayed model and image.
- Check that the loading state is visible while inference runs, and that failures clearly report unavailable runtime or model files.
- Confirm Study and Samples work with their local stored data, and About is reachable from navigation. Open the slide PDF independently in case the application cannot start.

## Approximately 50-second demo

1. **Classify:** use the compact bottom toolbar beneath the single image: **Open image**, **Samples**, the **model menu**, then **Classify**. Choose an image or sample and model, then start classification. Point out the loading state. After genuine runtime completion, explain the concise predicted class and confidence. Open **Details** for the full probability distribution. Say that these are model scores and can be confidently wrong.
2. **Study:** show the four stored clean-test results. Identify the frozen primary at 80.16% and the exploratory DINOv2 comparison at 86.83%. Show the primary confusion matrix, especially ceramic predicted as stone.
3. **Samples:** browse examples across the seven classes. Explain that related render views share a source group and must stay together during splitting. **About** is also available in navigation, if needed during questions.
4. Return to the conclusion: the point target is met on this clean test, but wide uncertainty and JPEG sensitivity limit the claim. Provisionally recommend fine-tuned DINOv2 for classroom use with comparable renders; retain fusion as the predeclared study primary; independently confirm before broader deployment.

If inference takes longer than the available time, move to Study and identify it explicitly as the stored evaluation. Resume the live prediction only when it actually finishes. Do not describe a stored or cached prediction as live inference.

## Offline fallback

Say: "The live runtime is unavailable on this machine. I will show the stored experimental results and saved example predictions."

Use the app's Study and Samples views if available. Otherwise use slides 7-11 and the saved correct/incorrect examples on slide 9. These are measured stored results, not newly computed predictions. The PDF contains all of the evidence needed to complete the explanation without the application.

## Integration sign-off for the parent

- [x] Build and open the final GPUI application, including the portable executable outside the checkout.
- [x] Confirm Classify, Study, Samples and About navigation, plus the single image and compact bottom toolbar with Open image, Samples, model menu and Classify.
- [x] Run fresh inference for a file-picker image and built-in samples on this machine; repeat on any different classroom device.
- [x] Check model switching, the concise class/confidence result, Details probabilities, loading, failure handling and stale-result rejection through native interaction and integration tests.
- [x] Check stored comparison metrics and confusions against the slide values.
- [ ] Rehearse the offline fallback and complete presenter hand-offs.

The user retains the Kaggle upload and dataset-release decision. No upload is part of this demo or presentation build.
