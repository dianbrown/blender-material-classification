use crate::{
    config::Config,
    data::{CLASSES, Model, Study, title},
    worker::{Event, Prediction, Worker},
};
use gpui::{prelude::*, *};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

const BG: u32 = 0x10171f;
const PANEL: u32 = 0x19232e;
const BORDER: u32 = 0x2b3a47;
const TEXT: u32 = 0xe8eff5;
const MUTED: u32 = 0x99acbd;
const ACCENT: u32 = 0x67d9cc;
actions!(
    studio,
    [
        OpenImage,
        Cancel,
        Classify,
        ShowClassify,
        ShowResults,
        ShowDataset,
        ShowAbout
    ]
);
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Classify,
    Results,
    Dataset,
    About,
}
pub struct Studio {
    config: Config,
    study: Option<Study>,
    resource_error: Option<String>,
    worker: Worker,
    ready: bool,
    status: String,
    page: Page,
    selected_model: usize,
    results_model: usize,
    image: Option<PathBuf>,
    image_label: String,
    prediction: Option<Prediction>,
    error: Option<String>,
    active_id: Option<String>,
    serial: u64,
    started: Option<Instant>,
    focus: FocusHandle,
    scroll: ScrollHandle,
    _poll: Task<()>,
}

impl Studio {
    fn results_page(&self, cx: &mut Context<Self>) -> Div {
        let mut content = div()
            .flex()
            .flex_col()
            .gap_5()
            .child(self.header(
                "The study, in perspective.",
                "Measured results on independent held-out material sources.",
                "STORED STUDY RESULTS",
            ))
            .child(self.model_picker(cx, true));
        if let Some(study) = &self.study {
            if let Some(model) = study.models.get(self.results_model) {
                content = content.child(div().flex().gap_5().items_start().child(self.confusion(model).flex_1())
                .child(panel().w(px(330.)).flex_shrink_0().child(eyebrow("UNCERTAINTY & COST")).child(heading(model.display_name.clone()))
                    .child(metric("Accuracy", format!("{:.2}%", model.accuracy * 100.), format!("{:.0}% CI  {:.1}–{:.1}%", model.accuracy_ci.level*100., model.accuracy_ci.low*100., model.accuracy_ci.high*100.)))
                    .child(metric("Macro F1", format!("{:.3}", model.macro_f1), format!("{:.0}% CI  {:.3}–{:.3}", model.macro_f1_ci.level*100., model.macro_f1_ci.low, model.macro_f1_ci.high)))
                    .child(metric("Warm study latency", format!("{:.1} ms", model.latency.median_ms), format!("Median · p95 {:.1} ms · batch {}", model.latency.p95_ms, model.latency.batch_size)))
                    .child(metric("Deployment size", format!("{:.1} MiB", model.size.total_deployment_bytes as f64 / 1048576.), format!("Saved artifact {:.1} MiB", model.size.artifact_bytes as f64 / 1048576.)))
                    .child(muted(format!("{} repetitions · {}", model.latency.repetitions, model.latency.device)))))
                .child(panel().child(eyebrow("HOW TO READ THESE RESULTS"))
                    .child(muted(format!("{} holdout images · {} materials · {} independent source groups. Confidence intervals use source-group resampling, so visually related renders stay together.", study.dataset.holdout.images, study.dataset.holdout.materials, study.dataset.holdout.source_groups)))
                    .child(muted("The preselected frozen DINOv2 + ConvNeXt fusion is the primary study. Fine-tuned DINOv2 and ResNet-50 are comparisons; the demo default does not change the study's primary model."))
                    .child(muted("Robustness is limited to the study's synthetic sources and tested corruptions. Strong holdout accuracy does not establish recognition of arbitrary photographs, unseen domains or physical composition.")))
                .child(panel().child(eyebrow("PERFORMANCE MEASUREMENT")).child(muted("Warm models, predecoded RGB images, exact family preprocessing and device transfer. Includes network inference, the saved classifier and class probabilities. Excludes disk access, image decoding and model loading."))
                    .child(muted("Live demo timing also includes local image decoding and may differ from these stored benchmark measurements. Model loading is reported separately.")));
            }
        }
        content
    }
    fn confusion(&self, model: &Model) -> Div {
        let mut matrix = div().flex().flex_col().flex_shrink_0().gap_1();
        let mut labels = div()
            .flex()
            .items_center()
            .h(px(28.))
            .min_h(px(28.))
            .flex_shrink_0()
            .gap_1()
            .child(div().w(px(74.)).flex_shrink_0());
        for class in CLASSES {
            labels = labels.child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .line_height(px(20.))
                    .text_xs()
                    .text_color(rgb(MUTED))
                    .child(title(class)),
            );
        }
        matrix = matrix.child(labels);
        for (i, row) in model.confusion_matrix.iter().enumerate() {
            let sum = row.iter().sum::<u64>().max(1) as f64;
            let mut cells = div()
                .flex()
                .flex_shrink_0()
                .gap_1()
                .items_center()
                .child(div().w(px(74.)).text_sm().child(title(CLASSES[i])));
            for (j, value) in row.iter().enumerate() {
                let fraction = *value as f64 / sum;
                let color = if i == j {
                    rgba(0x47b6a900 | ((45. + 170. * fraction) as u32))
                } else {
                    rgba(0x6897cf00 | ((25. + 170. * fraction) as u32))
                };
                cells = cells.child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .h(px(42.))
                        .flex()
                        .justify_center()
                        .items_center()
                        .rounded_md()
                        .bg(color)
                        .text_sm()
                        .child(value.to_string()),
                );
            }
            matrix = matrix.child(cells);
        }
        panel().min_w_0().child(eyebrow("CONFUSION MATRIX")).child(heading("Where predictions agree"))
            .child(muted("Rows: true material    ·    Columns: predicted material")).child(matrix).child(muted("Counts on the held-out set. Brighter diagonal cells indicate more correct predictions."))
    }
    fn dataset_page(&self, cx: &mut Context<Self>) -> Div {
        let mut content = div().flex().flex_col().gap_5().child(self.header(
            "A material library.",
            "Seven appearance classes, rendered under controlled conditions.",
            "SYNTHETIC DATASET",
        ));
        if let Some(study) = &self.study {
            content = content.child(
                div()
                    .flex()
                    .gap_5()
                    .child(
                        panel()
                            .flex_1()
                            .child(eyebrow("DEVELOPMENT"))
                            .child(heading(format!(
                                "{} images",
                                study.dataset.development.images
                            )))
                            .child(muted(format!(
                                "{} materials · {} source groups",
                                study.dataset.development.materials,
                                study.dataset.development.source_groups
                            ))),
                    )
                    .child(
                        panel()
                            .flex_1()
                            .child(eyebrow("HELD-OUT EVALUATION"))
                            .child(heading(format!("{} images", study.dataset.holdout.images)))
                            .child(muted(format!(
                                "{} materials · {} source groups",
                                study.dataset.holdout.materials,
                                study.dataset.holdout.source_groups
                            ))),
                    ),
            );
            for (i, class) in study.dataset.classes.iter().enumerate() {
                let mut images = div().flex().gap_3();
                for (j, sample) in study
                    .samples
                    .iter()
                    .filter(|s| s.class_name == class.name)
                    .enumerate()
                {
                    let path = self.config.root.join("app/assets").join(&sample.path);
                    let label = sample.label.clone();
                    images = images.child(
                        div()
                            .id((SharedString::from(format!("gallery-{i}")), j))
                            .w(px(145.))
                            .flex()
                            .flex_col()
                            .flex_shrink_0()
                            .cursor_pointer()
                            .rounded_lg()
                            .overflow_hidden()
                            .border_1()
                            .border_color(rgb(BORDER))
                            .bg(rgb(0x111b25))
                            .child(thumbnail(path.clone(), 115.))
                            .child(
                                div()
                                    .px_2()
                                    .py_2()
                                    .min_h(px(48.))
                                    .flex_shrink_0()
                                    .text_xs()
                                    .line_height(px(16.))
                                    .child(label.clone()),
                            )
                            .on_click(cx.listener(move |s, _, _, cx| {
                                s.choose_image(path.clone(), label.clone(), cx);
                                s.page = Page::Classify;
                                s.scroll.set_offset(point(px(0.), px(0.)));
                            })),
                    );
                }
                content = content.child(
                    panel()
                        .flex_row()
                        .items_center()
                        .gap_6()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap_3()
                                .child(heading(title(&class.name)))
                                .child(muted(class.definition.clone()))
                                .child(muted(format!(
                                    "{} development · {} held-out images",
                                    class.development_images, class.holdout_images
                                ))),
                        )
                        .child(images),
                );
            }
            content = content.child(panel().child(eyebrow("DATASET SCOPE")).child(muted(study.dataset.scope.clone())).child(muted("Gallery images are development examples from distinct source groups, selected without model scores. Click any example to run a fresh local prediction.")));
            let mut credits = panel().child(eyebrow("SAMPLE CREDITS"));
            for sample in &study.samples {
                credits = credits.child(muted(format!(
                    "{} · {} · {}",
                    sample.label, sample.attribution, sample.license
                )));
            }
            content = content.child(credits);
        }
        content
    }
    fn about_page(&self) -> Div {
        div().flex().flex_col().gap_5().child(self.header("Material Studio", "A native workspace for exploring material recognition.", "LOCAL & NATIVE"))
            .child(panel().child(eyebrow("A CLASSROOM DEMONSTRATION")).child(heading("From appearance to prediction")).child(muted("Open a local image, choose one of four studied models, and inspect the full seven-class probability distribution. Compare the prediction with measured study results and explore the source material gallery.")))
            .child(panel().child(eyebrow("THE EXPERIMENT")).child(heading("Three model families, one shared task"))
                .child(muted("Frozen visual representations with a learned classifier; a fine-tuned ResNet-50; and partially fine-tuned DINOv2. The frozen DINOv2 + ConvNeXt fusion remains the preselected primary study. Other models provide complementary comparisons."))
                .child(muted("Source groups are kept independent across development and holdout. Rendered views of related materials must not be treated as independent evidence.")))
            .child(panel().child(eyebrow("INTERPRET WITH CARE")).child(heading("Appearance is not physical composition"))
                .child(muted("The dataset is synthetic and follows explicit archival class conventions. Lighting, geometry, texture scale and source selection affect predictions. Probability scores are model outputs, not a guarantee of correctness. This demo has not been validated for arbitrary real-world images or safety-critical use.")))
            .child(panel().child(eyebrow("BUILT FOR EXPLORATION")).child(muted("Native Rust interface rendered by GPUI. A persistent local Python process loads the saved models on demand. Images are processed on this computer; there is no remote inference service."))
                .child(muted("Ctrl + O opens an image · Ctrl + Enter classifies · Escape dismisses the current result · Ctrl + 1–4 switches views."))
                .child(muted("Native framework and dependency credits are included in THIRD_PARTY_NOTICES.txt with the application.")))
    }
}
fn metric(label: &str, value: String, note: String) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .pb_3()
        .border_b_1()
        .border_color(rgb(BORDER))
        .child(muted(label.to_owned()))
        .child(div().text_2xl().child(value))
        .child(muted(note))
}
impl Render for Studio {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content = if let Some(error) = &self.resource_error {
            panel()
                .child(heading("Study resources are unavailable"))
                .child(muted(error.clone()))
                .child(muted(
                    "Launch with --resources pointing to the supplied Material Studio folder.",
                ))
        } else {
            match self.page {
                Page::Classify => self.classify_page(cx),
                Page::Results => self.results_page(cx),
                Page::Dataset => self.dataset_page(cx),
                Page::About => self.about_page(),
            }
        };
        div()
            .size_full()
            .flex()
            .font_family("Segoe UI")
            .text_size(px(14.))
            .text_color(rgb(TEXT))
            .bg(if self.config.opaque {
                rgb(BG)
            } else {
                rgba(0x10171fee)
            })
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::open))
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(|s, _: &Classify, _, cx| s.run(cx)))
            .on_action(cx.listener(|s, _: &ShowClassify, _, cx| {
                s.page = Page::Classify;
                s.scroll.set_offset(point(px(0.), px(0.)));
                cx.notify();
            }))
            .on_action(cx.listener(|s, _: &ShowResults, _, cx| {
                s.page = Page::Results;
                s.scroll.set_offset(point(px(0.), px(0.)));
                cx.notify();
            }))
            .on_action(cx.listener(|s, _: &ShowDataset, _, cx| {
                s.page = Page::Dataset;
                s.scroll.set_offset(point(px(0.), px(0.)));
                cx.notify();
            }))
            .on_action(cx.listener(|s, _: &ShowAbout, _, cx| {
                s.page = Page::About;
                s.scroll.set_offset(point(px(0.), px(0.)));
                cx.notify();
            }))
            .child(self.sidebar(cx))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("main-scroll")
                            .track_scroll(&self.scroll)
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .p_8()
                            .child(content),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .px_8()
                            .py_3()
                            .border_t_1()
                            .border_color(rgb(BORDER))
                            .bg(rgba(0x111c26e0))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(div().size_2().rounded_full().bg(rgb(if self.ready {
                                        ACCENT
                                    } else {
                                        0xe9b87b
                                    })))
                                    .child(muted(self.status.clone())),
                            )
                            .child(if !self.ready {
                                button("retry-worker", "Retry worker", false)
                                    .py_1()
                                    .on_click(cx.listener(|s, _, _, cx| {
                                        if let Err(e) = s.worker.retry() {
                                            s.error = Some(e);
                                        } else {
                                            s.status = "Restarting local inference…".into();
                                            s.error = None;
                                        }
                                        cx.notify();
                                    }))
                                    .into_any_element()
                            } else {
                                muted("7 classes  ·  4 models  ·  Local inference")
                                    .into_any_element()
                            }),
                    ),
            )
    }
}
fn panel() -> Div {
    div()
        .flex()
        .flex_col()
        .gap_4()
        .p_6()
        .rounded_xl()
        .bg(rgba(0x19232eef))
        .border_1()
        .border_color(rgb(BORDER))
}

// Keep intrinsic image aspect ratios out of the card's measured height. The
// image paints inside this fixed viewport; captions retain their own space.
fn thumbnail(path: PathBuf, height: f32) -> Div {
    div()
        .relative()
        .w_full()
        .h(px(height))
        .min_h(px(height))
        .flex_shrink_0()
        .overflow_hidden()
        .child(
            img(path)
                .absolute()
                .inset_0()
                .size_full()
                .object_fit(ObjectFit::Cover),
        )
}
fn muted(text: impl Into<SharedString>) -> Div {
    div().text_sm().text_color(rgb(MUTED)).child(text.into())
}
fn eyebrow(text: impl Into<SharedString>) -> Div {
    div()
        .text_xs()
        .text_color(rgb(ACCENT))
        .font_weight(FontWeight::SEMIBOLD)
        .child(text.into())
}
fn heading(text: impl Into<SharedString>) -> Div {
    div()
        .text_xl()
        .font_weight(FontWeight::SEMIBOLD)
        .child(text.into())
}
fn badge(text: impl Into<SharedString>) -> Div {
    div()
        .px_3()
        .py_1()
        .rounded_full()
        .bg(rgb(0x213b40))
        .text_xs()
        .text_color(rgb(ACCENT))
        .child(text.into())
}
fn button(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    primary: bool,
) -> Stateful<Div> {
    div()
        .id(id)
        .cursor_pointer()
        .px_4()
        .py_3()
        .rounded_lg()
        .border_1()
        .border_color(rgb(if primary { ACCENT } else { BORDER }))
        .bg(rgb(if primary { ACCENT } else { 0x24313e }))
        .text_color(rgb(if primary { 0x102728 } else { TEXT }))
        .font_weight(FontWeight::SEMIBOLD)
        .hover(move |style| style.bg(rgb(if primary { 0x8be7dc } else { 0x314453 })))
        .child(label.into())
}
fn model_short(id: &str) -> &'static str {
    match id {
        "frozen_selected" => "DINO + ConvNeXt",
        "frozen_control" => "Frozen ConvNeXt",
        "resnet_fixed" => "ResNet-50",
        _ => "Fine-tuned DINOv2",
    }
}
fn model_role(id: &str) -> &'static str {
    match id {
        "frozen_selected" => "PRIMARY STUDY",
        "frozen_control" => "FROZEN CONTROL",
        "resnet_fixed" => "CNN COMPARISON",
        _ => "DINO COMPARISON",
    }
}
impl Studio {
    pub fn new(config: Config, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let loaded = Study::load(&config.root);
        let resource_error = loaded.as_ref().err().cloned();
        let study = loaded.ok();
        let selected_model = study
            .as_ref()
            .and_then(|s| s.models.iter().position(|m| m.id == s.default_model))
            .unwrap_or(3);
        let image = config.image.clone().or_else(|| {
            study.as_ref().and_then(|s| {
                s.samples
                    .first()
                    .map(|s| config.root.join("app/assets").join(&s.path))
            })
        });
        let image_label = image
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Choose a local image".into());
        let page = match config.view.as_str() {
            "results" => Page::Results,
            "dataset" => Page::Dataset,
            "about" => Page::About,
            _ => Page::Classify,
        };
        let worker = Worker::start(config.clone());
        let focus = cx.focus_handle();
        focus.focus(window, cx);
        let executor = cx.background_executor().clone();
        let poll = cx.spawn(async move |this, cx| {
            loop {
                executor.timer(Duration::from_millis(80)).await;
                if this.update(cx, |studio, cx| studio.poll(cx)).is_err() {
                    break;
                }
            }
        });
        Self {
            config,
            study,
            resource_error,
            worker,
            ready: false,
            status: "Starting local inference…".into(),
            page,
            selected_model,
            results_model: 0,
            image,
            image_label,
            prediction: None,
            error: None,
            active_id: None,
            serial: 0,
            started: None,
            focus,
            scroll: ScrollHandle::new(),
            _poll: poll,
        }
    }
    fn poll(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        while let Ok(event) = self.worker.events.try_recv() {
            changed = true;
            match event {
                Event::Ready => {
                    self.ready = true;
                    self.status = "Local inference ready".into();
                    if self.config.classify {
                        self.config.classify = false;
                        self.run(cx);
                    }
                }
                Event::Failed(message) => {
                    self.ready = false;
                    self.status = "Local inference unavailable".into();
                    self.error = Some(message);
                }
                Event::Finished { id, result } => {
                    self.ready = result
                        .as_ref()
                        .map_or_else(|error| !error.requires_restart(), |_| true);
                    self.status = if self.ready {
                        "Local inference ready"
                    } else {
                        "Retry local inference"
                    }
                    .into();
                    if self.active_id.as_deref() == Some(&id) {
                        self.active_id = None;
                        match result {
                            Ok(prediction) => {
                                self.prediction = Some(prediction);
                                self.error = None;
                                self.ready = true;
                            }
                            Err(e) => {
                                self.error = Some(e.to_string());
                            }
                        }
                    }
                }
            }
        }
        if changed || self.worker.busy() {
            cx.notify();
        }
    }
    fn invalidate(&mut self) {
        self.active_id = None;
        self.prediction = None;
        self.error = None;
        self.started = None;
    }
    fn choose_image(&mut self, path: PathBuf, label: String, cx: &mut Context<Self>) {
        self.invalidate();
        let valid = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
                .contains(&e.to_ascii_lowercase().as_str())
        });
        if valid && path.is_file() {
            self.image = Some(path);
            self.image_label = label;
        } else {
            self.image = None;
            self.error =
                Some("Choose a PNG, JPEG, WebP, BMP or TIFF image stored on this computer.".into());
        }
        cx.notify();
    }
    fn open(&mut self, _: &OpenImage, _: &mut Window, cx: &mut Context<Self>) {
        let picker = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Choose a material image".into()),
        });
        cx.spawn(async move |this, cx| match picker.await {
            Ok(Ok(Some(paths))) => {
                if let Some(path) = paths.into_iter().next() {
                    let label = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    let _ = this.update(cx, |s, cx| s.choose_image(path, label, cx));
                }
            }
            Ok(Err(e)) => {
                let _ = this.update(cx, |s, cx| {
                    s.error = Some(format!("The image picker could not open: {e}"));
                    cx.notify();
                });
            }
            _ => {}
        })
        .detach();
    }
    fn run(&mut self, cx: &mut Context<Self>) {
        if self.worker.busy() {
            return;
        }
        let Some(path) = self.image.clone() else {
            self.error = Some("Choose a local image first.".into());
            cx.notify();
            return;
        };
        let Some(model) = self
            .study
            .as_ref()
            .and_then(|s| s.models.get(self.selected_model))
        else {
            return;
        };
        self.serial += 1;
        let id = format!("prediction-{}", self.serial);
        match self.worker.predict(id.clone(), model.id.clone(), path) {
            Ok(()) => {
                self.active_id = Some(id);
                self.prediction = None;
                self.error = None;
                self.started = Some(Instant::now());
                self.status = "Running local inference…".into();
            }
            Err(e) => self.error = Some(e),
        }
        cx.notify();
    }
    fn cancel(&mut self, _: &Cancel, _: &mut Window, cx: &mut Context<Self>) {
        self.invalidate();
        if self.worker.busy() {
            self.status = "Finishing previous request; result dismissed".into();
        }
        cx.notify();
    }
    fn header(&self, title_text: &str, subtitle: &str, tag: &str) -> Div {
        div()
            .flex()
            .justify_between()
            .items_center()
            .gap_4()
            .mb_6()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_3xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(title_text.to_owned()),
                    )
                    .child(muted(subtitle.to_owned())),
            )
            .child(badge(tag.to_owned()))
    }
    fn sidebar(&self, cx: &mut Context<Self>) -> Div {
        let mut nav = div().flex().flex_col().gap_2().mt_8();
        for (index, (page, label, icon)) in [
            (Page::Classify, "Classify", "◈"),
            (Page::Results, "Results", "▤"),
            (Page::Dataset, "Dataset", "▦"),
            (Page::About, "About", "○"),
        ]
        .into_iter()
        .enumerate()
        {
            let selected = self.page == page;
            nav = nav.child(
                div()
                    .id(("nav", index))
                    .flex()
                    .items_center()
                    .gap_3()
                    .px_4()
                    .py_3()
                    .rounded_lg()
                    .cursor_pointer()
                    .bg(if selected { rgb(0x243b43) } else { rgba(0) })
                    .text_color(rgb(if selected { ACCENT } else { MUTED }))
                    .hover(|s| s.bg(rgb(0x23333f)))
                    .child(div().text_xl().child(icon))
                    .child(label)
                    .on_click(cx.listener(move |s, _, _, cx| {
                        s.page = page;
                        s.scroll.set_offset(point(px(0.), px(0.)));
                        cx.notify();
                    })),
            );
        }
        div()
            .w(px(205.))
            .h_full()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .p_5()
            .border_r_1()
            .border_color(rgb(BORDER))
            .bg(rgba(0x101a24c9))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .mt_3()
                    .child(
                        div()
                            .size_9()
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_lg()
                            .bg(rgb(0x29434c))
                            .text_color(rgb(ACCENT))
                            .text_2xl()
                            .child("◈"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child("Material Studio"),
                            )
                            .child(muted("VISION LAB")),
                    ),
            )
            .child(nav)
            .child(div().flex_1())
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(eyebrow("ON THIS COMPUTER"))
                    .child(muted("Images stay local.\nNo upload required."))
                    .child(div().h(px(1.)).bg(rgb(BORDER)).my_2())
                    .child(muted("Ctrl + O  Open image"))
                    .child(muted("Ctrl + Enter  Classify"))
                    .child(muted("Esc  Dismiss result")),
            )
    }
    fn model_picker(&self, cx: &mut Context<Self>, results: bool) -> Div {
        let mut row = div().flex().gap_3().w_full();
        if let Some(study) = &self.study {
            for (i, model) in study.models.iter().enumerate() {
                let selected = if results {
                    self.results_model == i
                } else {
                    self.selected_model == i
                };
                let mut card = div()
                    .id((if results { "result-model" } else { "model" }, i))
                    .flex_1()
                    .min_w_0()
                    .p_4()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgb(if selected { ACCENT } else { BORDER }))
                    .bg(rgb(if selected { 0x213b42 } else { PANEL }))
                    .cursor_pointer()
                    .hover(|s| s.bg(rgb(0x293d49)))
                    .child(eyebrow(model_role(&model.id)))
                    .child(
                        div()
                            .mt_2()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(model_short(&model.id)),
                    );
                if results {
                    card = card
                        .child(
                            div()
                                .mt_3()
                                .text_3xl()
                                .child(format!("{:.1}%", model.accuracy * 100.)),
                        )
                        .child(muted(format!("Macro F1  {:.3}", model.macro_f1)));
                }
                row = row.child(card.on_click(cx.listener(move |s, _, _, cx| {
                    if results {
                        s.results_model = i;
                    } else {
                        s.selected_model = i;
                        s.invalidate();
                    }
                    cx.notify();
                })));
            }
        }
        row
    }
    fn classify_page(&self, cx: &mut Context<Self>) -> Div {
        let busy = self.worker.busy();
        let mut preview = panel().flex_1().min_w_0().gap_3().child(
            div()
                .flex()
                .justify_between()
                .items_center()
                .child(eyebrow("01 / IMAGE INPUT"))
                .child(muted("Local image")),
        );
        let mut image_box = div()
            .w_full()
            .h(px(290.))
            .rounded_lg()
            .overflow_hidden()
            .bg(rgb(0x0d141b))
            .flex()
            .items_center()
            .justify_center();
        if let Some(path) = &self.image {
            image_box = image_box.child(
                img(path.clone())
                    .w_full()
                    .h_full()
                    .object_fit(ObjectFit::Contain),
            );
        } else {
            image_box = image_box.child(muted("Open an image or choose a study sample below."));
        }
        preview = preview.child(image_box).child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap_3()
                .child(
                    div()
                        .min_w_0()
                        .flex_1()
                        .text_sm()
                        .truncate()
                        .child(self.image_label.clone()),
                )
                .child(
                    button("open-image", "Open image", false)
                        .on_click(cx.listener(|s, _, window, cx| s.open(&OpenImage, window, cx))),
                ),
        );
        let mut output = panel()
            .w(px(355.))
            .flex_shrink_0()
            .gap_3()
            .child(eyebrow("02 / LIVE LOCAL INFERENCE"));
        if let Some(prediction) = &self.prediction {
            let max = prediction
                .probabilities
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            output = output.child(
                div()
                    .flex()
                    .justify_between()
                    .items_end()
                    .child(
                        div()
                            .text_3xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(title(&prediction.predicted_class)),
                    )
                    .child(
                        div()
                            .text_xl()
                            .text_color(rgb(ACCENT))
                            .child(format!("{:.1}%", max * 100.)),
                    ),
            );
        } else {
            output = output
                .child(heading(if busy {
                    "Analyzing material…"
                } else {
                    "Ready to explore"
                }))
                .child(muted(if busy {
                    "First use loads the model. Navigation stays available."
                } else {
                    "Choose a model, then classify the image."
                }));
        }
        for (i, class) in CLASSES.iter().enumerate() {
            let probability = self.prediction.as_ref().map(|p| p.probabilities[i]);
            let top = self
                .prediction
                .as_ref()
                .is_some_and(|p| p.predicted_class == *class);
            output = output.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .text_sm()
                            .child(title(class))
                            .child(muted(
                                probability
                                    .map(|p| format!("{:.1}%", p * 100.))
                                    .unwrap_or_else(|| "—".into()),
                            )),
                    )
                    .child(
                        div()
                            .w_full()
                            .h(px(5.))
                            .rounded_full()
                            .overflow_hidden()
                            .bg(rgb(0x30404d))
                            .child(
                                div()
                                    .h_full()
                                    .w(relative(probability.unwrap_or(0.) as f32))
                                    .bg(rgb(if top { ACCENT } else { 0x568fb5 })),
                            ),
                    ),
            );
        }
        if let Some(p) = &self.prediction {
            output = output
                .child(muted(format!("{:.1} ms  ·  {}", p.inference_ms, p.device)))
                .child(muted(if p.model_cached {
                    "Model resident · decode + preprocessing included".into()
                } else {
                    format!(
                        "First load {:.1} s · excluded from inference",
                        p.load_ms / 1000.
                    )
                }));
        }
        let elapsed = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let label = if busy {
            format!("Analyzing…  {elapsed}s")
        } else {
            "Classify image  →".into()
        };
        let mut action = button("classify", label, !busy && self.image.is_some())
            .w_full()
            .flex()
            .justify_center();
        if !busy {
            action = action.on_click(cx.listener(|s, _, _, cx| s.run(cx)));
        }
        let mut content = div().flex().flex_col().gap_5().child(self.header("Look closer.", "Explore seven material classes with local computer vision.", "CLASSROOM DEMO"))
            .child(self.model_picker(cx, false)).child(div().flex().gap_5().items_start().child(preview).child(output))
            .child(div().flex().gap_5().child(div().flex_1().child(muted("Model scores describe this image, not certainty about its physical material.\nSynthetic-study models have not been validated for arbitrary real-world images."))).child(div().w(px(355.)).child(action)));
        if let Some(error) = &self.error {
            content = content.child(
                div()
                    .p_4()
                    .rounded_lg()
                    .bg(rgb(0x3b2c2d))
                    .text_color(rgb(0xffc4b4))
                    .child(error.clone()),
            );
        }
        let mut samples = div().flex().gap_3();
        if let Some(study) = &self.study {
            for (i, class) in CLASSES.iter().enumerate() {
                if let Some(sample) = study.samples.iter().find(|s| s.class_name == *class) {
                    let path = self.config.root.join("app/assets").join(&sample.path);
                    let label = sample.label.clone();
                    let selected = self.image.as_ref() == Some(&path);
                    samples = samples.child(
                        div()
                            .id(("sample", i))
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .min_h(px(120.))
                            .rounded_lg()
                            .overflow_hidden()
                            .border_1()
                            .border_color(rgb(if selected { ACCENT } else { BORDER }))
                            .cursor_pointer()
                            .bg(rgb(PANEL))
                            .child(thumbnail(path.clone(), 80.))
                            .child(
                                div()
                                    .px_2()
                                    .h(px(38.))
                                    .min_h(px(38.))
                                    .flex_shrink_0()
                                    .flex()
                                    .items_center()
                                    .text_sm()
                                    .line_height(px(20.))
                                    .child(title(class)),
                            )
                            .on_click(cx.listener(move |s, _, _, cx| {
                                s.choose_image(path.clone(), label.clone(), cx)
                            })),
                    );
                }
            }
        }
        content
            .child(
                div()
                    .flex()
                    .justify_between()
                    .child(eyebrow("TRY A STUDY SAMPLE"))
                    .child(muted("Development examples · labels are ground truth")),
            )
            .child(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::model_short;

    #[test]
    fn frozen_control_label_identifies_convnext() {
        assert_eq!(model_short("frozen_control"), "Frozen ConvNeXt");
    }
}
