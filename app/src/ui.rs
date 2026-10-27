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

const BG: u32 = 0x151515;
const BORDER: u32 = 0x323232;
const TEXT: u32 = 0xe7e7e7;
const MUTED: u32 = 0x939393;

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
#[derive(Clone, Copy, PartialEq, Debug)]
enum Menu {
    Models,
    Samples,
}

#[derive(Default)]
struct ViewOptions {
    menu: Option<Menu>,
    details: bool,
    study_details: bool,
    credits: bool,
    sample_filter: Option<usize>,
}
impl ViewOptions {
    fn toggle_menu(&mut self, menu: Menu) {
        self.menu = if self.menu == Some(menu) {
            None
        } else {
            Some(menu)
        };
    }
    fn dismiss(&mut self, page: Page, has_prediction: bool) -> bool {
        if page == Page::Classify && self.menu.take().is_some() {
            return true;
        }
        if page == Page::Classify && has_prediction && self.details {
            self.details = false;
            return true;
        }
        if page == Page::Results && self.study_details {
            self.study_details = false;
            return true;
        }
        false
    }
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
    options: ViewOptions,
    _poll: Task<()>,
}

fn muted(text: impl Into<SharedString>) -> Div {
    div().text_color(rgb(MUTED)).child(text.into())
}
fn section_title(text: impl Into<SharedString>) -> Div {
    div()
        .font_weight(FontWeight::MEDIUM)
        .text_size(px(14.))
        .child(text.into())
}
fn control(id: impl Into<ElementId>, text: impl Into<SharedString>) -> Stateful<Div> {
    div()
        .id(id)
        .h(px(34.))
        .flex_shrink_0()
        .px_3()
        .flex()
        .items_center()
        .justify_center()
        .rounded_md()
        .cursor_pointer()
        .text_size(px(13.))
        .text_color(rgb(0xc7c7c7))
        .hover(|s| s.bg(rgba(0xffffff0a)).text_color(rgb(TEXT)))
        .child(text.into())
}
fn thumbnail(path: PathBuf, height: f32) -> Div {
    div()
        .relative()
        .w_full()
        .h(px(height))
        .min_h(px(height))
        .flex_shrink_0()
        .overflow_hidden()
        .rounded_md()
        .child(
            img(path)
                .absolute()
                .inset_0()
                .size_full()
                .object_fit(ObjectFit::Cover),
        )
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
        "frozen_selected" => "Primary study",
        "frozen_control" => "Frozen control",
        "resnet_fixed" => "CNN comparison",
        _ => "DINO comparison",
    }
}

impl Studio {
    fn navigate(&mut self, page: Page, cx: &mut Context<Self>) {
        self.page = page;
        self.options.menu = None;
        self.scroll.set_offset(point(px(0.), px(0.)));
        cx.notify();
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> Div {
        let mut sidebar = div()
            .w(px(160.))
            .h_full()
            .flex_shrink_0()
            .flex()
            .flex_col()
            .px_3()
            .py_5()
            .bg(rgba(0x1c1c1cb8))
            .border_r_1()
            .border_color(rgba(0xffffff0b))
            .child(
                div()
                    .px_2()
                    .mb_6()
                    .text_size(px(13.))
                    .font_weight(FontWeight::MEDIUM)
                    .child("Material Studio"),
            );
        for (i, (page, label)) in [
            (Page::Classify, "Classify"),
            (Page::Results, "Study"),
            (Page::Dataset, "Samples"),
        ]
        .into_iter()
        .enumerate()
        {
            sidebar = sidebar.child(
                control(("nav", i), label)
                    .w_full()
                    .justify_start()
                    .mb_1()
                    .bg(if self.page == page {
                        rgba(0xffffff0c)
                    } else {
                        rgba(0)
                    })
                    .text_color(rgb(if self.page == page { TEXT } else { MUTED }))
                    .on_click(cx.listener(move |s, _, _, cx| s.navigate(page, cx))),
            );
        }
        sidebar.child(div().flex_1()).child(
            control("about", "About")
                .justify_start()
                .text_color(rgb(MUTED))
                .on_click(cx.listener(|s, _, _, cx| s.navigate(Page::About, cx))),
        )
    }

    fn classify_page(&self, cx: &mut Context<Self>) -> Div {
        let mut image = div()
            .relative()
            .w_full()
            .h_full()
            .max_w(px(520.))
            .max_h(px(520.))
            .rounded_md()
            .overflow_hidden();
        if let Some(path) = &self.image {
            image = image.child(
                img(path.clone())
                    .absolute()
                    .inset_0()
                    .size_full()
                    .object_fit(ObjectFit::Contain),
            );
        } else {
            image = image
                .flex()
                .items_center()
                .justify_center()
                .child(muted("Open an image to begin."));
        }
        let mut bottom = div()
            .w_full()
            .max_w(px(820.))
            .flex()
            .flex_col()
            .gap_3()
            .flex_shrink_0();
        if let Some(prediction) = &self.prediction {
            let confidence = prediction
                .probabilities
                .iter()
                .copied()
                .fold(0.0_f64, f64::max);
            bottom = bottom.child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_1()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_3()
                            .child(section_title(title(&prediction.predicted_class)))
                            .child(muted(format!("{:.1}%", confidence * 100.)))
                            .child(muted("· Live result").text_size(px(11.))),
                    )
                    .child(
                        control(
                            "prediction-details",
                            if self.options.details {
                                "Hide details"
                            } else {
                                "Details"
                            },
                        )
                        .on_click(cx.listener(|s, _, _, cx| {
                            s.options.details = !s.options.details;
                            s.options.menu = None;
                            cx.notify();
                        })),
                    ),
            );
            if self.options.details {
                bottom = bottom.child(self.prediction_details(prediction));
            }
        }
        if let Some(error) = &self.error {
            bottom = bottom.child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .px_3()
                    .py_2()
                    .border_1()
                    .border_color(rgb(BORDER))
                    .rounded_md()
                    .child(div().flex_1().min_w_0().child(error.clone()))
                    .when(!self.ready, |row| {
                        row.child(
                            control("retry", "Retry").on_click(cx.listener(|s, _, _, cx| {
                                if let Err(e) = s.worker.retry() {
                                    s.error = Some(e);
                                } else {
                                    s.error = None;
                                    s.status = "Starting local inference…".into();
                                }
                                cx.notify();
                            })),
                        )
                    }),
            );
        }
        bottom = bottom.child(self.toolbar(cx));
        div()
            .size_full()
            .flex()
            .flex_col()
            .px_8()
            .pb_7()
            .child(
                div()
                    .h(px(48.))
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .child(muted(self.image_label.clone()).text_size(px(12.))),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .py_8()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|s, _, _, cx| {
                            s.options.menu = None;
                            cx.notify();
                        }),
                    )
                    .child(image),
            )
            .child(div().w_full().flex().justify_center().child(bottom))
    }

    fn toolbar(&self, cx: &mut Context<Self>) -> Div {
        let busy = self.worker.busy();
        let name = self
            .study
            .as_ref()
            .and_then(|s| s.models.get(self.selected_model))
            .map(|m| model_short(&m.id))
            .unwrap_or("Choose model");
        let mut toolbar = div().relative().w_full().flex().flex_col().gap_2();
        let mut row = div()
            .flex()
            .items_center()
            .gap_1()
            .p_2()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER))
            .bg(rgba(0x202020ed))
            .child(
                control("open-image", "Open image")
                    .on_click(cx.listener(|s, _, window, cx| s.open(&OpenImage, window, cx))),
            )
            .child(
                control("sample-menu", "Samples  ⌄")
                    .bg(if self.options.menu == Some(Menu::Samples) {
                        rgba(0xffffff0c)
                    } else {
                        rgba(0)
                    })
                    .on_click(cx.listener(|s, _, _, cx| {
                        s.options.toggle_menu(Menu::Samples);
                        cx.notify();
                    })),
            )
            .child(div().flex_1().min_w_0());
        if busy {
            let elapsed = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
            row = row.child(
                muted(if self.active_id.is_some() {
                    format!("Working… {elapsed}s")
                } else {
                    "Finishing previous image…".into()
                })
                .text_size(px(11.))
                .flex_shrink_0()
                .px_2(),
            );
        } else if !self.ready && self.error.is_none() {
            row = row.child(muted("Starting…").text_size(px(11.)).flex_shrink_0().px_2());
        }
        row = row.child(
            control("model-menu", format!("{name}  ⌄"))
                .bg(if self.options.menu == Some(Menu::Models) {
                    rgba(0xffffff0c)
                } else {
                    rgba(0)
                })
                .on_click(cx.listener(|s, _, _, cx| {
                    s.options.toggle_menu(Menu::Models);
                    cx.notify();
                })),
        );
        let enabled = !busy && self.image.is_some();
        let mut classify = control("classify", "Classify")
            .px_5()
            .ml_2()
            .bg(rgb(if enabled { 0xd6d6d6 } else { 0x333333 }))
            .text_color(rgb(if enabled { 0x191919 } else { MUTED }))
            .font_weight(FontWeight::MEDIUM)
            .hover(move |s| {
                s.bg(rgb(if enabled { 0xeeeeee } else { 0x333333 }))
                    .text_color(rgb(if enabled { 0x191919 } else { MUTED }))
            });
        if enabled {
            classify = classify.on_click(cx.listener(|s, _, _, cx| s.run(cx)));
        }
        toolbar = toolbar.child(row.child(classify));
        if let Some(menu) = self.options.menu {
            let mut popup = div()
                .id("toolbar-popup")
                .absolute()
                .bottom(px(62.))
                .w(px(272.))
                .p_1()
                .rounded_lg()
                .border_1()
                .border_color(rgb(0x3a3a3a))
                .bg(rgb(0x232323))
                .occlude()
                .flex()
                .flex_col();
            match menu {
                Menu::Models => {
                    popup = popup.right(px(104.));
                    if let Some(study) = &self.study {
                        for (i, model) in study.models.iter().enumerate() {
                            popup = popup.child(
                                div()
                                    .id(("model-choice", i))
                                    .flex()
                                    .items_center()
                                    .gap_3()
                                    .px_3()
                                    .py_2()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .hover(|s| s.bg(rgb(0x303030)))
                                    .child(
                                        div().w(px(12.)).text_color(rgb(MUTED)).child(
                                            if self.selected_model == i { "✓" } else { "" },
                                        ),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(model_short(&model.id))
                                            .child(muted(model_role(&model.id)).text_size(px(11.))),
                                    )
                                    .on_click(cx.listener(move |s, _, _, cx| {
                                        s.selected_model = i;
                                        s.invalidate();
                                        cx.notify();
                                    })),
                            );
                        }
                    }
                }
                Menu::Samples => {
                    popup = popup.left(px(76.));
                    if let Some(study) = &self.study {
                        for (i, class) in CLASSES.iter().enumerate() {
                            if let Some(sample) =
                                study.samples.iter().find(|s| s.class_name == *class)
                            {
                                let path = self.config.root.join("app/assets").join(&sample.path);
                                let label = sample.label.clone();
                                popup = popup.child(
                                    div()
                                        .id(("quick-sample", i))
                                        .flex()
                                        .items_center()
                                        .gap_3()
                                        .p_2()
                                        .rounded_md()
                                        .cursor_pointer()
                                        .hover(|s| s.bg(rgb(0x303030)))
                                        .child(div().w(px(36.)).child(thumbnail(path.clone(), 36.)))
                                        .child(title(class))
                                        .on_click(cx.listener(move |s, _, _, cx| {
                                            s.choose_image(path.clone(), label.clone(), cx)
                                        })),
                                );
                            }
                        }
                    }
                    popup = popup.child(
                        control("browse-samples", "Browse all samples →")
                            .justify_start()
                            .mt_1()
                            .on_click(cx.listener(|s, _, _, cx| s.navigate(Page::Dataset, cx))),
                    );
                }
            }
            toolbar = toolbar.child(popup);
        }
        toolbar
    }

    fn prediction_details(&self, prediction: &Prediction) -> Div {
        let mut details = div()
            .flex()
            .flex_col()
            .gap_2()
            .px_1()
            .py_3()
            .border_t_1()
            .border_color(rgb(BORDER));
        for (i, class) in CLASSES.iter().enumerate() {
            let probability = prediction.probabilities[i];
            details = details.child(
                div()
                    .flex()
                    .items_center()
                    .gap_4()
                    .h(px(20.))
                    .flex_shrink_0()
                    .child(div().w(px(78.)).child(title(class)))
                    .child(
                        div()
                            .flex_1()
                            .h(px(3.))
                            .bg(rgb(0x303030))
                            .rounded_full()
                            .child(div().h_full().w(relative(probability as f32)).bg(rgb(
                                if prediction.predicted_class == *class {
                                    0xc4c4c4
                                } else {
                                    0x666666
                                },
                            ))),
                    )
                    .child(
                        muted(format!("{:.1}%", probability * 100.))
                            .w(px(48.))
                            .text_right(),
                    ),
            );
        }
        details.child(muted(format!("{:.1} ms · {} · {}", prediction.inference_ms, prediction.device,
            if prediction.model_cached { "model cached".into() } else { format!("first load {:.1}s", prediction.load_ms / 1000.) })).text_size(px(11.)).mt_2())
            .child(muted("Live scores, not calibrated certainty. Timing includes image decoding; model loading is separate.").text_size(px(11.)))
    }

    fn study_page(&self, cx: &mut Context<Self>) -> Div {
        let mut content = div()
            .w_full()
            .max_w(px(880.))
            .flex()
            .flex_col()
            .gap_6()
            .child(section_title("Study"));
        if let Some(study) = &self.study {
            content = content.child(muted(format!(
                "Stored results · {} held-out images · {} independent source groups",
                study.dataset.holdout.images, study.dataset.holdout.source_groups
            )));
            let mut table = div().flex().flex_col().w_full();
            table = table.child(
                div()
                    .flex()
                    .items_center()
                    .h(px(32.))
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .text_size(px(11.))
                    .text_color(rgb(MUTED))
                    .child(div().flex_1().px_3().child("Model"))
                    .child(table_cell("Accuracy".into()))
                    .child(table_cell("Macro F1".into()))
                    .child(table_cell("Median ms".into())),
            );
            for (i, model) in study.models.iter().enumerate() {
                table = table.child(
                    div()
                        .id(("study-model", i))
                        .flex()
                        .items_center()
                        .min_h(px(58.))
                        .py_2()
                        .border_b_1()
                        .border_color(rgba(0xffffff0b))
                        .cursor_pointer()
                        .bg(if self.results_model == i {
                            rgba(0xffffff08)
                        } else {
                            rgba(0)
                        })
                        .hover(|s| s.bg(rgba(0xffffff0d)))
                        .child(
                            div()
                                .flex_1()
                                .px_3()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(model_short(&model.id))
                                .child(muted(model_role(&model.id)).text_size(px(11.))),
                        )
                        .child(table_cell(format!("{:.1}%", model.accuracy * 100.)))
                        .child(table_cell(format!("{:.3}", model.macro_f1)))
                        .child(table_cell(format!("{:.1}", model.latency.median_ms)))
                        .on_click(cx.listener(move |s, _, _, cx| {
                            s.results_model = i;
                            cx.notify();
                        })),
                );
            }
            content = content.child(table);
            if let Some(model) = study.models.get(self.results_model) {
                content = content.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .flex()
                                .justify_between()
                                .child(section_title("Confusion matrix"))
                                .child(muted(model_short(&model.id))),
                        )
                        .child(muted("Rows: actual · Columns: predicted").text_size(px(11.)))
                        .child(self.confusion(model)),
                );
                content = content.child(
                    control(
                        "study-details",
                        if self.options.study_details {
                            "Hide study details  ⌃"
                        } else {
                            "Uncertainty & study details  ⌄"
                        },
                    )
                    .justify_start()
                    .px_0()
                    .on_click(cx.listener(|s, _, _, cx| {
                        s.options.study_details = !s.options.study_details;
                        cx.notify();
                    })),
                );
                if self.options.study_details {
                    content = content.child(self.study_details(study, model));
                }
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
            .child(div().w(px(76.)).flex_shrink_0());
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
                    .text_size(px(11.))
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
                .child(div().w(px(76.)).flex_shrink_0().child(title(CLASSES[i])));
            for value in row {
                let shade = 28 + (55. * *value as f64 / sum) as u32;
                cells = cells.child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .h(px(36.))
                        .flex()
                        .justify_center()
                        .items_center()
                        .rounded_sm()
                        .bg(rgb((shade << 16) | (shade << 8) | shade))
                        .child(value.to_string()),
                );
            }
            matrix = matrix.child(cells);
        }
        matrix
    }

    fn study_details(&self, study: &Study, model: &Model) -> Div {
        div().flex().flex_col().gap_4().pt_4().border_t_1().border_color(rgb(BORDER))
            .child(section_title(model.display_name.clone()))
            .child(detail_row("Accuracy interval", format!("{:.0}% CI · {:.1}–{:.1}%", model.accuracy_ci.level*100., model.accuracy_ci.low*100., model.accuracy_ci.high*100.)))
            .child(detail_row("Macro F1 interval", format!("{:.0}% CI · {:.3}–{:.3}", model.macro_f1_ci.level*100., model.macro_f1_ci.low, model.macro_f1_ci.high)))
            .child(detail_row("Warm latency", format!("Median {:.1} ms · p95 {:.1} ms · batch {} · {} repetitions", model.latency.median_ms, model.latency.p95_ms, model.latency.batch_size, model.latency.repetitions)))
            .child(detail_row("Measured device", model.latency.device.clone()))
            .child(detail_row("Deployment size", format!("{:.1} MiB · saved artifact {:.1} MiB", model.size.total_deployment_bytes as f64 / 1048576., model.size.artifact_bytes as f64 / 1048576.)))
            .child(muted("Intervals resample source groups, keeping related renders together. The frozen DINOv2 + ConvNeXt fusion is the preselected primary study; the other models are comparisons."))
            .child(muted("Stored timing uses warm models and predecoded images. It includes preprocessing, device transfer, inference and probabilities; excludes disk access, decoding and loading. Live demo timing includes decoding."))
            .child(muted(study.dataset.scope.clone()))
            .child(muted("Robustness evidence is limited to synthetic sources and tested corruptions. It does not establish recognition of arbitrary photographs or physical composition."))
    }

    fn samples_page(&self, cx: &mut Context<Self>) -> Div {
        let mut content = div()
            .w_full()
            .max_w(px(1020.))
            .flex()
            .flex_col()
            .gap_5()
            .child(section_title("Samples"));
        let mut filters = div().flex().flex_wrap().gap_1().child(
            control("filter-all", "All")
                .bg(if self.options.sample_filter.is_none() {
                    rgba(0xffffff0c)
                } else {
                    rgba(0)
                })
                .on_click(cx.listener(|s, _, _, cx| {
                    s.options.sample_filter = None;
                    cx.notify();
                })),
        );
        for (i, class) in CLASSES.iter().enumerate() {
            filters = filters.child(
                control(("filter", i), title(class))
                    .bg(if self.options.sample_filter == Some(i) {
                        rgba(0xffffff0c)
                    } else {
                        rgba(0)
                    })
                    .on_click(cx.listener(move |s, _, _, cx| {
                        s.options.sample_filter = Some(i);
                        cx.notify();
                    })),
            );
        }
        content = content.child(filters);
        if let Some(study) = &self.study {
            if let Some(class) = self
                .options
                .sample_filter
                .and_then(|i| study.dataset.classes.iter().find(|c| c.name == CLASSES[i]))
            {
                content = content.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(muted(class.definition.clone()))
                        .child(
                            muted(format!(
                                "{} development · {} held-out images",
                                class.development_images, class.holdout_images
                            ))
                            .text_size(px(11.)),
                        ),
                );
            } else {
                content = content.child(
                    muted("Development examples · choose an image to classify.").text_size(px(12.)),
                );
            }
            let mut grid = div().flex().flex_wrap().gap_5();
            for (i, sample) in study.samples.iter().enumerate().filter(|(_, s)| {
                self.options
                    .sample_filter
                    .is_none_or(|class| s.class_name == CLASSES[class])
            }) {
                let path = self.config.root.join("app/assets").join(&sample.path);
                let label = sample.label.clone();
                grid = grid.child(
                    div()
                        .id(("gallery", i))
                        .w(px(170.))
                        .flex()
                        .flex_col()
                        .gap_2()
                        .cursor_pointer()
                        .rounded_md()
                        .p_1()
                        .hover(|s| s.bg(rgba(0xffffff08)))
                        .child(thumbnail(path.clone(), 162.))
                        .child(
                            div()
                                .min_h(px(34.))
                                .line_height(px(17.))
                                .text_size(px(12.))
                                .text_color(rgb(0xb0b0b0))
                                .child(label.clone()),
                        )
                        .on_click(cx.listener(move |s, _, _, cx| {
                            s.choose_image(path.clone(), label.clone(), cx);
                            s.navigate(Page::Classify, cx);
                        })),
                );
            }
            content = content.child(grid);
        }
        content
    }

    fn about_page(&self, cx: &mut Context<Self>) -> Div {
        let mut content = div().w_full().max_w(px(680.)).flex().flex_col().gap_6().child(section_title("About Material Studio"))
            .child(muted("A local classroom demonstration of material recognition. Images stay on this computer."))
            .child(div().flex().flex_col().gap_3().child(section_title("The study")).child(muted("Seven appearance classes, three model families and four saved models. The preselected frozen DINOv2 + ConvNeXt fusion remains the primary study. Fine-tuned DINOv2 is the default for the demonstration, and a comparison in the study."))
                .child(muted("Source groups stay independent between development and holdout. Related rendered views are not independent evidence.")))
            .child(div().flex().flex_col().gap_3().child(section_title("Limits")).child(muted("Appearance is not physical composition. These models were studied on synthetic images with archival class conventions. Lighting, geometry and texture scale affect predictions. Live scores are not calibrated certainty, and arbitrary real-world images have not been validated.")))
            .child(div().flex().flex_col().gap_3().child(section_title("Keyboard")).child(detail_row("Open image", "Ctrl + O".into())).child(detail_row("Classify", "Ctrl + Enter".into())).child(detail_row("Close menu or dismiss result", "Escape".into())).child(detail_row("Switch views", "Ctrl + 1–4".into())));
        if let Some(study) = &self.study {
            content = content.child(div().flex().flex_col().gap_3().child(section_title("Dataset"))
                .child(muted(format!("Development: {} images, {} materials, {} source groups. Holdout: {} images, {} materials, {} source groups.", study.dataset.development.images, study.dataset.development.materials, study.dataset.development.source_groups, study.dataset.holdout.images, study.dataset.holdout.materials, study.dataset.holdout.source_groups)))
                .child(muted("Gallery examples come from distinct development source groups, selected without model scores. Their labels are archived ground truth; Classify always runs fresh local inference.")));
            content = content.child(
                control(
                    "credits",
                    if self.options.credits {
                        "Hide sample credits  ⌃"
                    } else {
                        "Sample credits  ⌄"
                    },
                )
                .justify_start()
                .px_0()
                .on_click(cx.listener(|s, _, _, cx| {
                    s.options.credits = !s.options.credits;
                    cx.notify();
                })),
            );
            if self.options.credits {
                for sample in &study.samples {
                    content = content.child(
                        muted(format!(
                            "{} · {} · {}",
                            sample.label, sample.attribution, sample.license
                        ))
                        .text_size(px(12.)),
                    );
                }
            }
        }
        content.child(muted("Native Rust / GPUI. Local Python inference. Dependency credits are included in THIRD_PARTY_NOTICES.txt.").text_size(px(11.)))
    }
}

fn table_cell(value: String) -> Div {
    div()
        .w(px(104.))
        .flex_shrink_0()
        .px_3()
        .text_right()
        .child(value)
}
fn detail_row(label: &str, value: String) -> Div {
    div()
        .flex()
        .gap_4()
        .child(muted(label.to_owned()).w(px(170.)).flex_shrink_0())
        .child(div().flex_1().child(value))
}

impl Render for Studio {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let body = if let Some(error) = &self.resource_error {
            div()
                .p_8()
                .flex()
                .flex_col()
                .gap_4()
                .child(section_title("Study resources unavailable"))
                .child(muted(error.clone()))
                .child(muted(
                    "Launch with --resources pointing to the supplied Material Studio folder.",
                ))
                .into_any_element()
        } else if self.page == Page::Classify {
            self.classify_page(cx).into_any_element()
        } else {
            let content = match self.page {
                Page::Results => self.study_page(cx),
                Page::Dataset => self.samples_page(cx),
                Page::About => self.about_page(cx),
                Page::Classify => unreachable!(),
            };
            div()
                .id("secondary-scroll")
                .size_full()
                .overflow_y_scroll()
                .track_scroll(&self.scroll)
                .p_8()
                .child(div().flex().justify_center().child(content))
                .into_any_element()
        };
        div()
            .size_full()
            .flex()
            .font_family("Segoe UI")
            .text_size(px(13.))
            .line_height(px(20.))
            .text_color(rgb(TEXT))
            .bg(if self.config.opaque {
                rgb(BG)
            } else {
                rgba(0x151515e8)
            })
            .track_focus(&self.focus)
            .on_action(cx.listener(Self::open))
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(|s, _: &Classify, _, cx| s.run(cx)))
            .on_action(cx.listener(|s, _: &ShowClassify, _, cx| s.navigate(Page::Classify, cx)))
            .on_action(cx.listener(|s, _: &ShowResults, _, cx| s.navigate(Page::Results, cx)))
            .on_action(cx.listener(|s, _: &ShowDataset, _, cx| s.navigate(Page::Dataset, cx)))
            .on_action(cx.listener(|s, _: &ShowAbout, _, cx| s.navigate(Page::About, cx)))
            .child(self.sidebar(cx))
            .child(div().h_full().flex_1().min_w_0().child(body))
    }
}

#[cfg(test)]
mod tests {
    use super::{Menu, Page, ViewOptions, model_short};
    #[test]
    fn frozen_control_label_identifies_convnext() {
        assert_eq!(model_short("frozen_control"), "Frozen ConvNeXt");
    }
    #[test]
    fn secondary_controls_are_hidden_by_default() {
        let view = ViewOptions::default();
        assert_eq!(view.menu, None);
        assert!(!view.details && !view.study_details && !view.credits);
    }
    #[test]
    fn menu_toggle_keeps_only_one_picker_open() {
        let mut view = ViewOptions::default();
        view.toggle_menu(Menu::Models);
        assert_eq!(view.menu, Some(Menu::Models));
        view.toggle_menu(Menu::Samples);
        assert_eq!(view.menu, Some(Menu::Samples));
        view.toggle_menu(Menu::Samples);
        assert_eq!(view.menu, None);
    }
    #[test]
    fn escape_dismisses_menu_before_prediction_details() {
        let mut view = ViewOptions {
            menu: Some(Menu::Models),
            details: true,
            ..Default::default()
        };
        assert!(view.dismiss(Page::Classify, true));
        assert_eq!(view.menu, None);
        assert!(view.details);
        assert!(view.dismiss(Page::Classify, true));
        assert!(!view.details);
        assert!(!view.dismiss(Page::Classify, true));
    }
    #[test]
    fn hidden_disclosures_do_not_consume_prediction_cancellation() {
        let mut view = ViewOptions {
            details: true,
            study_details: true,
            ..Default::default()
        };
        // A new inference removes the previous prediction: its details are hidden.
        assert!(!view.dismiss(Page::Classify, false));
        // A disclosure on another page cannot intercept Escape on Classify.
        view.details = false;
        assert!(!view.dismiss(Page::Classify, true));
        assert!(view.dismiss(Page::Results, false));
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
            options: ViewOptions::default(),
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
        self.options.menu = None;
        self.options.details = false;
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
        self.options.menu = None;
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
        self.options.menu = None;
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
                self.options.details = false;
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
        if self.options.dismiss(self.page, self.prediction.is_some()) {
            cx.notify();
            return;
        }
        self.invalidate();
        if self.worker.busy() {
            self.status = "Finishing previous request; result dismissed".into();
        }
        cx.notify();
    }
}
