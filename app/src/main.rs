#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]
mod config;
mod data;
mod jellyfish;
mod ui;
mod worker;
use config::Config;
use data::Study;
use gpui::{prelude::*, *};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use ui::{Cancel, Classify, OpenImage, ShowAbout, ShowClassify, ShowDataset, ShowResults, Studio};
use worker::{Event, Worker};

fn smoke(config: Config, output: PathBuf) {
    let mut worker = Worker::start(config.clone());
    let mut result = serde_json::json!({"ok": false});
    let start = Instant::now();
    let image = config.image.clone().or_else(|| {
        Study::load(&config.root).ok().and_then(|s| {
            s.samples
                .first()
                .map(|s| config.root.join("app/assets").join(&s.path))
        })
    });
    loop {
        match worker.events.recv_timeout(Duration::from_secs(200)) {
            Ok(Event::Ready) => {
                result["startup_ms"] = serde_json::json!(start.elapsed().as_secs_f64() * 1000.);
                if let Some(path) = image.clone() {
                    let _ = worker.predict("smoke".into(), "dinov2_finetuned".into(), path);
                } else {
                    result["error"] = serde_json::json!("No image for smoke test");
                    break;
                }
            }
            Ok(Event::Finished {
                result: prediction, ..
            }) => {
                match prediction {
                    Ok(p) => {
                        result["ok"] = serde_json::json!(true);
                        result["predicted_class"] = serde_json::json!(p.predicted_class);
                        result["probabilities"] = serde_json::json!(p.probabilities);
                        result["inference_ms"] = serde_json::json!(p.inference_ms);
                        result["load_ms"] = serde_json::json!(p.load_ms);
                        result["device"] = serde_json::json!(p.device);
                    }
                    Err(e) => {
                        if let worker::WorkerError::Rejected { code, .. } = &e {
                            result["error_code"] = serde_json::json!(code);
                        }
                        result["error"] = serde_json::json!(e.to_string());
                    }
                }
                break;
            }
            Ok(Event::Failed(e)) => {
                result["error"] = serde_json::json!(e);
                break;
            }
            Err(e) => {
                result["error"] = serde_json::json!(e.to_string());
                break;
            }
        }
    }
    worker.shutdown();
    let _ = std::fs::write(
        output,
        serde_json::to_string_pretty(&result).unwrap_or_default(),
    );
}

fn main() {
    let config = match Config::from_args() {
        Ok(config) => config,
        Err(message) => {
            gpui_platform::application().run(move |cx| {
                let _ = cx.open_window(WindowOptions::default(), |_, cx| {
                    cx.new(|_| StartupError(message))
                });
                cx.on_window_closed(|cx, _| {
                    if cx.windows().is_empty() {
                        cx.quit();
                    }
                })
                .detach();
                cx.activate(true);
            });
            return;
        }
    };
    if let Some(output) = config.smoke_output.clone() {
        smoke(config, output);
        return;
    }
    gpui_platform::application().run(move |cx| {
        cx.bind_keys([
            KeyBinding::new("ctrl-o", OpenImage, None),
            KeyBinding::new("escape", Cancel, None),
            KeyBinding::new("ctrl-enter", Classify, None),
            KeyBinding::new("ctrl-1", ShowClassify, None),
            KeyBinding::new("ctrl-2", ShowResults, None),
            KeyBinding::new("ctrl-3", ShowDataset, None),
            KeyBinding::new("ctrl-4", ShowAbout, None),
        ]);
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        let bounds = Bounds::centered(None, size(px(1280.), px(860.)), cx);
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            titlebar: Some(TitlebarOptions {
                title: Some("Material Studio".into()),
                ..Default::default()
            }),
            window_min_size: Some(size(px(1000.), px(720.))),
            window_background: if config.opaque {
                WindowBackgroundAppearance::Opaque
            } else {
                WindowBackgroundAppearance::Blurred
            },
            ..Default::default()
        };
        cx.open_window(options, |window, cx| {
            cx.new(|cx| Studio::new(config, window, cx))
        })
        .expect("Could not create the native Material Studio window");
        cx.activate(true);
    });
}
struct StartupError(String);
impl Render for StartupError {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .bg(rgb(0x151515))
            .text_color(rgb(0xe7e7e7))
            .p_8()
            .font_family("Segoe UI")
            .child(div().text_2xl().child("Material Studio"))
            .child(div().mt_4().child(self.0.clone()))
    }
}
