//! A transparent, paint-only ASCII backdrop for the classification page.
//!
//! Adapts the supplied ascii-sharks renderer's silhouette sampling, stable
//! character texture and quantized shading to jellyfish. Only occupied body
//! cells are painted: no background grid, cell rectangles, wake or particles.

use gpui::{prelude::*, *};
use std::{f32::consts::PI, rc::Rc, time::Instant};

const GLYPHS: &[u8] = b".,:;+=xXoO08";
const SHADES: usize = 7;
const CELL_X: f32 = 5.8;
const CELL_Y: f32 = 8.0;

pub struct Jellyfish {
    time: f32,
    last_frame: Instant,
    atlas: Option<Rc<Vec<ShapedLine>>>,
}

impl Jellyfish {
    pub fn new() -> Self {
        Self {
            time: 0.,
            last_frame: Instant::now(),
            atlas: None,
        }
    }
}

impl Render for Jellyfish {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let now = Instant::now();
        if window.is_window_active() && !cx.reduce_motion() {
            // Clamp after returning from another page or a suspended window so
            // the creatures resume in place instead of jumping across the page.
            self.time += now.duration_since(self.last_frame).as_secs_f32().min(0.05);
            window.request_animation_frame();
        }
        self.last_frame = now;
        let time = self.time;
        let atlas = self
            .atlas
            .get_or_insert_with(|| {
                // Shape the small glyph palette once; painting never builds a tree
                // of individual text elements or re-shapes strings every frame.
                Rc::new(
                    (0..SHADES)
                        .flat_map(|shade| {
                            GLYPHS.iter().map(move |&glyph| {
                                let mut color = rgb(0xa4ccdc);
                                color.a = 0.10 + shade as f32 * 0.075;
                                let text: SharedString = (glyph as char).to_string().into();
                                let run = TextRun {
                                    len: 1,
                                    font: font("Consolas"),
                                    color: color.into(),
                                    background_color: None,
                                    underline: None,
                                    strikethrough: None,
                                };
                                (text, run)
                            })
                        })
                        .map(|(text, run)| {
                            window.text_system().shape_line(text, px(9.), &[run], None)
                        })
                        .collect(),
                )
            })
            .clone();

        canvas(
            |_, _, _| (),
            move |bounds, _, window, cx| {
                window.with_content_mask(Some(ContentMask { bounds }), |window| {
                    paint_jellyfish(bounds, time, &atlas, window, cx);
                });
            },
        )
        .size_full()
    }
}

fn paint_jellyfish(
    bounds: Bounds<Pixels>,
    time: f32,
    atlas: &[ShapedLine],
    window: &mut Window,
    cx: &mut App,
) {
    let width = f32::from(bounds.size.width);
    let height = f32::from(bounds.size.height);
    // Positions leave the image in the middle as the visual focus. All three
    // have independent phases, depths and proportions, but no random respawns.
    for (index, (anchor_x, anchor_y, radius, depth)) in [
        (0.12, 0.20, 70., 0.85),
        (0.88, 0.39, 84., 1.0),
        (0.18, 0.73, 54., 0.65),
    ]
    .into_iter()
    .enumerate()
    {
        let phase = index as f32 * 2.3;
        let pulse = (time * 1.15 + phase).sin();
        let radius = radius * (width / 1120.).clamp(0.78, 1.15);
        let rx = radius * (1. + 0.075 * pulse);
        let ry = radius * (0.64 - 0.045 * pulse);
        let x = width * anchor_x + (time * 0.12 + phase).sin() * 23.;
        let y = height * anchor_y + (time * 0.17 + phase).cos() * 22. - pulse * 3.;
        let tilt = (time * 0.16 + phase).sin() * 0.18;
        let (sin, cos) = tilt.sin_cos();

        // A body-local lattice travels with each animal, as in the shark
        // reference. Empty cells remain fully transparent.
        let columns = (rx * 1.65 / CELL_X).ceil() as i32;
        let rows = (radius * 3.3 / CELL_Y).ceil() as i32;
        for row in -((ry / CELL_Y).ceil() as i32)..rows {
            for col in -columns..=columns {
                let local_x = col as f32 * CELL_X;
                let local_y = row as f32 * CELL_Y;
                let strength = sample(local_x, local_y, rx, ry, radius, time, phase);
                if strength <= 0. {
                    continue;
                }
                let key = (col * 73 + row * 137 + index as i32 * 29).unsigned_abs() as usize;
                let shimmer = 0.92 + 0.08 * (time * 0.7 + key as f32 * 0.31).sin();
                let level = (strength * depth * shimmer * SHADES as f32)
                    .clamp(0., (SHADES - 1) as f32) as usize;
                let glyph = if strength < 0.2 {
                    key % 4
                } else {
                    4 + key % (GLYPHS.len() - 4)
                };
                let position = bounds.origin
                    + point(
                        px(x + local_x * cos - local_y * sin),
                        px(y + local_x * sin + local_y * cos),
                    );
                let _ = atlas[level * GLYPHS.len() + glyph].paint(
                    position,
                    px(CELL_Y),
                    TextAlign::Left,
                    None,
                    window,
                    cx,
                );
            }
        }
    }
}

/// Sample a domed bell with a scalloped rim, translucent ribs and trailing
/// oral arms / tentacles. Coordinates follow the animal, not the window.
fn sample(x: f32, y: f32, rx: f32, ry: f32, radius: f32, time: f32, phase: f32) -> f32 {
    let u = x / rx;
    let v = y / ry;
    let rim = 0.17 + 0.045 * (u * PI * 5. + phase).cos();
    if v <= rim && v >= -1. && u.abs() < 1. {
        let dome = u * u + v.min(0.).powi(2);
        if dome < 1. {
            let edge = ((dome - 0.72) / 0.28).clamp(0., 1.);
            let lip = (1. - (v - rim).abs() / 0.17).max(0.);
            let ribs = (u * PI * 4. + v * 1.4).cos().abs().powi(8);
            let light = (1. - ((u + 0.28).powi(2) + (v + 0.45).powi(2))).max(0.);
            return (0.22 + edge * 0.30 + lip * 0.35 + ribs * 0.16 + light * 0.15).min(1.);
        }
    }

    let length = y - ry * 0.17;
    if length < 0. {
        return 0.;
    }
    let mut strength: f32 = 0.;
    for strand in 0..9 {
        let spread = (strand as f32 - 4.) / 4.;
        let strand_phase = strand as f32 * 1.73 + phase;
        let reach = radius * (1.8 + 0.8 * strand_phase.sin().abs());
        let progress = length / reach;
        if progress > 1. {
            continue;
        }
        let sway = (progress * 7. - time * 1.05 + strand_phase).sin();
        let drift = (progress * 3. - time * 0.55 + phase).sin();
        let center = spread * rx * (0.77 - progress * 0.15)
            + sway * radius * (0.025 + progress * 0.14)
            + drift * radius * progress * 0.23;
        let oral_arm = strand == 3 || strand == 5;
        let thickness = if oral_arm {
            radius * 0.105 * (1. - progress * 0.72) * (0.8 + 0.2 * (progress * 24. + time).sin())
        } else {
            3.8 * (1. - progress * 0.5)
        };
        let coverage = (1. - (x - center).abs() / (thickness + 2.)).max(0.);
        let fade = (1. - progress).powf(0.65);
        strength = strength.max(coverage * fade * if oral_arm { 0.75 } else { 0.6 });
    }
    strength
}
