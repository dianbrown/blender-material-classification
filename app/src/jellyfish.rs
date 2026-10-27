//! Native ASCII jellyfish, painted without a background grid or cell tiles.
//! Silhouette sampling and articulated motion adapt the supplied ascii-sharks
//! reference; the upright pulsing bells and drifting tentacles follow the video.

mod field;
mod motion;

use field::{CELL_X, CELL_Y, GLYPHS, SHADES, rasterize};
use gpui::{prelude::*, *};
use motion::{Ocean, Vec2};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

/// Teal like the reference, with an occasional violet animal.
const HUES: [u32; 2] = [0x8ad4c4, 0xab9de2];

pub struct Jellyfish {
    last_frame: Instant,
    ocean: Rc<RefCell<Ocean>>,
    atlas: Option<Rc<Vec<ShapedLine>>>,
    next_frame: Option<Task<()>>,
}

impl Jellyfish {
    pub fn new() -> Self {
        Self {
            last_frame: Instant::now(),
            ocean: Rc::new(RefCell::new(Ocean::new(Vec2::new(1120., 860.)))),
            atlas: None,
            next_frame: None,
        }
    }
}

impl Render for Jellyfish {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let now = Instant::now();
        let dt = if window.is_window_active() && !cx.reduce_motion() {
            // Cap background work at 30 Hz, independently of a 144/240 Hz
            // monitor. The pending task is dropped with this view.
            if self.next_frame.is_none() {
                let executor = cx.background_executor().clone();
                self.next_frame = Some(cx.spawn(async move |this, cx| {
                    executor.timer(Duration::from_millis(33)).await;
                    let _ = this.update(cx, |this, cx| {
                        this.next_frame = None;
                        cx.notify();
                    });
                }));
            }
            now.duration_since(self.last_frame).as_secs_f32().min(0.05)
        } else {
            0.
        };
        self.last_frame = now;
        let ocean = self.ocean.clone();
        let atlas = self
            .atlas
            .get_or_insert_with(|| {
                // Reuse shaped glyphs; no per-character elements or per-frame text shaping.
                Rc::new(
                    (0..HUES.len() * SHADES)
                        .flat_map(|index| {
                            GLYPHS.iter().map(move |&glyph| {
                                let mut color = rgb(HUES[index / SHADES]);
                                color.a = 0.05 + (index % SHADES) as f32 * 0.068;
                                let run = TextRun {
                                    len: 1,
                                    font: Font {
                                        weight: FontWeight::BOLD,
                                        ..font("Consolas")
                                    },
                                    color: color.into(),
                                    background_color: None,
                                    underline: None,
                                    strikethrough: None,
                                };
                                ((glyph as char).to_string().into(), run)
                            })
                        })
                        .map(|(text, run)| {
                            window.text_system().shape_line(text, px(11.), &[run], None)
                        })
                        .collect(),
                )
            })
            .clone();

        canvas(
            move |bounds, _, _| {
                let mut ocean = ocean.borrow_mut();
                ocean.resize(Vec2::new(
                    bounds.size.width.into(),
                    bounds.size.height.into(),
                ));
                ocean.advance(dt);
                rasterize(&ocean)
            },
            move |bounds, field, window, cx| {
                window.with_content_mask(Some(ContentMask { bounds }), |window| {
                    for (index, cell) in field.cells.iter().enumerate() {
                        if cell.light < 0.05 {
                            continue;
                        }
                        let shade = (cell.light * (SHADES - 1) as f32)
                            .clamp(0., (SHADES - 1) as f32)
                            as usize;
                        let glyph = (cell.hue as usize * SHADES + shade) * GLYPHS.len()
                            + cell.glyph as usize;
                        let position = bounds.origin
                            + point(
                                px((index % field.cols) as f32 * CELL_X),
                                px((index / field.cols) as f32 * CELL_Y),
                            );
                        let _ = atlas[glyph].paint(
                            position,
                            px(CELL_Y),
                            TextAlign::Left,
                            None,
                            window,
                            cx,
                        );
                    }
                });
            },
        )
        .size_full()
    }
}
