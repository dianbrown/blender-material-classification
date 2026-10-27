//! Native ASCII jellyfish, painted without a background grid or cell tiles.
//! Silhouette sampling and articulated motion adapt the supplied ascii-sharks
//! reference; the swimming bell and trailing oral arms follow the video.

mod motion;

use gpui::{prelude::*, *};
use motion::{Ocean, Vec2};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

const GLYPHS: &[u8] = b".,:;t%SXo08#@";
const SHADES: usize = 12;
const CELL_X: f32 = 6.;
const CELL_Y: f32 = 8.;

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
                    (0..SHADES)
                        .flat_map(|shade| {
                            GLYPHS.iter().map(move |&glyph| {
                                let mut color = rgb(0xb2d9e6);
                                color.a = 0.045 + shade as f32 * 0.073;
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
                            window.text_system().shape_line(text, px(9.), &[run], None)
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
                        if cell.light < 0.045 {
                            continue;
                        }
                        let shade = (cell.light * (SHADES - 1) as f32)
                            .clamp(0., (SHADES - 1) as f32)
                            as usize;
                        let glyph = if cell.light < 0.16 {
                            cell.key % 4
                        } else {
                            4 + cell.key % (GLYPHS.len() - 4)
                        };
                        let position = bounds.origin
                            + point(
                                px((index % field.cols) as f32 * CELL_X),
                                px((index / field.cols) as f32 * CELL_Y),
                            );
                        let _ = atlas[shade * GLYPHS.len() + glyph].paint(
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

#[derive(Clone, Copy, Default)]
struct Cell {
    light: f32,
    key: usize,
}

struct Field {
    cols: usize,
    rows: usize,
    cells: Vec<Cell>,
}

impl Field {
    fn new(size: Vec2) -> Self {
        let cols = (size.x / CELL_X).ceil() as usize;
        let rows = (size.y / CELL_Y).ceil() as usize;
        Self {
            cols,
            rows,
            cells: vec![Cell::default(); cols * rows],
        }
    }

    fn put(&mut self, col: usize, row: usize, light: f32, key: usize) {
        let cell = &mut self.cells[row * self.cols + col];
        if light > cell.light {
            *cell = Cell { light, key };
        }
    }

    fn segment(&mut self, a: Vec2, b: Vec2, radius: f32, light: f32, key: usize) {
        let padding = radius + CELL_X;
        let left = ((a.x.min(b.x) - padding) / CELL_X).max(0.) as usize;
        let right = (((a.x.max(b.x) + padding) / CELL_X).ceil() as usize).min(self.cols);
        let top = ((a.y.min(b.y) - padding) / CELL_Y).max(0.) as usize;
        let bottom = (((a.y.max(b.y) + padding) / CELL_Y).ceil() as usize).min(self.rows);
        let delta = b - a;
        let length_squared = delta.dot(delta).max(0.001);
        for row in top..bottom {
            for col in left..right {
                let p = Vec2::new((col as f32 + 0.5) * CELL_X, (row as f32 + 0.5) * CELL_Y);
                let t = ((p - a).dot(delta) / length_squared).clamp(0., 1.);
                let distance = (p - (a + delta * t)).length();
                let coverage = ((radius + 3. - distance) / 4.).clamp(0., 1.);
                self.put(col, row, light * coverage, key + col * 13 + row * 7);
            }
        }
    }
}

fn rasterize(ocean: &Ocean) -> Field {
    let mut field = Field::new(ocean.size);
    for (id, body) in ocean.bodies.iter().enumerate() {
        let bell = body.bell();
        // Oral arms are broad, ruffled ribbons; the marginal tentacles are
        // long filaments. Both use the same simulated, world-space chains.
        for (strand_id, strand) in body.strands.iter().enumerate() {
            for node in 0..strand.nodes.len() - 1 {
                let progress = node as f32 / (strand.nodes.len() - 1) as f32;
                let ruffle = if strand.oral {
                    0.75 + 0.25 * (progress * 39. - ocean.time * 1.8 + strand_id as f32).sin()
                } else {
                    1.
                };
                let radius = if strand.oral {
                    body.radius * 0.075 * ruffle
                } else {
                    1.6
                };
                let radius = radius * (1. - progress * 0.82);
                let light =
                    body.depth * (1. - progress).powf(0.5) * if strand.oral { 0.88 } else { 0.63 };
                let a = strand.nodes[node.saturating_sub(1)];
                let b = strand.nodes[node];
                let c = strand.nodes[node + 1];
                let d = strand.nodes[(node + 2).min(strand.nodes.len() - 1)];
                let mut previous = b;
                for sub in 1..=3 {
                    let next = spline(a, b, c, d, sub as f32 / 3.);
                    field.segment(
                        previous,
                        next,
                        radius,
                        light,
                        id * 431 + strand_id * 79 + node * 11,
                    );
                    previous = next;
                }
            }
        }

        let reach = bell.width + bell.height;
        let left = ((body.position.x - reach) / CELL_X).max(0.) as usize;
        let right = (((body.position.x + reach) / CELL_X).ceil() as usize).min(field.cols);
        let top = ((body.position.y - reach) / CELL_Y).max(0.) as usize;
        let bottom = (((body.position.y + reach) / CELL_Y).ceil() as usize).min(field.rows);
        for row in top..bottom {
            for col in left..right {
                let p = Vec2::new((col as f32 + 0.5) * CELL_X, (row as f32 + 0.5) * CELL_Y);
                let relative = p - body.position;
                let x = relative.dot(body.side());
                let y = relative.dot(body.tail());
                let u = x / bell.width;
                let v = y / bell.height;
                let rim = 0.14 + (u * 16.).cos() * 0.035;
                let ellipse = u * u + v.min(0.).powi(2);
                if u.abs() >= 1. || v > rim || ellipse > 1. {
                    continue;
                }
                let edge = ((1. - ellipse) * 12.).clamp(0., 1.);
                let lip = (1. - (v - rim).abs() / 0.16).max(0.);
                let rib = (u * 13. + v * 1.6).cos().abs().powi(10);
                let highlight = (1. - (u + 0.32).powi(2) - (v + 0.48).powi(2)).max(0.);
                let light = (0.32 + highlight * 0.36 + rib * 0.15 + lip * 0.30) * edge * body.depth;
                // Texture sticks to the deforming bell, not to the screen.
                let key = ((u * 28.) as i32 * 73 + (v * 19.) as i32 * 137 + id as i32 * 29)
                    .unsigned_abs() as usize;
                field.put(col, row, light, key);
            }
        }
    }
    field
}

fn spline(a: Vec2, b: Vec2, c: Vec2, d: Vec2, t: f32) -> Vec2 {
    (b * 2.
        + (c - a) * t
        + (a * 2. - b * 5. + c * 4. - d) * (t * t)
        + (b * 3. - a - c * 3. + d) * (t * t * t))
        * 0.5
}
