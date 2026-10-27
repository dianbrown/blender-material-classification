//! Rasterizes the simulated jellyfish into sparse ASCII cells. Only lit cells
//! are painted, so there is no grid, tile or background behind the shapes.
use super::motion::{Ocean, Swimmer, Vec2};

pub const CELL_X: f32 = 8.;
pub const CELL_Y: f32 = 11.;
pub const SHADES: usize = 12;
/// Faint punctuation for filaments and frills, then glyphs ordered by ink
/// density for the bell and oral arms.
pub const GLYPHS: &[u8] = b".':,;to%xsSX08#";
const FAINT: usize = 5;

#[derive(Clone, Copy, Default)]
pub struct Cell {
    pub light: f32,
    pub glyph: u8,
    pub hue: u8,
}

pub struct Field {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<Cell>,
}

impl Field {
    fn new(size: Vec2) -> Self {
        let cols = (size.x / CELL_X).ceil().max(0.) as usize;
        let rows = (size.y / CELL_Y).ceil().max(0.) as usize;
        Self {
            cols,
            rows,
            cells: vec![Cell::default(); cols * rows],
        }
    }

    fn put(&mut self, col: usize, row: usize, light: f32, glyph: usize, hue: usize) {
        let cell = &mut self.cells[row * self.cols + col];
        if light > cell.light {
            *cell = Cell {
                light,
                glyph: glyph as u8,
                hue: hue as u8,
            };
        }
    }

    fn span(&self, low: Vec2, high: Vec2) -> (usize, usize, usize, usize) {
        let left = (low.x / CELL_X).max(0.) as usize;
        let right = ((high.x / CELL_X).ceil().max(0.) as usize).min(self.cols);
        let top = (low.y / CELL_Y).max(0.) as usize;
        let bottom = ((high.y / CELL_Y).ceil().max(0.) as usize).min(self.rows);
        (left, right, top, bottom)
    }

    /// Lights cells along a capsule, fading with distance so strands slide
    /// smoothly between rows instead of jumping.
    fn segment(&mut self, a: Vec2, b: Vec2, radius: f32, ink: Ink, hue: usize) {
        let pad = Vec2::new(radius + CELL_X, radius + CELL_Y);
        let low = Vec2::new(a.x.min(b.x), a.y.min(b.y)) - pad;
        let high = Vec2::new(a.x.max(b.x), a.y.max(b.y)) + pad;
        let (left, right, top, bottom) = self.span(low, high);
        let delta = b - a;
        let length_squared = delta.dot(delta).max(0.001);
        for row in top..bottom {
            for col in left..right {
                let p = center(col, row);
                let t = ((p - a).dot(delta) / length_squared).clamp(0., 1.);
                let distance = (p - (a + delta * t)).length();
                let coverage = ((radius + 2.5 - distance) / 3.5).clamp(0., 1.);
                let light = ink.light * coverage;
                self.put(col, row, light, ink.glyph(light), hue);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Ink {
    light: f32,
    faint: bool,
    key: usize,
}

impl Ink {
    fn glyph(self, light: f32) -> usize {
        if self.faint {
            return self.key % FAINT;
        }
        let dense = GLYPHS.len() - FAINT;
        let jitter = (self.key % 7) as f32 * 0.06 - 0.18;
        FAINT + (((light * 0.95 + jitter) * dense as f32).max(0.) as usize).min(dense - 1)
    }
}

fn center(col: usize, row: usize) -> Vec2 {
    Vec2::new((col as f32 + 0.5) * CELL_X, (row as f32 + 0.5) * CELL_Y)
}

fn spline(a: Vec2, b: Vec2, c: Vec2, d: Vec2, t: f32) -> Vec2 {
    (b * 2.
        + (c - a) * t
        + (a * 2. - b * 5. + c * 4. - d) * (t * t)
        + (b * 3. - a - c * 3. + d) * (t * t * t))
        * 0.5
}

pub fn rasterize(ocean: &Ocean) -> Field {
    let mut field = Field::new(ocean.size);
    for (id, body) in ocean.bodies.iter().enumerate() {
        strands(&mut field, body, id, ocean.time);
        bell(&mut field, body, id);
    }
    field
}

fn strands(field: &mut Field, body: &Swimmer, id: usize, time: f32) {
    for (strand_id, strand) in body.strands.iter().enumerate() {
        let last = strand.nodes.len() - 1;
        for node in 0..last {
            let t = node as f32 / last as f32;
            let (radius, light) = if strand.thick {
                // Broad, slightly ruffled ribbons that taper toward the tip.
                let ruffle = 0.85 + 0.15 * (t * 22. - time * 1.3 + strand_id as f32).sin();
                (
                    body.radius * 0.085 * (1. - 0.55 * t) * ruffle,
                    0.6 * (1. - t).powf(0.45) + 0.1,
                )
            } else {
                // Fine filaments, fading toward the tip.
                (1.2, 0.42 * (1. - t).powf(0.6) + 0.1)
            };
            let ink = Ink {
                light: light * body.depth,
                faint: !strand.thick,
                key: strand_id * 3 + node + id * 7,
            };
            let a = strand.nodes[node.saturating_sub(1)];
            let b = strand.nodes[node];
            let c = strand.nodes[node + 1];
            let d = strand.nodes[(node + 2).min(last)];
            let mut previous = b;
            for sub in 1..=3 {
                let next = spline(a, b, c, d, sub as f32 / 3.);
                field.segment(previous, next, radius, ink, body.hue);
                previous = next;
            }
        }
    }
}

fn bell(field: &mut Field, body: &Swimmer, id: usize) {
    let bell = body.bell();
    let reach = Vec2::new(1., 1.) * (bell.width + bell.height);
    let (left, right, top, bottom) = field.span(body.position - reach, body.position + reach);
    let (side, tail) = (body.side(), body.tail());
    for row in top..bottom {
        for col in left..right {
            let relative = center(col, row) - body.position;
            // Bell coordinates: u across the margin, v down from the dome base.
            let u = relative.dot(side) / bell.width;
            let v = relative.dot(tail) / bell.height;
            if u.abs() >= 1. {
                continue;
            }
            // A rounded dome over short, nearly straight sides, like a helmet.
            let margin = 0.3 * (1. - u.powi(6)) + 0.025 * (u * 15.).cos();
            let inside = if v <= 0. {
                1. - (u * u + v * v).sqrt()
            } else {
                (margin - v).min(1. - u.abs() - v * 0.15)
            };
            let key =
                ((u * 20.).floor() as i32 * 73 + (v * 9.).floor() as i32 * 137 + id as i32 * 29)
                    .unsigned_abs() as usize;
            if inside <= 0. {
                // A sparse dotted frill fringes the margin.
                let below = v - margin;
                if below > 0. && below < 0.16 && u.abs() < 0.94 && key % 3 != 0 {
                    let light = 0.24 * (1. - below / 0.16) * body.depth;
                    field.put(col, row, light, key % FAINT, body.hue);
                }
                continue;
            }
            let edge = (inside * 9.).min(1.);
            let rim = (1. - inside * 4.).max(0.);
            let highlight = (1. - ((u + 0.3).powi(2) + (v + 0.55).powi(2)) / 0.3).max(0.);
            let hollow = (1. - (u * u / 0.4 + (v - 0.2).powi(2) / 0.05)).max(0.);
            let canal = (u.atan2(-v + 0.15) * 4.5).cos().abs().powi(12);
            let light = (0.5 + highlight * 0.3 + rim * 0.22 + canal * 0.1 - hollow * 0.24
                + body.contraction * 0.06)
                * edge
                * body.depth;
            let ink = Ink {
                light,
                faint: false,
                key,
            };
            field.put(col, row, light, ink.glyph(light), body.hue);
        }
    }
}
