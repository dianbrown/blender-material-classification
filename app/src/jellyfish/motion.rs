//! Fixed-step jellyfish swimming, independent of the UI frame rate.
//!
//! Bells stay roughly upright and glide upward on each pulse, drifting
//! sideways with a gentle tilt, as in the reference video. Tentacles are damped
//! rope chains pulled toward a slowly undulating rest shape: they lag behind
//! the bell, sway and settle, with no angle limits that could snap them.
use std::{
    f32::consts::{FRAC_PI_2, TAU},
    ops::{Add, Mul, Sub},
};

const STEP: f32 = 1. / 60.;
const NODES: usize = 24;
const ITERATIONS: usize = 6;
/// Largest lean away from upright; jellyfish in the reference never roll over.
const MAX_TILT: f32 = 0.5;
/// Water drag on tentacle nodes, per second.
const TENTACLE_DRAG: f32 = 2.2;
/// Water drag on the bell, per second.
const BELL_DRAG: f32 = 1.15;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Vec2 {
    pub x: f32,
    pub y: f32,
}
impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn direction(angle: f32) -> Self {
        Self::new(angle.cos(), angle.sin())
    }
    fn scaled(self, size: Self) -> Self {
        Self::new(self.x * size.x, self.y * size.y)
    }
}
impl Add for Vec2 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.x + r.x, self.y + r.y)
    }
}
impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.x - r.x, self.y - r.y)
    }
}
impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, r: f32) -> Self {
        Self::new(self.x * r, self.y * r)
    }
}

pub(super) struct Strand {
    pub nodes: [Vec2; NODES],
    previous: [Vec2; NODES],
    pub oral: bool,
    /// Position across the bell margin, from -1 (left) to 1 (right).
    attachment: f32,
    pub segment: f32,
    /// Outward spread of the tip, in bell widths.
    splay: f32,
    /// Undulation: waves along the strand, angular speed and phase.
    waves: f32,
    speed: f32,
    seed: f32,
}

pub(super) struct Bell {
    pub width: f32,
    pub height: f32,
}

pub(super) struct Swimmer {
    /// Centre of the bell margin; the dome rises ahead of it.
    pub position: Vec2,
    velocity: Vec2,
    tilt: f32,
    spin: f32,
    pub radius: f32,
    pub depth: f32,
    pub hue: usize,
    phase: f32,
    period: f32,
    pub contraction: f32,
    seed: f32,
    pub strands: Vec<Strand>,
}

impl Swimmer {
    fn new(position: Vec2, radius: f32, tilt: f32, id: usize) -> Self {
        let seed = id as f32 * 1.71 + 0.4;
        let phase = id as f32 * 0.31;
        let mut body = Self {
            position,
            velocity: Vec2::direction(tilt - FRAC_PI_2) * 22.,
            tilt,
            spin: 0.,
            radius,
            depth: [0.86, 0.74, 1., 0.66][id % 4],
            hue: usize::from(id == 1),
            phase,
            period: 3.1 + id as f32 * 0.23,
            contraction: contraction(phase),
            seed,
            strands: Vec::new(),
        };
        // Eight fine marginal tentacles around the rim, and four ruffled oral
        // arms hanging from the centre of the bell.
        for strand in 0..12 {
            let oral = strand >= 8;
            let k = strand as f32;
            let (attachment, length, splay, waves, speed) = if oral {
                let attachment = ((strand - 8) as f32 / 3. - 0.5) * 0.66;
                let length = 1.9 + (k * 1.3).sin().abs() * 0.6;
                (attachment, length, attachment * 0.9, 1.1, 0.9)
            } else {
                let attachment = k / 7. * 1.84 - 0.92;
                let length = 2.7 + (k * 2.1).sin().abs() * 1.2;
                let splay = attachment * (0.15 + (k * 0.7).cos().abs() * 0.3);
                (
                    attachment,
                    length,
                    splay,
                    1. + (k * 0.9).sin().abs() * 0.6,
                    0.9 + k * 0.06,
                )
            };
            let length = radius * length;
            body.strands.push(Strand {
                nodes: [position; NODES],
                previous: [position; NODES],
                oral,
                attachment,
                segment: length / (NODES - 1) as f32,
                splay,
                waves,
                speed,
                seed: seed * 3. + k * 2.4,
            });
        }
        for strand in 0..body.strands.len() {
            let nodes = std::array::from_fn(|node| body.rest(&body.strands[strand], node, 0.));
            let strand = &mut body.strands[strand];
            strand.nodes = nodes;
            strand.previous = nodes;
        }
        body
    }

    fn heading(&self) -> f32 {
        self.tilt - FRAC_PI_2
    }
    pub fn forward(&self) -> Vec2 {
        Vec2::direction(self.heading())
    }
    pub fn side(&self) -> Vec2 {
        Vec2::direction(self.tilt)
    }
    pub fn tail(&self) -> Vec2 {
        self.forward() * -1.
    }
    pub fn bell(&self) -> Bell {
        // A contracting bell narrows and stretches taller.
        Bell {
            width: self.radius * (1. - 0.16 * self.contraction),
            height: self.radius * (0.82 + 0.18 * self.contraction),
        }
    }
    fn root(&self, strand: &Strand) -> Vec2 {
        let bell = self.bell();
        let (across, down) = if strand.oral {
            (0.7, 0.06)
        } else {
            (0.94, 0.14)
        };
        self.position
            + self.side() * (bell.width * strand.attachment * across)
            + self.tail() * (bell.height * down)
    }
    /// Rest shape of a strand in the body frame: hanging behind the bell,
    /// spreading toward the tip, with a slow wave travelling down its length.
    fn rest(&self, strand: &Strand, node: usize, time: f32) -> Vec2 {
        let t = node as f32 / (NODES - 1) as f32;
        let along = strand.segment * node as f32 * 0.94;
        let wave = (TAU * strand.waves * t - time * strand.speed + strand.seed).sin();
        let amplitude = self.radius * if strand.oral { 0.18 } else { 0.3 };
        let lateral = self.bell().width * strand.splay * t + wave * amplitude * t.powf(1.2);
        self.root(strand) + self.side() * lateral + self.tail() * along
    }

    fn extent(&self) -> (Vec2, Vec2) {
        let bell = self.bell();
        let crown = self.position + self.forward() * bell.height;
        let (mut low, mut high) = (crown, crown);
        let rim = [
            self.position + self.side() * bell.width,
            self.position - self.side() * bell.width,
        ];
        let nodes = self.strands.iter().flat_map(|s| s.nodes.iter());
        for p in rim.iter().chain(nodes) {
            low = Vec2::new(low.x.min(p.x), low.y.min(p.y));
            high = Vec2::new(high.x.max(p.x), high.y.max(p.y));
        }
        (low, high)
    }

    fn translate(&mut self, offset: Vec2) {
        self.position = self.position + offset;
        for strand in &mut self.strands {
            for p in strand.nodes.iter_mut().chain(strand.previous.iter_mut()) {
                *p = *p + offset;
            }
        }
    }

    fn step(&mut self, time: f32, size: Vec2, others: &[(Vec2, f32)], id: usize) {
        let next = contraction(time / self.period + self.phase);
        let squeeze = ((next - self.contraction) / STEP).max(0.);
        self.contraction = next;

        // Steering only changes the lean, through a critically damped spring,
        // so turns build and fade gradually.
        let lane = 0.5 + 0.36 * (time * 0.041 + self.seed * 2.7).sin();
        let mut goal = ((lane * size.x - self.position.x) / size.x * 2.).clamp(-0.4, 0.4)
            + 0.22 * (time * 0.13 + self.seed).sin()
            + 0.12 * (time * 0.31 + self.seed * 1.9).sin();
        let margin = self.radius * 1.5;
        goal += 0.7
            * (((margin - self.position.x) / margin).max(0.)
                - ((self.position.x - size.x + margin) / margin).max(0.));
        for (other, &(position, radius)) in others.iter().enumerate() {
            let away = self.position - position;
            let range = (self.radius + radius) * 1.7;
            if other != id && away.length() < range {
                goal += away.x.signum() * 0.6 * (1. - away.length() / range);
            }
        }
        let goal = goal.clamp(-MAX_TILT, MAX_TILT);
        let omega = 0.85;
        self.spin += ((goal - self.tilt) * omega * omega - self.spin * 2. * omega) * STEP;
        self.tilt += self.spin * STEP;

        // Each contraction pushes the bell forward; drag slows the glide while
        // the bell refills, and a slight negative buoyancy sinks it a little.
        let scale = self.radius / 85.;
        let thrust = self.forward() * ((12. + squeeze * 46.) * scale);
        let sink = Vec2::new(0., 3.5);
        self.velocity = (self.velocity + (thrust + sink) * STEP) * (-BELL_DRAG * STEP).exp();
        self.position = self.position + self.velocity * STEP;

        // Once the whole animal has left through the top, it re-enters from
        // below, out of view. Nodes move with it, so nothing jolts.
        let (low, high) = self.extent();
        if high.y < -6. {
            let x = (0.14 + 0.72 * hash(self.seed + time)) * size.x;
            self.translate(Vec2::new(x - self.position.x, size.y + 6. - low.y));
        }

        let damping = (-TENTACLE_DRAG * STEP).exp();
        let down = Vec2::new(0., 1.);
        for index in 0..self.strands.len() {
            let strand = &self.strands[index];
            let root = self.root(strand);
            let stiffness = if strand.oral { 9. } else { 6. };
            let targets: [Vec2; NODES] = std::array::from_fn(|node| self.rest(strand, node, time));
            let strand = &mut self.strands[index];
            strand.nodes[0] = root;
            strand.previous[0] = root;
            for node in 1..NODES {
                let t = node as f32 / (NODES - 1) as f32;
                let x = strand.nodes[node];
                let velocity = (x - strand.previous[node]) * damping;
                strand.previous[node] = x;
                // Stiff near the bell, loose at the tips; tips are slightly heavy.
                let pull = stiffness * (1. - t).powi(3) + 0.9;
                let force = (targets[node] - x) * pull + down * (5. * t);
                strand.nodes[node] = x + velocity + force * (STEP * STEP);
            }
            // Symmetric distance constraints spread corrections along the rope;
            // a final pass from the root makes it exactly inextensible.
            for _ in 0..ITERATIONS {
                for node in 1..NODES {
                    let delta = strand.nodes[node] - strand.nodes[node - 1];
                    let length = delta.length().max(0.0001);
                    let error = delta * ((length - strand.segment) / length);
                    if node == 1 {
                        strand.nodes[node] = strand.nodes[node] - error;
                    } else {
                        strand.nodes[node - 1] = strand.nodes[node - 1] + error * 0.5;
                        strand.nodes[node] = strand.nodes[node] - error * 0.5;
                    }
                }
            }
            for node in 1..NODES {
                let delta = strand.nodes[node] - strand.nodes[node - 1];
                let length = delta.length().max(0.0001);
                strand.nodes[node] = strand.nodes[node - 1] + delta * (strand.segment / length);
            }
        }
    }
}

pub(super) struct Ocean {
    pub size: Vec2,
    pub bodies: Vec<Swimmer>,
    pub time: f32,
    accumulator: f32,
}

impl Ocean {
    pub fn new(size: Vec2) -> Self {
        let bodies = [
            (0.18, 0.42, 82., -0.3),
            (0.6, 0.1, 62., 0.35),
            (0.84, 0.64, 88., 0.2),
            (0.42, 0.94, 100., -0.1),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (x, y, radius, tilt))| {
            Swimmer::new(Vec2::new(x, y).scaled(size), radius, tilt, id)
        })
        .collect();
        let mut ocean = Self {
            size,
            bodies,
            time: 0.,
            accumulator: 0.,
        };
        // Let the tentacles settle into a natural drape before the first frame.
        for _ in 0..90 {
            ocean.step();
        }
        ocean
    }
    pub fn resize(&mut self, size: Vec2) {
        if size.x < 1. || size.y < 1. || (size - self.size).length() < 0.5 {
            return;
        }
        let scale = Vec2::new(size.x / self.size.x, size.y / self.size.y);
        for body in &mut self.bodies {
            // Move each animal rigidly so its tentacles keep their shape.
            let target = body.position.scaled(scale);
            body.translate(target - body.position);
        }
        self.size = size;
    }
    pub fn advance(&mut self, elapsed: f32) {
        self.accumulator += elapsed.clamp(0., 0.05);
        while self.accumulator + 0.000001 >= STEP {
            self.accumulator = (self.accumulator - STEP).max(0.);
            self.step();
        }
    }
    fn step(&mut self) {
        self.time += STEP;
        let positions: Vec<_> = self
            .bodies
            .iter()
            .map(|body| (body.position, body.radius))
            .collect();
        for (id, body) in self.bodies.iter_mut().enumerate() {
            body.step(self.time, self.size, &positions, id);
        }
    }
}

fn contraction(phase: f32) -> f32 {
    let phase = phase.rem_euclid(1.);
    // Short squeeze, long refill, both with zero velocity at their endpoints.
    let t = if phase < 0.3 {
        phase / 0.3
    } else {
        1. - (phase - 0.3) / 0.7
    };
    t * t * (3. - 2. * t)
}

fn hash(seed: f32) -> f32 {
    ((seed * 12.9898).sin() * 43758.547).rem_euclid(1.)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(ocean: &Ocean) -> Vec<Vec2> {
        ocean
            .bodies
            .iter()
            .flat_map(|b| b.strands.iter().flat_map(|s| s.nodes))
            .collect()
    }

    #[test]
    fn swimming_crosses_the_view_instead_of_hovering() {
        let mut ocean = Ocean::new(Vec2::new(1120., 860.));
        let mut travel = [0_f32; 4];
        let mut span = [(f32::MAX, f32::MIN); 4];
        for _ in 0..60 * 120 {
            let before: Vec<_> = ocean.bodies.iter().map(|b| b.position).collect();
            ocean.advance(STEP);
            for (id, body) in ocean.bodies.iter().enumerate() {
                let p = body.position;
                assert!(p.x.is_finite() && p.y.is_finite());
                assert!(p.x > -body.radius && p.x < ocean.size.x + body.radius);
                let moved = (p - before[id]).length();
                if moved < 50. {
                    travel[id] += moved;
                }
                span[id] = (span[id].0.min(p.x), span[id].1.max(p.x));
            }
        }
        // Each jellyfish swims well over a screen height and drifts sideways.
        assert!(travel.iter().all(|&d| d > 1200.), "travel: {travel:?}");
        assert!(span.iter().all(|s| s.1 - s.0 > 200.), "span: {span:?}");
    }

    #[test]
    fn tentacles_move_fluidly_without_snapping() {
        let mut ocean = Ocean::new(Vec2::new(1120., 860.));
        let mut history = vec![nodes(&ocean), nodes(&ocean)];
        let mut worst = 0_f32;
        // Skip the two samples straddling an off-screen wrap.
        let mut settle = 2;
        for _ in 0..60 * 90 {
            let before: Vec<_> = ocean.bodies.iter().map(|b| b.position).collect();
            ocean.advance(STEP);
            if ocean
                .bodies
                .iter()
                .zip(&before)
                .any(|(b, p)| (b.position - *p).length() > 50.)
            {
                settle = 2;
            }
            let current = nodes(&ocean);
            if settle == 0 {
                for i in 0..current.len() {
                    let jerk = (current[i] - history[1][i]) - (history[1][i] - history[0][i]);
                    worst = worst.max(jerk.length() / (STEP * STEP));
                }
            }
            settle = settle - usize::from(settle > 0);
            history = vec![history.swap_remove(1), current];
        }
        // The previous angle-limited chain peaked above 600,000 px/s².
        assert!(worst < 400., "peak tentacle acceleration {worst} px/s²");
    }

    #[test]
    fn tentacles_remain_attached_and_inextensible_through_wraps_and_resize() {
        let mut ocean = Ocean::new(Vec2::new(1120., 860.));
        for frame in 0..60 * 60 {
            if frame == 600 {
                ocean.resize(Vec2::new(840., 690.));
            }
            ocean.advance(STEP);
            for body in &ocean.bodies {
                for strand in &body.strands {
                    assert!((strand.nodes[0] - body.root(strand)).length() < 0.01);
                    for pair in strand.nodes.windows(2) {
                        assert!(((pair[1] - pair[0]).length() - strand.segment).abs() < 0.01);
                    }
                }
            }
        }
    }

    #[test]
    fn simulation_is_independent_of_display_refresh_rate() {
        let run = |hz: usize| {
            let mut ocean = Ocean::new(Vec2::new(1120., 860.));
            for _ in 0..hz * 12 {
                ocean.advance(1. / hz as f32);
            }
            ocean
        };
        let slow = run(30);
        let fast = run(144);
        for (a, b) in slow.bodies.iter().zip(&fast.bodies) {
            assert!((a.position - b.position).length() < 0.1);
            for (a, b) in a.strands.iter().zip(&b.strands) {
                assert!((a.nodes[NODES - 1] - b.nodes[NODES - 1]).length() < 0.1);
            }
        }
    }
}
