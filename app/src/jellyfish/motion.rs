//! Fixed-step jellyfish swimming, independent of the UI frame rate.
//!
//! Bells stay roughly upright and rise on each pulse in lanes along the left
//! and right of the view, around the centred content, as in the reference
//! video. Each pulse flares the tentacle roots open and shut; every segment
//! eases toward the direction of the one above it, curled a little and drawn
//! toward the way the water streams past. A swing therefore travels down the
//! tentacle as a smooth wave while the ends droop, and there are no
//! constraints or limits that could make it snap.
use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    ops::{Add, Mul, Sub},
};

const STEP: f32 = 1. / 60.;
const NODES: usize = 26;
/// Largest lean away from upright; jellyfish in the reference never roll over.
const MAX_TILT: f32 = 0.5;
/// Water drag on the bell, per second.
const BELL_DRAG: f32 = 1.15;
/// Bells keep to the outer columns of the view, where the content leaves room.
const LANES: [f32; 2] = [0.14, 0.86];

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
    fn angle(self) -> f32 {
        self.y.atan2(self.x)
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
    /// Absolute direction of each segment.
    angles: [f32; NODES - 1],
    /// Broad ribbon-like tentacle, or a fine trailing filament.
    pub thick: bool,
    /// Position across the bell margin, from -1 (left) to 1 (right).
    attachment: f32,
    pub segment: f32,
    /// Curl per segment in radians, waves along the strand, angular speed,
    /// phase, and how far the root sways on its own.
    curl: f32,
    waves: f32,
    speed: f32,
    seed: f32,
    sway: f32,
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
    lane: usize,
    pub strands: Vec<Strand>,
}

impl Swimmer {
    fn new(position: Vec2, radius: f32, lane: usize, id: usize) -> Self {
        let seed = id as f32 * 1.71 + 0.4;
        let phase = id as f32 * 0.37;
        let mut body = Self {
            position,
            velocity: Vec2::new(0., -18.),
            tilt: 0.,
            spin: 0.,
            radius,
            depth: [0.9, 0.8, 1., 0.84][id % 4],
            hue: usize::from(id == 1),
            phase,
            period: 3.3 + id as f32 * 0.29,
            contraction: contraction(phase),
            seed,
            lane,
            strands: Vec::new(),
        };
        // Six broad tentacles across the margin and two fine filaments.
        let attachments = [-0.8, -0.5, -0.18, 0.16, 0.48, 0.78, -0.62, 0.66];
        for (strand, &attachment) in attachments.iter().enumerate() {
            let thick = strand < 6;
            let k = strand as f32 + seed;
            let length = radius
                * if thick {
                    2.1 + (k * 1.7).sin().abs() * 0.7
                } else {
                    3.1 + (k * 1.3).sin().abs() * 0.5
                };
            body.strands.push(Strand {
                nodes: [position; NODES],
                angles: [0.; NODES - 1],
                thick,
                attachment,
                segment: length / (NODES - 1) as f32,
                curl: if thick { 0.05 } else { 0.07 } + (k * 2.3).sin().abs() * 0.025,
                waves: 0.7 + (k * 1.1).sin().abs() * 0.5,
                speed: 1.1 + (k * 0.7).sin().abs() * 0.5,
                seed: k * 2.4,
                sway: 0.22 + (k * 1.9).sin().abs() * 0.18,
            });
        }
        for index in 0..body.strands.len() {
            body.strands[index].angles = [body.tilt + FRAC_PI_2; NODES - 1];
            body.rebuild(index);
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
            width: self.radius * (1. - 0.15 * self.contraction),
            height: self.radius * (0.8 + 0.16 * self.contraction),
        }
    }
    fn root(&self, strand: &Strand) -> Vec2 {
        let bell = self.bell();
        let across = if strand.thick { 0.92 } else { 0.7 };
        self.position
            + self.side() * (bell.width * strand.attachment * across)
            + self.tail() * (bell.height * 0.24)
    }

    /// Eases every segment toward its target: the root flares open as the
    /// bell refills and closes on each squeeze, and each further segment
    /// follows the one above it, curling slightly and settling toward the
    /// direction the water streams past. Swings thus flow toward the tip.
    fn swim_tentacles(&mut self, time: f32) {
        let base = self.tilt + FRAC_PI_2;
        let open = 1. - self.contraction;
        // Tentacles stream away from the bell's motion and sink a little.
        let stream = (self.tail() - self.velocity * (1. / 40.) + Vec2::new(0., 0.35)).angle();
        let follow = 1. - (-13. * STEP).exp();
        for index in 0..self.strands.len() {
            let strand = &mut self.strands[index];
            let sway = strand.sway * (time * strand.speed * 0.55 + strand.seed).sin();
            let mut target = base - strand.attachment * (0.25 + 1.1 * open) + sway;
            for i in 0..NODES - 1 {
                if i > 0 {
                    let s = i as f32 / (NODES - 2) as f32;
                    let above = strand.angles[i - 1];
                    let wave = (TAU * strand.waves * s - time * strand.speed + strand.seed).sin();
                    target = above + strand.curl * wave + 0.035 * angle_difference(above, stream);
                }
                // Ease off before a tentacle could fold back over the bell.
                let bend = angle_difference(base, target);
                let target = base + 2.2 * (bend / 2.2).tanh();
                let angle = &mut strand.angles[i];
                *angle += angle_difference(*angle, target) * follow;
            }
            self.rebuild(index);
        }
    }

    fn rebuild(&mut self, index: usize) {
        let root = self.root(&self.strands[index]);
        let strand = &mut self.strands[index];
        strand.nodes[0] = root;
        for i in 1..NODES {
            strand.nodes[i] =
                strand.nodes[i - 1] + Vec2::direction(strand.angles[i - 1]) * strand.segment;
        }
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
            for p in &mut strand.nodes {
                *p = *p + offset;
            }
        }
    }

    fn step(&mut self, time: f32, size: Vec2, others: &[(Vec2, f32, usize)], id: usize) {
        let next = contraction(time / self.period + self.phase);
        let squeeze = ((next - self.contraction) / STEP).max(0.);
        self.contraction = next;

        // Steering only changes the lean, through a critically damped spring,
        // so turns build and fade gradually.
        let lane = LANES[self.lane] + 0.05 * (time * 0.05 + self.seed * 2.7).sin();
        let mut goal = ((lane * size.x - self.position.x) / (0.1 * size.x) * 0.3)
            .clamp(-0.35, 0.35)
            + 0.16 * (time * 0.13 + self.seed).sin()
            + 0.08 * (time * 0.31 + self.seed * 1.9).sin();
        let margin = self.radius * 1.1;
        goal += 0.6
            * (((margin - self.position.x) / margin).max(0.)
                - ((self.position.x - size.x + margin) / margin).max(0.));
        // Give way to a jellyfish just ahead instead of swimming through it.
        let mut yield_to = 1_f32;
        for (other, &(position, radius, _)) in others.iter().enumerate() {
            let away = self.position - position;
            let range = (self.radius + radius) * 1.6;
            let closeness = (1. - away.length() / range).max(0.);
            if other != id && closeness > 0. {
                goal += away.x.signum() * 0.3 * closeness;
                if away.y > 0. {
                    yield_to = yield_to.min(1. - 0.6 * closeness);
                }
            }
        }
        let goal = goal.clamp(-MAX_TILT, MAX_TILT);
        let omega = 0.85;
        self.spin += ((goal - self.tilt) * omega * omega - self.spin * 2. * omega) * STEP;
        self.tilt += self.spin * STEP;

        // Each contraction pushes the bell forward; drag slows the glide while
        // the bell refills, and a slight negative buoyancy sinks it a little.
        let scale = self.radius / 85.;
        let thrust = self.forward() * ((12. + squeeze * 46.) * scale * yield_to);
        let sink = Vec2::new(0., 3.5);
        self.velocity = (self.velocity + (thrust + sink) * STEP) * (-BELL_DRAG * STEP).exp();
        self.position = self.position + self.velocity * STEP;

        // Once the whole animal has left through the top it re-enters from
        // below, out of view, in the emptier lane. If both lanes are busy near
        // the bottom it waits off screen.
        let (low, high) = self.extent();
        if high.y < -6. {
            let crowd = |lane: usize| {
                others.iter().enumerate().any(|(other, &(p, radius, l))| {
                    other != id && l == lane && p.y > size.y - (self.radius + radius) * 2.6
                })
            };
            let count = |lane: usize| others.iter().filter(|o| o.2 == lane).count();
            let mut lanes = [1 - self.lane, self.lane];
            if count(lanes[1]) < count(lanes[0]) {
                lanes.swap(0, 1);
            }
            if let Some(&lane) = lanes.iter().find(|&&lane| !crowd(lane)) {
                self.lane = lane;
                let x = (LANES[lane] + 0.06 * (hash(self.seed + time) - 0.5)) * size.x;
                self.translate(Vec2::new(x - self.position.x, size.y + 6. - low.y));
            }
        }

        self.swim_tentacles(time);
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
            (0, 0.24, 84.),
            (1, 0.3, 66.),
            (0, 0.84, 92.),
            (1, 0.94, 100.),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (lane, y, radius))| {
            let position = Vec2::new(LANES[lane], y).scaled(size);
            Swimmer::new(position, radius, lane, id)
        })
        .collect();
        let mut ocean = Self {
            size,
            bodies,
            time: 0.,
            accumulator: 0.,
        };
        // Let the tentacles settle into their motion before the first frame.
        for _ in 0..240 {
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
        let others: Vec<_> = self
            .bodies
            .iter()
            .map(|body| (body.position, body.radius, body.lane))
            .collect();
        for (id, body) in self.bodies.iter_mut().enumerate() {
            body.step(self.time, self.size, &others, id);
        }
    }
}

fn angle_difference(from: f32, to: f32) -> f32 {
    (to - from + PI).rem_euclid(TAU) - PI
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

    const SIZE: Vec2 = Vec2::new(1120., 860.);

    fn nodes(ocean: &Ocean) -> Vec<Vec2> {
        ocean
            .bodies
            .iter()
            .flat_map(|b| b.strands.iter().flat_map(|s| s.nodes))
            .collect()
    }

    #[test]
    fn swimming_circulates_along_the_sides_of_the_view() {
        let mut ocean = Ocean::new(SIZE);
        let mut travel = [0_f32; 4];
        let mut lanes = [[false; 2]; 4];
        let (mut visible, mut central) = (0, 0);
        for _ in 0..60 * 240 {
            let before: Vec<_> = ocean.bodies.iter().map(|b| b.position).collect();
            ocean.advance(STEP);
            for (id, body) in ocean.bodies.iter().enumerate() {
                let p = body.position;
                assert!(p.x.is_finite() && p.y.is_finite());
                assert!(p.x > -body.radius && p.x < SIZE.x + body.radius);
                let moved = (p - before[id]).length();
                if moved < 50. {
                    travel[id] += moved;
                }
                lanes[id][body.lane] = true;
                if p.y > 0. && p.y < SIZE.y {
                    visible += 1;
                    central += usize::from((p.x / SIZE.x - 0.5).abs() < 0.2);
                }
            }
        }
        // Every jellyfish keeps swimming, visits both sides, and stays out of
        // the middle where the image sits.
        assert!(travel.iter().all(|&d| d > 2000.), "travel: {travel:?}");
        assert!(lanes.iter().all(|l| l[0] && l[1]), "lanes: {lanes:?}");
        assert!(
            central * 20 < visible,
            "{central} of {visible} samples central"
        );
    }

    #[test]
    fn tentacles_sway_visibly() {
        let mut ocean = Ocean::new(SIZE);
        let body = 0;
        let mut samples = Vec::new();
        for _ in 0..60 * 20 {
            ocean.advance(STEP);
            let b = &ocean.bodies[body];
            for strand in &b.strands {
                let tip = strand.nodes[NODES - 1] - b.position;
                samples.push((tip.dot(b.side()) / b.radius, strand.thick));
            }
        }
        let thick: Vec<_> = samples.iter().filter(|s| s.1).map(|s| s.0).collect();
        let mean = thick.iter().sum::<f32>() / thick.len() as f32;
        let spread = thick.iter().map(|x| (x - mean).abs()).sum::<f32>() / thick.len() as f32;
        // Tips sweep sideways by around a bell radius, as in the reference.
        assert!(spread > 0.6, "mean sideways tip deviation {spread} radii");
    }

    #[test]
    fn tentacles_move_fluidly_without_snapping() {
        let mut ocean = Ocean::new(SIZE);
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
        // The original angle-limited chain peaked above 600,000 px/s².
        assert!(worst < 600., "peak tentacle acceleration {worst} px/s²");
    }

    #[test]
    fn tentacles_remain_attached_and_inextensible_through_wraps_and_resize() {
        let mut ocean = Ocean::new(SIZE);
        for frame in 0..60 * 90 {
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
            let mut ocean = Ocean::new(SIZE);
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
