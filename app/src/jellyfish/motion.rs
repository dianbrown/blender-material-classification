//! Fixed-step swimming and inertial tentacles, independent of the UI frame rate.
use std::{
    f32::consts::{PI, TAU},
    ops::{Add, Mul, Sub},
};

const STEP: f32 = 1. / 60.;
const NODES: usize = 25;
const DESTINATIONS: [Vec2; 8] = [
    Vec2::new(0.82, 0.16),
    Vec2::new(0.16, 0.36),
    Vec2::new(0.72, 0.83),
    Vec2::new(0.35, 0.12),
    Vec2::new(0.14, 0.78),
    Vec2::new(0.86, 0.44),
    Vec2::new(0.46, 0.87),
    Vec2::new(0.58, 0.22),
];

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
    fn unit(self) -> Self {
        self * (1. / self.length().max(0.001))
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
    previous: [Vec2; NODES],
    pub oral: bool,
    attachment: f32,
    segment: f32,
}

pub(super) struct Bell {
    pub width: f32,
    pub height: f32,
}

pub(super) struct Swimmer {
    pub position: Vec2,
    velocity: Vec2,
    heading: f32,
    pub radius: f32,
    pub depth: f32,
    phase: f32,
    period: f32,
    contraction: f32,
    destination: usize,
    pub strands: Vec<Strand>,
}

impl Swimmer {
    fn new(position: Vec2, radius: f32, heading: f32, id: usize) -> Self {
        let phase = id as f32 * 0.27;
        let mut body = Self {
            position,
            velocity: Vec2::direction(heading) * 25.,
            heading,
            radius,
            depth: [0.78, 0.92, 0.72, 1.][id],
            phase,
            period: 2.8 + id as f32 * 0.27,
            contraction: contraction(phase),
            destination: [0, 1, 3, 5][id],
            strands: Vec::new(),
        };
        // Eleven marginal filaments and four thicker, folded oral arms.
        for strand in 0..15 {
            let oral = strand >= 11;
            let attachment = if oral {
                (strand - 11) as f32 / 3. * 0.9 - 0.45
            } else {
                strand as f32 / 10. * 1.76 - 0.88
            };
            let length = radius
                * if oral {
                    2.45 + (strand as f32 * 1.7).sin().abs() * 0.65
                } else {
                    3.0 + (strand as f32 * 1.9).sin().abs() * 1.15
                };
            let root = body.root(attachment);
            let nodes = std::array::from_fn(|node| {
                let t = node as f32 / (NODES - 1) as f32;
                root + body.tail() * (length * t)
                    + body.side() * ((t * 7. + strand as f32).sin() * radius * 0.12 * t)
            });
            body.strands.push(Strand {
                nodes,
                previous: nodes,
                oral,
                attachment,
                segment: length / (NODES - 1) as f32,
            });
        }
        body
    }

    pub fn side(&self) -> Vec2 {
        Vec2::direction(self.heading + PI * 0.5)
    }
    pub fn tail(&self) -> Vec2 {
        Vec2::direction(self.heading + PI)
    }
    pub fn bell(&self) -> Bell {
        Bell {
            width: self.radius * (1. - 0.32 * self.contraction),
            height: self.radius * (0.54 + 0.30 * self.contraction),
        }
    }
    fn root(&self, attachment: f32) -> Vec2 {
        let bell = self.bell();
        self.position + self.side() * (bell.width * attachment) + self.tail() * (bell.height * 0.13)
    }

    fn step(&mut self, time: f32, size: Vec2, others: &[(Vec2, f32)], id: usize) {
        let next_contraction = contraction(time / self.period + self.phase);
        let squeeze = ((next_contraction - self.contraction) / STEP).max(0.);
        self.contraction = next_contraction;
        let target = DESTINATIONS[self.destination].scaled(size);
        if (target - self.position).length() < self.radius {
            self.destination = (self.destination + 1) % DESTINATIONS.len();
        }
        let mut desired = (target - self.position).unit();
        // Slow changes in heading, plus a soft edge current. There is no
        // center exclusion zone, anchor, teleport or repeating hover orbit.
        desired =
            desired + Vec2::direction(self.heading + (time * 0.31 + self.phase * 13.).sin()) * 0.28;
        for (other_id, &(position, radius)) in others.iter().enumerate() {
            let away = self.position - position;
            let range = self.radius + radius;
            if other_id != id && away.length() < range {
                desired = desired + away.unit() * (1. - away.length() / range) * 1.5;
            }
        }
        let margin = self.radius * 0.8;
        let edge = Vec2::new(
            ((margin - self.position.x) / margin).max(0.)
                - ((self.position.x - size.x + margin) / margin).max(0.),
            ((margin - self.position.y) / margin).max(0.)
                - ((self.position.y - size.y + margin) / margin).max(0.),
        );
        desired = desired + edge * 4.;
        let turn = angle_difference(self.heading, desired.angle()).clamp(-0.48 * STEP, 0.48 * STEP);
        self.heading += turn;
        // A quick power stroke accelerates the animal; drag slows the glide
        // while the bell refills. Bell deformation and travel share one cycle.
        let forward = Vec2::direction(self.heading);
        self.velocity =
            (self.velocity + forward * ((12. + squeeze * 48.) * STEP)) * (-0.78 * STEP).exp();
        self.position = self.position + self.velocity * STEP;

        let side = self.side();
        let tail = self.tail();
        let bell = self.bell();
        for (strand_id, strand) in self.strands.iter_mut().enumerate() {
            let root = self.position
                + side * (bell.width * strand.attachment)
                + tail * (bell.height * 0.13);
            strand.nodes[0] = root;
            strand.previous[0] = root;
            let mut previous_angle = tail.angle();
            for node in 1..NODES {
                let old = strand.nodes[node];
                let progress = node as f32 / (NODES - 1) as f32;
                let current = Vec2::new(
                    (old.y * 0.009 + time * 0.7).sin() * 18.,
                    (old.x * 0.007 - time * 0.5).sin() * 10. + 6.,
                ) + side
                    * ((progress * 9. - time * 1.8 + strand_id as f32 * 0.8).sin()
                        * 14.
                        * progress);
                let predicted =
                    old + (old - strand.previous[node]) * 0.92 + current * (STEP * STEP);
                strand.previous[node] = old;
                // Inertial follow-the-leader chain, adapting the shark spine:
                // preserve segment lengths and limit bends, while allowing the
                // free tips to retain momentum through turns and contractions.
                let direction = (predicted - strand.nodes[node - 1]).angle();
                let max_bend = 0.10 + progress * 0.10;
                let angle = previous_angle
                    + angle_difference(previous_angle, direction).clamp(-max_bend, max_bend);
                strand.nodes[node] =
                    strand.nodes[node - 1] + Vec2::direction(angle) * strand.segment;
                previous_angle = angle;
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
            (0.15, 0.30, 76., -1.3),
            (0.60, 0.19, 92., -0.7),
            (0.88, 0.59, 78., -1.9),
            (0.38, 0.82, 102., -1.2),
        ]
        .into_iter()
        .enumerate()
        .map(|(id, (x, y, radius, heading))| {
            Swimmer::new(Vec2::new(x, y).scaled(size), radius, heading, id)
        })
        .collect();
        Self {
            size,
            bodies,
            time: 0.,
            accumulator: 0.,
        }
    }
    pub fn resize(&mut self, size: Vec2) {
        if size.x < 1. || size.y < 1. || (size - self.size).length() < 0.5 {
            return;
        }
        let scale = Vec2::new(size.x / self.size.x, size.y / self.size.y);
        let creature_scale =
            (size.x / 1120.).clamp(0.7, 1.25) / (self.size.x / 1120.).clamp(0.7, 1.25);
        for body in &mut self.bodies {
            body.position = body.position.scaled(scale);
            body.radius *= creature_scale;
            for strand in &mut body.strands {
                strand.segment *= creature_scale;
                for p in strand.nodes.iter_mut().chain(strand.previous.iter_mut()) {
                    *p = p.scaled(scale);
                }
            }
        }
        self.size = size;
    }
    pub fn advance(&mut self, elapsed: f32) {
        self.accumulator += elapsed.clamp(0., 0.05);
        while self.accumulator + 0.000001 >= STEP {
            self.accumulator = (self.accumulator - STEP).max(0.);
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
}

fn angle_difference(from: f32, to: f32) -> f32 {
    (to - from + PI).rem_euclid(TAU) - PI
}
fn contraction(phase: f32) -> f32 {
    let phase = phase.rem_euclid(1.);
    // Short squeeze, long refill, both with zero velocity at their endpoints.
    let t = if phase < 0.24 {
        phase / 0.24
    } else {
        1. - (phase - 0.24) / 0.76
    };
    t * t * (3. - 2. * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swimming_traverses_the_view_instead_of_hovering() {
        let mut ocean = Ocean::new(Vec2::new(1120., 860.));
        let initial: Vec<_> = ocean.bodies.iter().map(|b| b.position).collect();
        let mut farthest = [0_f32; 4];
        for _ in 0..60 * 90 {
            ocean.advance(STEP);
            for (id, body) in ocean.bodies.iter().enumerate() {
                farthest[id] = farthest[id].max((body.position - initial[id]).length());
                assert!(body.position.x.is_finite() && body.position.y.is_finite());
                assert!(body.position.x > -150. && body.position.x < ocean.size.x + 150.);
                assert!(body.position.y > -150. && body.position.y < ocean.size.y + 150.);
            }
        }
        assert!(
            farthest.iter().all(|&distance| distance > 500.),
            "travel: {farthest:?}"
        );
    }

    #[test]
    fn tentacles_remain_attached_and_inextensible_through_turns_and_resize() {
        let mut ocean = Ocean::new(Vec2::new(1120., 860.));
        for frame in 0..60 * 30 {
            if frame == 600 {
                ocean.resize(Vec2::new(840., 690.));
            }
            ocean.advance(STEP);
            for body in &ocean.bodies {
                for strand in &body.strands {
                    assert!((strand.nodes[0] - body.root(strand.attachment)).length() < 0.01);
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
