use crate::common::pairs;
use crate::common::*;
use crate::math::*;
use crate::nn::*;
use crate::physics::walls_from_points;
use crate::physics::Car;
use crate::physics::CarInput;
use crate::physics::PhysicsParameters;
use crate::physics::Reward;
use crate::physics::Wall;
use crate::physics::*;
use cmaes::DVector;
use cmaes::PlotOptions;
use core::f32::consts::TAU;
use differential_evolution::self_adaptive_de;
use egui::emath::RectTransform;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::Painter;
use egui::Pos2;
use egui::Shape;
use egui::Stroke;
use egui::Vec2;
use rand::thread_rng;
use rand::Rng;
use spiril::population::Population;
use spiril::unit::Unit;
use std::time::Instant;

pub fn mirror_horizontally(mut input: Track) -> Track {
    let car_center = 250.;
    input.walls.iter_mut().for_each(|x| {
        x.center.y = car_center - (x.center.y - car_center);
        x.angle_sin = -x.angle_sin;
    });
    input
        .rewards
        .iter_mut()
        .for_each(|x| x.center.y = car_center - (x.center.y - car_center));
    input.name += "_mirror";
    input
}

#[derive(Clone)]
pub struct Track {
    pub name: String,
    pub walls: Vec<Wall>,
    pub rewards: Vec<Reward>,
}

impl Track {
    fn new(name: String, walls: Vec<Wall>, rewards: Vec<Reward>) -> Self {
        Self {
            name,
            walls,
            rewards,
        }
    }
}

// straight line
pub fn track_straight_line() -> Track {
    Track::new(
        "straight".to_owned(),
        walls_from_points(
            vec![
                pos2(-8.00, 13.00),
                pos2(5635.54, 246.73),
                pos2(5630.75, 488.32),
                pos2(-100.47, 409.92),
                pos2(-14.23, 17.90),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(516.17, 234.58),
                pos2(1918.37, 265.53),
                pos2(4208.82, 325.74),
                pos2(5510.92, 362.09),
            ]
            .into_iter(),
        ),
    )
}

// turn right smooth
pub fn track_turn_right_smooth() -> Track {
    Track::new(
        "turn_right_smooth".to_owned(),
        walls_from_points(
            vec![
                pos2(83.33, 3.46),
                pos2(1227.77, -3.24),
                pos2(2054.61, 103.97),
                pos2(2971.24, 374.67),
                pos2(3590.36, 815.56),
                pos2(3966.93, 1354.28),
                pos2(4023.21, 1911.76),
                pos2(4039.29, 2559.02),
                pos2(3669.42, 2560.36),
                pos2(3660.04, 1903.72),
                pos2(3590.36, 1490.97),
                pos2(3282.13, 1095.64),
                pos2(2814.44, 787.42),
                pos2(1999.67, 558.26),
                pos2(1249.21, 438.99),
                pos2(86.01, 445.69),
                pos2(84.67, -1.90),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(651.25, 247.73),
                pos2(1172.18, 240.78),
                pos2(1725.53, 297.51),
                pos2(2405.05, 438.74),
                pos2(2972.29, 632.06),
                pos2(3428.39, 942.30),
                pos2(3768.74, 1398.41),
                pos2(3845.14, 1955.22),
                pos2(3852.09, 2154.33),
                pos2(3848.61, 2261.99),
                pos2(3852.09, 2371.97),
                pos2(3846.30, 2448.37),
            ]
            .into_iter(),
        ),
    )
}

// harn turn left
pub fn track_turn_left_90() -> Track {
    Track::new(
        "turn_left_90".to_owned(),
        walls_from_points(
            vec![
                pos2(80.34, -5.05),
                pos2(997.84, -6.53),
                pos2(997.84, -1646.50),
                pos2(1457.33, -1649.46),
                pos2(1466.19, 534.11),
                pos2(87.06, 524.75),
                pos2(78.86, -5.05),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(581.19, 257.00),
                pos2(1161.83, 110.73),
                pos2(1229.80, -449.22),
                pos2(1229.80, -579.24),
                pos2(1228.32, -704.82),
                pos2(1228.32, -814.16),
                pos2(1226.84, -941.22),
                pos2(1223.89, -1047.59),
                pos2(1222.41, -1186.47),
                pos2(1222.41, -1295.81),
                pos2(1220.93, -1406.62),
                pos2(1220.93, -1508.56),
            ]
            .into_iter(),
        ),
    )
}

pub fn track_turn_left_180() -> Track {
    Track::new(
        "turn_left_180".to_owned(),
        walls_from_points(
            vec![
                pos2(69.00, 6.00),
                pos2(75.16, 494.65),
                pos2(1628.53, 506.55),
                pos2(1626.82, -437.56),
                pos2(70.41, -430.71),
                pos2(68.70, -37.34),
                pos2(1219.76, -44.18),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(578.38, 244.87),
                pos2(976.89, 246.58),
                pos2(1382.24, 149.09),
                pos2(1378.82, -187.85),
                pos2(1035.04, -246.00),
                pos2(841.78, -239.16),
                pos2(674.16, -235.74),
                pos2(551.02, -237.45),
                pos2(429.59, -237.45),
                pos2(270.52, -230.61),
            ]
            .into_iter(),
        ),
    )
}

pub fn track_turn_around() -> Track {
    Track::new(
        "turn_around".to_owned(),
        walls_from_points(
            vec![
                pos2(550.85, -8.85),
                pos2(552.00, 501.00),
                pos2(-2048.97, 486.81),
                pos2(-2051.63, -17.28),
                pos2(521.62, -12.89),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(-200.11, 253.13),
                pos2(-959.28, 250.08),
                pos2(-1433.05, 248.37),
                pos2(-1783.67, 248.37),
            ]
            .into_iter(),
        ),
    )
}

pub fn track_smooth_left_and_right() -> Track {
    Track::new(
        "smooth_left_and_right".to_owned(),
        walls_from_points(
            vec![
                pos2(116.58, 91.49),
                pos2(1391.74, -7.79),
                pos2(1805.02, -432.51),
                pos2(1927.44, -1337.17),
                pos2(2159.11, -1920.08),
                pos2(2786.96, -2298.42),
                pos2(3761.07, -2298.42),
                pos2(4246.98, -1890.44),
                pos2(4414.30, -1315.14),
                pos2(4428.05, -613.78),
                pos2(4150.72, 46.32),
                pos2(3804.62, -109.54),
                pos2(4024.65, -666.50),
                pos2(4017.78, -1223.46),
                pos2(3839.00, -1718.54),
                pos2(3694.60, -1917.94),
                pos2(2876.35, -1911.06),
                pos2(2507.34, -1652.07),
                pos2(2351.48, -1347.23),
                pos2(2285.36, -489.60),
                pos2(1992.89, 117.08),
                pos2(1633.33, 401.70),
                pos2(976.30, 512.67),
                pos2(100.88, 529.78),
                pos2(110.83, 101.35),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(764.37, 265.17),
                pos2(1297.73, 199.50),
                pos2(1729.24, -2.85),
                pos2(1961.08, -274.89),
                pos2(2045.50, -588.47),
                pos2(2108.49, -898.03),
                pos2(2144.67, -1210.28),
                pos2(2221.06, -1507.78),
                pos2(2369.81, -1773.12),
                pos2(2631.13, -1966.09),
                pos2(2907.18, -2101.44),
                pos2(3228.81, -2106.80),
                pos2(3555.79, -2112.16),
                pos2(3842.57, -2023.71),
                pos2(4023.48, -1771.78),
                pos2(4121.31, -1490.36),
                pos2(4200.37, -1187.49),
                pos2(4200.38, -871.23),
                pos2(4191.00, -558.99),
                pos2(4128.01, -404.88),
                pos2(4074.41, -274.89),
                pos2(4048.94, -197.16),
            ]
            .into_iter(),
        ),
    )
}

#[allow(clippy::approx_constant)]
pub fn track_complex() -> Track {
    Track::new(
        "complex".to_owned(),
        vec![
            Wall::new(pos2(1396.0, 600.0), vec2(25.0, 1500.0), 1.5708),
            Wall::new(pos2(1065.0, 0.0), vec2(25.0, 1121.2), 1.5708),
            Wall::new(pos2(-13.0, 300.0), vec2(25.0, 353.2), 0.0000),
            Wall::new(pos2(2882.0, -127.0), vec2(25.0, 750.0), 0.0000),
            Wall::new(pos2(2146.0, -721.0), vec2(25.0, 750.0), 0.0000),
            Wall::new(pos2(2912.0, -1423.0), vec2(25.0, 828.4), 1.5708),
            Wall::new(pos2(3690.0, 861.0), vec2(25.0, 2416.0), 0.0000),
            Wall::new(pos2(445.0, 1694.0), vec2(25.0, 3689.0), 5.1487),
            Wall::new(pos2(-2830.0, -1535.0), vec2(25.0, 1905.0), 0.0000),
            Wall::new(pos2(-913.0, 112.0), vec2(25.0, 1104.0), 2.0420),
            Wall::new(pos2(-2062.0, -1517.0), vec2(25.0, 1375.0), 6.0563),
            Wall::new(pos2(-1055.0, -3250.0), vec2(25.0, 1905.0), 1.5708),
            Wall::new(pos2(-760.0, -2845.0), vec2(25.0, 1625.0), 1.5882),
            Wall::new(pos2(750.0, -3050.0), vec2(25.0, 320.0), 0.0000),
        ],
        vec![
            Reward::new(pos2(600.0, 290.0), 115.0),
            Reward::new(pos2(1215.0, 290.0), 120.0),
            Reward::new(pos2(1825.0, 310.0), 280.0),
            Reward::new(pos2(2470.0, 180.0), 400.0),
            Reward::new(pos2(2505.0, -515.0), 355.0),
            Reward::new(pos2(2525.0, -1040.0), 350.0),
            Reward::new(pos2(2920.0, -1140.0), 245.0),
            Reward::new(pos2(3300.0, -1040.0), 330.0),
            Reward::new(pos2(3280.0, -245.0), 345.0),
            Reward::new(pos2(3295.0, 510.0), 350.0),
            Reward::new(pos2(2860.0, 1020.0), 300.0),
            Reward::new(pos2(-660.0, 725.0), 500.0),
            Reward::new(pos2(-2465.0, -185.0), 705.0),
            Reward::new(pos2(-2560.0, -2945.0), 500.0),
            Reward::new(pos2(-560.0, -3030.0), 500.0),
            Reward::new(pos2(-140.0, -3065.0), 500.0),
            Reward::new(pos2(150.0, -3075.0), 500.0),
            Reward::new(pos2(395.0, -3040.0), 500.0),
        ],
    )
}

pub fn track_straight_45() -> Track {
    Track::new(
        "straight_45".to_owned(),
        walls_from_points(
            vec![
                pos2(-77.00, 178.00),
                pos2(2558.78, -2103.75),
                pos2(2892.96, -1732.53),
                pos2(256.05, 574.77),
                pos2(-86.66, 186.21),
            ]
            .into_iter(),
        ),
        rewards_from_points(
            vec![
                pos2(430.68, 109.81),
                pos2(478.70, 57.43),
                pos2(548.55, -1.51),
                pos2(642.42, -88.83),
                pos2(725.37, -169.59),
                pos2(823.60, -256.91),
                pos2(1048.43, -459.92),
                pos2(1266.72, -647.64),
                pos2(1714.21, -1038.38),
                pos2(1934.68, -1237.02),
                pos2(2083.11, -1354.89),
                pos2(2242.46, -1485.87),
                pos2(2390.90, -1616.84),
                pos2(2506.59, -1719.43),
                pos2(2615.74, -1804.57),
            ]
            .into_iter(),
        ),
    )
}

pub fn get_all_tracks() -> Vec<Track> {
    vec![
        track_straight_line(),
        track_straight_45(),
        track_turn_right_smooth(),
        track_smooth_left_and_right(),
        track_turn_left_90(),
        track_turn_left_180(),
        track_turn_around(),
        track_complex(),
    ]
    .into_iter()
    .flat_map(|x| vec![x.clone(), mirror_horizontally(x)].into_iter())
    .collect()
}

pub struct RewardPathProcessor {
    rewards: Vec<Reward>,
    // rewards_acquired: usize,
    current_segment: usize,
    prev_distance: f32,
    max_distance: f32,
}

impl RewardPathProcessor {
    pub fn new(rewards: Vec<Reward>) -> Self {
        Self {
            rewards,
            // rewards_acquired: 0,
            current_segment: 0,
            prev_distance: 0.,
            max_distance: 0.,
        }
    }

    fn process_point(&mut self, point: Pos2) -> f32 {
        let (
            rewards,
            prev,
            max,
            // rewards_acquired,
            current_segment,
        ) = (
            &mut self.rewards,
            &mut self.prev_distance,
            &mut self.max_distance,
            // &mut self.rewards_acquired,
            &mut self.current_segment,
        );

        #[allow(unused_mut)]
        let mut reward = 0.;

        if rewards.last_mut().unwrap().process_pos(point) {
            reward += 1.;
        }
        // if *rewards_acquired < rewards.len() && rewards[*rewards_acquired].process_pos(point) {
        //     reward += 1.;
        //     *rewards_acquired += 1;
        // }

        if 2 <= rewards.len() && *current_segment + 1 < rewards.len() {
            let a = rewards[*current_segment].center;
            let b = rewards[*current_segment + 1].center;
            let (projection, mut t) = project_to_segment(point, a, b);
            let dist_on_line = (projection - a).length();
            let dist_from_point = (projection - point).length();
            let is_adequate = dist_from_point < 200.;
            let current_dist = *prev + dist_on_line;
            if current_dist > *max && is_adequate {
                // reward += (current_dist - *max) * 1. / (dist_from_point + 1.0);
                *max = current_dist;
            }

            if *current_segment + 2 < rewards.len() {
                let a2 = rewards[*current_segment + 1].center;
                let b2 = rewards[*current_segment + 2].center;
                let (projection2, _) = project_to_segment(point, a2, b2);
                let dist_from_point2 = (projection2 - point).length();
                let is_adequate2 = dist_from_point2 < 200.;
                if dist_from_point2 < dist_from_point && is_adequate2 {
                    t = 1.;
                    let dist_on_line2 = (projection - a2).length();
                    let current_dist2 = *prev + (b - a).length() + dist_on_line2;
                    if current_dist2 > *max {
                        // reward += (current_dist2 - *max) * 1. / (dist_from_point2 + 1.0);
                        *max = current_dist2;
                    }
                }
            }

            if t == 1. && is_adequate {
                *prev += (b - a).length();
                *current_segment += 1;
            }
        }

        reward
    }

    fn reset(&mut self) {
        // self.rewards_acquired = 0;
        self.current_segment = 0;
        self.prev_distance = 0.;
        self.max_distance = 0.;
        self.rewards.iter_mut().for_each(|x| x.acquired = false);
    }

    fn draw(&self, point: Pos2, painter: &Painter, to_screen: &RectTransform) {
        if 2 <= self.rewards.len() && self.current_segment + 1 < self.rewards.len() {
            let a = self.rewards[self.current_segment].center;
            let b = self.rewards[self.current_segment + 1].center;
            let (projection, _) = project_to_segment(point, a, b);
            painter.add(Shape::line(
                vec![
                    to_screen.transform_pos(point),
                    to_screen.transform_pos(projection),
                ],
                Stroke::new(1.0, Color32::DARK_GREEN),
            ));
        }

        for (i, (a, b)) in pairs(self.rewards.iter()).enumerate() {
            painter.add(Shape::line(
                vec![
                    to_screen.transform_pos(a.center),
                    to_screen.transform_pos(b.center),
                ],
                Stroke::new(
                    1.0,
                    if i < self.current_segment {
                        Color32::DARK_RED
                    } else {
                        Color32::DARK_GREEN
                    },
                ),
            ));
        }

        for reward in &self.rewards {
            painter.add(Shape::closed_line(
                reward
                    .get_points()
                    .into_iter()
                    .map(|p| to_screen.transform_pos(p))
                    .collect(),
                Stroke::new(
                    1.0,
                    if reward.acquired {
                        Color32::RED
                    } else {
                        Color32::GREEN
                    },
                ),
            ));
        }
    }

    pub fn max_possible_distance(&self) -> f32 {
        pairs(self.rewards.iter())
            .map(|(a, b)| (b.center - a.center).length())
            .sum()
    }

    pub fn distance_percent(&self) -> f32 {
        self.max_distance / self.max_possible_distance()
    }

    pub fn all_acquired(&self) -> bool {
        self.rewards_acquired() == self.rewards.len()
        // self.rewards_acquired == self.rewards.len()
    }

    pub fn rewards_acquired(&self) -> usize {
        if self.rewards.last().unwrap().acquired {
            self.rewards.len()
        } else {
            0
        }
        // self.rewards.iter().filter(|x| x.acquired).count()
        // self.rewards_acquired
    }

    pub fn rewards_acquired_percent(&self) -> f32 {
        if self.rewards.is_empty() {
            0.
        } else {
            self.rewards_acquired() as f32 / self.rewards.len() as f32
        }
    }
}

pub struct CarSimulation {
    pub car: Car,
    pub penalty: f32,
    pub reward: f32,
    pub time_passed: f32,
    pub input_values: Vec<f32>,
    pub next_values: Vec<f32>,
    pub prev_output: Vec<f32>,
    pub dirs: Vec<Pos2>,
    pub distances: Vec<f32>,
    pub nn: NeuralNetwork,
    pub walls: Vec<Wall>,
    pub reward_path_processor: RewardPathProcessor,
}

impl CarSimulation {
    const PASS_TIME: bool = false;
    const PASS_DIRS: bool = true;
    const PASS_INTERNALS: bool = false;
    const PASS_PREV_OUTPUT: bool = false;
    const DIRS_N: usize = 5;
    const PASS_NEXT_SIZE: usize = 0;

    const CAR_INPUT_SIZE: usize = InternalCarValues::SIZE;
    const CAR_OUTPUT_SIZE: usize = CarInput::SIZE;

    pub fn get_total_input_neurons() -> usize {
        Self::PASS_TIME as usize
            + Self::PASS_DIRS as usize * Self::DIRS_N
            + Self::PASS_INTERNALS as usize * Self::CAR_INPUT_SIZE
            + Self::PASS_NEXT_SIZE
            + Self::PASS_PREV_OUTPUT as usize * Self::CAR_OUTPUT_SIZE
    }

    pub fn new(car: Car, nn: NeuralNetwork, walls: Vec<Wall>, rewards: Vec<Reward>) -> Self {
        let total_input_values = Self::get_total_input_neurons();
        Self {
            car,
            penalty: 0.,
            reward: 0.,
            time_passed: 0.,
            input_values: vec![0.; total_input_values],
            next_values: vec![0.; Self::PASS_NEXT_SIZE],
            prev_output: vec![0.; Self::CAR_OUTPUT_SIZE],
            dirs: (0..Self::DIRS_N)
                .map(|i| (i as f32 / (Self::DIRS_N - 1) as f32 - 0.5) * TAU * 2. / 6.)
                .map(|t| rotate_around_origin(pos2(1., 0.), t))
                .collect(),
            distances: vec![0.; total_input_values],
            nn,
            walls,
            reward_path_processor: RewardPathProcessor::new(rewards),
        }
    }

    pub fn reset(&mut self) {
        self.car = Default::default();
        self.penalty = 0.;
        self.reward = 0.;
        self.time_passed = 0.;
        self.reward_path_processor.reset();
    }

    pub fn step(
        &mut self,
        params: &PhysicsParameters,
        observe_distance: &mut impl FnMut(Pos2, Pos2, f32),
        process_user_input: &mut impl FnMut(&mut Car) -> bool,
        drift_observer: &mut impl FnMut(usize, Vec2, f32),
        observe_car_forces: &mut impl FnMut(&Car),
    ) -> bool {
        let mut input_values_iter = self.input_values.iter_mut();

        if Self::PASS_TIME {
            *input_values_iter.next().unwrap() = self.time_passed;
        }

        if Self::PASS_DIRS {
            for dir in &self.dirs {
                let dir_pos = self.car.from_local_coordinates(*dir);
                let origin = self.car.get_center();
                let mut intersection: Option<f32> = None;
                for wall in &self.walls {
                    let temp = wall.intersect_ray(origin, dir_pos);
                    intersection = intersection.any_or_both_with(temp, |a, b| a.min(b));
                }
                if let Some(t) = intersection {
                    observe_distance(origin, dir_pos, t);
                }
                *input_values_iter.next().unwrap() =
                    // intersection.map(|x| x.max(1000.)).unwrap_or(1000.);
                    intersection.map(|x| 20. / (x + 1.).sqrt()).unwrap_or(0.);
            }
        }

        if Self::PASS_INTERNALS {
            for y in &self.car.get_internal_values().to_f32() {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        #[allow(clippy::absurd_extreme_comparisons)]
        if Self::PASS_NEXT_SIZE > 0 {
            for y in &self.next_values {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        if Self::PASS_PREV_OUTPUT {
            for y in &self.prev_output {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        debug_assert!(input_values_iter.next().is_none());

        let values = self.nn.calc(&self.input_values);

        if Self::PASS_PREV_OUTPUT {
            values[..Self::CAR_OUTPUT_SIZE]
                .iter()
                .zip(self.prev_output.iter_mut())
                .for_each(|(x, y)| *y = *x);
        }

        #[allow(clippy::absurd_extreme_comparisons)]
        if Self::PASS_NEXT_SIZE > 0 {
            values[Self::CAR_OUTPUT_SIZE - 1..]
                .iter()
                .zip(self.next_values.iter_mut())
                .for_each(|(x, y)| *y = *x);
        }

        let input = CarInput::from_f32(&values[0..Self::CAR_OUTPUT_SIZE]);
        if !process_user_input(&mut self.car) {
            self.car.process_input(&input, params);
        }

        for i in 0..params.steps_per_time {
            let time = params.time / params.steps_per_time as f32;

            self.car.apply_wheels_force(drift_observer, params);

            for wall in &self.walls {
                if self.car.process_collision(wall, params) {
                    self.penalty += time;
                    // return true;
                }
            }

            if i == 0 {
                observe_car_forces(&self.car);
            }

            self.car.step(time, params);
            self.time_passed += time;
        }

        let reward = self
            .reward_path_processor
            .process_point(self.car.get_center());

        self.reward += reward * 10. / self.time_passed;
        false
    }

    pub fn draw(&mut self, painter: &Painter, to_screen: &RectTransform) {
        self.car.draw_car(painter, to_screen);
        for wall in &self.walls {
            painter.add(Shape::closed_line(
                wall.get_points()
                    .into_iter()
                    .map(|p| to_screen.transform_pos(p))
                    .collect(),
                Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
            ));
        }

        self.reward_path_processor
            .draw(self.car.get_center(), painter, to_screen);
    }
}

#[derive(Debug)]
struct TrackEvaluation {
    name: String,
    penalty: f32,
    reward: f32,
    early_finish_percent: f32,
    distance_percent: f32,
    rewards_acquired_percent: f32,
    all_acquired: bool,
}

impl std::fmt::Display for TrackEvaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} | d {:>5.1}% | a {:>5.1}% | e {:>5.1}% | penalty {:>6.1} | reward {:>6.1} | {}",
            if self.all_acquired { "✅" } else { "❌" },
            self.distance_percent * 100.,
            self.rewards_acquired_percent * 100.,
            self.early_finish_percent * 100.,
            self.penalty,
            self.reward,
            self.name,
        )
    }
}

fn print_evals(evals: &[TrackEvaluation]) {
    for eval in evals {
        println!("{}", eval);
    }
}

fn sum_evals(evals: &[TrackEvaluation]) -> f32 {
    let enable_steps = false;
    let enable_min_dist = false;
    evals
        .iter()
        .map(|x| {
            if x.all_acquired && enable_steps {
                TrackEvaluation {
                    name: Default::default(),
                    penalty: 0.,
                    reward: 1000.,
                    early_finish_percent: 1.,
                    distance_percent: 1.,
                    rewards_acquired_percent: 1.,
                    all_acquired: true,
                }
                .to_f32()
            } else {
                x.to_f32()
            }
        })
        .sum::<f32>()
        / evals.len() as f32
        + if enable_min_dist {
            evals
                .iter()
                .map(|x| x.distance_percent)
                .reduce(|a, b| a.min(b))
                .unwrap_or(0.)
                * 10000.
        } else {
            0.
        }
}

impl TrackEvaluation {
    fn to_f32(&self) -> f32 {
        self.reward * 10.
            + self.early_finish_percent * 1000.
            // + self.distance_percent * 1000.
            // + self.rewards_acquired_percent * 1000.
            - self.penalty * 50.
    }
}

fn eval_nn(nn: NeuralNetwork, params: &PhysicsParameters) -> Vec<TrackEvaluation> {
    let mut result: Vec<TrackEvaluation> = Default::default();
    for Track {
        name,
        walls,
        rewards,
    } in get_all_tracks()
    {
        let mut simulation = CarSimulation::new(Default::default(), nn.clone(), walls, rewards);

        let mut early_finish_percent = 0.;
        let steps_quota = 2000;
        for i in 0..steps_quota {
            if simulation.step(
                params,
                &mut |_, _, _| (),
                &mut |_| false,
                &mut |_, _, _| (),
                &mut |_| (),
            ) {
                break;
            }

            if simulation.reward_path_processor.all_acquired() {
                early_finish_percent = (steps_quota - i) as f32 / steps_quota as f32;
                break;
            }
        }
        result.push(TrackEvaluation {
            name,
            penalty: simulation.penalty,
            reward: simulation.reward,
            early_finish_percent,
            distance_percent: simulation.reward_path_processor.distance_percent(),
            rewards_acquired_percent: simulation.reward_path_processor.rewards_acquired_percent(),
            all_acquired: simulation.reward_path_processor.all_acquired(),
        });
    }
    result
}

fn from_slice_to_nn(sizes: Vec<usize>, pos: &[f32]) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(sizes);
    nn.get_values_mut()
        .iter_mut()
        .zip(pos.iter())
        .for_each(|(x, y)| *x = *y);
    nn
}

fn from_slice_to_nn_f64(sizes: Vec<usize>, pos: &[f64]) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(sizes);
    nn.get_values_mut()
        .iter_mut()
        .zip(pos.iter())
        .for_each(|(x, y)| *x = *y as f32);
    nn
}

fn from_dvector_to_nn(sizes: Vec<usize>, pos: &DVector<f64>) -> NeuralNetwork {
    let mut nn = NeuralNetwork::new(sizes);
    nn.get_values_mut()
        .iter_mut()
        .zip(pos.iter())
        .for_each(|(x, y)| *x = *y as f32);
    nn
}

#[derive(Clone)]
struct NnToEvolve {
    nn: NeuralNetwork,
}

impl Unit for NnToEvolve {
    fn fitness(&self) -> f64 {
        let evals = eval_nn(self.nn.clone(), &PhysicsParameters::default());
        let result = sum_evals(&evals);
        result as f64
    }

    fn breed_with(&self, other: &Self) -> Self {
        let mut rng = thread_rng();
        let mut result = self.clone();
        result
            .nn
            .get_values_mut()
            .iter_mut()
            .zip(other.nn.get_values().iter())
            .for_each(|(a, b)| {
                if rng.gen() {
                    *a = *b;
                }
            });
        for _ in 0..10 {
            result.nn.mutate_float_value(&mut rng);
        }
        result
    }
}

pub fn evolve_by_differential_evolution(nn_sizes: Vec<usize>) {
    let params = PhysicsParameters::default();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();

    let input_done: Vec<(f32, f32)> = include!("nn.data").1.into_iter().map(|x| (x - 0.01, x + 0.01)).collect();
    assert_eq!(input_done.len(), nn_len);
    // let input_new: Vec<(f32, f32)> = vec![(-10., 10.); nn_len];

    let now = Instant::now();
    let mut de = self_adaptive_de(input_done, |pos| {
        let evals = eval_nn(
            from_slice_to_nn(nn_sizes.clone(), pos),
            &PhysicsParameters::default(),
        );
        -sum_evals(&evals)
    });
    for pos in 0..100_000 {
        let value = de.iter().next().unwrap();
        if pos % 20 == 0 && pos != 0 {
            println!("{pos}. {value}, {:?}", now.elapsed() / pos as u32);
        }
        if pos % 300 == 0 && pos != 0 {
            let (cost, vec) = de.best().unwrap();
            println!("cost: {}", cost);
            print_evals(&eval_nn(from_slice_to_nn(nn_sizes.clone(), vec), &params));
        }
        if pos % 1000 == 0 && pos != 0 {
            let (_, vec) = de.best().unwrap();
            println!("(vec!{:?}, vec!{:?})", nn_sizes, vec);
        }
    }
    // show the result
    let (cost, pos) = de.best().unwrap();
    println!("cost: {}", cost);
    print_evals(&eval_nn(from_slice_to_nn(nn_sizes.clone(), pos), &params));
    println!("(vec!{:?}, vec!{:?})", nn_sizes, pos);
}

pub fn evolve_by_genetic_algorithm(nn_sizes: Vec<usize>) {
    let mut rng = thread_rng();
    let population_size = 100;
    let units = (0..population_size)
        .map(|_| NeuralNetwork::generate_random(nn_sizes.clone(), &mut rng))
        .map(|nn| NnToEvolve { nn })
        .collect::<Vec<_>>();
    let mut population = Population::new(units);
    population
        .set_size(population_size)
        .set_breed_factor(0.3)
        .set_survival_factor(0.5);

    let res = population.epochs_parallel(100, 4).finish().remove(0);
    dbg!(&res.nn);
    let evals = eval_nn(res.nn, &PhysicsParameters::default());
    print_evals(&evals);
    let cost = -sum_evals(&evals);
    println!("cost: {}", cost);
}

pub fn evolve_by_cma_es(nn_sizes: Vec<usize>) {
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let mut state = cmaes::options::CMAESOptions::new(vec![3.0; nn_len], 10.0)
        .population_size(50)
        .enable_plot(PlotOptions::new(0, false))
        .enable_printing(1000)
        .max_generations(200)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(
                from_dvector_to_nn(nn_sizes.clone(), x),
                &PhysicsParameters::default(),
            );
            -sum_evals(&evals) as f64
        })
        .unwrap();
    let solution = state.run().overall_best.unwrap().point;
    println!("(vec!{:?}, vec!{:?})", nn_sizes, solution.as_slice());
    let evals = eval_nn(
        from_dvector_to_nn(nn_sizes.clone(), &solution),
        &PhysicsParameters::default(),
    );
    print_evals(&evals);
    state
        .get_plot()
        .unwrap()
        .save_to_file("plot.png", true)
        .unwrap();
}

 #[inline(always)]
fn mod_and_calc_vec<T>(
    x: &mut Vec<f32>,
    f: &mut dyn FnMut(&Vec<f32>) -> T,
    idx: usize,
    y: f32,
) -> T {
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(x);
    x[idx] = xtmp;
    fx1
}

fn forward_diff_vec(
    x: &Vec<f32>,
    f: &mut dyn FnMut(&Vec<f32>) -> f32,
) -> Vec<f32> {
    let step = 0.01;
    let fx = (f)(x);
    let mut xt = x.clone();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec(&mut xt, f, i, step);
            (fx1 - fx) / step
        })
        .collect()
}

#[allow(unused_imports, unused_variables)]
fn evolve_by_bfgs(nn_sizes: Vec<usize>) {
    use argmin::solver::conjugategradient::beta::*;
    use argmin::solver::conjugategradient::*;
    use argmin::solver::neldermead::*;
    use argmin::solver::particleswarm::ParticleSwarm;
    use argmin::solver::simulatedannealing::*;
    use argmin::{
        core::{CostFunction, Error, Executor, Gradient},
        solver::{linesearch::MoreThuenteLineSearch, quasinewton::BFGS},
    };
    use finitediff::FiniteDiff;
    use ndarray::{Array1, Array2};
    use rand::{distributions::Uniform, prelude::*};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct NnStruct {
        nn_sizes: Vec<usize>,

        rng: Arc<Mutex<Xoshiro256PlusPlus>>,
    }

    #[inline(always)]
    fn mod_and_calc_ndarray_f64<T>(
        x: &mut ndarray::Array1<f64>,
        f: &dyn Fn(&ndarray::Array1<f64>) -> T,
        idx: usize,
        y: f64,
    ) -> T {
        let xtmp = x[idx];
        x[idx] = xtmp + y;
        let fx1 = (f)(x);
        x[idx] = xtmp;
        fx1
    }

    fn forward_diff_ndarray_f64(
        x: &ndarray::Array1<f64>,
        f: &dyn Fn(&ndarray::Array1<f64>) -> f64,
    ) -> ndarray::Array1<f64> {
        let step = 0.2;
        let fx = (f)(x);
        let mut xt = x.clone();
        (0..x.len())
            .map(|i| {
                let fx1 = mod_and_calc_ndarray_f64(&mut xt, f, i, step);
                (fx1 - fx) / step
            })
            .collect()
    }

    impl CostFunction for NnStruct {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let evals = eval_nn(
                from_slice_to_nn_f64(self.nn_sizes.clone(), &p.to_vec()),
                &PhysicsParameters::default(),
            );
            Ok(-sum_evals(&evals) as f64)
        }
    }
    impl Gradient for NnStruct {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok(forward_diff_ndarray_f64(p, &|x| {
                let evals = eval_nn(
                    from_slice_to_nn_f64(self.nn_sizes.clone(), &x.to_vec()),
                    &PhysicsParameters::default(),
                );
                -sum_evals(&evals) as f64
            }))
        }
    }

    impl Anneal for NnStruct {
        type Param = Array1<f64>;
        type Output = Array1<f64>;
        type Float = f64;

        /// Anneal a parameter vector
        fn anneal(&self, param: &Array1<f64>, temp: f64) -> Result<Array1<f64>, Error> {
            let mut param_n = param.clone();
            let mut rng = self.rng.lock().unwrap();
            let distr = Uniform::from(0..param.len());
            // Perform modifications to a degree proportional to the current temperature `temp`.
            for _ in 0..(temp.floor() as u64 + 1) {
                // Compute random index of the parameter vector using the supplied random number
                // generator.
                let idx = rng.sample(distr);

                // Compute random number in [0.1, 0.1].
                let val = rng.sample(Uniform::new_inclusive(-0.1, 0.1));

                // modify previous parameter value at random position `idx` by `val`
                param_n[idx] += val;
            }
            Ok(param_n)
        }
    }

    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let cost = NnStruct {
        nn_sizes: nn_sizes.clone(),
        rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
    };
    let mut rng = thread_rng();
    let init_param: Array1<f64> = (0..nn_len).map(|_| rng.gen_range(-10.0..10.0)).collect();
    // let init_param: Array1<f64> = vec![4.688306097853486, 0.6204017367645083, -7.859635664475743, -7.635218409421679, 7.717967786436421, -0.54923420739562, 8.066439164645534, -8.897632538763437, 9.865049574233668, -4.068853301884416, 9.995907498968416, -5.9458871673975295, 9.014559744030446, -1.9768150622806946, -0.43603010816141996, 7.763405146364665, -6.145013892767182, -3.3949860939939436, -0.5952302737544315, -0.03173308202889302, -10.0, 3.5596750513168534, -3.1339574977700995, -1.7263951533717297, -5.803931353512225, -6.606952410815025, 9.751160709892426, -7.052545480460957, -9.608931236527063, -9.971011080222466, -9.089350456184636, -4.675328307219448, -4.765826891536515, 1.9106005956427523, 7.2317476442113655, 5.197172524515189, 0.7044440726356216, -3.0410425227088056, -7.00246850779088, -0.914321933341788, -6.36322586787105, -0.40145501100564773, -6.783476137785582, 8.93626982172199, -0.0024591707864168866, -10.0, -9.910070499266546, -2.164459640722657, 4.1816780767118145, -7.754302079816634, -8.802416740030084, 0.08398342448757656, -9.669314963459694, 1.8973137389881352, -8.470348743142802, -5.627227755070736, 1.7625263817770682, -10.0].into_iter().map(|x| x + rng.gen_range(-0.05..0.05)).collect();

    let min_param: Array1<f64> = (0..nn_len).map(|_| -10.).collect();
    let max_param: Array1<f64> = (0..nn_len).map(|_| 10.).collect();
    let init_hessian: Array2<f64> = Array2::eye(nn_len);
    // let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    // let solver = BFGS::new(linesearch);
    let solver = ParticleSwarm::new((min_param, max_param), 100);
    // let solver = NelderMead::new((0..nn_len+1).map(|_| (0..nn_len).map(|_| rng.gen_range(-10.0..10.0)).collect::<Array1<f64>>()).collect());
    // let solver = NonlinearConjugateGradient::new(linesearch, FletcherReeves::new());
    // let solver = SimulatedAnnealing::new(2000.0).unwrap()
    //     // .with_temp_func(SATempFunc::Boltzmann)
    //     .with_stall_best(1000)
    //     .with_stall_accepted(1000)
    //     .with_reannealing_fixed(300)
    //     .with_reannealing_accepted(300)
    //     .with_reannealing_best(300);
    let res = Executor::new(cost.clone(), solver)
        .configure(|state| {
            state
                // .param(init_param)
                // .inv_hessian(init_hessian)
                .max_iters(15)
                .target_cost(-10000000.)
        })
        .add_observer(
            argmin_observer_slog::SlogLogger::term(),
            argmin::core::observers::ObserverMode::Always,
        )
        .run()
        .unwrap();

    // let input_vec = &res.state.param.clone().unwrap();
    let input_vec = &res.state.best_individual.clone().unwrap().position;
    // let input_vec = nn_sizes.clone(), &res.state.best_param.clone().unwrap().to_vec();

    let time = Instant::now();
    dbg!(cost.gradient(input_vec));
    dbg!(time.elapsed());

    let evals = eval_nn(
        from_slice_to_nn_f64(nn_sizes.clone(), &input_vec.to_vec()),
        &PhysicsParameters::default(),
    );
    print_evals(&evals);

    println!("{res}");
}

pub fn evolution() {
    let nn_sizes = vec![
        CarSimulation::get_total_input_neurons(),
        10,
        CarSimulation::CAR_OUTPUT_SIZE,
    ];
    // evolve_by_differential_evolution(nn_sizes);
    // evolve_by_genetic_algorithm(nn_sizes);
    // evolve_by_cma_es(nn_sizes);
    // evolve_by_bfgs(nn_sizes);

    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let input_done: Vec<f32> = include!("nn.data").1;
    assert_eq!(input_done.len(), nn_len);
    let time = Instant::now();
    let mut count = 0;
    dbg!(forward_diff_vec(&input_done, &mut |x| {
        count += 1;
        let evals = eval_nn(
            from_slice_to_nn(nn_sizes.clone(), x),
            &PhysicsParameters::default(),
        );
        -sum_evals(&evals)
    }));
    dbg!(count);
    dbg!(time.elapsed());
}
