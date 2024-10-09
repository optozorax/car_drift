use differential_evolution::Population;
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
use crate::storage::egui_array_inner;
use cmaes::DVector;
use core::f32::consts::TAU;
use differential_evolution::self_adaptive_de;
use egui::emath::RectTransform;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::Painter;
use egui::Pos2;
use egui::Shape;
use egui::Slider;
use egui::Stroke;
use egui::Ui;
use egui::Vec2;
use rand::thread_rng;
use rand::Rng;
use std::collections::BTreeMap;
use std::time::Instant;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Serialize;
use serde::Deserialize;

// Number from 0 to 1
#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Clamped {
    pub value: f32,
    pub start: f32,
    pub end: f32,
}

impl Clamped {
    pub fn new(value: f32, start: f32, end: f32) -> Self {
        Self { value, start, end }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        ui.add(
            Slider::new(&mut self.value, self.start..=self.end)
                .clamp_to_range(true)
                .min_decimals(0)
                .max_decimals(1),
        );
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct NnParameters {
    pub pass_time: bool,
    pub pass_distance: bool,
    pub pass_dpenalty: bool,
    pub pass_dirs: bool,
    pub pass_internals: bool,
    pub pass_prev_output: bool,
    pub dirs_size: usize,
    pub pass_next_size: usize,
    pub hidden_layers: Vec<usize>,
    pub inv_distance: bool,
    pub view_angle_ratio: f32,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct SimulationParameters {
    pub tracks_enabled: BTreeMap<String, bool>,
    pub tracks_enable_mirror: bool, // mirrors track horizontally

    pub mutate_car_enable: bool,         // mutates position of a car
    pub mutate_car_count: usize,         // hom much times run same track with mutation
    pub mutate_car_angle_range: Clamped, // ratio to PI/4

    pub rewards_enable_early_acquire: bool, // if disabled, then only last reward works
    pub rewards_add_each_acquire: bool,     // to get 1 when reward is acquired
    pub rewards_enable_distance_integral: bool,
    pub rewards_progress_distance: Clamped, // ratio from 0 to 1000
    pub rewards_second_way: bool,
    pub rewards_second_way_penalty: bool,

    pub simulation_stop_penalty: Clamped,      // ratio from 0 to 50
    pub simulation_scale_reward_to_time: bool, // if reward acquired earlier, it will be bigger
    pub simulation_steps_quota: usize,         // reasonable values from 1000 to 3000
    pub simulation_simple_physics: f32,
    pub simulation_enable_random_nn_output: bool,
    pub simulation_random_output_range: f32,

    pub eval_skip_passed_tracks: bool,
    pub eval_add_min_distance: bool,
    pub eval_reward: Clamped,       // ratio from 0 to 5000
    pub eval_early_finish: Clamped, // ratio from 0 to 5000
    pub eval_distance: Clamped,     // ratio from 0 to 5000
    pub eval_acquired: Clamped,     // ratio from 0 to 5000
    pub eval_penalty: Clamped,      // ratio from 0 to 5000

    pub nn: NnParameters,
}

impl NnParameters {
    pub fn grid_ui(&mut self, ui: &mut Ui) {
        ui.label("Pass time:");
        ui.checkbox(&mut self.pass_time, "");
        ui.end_row();

        ui.label("Pass distance:");
        ui.checkbox(&mut self.pass_distance, "");
        ui.end_row();

        ui.label("Pass dpenalty:");
        ui.checkbox(&mut self.pass_dpenalty, "");
        ui.end_row();

        ui.label("Pass dirs:");
        ui.checkbox(&mut self.pass_dirs, "");
        ui.end_row();

        ui.label("Pass internals:");
        ui.checkbox(&mut self.pass_internals, "");
        ui.end_row();

        ui.label("Pass prev output:");
        ui.checkbox(&mut self.pass_prev_output, "");
        ui.end_row();

        ui.label("Dirs size:");
        egui_usize(ui, &mut self.dirs_size);
        ui.end_row();

        ui.label("Pass next size:");
        egui_usize(ui, &mut self.pass_next_size);
        ui.end_row();

        ui.label("Hidden layers:");
        ui.vertical(|ui| {
            egui_array_inner(
                &mut self.hidden_layers,
                ui,
                "layers".into(),
                |elem, ui, _| egui_usize(ui, elem),
                true,
            );
        });
        ui.end_row();

        ui.label("Inv distance:");
        ui.checkbox(&mut self.inv_distance, "");
        ui.end_row();

        ui.label("Total inputs:");
        ui.label(self.get_total_input_neurons().to_string());
        ui.end_row();
    }

    const CAR_INPUT_SIZE: usize = InternalCarValues::SIZE;
    const CAR_OUTPUT_SIZE: usize = CarInput::SIZE;

    pub fn get_total_input_neurons(&self) -> usize {
        self.pass_time as usize
            + self.pass_distance as usize
            + self.pass_dpenalty as usize
            + self.pass_dirs as usize * self.dirs_size
            + self.pass_internals as usize * Self::CAR_INPUT_SIZE
            + self.pass_next_size
            + self.pass_prev_output as usize * Self::CAR_OUTPUT_SIZE
    }

    pub fn get_nn_sizes(&self) -> Vec<usize> {
        std::iter::once(self.get_total_input_neurons())
            .chain(self.hidden_layers.iter().copied())
            .chain(std::iter::once(Self::CAR_OUTPUT_SIZE))
            .collect()
    }

    pub fn get_nn_len(&self) -> usize {
        NeuralNetwork::new(self.get_nn_sizes()).get_values().len()
    }
}

impl SimulationParameters {
    pub fn grid_ui(&mut self, ui: &mut Ui) {
        for (track, enabled) in self.tracks_enabled.iter_mut() {
            ui.label(format!("Track {track}"));
            ui.checkbox(enabled, "");
            ui.end_row();
        }

        ui.separator();
        ui.end_row();

        ui.label("Mirror tracks");
        ui.checkbox(&mut self.tracks_enable_mirror, "");
        ui.end_row();

        ui.separator();
        ui.end_row();

        ui.label("Mutate car");
        ui.checkbox(&mut self.mutate_car_enable, "");
        ui.end_row();

        ui.label("Mutate car count:");
        egui_usize(ui, &mut self.mutate_car_count);
        ui.end_row();

        ui.label("Mutate car angle range:");
        self.mutate_car_angle_range.ui(ui);
        ui.end_row();

        ui.separator();
        ui.end_row();

        ui.label("Enable early acquire:");
        ui.checkbox(&mut self.rewards_enable_early_acquire, "");
        ui.end_row();

        ui.label("Add each acquire:");
        ui.checkbox(&mut self.rewards_add_each_acquire, "");
        ui.end_row();

        ui.label("Enable distance integral:");
        ui.checkbox(&mut self.rewards_enable_distance_integral, "");
        ui.end_row();

        ui.label("Progress distance:");
        self.rewards_progress_distance.ui(ui);
        ui.end_row();

        ui.separator();
        ui.end_row();

        ui.label("Stop penalty:");
        self.simulation_stop_penalty.ui(ui);
        ui.end_row();

        ui.label("Scale reward to time:");
        ui.checkbox(&mut self.simulation_scale_reward_to_time, "");
        ui.end_row();

        ui.label("Steps quota:");
        egui_usize(ui, &mut self.simulation_steps_quota);
        ui.end_row();

        ui.label("Simple physics:");
        egui_0_1(ui, &mut self.simulation_simple_physics);
        ui.end_row();

        ui.separator();
        ui.end_row();

        ui.label("Skip passed tracks:");
        ui.checkbox(&mut self.eval_skip_passed_tracks, "");
        ui.end_row();

        ui.label("Add min distance:");
        ui.checkbox(&mut self.eval_add_min_distance, "");
        ui.end_row();

        ui.label("Eval reward:");
        self.eval_reward.ui(ui);
        ui.end_row();

        ui.label("Early finish:");
        self.eval_early_finish.ui(ui);
        ui.end_row();

        ui.label("Eval distance:");
        self.eval_distance.ui(ui);
        ui.end_row();

        ui.label("Eval acquired:");
        self.eval_acquired.ui(ui);
        ui.end_row();

        ui.label("Eval penalty:");
        self.eval_penalty.ui(ui);
        ui.end_row();

        ui.separator();
        ui.end_row();

        self.nn.grid_ui(ui);
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        egui::Grid::new("simulation params")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
                ui.separator();
                ui.end_row();
            });
    }

    pub fn patch_physics_parameters(&self, mut params: PhysicsParameters) -> PhysicsParameters {
        params.simple_physics_ratio = self.simulation_simple_physics;
        params
    }
}

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

pub fn mutate_car(mut car: Car, params: &SimulationParameters) -> Car {
    let angle_range = params.mutate_car_angle_range.value;
    let angle_speed_range = 0.1;
    let pos_range = 0.5;
    let speed_range = 0.01;
    let mut rng = thread_rng();
    car.change_position(
        rng.gen_range(-angle_range..angle_range),
        vec2(
            rng.gen_range(-pos_range..pos_range),
            rng.gen_range(-pos_range..pos_range),
        ),
        rng.gen_range(-angle_speed_range..angle_speed_range),
        vec2(
            rng.gen_range(-speed_range..speed_range),
            rng.gen_range(-speed_range..speed_range),
        ),
    );
    car
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
        // track_turn_around(),
        track_complex(),
    ]
}

pub struct RewardPathProcessor {
    rewards: Vec<Reward>,
    current_segment: usize,
    prev_distance: f32,
    max_distance: f32,
    max_possible_distance: f32,
}

impl RewardPathProcessor {
    const ENABLE_EARLY_ACQUIRE: bool = true;

    pub fn new(rewards: Vec<Reward>) -> Self {
        let max_possible_distance = pairs(rewards.iter())
            .map(|(a, b)| (b.center - a.center).length())
            .sum();
        Self {
            rewards,
            current_segment: 0,
            prev_distance: 0.,
            max_distance: 0.,
            max_possible_distance,
        }
    }

    fn process_point(&mut self, point: Pos2, params: &SimulationParameters) -> f32 {
        let (rewards, prev, max, current_segment) = (
            &mut self.rewards,
            &mut self.prev_distance,
            &mut self.max_distance,
            &mut self.current_segment,
        );

        #[allow(unused_mut)]
        let mut reward_sum = 0.;

        if params.rewards_enable_early_acquire {
            let mut to_acquire_before: Option<usize> = None;
            for (pos, reward) in rewards.iter_mut().enumerate() {
                if !reward.acquired && reward.process_pos(point) {
                    if params.rewards_add_each_acquire {
                        reward_sum += 1.;
                    }
                    to_acquire_before = Some(pos);
                }
            }
            if let Some(to_acquire_before) = to_acquire_before {
                for reward in rewards.iter_mut().take(to_acquire_before) {
                    if !reward.acquired {
                        if params.rewards_add_each_acquire {
                            reward_sum += 1.;
                        }
                        reward.acquired = true;
                    }
                }
                let mut new_prev = 0.;
                for (a, b) in pairs(rewards.iter().take(to_acquire_before + 1)) {
                    new_prev += (b.center - a.center).length();
                }
                *prev = new_prev;
                *current_segment = to_acquire_before;
                if *max < *prev {
                    *max = *prev;
                }
            }
        } else if rewards.last_mut().unwrap().process_pos(point) {
            reward_sum += 1.;
        }

        if 2 <= rewards.len() && *current_segment + 1 < rewards.len() {
            let a = rewards[*current_segment].center;
            let b = rewards[*current_segment + 1].center;
            let (projection, mut t) = project_to_segment(point, a, b);
            let dist_on_line = (projection - a).length();
            let dist_from_point = (projection - point).length();
            let is_adequate = dist_from_point < params.rewards_progress_distance.value;
            let current_dist = *prev + dist_on_line;
            if current_dist > *max && is_adequate {
                if params.rewards_enable_distance_integral {
                    reward_sum += (current_dist - *max) * 1. / (dist_from_point + 1.0);
                }
                *max = current_dist;
            }

            if *current_segment + 2 < rewards.len() {
                let a2 = rewards[*current_segment + 1].center;
                let b2 = rewards[*current_segment + 2].center;
                let (projection2, _) = project_to_segment(point, a2, b2);
                let dist_from_point2 = (projection2 - point).length();
                let is_adequate2 = dist_from_point2 < params.rewards_progress_distance.value;
                if dist_from_point2 < dist_from_point && is_adequate2 {
                    t = 1.;
                    let dist_on_line2 = (projection - a2).length();
                    let current_dist2 = *prev + (b - a).length() + dist_on_line2;
                    if current_dist2 > *max {
                        if params.rewards_enable_distance_integral {
                            reward_sum += (current_dist2 - *max) * 1. / (dist_from_point2 + 1.0);
                        }
                        *max = current_dist2;
                    }
                }
            }

            if t == 1. && is_adequate {
                *prev += (b - a).length();
                *current_segment += 1;
            }
        }

        reward_sum
    }

    fn reset(&mut self) {
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

    pub fn distance_percent(&self) -> f32 {
        self.max_distance / self.max_possible_distance
    }

    pub fn all_acquired(&self, params: &SimulationParameters) -> bool {
        self.rewards_acquired(params) == self.rewards.len()
    }

    pub fn rewards_acquired(&self, params: &SimulationParameters) -> usize {
        if params.rewards_enable_early_acquire {
            self.rewards.iter().filter(|x| x.acquired).count()
        } else if self.rewards.last().unwrap().acquired {
            self.rewards.len()
        } else {
            0
        }
    }

    pub fn rewards_acquired_percent(&self, params: &SimulationParameters) -> f32 {
        if self.rewards.is_empty() {
            0.
        } else {
            self.rewards_acquired(params) as f32 / self.rewards.len() as f32
        }
    }
}

pub struct NnProcessor {
    params: NnParameters,
    input_values: Vec<f32>,
    next_values: Vec<f32>,
    prev_output: Vec<f32>,
    nn: NeuralNetwork,
}

impl NnProcessor {
    pub fn new(nn: NeuralNetwork, params: NnParameters) -> Self {
        Self {
            input_values: vec![0.; params.get_total_input_neurons()],
            next_values: vec![0.; params.pass_next_size],
            prev_output: vec![0.; NnParameters::CAR_OUTPUT_SIZE],
            params,
            nn,
        }
    }

    pub fn new_from_nn_data(params: NnParameters) -> Self {
        let (sizes, values) = include!("nn.data");
        assert_eq!(params.get_nn_sizes(), sizes);
        let mut result = NeuralNetwork::new(sizes);
        result
            .get_values_mut()
            .iter_mut()
            .zip(values.iter())
            .for_each(|(x, y)| *x = *y);
        Self::new(result, params)
    }

    pub fn process(
        &mut self,
        time_passed: f32,
        distance_percent: f32,
        dpenalty: f32,
        dirs: &[Option<f32>],
        internals: &InternalCarValues,
        params_sim: &SimulationParameters,
    ) -> CarInput {
        let mut input_values_iter = self.input_values.iter_mut();

        if self.params.pass_time {
            *input_values_iter.next().unwrap() = time_passed;
        }

        if self.params.pass_distance {
            *input_values_iter.next().unwrap() = distance_percent;
        }

        if self.params.pass_dpenalty {
            *input_values_iter.next().unwrap() = dpenalty;
        }

        if self.params.pass_dirs {
            for intersection in dirs {
                *input_values_iter.next().unwrap() = if self.params.inv_distance {
                    intersection.map(|x| 20. / (x + 1.).sqrt()).unwrap_or(0.)
                } else {
                    intersection.map(|x| x.max(1000.)).unwrap_or(1000.)
                };
            }
        }

        if self.params.pass_internals {
            for y in &internals.to_f32() {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        if self.params.pass_next_size > 0 {
            for y in &self.next_values {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        if self.params.pass_prev_output {
            for y in &self.prev_output {
                *input_values_iter.next().unwrap() = *y;
            }
        }

        debug_assert!(input_values_iter.next().is_none());

        let values = self.nn.calc(&self.input_values);

        if self.params.pass_prev_output {
            values[..NnParameters::CAR_OUTPUT_SIZE]
                .iter()
                .zip(self.prev_output.iter_mut())
                .for_each(|(x, y)| *y = *x);
        }

        if self.params.pass_next_size > 0 {
            values[NnParameters::CAR_OUTPUT_SIZE - 1..]
                .iter()
                .zip(self.next_values.iter_mut())
                .for_each(|(x, y)| *y = *x);
        }

        if params_sim.simulation_enable_random_nn_output {
            let mut values = values.iter().copied().collect::<Vec<_>>();
            let mut rng = thread_rng();
            for i in 0..NnParameters::CAR_OUTPUT_SIZE {
                values[i] += rng.gen_range(-params_sim.simulation_random_output_range..params_sim.simulation_random_output_range);
            }
        }

        CarInput::from_f32(&values[0..NnParameters::CAR_OUTPUT_SIZE])
    }
}

pub struct CarSimulation {
    pub car: Car,
    pub penalty: f32,
    prev_penalty: f32,
    pub reward: f32,
    pub time_passed: f32,
    pub dirs: Vec<Pos2>,
    dirs_values: Vec<Option<f32>>,
    pub walls: Vec<Wall>,
    pub reward_path_processor: RewardPathProcessor,
}

impl CarSimulation {
    pub fn new(
        car: Car,
        walls: Vec<Wall>,
        rewards: Vec<Reward>,
        params: &SimulationParameters,
    ) -> Self {
        Self {
            car,
            penalty: 0.,
            prev_penalty: 0.,
            reward: 0.,
            time_passed: 0.,
            dirs: (0..params.nn.dirs_size)
                .map(|i| (i as f32 / (params.nn.dirs_size - 1) as f32 - 0.5) * TAU * params.nn.view_angle_ratio)
                .map(|t| rotate_around_origin(pos2(1., 0.), t))
                .collect(),
            dirs_values: (0..params.nn.dirs_size).map(|_| None).collect(),
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
        params_phys: &PhysicsParameters,
        params_sim: &SimulationParameters,
        observe_distance: &mut impl FnMut(Pos2, Pos2, f32),
        get_input: &mut impl FnMut(f32, f32, f32, &[Option<f32>], &InternalCarValues) -> CarInput,
        drift_observer: &mut impl FnMut(usize, Vec2, f32),
        observe_car_forces: &mut impl FnMut(&Car),
    ) -> bool {
        for (dir, value) in self.dirs.iter().zip(self.dirs_values.iter_mut()) {
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
            *value = intersection;
        }

        let input = get_input(
            self.time_passed,
            self.reward_path_processor.distance_percent(),
            self.penalty - self.prev_penalty,
            &self.dirs_values,
            &self.car.get_internal_values(),
        );
        self.car.process_input(&input, params_phys);
        self.prev_penalty = self.penalty;

        for i in 0..params_phys.steps_per_time {
            let time = params_phys.time / params_phys.steps_per_time as f32;

            self.car.apply_wheels_force(drift_observer, params_phys);

            for wall in &self.walls {
                if self.car.process_collision(wall, params_phys) {
                    self.penalty += time;
                }
                if self.penalty > params_sim.simulation_stop_penalty.value {
                    return true;
                }
            }

            if i == 0 {
                observe_car_forces(&self.car);
            }

            self.car.step(time, params_phys);
            self.time_passed += time;
        }

        let reward = self
            .reward_path_processor
            .process_point(self.car.get_center(), params_sim);

        self.reward += if params_sim.simulation_scale_reward_to_time {
            reward * 10. / self.time_passed
        } else {
            reward
        };

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

#[derive(Debug, Serialize, Deserialize)]
pub struct TrackEvaluation {
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

pub fn print_evals(evals: &[TrackEvaluation]) {
    for eval in evals {
        println!("{}", eval);
    }
}

pub fn sum_evals(evals: &[TrackEvaluation], params: &SimulationParameters) -> f32 {
    if params.rewards_second_way {
        let len = evals.len() as f32;
        let tracks_sum = evals.iter().map(|x| x.all_acquired as usize as f32).sum::<f32>();
        let penalty_level_1 = evals.iter().map(|x| (x.penalty < 15.) as usize as f32).sum::<f32>();
        let penalty_level_2 = evals.iter().map(|x| (x.penalty < 5.) as usize as f32).sum::<f32>();
        let penalty_level_3 = evals.iter().map(|x| (x.penalty < 2.) as usize as f32).sum::<f32>();
        let penalty_level_4 = evals.iter().map(|x| (x.penalty == 0.) as usize as f32).sum::<f32>();
        let penalty_levels_len = 4. * len;
        let penalty_levels = (penalty_level_1 + penalty_level_2 + penalty_level_3 + penalty_level_4) / penalty_levels_len;
        let smooth_metrics = evals.iter().map(|x| x.to_f32_second_way()).sum::<f32>();
        if params.rewards_second_way_penalty {
            tracks_sum + penalty_levels + smooth_metrics / penalty_levels_len
        } else {
            tracks_sum + smooth_metrics / len
        }
    } else {
        evals
            .iter()
            .map(|x| {
                if x.all_acquired && params.eval_skip_passed_tracks {
                    TrackEvaluation {
                        name: Default::default(),
                        penalty: 0.,
                        reward: 1000.,
                        early_finish_percent: 1.,
                        distance_percent: 1.,
                        rewards_acquired_percent: 1.,
                        all_acquired: true,
                    }
                    .to_f32(params)
                } else {
                    x.to_f32(params)
                }
            })
            .sum::<f32>()
            / evals.len() as f32
            + if params.eval_add_min_distance {
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
}

impl TrackEvaluation {
    fn to_f32(&self, params: &SimulationParameters) -> f32 {
        0. + self.reward * params.eval_reward.value
            + self.early_finish_percent * params.eval_early_finish.value
            + self.distance_percent * params.eval_distance.value
            + self.rewards_acquired_percent * params.eval_acquired.value
            - self.penalty * params.eval_penalty.value
    }

    // from 0 to 1
    fn to_f32_second_way(&self) -> f32 {
        (1. / (1. + self.penalty) + (1. - 1. / (1. + self.reward)) + self.early_finish_percent + self.distance_percent + self.rewards_acquired_percent) / 5.
    }
}

pub fn eval_nn(
    nn: NeuralNetwork,
    params_phys: &PhysicsParameters,
    params_sim: &SimulationParameters,
) -> Vec<TrackEvaluation> {
    let params_phys = params_sim.patch_physics_parameters(params_phys.clone());

    let mut result: Vec<TrackEvaluation> = Default::default();
    let tracks: Vec<Track> = get_all_tracks()
        .into_iter()
        .filter(|x| {
            params_sim
                .tracks_enabled
                .get(&x.name)
                .copied()
                .unwrap_or_default()
        })
        .flat_map(|x| {
            if params_sim.tracks_enable_mirror {
                vec![x.clone(), mirror_horizontally(x)]
            } else {
                vec![x]
            }
        })
        .collect();
    for Track {
        name,
        walls,
        rewards,
    } in tracks
    {
        let max = if params_sim.mutate_car_enable {
            params_sim.mutate_car_count
        } else {
            1
        };
        for i in 0..max {
            let car = {
                let result: Car = Default::default();
                if i != 0 && params_sim.mutate_car_enable {
                    mutate_car(result, params_sim)
                } else {
                    result
                }
            };
            let mut nn_processor = NnProcessor::new(nn.clone(), params_sim.nn.clone());
            let mut simulation =
                CarSimulation::new(car, walls.clone(), rewards.clone(), params_sim);

            let mut early_finish_percent = 0.;
            let steps_quota = params_sim.simulation_steps_quota;
            for i in 0..steps_quota {
                if simulation.step(
                    &params_phys,
                    params_sim,
                    &mut |_, _, _| (),
                    &mut |time, dist, dpenalty, dirs, internals| {
                        nn_processor.process(time, dist, dpenalty, dirs, internals, params_sim)
                    },
                    &mut |_, _, _| (),
                    &mut |_| (),
                ) {
                    break;
                }

                if simulation.reward_path_processor.all_acquired(params_sim) {
                    early_finish_percent = (steps_quota - i) as f32 / steps_quota as f32;
                    break;
                }
            }
            result.push(TrackEvaluation {
                name: name.clone()
                    + &if i != 0 {
                        format!(":{i}")
                    } else {
                        Default::default()
                    },
                penalty: simulation.penalty,
                reward: simulation.reward,
                early_finish_percent,
                distance_percent: simulation.reward_path_processor.distance_percent(),
                rewards_acquired_percent: simulation
                    .reward_path_processor
                    .rewards_acquired_percent(params_sim),
                all_acquired: simulation.reward_path_processor.all_acquired(params_sim),
            });
        }
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

pub fn evolve_by_differential_evolution(params_sim: &SimulationParameters) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let params_phys = params_sim.patch_physics_parameters(PhysicsParameters::default());
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();

    let input_done: Vec<(f32, f32)> = if RUN_FROM_PREV_NN {
        include!("nn.data")
            .1
            .into_iter()
            .map(|x| (x - 0.01, x + 0.01))
            .collect()
    } else {
        vec![(-10., 10.); nn_len]
    };

    assert_eq!(input_done.len(), nn_len);

    let now = Instant::now();
    let mut de = self_adaptive_de(input_done, |pos| {
        let evals = eval_nn(
            from_slice_to_nn(nn_sizes.clone(), pos),
            &params_phys,
            params_sim,
        );
        -sum_evals(&evals, params_sim)
    });
    for pos in 0..100_000 {
        let value = de.iter().next().unwrap();
        if pos % 20 == 0 && pos != 0 {
            println!("{pos}. {value}, {:?}", now.elapsed() / pos as u32);
        }
        if pos % 300 == 0 && pos != 0 {
            let (cost, vec) = de.best().unwrap();
            println!("cost: {}", cost);
            print_evals(&eval_nn(
                from_slice_to_nn(nn_sizes.clone(), vec),
                &params_phys,
                params_sim,
            ));
        }
        if pos % 1000 == 0 && pos != 0 {
            let (_, vec) = de.best().unwrap();
            println!("(vec!{:?}, vec!{:?})", nn_sizes, vec);
        }
    }
    // show the result
    let (cost, pos) = de.best().unwrap();
    println!("cost: {}", cost);
    print_evals(&eval_nn(
        from_slice_to_nn(nn_sizes.clone(), pos),
        &params_phys,
        params_sim,
    ));
    println!("(vec!{:?}, vec!{:?})", nn_sizes, pos);
}

pub fn evolve_by_differential_evolution_custom(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    population_size: usize,
    generations_count: usize,
) -> Vec<EvolveOutputEachStep> {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let input_done: Vec<(f32, f32)> = vec![(-10., 10.); nn_len];

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut de = self_adaptive_de(input_done, |pos| {
        let evals = eval_nn(
            from_slice_to_nn(nn_sizes.clone(), pos),
            &params_phys,
            params_sim,
        );
        -sum_evals(&evals, params_sim)
    });

    let true_params_sim = SimulationParameters::true_metric(params_sim.simulation_simple_physics);

    for pos in 0..generations_count {
        let now = Instant::now();
        for _ in 0..population_size {
            let _ = de.iter().next().unwrap();
        }
        let (value, point) = de.best().unwrap();

        let nn = from_slice_to_nn(nn_sizes.clone(), point);
        let evals = eval_nn(
            nn.clone(),
            params_phys,
            &params_sim,
        );
        let true_evals = eval_nn(
            nn.clone(),
            params_phys,
            &true_params_sim,
        );

        result.push(EvolveOutputEachStep {
            nn: nn.get_values().iter().copied().collect(),
            evals_cost: sum_evals(&evals, params_sim),
            true_evals_cost: sum_evals(&true_evals, &true_params_sim),
            evals,
            true_evals,
        });
    }

    result
}

pub fn evolve_by_cma_es(params_sim: &SimulationParameters) {
    let mut params_sim = params_sim.clone();
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let mut params_phys = params_sim.patch_physics_parameters(PhysicsParameters::default());

    let mut input_done: Vec<f64> = if RUN_FROM_PREV_NN {
        include!("nn.data").1
    } else {
        vec![1.0; nn_len]
    };

    assert_eq!(input_done.len(), nn_len);

    // while params_sim.simulation_simple_physics >= 0. {
        let mut state = cmaes::options::CMAESOptions::new(input_done, 10.0)
            .population_size(100)
            .build(|x: &DVector<f64>| -> f64 {
                let evals = eval_nn(
                    from_dvector_to_nn(nn_sizes.clone(), x),
                    &params_phys,
                    &params_sim,
                );
                -sum_evals(&evals, &params_sim) as f64
            })
            .unwrap();
        let now = Instant::now();
        for pos in 0..500 {
            // let _ = state.next();
            let _ = state.next_parallel();
            let cmaes::Individual { point, value } = state.overall_best_individual().unwrap();
            if pos == 0 {
                print_evals(&eval_nn(
                    from_dvector_to_nn(nn_sizes.clone(), point),
                    &params_phys,
                    &params_sim,
                ));
            }
            println!("{pos}. {value}, {:?}", now.elapsed() / (pos + 1) as u32);
            if pos % 10 == 0 && pos != 0 {
                print_evals(&eval_nn(
                    from_dvector_to_nn(nn_sizes.clone(), point),
                    &params_phys,
                    &params_sim,
                ));
            }
            if pos % 10 == 0 && pos != 0 {
                println!("(vec!{:?}, vec!{:?})", nn_sizes, from_dvector_to_nn(nn_sizes.clone(), point).get_values());
            }
        }
        let solution = state.overall_best_individual().unwrap().point.clone();

        let params_phys_clone = params_phys.clone();
        params_phys = params_sim.patch_physics_parameters(params_phys_clone);
        input_done = solution.iter().copied().collect();
        println!("(vec!{:?}, vec!{:?})", nn_sizes, solution.as_slice());
        let evals = eval_nn(
            from_dvector_to_nn(nn_sizes.clone(), &solution),
            &params_phys,
            &params_sim,
        );
        print_evals(&evals);

        params_phys.acceleration_ratio += 0.01;
        // params_sim.simulation_simple_physics -= 0.02;
        println!(
            "new value of ACCELERATION RATIO: {}",
            params_phys.acceleration_ratio
        );
        println!(
            "new value of SIMPLE PHYSICS: {}",
            params_sim.simulation_simple_physics
        );
        println!("------------------------------------");
        println!("------------------------------------");
        println!("------------------------------------");
    // }
}

#[derive(Serialize, Deserialize)]
struct EvolveOutputEachStep {
    nn: Vec<f32>,
    evals: Vec<TrackEvaluation>,
    evals_cost: f32,
    true_evals: Vec<TrackEvaluation>,
    true_evals_cost: f32,
}

pub fn evolve_by_cma_es_custom(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    nn_input: &[f32],
    population_size: usize,
    generations_count: usize,
) -> Vec<EvolveOutputEachStep> {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let input_done: Vec<f64> = nn_input.iter().map(|x| *x as f64).collect();
    assert_eq!(input_done.len(), nn_len);

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut state = cmaes::options::CMAESOptions::new(input_done, 10.0)
        .population_size(population_size)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(
                from_dvector_to_nn(nn_sizes.clone(), x),
                params_phys,
                params_sim,
            );
            -sum_evals(&evals, params_sim) as f64
        })
        .unwrap();

    let true_params_sim = SimulationParameters::true_metric(params_sim.simulation_simple_physics);

    for pos in 0..generations_count {
        let now = Instant::now();
        // let _ = state.next();
        let _ = state.next_parallel();
        let cmaes::Individual { point, value } = state.overall_best_individual().unwrap();

        let nn = from_dvector_to_nn(nn_sizes.clone(), point);
        let evals = eval_nn(
            nn.clone(),
            params_phys,
            &params_sim,
        );
        let true_evals = eval_nn(
            nn.clone(),
            params_phys,
            &true_params_sim,
        );

        result.push(EvolveOutputEachStep {
            nn: nn.get_values().iter().copied().collect(),
            evals_cost: sum_evals(&evals, params_sim),
            true_evals_cost: sum_evals(&true_evals, &true_params_sim),
            evals,
            true_evals,
        });
    }

    result
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

fn forward_diff_vec(x: &Vec<f32>, f: &mut dyn FnMut(&Vec<f32>) -> f32) -> Vec<f32> {
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
fn evolve_by_bfgs(params_sim: &SimulationParameters) {
    use argmin::solver::conjugategradient::beta::*;
    use argmin::solver::conjugategradient::*;
    use argmin::solver::gradientdescent::SteepestDescent;
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
        params_sim: SimulationParameters,
        params_phys: PhysicsParameters,

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
                from_slice_to_nn_f64(self.params_sim.nn.get_nn_sizes(), &p.to_vec()),
                &self.params_phys,
                &self.params_sim,
            );
            Ok(-sum_evals(&evals, &self.params_sim) as f64)
        }
    }
    impl Gradient for NnStruct {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok(forward_diff_ndarray_f64(p, &|x| {
                let evals = eval_nn(
                    from_slice_to_nn_f64(self.params_sim.nn.get_nn_sizes(), &x.to_vec()),
                    &self.params_phys,
                    &self.params_sim,
                );
                -sum_evals(&evals, &self.params_sim) as f64
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

    let params_phys = params_sim.patch_physics_parameters(PhysicsParameters::default());

    let nn_len = NeuralNetwork::new(params_sim.nn.get_nn_sizes())
        .get_values()
        .len();
    let cost = NnStruct {
        params_sim: params_sim.clone(),
        params_phys: params_phys.clone(),
        rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
    };
    let mut rng = thread_rng();

    let init_param: Array1<f64> = if RUN_FROM_PREV_NN {
        include!("nn.data").1.into_iter().collect()
    } else {
        (0..nn_len).map(|_| rng.gen_range(-10.0..10.0)).collect()
    };
    assert_eq!(nn_len, init_param.len());

    let min_param: Array1<f64> = (0..nn_len).map(|_| -10.).collect();
    let max_param: Array1<f64> = (0..nn_len).map(|_| 10.).collect();
    let init_hessian: Array2<f64> = Array2::eye(nn_len);
    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    let solver = BFGS::new(linesearch);
    // let solver = SteepestDescent::new(linesearch);
    // let solver = ParticleSwarm::new((min_param, max_param), 100);
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
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(15)
                .target_cost(-10000000.)
        })
        .add_observer(
            argmin_observer_slog::SlogLogger::term(),
            argmin::core::observers::ObserverMode::Always,
        )
        .run()
        .unwrap();

    let input_vec = &res.state.param.clone().unwrap();
    // let input_vec = &res.state.best_individual.clone().unwrap().position;
    // let input_vec = nn_sizes.clone(), &res.state.best_param.clone().unwrap().to_vec();

    let evals = eval_nn(
        from_slice_to_nn_f64(params_sim.nn.get_nn_sizes(), &input_vec.to_vec()),
        &params_phys,
        params_sim,
    );
    print_evals(&evals);

    println!("{res}");
}

#[allow(unused_imports, unused_variables)]
pub fn evolve_by_particle_swarm_custom(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    population_size: usize,
    generations_count: usize,
) -> Vec<EvolveOutputEachStep> {
    use argmin::solver::conjugategradient::beta::*;
    use argmin::solver::conjugategradient::*;
    use argmin::solver::gradientdescent::SteepestDescent;
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
    use argmin::core::Solver;
    use argmin::core::Problem;
    use argmin::core::PopulationState;
    use argmin::core::State;

    #[derive(Clone)]
    struct NnStruct {
        params_sim: SimulationParameters,
        params_phys: PhysicsParameters,
    }

    impl CostFunction for NnStruct {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let evals = eval_nn(
                from_slice_to_nn_f64(self.params_sim.nn.get_nn_sizes(), &p.to_vec()),
                &self.params_phys,
                &self.params_sim,
            );
            Ok(-sum_evals(&evals, &self.params_sim) as f64)
        }
    }

    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();
    let cost = NnStruct {
        params_sim: params_sim.clone(),
        params_phys: params_phys.clone(),
    };

    let min_param: Array1<f64> = (0..nn_len).map(|_| -10.).collect();
    let max_param: Array1<f64> = (0..nn_len).map(|_| 10.).collect();
    let mut solver = ParticleSwarm::new((min_param, max_param), population_size);
    let mut problem = Problem::new(cost);
    let mut state = PopulationState::new();
    state = solver.init(&mut problem, state).unwrap().0;

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let true_params_sim = SimulationParameters::true_metric(params_sim.simulation_simple_physics);

    for pos in 0..generations_count {
        let now = Instant::now();
        state = solver.next_iter(&mut problem, state).unwrap().0;
        let (point, value) = (&state.individual.as_ref().unwrap().position, state.cost);
        
        let nn = from_slice_to_nn_f64(nn_sizes.clone(), point.as_slice().unwrap());
        let evals = eval_nn(
            nn.clone(),
            params_phys,
            &params_sim,
        );
        let true_evals = eval_nn(
            nn.clone(),
            params_phys,
            &true_params_sim,
        );

        result.push(EvolveOutputEachStep {
            nn: nn.get_values().iter().copied().collect(),
            evals_cost: sum_evals(&evals, params_sim),
            true_evals_cost: sum_evals(&true_evals, &true_params_sim),
            evals,
            true_evals,
        });
    }
    result
}

fn calc_gradient(params_sim: &SimulationParameters) {
    let nn_len = NeuralNetwork::new(params_sim.nn.get_nn_sizes())
        .get_values()
        .len();
    let input_done: Vec<f32> = include!("nn.data").1;
    assert_eq!(input_done.len(), nn_len);
    let time = Instant::now();
    let mut count = 0;
    dbg!(forward_diff_vec(&input_done, &mut |x| {
        count += 1;
        let evals = eval_nn(
            from_slice_to_nn(params_sim.nn.get_nn_sizes(), x),
            &params_sim.patch_physics_parameters(PhysicsParameters::default()),
            params_sim,
        );
        -sum_evals(&evals, params_sim)
    }));
    dbg!(count);
    dbg!(time.elapsed());
}

fn mutate_nn() {
    let nn = from_slice_to_nn(include!("nn.data").0, &include!("nn.data").1);
    let mut nn_uno = nn.to_unoptimized();
    nn_uno.add_hidden_layer(1);
    let nn = nn_uno.to_optimized();
    println!("(vec!{:?}, vec!{:?})", nn.get_sizes(), nn.get_values());
}

impl Default for NnParameters {
    fn default() -> Self {
        Self {
            pass_time: false,
            pass_distance: false,
            pass_dpenalty: true,
            pass_dirs: true,
            pass_internals: false,
            pass_prev_output: false,
            // dirs_size: 5,
            dirs_size: 21,
            pass_next_size: 0,
            // pass_next_size: 3,
            // hidden_layers: vec![6, 6],
            // hidden_layers: vec![4, 4, 4, 4],
            hidden_layers: vec![6],
            inv_distance: true,
            view_angle_ratio: 2. / 6.,
        }
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        let all_tracks: BTreeMap<String, bool> = get_all_tracks()
            .into_iter()
            .map(|x| (x.name, false))
            .collect();
        Self {
            tracks_enabled: all_tracks
                .into_iter()
                .chain(
                    vec![
                        "straight",
                        "straight_45",
                        "turn_right_smooth",
                        "smooth_left_and_right",
                        "turn_left_90",
                        "turn_left_180",
                        "complex",
                    ]
                    .into_iter()
                    .map(|x| (x.to_owned(), true)),
                )
                .collect(),
            tracks_enable_mirror: true,

            mutate_car_enable: false,
            mutate_car_count: 3,
            mutate_car_angle_range: Clamped::new(0.7, 0., TAU / 8.),

            rewards_enable_early_acquire: true,
            rewards_add_each_acquire: false,
            rewards_enable_distance_integral: false,
            rewards_progress_distance: Clamped::new(200., 0., 1000.),
            rewards_second_way: false,
            rewards_second_way_penalty: false,

            simulation_stop_penalty: Clamped::new(20., 0., 50.),
            simulation_scale_reward_to_time: false,
            simulation_steps_quota: 3500,
            simulation_simple_physics: 0.0,
            simulation_enable_random_nn_output: false,
            simulation_random_output_range: 0.1,

            eval_skip_passed_tracks: false,
            eval_add_min_distance: false,
            eval_reward: Clamped::new(10., 0., 5000.),
            eval_early_finish: Clamped::new(10000., 0., 10000.),
            eval_distance: Clamped::new(1000., 0., 5000.),
            eval_acquired: Clamped::new(1000., 0., 5000.),
            eval_penalty: Clamped::new(20., 0., 5000.),

            nn: Default::default(),
        }
    }
}

impl SimulationParameters {
    pub fn true_metric(simple_physics: f32) -> Self {
        let mut result = Self::default();
        result.enable_all_tracks();
        result.disable_track("straight_45"); // todo: удалить это и снова проверить
        result.rewards_second_way = true;
        result.mutate_car_enable = false;
        result.simulation_stop_penalty.value = 100.;
        result.simulation_steps_quota = 3000;
        result.simulation_simple_physics = simple_physics;
        result.eval_reward.value = 1.;
        result.eval_acquired.value = 1.;
        result.eval_penalty.value = 1.;

        // result.nn.hidden_layers = vec![6, 6];

        result
    }

    pub fn disable_all_tracks(&mut self) {
        for enabled in self.tracks_enabled.values_mut() {
            *enabled = false;
        }
    }
    pub fn enable_all_tracks(&mut self) {
        for enabled in self.tracks_enabled.values_mut() {
            *enabled = true;
        }
    }
    pub fn enable_track(&mut self, track: &str) {
        *self.tracks_enabled.get_mut(track).unwrap() = true;
    }
    pub fn disable_track(&mut self, track: &str) {
        *self.tracks_enabled.get_mut(track).unwrap() = false;
    }
}

pub const RUN_EVOLUTION: bool = true;

pub const RUN_FROM_PREV_NN: bool = false;

const RUNS_COUNT: usize = 30;
const POPULATION_SIZE: usize = 30;
const GENERATIONS_COUNT: usize = 100;

// const RUNS_COUNT: usize = 3;
// const POPULATION_SIZE: usize = 3;
// const GENERATIONS_COUNT: usize = 3;

fn test_params_sim(params_sim: &SimulationParameters, params_phys: &PhysicsParameters, name: &str) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();

    let now = Instant::now();
    let result: Vec<Vec<EvolveOutputEachStep>> = (0..RUNS_COUNT).into_par_iter().map(|_| {
        let mut rng = thread_rng();
        let mut input = (0..nn_len).map(|_| rng.gen_range(-10.0..10.0)).collect::<Vec<f32>>();
        evolve_by_cma_es_custom(&params_sim, &params_phys, &input, POPULATION_SIZE, GENERATIONS_COUNT)
    }).collect();

    use std::io::Write;
    let filename = format!("graphs/{name}.json");
    let mut file = std::fs::File::create(&filename).unwrap();
    let json = serde_json::to_string(&result).unwrap();
    write!(file, "{}", json).unwrap();

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

fn test_params_sim_differential_evolution(params_sim: &SimulationParameters, params_phys: &PhysicsParameters, name: &str) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();

    let now = Instant::now();
    let result: Vec<Vec<EvolveOutputEachStep>> = (0..RUNS_COUNT).into_par_iter().map(|_| {
        let mut rng = thread_rng();
        let mut input = (0..nn_len).map(|_| rng.gen_range(-10.0..10.0)).collect::<Vec<f32>>();
        evolve_by_differential_evolution_custom(&params_sim, &params_phys, POPULATION_SIZE, GENERATIONS_COUNT)
    }).collect();

    use std::io::Write;
    let filename = format!("graphs/{name}.json");
    let mut file = std::fs::File::create(&filename).unwrap();
    let json = serde_json::to_string(&result).unwrap();
    write!(file, "{}", json).unwrap();

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

fn test_params_sim_particle_swarm(params_sim: &SimulationParameters, params_phys: &PhysicsParameters, name: &str) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = NeuralNetwork::new(nn_sizes.clone()).get_values().len();

    let now = Instant::now();
    let result: Vec<Vec<EvolveOutputEachStep>> = (0..RUNS_COUNT).into_par_iter().map(|_| {
        evolve_by_particle_swarm_custom(&params_sim, &params_phys, POPULATION_SIZE, GENERATIONS_COUNT)
    }).collect();

    use std::io::Write;
    let filename = format!("graphs/{name}.json");
    let mut file = std::fs::File::create(&filename).unwrap();
    let json = serde_json::to_string(&result).unwrap();
    write!(file, "{}", json).unwrap();

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

pub fn evolution() {
    let mut params_sim = SimulationParameters::default();
    let mut params_phys = PhysicsParameters::default();

    params_sim.enable_all_tracks();
    params_sim.disable_track("straight_45");
    params_sim.simulation_enable_random_nn_output = false;
    params_sim.eval_penalty.value = 200.;
    params_sim.rewards_add_each_acquire = true;
    params_sim.rewards_enable_distance_integral = true;
    params_sim.mutate_car_enable = false;
    params_sim.eval_add_min_distance = false;
    params_sim.simulation_stop_penalty.value = 100.;
    params_sim.simulation_simple_physics = 1.;

    let mut params_sim_copy = params_sim.clone();
    
    // test_params_sim(&params_sim, &params_phys, "default");

    //------------------------------------------------------------------------

    // test_params_sim_differential_evolution(&params_sim, &params_phys, "differential_evolution");
    // test_params_sim_particle_swarm(&params_sim, &params_phys, "particle_swarm");

    //------------------------------------------------------------------------

    /*
    params_sim.rewards_second_way = true;
    test_params_sim(&params_sim, &params_phys, "second_way");
    params_sim = params_sim_copy.clone();

    params_sim.rewards_add_each_acquire = false;
    test_params_sim(&params_sim, &params_phys, "no_add_each_acquire");
    params_sim = params_sim_copy.clone();

    params_sim.rewards_enable_distance_integral = false;
    test_params_sim(&params_sim, &params_phys, "no_distance_integral");
    params_sim = params_sim_copy.clone();

    params_sim.mutate_car_enable = true;
    test_params_sim(&params_sim, &params_phys, "mutate_car_enabled");
    params_sim = params_sim_copy.clone();

    params_sim.eval_add_min_distance = true;
    test_params_sim(&params_sim, &params_phys, "with_min_distance");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_scale_reward_to_time = true;
    test_params_sim(&params_sim, &params_phys, "scale_reward_to_time");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_simple_physics = 1.0;
    test_params_sim(&params_sim, &params_phys, "simple_physics_1.0");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_simple_physics = 0.5;
    test_params_sim(&params_sim, &params_phys, "simple_physics_0.5");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_simple_physics = 0.0;
    test_params_sim(&params_sim, &params_phys, "simple_physics_0.0");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.disable_track("complex");
    test_params_sim(&params_sim, &params_phys, "no_tr_complex");
    params_sim = params_sim_copy.clone();

    params_sim.enable_track("straight_45");
    test_params_sim(&params_sim, &params_phys, "with_tr_45");
    params_sim = params_sim_copy.clone();

    params_sim.disable_track("turn_left_90");
    params_sim.disable_track("turn_left_180");
    test_params_sim(&params_sim, &params_phys, "no_tr_turns");
    params_sim = params_sim_copy.clone();

    params_sim.disable_all_tracks();
    params_sim.enable_track("complex");
    test_params_sim(&params_sim, &params_phys, "only_complex_track");
    params_sim = params_sim_copy.clone();
    
    params_sim.disable_track("turn_left_90");
    params_sim.disable_track("turn_left_180");
    params_sim.disable_track("complex");
    params_sim.disable_track("straight_45");
    test_params_sim(&params_sim, &params_phys, "only_tr_simple");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_enable_random_nn_output = false;
    test_params_sim(&params_sim, &params_phys, "random_output_no");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.01;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.01");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.05;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.05");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.1;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.1");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.2;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.2");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.4;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.4");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.eval_penalty.value = 0.;
    test_params_sim(&params_sim, &params_phys, "penalty_0");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 10.;
    test_params_sim(&params_sim, &params_phys, "penalty_10");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 50.;
    test_params_sim(&params_sim, &params_phys, "penalty_50");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 50.;
    test_params_sim(&params_sim, &params_phys, "penalty_100");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 200.;
    test_params_sim(&params_sim, &params_phys, "penalty_200");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 500.;
    test_params_sim(&params_sim, &params_phys, "penalty_500");
    params_sim = params_sim_copy.clone();

    params_sim.eval_penalty.value = 1000.;
    test_params_sim(&params_sim, &params_phys, "penalty_1000");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_stop_penalty.value = 0.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_0");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 1.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_1");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 5.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_5");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 10.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_10");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 20.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_20");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 50.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_50");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 100.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_100");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.eval_reward.value = 0.;
    test_params_sim(&params_sim, &params_phys, "reward_0");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 1.;
    test_params_sim(&params_sim, &params_phys, "reward_1");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 1.;
    test_params_sim(&params_sim, &params_phys, "reward_10");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 50.;
    test_params_sim(&params_sim, &params_phys, "reward_50");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 100.;
    test_params_sim(&params_sim, &params_phys, "reward_100");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 200.;
    test_params_sim(&params_sim, &params_phys, "reward_200");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 500.;
    test_params_sim(&params_sim, &params_phys, "reward_500");
    params_sim = params_sim_copy.clone();

    params_sim.eval_reward.value = 1000.;
    test_params_sim(&params_sim, &params_phys, "reward_1000");
    params_sim = params_sim_copy.clone();
    */

    params_sim.rewards_second_way = true;
    params_sim_copy.rewards_second_way = true;

    test_params_sim_differential_evolution(&params_sim, &params_phys, "differential_evolution_2w");
    test_params_sim_particle_swarm(&params_sim, &params_phys, "particle_swarm_2w");

    /*params_sim.mutate_car_enable = true;
    test_params_sim(&params_sim, &params_phys, "mutate_car_enabled_2w");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_simple_physics = 1.0;
    test_params_sim(&params_sim, &params_phys, "simple_physics_1.0_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_simple_physics = 0.5;
    test_params_sim(&params_sim, &params_phys, "simple_physics_0.5_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_simple_physics = 0.0;
    test_params_sim(&params_sim, &params_phys, "simple_physics_0.0_2w");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.disable_track("complex");
    test_params_sim(&params_sim, &params_phys, "no_tr_complex_2w");
    params_sim = params_sim_copy.clone();

    params_sim.enable_track("straight_45");
    test_params_sim(&params_sim, &params_phys, "with_tr_45_2w");
    params_sim = params_sim_copy.clone();

    params_sim.disable_track("turn_left_90");
    params_sim.disable_track("turn_left_180");
    test_params_sim(&params_sim, &params_phys, "no_tr_turns_2w");
    params_sim = params_sim_copy.clone();

    params_sim.disable_all_tracks();
    params_sim.enable_track("complex");
    test_params_sim(&params_sim, &params_phys, "only_complex_track_2w");
    params_sim = params_sim_copy.clone();
    
    params_sim.disable_track("turn_left_90");
    params_sim.disable_track("turn_left_180");
    params_sim.disable_track("complex");
    params_sim.disable_track("straight_45");
    test_params_sim(&params_sim, &params_phys, "only_tr_simple_2w");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_enable_random_nn_output = false;
    test_params_sim(&params_sim, &params_phys, "random_output_no_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.01;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.01_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.05;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.05_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.1;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.1_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.2;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.2_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_random_output_range = 0.4;
    params_sim.simulation_enable_random_nn_output = true;
    test_params_sim(&params_sim, &params_phys, "random_output_0.4_2w");
    params_sim = params_sim_copy.clone();

    //------------------------------------------------------------------------

    params_sim.simulation_stop_penalty.value = 0.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_0_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 1.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_1_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 5.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_5_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 10.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_10_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 20.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_20_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 50.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_50_2w");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 100.;
    test_params_sim(&params_sim, &params_phys, "stop_penalty_100_2w");
    params_sim = params_sim_copy.clone();*/
}
