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
use differential_evolution::Population;
use egui::emath::RectTransform;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::DragValue;
use egui::Painter;
use egui::Pos2;
use egui::Shape;
use egui::Slider;
use egui::Stroke;
use egui::Ui;
use egui::Vec2;
use rand::thread_rng;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Instant;

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
    pub pass_internals: bool,
    pub pass_prev_output: bool,
    pub pass_simple_physics_value: bool,
    pub pass_next_size: usize,
    pub hidden_layers: Vec<usize>,
    pub inv_distance: bool,
    pub inv_distance_coef: f32,
    pub inv_distance_pow: f32,
    pub view_angle_ratio: f32,

    pub pass_dirs: bool,
    pub dirs_size: usize,
    pub pass_dirs_diff: bool,
    pub pass_dirs_second_layer: bool,

    pub pass_current_track: bool,
    pub max_tracks: usize,

    pub pass_current_segment: bool,
    pub max_segments: usize,

    pub use_dirs_autoencoder: bool,
    pub autoencoder_exits: usize,
    pub autoencoder_hidden_layers: Vec<usize>,

    pub pass_time_mods: Vec<f32>,
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
    pub rewards_early_finish_zero_penalty: bool,

    pub simulation_stop_penalty: Clamped, // ratio from 0 to 50
    pub simulation_scale_reward_to_time: bool, // if reward acquired earlier, it will be bigger
    pub simulation_steps_quota: usize,    // reasonable values from 1000 to 3000
    pub simulation_simple_physics: f32,
    pub simulation_enable_random_nn_output: bool,
    pub simulation_random_output_range: f32,
    pub simulation_use_output_regularization: bool,

    pub evolution_sample_mean: bool,
    pub evolution_generation_count: usize,
    pub evolution_population_size: usize,
    pub evolution_learning_rate: f64,
    pub evolution_distance_to_solution: f64,

    pub eval_skip_passed_tracks: bool,
    pub eval_add_min_distance: bool,
    pub eval_reward: Clamped,       // ratio from 0 to 5000
    pub eval_early_finish: Clamped, // ratio from 0 to 5000
    pub eval_distance: Clamped,     // ratio from 0 to 5000
    pub eval_acquired: Clamped,     // ratio from 0 to 5000
    pub eval_penalty: Clamped,      // ratio from 0 to 5000
    pub eval_calc_all_physics: bool,
    pub eval_calc_all_physics_count: usize,
    pub eval_add_other_physics: bool,

    pub evolve_simple_physics: bool,
    pub hard_physics_reward: Clamped,

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

        ui.label("Pass simple physics value:");
        ui.checkbox(&mut self.pass_simple_physics_value, "");
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
            + self.pass_dirs as usize
                * if self.use_dirs_autoencoder {
                    self.autoencoder_exits
                } else {
                    self.dirs_size
                }
            + self.pass_internals as usize * Self::CAR_INPUT_SIZE
            + self.pass_next_size
            + self.pass_prev_output as usize * Self::CAR_OUTPUT_SIZE
            + self.pass_simple_physics_value as usize
            + self.pass_current_track as usize * self.max_tracks
            + self.pass_dirs_diff as usize * self.dirs_size
            + self.pass_time_mods.len()
            + self.pass_current_segment as usize * self.max_segments
            + self.pass_dirs_second_layer as usize * self.dirs_size
    }

    pub fn get_total_output_neurons(&self) -> usize {
        Self::CAR_OUTPUT_SIZE + self.pass_next_size
    }

    pub fn get_nn_sizes(&self) -> Vec<usize> {
        std::iter::once(self.get_total_input_neurons())
            .chain(self.hidden_layers.iter().copied())
            .chain(std::iter::once(self.get_total_output_neurons()))
            .collect()
    }

    pub fn get_nn_autoencoder_input_sizes(&self) -> Vec<usize> {
        std::iter::once(self.dirs_size)
            .chain(self.autoencoder_hidden_layers.iter().copied())
            .chain(std::iter::once(self.autoencoder_exits))
            .collect()
    }

    pub fn get_nn_autoencoder_output_sizes(&self) -> Vec<usize> {
        let mut result = self.get_nn_autoencoder_input_sizes();
        result.reverse();
        result
    }

    pub fn get_nn_len(&self) -> usize {
        NeuralNetwork::calc_nn_len(&self.get_nn_sizes())
    }

    pub fn get_nns_len(&self) -> usize {
        self.get_nn_len() // + self.get_nns_autoencoder_len()
    }

    pub fn get_nns_autoencoder_len(&self) -> usize {
        if self.use_dirs_autoencoder {
            self.get_nn_autoencoder_input_len() + self.get_nn_autoencoder_output_len()
        } else {
            0
        }
    }

    pub fn get_nn_autoencoder_input_len(&self) -> usize {
        if self.use_dirs_autoencoder {
            NeuralNetwork::calc_nn_len(&self.get_nn_autoencoder_input_sizes())
        } else {
            0
        }
    }

    pub fn get_nn_autoencoder_output_len(&self) -> usize {
        if self.use_dirs_autoencoder {
            NeuralNetwork::calc_nn_len(&self.get_nn_autoencoder_output_sizes())
        } else {
            0
        }
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

pub fn mirror_horizontally_track2(
    (name, mut points): (String, Vec<PointsStorage>),
) -> (String, Vec<PointsStorage>) {
    let car_center = 250.;
    let points = points
        .into_iter()
        .map(|mut storage| {
            storage.points.iter_mut().for_each(|point| {
                point.y = car_center - (point.y - car_center);
            });
            storage
        })
        .collect();
    (name + "_mirror", points)
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

#[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct PointsStorage {
    pub is_reward: bool,
    pub points: Vec<Pos2>,
}

impl PointsStorage {
    pub fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        ui.selectable_value(&mut self.is_reward, false, "Walls");
        ui.selectable_value(&mut self.is_reward, true, "Rewards");
        egui_array_inner(&mut self.points, ui, data_id, egui_pos2, false);
    }
}

pub fn egui_pos2(pos: &mut Pos2, ui: &mut Ui, data_id: egui::Id) {
    egui::Grid::new(data_id.with("wall_grid"))
        .num_columns(2)
        .spacing([40.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("Pos X:");
            ui.add(
                DragValue::new(&mut pos.x)
                    .speed(5.0)
                    .min_decimals(0)
                    .max_decimals(0),
            );
            ui.end_row();

            ui.label("Pos Y:");
            ui.add(
                DragValue::new(&mut pos.y)
                    .speed(5.0)
                    .min_decimals(0)
                    .max_decimals(0),
            );
            ui.end_row();
        });
}

pub fn points_storage_to_track((name, points): (String, Vec<PointsStorage>)) -> Track {
    let mut walls: Vec<Wall> = Default::default();
    let mut rewards: Vec<Reward> = Default::default();
    for elem in &points {
        if elem.is_reward {
            rewards.extend(rewards_from_points(elem.points.iter().copied()));
        } else {
            walls.extend(walls_from_points(elem.points.iter().copied()));
        }
    }
    Track::new(name, walls, rewards)
}

// straight line
pub fn track_straight_line() -> (String, Vec<PointsStorage>) {
    (
        "straight".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(-14.23, 17.90),
                    pos2(5635.54, 246.73),
                    pos2(5640.00, -98.93),
                    pos2(6238.29, -76.85),
                    pos2(6273.62, 485.02),
                    pos2(5630.75, 488.32),
                    pos2(-100.47, 409.92),
                    pos2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![pos2(255.56, 251.16), pos2(5484.87, 358.62)],
            },
        ],
    )
}

// turn right smooth
pub fn track_turn_right_smooth() -> (String, Vec<PointsStorage>) {
    (
        "turn_right_smooth".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(84.64, -1.89),
                    pos2(1227.77, -3.24),
                    pos2(2054.61, 103.97),
                    pos2(2971.24, 374.67),
                    pos2(3590.36, 815.56),
                    pos2(3966.93, 1354.28),
                    pos2(4023.21, 1911.76),
                    pos2(4039.29, 2559.02),
                    pos2(4052.83, 2951.00),
                    pos2(2972.12, 2941.26),
                    pos2(2973.34, 2549.38),
                    pos2(3669.42, 2560.36),
                    pos2(3660.04, 1903.72),
                    pos2(3590.36, 1490.97),
                    pos2(3282.13, 1095.64),
                    pos2(2814.44, 787.42),
                    pos2(1999.67, 558.26),
                    pos2(1249.21, 438.99),
                    pos2(86.01, 445.69),
                    pos2(84.67, -1.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(251.92, 250.12),
                    pos2(1234.09, 215.53),
                    pos2(2034.77, 349.62),
                    pos2(2942.04, 604.14),
                    pos2(3430.01, 958.38),
                    pos2(3786.43, 1417.46),
                    pos2(3839.26, 1887.05),
                    pos2(3851.00, 2485.98),
                ],
            },
        ],
    )
}

// harn turn left
pub fn track_turn_left_90() -> (String, Vec<PointsStorage>) {
    (
        "turn_left_90".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(80.34, -5.05),
                    pos2(997.84, -6.53),
                    pos2(997.84, -1646.50),
                    pos2(994.08, -2093.93),
                    pos2(1880.17, -2085.06),
                    pos2(1878.69, -1653.10),
                    pos2(1457.33, -1649.46),
                    pos2(1466.19, 534.11),
                    pos2(87.06, 524.75),
                    pos2(78.86, -5.05),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(258.24, 250.23),
                    pos2(785.25, 251.82),
                    pos2(1063.20, 200.65),
                    pos2(1211.43, 80.65),
                    pos2(1231.00, -149.45),
                    pos2(1220.93, -1508.56),
                ],
            },
        ],
    )
}

pub fn track_turn_left_180() -> (String, Vec<PointsStorage>) {
    (
        "turn_left_180".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(68.68, -37.37),
                    pos2(68.39, 495.62),
                    pos2(1628.53, 506.55),
                    pos2(1626.82, -437.56),
                    pos2(70.41, -430.71),
                    pos2(75.44, -1071.74),
                    pos2(-334.32, -1068.79),
                    pos2(-332.84, -36.24),
                    pos2(68.70, -37.34),
                    pos2(1219.76, -44.18),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(251.38, 247.87),
                    pos2(987.99, 246.03),
                    pos2(1315.29, 175.24),
                    pos2(1449.97, -39.40),
                    pos2(1328.72, -223.41),
                    pos2(953.16, -260.21),
                    pos2(238.07, -248.15),
                ],
            },
        ],
    )
}

pub fn track_turn_around() -> (String, Vec<PointsStorage>) {
    (
        "turn_around".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(550.85, -8.85),
                    pos2(552.00, 501.00),
                    pos2(-2048.97, 486.81),
                    pos2(-2051.63, -17.28),
                    pos2(521.62, -12.89),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(-200.11, 253.13),
                    pos2(-959.28, 250.08),
                    pos2(-1433.05, 248.37),
                    pos2(-1783.67, 248.37),
                ],
            },
        ],
    )
}

pub fn track_smooth_left_and_right() -> (String, Vec<PointsStorage>) {
    (
        "smooth_left_and_right".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(110.72, 101.35),
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
                    pos2(4778.57, 322.66),
                    pos2(4638.04, 662.90),
                    pos2(3654.31, 253.14),
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
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(257.32, 247.26),
                    pos2(1508.32, 197.59),
                    pos2(2056.80, -456.76),
                    pos2(2158.07, -1332.80),
                    pos2(2369.81, -1773.12),
                    pos2(2813.37, -2095.70),
                    pos2(3726.17, -2087.27),
                    pos2(4042.62, -1794.75),
                    pos2(4209.94, -1277.47),
                    pos2(4215.89, -625.99),
                    pos2(4016.48, -138.96),
                ],
            },
        ],
    )
}

#[allow(clippy::approx_constant)]
pub fn track_complex() -> (String, Vec<PointsStorage>) {
    (
        "complex".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(-8.88, -9.06),
                    pos2(2150.00, -1.43),
                    pos2(2142.86, -1431.43),
                    pos2(3690.00, -1424.29),
                    pos2(3687.09, 3202.46),
                    pos2(-2832.30, 157.16),
                    pos2(-2834.52, -3251.33),
                    pos2(755.70, -3253.55),
                    pos2(751.49, -4097.08),
                    pos2(1868.51, -4094.22),
                    pos2(1916.49, -2829.04),
                    pos2(751.27, -2828.04),
                    pos2(-2377.98, -2861.28),
                    pos2(-1792.91, -341.48),
                    pos2(-11.10, 580.45),
                    pos2(-8.88, -9.06),
                    pos2(-11.10, 580.45),
                    pos2(2883.23, 591.53),
                    pos2(2883.23, -862.84),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    pos2(249.86, 250.44),
                    pos2(1953.63, 285.30),
                    pos2(2266.79, 233.48),
                    pos2(2435.79, 66.79),
                    pos2(2485.06, -204.70),
                    pos2(2484.85, -717.33),
                    pos2(2609.34, -1050.41),
                    pos2(2874.87, -1191.42),
                    pos2(3175.06, -1060.41),
                    pos2(3301.25, -774.23),
                    pos2(3302.97, 488.77),
                    pos2(2895.47, 1000.49),
                    pos2(333.65, 1022.85),
                    pos2(-1671.56, 191.94),
                    pos2(-2228.68, -312.10),
                    pos2(-2595.71, -2323.18),
                    pos2(-2644.09, -2728.56),
                    pos2(-2598.97, -2967.43),
                    pos2(-2372.19, -3075.67),
                    pos2(393.66, -3040.27),
                ],
            },
        ],
    )
}

pub fn track_straight_45() -> (String, Vec<PointsStorage>) {
    (
        "straight_45".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    pos2(-77.00, 178.00),
                    pos2(2558.78, -2103.75),
                    pos2(2892.96, -1732.53),
                    pos2(256.05, 574.77),
                    pos2(-86.66, 186.21),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
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
                ],
            },
        ],
    )
}

pub fn get_all_tracks() -> Vec<(String, Vec<PointsStorage>)> {
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
    max_distance: f32,
    current_segment_f32: f32,
}

impl RewardPathProcessor {
    pub fn new(rewards: Vec<Reward>) -> Self {
        Self {
            rewards,
            max_distance: 0.,
            current_segment_f32: 0.,
        }
    }

    fn process_point(&mut self, point: Pos2, params: &SimulationParameters) -> f32 {
        let mut reward_sum = 0.;

        let (segm, dist) = if self.rewards.len() >= 2 {
            let (pos, _, dist, t) = self
                .rewards
                .windows(2)
                .enumerate()
                .map(|(pos, x)| {
                    let (a, b) = (x[0].center, x[1].center);
                    let (res, t) = project_to_segment(point, a, b);
                    (pos, res, (point - res).length(), t)
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or_else(|| {
                    dbg!(a);
                    dbg!(b);
                    dbg!(&self.rewards);
                    dbg!(&point);
                    todo!()
                }))
                .unwrap();
            (pos as f32 + t, dist)
        } else {
            (0., 0.)
        };
        self.current_segment_f32 = segm;

        if self.max_distance < segm {
            if params.rewards_enable_distance_integral {
                reward_sum += (self.current_segment_f32 - self.max_distance) / (dist + 1.0);
            }
            self.max_distance = self.current_segment_f32;
        }

        reward_sum
    }

    fn reset(&mut self) {
        self.current_segment_f32 = 0.;
        self.max_distance = 0.;
    }

    fn draw(&self, point: Pos2, painter: &Painter, to_screen: &RectTransform) {
        let projection1 = self
            .rewards
            .windows(2)
            .enumerate()
            .map(|(pos, x)| {
                let (a, b) = (x[0].center, x[1].center);
                let (res, t) = project_to_segment(point, a, b);
                (pos, res, (point - res).length(), t)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap_or_default()
            .1;
        painter.add(Shape::line(
            vec![
                to_screen.transform_pos(point),
                to_screen.transform_pos(projection1),
            ],
            Stroke::new(1.0, Color32::DARK_RED),
        ));

        for (i, (a, b)) in pairs(self.rewards.iter()).enumerate() {
            painter.add(Shape::line(
                vec![
                    to_screen.transform_pos(a.center),
                    to_screen.transform_pos(b.center),
                ],
                Stroke::new(1.0, Color32::DARK_GREEN),
            ));
        }
    }

    pub fn distance_percent(&self) -> f32 {
        self.max_distance / (self.rewards.len() as f32 - 1.0)
    }

    pub fn all_acquired(&self) -> bool {
        (self.distance_percent() - 1.0).abs() < 1e-6
    }

    pub fn get_current_segment_f32(&self) -> f32 {
        self.current_segment_f32
    }
}

fn convert_dir(params: &NnParameters, intersection: Option<f32>) -> f32 {
    if params.inv_distance {
        intersection
            .map(|x| params.inv_distance_coef / (x + 1.).powf(params.inv_distance_pow))
            .unwrap_or(0.)
    } else {
        intersection.map(|x| x.max(1000.)).unwrap_or(1000.)
    }
}

pub struct NnProcessorAutoencoder {
    input: Vec<f32>,
    output: Vec<f32>,
    nn_input: NeuralNetwork,
    nn_output: NeuralNetwork,
    autoencoder_loss: f32,
    calc_counts: usize,
    params: NnParameters,
}

impl NnProcessorAutoencoder {
    pub fn new_zeros(nn_params: NnParameters) -> Self {
        Self {
            input: vec![0.; nn_params.dirs_size],
            output: vec![0.; nn_params.autoencoder_exits],
            nn_input: NeuralNetwork::new_zeros(nn_params.get_nn_autoencoder_input_sizes()),
            nn_output: NeuralNetwork::new_zeros(nn_params.get_nn_autoencoder_output_sizes()),
            autoencoder_loss: 0.,
            calc_counts: 0,
            params: nn_params,
        }
    }

    pub fn new(params: &[f32], nn_params: NnParameters) -> Self {
        let (params_input, params_output) =
            params.split_at(nn_params.get_nn_autoencoder_input_len());
        Self {
            input: vec![0.; nn_params.dirs_size],
            output: vec![0.; nn_params.autoencoder_exits],
            nn_input: NeuralNetwork::new_params(
                nn_params.get_nn_autoencoder_input_sizes(),
                params_input,
            ),
            nn_output: NeuralNetwork::new_params(
                nn_params.get_nn_autoencoder_output_sizes(),
                params_output,
            ),
            autoencoder_loss: 0.,
            calc_counts: 0,
            params: nn_params,
        }
    }

    pub fn process<'a>(&'a mut self, dirs: &[Option<f32>]) -> &'a [f32] {
        self.calc_counts += 1;

        let mut autoencoder_input_iter = self.input.iter_mut();
        for intersection in dirs {
            *autoencoder_input_iter.next().unwrap() = convert_dir(&self.params, *intersection);
        }
        assert!(autoencoder_input_iter.next().is_none());

        let values = self.nn_input.calc(&self.input);

        let values_output = self.nn_output.calc(values);
        let mut sum = 0.;
        for (intersection, prediction) in dirs.iter().zip(values_output.iter()) {
            sum += 1. / (1. + (convert_dir(&self.params, *intersection) - prediction).abs());
        }
        self.autoencoder_loss += sum / dirs.len() as f32;

        values
    }

    pub fn get_regularization_loss(&self) -> f32 {
        self.nn_input.get_regularization() + self.nn_output.get_regularization()
    }

    pub fn get_autoencoder_loss(&self) -> f32 {
        if self.calc_counts == 0 {
            0.
        } else {
            self.autoencoder_loss / self.calc_counts as f32
        }
    }

    pub fn reset(&mut self) {
        self.input.iter_mut().for_each(|x| *x = 0.);
        self.output.iter_mut().for_each(|x| *x = 0.);
        self.autoencoder_loss = 0.;
    }
}

const BEST_AUTOENCODER_21_10_10_10_5: [f32; 1006] = [
    0.45737186,
    -0.17592742,
    0.41990036,
    0.3023682,
    0.46085286,
    0.09734252,
    0.20202918,
    0.18677261,
    0.15259826,
    0.08691186,
    0.16731122,
    0.10409911,
    -0.27700168,
    -0.013539409,
    -0.09711216,
    0.14397214,
    -0.15910262,
    0.21080758,
    0.43845347,
    0.07938459,
    0.47766662,
    0.49709564,
    0.5581285,
    0.5085441,
    0.37938422,
    0.13941869,
    -0.011706705,
    0.111483075,
    -0.3481085,
    -0.17621183,
    -0.16999112,
    0.33578566,
    0.33830214,
    0.3392177,
    0.46204475,
    0.43641356,
    0.02911971,
    -0.24209979,
    -0.13739654,
    0.07810422,
    -0.42370325,
    0.048519064,
    0.47701773,
    0.36765498,
    0.25073645,
    0.34227213,
    0.28530744,
    0.12449606,
    0.33620736,
    0.4206451,
    0.37811056,
    0.48892096,
    0.31235722,
    -0.019208623,
    0.28711075,
    -0.32138065,
    -0.48255187,
    -0.073856294,
    0.21494687,
    0.2527926,
    0.25357565,
    0.06038692,
    0.21669765,
    -0.4017394,
    0.0030092,
    0.027453631,
    -0.008625239,
    -0.12991595,
    -0.3729226,
    0.27464026,
    -0.35417527,
    -0.32996136,
    -0.3039164,
    -0.41292477,
    -0.008672744,
    -0.16740495,
    0.27102596,
    -0.25378257,
    -0.09404045,
    -0.34924185,
    0.4185815,
    -0.19595659,
    0.06775886,
    -0.40013322,
    0.0044076815,
    0.22488806,
    0.038864266,
    -0.38992977,
    -0.3962791,
    -0.008726857,
    -0.08023614,
    0.045806993,
    0.23833953,
    0.5205801,
    0.768315,
    0.30037403,
    -0.008285542,
    0.13156843,
    0.016080525,
    -0.41769928,
    -0.5351447,
    -0.2394889,
    0.4227838,
    0.29043153,
    0.3176221,
    -0.42664915,
    -0.20242606,
    -0.25436103,
    -0.005583709,
    0.16876525,
    0.41205645,
    -0.28080022,
    0.10476436,
    -0.31362674,
    -0.2677709,
    -0.42060456,
    -0.3070178,
    0.15683068,
    0.42874318,
    0.22397202,
    0.3479392,
    -0.08452995,
    0.15452468,
    0.3514351,
    -0.01399497,
    -0.40478998,
    -0.2482755,
    -0.1356977,
    -0.2107391,
    0.1617366,
    -0.24560514,
    -0.09844121,
    -0.05664088,
    0.016249405,
    -0.20677516,
    -0.057893697,
    -0.3120921,
    0.14034316,
    0.19313832,
    0.2763481,
    0.3531536,
    0.56920975,
    0.5262653,
    0.38728634,
    -0.030576818,
    0.6514924,
    -0.10670456,
    -0.069721915,
    0.25045338,
    -0.14655343,
    0.35060158,
    -0.10266501,
    0.63437945,
    0.32942742,
    0.45425716,
    0.074557275,
    0.39037,
    0.0637424,
    -0.17551796,
    -0.20605043,
    0.3715435,
    0.44256073,
    -0.0024101275,
    -0.19201526,
    -0.24129438,
    0.39782032,
    0.5097004,
    0.1726135,
    -0.3583182,
    -0.23892967,
    0.28712755,
    -0.21878761,
    0.21266371,
    -0.3139548,
    0.2520895,
    0.20426053,
    -0.38272342,
    -0.13531768,
    -0.37770674,
    -0.07767549,
    -0.079563916,
    0.076762915,
    -0.09228301,
    -0.15359625,
    0.39501822,
    -0.32253093,
    -0.05489093,
    -0.10004258,
    0.043926954,
    -0.21595538,
    0.42019904,
    -0.19991599,
    -0.2796001,
    -0.3063535,
    -0.1659193,
    0.11443508,
    -0.28578854,
    0.07319701,
    -0.2500814,
    -0.015817225,
    0.39411527,
    -0.14725995,
    0.39196688,
    -0.25877836,
    -0.04152623,
    -0.095975876,
    -0.15781116,
    0.028069556,
    -0.14534119,
    0.019865453,
    -0.06348205,
    -0.038866445,
    -0.12543958,
    0.0,
    -0.049645416,
    -0.008875614,
    -0.34252134,
    -0.02051096,
    0.0,
    0.0,
    0.027617633,
    0.20035297,
    -0.35912716,
    0.33826768,
    0.24858415,
    0.13768375,
    -0.03795594,
    -0.09491876,
    -0.5105505,
    -0.2659443,
    -0.45389396,
    0.11748606,
    0.09288216,
    0.32547277,
    -0.030020177,
    0.24329084,
    0.08851558,
    -0.42366457,
    -0.26695818,
    0.101017475,
    -0.49297047,
    0.36307758,
    -0.34514493,
    0.2988848,
    0.035871148,
    0.5472412,
    0.50963855,
    -0.45673212,
    0.40956378,
    -0.1742078,
    0.31611833,
    0.5084936,
    0.40096274,
    -0.1617916,
    0.42529443,
    0.28289756,
    0.31026813,
    0.1375107,
    0.032167792,
    -0.39572728,
    0.28746232,
    -0.04957891,
    0.20623049,
    0.15467656,
    -0.4147029,
    -0.11097115,
    0.8231028,
    0.20070969,
    0.4504164,
    -0.1172778,
    0.43438476,
    -0.3721964,
    0.4799559,
    0.5127816,
    0.22977056,
    0.5342281,
    -0.49618167,
    0.48291194,
    -0.54680806,
    -0.41188288,
    -0.46225387,
    0.02290225,
    -0.30547047,
    -0.4821162,
    0.16012192,
    0.38117933,
    0.09050703,
    -0.19624546,
    -0.15609431,
    0.07042986,
    0.3148246,
    -0.05269588,
    -0.039140135,
    -0.27311864,
    -0.5313749,
    -0.07278303,
    -0.28564167,
    0.40633324,
    0.27438074,
    -0.33285016,
    0.48366016,
    -0.1868916,
    -0.11395642,
    0.41317356,
    0.6122248,
    -0.17725272,
    0.5885771,
    -0.043597978,
    -0.12985998,
    -0.48262614,
    0.036938548,
    -0.23215446,
    -0.4586741,
    0.40202367,
    0.39871365,
    0.40458053,
    0.11369836,
    0.047267616,
    -0.33472955,
    -0.0736922,
    0.0,
    0.0,
    0.0,
    -0.04239895,
    -0.048857134,
    -0.07255001,
    0.0,
    0.02166893,
    0.003860553,
    0.0,
    0.015568614,
    -0.12189758,
    0.1656887,
    0.56778526,
    0.2573384,
    -0.14126685,
    0.26964617,
    0.38049787,
    -0.36964574,
    0.5429429,
    0.4858694,
    0.5392759,
    0.33397192,
    -0.5292928,
    0.48138225,
    -0.3814337,
    -0.26645464,
    -0.22139981,
    0.0393188,
    0.540257,
    0.3732344,
    0.061119914,
    -0.40323785,
    -0.027448416,
    -0.09904677,
    -0.39385843,
    0.51572704,
    0.051013887,
    0.010567129,
    -0.45928824,
    -0.046225548,
    0.35471177,
    -0.11607221,
    0.4036979,
    0.04851145,
    0.31053594,
    -0.20114025,
    -0.06318814,
    -0.11971743,
    0.2558158,
    0.4767382,
    0.06389463,
    0.5359721,
    0.27281535,
    0.61154187,
    -0.17682181,
    -0.5467057,
    -0.15335092,
    0.16274224,
    -0.2988279,
    -0.21623209,
    0.47806942,
    0.51816785,
    -0.3264413,
    0.3658301,
    0.23505229,
    -0.064686,
    0.58192694,
    0.036728177,
    -0.19700703,
    -0.24694583,
    0.36191887,
    0.12063205,
    -0.07626569,
    -0.3799886,
    -0.45023346,
    0.45717847,
    0.10571009,
    -0.30130303,
    0.34269577,
    -0.20633999,
    0.0499897,
    -0.097061306,
    -0.4296894,
    -0.44272336,
    0.5295114,
    0.40523547,
    0.47777525,
    -0.47180068,
    0.43413532,
    0.14695245,
    -0.039705575,
    0.15221053,
    -0.10159143,
    0.1270639,
    0.17898011,
    -0.32885066,
    -0.5786,
    -0.023145985,
    0.24564582,
    0.2326225,
    -0.0007247329,
    0.21068501,
    -0.068891704,
    -0.2498591,
    0.38694972,
    0.25435388,
    -0.26001695,
    -0.53119427,
    0.42076278,
    -0.020152673,
    0.0,
    0.0,
    -0.04903647,
    -0.021485018,
    0.0063240416,
    0.0,
    -0.013520969,
    -0.04661106,
    0.0,
    0.25270316,
    -0.044974983,
    -0.515672,
    0.079448774,
    0.50788134,
    0.65939474,
    0.326761,
    -0.44779658,
    -0.6086799,
    -0.48577356,
    -0.47844335,
    -0.041437924,
    -0.6244541,
    0.6047946,
    0.60107666,
    -0.8273604,
    0.25618458,
    -0.1575329,
    0.027038105,
    -0.23821822,
    0.5200817,
    0.5768277,
    0.2281723,
    -0.57039213,
    -0.59204894,
    -0.4897523,
    -0.20118606,
    -0.10484761,
    -0.14021656,
    -0.38588452,
    0.3546,
    0.61131567,
    -0.4037295,
    0.7011073,
    -0.33956572,
    0.04620619,
    -0.32616884,
    -0.4394969,
    -0.49657133,
    -0.33803958,
    0.40583158,
    -0.35029602,
    -0.5258989,
    -0.026526408,
    0.27576658,
    -0.792013,
    -0.20139068,
    0.011485338,
    -0.31658253,
    0.2183519,
    -0.018916938,
    0.0050536706,
    -0.02281617,
    -0.038263872,
    0.0310839,
    0.56170845,
    -0.6282362,
    -0.5168994,
    -0.15849586,
    0.096899875,
    -0.060975116,
    0.33497098,
    0.17076254,
    0.3759598,
    -0.5072324,
    0.13570258,
    -0.1473247,
    -0.17493798,
    0.10895595,
    0.43225223,
    -0.5622761,
    0.22041798,
    0.25107282,
    -0.3827208,
    -0.3367378,
    -0.53930104,
    0.06395388,
    -0.21373653,
    -0.13711393,
    -0.17852244,
    -0.014904106,
    0.7355005,
    0.25485387,
    0.20877622,
    0.59275186,
    -0.28384012,
    0.39728564,
    0.021546245,
    -0.22174235,
    -0.5313238,
    0.29870403,
    -0.7480616,
    0.23096626,
    0.35752147,
    0.31776556,
    0.60854894,
    0.7437414,
    0.40992495,
    0.0069172373,
    -0.6218858,
    0.4920594,
    0.07133994,
    -0.070414774,
    0.6690848,
    0.041190106,
    0.016126594,
    -0.0058748773,
    -0.07512411,
    0.0,
    0.0,
    -0.005659065,
    -0.011941007,
    -0.006117512,
    -0.031855706,
    -0.040702112,
    0.648664,
    -0.41656962,
    0.22611614,
    0.07048929,
    0.38365042,
    0.43846026,
    -0.4582126,
    0.52142835,
    -0.41739795,
    0.20872425,
    -0.74500805,
    -0.5456883,
    0.29133123,
    0.0045635104,
    0.45645988,
    0.0218608,
    0.37450957,
    0.19543046,
    -0.2703051,
    0.37894905,
    0.3440333,
    0.42775798,
    0.06840312,
    -0.14772394,
    -0.13403201,
    0.49949837,
    0.16941537,
    -0.42480052,
    0.33693975,
    0.5180814,
    -0.23840593,
    0.34738067,
    -0.14095815,
    0.4771434,
    0.0059478283,
    0.3157413,
    -0.4141257,
    -0.32466415,
    0.30409428,
    0.073748946,
    0.30155033,
    -0.17202307,
    0.31254074,
    0.34113133,
    -0.4173333,
    -0.30333576,
    0.3580578,
    -0.058534212,
    0.42712164,
    0.46393025,
    0.19257312,
    0.91011685,
    0.17239583,
    -0.01824528,
    -0.12686616,
    -0.19433935,
    -0.3245527,
    0.43490216,
    0.22452417,
    0.1861319,
    -0.4840148,
    -0.01780321,
    0.18180817,
    0.38608533,
    -0.09568596,
    -0.057836026,
    0.31867123,
    0.5001768,
    -0.5310913,
    -0.23036578,
    -0.18935914,
    0.44626456,
    -0.014953927,
    -0.005733669,
    -0.4992405,
    0.40648514,
    0.236542,
    -0.47628355,
    -0.32074094,
    0.43714648,
    0.073094346,
    0.43527675,
    0.13248475,
    0.12132627,
    0.37790197,
    -0.17227016,
    -0.52738965,
    0.039165292,
    -0.16698352,
    0.4481608,
    -0.061499804,
    0.26578775,
    -0.3720556,
    0.3283339,
    0.43164974,
    0.27652317,
    0.059041668,
    0.36649668,
    0.30042675,
    0.5507893,
    0.031096091,
    -0.023072015,
    0.015341973,
    0.063868605,
    0.05314186,
    -0.10451103,
    0.012260199,
    -0.033602644,
    -0.007871839,
    -0.028260214,
    0.4560528,
    -0.43863538,
    -0.1972818,
    -0.54110587,
    -0.11297603,
    0.5834811,
    -0.3068129,
    0.16640697,
    0.15583868,
    0.306662,
    -0.1322146,
    -0.25152695,
    -0.36682713,
    -0.40586734,
    -0.52713156,
    0.08048719,
    -0.4596381,
    -0.16805714,
    0.22507417,
    0.01995802,
    0.41539112,
    0.12115909,
    0.61171275,
    0.17476349,
    -0.17620562,
    -0.38986272,
    -0.1986934,
    -0.07216763,
    -0.2522651,
    0.1745536,
    -0.00065242837,
    0.33300617,
    -0.16377176,
    0.47842458,
    0.1233035,
    -0.11198796,
    -0.5111505,
    0.44108307,
    -0.3224152,
    -0.34009638,
    -0.19228707,
    -0.30730093,
    0.25576028,
    -0.10244663,
    -0.31067827,
    0.37724394,
    0.49409118,
    0.0061814627,
    0.5397924,
    0.32008857,
    -0.30164802,
    -0.50917286,
    0.17394805,
    -0.0714896,
    -0.44997922,
    0.15607458,
    -0.53747565,
    -0.5079738,
    -0.0361138,
    0.21564847,
    -0.57721186,
    -0.90376776,
    0.52751344,
    0.44228637,
    -0.07307916,
    -0.33051163,
    0.03798145,
    -0.7166991,
    0.0744902,
    0.533621,
    -0.58328587,
    0.2642905,
    0.30252588,
    0.12714164,
    -0.34574488,
    0.023801422,
    -0.13283314,
    0.2592124,
    0.11557618,
    0.27972016,
    0.22628273,
    0.4573506,
    -0.03671455,
    0.5283449,
    0.016419219,
    -0.30363163,
    -0.51035285,
    0.22357996,
    -0.4086221,
    -0.15721984,
    0.47282434,
    -0.09929528,
    0.5269532,
    0.14871538,
    0.47924593,
    0.12536359,
    0.06255887,
    -0.4338604,
    -0.17016983,
    0.039393302,
    -0.021478916,
    0.0,
    -0.0011150183,
    -0.02002353,
    0.013970958,
    0.0,
    0.08725746,
    -0.043776356,
    0.14214514,
    0.026229413,
    -0.2595059,
    -0.12774545,
    -0.0494774,
    -0.12511805,
    0.4504645,
    -0.08772072,
    -0.73110783,
    0.125394,
    0.31506735,
    0.42245546,
    0.3609071,
    -0.2768759,
    0.36766338,
    -0.24215254,
    0.1554749,
    0.31662637,
    -0.4220921,
    0.14024962,
    0.53150356,
    -0.052198645,
    0.24542153,
    -0.27837205,
    0.301992,
    -0.30569372,
    0.24076378,
    -0.045823783,
    -0.43873274,
    0.102294445,
    0.28654665,
    0.004846038,
    0.00082837255,
    -0.2729205,
    -0.24653284,
    0.11174075,
    -0.07795748,
    0.24567664,
    -0.62129486,
    0.5075251,
    -0.5105555,
    0.5848545,
    0.38590983,
    -0.16161081,
    0.2442681,
    0.37855762,
    0.1243355,
    0.25341237,
    -0.23978654,
    0.18581568,
    -0.5839694,
    -0.06217745,
    0.018985085,
    -0.0029155314,
    -0.11399212,
    0.34180018,
    0.38483477,
    0.06241387,
    -0.28890932,
    0.11825153,
    -0.3466623,
    0.17485446,
    0.20287298,
    0.3890488,
    -0.036215372,
    0.46085256,
    0.45194423,
    0.42120856,
    -0.021849155,
    0.019964645,
    0.087934755,
    -0.08417687,
    0.2719986,
    0.105430365,
    -0.38356945,
    0.23822773,
    0.23885334,
    0.3386371,
    0.09151632,
    0.3015638,
    0.5727032,
    0.07485627,
    0.48018882,
    0.08769888,
    -0.04850754,
    -0.36953154,
    0.1649228,
    -0.015656859,
    0.30176848,
    0.2974766,
    0.5414424,
    -0.2573507,
    -0.0125634745,
    0.25059485,
    0.0073877983,
    0.32782456,
    0.39748195,
    -0.09214687,
    0.08788801,
    0.24517,
    0.113038145,
    -0.12556338,
    0.14891444,
    -0.3637699,
    0.16135764,
    -0.24549964,
    -0.048727535,
    -0.4137956,
    -0.064132504,
    0.6118638,
    -0.39429182,
    -0.08805484,
    -0.18773945,
    0.2982645,
    0.11784611,
    -0.25582108,
    -0.112161316,
    0.05399874,
    -0.3052031,
    0.4420171,
    -0.3254955,
    0.26785895,
    -0.41021463,
    0.2770481,
    0.4295652,
    0.01817906,
    0.16657665,
    -0.093443215,
    -0.3001354,
    -0.18076026,
    -0.14202349,
    0.17395072,
    -0.115156375,
    -0.09231439,
    0.49168733,
    -0.006198864,
    -0.27805942,
    -0.28996944,
    -0.03537516,
    -0.21342821,
    0.054626282,
    0.1938549,
    -0.08305472,
    -0.106129885,
    0.6057189,
    -0.08438851,
    -0.18504211,
    0.12727177,
    0.17185001,
    -0.4829378,
    0.1772602,
    0.071630456,
    -0.11114428,
    0.41013658,
    0.26588863,
    -0.067258045,
    -0.40872657,
    -0.32674155,
    0.24508859,
    -0.29301828,
    0.24470176,
    0.36417535,
    -0.1254141,
    -0.3829799,
    0.3146924,
    -0.07486214,
    -0.22775005,
    -0.41284937,
    0.38637522,
    -0.40476888,
    0.16482165,
    0.23975624,
    -0.33793283,
    0.3607992,
    0.049563147,
    -0.07752391,
    -0.07130672,
    -0.43954402,
    0.35074002,
    -0.267593,
    0.007313381,
    0.41151854,
    -0.33716315,
    -0.35077953,
    0.11318766,
    -0.3380483,
    0.20592208,
    0.035967678,
    0.39766645,
    -0.21563718,
    -0.1851213,
    0.22164433,
    -0.31974375,
    -0.4119708,
    0.09578852,
    -0.05242302,
    0.23938423,
    -0.23844309,
    0.42013696,
    -0.14793545,
    -0.2336527,
    0.19472612,
    -0.12992854,
    0.32164264,
    0.09721068,
    0.4162817,
    0.016214421,
    0.25102162,
    0.4798254,
    -0.012202531,
    -0.17921817,
    0.1839505,
    0.12687603,
    0.1467754,
    0.11232976,
    0.15392108,
    0.09507806,
    0.0960064,
    0.054173872,
    0.10056898,
    0.08604917,
    0.12875709,
    0.2974497,
    0.13084707,
    0.03666326,
    0.001766446,
    -0.039353047,
    -0.016559495,
    -0.014885214,
    0.016923485,
    0.020065885,
    0.03429328,
    0.04688359,
];

pub struct NnProcessor {
    params: NnParameters,
    input_values: Vec<f32>,
    next_values: Vec<f32>,
    prev_output: Vec<f32>,
    prev_dirs: Vec<Option<f32>>,
    nn: NeuralNetwork,
    simple_physics: f32,

    calc_counts: usize,

    output_diff_loss: f32,
    output_regularization_loss: f32,

    current_track: usize,

    nn_autoencoder: NnProcessorAutoencoder,
}

impl NnProcessor {
    pub fn new_zeros(nn_params: NnParameters, simple_physics: f32, current_track: usize) -> Self {
        Self {
            input_values: vec![0.; nn_params.get_total_input_neurons()],
            next_values: vec![0.; nn_params.pass_next_size],
            prev_output: vec![0.; NnParameters::CAR_OUTPUT_SIZE],
            prev_dirs: vec![None; nn_params.dirs_size],
            params: nn_params.clone(),
            nn: NeuralNetwork::new_zeros(nn_params.get_nn_sizes()),
            simple_physics,
            calc_counts: 0,
            output_diff_loss: 0.,
            output_regularization_loss: 0.,
            current_track,
            nn_autoencoder: NnProcessorAutoencoder::new_zeros(nn_params),
        }
    }

    pub fn new(
        params: &[f32],
        nn_params: NnParameters,
        simple_physics: f32,
        current_track: usize,
    ) -> Self {
        assert_eq!(params.len(), nn_params.get_nns_len());
        let (params_nn, params_other) = params.split_at(nn_params.get_nn_len());
        Self {
            input_values: vec![0.; nn_params.get_total_input_neurons()],
            next_values: vec![0.; nn_params.pass_next_size],
            prev_output: vec![0.; NnParameters::CAR_OUTPUT_SIZE],
            prev_dirs: vec![None; nn_params.dirs_size],
            params: nn_params.clone(),
            nn: NeuralNetwork::new_params(nn_params.get_nn_sizes(), params_nn),
            simple_physics,
            calc_counts: 0,
            output_diff_loss: 0.,
            output_regularization_loss: 0.,
            current_track,
            nn_autoencoder: if nn_params.use_dirs_autoencoder {
                // NnProcessorAutoencoder::new(params_other, nn_params)
                NnProcessorAutoencoder::new(&BEST_AUTOENCODER_21_10_10_10_5, nn_params)
            } else {
                NnProcessorAutoencoder::new_zeros(nn_params)
            },
        }
    }

    pub fn process(
        &mut self,
        time_passed: f32,
        distance_percent: f32,
        dpenalty: f32,
        dirs: &[Option<f32>],
        dirs_second_layer: &[Option<f32>],
        current_segment_f32: f32,
        internals: &InternalCarValues,
        params_sim: &SimulationParameters,
    ) -> CarInput {
        self.calc_counts += 1;

        let mut input_values_iter = self.input_values.iter_mut();

        if self.params.pass_time {
            *input_values_iter.next().unwrap() = time_passed;
        }

        for time_mod in &self.params.pass_time_mods {
            *input_values_iter.next().unwrap() = (time_passed % time_mod) / time_mod;
        }

        if self.params.pass_distance {
            *input_values_iter.next().unwrap() = distance_percent;
        }

        if self.params.pass_dpenalty {
            *input_values_iter.next().unwrap() = dpenalty;
        }

        if self.params.pass_dirs_diff {
            for (prev, current) in self.prev_dirs.iter_mut().zip(dirs.iter()) {
                *input_values_iter.next().unwrap() =
                    convert_dir(&self.params, *prev) - convert_dir(&self.params, *current);
                *prev = *current;
            }
        }

        if self.params.pass_dirs {
            if self.params.use_dirs_autoencoder {
                let values = self.nn_autoencoder.process(dirs);
                for value in values {
                    *input_values_iter.next().unwrap() = *value;
                }
            } else {
                for intersection in dirs {
                    *input_values_iter.next().unwrap() = convert_dir(&self.params, *intersection);
                }
            }
        }

        if self.params.pass_dirs_second_layer {
            for intersection in dirs_second_layer {
                *input_values_iter.next().unwrap() = convert_dir(&self.params, *intersection);
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

        if self.params.pass_simple_physics_value {
            *input_values_iter.next().unwrap() = self.simple_physics;
        }

        if self.params.pass_current_track {
            for i in 0..self.params.max_tracks {
                *input_values_iter.next().unwrap() = (i == self.current_track) as usize as f32;
            }
        }

        if self.params.pass_current_segment {
            for i in 0..self.params.max_segments {
                *input_values_iter.next().unwrap() =
                    if i as f32 <= current_segment_f32 && current_segment_f32 < (i + 1) as f32 {
                        1. + (current_segment_f32 - i as f32)
                    } else {
                        0.
                    };
            }
        }

        assert!(input_values_iter.next().is_none());

        let values = self.nn.calc(&self.input_values);

        let mut output_values_iter = values.iter();

        let prev_output_len = self.prev_output.len();
        for y in &mut self.prev_output {
            let new_value = *output_values_iter.next().unwrap();
            self.output_diff_loss += (*y - new_value).abs() / prev_output_len as f32;
            *y = new_value;
            if new_value.abs() > 10. {
                self.output_regularization_loss += (new_value.abs() - 10.) / prev_output_len as f32;
            }
        }

        if self.params.pass_next_size > 0 {
            for y in &mut self.next_values {
                *y = *output_values_iter.next().unwrap();
            }
        }

        assert!(output_values_iter.next().is_none());

        if params_sim.simulation_enable_random_nn_output {
            let mut values = values.iter().copied().collect::<Vec<_>>();
            let mut rng = thread_rng();
            for i in 0..NnParameters::CAR_OUTPUT_SIZE {
                values[i] += rng.gen_range(
                    -params_sim.simulation_random_output_range
                        ..params_sim.simulation_random_output_range,
                );
            }
        }

        CarInput::from_f32(&values[0..NnParameters::CAR_OUTPUT_SIZE])
    }

    pub fn get_autoencoder_loss(&self) -> f32 {
        self.nn_autoencoder.get_autoencoder_loss()
    }

    pub fn get_regularization_loss(&self) -> f32 {
        self.nn.get_regularization() + self.nn_autoencoder.get_regularization_loss()
    }

    pub fn get_output_diff_loss(&self) -> f32 {
        if self.calc_counts == 0 {
            0.
        } else {
            self.output_diff_loss / self.calc_counts as f32
        }
    }

    pub fn get_output_regularization_loss(&self) -> f32 {
        if self.calc_counts == 0 {
            0.
        } else {
            self.output_regularization_loss / self.calc_counts as f32
        }
    }

    pub fn set_current_track(&mut self, current_track: usize) {
        self.current_track = current_track;
    }

    pub fn reset(&mut self) {
        self.next_values.iter_mut().for_each(|x| *x = 0.);
        self.prev_output.iter_mut().for_each(|x| *x = 0.);
        self.nn_autoencoder.reset();
        self.prev_dirs.iter_mut().for_each(|x| *x = None);
        self.calc_counts = 0;
        self.output_diff_loss = 0.;
        self.output_regularization_loss = 0.;
    }
}

fn update_two_nearest(
    current: (Option<f32>, Option<f32>),
    new_value: Option<f32>,
) -> (Option<f32>, Option<f32>) {
    if let Some(dist) = new_value {
        match current {
            (None, None) => (Some(dist), None),
            (Some(nearest), None) => {
                if dist < nearest {
                    (Some(dist), Some(nearest))
                } else {
                    (Some(nearest), Some(dist))
                }
            }
            (Some(nearest), Some(second)) => {
                if dist < nearest {
                    (Some(dist), Some(nearest))
                } else if dist < second {
                    (Some(nearest), Some(dist))
                } else {
                    (Some(nearest), Some(second))
                }
            }
            (None, Some(_)) => unreachable!(), // This case shouldn't be possible
        }
    } else {
        current
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
    dirs_values_second_layer: Vec<Option<f32>>,
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
                .map(|i| {
                    (i as f32 / (params.nn.dirs_size - 1) as f32 - 0.5)
                        * TAU
                        * params.nn.view_angle_ratio
                })
                .map(|t| rotate_around_origin(pos2(1., 0.), t))
                .collect(),
            dirs_values: (0..params.nn.dirs_size).map(|_| None).collect(),
            dirs_values_second_layer: (0..params.nn.dirs_size).map(|_| None).collect(),
            walls,
            reward_path_processor: RewardPathProcessor::new(rewards),
        }
    }

    pub fn reset(&mut self) {
        self.car = Default::default();
        self.penalty = 0.;
        self.prev_penalty = 0.;
        self.reward = 0.;
        self.time_passed = 0.;
        self.dirs_values.iter_mut().for_each(|x| *x = None);
        self.dirs_values_second_layer
            .iter_mut()
            .for_each(|x| *x = None);
        self.reward_path_processor.reset();
    }

    pub fn step(
        &mut self,
        params_phys: &PhysicsParameters,
        params_sim: &SimulationParameters,
        observe_distance: &mut impl FnMut(Pos2, Pos2, f32, Option<f32>),
        get_input: &mut impl FnMut(
            f32,
            f32,
            f32,
            &[Option<f32>],
            &[Option<f32>],
            &InternalCarValues,
            f32,
        ) -> CarInput,
        drift_observer: &mut impl FnMut(usize, Vec2, f32),
        observe_car_forces: &mut impl FnMut(&Car),
    ) -> bool {
        for ((dir, value), value_second_layer) in self
            .dirs
            .iter()
            .zip(self.dirs_values.iter_mut())
            .zip(self.dirs_values_second_layer.iter_mut())
        {
            let dir_pos = self.car.from_local_coordinates(*dir);
            let origin = self.car.get_center();
            let mut intersections = (None, None);
            for wall in &self.walls {
                intersections =
                    update_two_nearest(intersections, wall.intersect_ray(origin, dir_pos));
            }
            if let Some(t) = intersections.0 {
                observe_distance(origin, dir_pos, t, intersections.1);
            }
            *value = intersections.0;
            *value_second_layer = intersections.1;
        }

        let input = get_input(
            self.time_passed,
            self.reward_path_processor.distance_percent(),
            self.penalty - self.prev_penalty,
            &self.dirs_values,
            &self.dirs_values_second_layer,
            &self.car.get_internal_values(),
            self.reward_path_processor.get_current_segment_f32(),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackEvaluation {
    name: String,
    penalty: f32,
    reward: f32,
    early_finish_percent: f32,
    distance_percent: f32,
    all_acquired: bool,
    simple_physics: f32,
    simple_physics_raw: f32,
    autoencoder_loss: f32,
    regularization_loss: f32,
    output_diff_loss: f32,
    output_regularization_loss: f32,
}

impl std::fmt::Display for TrackEvaluation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} | d {:>5.1}% | e {:>5.1}% | penalty {:>6.1} | {}",
            if self.all_acquired { "✅" } else { "❌" },
            self.distance_percent * 100.,
            self.early_finish_percent * 100.,
            self.penalty,
            // self.reward,
            // self.autoencoder_loss,
            // self.regularization_loss,
            // self.output_diff_loss,
            // self.output_regularization_loss,
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
        let all_acquired = evals.iter().all(|x| x.all_acquired);
        let len = evals.len() as f32;
        let hard_physics_reward = if all_acquired {
            0.01 + (1.0 - evals[0].simple_physics)
        } else {
            evals[0].simple_physics - 1.
        } * len;
        let simple_physics_raw = evals[0].simple_physics_raw;
        let raw_penalty_start = 5.;
        let simple_physics_raw_penalty = if simple_physics_raw.abs() < raw_penalty_start {
            1.
        } else {
            1. / (1. + (simple_physics_raw.abs() - raw_penalty_start))
        };
        let tracks_sum = evals
            .iter()
            .map(|x| x.all_acquired as usize as f32)
            .sum::<f32>();
        // let regularization_sum = evals.iter().map(|x| x.regularization_loss).sum::<f32>();
        let regularization_sum = evals
            .iter()
            .map(|x| x.output_regularization_loss)
            .sum::<f32>()
            / evals.len() as f32;
        let penalty_level_1 = evals
            .iter()
            .map(|x| (x.penalty < 15.) as usize as f32)
            .sum::<f32>();
        let penalty_level_2 = evals
            .iter()
            .map(|x| (x.penalty < 5.) as usize as f32)
            .sum::<f32>();
        let penalty_level_3 = evals
            .iter()
            .map(|x| (x.penalty < 2.) as usize as f32)
            .sum::<f32>();
        let penalty_level_4 = evals
            .iter()
            .map(|x| (x.penalty == 0.) as usize as f32)
            .sum::<f32>();
        let penalty_levels_len = 4. * len;
        let penalty_levels =
            (penalty_level_1 + penalty_level_2 + penalty_level_3 + penalty_level_4)
                / penalty_levels_len;
        let smooth_metrics = evals.iter().map(|x| x.to_f32_second_way()).sum::<f32>();
        return (tracks_sum
            + if params.rewards_second_way_penalty {
                penalty_levels + smooth_metrics / penalty_levels_len
            } else {
                smooth_metrics / len
            }
            + if params.evolve_simple_physics {
                hard_physics_reward + simple_physics_raw_penalty
            } else {
                0.
            }
            + if params.simulation_use_output_regularization {
                1. / (1. + regularization_sum.sqrt().sqrt())
            } else {
                0.
            })
            / len;
    } else if params.rewards_early_finish_zero_penalty {
        let len = evals.len() as f32;
        let tracks_sum = evals
            .iter()
            .map(|x| x.all_acquired as usize as f32 + x.distance_percent)
            .sum::<f32>();
        let penalty_sum = evals
            .iter()
            .map(|x| (x.penalty == 0.) as usize as f32 + 1. / (1. + x.penalty))
            .sum::<f32>();
        let early_finish_reward = evals
            .iter()
            .map(|x| {
                if x.penalty == 0. {
                    x.early_finish_percent * 2.
                } else {
                    0.
                }
            })
            .sum::<f32>();
        (tracks_sum + penalty_sum + early_finish_reward) / len
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
                        all_acquired: true,
                        simple_physics: x.simple_physics,
                        simple_physics_raw: 0.,
                        autoencoder_loss: 0.,
                        regularization_loss: 0.,
                        output_diff_loss: 0.,
                        output_regularization_loss: 0.,
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
            + if params.evolve_simple_physics {
                evals[0].simple_physics * params.hard_physics_reward.value
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
            - self.penalty * params.eval_penalty.value
    }

    // from 0 to 1
    fn to_f32_second_way(&self) -> f32 {
        (1. / (1. + self.penalty)
            + (1. - 1. / (1. + self.reward))
            + self.early_finish_percent
            + self.distance_percent
            + self.autoencoder_loss)
            / 6.
    }
}

pub fn patch_params_sim(params: &[f32], params_sim: &SimulationParameters) -> SimulationParameters {
    let nn_len = params_sim.nn.get_nns_len();
    let (_, other_params) = params.split_at(nn_len);
    assert_eq!(other_params.len(), OTHER_PARAMS_SIZE);
    let mut params_sim = params_sim.clone();
    if params_sim.evolve_simple_physics {
        params_sim.simulation_simple_physics = sigmoid(other_params[0]);
    }
    params_sim
}

pub fn eval_nn(
    params: &[f32],
    params_phys: &PhysicsParameters,
    params_sim: &SimulationParameters,
) -> Vec<TrackEvaluation> {
    let nn_len = params_sim.nn.get_nns_len();
    let (nn_params, other_params) = params.split_at(nn_len);
    assert_eq!(other_params.len(), OTHER_PARAMS_SIZE);

    let params_sim = patch_params_sim(params, params_sim);
    let params_phys = params_sim.patch_physics_parameters(params_phys.clone());

    let mut result: Vec<TrackEvaluation> = Default::default();
    let tracks: Vec<Track> = get_all_tracks()
        .into_iter()
        .map(|x| points_storage_to_track(x))
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

    for (
        track_no,
        Track {
            name,
            walls,
            rewards,
        },
    ) in tracks.into_iter().enumerate()
    {
        let max = if params_sim.mutate_car_enable {
            params_sim.mutate_car_count
        } else {
            1
        };
        for i in 0..max {
            let physics_max = if params_sim.eval_calc_all_physics {
                params_sim.eval_calc_all_physics_count
            } else {
                1
            };
            let default = (0.0, 0.9, 0.4, 0.5, 0.1);
            for (simple_physics, traction, acceleration_ratio, friction_coef, turn_speed) in (0
                ..physics_max)
                .map(|i| {
                    if i == physics_max - 1 {
                        1.0
                    } else {
                        i as f32 / (physics_max - 1) as f32
                    }
                    .sqrt()
                    .sqrt()
                })
                .map(|s| {
                    let mut x = default;
                    x.0 = s;
                    x
                })
                .chain(if params_sim.eval_add_other_physics {
                    vec![
                        // traction
                        // { let mut x = default; x.1 = 0.5; x },
                        // { let mut x = default; x.1 = 0.15; x },

                        // acceleration ratio
                        {
                            let mut x = default;
                            x.2 = 0.8;
                            x
                        },
                        {
                            let mut x = default;
                            x.2 = 1.0;
                            x
                        },
                        // friction
                        // { let mut x = default; x.3 = 0.0; x },
                        // { let mut x = default; x.3 = 1.0; x },

                        // turn speed
                        // { let mut x = default; x.4 = 0.3; x },

                        // combined
                        // { let mut x = default; x.2 = 1.0; x.1 = 1.0; x.3 = 0.15; x },
                    ]
                    .into_iter()
                } else {
                    vec![].into_iter()
                })
            {
                let mut params_sim = params_sim.clone();
                let mut params_phys = params_phys.clone();
                if params_sim.eval_calc_all_physics || params_sim.eval_add_other_physics {
                    // params_sim.simulation_simple_physics = simple_physics;
                    params_sim.simulation_simple_physics = 0.;
                    params_phys = params_sim.patch_physics_parameters(params_phys.clone());
                }

                if params_sim.eval_add_other_physics {
                    params_phys.traction_coefficient = traction;
                    params_phys.acceleration_ratio = acceleration_ratio;
                    params_phys.friction_coefficient = friction_coef;
                    params_phys.wheel_turn_per_time = turn_speed;
                }

                let car = {
                    let result: Car = Default::default();
                    if i != 0 && params_sim.mutate_car_enable {
                        mutate_car(result, &params_sim)
                    } else {
                        result
                    }
                };
                let mut nn_processor = NnProcessor::new(
                    nn_params,
                    params_sim.nn.clone(),
                    params_sim.simulation_simple_physics,
                    track_no,
                );
                let mut simulation =
                    CarSimulation::new(car, walls.clone(), rewards.clone(), &params_sim);

                let mut early_finish_percent = 0.;
                let steps_quota = params_sim.simulation_steps_quota;
                for i in 0..steps_quota {
                    if simulation.step(
                        &params_phys,
                        &params_sim,
                        &mut |_, _, _, _| (),
                        &mut |time,
                              dist,
                              dpenalty,
                              dirs,
                              dirs_second_layer,
                              internals,
                              current_segment_f32| {
                            nn_processor.process(
                                time,
                                dist,
                                dpenalty,
                                dirs,
                                dirs_second_layer,
                                current_segment_f32,
                                internals,
                                &params_sim,
                            )
                        },
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
                    name: name.clone()
                        + &if i != 0 {
                            format!(":{i}")
                        } else {
                            Default::default()
                        } + &if params_sim.eval_calc_all_physics || params_sim.eval_add_other_physics {
                            format!(":simple{:.2}", simple_physics)
                        } else {
                            Default::default()
                        } + &if params_sim.eval_add_other_physics {
                            format!(":t{traction:.2}:a{acceleration_ratio:.2}:f{friction_coef:.2}:t{turn_speed:.2}")
                        } else {
                            Default::default()
                        },
                    penalty: simulation.penalty,
                    reward: simulation.reward,
                    early_finish_percent,
                    distance_percent: simulation.reward_path_processor.distance_percent(),
                    all_acquired: simulation.reward_path_processor.all_acquired(),
                    simple_physics: params_sim.simulation_simple_physics,
                    simple_physics_raw: other_params[0],
                    autoencoder_loss: nn_processor.get_autoencoder_loss(),
                    regularization_loss: nn_processor.get_regularization_loss(),
                    output_diff_loss: nn_processor.get_output_diff_loss(),
                    output_regularization_loss: nn_processor.get_output_regularization_loss(),
                });
            }
        }
    }
    result
}

fn from_f64_to_f32_vec(pos: &[f64]) -> Vec<f32> {
    pos.iter().copied().map(|x| x as f32).collect()
}

fn from_dvector_to_f32_vec(pos: &DVector<f64>) -> Vec<f32> {
    pos.iter().copied().map(|x| x as f32).collect()
}

pub fn evolve_by_differential_evolution(params_sim: &SimulationParameters) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let params_phys = params_sim.patch_physics_parameters(PhysicsParameters::default());
    let nn_len = params_sim.nn.get_nns_len();

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
        let evals = eval_nn(pos, &params_phys, params_sim);
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
            print_evals(&eval_nn(vec, &params_phys, params_sim));
        }
        if pos % 1000 == 0 && pos != 0 {
            let (_, vec) = de.best().unwrap();
            println!("(vec!{:?}, vec!{:?})", nn_sizes, vec);
        }
    }
    // show the result
    let (cost, pos) = de.best().unwrap();
    println!("cost: {}", cost);
    print_evals(&eval_nn(pos, &params_phys, params_sim));
    println!("(vec!{:?}, vec!{:?})", nn_sizes, pos);
}

pub fn evolve_by_differential_evolution_custom(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    population_size: usize,
    generations_count: usize,
) -> Vec<EvolveOutputEachStep> {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = params_sim.nn.get_nns_len();
    let input_done: Vec<(f32, f32)> = vec![(-10., 10.); nn_len + OTHER_PARAMS_SIZE];

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut de = self_adaptive_de(input_done, |pos| {
        let evals = eval_nn(pos, &params_phys, params_sim);
        -sum_evals(&evals, params_sim)
    });

    let mut true_params_sim = SimulationParameters::true_metric(&params_sim);

    for pos in 0..generations_count {
        let now = Instant::now();
        for _ in 0..population_size {
            let _ = de.iter().next().unwrap();
        }
        let (value, point) = de.best().unwrap();

        true_params_sim = patch_params_sim(point, &true_params_sim);

        let evals = eval_nn(point, params_phys, &params_sim);
        let true_evals = eval_nn(point, params_phys, &true_params_sim);

        result.push(EvolveOutputEachStep {
            nn: point.iter().copied().collect(),
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
    let nn_len = params_sim.nn.get_nns_len();
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
        .sample_mean(params_sim.evolution_sample_mean)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(&from_dvector_to_f32_vec(&x), &params_phys, &params_sim);
            -sum_evals(&evals, &params_sim) as f64
        })
        .unwrap();
    let now = Instant::now();
    for pos in 0..500 {
        if ONE_THREADED {
            let _ = state.next();
        } else {
            let _ = state.next_parallel();
        }
        let cmaes::Individual { point, value } = state.overall_best_individual().unwrap();
        if pos == 0 {
            print_evals(&eval_nn(
                &from_dvector_to_f32_vec(&point),
                &params_phys,
                &params_sim,
            ));
        }
        println!("{pos}. {value}, {:?}", now.elapsed() / (pos + 1) as u32);
        if pos % 10 == 0 && pos != 0 {
            print_evals(&eval_nn(
                &from_dvector_to_f32_vec(&point),
                &params_phys,
                &params_sim,
            ));
        }
        if pos % 10 == 0 && pos != 0 {
            println!(
                "(vec!{:?}, vec!{:?})",
                nn_sizes,
                &from_dvector_to_f32_vec(&point)
            );
        }
    }
    let solution = state.overall_best_individual().unwrap().point.clone();

    let params_phys_clone = params_phys.clone();
    params_phys = params_sim.patch_physics_parameters(params_phys_clone);
    input_done = solution.iter().copied().collect();
    println!("(vec!{:?}, vec!{:?})", nn_sizes, solution.as_slice());
    let evals = eval_nn(
        &from_dvector_to_f32_vec(&&solution),
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
    step_size: Option<f64>,
    stop_at: Option<f32>,
) -> Vec<EvolveOutputEachStep> {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = params_sim.nn.get_nns_len();
    let input_done: Vec<f64> = nn_input.iter().map(|x| *x as f64).collect();
    assert_eq!(input_done.len(), nn_len + OTHER_PARAMS_SIZE);

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut state = cmaes::options::CMAESOptions::new(input_done, step_size.unwrap_or(10.))
        .population_size(population_size)
        .cm(params_sim.evolution_learning_rate)
        .sample_mean(params_sim.evolution_sample_mean)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(&from_dvector_to_f32_vec(&x), params_phys, params_sim);
            -sum_evals(&evals, params_sim) as f64
        })
        .unwrap();

    let mut true_params_sim = SimulationParameters::true_metric(params_sim);

    for pos in 0..generations_count {
        let now = Instant::now();
        if ONE_THREADED {
            let _ = state.next();
        } else {
            let _ = state.next_parallel();
        }
        let cmaes::Individual { point, value } = state.overall_best_individual().unwrap();

        let nn = from_dvector_to_f32_vec(&point);
        true_params_sim = patch_params_sim(&nn, &true_params_sim);

        let evals = eval_nn(&nn, params_phys, &params_sim);
        let evals_cost = sum_evals(&evals, params_sim);
        let true_evals = eval_nn(&nn, params_phys, &true_params_sim);
        let true_evals_cost = sum_evals(&true_evals, &true_params_sim);

        result.push(EvolveOutputEachStep {
            nn: nn,
            evals_cost,
            true_evals_cost,
            evals: evals.clone(),
            true_evals,
        });

        if PRINT && (((pos % 10 == 0) && PRINT_EVERY_10) || !PRINT_EVERY_10) {
            println!("{pos}. {evals_cost}");
            if PRINT_EVALS
                && (((pos % 10 == 0) && PRINT_EVERY_10_ONLY_EVALS) || !PRINT_EVERY_10_ONLY_EVALS)
            {
                for i in &evals {
                    println!("{}", i);
                }
            }
        }
        if stop_at.map(|stop_at| evals_cost > stop_at).unwrap_or(false) {
            if PRINT {
                println!(
                    "Break, because current value {} is bigger than stop_at value {}",
                    evals_cost,
                    stop_at.unwrap()
                );
            }
            break;
        }
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
                &from_f64_to_f32_vec(&p.to_vec()),
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
                    &from_f64_to_f32_vec(&x.to_vec()),
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

    let nn_len = params_sim.nn.get_nns_len();
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
        &from_f64_to_f32_vec(&input_vec.to_vec()),
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
    best_start: Option<Vec<f32>>,
    stop_at: Option<f32>,
) -> Vec<EvolveOutputEachStep> {
    use argmin::core::PopulationState;
    use argmin::core::Problem;
    use argmin::core::Solver;
    use argmin::core::State;
    use argmin::solver::conjugategradient::beta::*;
    use argmin::solver::conjugategradient::*;
    use argmin::solver::gradientdescent::SteepestDescent;
    use argmin::solver::neldermead::*;
    use argmin::solver::particleswarm::Particle;
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
    }

    impl CostFunction for NnStruct {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let evals = eval_nn(
                &from_f64_to_f32_vec(&p.to_vec()),
                &self.params_phys,
                &self.params_sim,
            );
            Ok(-sum_evals(&evals, &self.params_sim) as f64)
        }
    }

    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = params_sim.nn.get_nns_len();
    let cost = NnStruct {
        params_sim: params_sim.clone(),
        params_phys: params_phys.clone(),
    };

    let min_param: Array1<f64> = (0..nn_len + OTHER_PARAMS_SIZE).map(|_| -1000.).collect();
    let max_param: Array1<f64> = (0..nn_len + OTHER_PARAMS_SIZE).map(|_| 1000.).collect();
    let mut solver = ParticleSwarm::new((min_param, max_param), population_size);
    let mut problem = Problem::new(cost);
    let mut state = PopulationState::new();
    state = solver.init(&mut problem, state).unwrap().0;

    if let Some(best_start) = best_start {
        let best_start = Array1::<f64>::from_iter(best_start.iter().map(|x| *x as f64));
        let best_start_velocity = Array1::<f64>::from_iter((0..best_start.len()).map(|x| 0.1));
        let best_start_cost = problem.cost(&best_start).unwrap();
        if state.cost > best_start_cost {
            state = state
                .individual(Particle::new(
                    best_start,
                    best_start_cost,
                    best_start_velocity,
                ))
                .cost(best_start_cost);
        } else {
            println!("!!! Your provided solution have cost {}, when random solution have cost {}, it's not used.", best_start_cost, state.cost);
        }
    }

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut true_params_sim = SimulationParameters::true_metric(&params_sim);

    for pos in 0..generations_count {
        let now = Instant::now();
        state = solver.next_iter(&mut problem, state).unwrap().0;
        let (point, value) = (&state.individual.as_ref().unwrap().position, state.cost);

        let nn = from_f64_to_f32_vec(point.as_slice().unwrap());
        true_params_sim = patch_params_sim(&nn, &true_params_sim);

        let evals = eval_nn(&nn, params_phys, &params_sim);
        let evals_cost = sum_evals(&evals, params_sim);
        let true_evals = eval_nn(&nn, params_phys, &true_params_sim);
        let true_evals_cost = sum_evals(&true_evals, &true_params_sim);

        result.push(EvolveOutputEachStep {
            nn: nn,
            evals_cost,
            true_evals_cost,
            evals,
            true_evals,
        });

        if PRINT {
            println!("{pos}. {evals_cost}");
        }
        if stop_at.map(|stop_at| evals_cost > stop_at).unwrap_or(false) {
            if PRINT {
                println!(
                    "Break, because current value {} is bigger than stop_at value {}",
                    evals_cost,
                    stop_at.unwrap()
                );
            }
            break;
        }
    }
    result
}

fn calc_gradient(params_sim: &SimulationParameters) {
    let nn_len = params_sim.nn.get_nns_len();
    let input_done: Vec<f32> = include!("nn.data").1;
    assert_eq!(input_done.len(), nn_len);
    let time = Instant::now();
    let mut count = 0;
    dbg!(forward_diff_vec(&input_done, &mut |x| {
        count += 1;
        let evals = eval_nn(
            x,
            &params_sim.patch_physics_parameters(PhysicsParameters::default()),
            params_sim,
        );
        -sum_evals(&evals, params_sim)
    }));
    dbg!(count);
    dbg!(time.elapsed());
}

// fn mutate_nn() {
//     let nn = from_slice_to_nn(include!("nn.data").0, &include!("nn.data").1);
//     let mut nn_uno = nn.to_unoptimized();
//     nn_uno.add_hidden_layer(1);
//     let nn = nn_uno.to_optimized();
//     println!("(vec!{:?}, vec!{:?})", nn.get_sizes(), nn.get_values());
// }

impl Default for NnParameters {
    fn default() -> Self {
        Self {
            pass_time: false,
            pass_distance: false,
            pass_dpenalty: true,
            pass_internals: false,
            pass_prev_output: false,
            pass_simple_physics_value: false,
            pass_next_size: 0,
            hidden_layers: vec![6],
            inv_distance: true,
            inv_distance_coef: 20.,
            inv_distance_pow: 0.5,
            view_angle_ratio: 2. / 6.,

            dirs_size: 21,
            pass_dirs: true,
            pass_dirs_diff: false,
            pass_dirs_second_layer: false,

            use_dirs_autoencoder: false,
            autoencoder_hidden_layers: vec![6],
            autoencoder_exits: 5,

            pass_current_track: false,
            max_tracks: 12,

            pass_current_segment: false,
            max_segments: 23,

            pass_time_mods: vec![],
        }
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        let all_tracks: BTreeMap<String, bool> =
            get_all_tracks().into_iter().map(|x| (x.0, false)).collect();
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
            rewards_early_finish_zero_penalty: false,

            simulation_stop_penalty: Clamped::new(20., 0., 50.),
            simulation_scale_reward_to_time: false,
            simulation_steps_quota: 3500,
            simulation_simple_physics: 0.0,
            simulation_enable_random_nn_output: false,
            simulation_random_output_range: 0.1,
            simulation_use_output_regularization: false,

            evolution_sample_mean: false,
            evolution_generation_count: 100,
            evolution_population_size: 30,
            evolution_learning_rate: 1.0,
            evolution_distance_to_solution: 10.,

            eval_skip_passed_tracks: false,
            eval_add_min_distance: false,
            eval_reward: Clamped::new(10., 0., 5000.),
            eval_early_finish: Clamped::new(10000., 0., 10000.),
            eval_distance: Clamped::new(1000., 0., 5000.),
            eval_acquired: Clamped::new(1000., 0., 5000.),
            eval_penalty: Clamped::new(20., 0., 5000.),
            eval_calc_all_physics: false,
            eval_calc_all_physics_count: 5,
            eval_add_other_physics: false,

            evolve_simple_physics: false,
            hard_physics_reward: Clamped::new(10000., 0., 10000.),

            nn: Default::default(),
        }
    }
}

impl SimulationParameters {
    pub fn true_metric(other: &SimulationParameters) -> Self {
        let mut result = Self::default();
        result.enable_all_tracks();
        result.disable_track("straight_45"); // todo: удалить это и снова проверить
        result.rewards_second_way = true;
        result.mutate_car_enable = false;
        result.simulation_stop_penalty.value = 100.;

        result.simulation_simple_physics = other.simulation_simple_physics;
        result.evolve_simple_physics = other.evolve_simple_physics;

        result.nn = other.nn.clone();

        // result.evolve_simple_physics = true;

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
const ONE_THREADED: bool = false;
const PRINT: bool = true;
const PRINT_EVERY_10: bool = false;
const PRINT_EVERY_10_ONLY_EVALS: bool = true;
const PRINT_EVALS: bool = true;

const RUNS_COUNT: usize = 30;
const POPULATION_SIZE: usize = 30;
const GENERATIONS_COUNT: usize = 100;
pub const OTHER_PARAMS_SIZE: usize = 1;

fn save_json_to_file<T: Serialize>(t: &T, name: &str) {
    use std::io::Write;
    let mut file = std::fs::File::create(&name).unwrap();
    let json = serde_json::to_string(&t).unwrap();
    write!(file, "{}", json).unwrap();
}

fn save_runs(result: Vec<Vec<EvolveOutputEachStep>>, name: &str) {
    save_json_to_file(&result, &format!("graphs/{name}.json"));
}

fn evolve_simple_physics(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    input: &[f32],
    population_size: usize,
    generations_count_main: usize,
    generations_count_adapt: usize,
) -> Vec<EvolveOutputEachStep> {
    let mut params_sim = params_sim.clone();
    let step = 0.02;

    let start = Instant::now();

    params_sim.simulation_simple_physics = 0.98;
    let mut result = evolve_by_cma_es_custom(
        &params_sim,
        &params_phys,
        &input,
        population_size,
        generations_count_main,
        None,
        None,
    );
    let best_simple_cost = result.last().unwrap().evals_cost;
    while params_sim.simulation_simple_physics > 0. {
        params_sim.simulation_simple_physics -= step;
        if params_sim.simulation_simple_physics < 0. {
            params_sim.simulation_simple_physics = 0.;
        }
        if PRINT {
            println!(
                "Use simple physics value: {}",
                params_sim.simulation_simple_physics
            );
        }
        let evals = eval_nn(&result.last().unwrap().nn, params_phys, &params_sim);
        let evals_cost = sum_evals(&evals, &params_sim);
        if evals_cost < best_simple_cost {
            let result2 = evolve_by_cma_es_custom(
                &params_sim,
                &params_phys,
                &result.last().unwrap().nn,
                population_size,
                generations_count_adapt,
                None,
                Some(best_simple_cost),
            );
            result.extend(result2);
        } else {
            if PRINT {
                println!("Skip evolution entirely.");
            }
        }
    }
    result.extend(evolve_by_cma_es_custom(
        &params_sim,
        &params_phys,
        &result.last().unwrap().nn,
        population_size,
        generations_count_main,
        None,
        None,
    ));
    println!(
        "FINISH! Time: {:?}, score: {}",
        start.elapsed(),
        result.last().unwrap().evals_cost
    );
    result
}

fn test_params_sim_evolve_simple(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    name: &str,
) {
    test_params_sim_fn(
        params_sim,
        params_phys,
        name,
        |params_sim, params_phys, input, population_size, generations_count| {
            evolve_simple_physics(
                &params_sim,
                params_phys,
                &input,
                population_size,
                generations_count,
                10,
            )
        },
    )
}

fn random_input_by_len(len: usize, limit: f32) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..len)
        .map(|_| rng.gen_range(-limit..limit))
        .collect::<Vec<f32>>()
}

fn random_input(params_sim: &SimulationParameters) -> Vec<f32> {
    random_input_by_len(params_sim.nn.get_nns_len() + OTHER_PARAMS_SIZE, 10.)
}

fn test_params_sim_fn<
    F: Fn(
            &SimulationParameters,
            &PhysicsParameters,
            &[f32],
            usize,
            usize,
        ) -> Vec<EvolveOutputEachStep>
        + std::marker::Sync,
>(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    name: &str,
    f: F,
) {
    let now = Instant::now();
    let map_lambda = |_| {
        f(
            params_sim,
            params_phys,
            &random_input(params_sim),
            POPULATION_SIZE,
            GENERATIONS_COUNT,
        )
    };

    let result: Vec<Vec<EvolveOutputEachStep>> = if ONE_THREADED {
        (0..RUNS_COUNT).into_iter().map(map_lambda).collect()
    } else {
        (0..RUNS_COUNT).into_par_iter().map(map_lambda).collect()
    };

    save_runs(result, name);

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

fn test_params_sim(params_sim: &SimulationParameters, params_phys: &PhysicsParameters, name: &str) {
    let now = Instant::now();
    let map_lambda = |_| {
        evolve_by_cma_es_custom(
            &params_sim,
            &params_phys,
            &random_input(params_sim),
            params_sim.evolution_population_size,
            params_sim.evolution_generation_count,
            Some(params_sim.evolution_distance_to_solution),
            None,
        )
    };

    let result: Vec<Vec<EvolveOutputEachStep>> = if ONE_THREADED {
        (0..RUNS_COUNT).into_iter().map(map_lambda).collect()
    } else {
        (0..RUNS_COUNT).into_par_iter().map(map_lambda).collect()
    };

    save_runs(result, name);

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

fn test_params_sim_differential_evolution(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    name: &str,
) {
    let now = Instant::now();
    let result: Vec<Vec<EvolveOutputEachStep>> = (0..RUNS_COUNT)
        .into_par_iter()
        .map(|_| {
            evolve_by_differential_evolution_custom(
                &params_sim,
                &params_phys,
                POPULATION_SIZE,
                GENERATIONS_COUNT,
            )
        })
        .collect();

    save_runs(result, name);

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

fn test_params_sim_particle_swarm(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    name: &str,
) {
    let now = Instant::now();
    let result: Vec<Vec<EvolveOutputEachStep>> = (0..RUNS_COUNT)
        .into_par_iter()
        .map(|_| {
            evolve_by_particle_swarm_custom(
                &params_sim,
                &params_phys,
                POPULATION_SIZE,
                GENERATIONS_COUNT,
                None,
                None,
            )
        })
        .collect();

    save_runs(result, name);

    println!("For `{}`, time: {:?}", name, now.elapsed());
}

#[derive(Default, Serialize, Deserialize, Clone)]
struct Dirs(Vec<Option<f32>>);
#[derive(Default, Serialize, Deserialize, Clone)]
struct TrackDirs(Vec<Dirs>);
#[derive(Default, Serialize, Deserialize, Clone)]
struct AllTrackDirs(Vec<TrackDirs>);

fn eval_tracks_dirs(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    input: &[f32],
) -> AllTrackDirs {
    let evals = eval_nn(&input, &params_phys, &params_sim);
    for i in evals {
        println!("{i}");
    }

    let nn_len = params_sim.nn.get_nn_len();
    let nn_params = &input[..nn_len];

    let tracks: Vec<Track> = get_all_tracks()
        .into_iter()
        .map(|x| points_storage_to_track(x))
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

    let mut result = AllTrackDirs(Default::default());

    for (
        track_no,
        Track {
            name,
            walls,
            rewards,
        },
    ) in tracks.into_iter().enumerate()
    {
        result.0.push(TrackDirs(Default::default()));
        let mut nn_processor = NnProcessor::new(
            nn_params,
            params_sim.nn.clone(),
            params_sim.simulation_simple_physics,
            track_no,
        );
        let mut simulation = CarSimulation::new(
            Default::default(),
            walls.clone(),
            rewards.clone(),
            &params_sim,
        );

        for i in 0..params_sim.simulation_steps_quota {
            if simulation.step(
                &params_phys,
                &params_sim,
                &mut |_, _, _, _| (),
                &mut |time,
                      dist,
                      dpenalty,
                      dirs,
                      dirs_second_layer,
                      internals,
                      current_segment_f32| {
                    result.0.last_mut().unwrap().0.push(Dirs(dirs.to_vec()));
                    nn_processor.process(
                        time,
                        dist,
                        dpenalty,
                        dirs,
                        dirs_second_layer,
                        current_segment_f32,
                        internals,
                        &params_sim,
                    )
                },
                &mut |_, _, _| (),
                &mut |_| (),
            ) {
                break;
            }

            if simulation.reward_path_processor.all_acquired() {
                break;
            }
        }
    }

    result
}

fn eval_ae(params: &[f32], all_tracks_dirs: &AllTrackDirs, nn_params: &NnParameters) -> f32 {
    let mut loss = 0.;
    let mut ae_processor = NnProcessorAutoencoder::new(&params, nn_params.clone());
    for TrackDirs(track_dirs) in &all_tracks_dirs.0 {
        ae_processor.reset();
        for Dirs(dirs) in track_dirs {
            ae_processor.process(&dirs);
        }
        loss += ae_processor.get_autoencoder_loss();
    }
    loss / all_tracks_dirs.0.len() as f32
}

fn evolve_by_cma_es_custom_ae(
    input: &[f32],
    all_tracks_dirs: &AllTrackDirs,
    nn_params: &NnParameters,
    population_size: usize,
    generations_count: usize,
) -> Vec<f32> {
    let input_done: Vec<f64> = input.iter().map(|x| *x as f64).collect();

    let mut state = cmaes::options::CMAESOptions::new(input_done, 10.)
        .population_size(population_size)
        .build(|x: &DVector<f64>| -> f64 {
            -eval_ae(&from_dvector_to_f32_vec(&x), all_tracks_dirs, nn_params) as f64
        })
        .unwrap();

    for pos in 0..generations_count {
        let now = Instant::now();
        if ONE_THREADED {
            let _ = state.next();
        } else {
            let _ = state.next_parallel();
        }
        let cmaes::Individual { point, value } = state.overall_best_individual().unwrap();

        if PRINT {
            println!("{pos}. {value:.4}, time: {:?}", now.elapsed());
        }
    }

    from_dvector_to_f32_vec(&state.overall_best_individual().unwrap().point)
}

#[allow(unused_imports, unused_variables)]
fn evolve_by_bfgs_autoencoder(
    input: &[f32],
    all_tracks_dirs: &AllTrackDirs,
    nn_params: &NnParameters,
) {
    use argmin::solver::conjugategradient::beta::*;
    use argmin::solver::conjugategradient::*;
    use argmin::solver::gradientdescent::SteepestDescent;
    use argmin::solver::linesearch::condition::*;
    use argmin::solver::neldermead::*;
    use argmin::solver::particleswarm::ParticleSwarm;
    use argmin::solver::simulatedannealing::*;
    use argmin::{
        core::{CostFunction, Error, Executor, Gradient},
        solver::{linesearch::*, quasinewton::BFGS},
    };
    use finitediff::FiniteDiff;
    use ndarray::{Array1, Array2};
    use rand::{distributions::Uniform, prelude::*};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct NnStruct {
        all_tracks_dirs: AllTrackDirs,
        nn_params: NnParameters,
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
        let step = 0.1;
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
            Ok(-eval_ae(
                &from_f64_to_f32_vec(&p.to_vec()),
                &self.all_tracks_dirs,
                &self.nn_params,
            ) as f64)
        }
    }
    impl Gradient for NnStruct {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok(forward_diff_ndarray_f64(p, &|x| {
                -eval_ae(
                    &from_f64_to_f32_vec(&p.to_vec()),
                    &self.all_tracks_dirs,
                    &self.nn_params,
                ) as f64
            }))
        }
    }

    let nn_len = nn_params.get_nns_autoencoder_len();
    let cost = NnStruct {
        all_tracks_dirs: all_tracks_dirs.clone(),
        nn_params: nn_params.clone(),
    };
    let mut rng = thread_rng();

    let init_param: Array1<f64> = input.iter().map(|x| *x as f64).collect();

    let min_param: Array1<f64> = (0..nn_len).map(|_| -10.).collect();
    let max_param: Array1<f64> = (0..nn_len).map(|_| 10.).collect();
    let init_hessian: Array2<f64> = Array2::eye(nn_len);
    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    // let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001f64));
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
                .max_iters(150)
                .target_cost(-100000.)
        })
        .add_observer(
            argmin_observer_slog::SlogLogger::term(),
            argmin::core::observers::ObserverMode::Always,
        )
        .run()
        .unwrap();

    let input_vec = &res.state.param.clone().unwrap();
    println!("{}", input_vec);
    // let input_vec = &res.state.best_individual.clone().unwrap().position;
    // let input_vec = nn_sizes.clone(), &res.state.best_param.clone().unwrap().to_vec();

    let evals = eval_ae(
        &from_f64_to_f32_vec(&input_vec.to_vec()),
        &all_tracks_dirs,
        &nn_params,
    );

    println!("{evals}");
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

    params_sim.rewards_second_way = true;
    params_sim.rewards_second_way_penalty = true;
    params_sim.evolve_simple_physics = false;

    params_sim.nn.pass_internals = true;
    params_sim.nn.pass_simple_physics_value = false;
    params_sim.simulation_stop_penalty.value = 20.;
    params_sim.simulation_simple_physics = 1.0;

    params_sim.evolution_generation_count = 200;

    params_sim.nn.view_angle_ratio = 3. / 6.;
    params_sim.nn.hidden_layers = vec![10];
    params_sim.nn.pass_next_size = 1;
    params_sim.nn.inv_distance_coef = 30.;

    let mut params_sim_copy = params_sim.clone();

    // params_sim.nn.pass_dirs_second_layer = true;
    // params_sim.nn.pass_current_track = true;
    // params_sim.nn.pass_dirs_diff = true;
    // params_sim.nn.pass_next_size = 3;
    // params_sim.nn.use_dirs_autoencoder = true;
    // params_sim.nn.autoencoder_hidden_layers = vec![10, 10, 10];
    // params_sim.nn.autoencoder_exits = 5;
    params_sim.simulation_simple_physics = 1.0;
    // save_runs(vec![evolve_simple_physics(&params_sim, &params_phys, &random_input(&params_sim), 50, 100, 5)], "nn_autoencoder_best");
    save_runs(
        vec![evolve_by_cma_es_custom(
            &params_sim,
            &params_phys,
            &random_input(&params_sim),
            50,
            100,
            None,
            None,
        )],
        "nn_autoencoder_best_complex",
    );
    params_sim = params_sim_copy.clone();

    params_sim.simulation_simple_physics = 0.0;
    params_sim.nn.pass_current_track = true;
    // params_sim.disable_all_tracks();
    // params_sim.enable_track("complex");
    // params_sim.enable_track("turn_left_90");
    // params_sim.enable_track("turn_left_180");
    // params_sim.enable_track("smooth_left_and_right");
    // params_sim.enable_track("turn_right_smooth");
    // params_sim.tracks_enable_mirror = false;
    // params_sim.eval_calc_all_physics = true;
    // params_sim.eval_add_other_physics = true;
    params_sim.nn.pass_dirs_diff = true;
    params_sim.nn.pass_next_size = 3;
    params_sim.nn.hidden_layers = vec![10];
    params_sim.simulation_stop_penalty.value = 80.;
    params_sim.eval_calc_all_physics_count = 8;
    params_sim.simulation_simple_physics = 0.0;
    params_sim.rewards_second_way = false;
    params_sim.rewards_early_finish_zero_penalty = true;

    let input = vec![
        50.82979,
        6.8255806,
        -212.33067,
        62.745567,
        45.237362,
        -140.87233,
        -72.98127,
        93.25558,
        -148.62775,
        -100.87053,
        -106.10733,
        62.22232,
        49.03537,
        -30.563215,
        26.71632,
        -69.72745,
        -50.753548,
        57.839073,
        -87.69297,
        19.02222,
        130.9145,
        -63.085625,
        65.10956,
        -85.650406,
        70.79011,
        -52.00149,
        -180.66316,
        -14.790712,
        40.251663,
        17.33697,
        105.49219,
        64.3382,
        -78.830376,
        -16.15948,
        27.656473,
        96.60939,
        79.10908,
        -3.3050725,
        55.359814,
        37.64303,
        -3.6646028,
        80.42091,
        -66.87056,
        251.49683,
        60.166653,
        136.15143,
        -16.154272,
        -22.46706,
        -26.938929,
        107.08363,
        -11.486605,
        86.18303,
        -71.15506,
        77.64043,
        -69.64464,
        49.550068,
        -40.087555,
        69.021706,
        8.264989,
        -82.418915,
        134.31361,
        -3.7207134,
        37.738583,
        -99.52632,
        -44.99712,
        92.17385,
        124.648224,
        -88.02564,
        250.21399,
        25.870653,
        -88.77891,
        -135.47725,
        -48.42257,
        35.93068,
        -131.85873,
        -216.89653,
        -75.10553,
        44.62893,
        -26.013088,
        61.770203,
        -3.4767735,
        -65.3946,
        57.212772,
        77.39033,
        21.73365,
        26.406633,
        19.281778,
        76.58683,
        -80.37626,
        71.662224,
        111.12739,
        -60.910496,
        -25.881977,
        -115.78931,
        63.433617,
        -69.30505,
        43.180832,
        -68.79547,
        27.846312,
        -29.47661,
        -169.82027,
        -2.7010465,
        25.906143,
        24.204971,
        -154.87703,
        -117.11982,
        72.440285,
        -40.214993,
        -90.72162,
        -83.748764,
        -81.2134,
        -116.93174,
        50.3682,
        -20.83887,
        158.06996,
        138.1619,
        57.19443,
        -9.595369,
        -127.12494,
        -190.78584,
        44.887077,
        -29.397337,
        34.447834,
        -130.67612,
        -81.49003,
        99.291145,
        71.481766,
        238.19449,
        -97.65335,
        92.890236,
        -2.9468393,
        -115.50133,
        38.888493,
        -2.8033266,
        35.471237,
        64.675995,
        -36.296505,
        -9.266836,
        -119.26758,
        -41.97364,
        40.923637,
        34.426544,
        -53.40543,
        -3.7035291,
        -32.267063,
        -68.91373,
        97.09721,
        -34.846046,
        3.907564,
        -66.201614,
        -118.9858,
        105.06361,
        178.53958,
        4.0544424,
        -94.79694,
        -20.102713,
        -98.62863,
        -39.861652,
        -13.361424,
        -4.9815593,
        -166.14812,
        -37.12321,
        5.371111,
        36.828648,
        -90.45125,
        -107.01102,
        131.4116,
        159.24843,
        -136.16092,
        26.962952,
        -50.639072,
        7.1206155,
        -13.854664,
        -41.75538,
        24.600922,
        -190.63876,
        -8.953553,
        -91.46681,
        142.02534,
        -44.27105,
        -115.83145,
        40.435047,
        157.87271,
        -29.26607,
        53.00716,
        -132.50919,
        -21.998045,
        -138.07574,
        -24.826086,
        20.951517,
        149.80478,
        -10.855478,
        49.51545,
        -42.998116,
        107.38452,
        -53.78259,
        -65.906044,
        8.26917,
        38.095924,
        -19.24304,
        214.2714,
        29.338884,
        28.545607,
        -90.226135,
        -139.84605,
        -13.020877,
        26.07847,
        -54.22772,
        97.5801,
        -42.63841,
        75.58603,
        21.45345,
        -31.229866,
        24.058033,
        -133.24316,
        83.35174,
        -18.754086,
        -15.691487,
        -163.5872,
        -226.31741,
        -42.148212,
        -57.019745,
        -60.946228,
        47.513885,
        6.5162983,
        81.02603,
        41.20429,
        32.849533,
        -24.057053,
        -57.74444,
        47.85135,
        -89.213486,
        -148.1284,
        18.285307,
        -58.581318,
        -39.500206,
        0.7722682,
        -17.545057,
        -111.25119,
        -164.64728,
        67.93589,
        47.84141,
        3.1108594,
        -84.065125,
        -22.2014,
        2.1666238,
        53.728764,
        -50.139416,
        -49.535164,
        -74.89746,
        71.563515,
        80.87902,
        12.340418,
        103.597824,
        -38.931637,
        88.236626,
        -4.4025974,
        31.189976,
        22.195051,
        154.10713,
        -38.06252,
        41.303932,
        -58.50313,
        57.145542,
        -34.834972,
        33.530277,
        -23.60256,
        -91.74458,
        56.94089,
        46.650093,
        0.5509671,
        -118.9931,
        -29.24474,
        -16.893341,
        -121.552574,
        -28.94258,
        32.25257,
        43.451004,
        15.8047285,
        -104.616,
        110.95029,
        23.57843,
        -86.98945,
        81.430435,
        18.395048,
        108.280525,
        -88.08824,
        -18.445538,
        -1.6042095,
        99.337555,
        -136.88745,
        186.5173,
        202.78429,
        -63.62551,
        153.67064,
        75.763916,
        -68.61824,
        -150.23825,
        -36.675125,
        -78.0957,
        -112.98722,
        -18.71968,
        -3.1673405,
        -107.34639,
        4.2084374,
        -82.71414,
        94.625984,
        -53.45087,
        66.59383,
        36.67231,
        46.42246,
        79.750565,
        147.81445,
        -57.945957,
        -34.3356,
        38.627342,
        -34.468903,
        -7.4333134,
        -26.826141,
        -70.40983,
        10.403972,
        63.41928,
        113.612724,
        304.24057,
        150.32907,
        39.444378,
        129.18619,
        -46.31621,
        -49.63668,
        15.489573,
        -11.812521,
        98.48037,
        20.820442,
        -56.481697,
        150.48865,
        27.430372,
        139.36862,
        11.737478,
        -47.422405,
        55.552326,
        -46.435333,
        25.82776,
        -277.44537,
        9.378829,
        -14.795844,
        -65.28654,
        -141.06755,
        147.61404,
        57.509842,
        21.00211,
        43.322735,
        -43.443024,
        -109.33668,
        -88.281876,
        27.002996,
        -68.6047,
        -7.291876,
        25.495424,
        49.050083,
        3.3126154,
        -34.34014,
        -39.38021,
        122.14926,
        163.6885,
        101.68576,
        87.306206,
        -34.215626,
        51.58703,
        92.512985,
        -133.32954,
        -23.16802,
        49.160656,
        -93.103,
        -15.362362,
        26.189367,
        -78.31026,
        -49.298275,
        -118.41848,
        -118.06167,
        -129.14561,
        87.65882,
        65.74473,
        -34.642963,
        0.59963614,
        -12.132362,
        -35.96119,
        -2.0369847,
        62.77015,
        -169.06023,
        17.740913,
        38.796852,
        48.766933,
        101.79444,
        -7.1656513,
        -28.951742,
        -7.115925,
        -2.1332786,
        -125.66881,
        111.65083,
        -104.83792,
        11.58429,
        75.50527,
        83.21223,
        -21.474243,
        44.069252,
        146.5395,
        32.605076,
        62.373127,
        -58.384495,
        -2.5292716,
        -54.27416,
        -95.79292,
        32.83187,
        -125.50223,
        -30.297606,
        -90.17498,
        182.35686,
        -129.64906,
        8.160638,
        -63.693634,
        211.54219,
        111.15669,
        63.879482,
        -81.08478,
        -90.232445,
        126.31275,
        86.770294,
        -71.32821,
        7.35478,
        74.93101,
        -8.507543,
        190.83658,
        -54.77851,
        89.28695,
        -86.47313,
        98.57758,
        -21.625732,
        -121.190414,
        27.44329,
        -41.286476,
        -40.039738,
        -48.831936,
        114.20205,
        38.468246,
        -187.29756,
        -12.893032,
        47.695614,
        -132.28278,
        11.25607,
        66.493996,
        35.055935,
        229.19946,
        -4.6952224,
        -94.06931,
        88.12903,
        179.99207,
        -0.064635314,
        -107.08653,
        -266.31717,
        -7.495263,
        89.36252,
        70.83124,
        -19.979351,
        -28.381914,
        -10.1463175,
        92.63006,
        85.96275,
        61.225914,
        -26.22539,
        53.100266,
        -25.584658,
        -215.76096,
        149.82184,
        -152.24033,
        101.10119,
        149.20746,
        31.906218,
        30.288544,
        -65.06056,
        -11.760653,
        2.1143365,
        -120.032936,
        -127.56416,
        -117.144905,
        -78.058495,
        19.138624,
        63.265617,
        40.55169,
        154.60936,
        -68.172325,
        29.528942,
        -36.717815,
        101.32167,
        -235.02087,
        55.383106,
        54.88794,
        -47.180695,
        -146.61148,
        15.954552,
        -1.7622387,
        -105.308075,
        134.97568,
        -28.4599,
        8.4662895,
        -54.207336,
        0.06041097,
        33.765663,
        125.03138,
        43.420567,
        15.91161,
        101.32111,
        -93.35973,
        -99.34158,
        66.06835,
        76.9416,
        -15.897721,
        -8.218464,
        100.15606,
        -89.13423,
        151.31926,
        70.55875,
        30.783707,
        -84.91548,
        -12.944083,
        -88.50834,
        84.452446,
        1.611835,
        -95.10164,
        110.356125,
        -50.71854,
        -28.931868,
        152.41895,
        144.00221,
        -67.06031,
        -40.841022,
        -112.50909,
        90.269966,
        -7.5978193,
        9.035043,
        -105.57618,
        43.7802,
        -56.10008,
        -121.07246,
        -26.111298,
        -49.39096,
        52.050762,
        40.169735,
        134.62231,
        4.658425,
        76.76731,
        1.7914641,
        48.851383,
        -4.0104885,
        -97.15206,
        40.20788,
        30.062815,
        127.34317,
        -46.095448,
        -25.987295,
        105.483894,
        -175.31894,
        -6.8107157,
        -30.053232,
        -124.755844,
        -31.537148,
        22.38251,
        -139.80946,
        -111.663475,
        4.888685,
        16.055223,
        81.48258,
        69.65019,
        -9.539486,
        -110.93246,
        -40.69777,
        -138.89558,
        32.355392,
        -19.229206,
        57.964916,
        106.66932,
        169.20358,
        -61.568253,
        -106.30939,
        -20.617561,
        57.357143,
        -49.600918,
        -48.44781,
        -40.72106,
        -33.232048,
        -13.702501,
        -188.87871,
        -43.97039,
        88.754944,
        -37.664383,
        37.93126,
        72.23161,
        -40.112034,
        -101.402504,
        -55.48282,
        70.56232,
        85.91094,
        -69.311516,
        -35.022205,
        -25.085213,
        60.674564,
        6.540574,
        57.025543,
        10.72184,
        -49.208954,
        31.5521,
        75.52286,
        37.385254,
        -17.899426,
        -52.354027,
        -26.511627,
        -13.960304,
        62.70954,
        -0.58831805,
        50.055943,
        -8.235503,
        154.98735,
        120.54744,
        -196.53255,
        -38.442146,
        8.691962,
        -35.4042,
        44.573586,
        102.93175,
        41.039097,
        23.049347,
        25.36164,
        39.409946,
        -47.8825,
        -13.539881,
        -56.808735,
        17.99452,
        -41.926304,
        -70.58573,
        -176.07678,
        -117.636505,
        54.081448,
        101.528885,
        93.17107,
        -72.44129,
        58.30619,
        -1.9841936,
        -97.52485,
        -115.21322,
        -51.738792,
        0.06626658,
        -157.64084,
        35.167534,
        49.319572,
        -125.531815,
        30.535234,
        52.744907,
        47.110416,
        -17.028273,
        104.30233,
        -5.507992,
        -68.37251,
        -128.49918,
        42.947613,
        34.27881,
        7.4876776,
        76.39832,
        8.727392,
        59.08267,
        18.38585,
        79.25063,
        3.1719,
        110.32226,
        -0.25803033,
        -64.72582,
        -106.29445,
        -107.62907,
        -128.25266,
        57.016,
        -4.0009613,
        68.63145,
        166.55951,
        16.245739,
        7.967949,
        55.68995,
        203.96786,
        126.397316,
        -13.356705,
        -102.254875,
        -19.872458,
        26.919744,
        39.288116,
        63.586567,
        84.60314,
        45.755398,
        -164.70078,
        -37.978497,
        -108.65916,
        8.707288,
        -92.352715,
        -12.429333,
        -14.801342,
        -20.504671,
        44.00929,
        36.68336,
        52.246624,
        130.94351,
        14.492783,
        -57.93018,
        -67.071526,
        14.621403,
        -138.22528,
        -2.6102273,
        -121.38736,
        -113.420975,
        -61.893253,
        -98.235664,
        -62.990517,
        -26.994469,
        -30.541307,
        -25.734997,
        -86.89197,
        -40.087112,
        9.094153,
        -83.56758,
        -20.74797,
        -9.748177,
        1.2632121,
        -13.608057,
        -6.9518933,
        24.433912,
        5.579871,
        118.29234,
        -56.721317,
        21.968445,
        134.80156,
        -39.210865,
        -9.40181,
        6.156143,
        -85.19102,
        111.48763,
        21.58796,
        122.76232,
        -90.41675,
        9.152984,
        -165.5826,
        40.304283,
        -8.051397,
        66.11045,
        -4.1537495,
        7.916171,
        -96.71719,
        -32.474934,
        -63.480034,
        111.893,
        -65.39603,
        86.92193,
        -20.23148,
        -29.124113,
        -6.574574,
        120.03416,
        -43.831738,
        59.963367,
        20.325058,
        -57.137333,
        -23.18118,
        -72.90113,
        -2.1142719,
        -48.737312,
        33.665985,
        -31.749983,
        -2.3302555,
        18.836487,
        -76.74881,
        63.42666,
        9.605455,
        -174.03116,
        68.61991,
        -110.84995,
        15.163495,
        -42.632076,
        -18.957863,
        -168.69986,
        100.11529,
        -40.10067,
        -65.51679,
        59.719948,
        -146.70041,
        -114.89861,
        -82.28902,
        -49.428722,
    ];

    // оптимизировано под complex трек с разными acceleration ratio
    // let input = vec![122.99959, 19.827106, -211.26527, 13.040962, 4.947537, -150.89253, -134.52582, 67.02215, -126.7093, -83.578224, -143.94803, 137.12057, 93.46251, 77.468834, -4.965892, -42.174675, 23.26137, 7.725583, -102.48187, -44.926895, 93.13805, -140.78406, 74.85608, -104.50879, -9.737485, -38.758556, -190.23108, 6.952379, 44.986176, -19.240522, 73.520515, 115.843666, -124.196304, -39.97365, 0.33581203, 124.3181, 137.07176, -11.680864, 120.91084, -27.713772, 4.86481, -10.919154, 12.1253195, 292.29993, 36.471653, 128.46844, -18.867287, -38.03176, -114.17947, 48.59214, -55.16136, 84.293655, -69.13536, 66.46822, -84.07144, 74.46106, -93.901, 131.61823, 98.502106, 23.387533, 138.20795, 11.513432, 61.076714, -161.86916, -80.03595, 54.036003, 177.89973, -123.51265, -12.68581, -10.163408, -5.5029364, -52.1474, -33.97062, 16.214655, -19.061657, -38.050247, -25.694986, 32.445023, -37.060207, 12.092802, 7.7428904, 3.9024227, -1.0637244, 4.1405783, 4.6181903, -1.994448, -26.614454, -3.3493242, -0.42428493, -20.593754, -20.4957, 221.62323, 69.113884, -174.87125, -30.660282, -27.860296, 26.953028, -102.82742, -120.70705, -111.60097, 6.6147623, -94.18072, -43.018856, -0.2264696, 7.012259, 116.73378, 105.11107, 25.51478, 67.131676, 112.54906, 36.980793, -174.75327, 6.2583904, 59.43666, -83.118195, -13.640844, -120.26798, 129.37666, -14.046353, 28.085266, -110.10327, -48.741016, -35.138203, -202.09608, 4.35473, -38.432728, 27.200731, -174.03261, -161.56529, 13.340796, -96.17907, -66.4376, -162.21623, -165.09845, -85.03037, 34.996487, -121.04108, 206.06812, 142.04987, 126.325806, 13.558274, -179.79202, -258.86414, 8.408983, -38.402977, 7.1786947, -118.46269, -9.440362, 28.279848, 118.21504, 187.53181, -208.68237, 111.83681, -95.19562, -94.08752, 66.03473, 139.88928, 82.302505, -7.5380006, -47.4826, -9.189448, 3.68119, 5.063537, -2.460134, 21.63082, 7.009046, -21.920567, 3.6390374, 30.05716, -43.169918, -6.1082187, 16.155361, -14.742553, 5.3807898, 30.064842, 7.1176596, -40.54399, -1.2845279, 17.563854, -19.38464, 29.168665, 15.641406, -44.69946, -48.739594, -23.891802, 35.849434, 39.0701, 59.153572, -8.283105, -43.946342, -66.65464, -86.20261, 115.83824, 30.741604, -50.45272, -173.03659, -173.63017, 52.203762, 157.90132, 14.522188, -122.95822, 61.35859, -197.12953, -67.4129, -34.69851, -58.356407, -170.83284, -88.621864, 54.21396, -12.215842, -199.71404, -165.7261, 88.86716, 172.37376, -95.013084, 41.53734, -33.525677, -22.056671, -80.77459, -80.64678, 23.352255, -159.21352, -2.857613, -70.786865, 148.48035, -60.47818, -148.36026, -10.843752, 139.25131, 27.307104, 1.0428718, -141.25333, -88.07537, -224.33244, -151.60013, 27.695406, 196.19617, -69.71327, 26.792381, -2.6967738, 60.89714, -187.4409, -114.544586, -8.03215, 23.354065, -19.75846, 258.2187, 27.869524, 49.256893, -159.17522, 16.695436, -47.734047, -12.682347, -10.944268, 20.183556, -44.537785, 22.498674, -17.13901, 16.648184, -17.734629, -7.7931056, -2.0747428, -13.240448, -9.656913, -52.063927, -19.459614, -31.515285, 34.76572, 30.252565, 21.290817, 23.98249, 2.5932806, -17.921696, -120.96738, -87.966446, 52.162117, 13.571177, 179.53156, -56.088505, 81.4875, 103.31219, 8.096389, 66.494545, -199.25508, 110.15054, -14.431482, -21.933157, -156.28705, -235.57251, -94.08241, -34.625614, -121.24315, -41.568718, 19.174393, 0.13343331, -4.758517, 15.349417, -23.368988, -74.938385, 16.446451, -126.45404, -157.75258, 79.59865, 4.7463946, -68.711426, -88.9106, -27.290733, -163.99294, -266.0483, 49.752075, 43.381767, -13.76998, -109.05556, -54.903816, 53.23545, 11.944242, -30.92163, -28.260891, -85.12609, 98.87049, 108.45636, -24.037987, 184.87808, -63.285107, 76.58202, 66.84948, 43.60226, -24.37463, 68.83407, 45.596523, 102.43553, -47.616043, 24.825367, -103.72831, 59.544235, -0.8615168, -109.948235, 83.46709, 70.35396, -77.10674, -127.325714, 17.85845, 0.7117891, -32.13957, -41.240532, -18.652359, 23.291674, 17.17413, -2.5685077, -23.756823, -14.977595, -6.19097, 11.018392, -14.173869, 11.450865, -51.298527, -8.732426, -37.446392, 28.921972, -0.9541577, -8.417754, -1.3469859, 23.367878, 5.7091236, -57.47608, -17.80437, -171.73737, 1.2423717, 104.27164, 64.96264, 96.94122, -70.67388, 206.09496, -0.37630612, -71.74157, 26.342537, -83.463356, 119.55424, -76.63775, -10.784824, -44.037033, 80.39154, -71.60061, 157.69061, 160.67427, -66.636055, 166.03781, 0.3155479, -43.148594, -133.6262, -80.40519, -31.883759, -71.493286, 7.3778825, -29.724482, -78.81311, 92.874664, -19.777433, 29.62443, -91.06756, 54.05024, 93.29753, 126.402115, 103.33693, 184.26587, -41.435905, -119.261795, 41.71476, 1.3194227, -4.0486474, -57.68281, -14.749181, -22.82639, 87.85227, 128.67091, 417.99393, 50.799873, 3.7595348, 117.756294, -37.724728, -72.63585, -33.72766, -28.508047, 75.1621, -0.39484444, -113.06965, 148.1157, -71.616905, 199.56114, 69.56616, -161.86072, 26.726463, 8.769664, -21.8275, 7.3365273, -11.28997, -15.884765, -21.198605, 4.433806, 13.66197, 4.7966785, -17.304102, 8.051556, 17.07117, -4.46984, 19.292236, -14.078759, -21.163984, -17.612173, -16.999462, -0.6972891, 1.4735016, -30.45063, 10.920243, -2.4385948, -134.02975, -17.144173, -315.3892, 0.7741976, -20.319696, -100.1649, -129.59822, 138.23871, 104.38525, 10.174865, 32.82715, -92.34157, -129.09929, -146.93927, -8.092951, -54.1663, -8.201969, -15.854421, 29.65851, 137.70001, -52.64514, -47.845776, 125.643936, 182.5821, 34.304092, 170.59276, -121.78587, 28.700987, 155.84846, -138.5825, -52.629047, -1.1842062, -76.41164, -52.71315, 7.135729, -120.568535, -83.100586, -113.697075, -187.6206, -76.108604, 174.15741, 91.05022, -97.249146, 40.677334, 100.98147, -15.181839, -6.8249226, 187.02515, -186.64671, -17.908085, 179.66696, -2.494835, 154.04053, -11.485787, -110.69902, -81.85018, 73.63738, -153.91739, 119.82563, -145.53996, 36.874344, -4.649821, 58.873215, -56.470165, 7.9158964, 163.33652, 46.074654, 51.171432, -1.0598547, 7.7256556, -41.097572, -2.7059972, 3.5243254, 8.4827175, 18.014238, -19.203074, -21.270306, -9.11266, 34.13311, 43.291637, 11.067754, 9.859995, -24.45606, 3.106928, 9.177856, -26.33383, -48.60897, 26.690262, 16.594873, 6.627214, -29.9645, 23.070951, 4.9783425, 2.5532756, -70.51258, 5.8154917, -56.729313, 42.9398, -136.37158, 152.51231, -123.692505, -49.167355, -106.735855, 257.24976, 58.70956, 119.66207, -130.19289, -58.52569, 113.00057, 129.89659, -121.95989, 84.09398, 15.172202, 21.328814, 168.86491, 104.26403, 78.98188, -66.415726, 132.1359, -205.9788, -151.06586, -20.50394, -25.512817, -84.99213, -12.729263, 108.81223, -17.960052, -204.86272, -57.251392, 77.905556, -210.54114, 72.94149, 30.545584, -76.62438, 95.23992, -42.886795, -53.15746, 128.6481, 271.77753, -22.5784, -170.47075, -202.72604, 4.9188886, 27.263474, 16.605814, 16.027468, -27.00095, -28.072968, 82.891014, 118.42437, 39.600403, -37.147354, 37.71816, 27.018393, -260.0361, 189.10378, -181.09023, 159.77896, 71.61578, -6.583144, 8.854284, 13.529202, -28.01932, -11.274318, -41.361458, -6.691892, -5.4507065, 6.831349, 26.644089, 12.097811, 2.9891553, 34.429916, -20.573795, 39.890224, 8.452365, 9.261578, -16.616125, 19.205849, 16.008883, -0.9894398, 13.324706, 0.5274746, 46.451107, 31.259476, -128.0129, -3.8304553, 16.6034, 7.50903, -169.89806, -110.45076, -32.46918, 69.70372, 82.43806, 125.056946, 130.48668, 35.400867, 66.54845, -62.363766, 134.05582, -344.0261, 12.246069, 131.65707, 23.646078, -137.8757, -103.92457, -44.76825, -135.6579, 161.3419, -70.003914, 117.705666, -97.337425, -2.7890542, -18.083456, 146.1364, -8.5377655, -14.151282, 151.97527, 0.12736008, -53.270283, 46.34215, -6.6736426, -136.74437, 7.9721055, 162.20155, -163.89084, 154.68704, 72.857254, 20.885948, -106.68227, 118.74488, 33.665096, 33.17587, -24.982368, -127.42294, 112.90078, -114.90778, -32.945854, 218.01007, 160.22778, -43.111347, -46.951153, -161.49449, 73.9815, -162.56192, 113.245575, -135.78526, 120.28083, -34.53491, -132.60675, 51.377666, 25.33688, -36.4114, 14.500134, 36.072502, -9.834729, 1.609948, -43.139175, -21.655184, 13.369449, -13.302529, 35.364277, -0.08175161, -18.753927, -30.703548, 29.431276, -25.208616, -23.459703, 0.55019826, -3.5945492, 27.99024, 4.2158713, 28.94006, 5.409066, 49.880024, -77.00551, 63.07318, 133.42157, 24.40869, 75.90943, -14.404419, 30.61718, 107.51647, -69.379524, 69.098915, 81.82334, 89.18672, -56.218006, 18.557999, 124.22922, -198.30544, 23.793196, -126.39359, -119.418594, 41.56279, 1.068643, -113.21623, -59.103565, 70.9482, 82.88249, 158.72823, 121.35577, -108.299515, -45.593468, -94.27765, -125.746475, 49.747093, -81.513275, 22.570824, 29.596018, 127.197365, -127.00812, 22.2864, -136.10323, 129.19394, -44.896954, -109.59769, -65.36646, -3.3540597, -40.17025, -148.32793, -64.62631, 40.73211, -34.83359, 10.4084635, 71.951416, -22.909983, -121.75351, -62.90402, 149.76962, 87.83706, -74.74144, -17.00519, -43.813694, 116.89784, 169.46278, 89.25524, -40.824512, 64.63284, 110.652176, 114.08055, 96.445366, 26.213337, -10.635912, -3.3012958, -14.059835, 0.5148824, 13.64769, -3.9799104, -5.7138796, 3.1735227, -26.556406, -1.0512146, -4.561666, 5.4022517, -9.57176, 2.0012631, 11.697146, 15.757824, -9.332068, -13.311998, 20.014914, -29.359543, -1.8725644, 34.641224, -39.882305, -76.80219, -121.479805, -3.363248, 142.08438, -70.73359, -25.789679, 15.320223, 177.63013, 181.60649, -159.20033, -126.1032, 2.695823, -26.588587, 80.45846, 93.166115, -19.434402, 60.850994, 73.039925, -10.429615, -77.36828, -69.15376, -14.130689, -22.358343, -42.197456, -127.51963, -269.5145, -77.50883, 85.091415, 65.199295, 125.40796, 8.112029, 106.35348, 51.437126, -47.30894, -199.0066, -67.52331, -73.13668, -58.222897, 62.77533, 82.02582, -151.26152, 59.4576, 29.011942, -20.200348, -83.2769, 195.48563, 39.66609, -113.76064, -13.01954, -49.861004, 29.122196, -55.78083, 72.983826, 22.508795, 62.43067, 14.0006, 0.7808462, 4.146634, 146.86989, 141.6578, -43.78507, -145.78897, -30.585613, -94.844315, 132.46165, -1.4806623, 85.48763, -32.68459, -25.138842, -36.741886, -57.979904, -14.880508, 5.5412946, 20.506756, 12.687407, 5.4875607, 33.846714, 11.132793, 3.8738523, -12.363412, 30.210928, 11.910528, 10.52164, 5.203176, -30.982586, -0.11799519, -37.155148, -16.286592, 16.755661, -31.562641, 147.64319, 5.6333985, 25.09432, 55.81939, 174.38048, 136.75691, -47.312916, -76.305, 12.577181, -42.303604, 107.74138, -2.8109164, 57.70692, -1.1849319, -200.27505, -89.01402, -127.93407, -44.623367, -37.5794, 56.96782, 25.303663, -4.1973653, -35.5241, 88.77342, 59.94585, 121.94038, -72.11045, -52.686855, -193.57816, 31.602356, -222.0859, -60.452904, -118.37708, -84.80176, -27.411047, -114.64215, -96.16341, -1.8132229, 11.9861, 4.9339414, -170.06865, -51.577255, -9.203213, -77.38308, -29.839935, 31.010883, 40.009853, -23.076828, -52.114147, -14.676016, 32.5232, 189.77307, -79.43642, -48.69232, 147.45631, -77.14896, 18.871378, -41.60389, -55.436363, 164.05853, 23.593302, 190.35449, 11.175751, -54.03387, -203.0115, 52.741753, -24.225012, 59.627983, -104.16808, 34.1287, -11.32434, -97.79916, -2.3898604, 61.765484, -12.784402, 52.179302, 30.581629, -59.12471, 5.8075876, 223.61646, -49.33609, 113.25617, -64.11098, -94.147736, -109.74751, -136.923, -36.84891, -40.38326, 96.71126, -80.25205, -43.19268, 55.141457, 16.585922, 33.700626, -87.294975, -384.48276, 8.251474, -138.03267, -11.951185, -9.195237, -25.559935, -124.5181, 196.41222, -3.8245058, -25.152508, 32.64748, -121.16522, -139.91415, -52.62504, 21.02214];

    // let nn_len = params_sim.nn.get_nn_len();
    // let nn_params = &input[..nn_len];
    // let nn = NeuralNetwork::new_params(params_sim.nn.get_nn_sizes(), nn_params);
    // let mut nn = nn.to_unoptimized();
    // for _ in 0..params_sim.nn.max_segments {
    //     nn.add_input_neuron();
    // }
    // let nn = nn.to_optimized();
    // let mut input = nn.get_values().iter().copied().collect::<Vec<_>>();
    // input.push(0.0);
    // params_sim.nn.pass_current_segment = true;

    // params_phys.acceleration_ratio = 1.0;

    // save_runs(vec![evolve_by_cma_es_custom(&params_sim, &params_phys, &input, 100, 200 + 1, None, None)], "early_finish_complex2");
    // params_sim = params_sim_copy.clone();

    // let all_tracks_dirs = eval_tracks_dirs(&params_sim, &params_phys, &input);
    // use std::io::Write;
    // let filename = format!("graphs/tracks_dirs.json");
    // let mut file = std::fs::File::create(&filename).unwrap();
    // let json = serde_json::to_string(&all_tracks_dirs).unwrap();
    // write!(file, "{}", json).unwrap();

    let all_tracks_dirs: AllTrackDirs =
        serde_json::from_str(&std::fs::read_to_string("graphs/tracks_dirs.json").unwrap()).unwrap();

    params_sim.nn.use_dirs_autoencoder = true;
    params_sim.nn.autoencoder_exits = 21;
    params_sim.nn.autoencoder_hidden_layers = vec![];
    let nn_ae_true = (0..22)
        .flat_map(|x| (0..21).map(move |y| if y == x { 1.0 } else { 0. }))
        .chain((0..22).flat_map(|x| (0..21).map(move |y| if y == x { 1.0 } else { 0. })))
        .collect::<Vec<f32>>();

    {
        // let nn_ae = vec![-10.402192, -12.81467, -1.334112, -54.846046, -16.628483, -15.882096, 5.2827773, -8.709642, -3.2156067, 25.40373, -10.916567, 36.08911, -9.660876, 3.837295, -1.8140374, 5.09231, -30.084229, -26.631052, -8.031221, 3.5133877, -16.342224, -14.879568, -20.339033, 22.362162, 28.225243, 8.65498, 9.549778, 28.15433, -2.1783109, -24.507343, -3.632232, -11.8642025, -8.263387, -8.246302, 9.588532, 5.5749345, -17.62662, 20.225092, 5.687475, 26.644722, -12.309989, 2.6157231, 13.518888, -7.001409, -12.401393, -20.812086, -34.745884, 15.620207, -1.1153742, 32.325115, 20.036617, 16.626688, -10.206387, 10.305707, -6.9369025, 2.418043, 14.494402, 25.377663, -24.043068, 12.711786, 42.7951, 15.390242, 11.896772, -28.701984, 8.827669, -1.7954232, 9.981912, 3.059857, 17.223806, -19.135246, 1.6607916, -33.349464, -3.8960204, -31.991394, 20.279718, -13.030778, -5.3444047, 32.23121, 31.205284, -7.752082, 11.028618, 18.51431, 26.290848, 5.9877095, 24.388117, -14.840845, -32.848263, -15.919551, -12.401217, 10.219399, 3.459926, 17.944166, -7.0370064, 1.9648772, 34.361713, -32.34455, 15.611576, -9.069553, -11.585644, -5.7530074, 35.910187, -28.130653, -30.583973, 31.743319, -7.2818093, 14.055178, 2.800713, -0.25402683, -0.029239891, 27.309935, 3.079662, 12.637695, 12.382198, -4.762073, 6.910957, -12.202516, -7.2248945, 15.394067, -5.606745, -16.811123, -59.15733, -5.5759706, -14.631457, 12.211524, -7.263173, -8.283347, 9.426487, 11.17143, 4.433884, 26.959188, 32.909798, 4.651769, -21.665623, 0.5406569, -8.548473, -13.20534, 16.948662, -11.020735, -45.99161, 14.225613, -12.138587, 17.888279, 35.03713, 28.922096, 31.75018, -7.535165, 7.266941, 28.625319, -44.60054, 6.7494693, 34.570156, -8.114177, 10.908494, -35.07604, 17.94224, -25.791101, 30.390848, 0.93601125, -48.133434, 27.78198, 9.257546, -9.369708, -4.350466, -13.946973, -2.59209, -6.9654274, -14.620564, -12.469917, 9.242237, 37.434383, 1.8004198, -9.754682, 2.6446605, 6.9597926, 18.125362, -8.1912775, -10.448769, 43.662403, 6.261963, 18.201944, 10.391566, 33.374947, -6.5989513, -4.7780538, 7.956817, -2.04103, 42.46626, 10.568605, -23.183004, -26.48126, 33.773216, 11.2762375, -4.017927, -46.213207, 10.602224, 24.108633, 25.363718, 6.2135215, 13.054002, 7.4923553, -6.9606447, -25.15934, -0.49015117, 17.3396, -10.865764, -46.42719, -32.304607, -31.254908, -0.8754342, -15.11934, 39.182884, 4.795882, 25.74899, 42.49588, 9.4887495, 36.302795, -7.015494, 29.476086, -3.3804512, -17.335752, -22.49072, 1.1690334, -5.2992663, 29.840431, 17.630184, 2.9305396, -11.806991, 13.317641, -24.795975, -5.2244244, -13.853342, 2.2161162, -5.7702518, -2.9729834, -12.128309, 4.8666997, -11.008327, 9.9623165, 15.621912, 8.56509, -11.542491, 9.372408, -26.17975, -10.359556, 3.7357278, 3.8641405, -14.195796, 26.019978, -28.630375, -24.06834, -24.65577, -13.131994, 13.31206, -16.421507, -12.634441, -22.024918, 1.0392586, -24.861893, -31.749197, 4.3569636, -8.507961, -63.755524, -22.41855, -58.190155, -1.8819113, -21.463888, -9.276476, -5.5474815, 30.150148, -18.107452, -25.254995, 11.052868, 22.04995, 21.374638, 12.030538, 12.190385, -9.01388, 14.950607, 18.47684, -3.461462, 4.037393, 20.265793, -2.8404257, 15.562462, -13.620628, 4.1476293, 28.413874, 15.426255, -4.0989256, -11.260354, 36.44368, -12.729996, -33.92409, -0.8787172, 36.813713, 12.985842, 31.693453, 10.079168, -0.6295351, -17.76084, 13.821363, 1.9657446, -5.858153, -6.9550104, 2.4301524, -15.773419, -18.923468, 37.982204, 7.492298, 17.53056, -14.574392, -31.933655, -27.232758, 22.303211, 13.088847, -2.088057, 17.049175, -8.926121, 6.6660504, -13.785842, -30.57913, 22.570255, -9.258044, 3.797602, -2.5006785, 4.095271, -25.840889, 1.6187761, -0.55601597, -8.808943, 4.1852117, -3.687526, 21.454412, 1.3934801, -2.2097778, -0.42945912, -33.96055, -26.76173, -8.334811, 33.57418, 5.107932, -19.478424, -22.715155, 16.272274, 5.572794, -27.071268, -2.6735964, -13.45185, 21.672874, -3.1931982, 12.923532, 12.489648, 8.935326, -20.408216, -0.88897073, 19.207579, -23.166027, 14.203613, 6.6405544, -9.70911, -3.7747543, 39.156418, 14.740594, 6.3442097, 1.195165, -1.0709428, 21.776655, -2.5554345, 25.078701, 34.812996, 1.4691111, -10.585149, 6.4991136, -8.891779, -3.6940129, 23.916084, 6.2499633, -52.329975, 7.750041, 45.63462, -18.123257, -10.0024, 30.323141, 17.18454, 13.7591095, -4.930341, -0.13706625, -18.057444, 23.803917, -31.6441, 17.299658, -6.0696177, 1.4992105, -17.818697, 19.185667, -20.193094, 5.5799785, 19.085653, 16.984776, -18.7103, 34.39665, 13.147737, -38.83694, -35.05855, -24.177048, 41.714027, -7.9462824, 3.4458513, -12.937017, 53.41195, 19.28394, 7.690452, 5.4922504, -9.25375, 31.087502, 20.920532, 15.236817, -39.427887, 0.6807137, -6.8717027, -7.2335696, -3.5944173, -53.315147, 27.089678, -12.413843, 2.9783828, -11.4738655, -2.5641265, -3.4632154, 9.522597, 6.935432, -51.13237, 23.136923, -14.197098, 24.916367, -11.407749, -46.52631, -7.396782, -14.659616, 6.603961, -17.220852, -0.10957733, -5.650636, -25.734434, 24.114012, -42.878395, 9.273736, 25.684786, -17.077387, -18.77252, -11.18972, 0.65073514, -20.841198, -1.7625315, -2.936679, -11.475946, -46.827293, 13.614556, 16.455715, -8.661035, -17.19266, -25.228245, -16.95726, 7.67715, 22.110437, -18.051264, -18.423014, 32.354374, 1.2610549, 17.346628, 25.427088, 2.0342872, 6.6190486, -2.912024, 8.5052395, 22.758698, 17.118412, 23.470568, -4.908269, 10.984505, 15.008988, -12.080434, 9.094064, 24.158188, -19.603046, 18.894161, 7.070206, 13.699905, -32.902107, 3.1501918, -35.220776, 33.16622, -33.19015, 3.2644298, 20.42653, -17.539122, -9.958948, 23.200514, 14.266396, -6.812111, 9.429072, -5.6587577, -19.665924, 0.1506698, -32.2089, 34.14472, 22.006447, -7.6289363, 8.400105, -6.839393, 38.013348, 9.867596, -10.54844, 10.773991, 16.14327, -23.717747, 5.968211, -18.333, -1.8089609, 23.634624, -31.620228, 18.176971, -37.979057, -0.74829024, 4.7924137, -21.95463, 36.17055, 12.678212, 11.26078, -2.8572671, 22.435976, 9.342743, -8.888319, -27.929117, -11.394844, 21.933855, -17.430073, 23.437654, -12.286692, 20.806799, -18.769848, 1.0399929, -11.654469, 14.968251, 26.495934, 5.005573, 5.6344132, 14.208514, 3.6094744, 7.4546185, -23.185652, -9.371112, 21.940815, 11.750916, 10.270708, 7.2531114, -12.674362, -7.968918, 4.0428076, 1.532424, 0.5423498, -0.3822809, 12.162314, 11.465307, 25.030281, 22.702919, 8.26633, -22.717798, -7.7406235, 28.589796, -9.558722, 7.0105004, -11.79277, 15.905154, 21.117659, 14.129385, 22.470482, 14.406423, -21.170244, -6.78817, -7.133777, -15.765576, -13.293311, -10.705407, -9.61853, 25.09602, -2.4420853, -28.72172, 2.024209, 12.052, 10.286016, -10.423052, -18.204578, 7.5217524, 15.90335, 21.757668, -11.731483, 1.8408504, 10.506032, 15.4024105, -2.2635045, 15.426017, 26.475983, -11.2278595, 10.406003, 8.239731, -9.744912, -1.8288183, 8.233478, -29.584389, 13.434305, -0.262883, 1.7461076, -5.884614, 8.504635, 16.47839, -10.352684, -16.763388, -41.138283, -9.05946, -29.01326, -12.360936, 5.230625, -1.0046263, 16.87837, 9.844948, 21.047327, -0.057271816, 7.729325, 3.1492941, -44.14263, 0.9784782, 28.139292, -5.536187, -27.57507, -23.028135, -8.702454, 18.1424, 2.107851, -0.020221276, 33.581173, -8.517357, -47.36999, 23.728851, -10.8426485, 5.582527, -20.076254, 13.180222, -15.724453, -13.117335, 35.96898, -24.225939, 22.448156, 9.656536, 13.347906, 14.346284, -0.08044238, -10.4974785, 15.320372, 8.961649, -20.295588, 1.563929, -32.09386, -4.6694565, -10.850539, 12.036532, -12.385367, 18.546635, 5.0814176, 22.031656, 11.376638, 2.1596029, -0.102024645, -11.996105, 18.20225, 14.583181, 23.856422, 0.29213953, -18.723782, 15.249735, 17.68424, -16.712534, 27.191149, 8.218385, 12.607009, 30.574932, 0.060334142, 1.9817942, -10.435183, -1.7783374, 16.574461, -5.253032, 2.254499, 4.4098425, 16.342941, -19.961582, -13.747298, 28.914028, 18.687431, 12.43136, -11.403279, 42.08944, 27.793236, 3.709095, 6.9413013, 37.347233, 7.4912615, 15.319032, -32.239662, -3.621704, 8.033373, -0.7498707, -33.513084, -14.443935, 9.359963, -4.727625, -15.121411, 46.627087, 17.975842, -30.049213, -4.0428505, -8.871616, -9.143748, -39.724567, 5.1404386, -6.6544714, -7.200495, -5.3227353, 26.876722, 4.2896423, -14.484675, 7.3419642, 2.8639913, 12.992275, -9.2744255, 23.326082, -5.2673297, 25.881891, 31.955164, -11.941436, 6.7384963, -6.709575, -8.550409, 1.6657376, -0.6219369, 17.284058, 5.534735, 18.430216, 21.25945, -14.615407, -10.068931, 21.723404, -0.5143439, 16.241512, 23.475903, 23.67134, -7.324945, 7.078916, 13.846919, 28.460903, -16.804182, 22.772625, 19.696138, 1.836437, 38.524315, 18.022242, 12.945707, 28.965097, 15.7240305, 1.7086297, -10.061468, 6.461726, -7.8129787, 10.943852, 9.155345, 4.0983725, -9.368745, -0.24757686, 45.416218, -9.87654, -15.203764, -1.2960294, -4.4287734, 27.277502, 18.553606, 16.410852, 5.2531896, 13.198769, 29.103214, 17.615368, -8.178389, 17.665274, 17.568453, 41.66804, 15.353958, 12.9696665, -6.8775964, 17.255962, 10.148002, 9.845259, 6.749038, -8.143438, 5.1317835, -20.445932, 5.906553, -18.056501, 13.894502, 3.2883627, 32.10415, -0.28647432, -14.977278, 9.060014, -3.5721684, 8.341531, -13.940936, 15.589018, -3.5364635, 20.657656, -21.963596, -13.304408, 7.925467, 7.025375, -3.5710862, -7.5190954, 15.156425, 2.7688332, 23.226873, 7.6592126, -14.045539, 0.9400547, 14.336419, 5.399253, 11.410572, 36.92519, -5.042029, -0.07001658, -14.884954, 21.029516, 18.471708, 32.67513, -11.061294, 27.663849, 14.6624775, -29.892727, 20.356745, 4.7093434, 15.963271, -20.63456, 2.4499183, -25.726318, 14.724038, 41.430073, 14.386813, 26.568254, 24.104473, 31.750753, 5.4195337, 0.9472553, 0.22706231, -4.7847295, 24.22206, -28.975145, 14.134741, 1.9530394, 43.262463, -7.7317452, -2.5314653, 14.31748, -12.227576, 2.4182682, 6.4123616, 45.15869, -8.500358, -10.293742, -3.3252983, 20.718964, -0.018237848, 5.065467, -16.371725, 25.829779, 14.211358, -18.214428, -3.119935, 66.59417, 26.247614, 16.588602, -7.2911835, -18.222338, -24.279478, -3.367562, 26.245724, -11.737988, -9.204783, -16.333954, -7.3411403, 9.660266, 14.545144, -2.850747, -6.4001656, 0.15658051, -5.732063, 26.251387, -21.660866, 7.382456, -7.995202, 49.660637, 20.830267, -6.308548, 7.973229, -25.881931, -34.41236, 24.16468, -23.503645, -20.655416, 6.0091724, 8.200621, 3.46788, 12.625018, 1.9606158, -2.3312433, 39.85789, 30.240776, -33.235523, -3.7548316];

        // let nn_ae = vec![-27.218224, 17.133417, -37.671772, 27.369268, 32.586292, -6.011719, 15.155539, -22.126303, 0.10359073, 4.2528553, -31.832127, -70.46214, 14.441221, -25.9086, 47.783928, -42.35313, 8.125991, -57.745426, -28.436216, -6.4652066, 45.400192, 7.232529, 2.5940673, -25.002401, -14.062383, -24.278849, 16.85447, 25.313282, 12.942309, 3.051526, 5.66826, 5.202133, 15.897694, -35.244583, 36.821568, 23.905182, -19.909006, -20.78944, -9.458158, -4.765595, -8.817683, -59.66113, 12.06762, 20.650862, -21.3492, -1.1322707, -6.9837394, 4.0257487, 41.939793, -6.009641, -27.31491, 9.146661, 8.494114, 34.611965, -21.214617, -7.7447114, -28.80437, -10.747716, 36.506924, 28.191666, -10.001554, 0.4062614, -35.24157, -27.165583, -6.3632383, -55.76939, 22.506142, -48.142605, -3.4321167, 2.8232996, 25.982569, 50.73253, 22.700388, -6.0489073, 9.222935, 7.228762, -35.145546, 17.6011, -7.353504, 39.911324, 9.743269, 10.273915, 9.766322, -31.013245, -1.3153051, 44.715, -8.663256, 13.430467, -69.18587, 15.980481, -6.9943743, -6.424999, -54.19015, -10.477118, -20.918966, -15.262623, -39.584606, -34.415787, -3.815318, -15.530305, 17.95494, -7.684779, -62.310665, 17.918556, -18.178528, -23.279337, 12.833176, 14.377593, 20.098833, -16.461126, 2.374039, 51.401463, -17.907959, 22.158588, 11.291428, -2.8593886, -0.5333427, -19.174654, -59.213146, 10.958673, -18.445726, -33.20053, 0.94084424, -7.1788397, 13.095681, -46.542774, 15.17462, 9.796274, 0.75057083, -6.948541, -1.5366769, 35.532883, 38.855583, -23.987785, -6.474254, -22.308893, 26.34997, -53.221157, -13.09908, -27.878742, -7.002189, 17.591986, -10.99833, -35.84307, 18.579245, 19.89024, -24.952478, 5.4289517, -20.97732, 26.299854, -48.27754, -22.03377, 14.035232, -33.089035, 27.793613, 11.387519, -53.847282, 2.4898074, -15.576017, 3.0472922, 9.415255, -59.35402, 13.483398, -12.393955, 18.12349, 8.370019, 39.68044, 18.738945, -35.822617, 0.90428126, 29.024115, 41.35968, 11.8133955, -46.01729, -14.2022705, -39.91097, 20.568045, -26.27885, -33.89822, 2.3489122, -38.24004, -22.318714, -16.114462, -12.658306, -10.672129, -5.4831634, 11.783103, 18.22828, 10.114795, -19.320982, -40.171867, -9.262074, 21.255606, 4.607934, 3.1629872, -71.32761, -37.877205, -46.81828, -13.39436, 23.424416, 15.84207, -45.387806, 17.094128, -39.528442, 9.025033, 32.382748, 42.94113, 37.126778, -13.026775, -33.684288, -19.624783, 9.875304, 10.598386, -20.865742, -2.2921095, 11.101764, 22.615978, -27.541248, -2.7266293, 5.101632, -34.663, 1.3166424, 17.171047, 3.881494, -23.186726, -43.39022, -12.396299, -5.098136, 39.402157, -20.04563, 13.29511, -36.597637, 34.480118, -13.7093115, -10.4663105, -25.939901, 6.0977664, -46.902176, -10.332387, 14.540328, -3.9014466, 27.33057, 21.191137, 22.930237, -29.800865, -12.7519455, 11.957602, -43.61371, 18.002604, -4.744519, -2.560486, 33.467007, -6.025956, -32.88362, 47.54581, 1.1214323, -8.421825, 31.509192, 33.577007, 5.781778, -5.4857545, -8.840458, -5.188505, 7.0918407, 14.552574, -13.843139, -41.45257, 6.5058427, 6.4101477, -57.12872, -25.805988, -17.144907, -24.218147, -31.579441, 16.152342, 39.4782, 13.217778, -42.114765, -10.815801, 6.8229413, -19.765598, -25.859865, 10.844099, -55.000656, 23.790873, -0.52260137, 14.205814, 3.038095, -47.92769, 16.335775, -68.01695, 24.446993, -3.4424083, 3.9189982, -12.501014, 10.79301, 5.3294373, 26.047009, -12.000756, 31.51651, -17.913137, 27.013546, -19.18208, -14.742183, -37.419743, -9.628022, -27.172781, -26.185343, 22.553864, -24.628069, -42.350346, 29.827925, 27.115936, 11.297965, -15.162453, -37.066406, 18.194555, -37.623573, -26.183893, -30.948467, 20.064276, 20.025944, -1.2982523, 62.65674, -71.059784, 3.4698117, 8.019327, -25.06673, -40.896015, 26.546679, -16.774218, 10.253869, 18.410522, -25.272451, 7.1470222, -31.978085, -28.062025, -31.69492, -0.009860812, 9.604144, 35.103653, -44.517723, 2.7475884, -3.997438, -15.996081, -0.93424207, 9.040598, 21.6876, 24.153135, -6.2590475, 0.95701975, -16.428984, 8.470741, 12.683017, -21.543081, 17.915531, 19.893377, 6.7507734, -18.454638, -7.980169, 13.670085, -44.022034, -2.3162508, 11.362533, 26.582441, 6.0728745, -17.823074, 18.246609, 12.093675, -34.80336, -6.487399, -20.468773, 0.8212358, -0.3660627, 33.934395, -26.801933, 1.6127168, -1.005266, 37.10935, 27.297472, -23.888857, 5.9357357, -16.727274, 15.346798, -72.64265, 24.702616, -37.801365, 5.304393, 50.744972, 20.24283, -33.1897, 19.625261, 8.396591, 31.392805, -40.289825, -12.166332, 28.119045, -9.99071, -37.559914, -8.597865, 25.751667, -21.983053, 10.100447, 16.494705, 12.11528, -27.88679, -29.4978, -14.236583, -28.226374, -42.312786, -15.79138, -54.847725, -1.146913, 18.250132, -21.992754, -39.540787, -27.678478, 31.65052, -4.6815042, -5.8404384, 1.080391, -25.529427, 9.1600485, 5.5037007, -10.823173, 1.017885, -0.9922621, -2.1241107, 2.8427932, -40.07707, -37.63145, -36.864735, -21.063427, 5.036238, -22.792841, -27.37792, -22.422243, -29.462648, 48.962147, 48.744297, -18.218874, 42.388924, 16.851818, -37.013573, -47.16117, 38.640186, 12.502603, 17.001556, -25.31748, 7.0709467, 7.9413023, 24.229197, 9.334005, -4.119898, 23.057415, 9.869038, 43.50448, -23.15929, -0.3160672, -40.792778, 50.004723, -4.0998087, 28.33392, 61.348583, -37.657818, -42.717937, -20.402487, -11.45, 8.598783, 29.438751, 12.615744, 24.47596, -19.268255, 8.696921, 7.681738, 4.5775237, 22.040943, -42.96049, 20.95878, 22.846228, 11.464182, -0.6632662, 9.150599, 45.59586, -2.161506, 8.147031, -11.150498, -1.9885492, -0.5311766, -15.243455, -12.172029, 21.16529, -10.321007, 0.65651214, -15.610069, -22.65476, 7.321276, -37.839523, 24.857853, 0.8724095, -14.273914, 36.48826, -18.49949, -9.318911, -26.006346, -24.179596, -8.184584, 0.42282, 16.875387, -13.988372, -8.494401, 20.927933, -29.755312, 43.91837, 0.2188631, -1.1963542, -25.277674, -6.25709, 29.271736, 23.408384, -30.571674, -35.138783, -13.734602, -1.11571, 16.295279, 57.824554, 3.4221158, -0.36409804, -15.332925, -3.6281333, 0.9504364, 6.255336, -16.573553, -22.014288, -20.582254, 25.700294, 62.022415, 2.925688, -7.261946, 51.137154, -15.2851305, -6.8468766, 2.7320373, 26.546747, -29.093042, 4.408895, -26.421255, -19.09467, -44.870667, 11.344451, -59.427715, -42.969433, -25.1041, -16.002913, -0.78171957, -6.3471837, 46.605545, 2.843452, 37.23997, 51.573257, 27.171162, 8.126075, 18.048183, 40.51124, 9.342686, 17.909565, 28.79814, 40.000854, -35.87046, -27.299053, 2.0021136, 3.6828096, -1.4047743, 45.154438, 15.187274, -36.33181, 28.84972, 22.662054, 12.550608, -14.969397, 10.6272545, 10.197804, 17.548082, 1.9003526, 15.089401, -13.097765, -11.128769, 42.954914, 28.741238, 23.59735, 15.481476, -7.440324, 46.078358, 15.4282465, -5.833129, -8.68864, 9.00061, 20.234728, -36.64382, -26.410038, -16.324717, 9.162067, 8.465219, 8.043057, -22.202774, 71.056946, -21.165031, -84.90857, -49.108166, 33.838833, -15.841871, -20.123152, -10.333649, -2.7600412, -6.4167676, -37.240368, 51.027283, -0.6604606, -1.4278914, -56.49742, 1.4049625, 37.586372, -25.49698, -11.897888, 20.65678, -5.576202, -12.785378, 15.473032, 4.7088118, -40.049385, -4.8947306, 19.082592, 29.992052, -30.352602, 29.767529, -56.465935, 6.5776496, -3.0858033, -2.6697495, 17.24652, -22.7547, -13.969575, 30.006643, -11.94065, 14.634809, 7.03015, -0.9055918, 12.968517, 28.702766, -38.657177, 5.32592, -18.66762, 2.6849995, 23.199778, 25.146824, -10.261995, -29.295176, -46.98718, -10.702109, -6.2836533, 14.837078, 17.999348, 53.304253, 1.0880307, 10.450532, -42.904343, -12.696536, 35.277096, 19.402893, 34.490253, -31.878426, -0.72327363, 3.2020931, 5.5829086, 1.2235193, 39.855217, 13.7512045, -20.072899, -18.492598, 35.418625, -2.75042, -46.972694, -13.164911, -1.5264325, 29.272524, -27.102371, -21.23982, 6.48199, -32.437576, 3.0106723, -0.9538729, -0.9672919, 19.445326, 27.01155, -28.820118, 4.350904, -5.2940717, 26.622917, -30.000076, 71.62214, -17.495926, 21.420828, 10.031766, 2.6134477, -64.00162, 3.1796324, 30.076479, 13.792619, -0.6556351, -27.03231, -26.451763, -71.25626, -31.094572, 8.026788, 14.267466, 49.809555, -2.8409095, -67.12405, 20.390942, -39.794884, -18.403955, 2.450811, -22.716702, -2.9938605, -0.831182, 2.927972, -14.324591, 16.308388, 32.922047, 8.236022, 28.615368, -5.403136, -22.039892, 11.565795, -3.4133062, 21.254242, -3.4795187, 58.546417, -11.531884, -4.3777957, -0.85333055, -3.2865646, 51.10584, 31.536188, 5.176219, 33.81827, -3.515083, -36.237007, -22.13741, 0.6701246, 3.4142716, -40.451454, 10.971165, 9.211313, -19.22131, -6.054189, 16.309637, -49.299183, 30.114532, -5.507632, -2.0490453, 8.642146, -24.653368, -63.77576, 79.8459, 43.28692, -1.4206473, -5.1330857, -13.822501, -2.4886785, 7.6773534, 11.758433, -30.847273, -20.18952, 3.3101163, -16.639421, 3.9290936, -10.324596, 8.90667, 3.6200078, 18.93073, -26.946514, 13.283892, 11.155951, 25.918808, -22.906862, 45.074986, -38.21903, 15.940609, -5.3236823, -14.374478, 22.265028, 19.385698, -25.114088, 27.884266, 47.49892, 13.197245, 25.572714, 9.492843, 43.36066, -26.470003, -7.3735704, -15.813127, -2.2756088, -29.205374, 14.630696, -5.4984117, -5.230003, 18.6151, -15.475535, 8.536966, -25.635777, 11.012137, -4.414873, 8.170055, 33.378815, 45.986893, -16.512009, 17.41139, 5.002574, -50.84762, 23.649818, -29.651089, 11.551647, -9.429941, 7.530942, -1.4688228, 14.640666, -5.169943, 7.424997, 2.3323495, 33.35069, 43.600555, 27.194101, -27.63035, 19.151285, 7.004185, -77.06581, 24.792881, -13.155574, 35.298965, 7.895106, -24.198652, -2.5329924, -34.566216, -0.779679, 16.079256, 25.234098, 0.5548226, -19.854725, 4.6859508, 46.4277, 18.473476, -27.562965, 6.179995, 20.413969, 10.246393, -45.724957, -9.839012, 10.605398, 19.971107, -56.2226, 6.8598537, -44.673363, -1.3906938, 6.4274077, 17.060549, 9.800216, 29.482376, 9.147155, -61.596046, -19.112818, -9.499973, -1.0059932, 12.50129, 29.34653, 33.91585, 32.06485, -13.480287, 47.12624, 46.79191, -6.2622585, -8.280455, -46.526222, -53.10293, -36.14883, 36.30288, 12.172733, -40.310997, -34.049152, 12.414795, 52.18904, -11.037516, -20.34952, 33.945065, -6.555611, 7.6605525, -19.187029, -17.203753, 16.419111, 26.29138, 24.250366, -28.60913, -43.63644, 40.7328, -26.134531, -4.8829937, 44.663723, 8.340811, -8.327685, 30.82668, -54.733795, -20.755455, 24.367435, 3.7074559, 4.2999797, -35.993618, -1.4416964, 58.486126, -35.729652, 30.637081, -5.8738, -30.31359, -36.894524];

        let nn_keras: NeuralNetworkUnoptimized = serde_json::from_str(
            &std::fs::read_to_string("graphs/neural_network_10_10_10_5.json").unwrap(),
        )
        .unwrap();

        let nn_input = NeuralNetworkUnoptimized {
            layers: nn_keras.layers.iter().cloned().take(4).collect(),
        }
        .to_optimized();
        let nn_output = NeuralNetworkUnoptimized {
            layers: nn_keras.layers.iter().cloned().skip(4).collect(),
        }
        .to_optimized();

        params_sim.nn.autoencoder_hidden_layers = vec![10, 10, 10];
        params_sim.nn.autoencoder_exits = 5;
        let nn_ae = nn_input
            .get_values()
            .iter()
            .chain(nn_output.get_values().iter())
            .copied()
            .collect::<Vec<f32>>();
        // let nn_ae = nn_ae_true.clone();

        println!("vec!{:?}", nn_ae);

        let nn_params = &params_sim.nn;
        let some_dirs = &all_tracks_dirs.0.last().unwrap().0[20].0;

        dbg!(some_dirs);

        let (params_input, params_output) =
            nn_ae.split_at(nn_params.get_nn_autoencoder_input_len());
        let mut nn_input =
            NeuralNetwork::new_params(nn_params.get_nn_autoencoder_input_sizes(), params_input);
        let mut nn_output =
            NeuralNetwork::new_params(nn_params.get_nn_autoencoder_output_sizes(), params_output);
        let mut autoencoder_loss = 0.;

        let mut input = vec![0.; nn_params.dirs_size];
        let output = vec![0.; nn_params.autoencoder_exits];

        let mut autoencoder_input_iter = input.iter_mut();
        for intersection in some_dirs {
            *autoencoder_input_iter.next().unwrap() = convert_dir(&nn_params, *intersection);
        }
        assert!(autoencoder_input_iter.next().is_none());

        dbg!(&input);

        let values = nn_input.calc(&input);

        dbg!(values);

        let values_output = nn_output.calc(values);

        dbg!(values_output);

        let mut sum = 0.;
        for (intersection, prediction) in some_dirs.iter().zip(values_output.iter()) {
            let diff = (convert_dir(&nn_params, *intersection) - prediction).abs();
            dbg!(diff);
            sum += 1. / (1. + diff);
        }
        autoencoder_loss += sum / some_dirs.len() as f32;

        dbg!(autoencoder_loss);
    }

    // params_sim.nn.autoencoder_exits = 5;
    // params_sim.nn.autoencoder_hidden_layers = vec![15, 10, 7];
    let input2 = random_input_by_len(params_sim.nn.get_nns_autoencoder_len(), 10.);
    // evolve_by_bfgs_autoencoder(&input2, &all_tracks_dirs, &params_sim.nn);
    // println!("{:?}", evolve_by_cma_es_custom_ae(&input2, &all_tracks_dirs, &params_sim.nn, 30, 1000));

    let converted = all_tracks_dirs
        .0
        .iter()
        .flat_map(|x| x.0.iter())
        .map(|x| {
            x.0.iter()
                .map(|y| convert_dir(&params_sim.nn, *y))
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();
    dbg!(converted.len());
    save_json_to_file(&converted, "graphs/tracks_dirs_converted.json");

    // params_sim.rewards_second_way_penalty = false;
    // params_sim.simulation_simple_physics = 1.0;
    // params_sim.nn.pass_current_track = false;
    // params_sim.disable_all_tracks();
    // params_sim.enable_track("smooth_left_and_right");
    // params_sim.tracks_enable_mirror = false;
    // params_sim.eval_calc_all_physics = false;
    // params_sim.eval_add_other_physics = false;
    // params_sim.nn.pass_dirs_diff = false;
    // // params_sim.nn.pass_next_size = 3;
    // params_sim.nn.hidden_layers = vec![10];
    // params_sim.simulation_stop_penalty.value = 0.;
    // // params_sim.eval_calc_all_physics_count = 8;
    // params_sim.nn.pass_dirs = false;
    // params_sim.nn.pass_internals = false;
    // params_sim.nn.pass_dpenalty = false;
    // params_sim.simulation_steps_quota = 3500;
    // params_sim.nn.pass_time_mods = vec![5., 10., 20., 40., 80., 160., 320., 640.];
    // let nn_len = params_sim.nn.get_nns_len();
    // let mut rng = thread_rng();
    // let mut input = (0..nn_len+OTHER_PARAMS_SIZE).map(|_| rng.gen_range(-10.0..10.0)).collect::<Vec<f32>>();
    // let input = vec![-66.89062, -108.62231, 246.58342, 102.292366, 16.429665, -2.724432, -99.40296, -35.57996, -263.48096, 83.635796, 18.46015, 76.38358, 70.452614, -84.21375, 158.27675, 137.68193, -10.380199, -21.433596, 9.821229, 191.32527, -93.878494, -28.524704, -103.30089, -113.96097, -104.865814, 56.465477, -25.502287, -39.230118, -37.96074, 42.85627, -199.43361, -95.35711, -168.01305, -77.226906, 4.9770904, 48.885315, 47.02196, 60.58137, 19.218166, 11.427331, 251.34377, -32.753994, -17.163734, 77.81432, 44.26807, 7.2312856, -184.56947, 68.427505, -170.47424, -20.194715, -114.07443, -15.41009, 192.25987, -83.33668, 98.05884, -67.91741, 122.65025, 66.55112, -145.74121, -48.74625, -122.81123, 51.124634, -16.607061, -44.31206, 38.894466, 39.023094, 76.02566, -273.48422, -201.81976, -121.03118, -167.11621, 44.701866, -48.906654, -45.661472, -58.7552, 106.04658, -98.842926, -75.19532, -174.29295, 35.22406, -46.034386, 105.86399, 58.663887, 51.90301, 131.86256, -94.69897, 65.618355, -105.38914, 67.53233, -71.86449, 36.646038, -25.75117, 60.76806, -9.037131, 121.59891, 160.16945, 33.891155, 120.24913, -107.05062, -54.338665, -72.24365, -78.362335, -103.01281, -228.51314, 301.365, -74.38117, -190.72139, -173.08618, -129.64525, -11.8894205, 177.76425, 197.36433, -14.698077, 29.290905, 26.224588, 179.34471, 76.110306, -57.028854, -96.78898, 207.78308, -81.46421, -148.95154, -215.66296, -84.84162, -28.153461, 12.577771, 180.15828, -37.221386, -8.027995, 94.25407, -54.531284, -18.881538, 167.27827, 38.636795, -147.6951, 239.49042, -10.458294, 172.2906, -64.73083, 93.17063, -123.70115, 14.096864, 111.40652, 20.212591, 25.037558, 73.16157, 85.4934, 92.06663, 165.43498, 80.16593, 22.897295, -256.73398, -90.83209, -195.5379, -28.888535, -65.82334, -98.49004, -8.540158, -189.55867, 1.7741178, -71.810936, -190.23196, -94.17067, -81.85241, -187.02585, -9.580281, -221.08873, 155.57037, -78.48842, -110.67668, 64.633224, -0.5951979, 46.092064, 70.95609, 222.77187, -101.262886, 12.213247, -0.13395002];
    // save_runs(vec![evolve_by_cma_es_custom(&params_sim, &params_phys, &input, 50, 300, None, None)], "blind");
    params_sim = params_sim_copy.clone();

    // test_params_sim(&params_sim, &params_phys, "evo_default_pop_30_rate_1_dist_10");
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_population_size = 15;
    // test_params_sim(&params_sim, &params_phys, "evo_population_10");
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_population_size = 60;
    // test_params_sim(&params_sim, &params_phys, "evo_population_60");
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_population_size = 100;
    // test_params_sim(&params_sim, &params_phys, "evo_population_100");
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_learning_rate = 0.8;
    // test_params_sim(&params_sim, &params_phys, "evo_rate_0_8");
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_learning_rate = 0.5;
    // test_params_sim(&params_sim, &params_phys, "evo_rate_0_5");

    // params_sim.evolution_learning_rate = 0.3;
    // test_params_sim(&params_sim, &params_phys, "evo_rate_0_3");

    // params_sim.evolution_distance_to_solution = 1.;
    // test_params_sim(&params_sim, &params_phys, "evo_dist_1");

    // params_sim.evolution_distance_to_solution = 5.;
    // test_params_sim(&params_sim, &params_phys, "evo_dist_5");

    // params_sim.evolution_distance_to_solution = 15.;
    // test_params_sim(&params_sim, &params_phys, "evo_dist_15");

    // params_sim.evolution_distance_to_solution = 20.;
    // test_params_sim(&params_sim, &params_phys, "evo_dist_20");

    // params_sim.evolution_sample_mean = true;
    // params_sim.nn.hidden_layers = vec![10];
    // test_params_sim_fn(
    //     &params_sim, &params_phys, "nn_restart_layer_sm",
    //     |params_sim, params_phys, input, population_size, generations_count| {
    //         let start = Instant::now();

    //         let mut result = evolve_by_cma_es_custom(&params_sim, &params_phys, &input, population_size, generations_count, None, None);

    //         let nn_len = params_sim.nn.get_nn_len();
    //         let nn_params = &result.last().unwrap().nn[..nn_len];
    //         let nn = NeuralNetwork::new_params(params_sim.nn.get_nn_sizes(), nn_params);
    //         let mut nn = nn.to_unoptimized();
    //         nn.add_hidden_layer(1);
    //         let nn = nn.to_optimized();

    //         let mut params_sim = params_sim.clone();
    //         params_sim.nn.hidden_layers = vec![10, 10];

    //         let mut params = nn.get_values().iter().copied().collect::<Vec<_>>();
    //         params.push(0.0);

    //         result.extend(evolve_by_cma_es_custom(&params_sim, &params_phys, &params, population_size, generations_count, None, None));
    //         println!("FINISH! Time: {:?}, score: {}", start.elapsed(), result.last().unwrap().evals_cost);
    //         result
    //     }
    // );
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_sample_mean = true;
    // params_sim.nn.hidden_layers = vec![10];
    // test_params_sim_fn(
    //     &params_sim, &params_phys, "nn_restart_neuron_sm",
    //     |params_sim, params_phys, input, population_size, generations_count| {
    //         let start = Instant::now();

    //         let mut result = evolve_by_cma_es_custom(&params_sim, &params_phys, &input, population_size, generations_count, None, None);

    //         let nn_len = params_sim.nn.get_nn_len();
    //         let nn_params = &result.last().unwrap().nn[..nn_len];
    //         let nn = NeuralNetwork::new_params(params_sim.nn.get_nn_sizes(), nn_params);
    //         let mut nn = nn.to_unoptimized();
    //         nn.add_hidden_neuron(0);
    //         let nn = nn.to_optimized();

    //         let mut params_sim = params_sim.clone();
    //         params_sim.nn.hidden_layers = vec![11];

    //         let mut params = nn.get_values().iter().copied().collect::<Vec<_>>();
    //         params.push(0.0);

    //         result.extend(evolve_by_cma_es_custom(&params_sim, &params_phys, &params, population_size, generations_count, None, None));
    //         println!("FINISH! Time: {:?}, score: {}", start.elapsed(), result.last().unwrap().evals_cost);
    //         result
    //     }
    // );
    // params_sim = params_sim_copy.clone();

    // params_sim.evolution_sample_mean = true;
    // params_sim.nn.hidden_layers = vec![10];
    // test_params_sim_fn(
    //     &params_sim, &params_phys, "nn_restart_sm",
    //     |params_sim, params_phys, input, population_size, generations_count| {
    //         let start = Instant::now();

    //         let mut result = evolve_by_cma_es_custom(&params_sim, &params_phys, &input, population_size, generations_count, None, None);
    //         result.extend(evolve_by_cma_es_custom(&params_sim, &params_phys, &result.last().unwrap().nn, population_size, generations_count, None, None));
    //         println!("FINISH! Time: {:?}, score: {}", start.elapsed(), result.last().unwrap().evals_cost);
    //         result
    //     }
    // );
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.view_angle_ratio = 3. / 6.;
    // params_sim.nn.hidden_layers = vec![10];
    // params_sim.nn.pass_next_size = 1;
    // params_sim.nn.inv_distance_coef = 30.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_best");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_coef = 40.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_coef_40");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_coef = 80.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_coef_80");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_coef = 160.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_coef_160");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.use_dirs_autoencoder = true;
    // params_sim.nn.autoencoder_exits = 5;
    // params_sim.nn.autoencoder_hidden_layers = vec![6];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_autoencoder");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.use_dirs_autoencoder = true;
    // params_sim.nn.autoencoder_exits = 5;
    // params_sim.nn.autoencoder_hidden_layers = vec![10, 8, 6];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_autoencoder_deep");
    // params_sim = params_sim_copy.clone();

    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_default");

    // params_sim.nn.hidden_layers = vec![];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_no");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![10];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_10");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![20];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_20");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![6, 6];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_6_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![6, 6, 6, 6];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_6_6_6_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![10, 4, 3, 3, 3];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_10_4_3_3_3");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.hidden_layers = vec![20, 20];
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_hidden_20_20");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_internals = false;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_no_internals");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.view_angle_ratio = 1. / 6.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_view_1_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.view_angle_ratio = 2. / 6.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_view_2_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.view_angle_ratio = 3. / 6.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_view_3_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.view_angle_ratio = 4. / 6.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_view_4_6");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_time = true;
    // params_sim.nn.pass_distance = true;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_with_time_distance");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_prev_output = true;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_with_prev_output");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.dirs_size = 11;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_dirs_11");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.dirs_size = 7;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_dirs_7");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_next_size = 1;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_next_1");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_next_size = 3;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_next_3");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.pass_next_size = 10;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_next_10");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance = false;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_no_inv_distance");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_coef = 10.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_coef_10");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_coef = 30.;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_coef_30");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_pow = 0.25;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_pow_0_25");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_pow = 1.0;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_pow_1");
    // params_sim = params_sim_copy.clone();

    // params_sim.nn.inv_distance_pow = 2.0;
    // test_params_sim_evolve_simple(&params_sim, &params_phys, "nn_inv_distance_pow_2");
    // params_sim = params_sim_copy.clone();

    // test_params_sim(&params_sim, &params_phys, "default");

    //------------------------------------------------------------------------

    // test_params_sim_differential_evolution(&params_sim, &params_phys, "differential_evolution");
    // test_params_sim_particle_swarm(&params_sim, &params_phys, "particle_swarm");

    //------------------------------------------------------------------------

    // params_sim.rewards_second_way = true;
    // test_params_sim(&params_sim, &params_phys, "second_way");
    // params_sim = params_sim_copy.clone();

    // test_params_sim_evolve_simple(&params_sim, &params_phys, "evolve_simple_NEW");

    // params_sim.simulation_simple_physics = 0.0;
    // test_params_sim(&params_sim, &params_phys, "simple_physics_0.0_2w_NEW");

    // params_sim.evolve_simple_physics = true;
    // test_params_sim(&params_sim, &params_phys, "evolve_simple_2w_NEW_MUL");

    // test_params_sim(&params_sim, &params_phys, "my2", Some(input_done), 1.0);
    // params_sim = params_sim_copy.clone();

    // params_sim.rewards_second_way = true;
    // params_sim.evolve_simple_physics = true;
    // params_sim.nn.pass_internals = true;
    // test_params_sim(&params_sim, &params_phys, "evolve_simple_2w_next_internals");
    // params_sim = params_sim_copy.clone();

    // params_sim.rewards_second_way = true;
    // params_sim.evolve_simple_physics = true;
    // params_sim.nn.pass_simple_physics_value = true;
    // test_params_sim(&params_sim, &params_phys, "evolve_simple_2w_next_value");
    // params_sim = params_sim_copy.clone();

    // params_sim.rewards_second_way = true;
    // params_sim.evolve_simple_physics = true;
    // params_sim.nn.pass_internals = true;
    // params_sim.nn.pass_simple_physics_value = true;
    // test_params_sim(&params_sim, &params_phys, "evolve_simple_2w_next_both");
    // params_sim = params_sim_copy.clone();

    // params_sim.rewards_second_way = true;
    // params_sim.simulation_simple_physics = 0.0;
    // test_params_sim(&params_sim, &params_phys, "simple_physics_0.0_2w_evolve");
    // params_sim = params_sim_copy.clone();

    /*
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

    // params_sim.rewards_second_way = true;
    // params_sim_copy.rewards_second_way = true;

    // test_params_sim_differential_evolution(&params_sim, &params_phys, "differential_evolution_2w");
    // test_params_sim_particle_swarm(&params_sim, &params_phys, "particle_swarm_2w");

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
