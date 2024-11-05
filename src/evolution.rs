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
use rayon::iter::IndexedParallelIterator;
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

    pub use_ranking_network: bool,
    pub ranking_hidden_layers: Vec<usize>,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, Default)]
pub struct PhysicsPatch {
    pub simple_physics: Option<f32>,
    pub traction: Option<f32>,
    pub acceleration_ratio: Option<f32>,
    pub friction_coef: Option<f32>,
    pub turn_speed: Option<f32>,
    pub name: Option<String>,
}

impl PhysicsPatch {
    pub fn get_text(name: &str, value: Option<f32>) -> String {
        if let Some(value) = value {
            format!(":{}{:.2}", name, value)
        } else {
            Default::default()
        }
    }

    pub fn get_text_all(&self) -> String {
        if let Some(name) = &self.name {
            name.clone()
        } else {
            Self::get_text("s", self.simple_physics)
                + &Self::get_text("t", self.traction)
                + &Self::get_text("a", self.acceleration_ratio)
                + &Self::get_text("f", self.friction_coef)
                + &Self::get_text("tr", self.turn_speed)
        }
    }

    pub fn simple_physics(value: f32) -> Self {
        Self {
            simple_physics: Some(value),
            ..Self::default()
        }
    }

    pub fn simple_physics_ignored(value: f32) -> Self {
        Self {
            simple_physics: Some(value),
            name: Some(Self::get_text("s", Some(value)) + "_ignore"),
            ..Self::default()
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct SimulationParameters {
    pub tracks_enabled: BTreeMap<String, bool>,
    pub tracks_enable_mirror: bool, // mirrors track horizontally

    pub start_places_enable: bool,
    pub start_places_for_tracks: BTreeMap<String, bool>,

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
    pub evolution_simple_add_mid_start: bool,

    pub eval_skip_passed_tracks: bool,
    pub eval_add_min_distance: bool,
    pub eval_reward: Clamped,       // ratio from 0 to 5000
    pub eval_early_finish: Clamped, // ratio from 0 to 5000
    pub eval_distance: Clamped,     // ratio from 0 to 5000
    pub eval_acquired: Clamped,     // ratio from 0 to 5000
    pub eval_penalty: Clamped,      // ratio from 0 to 5000
    pub eval_add_other_physics: Vec<PhysicsPatch>,

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

    pub fn get_ranking_input_size(&self) -> usize {
        Self::CAR_INPUT_SIZE * 2 + self.dirs_size * 3
    }

    pub fn get_total_input_neurons(&self) -> usize {
        if self.use_ranking_network {
            self.get_ranking_input_size()
        } else {
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
                + self.pass_current_segment as usize * self.max_segments * self.max_tracks
                + self.pass_dirs_second_layer as usize * self.dirs_size
        }
    }

    pub fn get_total_output_neurons(&self) -> usize {
        if self.use_ranking_network {
            1
        } else {
            Self::CAR_OUTPUT_SIZE + self.pass_next_size
        }
    }

    pub fn get_nn_sizes(&self) -> Vec<usize> {
        if self.use_ranking_network {
            std::iter::once(self.get_total_input_neurons())
                .chain(self.ranking_hidden_layers.iter().copied())
                .chain(std::iter::once(self.get_total_output_neurons()))
                .collect()
        } else {
            std::iter::once(self.get_total_input_neurons())
                .chain(self.hidden_layers.iter().copied())
                .chain(std::iter::once(self.get_total_output_neurons()))
                .collect()
        }
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

pub fn place_car_to_reward(mut car: Car, rewards: &[Reward], pos: usize) -> Car {
    let (car_start, car_start2) = (rewards[0].center, rewards[1].center);
    let start_dir = car_start2 - car_start;
    let start_angle = start_dir.y.atan2(start_dir.x);
    let (a, b) = (rewards[pos].center, rewards[pos + 1].center);
    let current_dir = b - a;
    let current_angle = current_dir.y.atan2(current_dir.x);
    car.change_position(current_angle - start_angle, a - car_start, 0., vec2(0., 0.));
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
                .min_by(|a, b| {
                    a.2.partial_cmp(&b.2).unwrap_or_else(|| {
                        dbg!(a);
                        dbg!(b);
                        dbg!(&self.rewards);
                        dbg!(&point);
                        todo!()
                    })
                })
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

    pub fn get_rewards(&self) -> &[Reward] {
        &self.rewards
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

#[rustfmt::skip]
const BEST_AUTOENCODER_21_10_10_10_5: [f32; 1006] = [0.45737186,-0.17592742,0.41990036,0.3023682,0.46085286,0.09734252,0.20202918,0.18677261,0.15259826,0.08691186,0.16731122,0.10409911,-0.27700168,-0.013539409,-0.09711216,0.14397214,-0.15910262,0.21080758,0.43845347,0.07938459,0.47766662,0.49709564,0.5581285,0.5085441,0.37938422,0.13941869,-0.011706705,0.111483075,-0.3481085,-0.17621183,-0.16999112,0.33578566,0.33830214,0.3392177,0.46204475,0.43641356,0.02911971,-0.24209979,-0.13739654,0.07810422,-0.42370325,0.048519064,0.47701773,0.36765498,0.25073645,0.34227213,0.28530744,0.12449606,0.33620736,0.4206451,0.37811056,0.48892096,0.31235722,-0.019208623,0.28711075,-0.32138065,-0.48255187,-0.073856294,0.21494687,0.2527926,0.25357565,0.06038692,0.21669765,-0.4017394,0.0030092,0.027453631,-0.008625239,-0.12991595,-0.3729226,0.27464026,-0.35417527,-0.32996136,-0.3039164,-0.41292477,-0.008672744,-0.16740495,0.27102596,-0.25378257,-0.09404045,-0.34924185,0.4185815,-0.19595659,0.06775886,-0.40013322,0.0044076815,0.22488806,0.038864266,-0.38992977,-0.3962791,-0.008726857,-0.08023614,0.045806993,0.23833953,0.5205801,0.768315,0.30037403,-0.008285542,0.13156843,0.016080525,-0.41769928,-0.5351447,-0.2394889,0.4227838,0.29043153,0.3176221,-0.42664915,-0.20242606,-0.25436103,-0.005583709,0.16876525,0.41205645,-0.28080022,0.10476436,-0.31362674,-0.2677709,-0.42060456,-0.3070178,0.15683068,0.42874318,0.22397202,0.3479392,-0.08452995,0.15452468,0.3514351,-0.01399497,-0.40478998,-0.2482755,-0.1356977,-0.2107391,0.1617366,-0.24560514,-0.09844121,-0.05664088,0.016249405,-0.20677516,-0.057893697,-0.3120921,0.14034316,0.19313832,0.2763481,0.3531536,0.56920975,0.5262653,0.38728634,-0.030576818,0.6514924,-0.10670456,-0.069721915,0.25045338,-0.14655343,0.35060158,-0.10266501,0.63437945,0.32942742,0.45425716,0.074557275,0.39037,0.0637424,-0.17551796,-0.20605043,0.3715435,0.44256073,-0.0024101275,-0.19201526,-0.24129438,0.39782032,0.5097004,0.1726135,-0.3583182,-0.23892967,0.28712755,-0.21878761,0.21266371,-0.3139548,0.2520895,0.20426053,-0.38272342,-0.13531768,-0.37770674,-0.07767549,-0.079563916,0.076762915,-0.09228301,-0.15359625,0.39501822,-0.32253093,-0.05489093,-0.10004258,0.043926954,-0.21595538,0.42019904,-0.19991599,-0.2796001,-0.3063535,-0.1659193,0.11443508,-0.28578854,0.07319701,-0.2500814,-0.015817225,0.39411527,-0.14725995,0.39196688,-0.25877836,-0.04152623,-0.095975876,-0.15781116,0.028069556,-0.14534119,0.019865453,-0.06348205,-0.038866445,-0.12543958,0.0,-0.049645416,-0.008875614,-0.34252134,-0.02051096,0.0,0.0,0.027617633,0.20035297,-0.35912716,0.33826768,0.24858415,0.13768375,-0.03795594,-0.09491876,-0.5105505,-0.2659443,-0.45389396,0.11748606,0.09288216,0.32547277,-0.030020177,0.24329084,0.08851558,-0.42366457,-0.26695818,0.101017475,-0.49297047,0.36307758,-0.34514493,0.2988848,0.035871148,0.5472412,0.50963855,-0.45673212,0.40956378,-0.1742078,0.31611833,0.5084936,0.40096274,-0.1617916,0.42529443,0.28289756,0.31026813,0.1375107,0.032167792,-0.39572728,0.28746232,-0.04957891,0.20623049,0.15467656,-0.4147029,-0.11097115,0.8231028,0.20070969,0.4504164,-0.1172778,0.43438476,-0.3721964,0.4799559,0.5127816,0.22977056,0.5342281,-0.49618167,0.48291194,-0.54680806,-0.41188288,-0.46225387,0.02290225,-0.30547047,-0.4821162,0.16012192,0.38117933,0.09050703,-0.19624546,-0.15609431,0.07042986,0.3148246,-0.05269588,-0.039140135,-0.27311864,-0.5313749,-0.07278303,-0.28564167,0.40633324,0.27438074,-0.33285016,0.48366016,-0.1868916,-0.11395642,0.41317356,0.6122248,-0.17725272,0.5885771,-0.043597978,-0.12985998,-0.48262614,0.036938548,-0.23215446,-0.4586741,0.40202367,0.39871365,0.40458053,0.11369836,0.047267616,-0.33472955,-0.0736922,0.0,0.0,0.0,-0.04239895,-0.048857134,-0.07255001,0.0,0.02166893,0.003860553,0.0,0.015568614,-0.12189758,0.1656887,0.56778526,0.2573384,-0.14126685,0.26964617,0.38049787,-0.36964574,0.5429429,0.4858694,0.5392759,0.33397192,-0.5292928,0.48138225,-0.3814337,-0.26645464,-0.22139981,0.0393188,0.540257,0.3732344,0.061119914,-0.40323785,-0.027448416,-0.09904677,-0.39385843,0.51572704,0.051013887,0.010567129,-0.45928824,-0.046225548,0.35471177,-0.11607221,0.4036979,0.04851145,0.31053594,-0.20114025,-0.06318814,-0.11971743,0.2558158,0.4767382,0.06389463,0.5359721,0.27281535,0.61154187,-0.17682181,-0.5467057,-0.15335092,0.16274224,-0.2988279,-0.21623209,0.47806942,0.51816785,-0.3264413,0.3658301,0.23505229,-0.064686,0.58192694,0.036728177,-0.19700703,-0.24694583,0.36191887,0.12063205,-0.07626569,-0.3799886,-0.45023346,0.45717847,0.10571009,-0.30130303,0.34269577,-0.20633999,0.0499897,-0.097061306,-0.4296894,-0.44272336,0.5295114,0.40523547,0.47777525,-0.47180068,0.43413532,0.14695245,-0.039705575,0.15221053,-0.10159143,0.1270639,0.17898011,-0.32885066,-0.5786,-0.023145985,0.24564582,0.2326225,-0.0007247329,0.21068501,-0.068891704,-0.2498591,0.38694972,0.25435388,-0.26001695,-0.53119427,0.42076278,-0.020152673,0.0,0.0,-0.04903647,-0.021485018,0.0063240416,0.0,-0.013520969,-0.04661106,0.0,0.25270316,-0.044974983,-0.515672,0.079448774,0.50788134,0.65939474,0.326761,-0.44779658,-0.6086799,-0.48577356,-0.47844335,-0.041437924,-0.6244541,0.6047946,0.60107666,-0.8273604,0.25618458,-0.1575329,0.027038105,-0.23821822,0.5200817,0.5768277,0.2281723,-0.57039213,-0.59204894,-0.4897523,-0.20118606,-0.10484761,-0.14021656,-0.38588452,0.3546,0.61131567,-0.4037295,0.7011073,-0.33956572,0.04620619,-0.32616884,-0.4394969,-0.49657133,-0.33803958,0.40583158,-0.35029602,-0.5258989,-0.026526408,0.27576658,-0.792013,-0.20139068,0.011485338,-0.31658253,0.2183519,-0.018916938,0.0050536706,-0.02281617,-0.038263872,0.0310839,0.56170845,-0.6282362,-0.5168994,-0.15849586,0.096899875,-0.060975116,0.33497098,0.17076254,0.3759598,-0.5072324,0.13570258,-0.1473247,-0.17493798,0.10895595,0.43225223,-0.5622761,0.22041798,0.25107282,-0.3827208,-0.3367378,-0.53930104,0.06395388,-0.21373653,-0.13711393,-0.17852244,-0.014904106,0.7355005,0.25485387,0.20877622,0.59275186,-0.28384012,0.39728564,0.021546245,-0.22174235,-0.5313238,0.29870403,-0.7480616,0.23096626,0.35752147,0.31776556,0.60854894,0.7437414,0.40992495,0.0069172373,-0.6218858,0.4920594,0.07133994,-0.070414774,0.6690848,0.041190106,0.016126594,-0.0058748773,-0.07512411,0.0,0.0,-0.005659065,-0.011941007,-0.006117512,-0.031855706,-0.040702112,0.648664,-0.41656962,0.22611614,0.07048929,0.38365042,0.43846026,-0.4582126,0.52142835,-0.41739795,0.20872425,-0.74500805,-0.5456883,0.29133123,0.0045635104,0.45645988,0.0218608,0.37450957,0.19543046,-0.2703051,0.37894905,0.3440333,0.42775798,0.06840312,-0.14772394,-0.13403201,0.49949837,0.16941537,-0.42480052,0.33693975,0.5180814,-0.23840593,0.34738067,-0.14095815,0.4771434,0.0059478283,0.3157413,-0.4141257,-0.32466415,0.30409428,0.073748946,0.30155033,-0.17202307,0.31254074,0.34113133,-0.4173333,-0.30333576,0.3580578,-0.058534212,0.42712164,0.46393025,0.19257312,0.91011685,0.17239583,-0.01824528,-0.12686616,-0.19433935,-0.3245527,0.43490216,0.22452417,0.1861319,-0.4840148,-0.01780321,0.18180817,0.38608533,-0.09568596,-0.057836026,0.31867123,0.5001768,-0.5310913,-0.23036578,-0.18935914,0.44626456,-0.014953927,-0.005733669,-0.4992405,0.40648514,0.236542,-0.47628355,-0.32074094,0.43714648,0.073094346,0.43527675,0.13248475,0.12132627,0.37790197,-0.17227016,-0.52738965,0.039165292,-0.16698352,0.4481608,-0.061499804,0.26578775,-0.3720556,0.3283339,0.43164974,0.27652317,0.059041668,0.36649668,0.30042675,0.5507893,0.031096091,-0.023072015,0.015341973,0.063868605,0.05314186,-0.10451103,0.012260199,-0.033602644,-0.007871839,-0.028260214,0.4560528,-0.43863538,-0.1972818,-0.54110587,-0.11297603,0.5834811,-0.3068129,0.16640697,0.15583868,0.306662,-0.1322146,-0.25152695,-0.36682713,-0.40586734,-0.52713156,0.08048719,-0.4596381,-0.16805714,0.22507417,0.01995802,0.41539112,0.12115909,0.61171275,0.17476349,-0.17620562,-0.38986272,-0.1986934,-0.07216763,-0.2522651,0.1745536,-0.00065242837,0.33300617,-0.16377176,0.47842458,0.1233035,-0.11198796,-0.5111505,0.44108307,-0.3224152,-0.34009638,-0.19228707,-0.30730093,0.25576028,-0.10244663,-0.31067827,0.37724394,0.49409118,0.0061814627,0.5397924,0.32008857,-0.30164802,-0.50917286,0.17394805,-0.0714896,-0.44997922,0.15607458,-0.53747565,-0.5079738,-0.0361138,0.21564847,-0.57721186,-0.90376776,0.52751344,0.44228637,-0.07307916,-0.33051163,0.03798145,-0.7166991,0.0744902,0.533621,-0.58328587,0.2642905,0.30252588,0.12714164,-0.34574488,0.023801422,-0.13283314,0.2592124,0.11557618,0.27972016,0.22628273,0.4573506,-0.03671455,0.5283449,0.016419219,-0.30363163,-0.51035285,0.22357996,-0.4086221,-0.15721984,0.47282434,-0.09929528,0.5269532,0.14871538,0.47924593,0.12536359,0.06255887,-0.4338604,-0.17016983,0.039393302,-0.021478916,0.0,-0.0011150183,-0.02002353,0.013970958,0.0,0.08725746,-0.043776356,0.14214514,0.026229413,-0.2595059,-0.12774545,-0.0494774,-0.12511805,0.4504645,-0.08772072,-0.73110783,0.125394,0.31506735,0.42245546,0.3609071,-0.2768759,0.36766338,-0.24215254,0.1554749,0.31662637,-0.4220921,0.14024962,0.53150356,-0.052198645,0.24542153,-0.27837205,0.301992,-0.30569372,0.24076378,-0.045823783,-0.43873274,0.102294445,0.28654665,0.004846038,0.00082837255,-0.2729205,-0.24653284,0.11174075,-0.07795748,0.24567664,-0.62129486,0.5075251,-0.5105555,0.5848545,0.38590983,-0.16161081,0.2442681,0.37855762,0.1243355,0.25341237,-0.23978654,0.18581568,-0.5839694,-0.06217745,0.018985085,-0.0029155314,-0.11399212,0.34180018,0.38483477,0.06241387,-0.28890932,0.11825153,-0.3466623,0.17485446,0.20287298,0.3890488,-0.036215372,0.46085256,0.45194423,0.42120856,-0.021849155,0.019964645,0.087934755,-0.08417687,0.2719986,0.105430365,-0.38356945,0.23822773,0.23885334,0.3386371,0.09151632,0.3015638,0.5727032,0.07485627,0.48018882,0.08769888,-0.04850754,-0.36953154,0.1649228,-0.015656859,0.30176848,0.2974766,0.5414424,-0.2573507,-0.0125634745,0.25059485,0.0073877983,0.32782456,0.39748195,-0.09214687,0.08788801,0.24517,0.113038145,-0.12556338,0.14891444,-0.3637699,0.16135764,-0.24549964,-0.048727535,-0.4137956,-0.064132504,0.6118638,-0.39429182,-0.08805484,-0.18773945,0.2982645,0.11784611,-0.25582108,-0.112161316,0.05399874,-0.3052031,0.4420171,-0.3254955,0.26785895,-0.41021463,0.2770481,0.4295652,0.01817906,0.16657665,-0.093443215,-0.3001354,-0.18076026,-0.14202349,0.17395072,-0.115156375,-0.09231439,0.49168733,-0.006198864,-0.27805942,-0.28996944,-0.03537516,-0.21342821,0.054626282,0.1938549,-0.08305472,-0.106129885,0.6057189,-0.08438851,-0.18504211,0.12727177,0.17185001,-0.4829378,0.1772602,0.071630456,-0.11114428,0.41013658,0.26588863,-0.067258045,-0.40872657,-0.32674155,0.24508859,-0.29301828,0.24470176,0.36417535,-0.1254141,-0.3829799,0.3146924,-0.07486214,-0.22775005,-0.41284937,0.38637522,-0.40476888,0.16482165,0.23975624,-0.33793283,0.3607992,0.049563147,-0.07752391,-0.07130672,-0.43954402,0.35074002,-0.267593,0.007313381,0.41151854,-0.33716315,-0.35077953,0.11318766,-0.3380483,0.20592208,0.035967678,0.39766645,-0.21563718,-0.1851213,0.22164433,-0.31974375,-0.4119708,0.09578852,-0.05242302,0.23938423,-0.23844309,0.42013696,-0.14793545,-0.2336527,0.19472612,-0.12992854,0.32164264,0.09721068,0.4162817,0.016214421,0.25102162,0.4798254,-0.012202531,-0.17921817,0.1839505,0.12687603,0.1467754,0.11232976,0.15392108,0.09507806,0.0960064,0.054173872,0.10056898,0.08604917,0.12875709,0.2974497,0.13084707,0.03666326,0.001766446,-0.039353047,-0.016559495,-0.014885214,0.016923485,0.020065885,0.03429328,0.04688359];

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

pub struct SimulationVars<'a> {
    car: &'a Car,
    walls: &'a [Wall],
    params_phys: &'a PhysicsParameters,
    dirs: &'a [Pos2],
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

    pub fn calc_input_vector(
        &mut self,
        time_passed: f32,
        distance_percent: f32,
        dpenalty: f32,
        dirs: &[Option<f32>],
        dirs_second_layer: &[Option<f32>],
        current_segment_f32: f32,
        internals: &InternalCarValues,
        params_sim: &SimulationParameters,
    ) -> &[f32] {
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
            for track in 0..self.params.max_tracks {
                for i in 0..self.params.max_segments {
                    *input_values_iter.next().unwrap() = if track == self.current_track {
                        if i as f32 <= current_segment_f32 && current_segment_f32 < (i + 1) as f32 {
                            1. + (current_segment_f32 - i as f32)
                        } else {
                            0.
                        }
                    } else {
                        0.
                    };
                }
            }
        }

        assert!(input_values_iter.next().is_none());

        &self.input_values
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
        simulation_vars: SimulationVars,
    ) -> CarInput {
        if self.params.use_ranking_network {
            const POSSIBLE_ACTIONS: [CarInput; 11] = [
                CarInput {
                    brake: 0.,
                    acceleration: 0.,
                    remove_turn: 0.,
                    turn: 0.,
                },
                CarInput {
                    brake: 1.,
                    acceleration: 0.,
                    remove_turn: 0.,
                    turn: 0.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 1.,
                    remove_turn: 0.,
                    turn: 0.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: -1.,
                    remove_turn: 0.,
                    turn: 0.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 0.,
                    remove_turn: 1.,
                    turn: 0.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 0.,
                    remove_turn: 0.,
                    turn: 1.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 0.,
                    remove_turn: 0.,
                    turn: -1.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 1.,
                    remove_turn: 0.,
                    turn: 1.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 1.,
                    remove_turn: 0.,
                    turn: -1.,
                },
                CarInput {
                    brake: 0.,
                    acceleration: 1.,
                    remove_turn: 1.,
                    turn: 0.,
                },
                CarInput {
                    brake: 1.,
                    acceleration: 0.,
                    remove_turn: 1.,
                    turn: 0.,
                },
            ];

            return POSSIBLE_ACTIONS
                .iter()
                .map(|action| {
                    let mut input_values_iter = self.input_values.iter_mut();

                    for intersection in dirs {
                        *input_values_iter.next().unwrap() =
                            convert_dir(&self.params, *intersection);
                    }

                    for y in &internals.to_f32() {
                        *input_values_iter.next().unwrap() = *y;
                    }

                    let mut car = simulation_vars.car.clone();
                    let walls = &simulation_vars.walls;
                    let params_phys = &simulation_vars.params_phys;
                    let precomputed_dirs = &simulation_vars.dirs;
                    car.process_input(action, params_phys);

                    for i in 0..params_phys.steps_per_time {
                        let time = params_phys.time / params_phys.steps_per_time as f32;
                        car.apply_wheels_force(&mut |_, _, _| {}, params_phys);
                        for wall in *walls {
                            car.process_collision(wall, params_phys);
                        }
                        car.step(time, params_phys);
                    }

                    for (precomputed_dir, dir) in precomputed_dirs.iter().zip(dirs.iter()) {
                        let dir_pos = car.from_local_coordinates(*precomputed_dir);
                        let origin = car.get_center();
                        let mut intersection = None;
                        for wall in *walls {
                            let temp = wall.intersect_ray(origin, dir_pos);
                            intersection = intersection.any_or_both_with(temp, |a, b| a.min(b));
                        }
                        *input_values_iter.next().unwrap() =
                            convert_dir(&self.params, intersection);
                        *input_values_iter.next().unwrap() =
                            convert_dir(&self.params, intersection)
                                - convert_dir(&self.params, *dir);
                    }

                    for y in &car.get_internal_values().to_f32() {
                        *input_values_iter.next().unwrap() = *y;
                    }

                    assert!(input_values_iter.next().is_none());

                    let values = self.nn.calc(&self.input_values);

                    (action, values[0])
                })
                .max_by(|(_, value1), (_, value2)| value1.partial_cmp(value2).unwrap())
                .unwrap()
                .0
                .clone();
        }

        self.calc_counts += 1;

        self.calc_input_vector(
            time_passed,
            distance_percent,
            dpenalty,
            dirs,
            dirs_second_layer,
            current_segment_f32,
            internals,
            params_sim,
        );
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
            SimulationVars,
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
            SimulationVars {
                car: &self.car,
                walls: &self.walls,
                params_phys,
                dirs: &self.dirs,
            },
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

pub fn sum_evals(
    evals: &[TrackEvaluation],
    params: &SimulationParameters,
    ignore_ignored: bool,
) -> f32 {
    let evals = evals
        .iter()
        .filter(|x| !x.name.ends_with("_ignore"))
        .collect::<Vec<_>>();
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
            // + (1. - 1. / (1. + self.reward))
            + self.early_finish_percent
            + self.distance_percent
            + self.autoencoder_loss)
            / 5.
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

    let ffn = |(
        track_no,
        Track {
            name,
            walls,
            rewards,
        },
    )| {
        let mut result: Vec<TrackEvaluation> = Default::default();

        let car_positions =
            if params_sim.start_places_enable && params_sim.start_places_for_tracks[&name] {
                rewards.len() - 1
            } else if params_sim.mutate_car_enable {
                params_sim.mutate_car_count
            } else {
                1
            };
        for car_position in 0..car_positions {
            let physics_max = if params_sim.eval_add_other_physics.len() > 0 {
                1 + params_sim.eval_add_other_physics.len()
            } else {
                1
            };
            for physics_patch in std::iter::once(&PhysicsPatch::default())
                .chain(params_sim.eval_add_other_physics.iter())
            {
                let mut params_sim = params_sim.clone();
                let mut params_phys = params_phys.clone();
                if let Some(simple_physics) = physics_patch.simple_physics {
                    params_sim.simulation_simple_physics = simple_physics;
                    params_phys = params_sim.patch_physics_parameters(params_phys.clone());
                }

                if let Some(traction) = physics_patch.traction {
                    params_phys.traction_coefficient = traction;
                }

                if let Some(acceleration_ratio) = physics_patch.acceleration_ratio {
                    params_phys.acceleration_ratio = acceleration_ratio;
                }

                if let Some(friction_coef) = physics_patch.friction_coef {
                    params_phys.friction_coefficient = friction_coef;
                }

                if let Some(turn_speed) = physics_patch.turn_speed {
                    params_phys.wheel_turn_per_time = turn_speed;
                }

                let car = {
                    let result: Car = Default::default();
                    if car_position != 0 && params_sim.start_places_enable {
                        place_car_to_reward(result, &rewards, car_position)
                    } else if car_position != 0 && params_sim.mutate_car_enable {
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
                              current_segment_f32,
                              simulation_vars| {
                            nn_processor.process(
                                time,
                                dist,
                                dpenalty,
                                dirs,
                                dirs_second_layer,
                                current_segment_f32,
                                internals,
                                &params_sim,
                                simulation_vars,
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
                        + &if car_position != 0 {
                            format!(":{car_position}")
                        } else {
                            Default::default()
                        }
                        + &physics_patch.get_text_all(),
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
        result.into_iter()
    };

    if ONE_THREADED {
        tracks.into_iter().enumerate().flat_map(ffn).collect()
    } else {
        tracks
            .into_par_iter()
            .enumerate()
            .flat_map_iter(ffn)
            .collect()
    }
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
        -sum_evals(&evals, params_sim, false)
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
        -sum_evals(&evals, params_sim, false)
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
            evals_cost: sum_evals(&evals, params_sim, false),
            true_evals_cost: sum_evals(&true_evals, &true_params_sim, false),
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
            -sum_evals(&evals, &params_sim, false) as f64
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
            -sum_evals(&evals, params_sim, false) as f64
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
        let evals_cost = sum_evals(&evals, params_sim, true);
        let true_evals = eval_nn(&nn, params_phys, &true_params_sim);
        let true_evals_cost = sum_evals(&true_evals, &true_params_sim, true);

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
            Ok(-sum_evals(&evals, &self.params_sim, false) as f64)
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
                -sum_evals(&evals, &self.params_sim, false) as f64
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
            Ok(-sum_evals(&evals, &self.params_sim, false) as f64)
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
        let evals_cost = sum_evals(&evals, params_sim, false);
        let true_evals = eval_nn(&nn, params_phys, &true_params_sim);
        let true_evals_cost = sum_evals(&true_evals, &true_params_sim, false);

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
        -sum_evals(&evals, params_sim, false)
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
            pass_dpenalty: false,
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
            max_segments: 20,

            pass_time_mods: vec![],

            use_ranking_network: false,
            ranking_hidden_layers: vec![6],
        }
    }
}

impl Default for SimulationParameters {
    fn default() -> Self {
        let all_tracks: BTreeMap<String, bool> =
            get_all_tracks().into_iter().map(|x| (x.0, false)).collect();
        Self {
            tracks_enabled: all_tracks
                .clone()
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

            start_places_enable: false,
            start_places_for_tracks: all_tracks
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
                    .flat_map(|x| vec![x.to_owned(), x.to_owned() + "_mirror"])
                    .map(|x| (x.to_owned(), false)),
                )
                .collect(),

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
            evolution_simple_add_mid_start: false,

            eval_skip_passed_tracks: false,
            eval_add_min_distance: false,
            eval_reward: Clamped::new(10., 0., 5000.),
            eval_early_finish: Clamped::new(10000., 0., 10000.),
            eval_distance: Clamped::new(1000., 0., 5000.),
            eval_acquired: Clamped::new(1000., 0., 5000.),
            eval_penalty: Clamped::new(20., 0., 5000.),
            eval_add_other_physics: vec![],

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

    pub fn start_places_disable_all_tracks(&mut self) {
        for enabled in self.start_places_for_tracks.values_mut() {
            *enabled = false;
        }
    }
    pub fn start_places_enable_all_tracks(&mut self) {
        for enabled in self.start_places_for_tracks.values_mut() {
            *enabled = true;
        }
    }
    pub fn start_places_enable_track(&mut self, track: &str) {
        *self.start_places_for_tracks.get_mut(track).unwrap() = true;
        *self
            .start_places_for_tracks
            .get_mut(&(track.to_owned() + "_mirror"))
            .unwrap() = true;
    }
    pub fn start_places_disable_track(&mut self, track: &str) {
        *self.start_places_for_tracks.get_mut(track).unwrap() = false;
        *self
            .start_places_for_tracks
            .get_mut(&(track.to_owned() + "_mirror"))
            .unwrap() = false;
    }
}

pub fn save_json_to_file<T: Serialize>(t: &T, name: &str) {
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
    let step = 0.1;
    let start_simple_physics = 0.98;

    let start = Instant::now();

    let start_other_physics = params_sim.eval_add_other_physics.clone();

    params_sim.simulation_simple_physics = start_simple_physics;
    let mut result = evolve_by_cma_es_custom(
        &params_sim,
        &params_phys,
        &input,
        population_size,
        generations_count_main,
        None,
        None,
    );
    let good_enough_percent = 0.999;
    let best_simple_cost = result.last().unwrap().evals_cost;
    while params_sim.simulation_simple_physics > 0. {
        if params_sim.simulation_simple_physics < 0.1 {
            params_sim.simulation_simple_physics -= step / 4.;
        } else if params_sim.simulation_simple_physics < 0.3 {
            params_sim.simulation_simple_physics -= step / 2.;
        } else {
            params_sim.simulation_simple_physics -= step;
        }
        if params_sim.simulation_simple_physics < 0. {
            params_sim.simulation_simple_physics = 0.;
        }
        if PRINT {
            println!(
                "Use simple physics value: {}",
                params_sim.simulation_simple_physics
            );
        }
        if params_sim.evolution_simple_add_mid_start {
            params_sim.eval_add_other_physics = start_other_physics.clone();
            params_sim
                .eval_add_other_physics
                .push(PhysicsPatch::simple_physics_ignored(
                    (start_simple_physics + params_sim.simulation_simple_physics) / 2.,
                ));
        }
        let evals = eval_nn(&result.last().unwrap().nn, params_phys, &params_sim);
        let evals_cost = sum_evals(&evals, &params_sim, true);
        if evals_cost < best_simple_cost * good_enough_percent {
            let result2 = evolve_by_cma_es_custom(
                &params_sim,
                &params_phys,
                &result.last().unwrap().nn,
                population_size,
                generations_count_adapt,
                None,
                Some(best_simple_cost * good_enough_percent),
            );
            result.extend(result2);
        } else {
            if PRINT {
                println!("Skip evolution entirely.");
            }
        }
    }
    params_sim.eval_add_other_physics = start_other_physics.clone();
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
                11,
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
                      current_segment_f32,
                      simulation_vars| {
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
                        simulation_vars,
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

fn test_python_nn(params_sim: &SimulationParameters, params_phys: &PhysicsParameters) {
    let file = std::fs::read_to_string("records/neural_network.json").unwrap();
    let nn: NeuralNetworkUnoptimized = serde_json::from_str(&file).unwrap();
    let nn = nn.to_optimized();
    let mut params = nn.get_values().to_owned();
    params.push(0.);

    // println!("{:?}", params);

    print_evals(&eval_nn(&params, params_phys, params_sim));
}

fn print_mean_std_of_best_nns(params_sim: &SimulationParameters) {
    let file = std::fs::read_to_string("graphs/graphs_to_copy/nn2_default.json").unwrap();
    let all_runs: Vec<Vec<EvolveOutputEachStep>> = serde_json::from_str(&file).unwrap();
    let all_nns = all_runs
        .iter()
        .map(|x| x.last().unwrap().nn.clone())
        .map(|x| {
            NeuralNetwork::new_params(
                params_sim.nn.get_nn_sizes(),
                &x[..params_sim.nn.get_nns_len()],
            )
            .to_unoptimized()
        })
        .collect::<Vec<NeuralNetworkUnoptimized>>();

    // For each layer position
    for layer_idx in 0..all_nns[0].layers.len() {
        println!("\nLayer {}:", layer_idx + 1);

        let input_size = all_nns[0].layers[layer_idx].input_size;
        let output_size = all_nns[0].layers[layer_idx].output_size;

        // Calculate matrix stats
        println!("Matrix stats ({}x{}):", output_size, input_size);
        for i in 0..output_size {
            for j in 0..input_size {
                // Collect all values at this matrix position across networks
                let values: Vec<f32> = all_nns
                    .iter()
                    .map(|nn| nn.layers[layer_idx].matrix[i][j])
                    .collect();

                // Calculate mean
                let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;

                // Calculate std dev
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
                let std_dev = variance.sqrt();

                println!(
                    "Position [{},{}]: mean = {:.3}, std = {:.3}",
                    i, j, mean, std_dev
                );
            }
        }

        // Calculate bias stats
        println!("\nBias stats ({}):", output_size);
        for i in 0..output_size {
            // Collect all values at this bias position across networks
            let values: Vec<f32> = all_nns
                .iter()
                .map(|nn| nn.layers[layer_idx].bias[i])
                .collect();

            // Calculate mean
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;

            // Calculate std dev
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = variance.sqrt();

            println!("Position [{}]: mean = {:.3}, std = {:.3}", i, mean, std_dev);
        }
    }
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
    params_sim.rewards_second_way = true;
    params_sim.rewards_second_way_penalty = true;
    params_sim.evolve_simple_physics = false;
    params_sim.nn.pass_internals = true;
    params_sim.nn.pass_simple_physics_value = false;
    params_sim.simulation_stop_penalty.value = 20.;
    params_sim.simulation_simple_physics = 1.0;
    params_sim.evolution_generation_count = 150;
    params_sim.nn.view_angle_ratio = 3. / 6.;
    params_sim.nn.hidden_layers = vec![10];
    params_sim.nn.inv_distance_coef = 30.;

    let mut params_sim_copy = params_sim.clone();

    #[rustfmt::skip]
    let input = vec![0.6738715, 1.3962375, -9.709969, -4.230642, -3.2676506, 4.8991504, 2.1108444, 10.450238, 8.088549, 0.50729334, -18.5119, 0.480514, -4.8349433, -11.2844095, -1.3844314, 1.4876034, 7.716983, -1.9619504, 0.052436914, -2.3475382, 5.768196, 6.2214384, -0.039991844, -6.6355953, -1.9093088, -1.6380897, -1.2529079, 2.1773398, -3.755879, -5.2469406, -3.3197396, -0.78104776, -5.0967484, -0.5678563, 7.521401, -15.174174, -3.9670808, -4.3771453, -15.471053, 3.721583, -0.7741944, -1.5156367, 5.740673, 0.32964358, 10.183514, 3.565154, -11.03147, 3.1263359, 6.9440174, 1.8576188, 15.882395, 10.648385, 7.9288163, -7.163074, -7.1614885, 9.041861, -1.8697565, 4.051439, 1.6114315, 4.6583066, -2.336989, -5.8087296, 0.6867051, 2.2763133, 0.022993434, 6.8953013, -6.3628225, 0.6505885, 0.63723874, -1.5032864, -0.31809494, -8.800018, -6.537378, 1.7357835, -6.0556755, 0.45933804, -7.503247, 6.7944207, 2.2170389, 8.29003, 2.2948947, 7.441893, 1.4637926, -5.7660832, 0.77774525, 9.301701, -12.705781, 3.8497784, 0.36084002, -4.947007, -3.5362866, 0.9230598, 8.213377, 1.028692, 0.06970441, 10.69342, 5.4226193, 0.35453978, 3.238097, -8.675721, -0.7991422, -5.3843412, -2.3975177, -12.104445, -6.1785636, 4.250744, -2.3712895, 2.9084098, -1.4647075, 11.217567, 7.9854937, -3.453878, 0.36347595, -7.904556, 5.1204815, 3.7474253, 1.1802336, -4.8779154, 6.036164, -4.2707524, 1.1436768, 2.6914713, 12.895648, 0.73372704, -6.678311, 8.248247, -5.9110055, 7.3122635, -7.690391, 12.582394, 2.257427, 11.796972, -5.4697948, 4.311518, -4.2042828, -3.3321474, -2.365094, -6.204951, -0.111704774, 3.2974808, 5.3967466, -0.3612317, -12.495264, 3.2147114, 2.5028021, -10.717106, -6.984461, 5.6015882, 2.839644, -0.96129626, -5.0600686, -4.7290306, 1.6347486, 0.17619619, 5.667323, 1.1224418, 1.7603192, -3.2291133, -17.77918, 3.7826953, -8.318663, 5.523494, -14.4559765, -0.32160738, 0.41269335, -14.409108, 1.7315196, -3.394852, 2.1009831, 3.1736698, -1.3285139, -1.6421064, -4.875844, 4.236841, 6.7807064, 9.338122, -4.356945, 0.49719056, 1.2529842, 5.4636097, -7.306507, 0.31483808, 3.084046, -8.3597145, -0.57200867, -1.7412217, 7.0985913, -2.6089218, -8.155095, 3.5816824, -2.449088, -5.0449896, -6.69862, 5.1846147, 4.1971545, -7.2106786, -0.5595767, 5.7564073, -7.161097, -3.6338475, 0.6429556, 4.877759, -6.042184, -3.9485717, -11.746075, 0.34487122, 8.859, -5.645634, -3.8512864, -3.079258, -12.133314, 6.4503703, -4.8906627, -5.9239416, 2.7629216, -5.972483, 3.2272227, -3.918746, 1.1683934, 0.7593199, 1.7962141, -6.3846965, 5.819411, 3.2576718, -1.3861779, 0.009636184, 2.6093872, -0.38392022, 6.8437643, 0.6078342, 7.03835, -8.964784, 3.6592817, -0.6676009, -0.35108447, -9.956499, -6.9970856, -3.549926, -8.028021, -4.7959075, 3.335188, -7.8962917, -10.882255, 5.267296, 6.4303236, -0.86066014, 5.057058, -1.1721102, 4.3153934, 6.2748747, -4.4899716, 6.363882, -1.9624325, 0.7796869, 2.8663805, 9.4819145, 0.8457912, -12.272658, 0.48730576, -12.598691, -6.3082957, 2.5982032, 2.5547955, -4.1446986, 3.2683392, 5.4278398, 9.001255, -9.3119955, -9.177208, 7.8054194, 6.670474, -1.4150115, -6.012771, -6.961443, 0.23287638, 0.42331907, -5.4000406, 2.6040766, 2.313967, 2.2828512, -0.17408733, 1.3877892, -8.418374, -1.1286818, 6.9635353, 0.3188041, 10.748581, 3.6848304, 9.230947, -0.12724033, -4.108149, 2.0885513, 1.6309887, 1.1156834, 6.707361, -3.6627262, -6.4920697, 0.13680422, -9.278443, 2.2974968, 7.619677, 5.7793436, 5.146217, -4.156317, 9.467098, -5.778484, -1.8848257, -8.346977, 8.019962, 10.038042, 8.982399, 7.7225413, 11.076224, -3.0284455, -6.3276963, -12.057372, -3.9442973, -5.6489196, -4.9268017, 3.8206322, 1.1970876, 7.072864, -2.459823, 6.127094, -12.714812, 0.6546931, 0.92610204, -2.2722116, -1.2640625, 11.366784, -10.431747, -7.4916573, -3.16176, -2.3829615, -3.0428858, 4.205239, 2.6928093, 6.046191, 6.2147436, -7.5239286, 2.9150157, -3.6168456, 6.5422955, 10.428736, 3.8776603, -8.015833, 5.3989463, -2.0266063, -7.1585774, -11.922148, -0.82644176, -2.0668254, -2.876186, 2.1092627, -2.4665015, -5.4396486, 4.628618, -5.641155, 6.9669733, -4.533144, -16.067842, 3.4714723, -8.334372, 2.938648, 4.0280724, -1.1693623, -11.070127, -0.74707437, -1.0486817, 3.1873343, -9.950971, -9.031637, 17.71806, -11.161386, 0.99262726, 3.5523226, -1.1323831, 0.22481433, 2.275123, 9.166289, 3.900746, -2.862967, -5.1764283, -8.35151, 0.37135583, -1.6255641, -0.032845207, -4.610136, -4.209204, 16.591026, 10.373624, -7.9896774, 4.5687447, -8.53435, 0.34964928, -2.9510033, 4.2848344, 13.116221, 6.352876, 3.4352124, -0.14823785, 0.25068462, -4.004348, -11.196231, -0.2874741, 6.624616, 2.8403475, 4.8818755, -11.8892565, -9.939011, -12.098582, -0.64249086, 6.4809756, -5.2674246, -2.994317, -1.3270046, -3.2846994, -0.24998243, 4.216144, -8.607953, -0.85697645, -8.015713, 4.1492796, 1.5119101, -1.1941965, -4.6170983, -10.110891, 1.0302607, -1.6220341, -0.43495977, 0.500797, 1.6730058, 1.0254459, 0.07864116, 6.3756704, -7.484386, 0.8378514, -8.570443, -4.393791, 2.106032, -5.6512403, -5.610057, -7.832937, -9.653405, -4.151795, -3.2778904, 1.1567644, -1.4021993, 1.8537433, 9.47908, -2.1099195, 11.487564, 3.992513, -5.2398105, 1.0303276, -6.5170403, -3.2477758, -1.0098257, 1.1901696, -6.5265203, -11.270964, -0.054669544, 9.822261, -5.2556977, 3.349265, 12.356305, 10.311793, 2.719666, -13.922078, 5.0607657, 4.3269887, -13.458885, 9.801892, -8.359823, 3.3954253, 0.325571, 2.3993316, 2.2144196, 5.702282, -4.231979, 4.7727413, 4.6621985, -10.61038, -6.1670327, 8.1245, 2.358152, -1.0311686, -9.832273, 7.2184277, -1.8040428, -6.2133455, -9.28476, 1.0713037, 11.15517, -1.2938684, 6.6882772, 0.52386546, -4.808851, 7.2910113, -1.8336754, -5.7920313, 0.12475824, -0.8097051, -3.822063, -4.706221, 0.26966828, 8.072553, 0.19075319, 1.0390017, -1.9806764, 2.0115683, 2.6446826, 2.3741515, 3.4480588, -9.380573, -1.2658017, 7.009859, 2.2639198, 1.2822767, 4.1514473, -5.7523236, -3.268694, 8.9077215, 4.0111794, 4.056735, -5.5720053, -0.9381929, 6.539324, -3.3424828, 3.7140653, 4.6977053, -3.9671173, 4.345394, 1.6373153, 3.046059, 3.9385967, 0.55000466, 8.174156, -2.3521214, -3.2224913, -3.8069167, 0.61029464, -1.2425714, 1.7699536, 8.560754, 1.405405, 3.3751774, -6.3295155, -1.5361547, 3.7462378, -0.6373796, 9.896731, 9.707323, 3.1303754, -4.9335017, -1.1532115, -9.257265, 4.187131, 3.5619738, 6.86471, -6.5316296, -6.7160306, -1.8332678, 1.2182236, 3.7553535, -3.1426256, -2.3777232, -12.228059, 8.446975, 0.6911875, 13.005052, -2.8603888, 6.343339, 9.834011, -2.7853923, 10.171492, 5.807957, -7.2283587, -6.192022, -3.5501425, -3.252275, -8.201218, -6.4209642, -5.775359, -8.691218, -5.83, -2.1448958, -0.5844958, -6.557204, -12.0022335, -4.4206324, 3.0301588, -4.6706266, 11.24643, -2.6479974, -7.442976, -3.979046, 4.461193, -1.5340668, 7.1464524, 1.0320016, -9.276143, -0.18155181, -0.7603979, -2.3123596, -0.11147067, -4.2938538, 12.923013, 6.6453643, 11.234474, 3.912316, 1.4529525, -4.4720254, -4.019155, -1.3242787, 0.10020086, -10.338614, 1.06701, -4.14226, -7.0309515, -6.435373, -2.6474764, -3.3250067, -8.156043, -0.024789132, -8.864347, 11.723535, 1.011791, -0.9775961, -6.261484, 3.2945123, -8.516468, 0.48645186, 10.532079, -2.4773521, -3.5174398, -3.8881316, -4.4456844, 8.762048, 2.3198946, -6.5467415, -4.1247945, -5.1098723, -0.92861164, 2.5971859, 4.270702, 0.67311597, 3.0782294, -7.9665346, -7.778015, -6.5079036, -2.3372808, -1.1086447, 1.3334146, 5.313313, 5.5396605, -0.8845387, 3.642151, 0.17856163, -9.458064, 4.703637, -5.177733, -6.533986, 0.11077616, -6.1123667, 1.0119509, -3.2709632, 1.5315372, -5.9039574, -1.9154682, 0.63521624, 4.1084967, 0.20591778, 0.16088028, 1.0014832, 3.4770222, 4.505648, 7.4210863, 0.9920085, 6.4650517, -3.286229, 4.00343, -3.4254344, -0.7693688, -3.6278143, 4.8354473, 0.4476805, -9.335662, -8.961105, -4.027863, -2.308868, -6.6420712, -10.731946, 2.1924844, -0.7884905, 5.9276495, -7.240472, -9.2087, -1.631058, -7.5621057, -4.711686, -3.4083626, -3.0844975, -7.4984922, 6.4751377, -2.5670013, -3.6147892, -3.337315, -1.4617364, 4.127257, -2.469072, 12.72022, 2.5541582, 1.182711, 5.739586, 1.5758116, -1.0468379, 0.5204578, -5.879354, 0.5365463, -3.624002, -0.34282434, -1.2316997, 5.9598393, -2.1030006, -1.4017736, 0.4112153, -2.9103777, -9.999597, -4.4700427, 2.4884741, 6.0900283, 6.4103513, -0.82919776, 5.101978, -7.4778614, 11.083736, 9.09637, 3.1157615, -7.07785, 5.028301, -3.6747553, -3.7659705, 6.7654552, 2.0573092, 0.6338553, 6.34328, -6.047862, -0.5955341, -5.090883, -10.698972, -0.99924344, -2.3775327, -2.4674616, 0.70818084, -2.4881525, 2.9095998, -9.855811, -0.57291234, 7.709623, 7.917493, -1.5765494, 12.528931, 8.3131, 17.14767, -0.7353941, -3.4648402, -0.86669195, 2.204483, 0.91970766, -0.21738544, 4.9712753, -0.9543555, 2.7695718, 4.298482, -4.518572, 8.71062, 2.354512, -4.0580378, 0.3128378, 8.260591, -3.227196, -1.468644, -0.9634521, -0.22802265, -6.0204988, -3.161431, 0.4183467, -5.6339564, -1.793592, -10.931556, 0.9016953, -4.6181116, 0.34925088, -7.3069196, -0.68601143, -5.3798437, -7.2715106, 6.4609756, -2.331606, 2.559343, 0.74353397, 4.7186437, -1.3672069, 3.7938619, -1.6476473, -2.3876286, 0.9800127, 4.8672266, -3.5541766, 1.6889145, -5.3168917, 5.225214, 5.786194, -3.6943362, 3.1404045, 1.5667737, -6.53754, 10.655074, 1.9665438, -2.6710315, -7.2964478, -0.06155831, -1.0515051, -3.8465307, 0.31120464, -2.2075243, 3.758048, 16.740057, -3.242626, -0.78621596, -1.7105806, -6.1364746, -12.045445, -9.317872, 1.6534698, -7.982636, 1.2758802, 0.44874343, 8.249254, 7.665387, -3.6444862, -1.458851, 3.0724099, 5.237047, 2.2367852, -4.628819, 2.2844563, -3.5600092, 5.6040626, 4.6064377, 5.8036313, 2.516544, -7.1462293, 0.61613363, -2.252139, -8.564062, -2.4067917, -1.1110994, -5.679913, -3.802304, -2.1357374, -1.8486506, 1.4200228, 0.25348327, -4.3411922, 11.874627, 7.6119967, -0.38366476, 6.3590417, 5.103949, 4.770087, 0.9576347, 4.8307853, -2.8504584, 4.3833103, -2.8354442, 3.0878096, -0.7844884, 0.81957394, 9.729682, -1.7481557, -0.04654137, 1.0249834, -7.8602133, -6.2685456, 1.8640699, -6.221731, -2.7769165, 10.108798, -0.5519361, 1.6500386, 7.2599587, -3.049358, 2.7386358, -4.051545, 6.6098676, -1.0050513, -12.044271, 0.79388875, -2.9755988, -3.168039];

    params_sim.disable_track("turn_right_smooth");
    params_sim.disable_track("straight");
    params_sim.disable_all_tracks();
    params_sim.enable_track("complex");
    params_sim.simulation_simple_physics = 0.0;
    // params_sim.tracks_enable_mirror = false;
    params_sim.nn.use_ranking_network = true;
    // params_sim.simulation_stop_penalty.value = 0.1;
    params_sim.nn.ranking_hidden_layers = vec![10, 5];
    params_sim.simulation_steps_quota = 3500;
    params_sim.eval_add_other_physics = vec![
        //     PhysicsPatch {
        //         traction: Some(0.15),
        //         ..PhysicsPatch::default()
        //     },
        //     PhysicsPatch {
        //         traction: Some(0.5),
        //         ..PhysicsPatch::default()
        //     },
        //     PhysicsPatch {
        //         traction: Some(1.0),
        //         ..PhysicsPatch::default()
        //     },
        //     PhysicsPatch {
        //         friction_coef: Some(1.0),
        //         ..PhysicsPatch::default()
        //     },
        //     PhysicsPatch {
        //         friction_coef: Some(0.0),
        //         ..PhysicsPatch::default()
        //     },
        PhysicsPatch {
            acceleration_ratio: Some(1.0),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            acceleration_ratio: Some(0.6),
            ..PhysicsPatch::default()
        },
    ];
    save_runs(
        vec![evolve_by_cma_es_custom(
            &params_sim,
            &params_phys,
            // &random_input_by_len(params_sim.nn.get_nns_len() + OTHER_PARAMS_SIZE, 1.),
            &input,
            30,
            300,
            Some(0.1),
            None,
        )],
        "ranking_network6",
    );

    return;

    params_sim.simulation_simple_physics = 0.0;
    params_sim.start_places_enable = true;
    params_sim.start_places_disable_all_tracks();
    params_sim.start_places_enable_track("complex");
    params_sim.start_places_enable_track("turn_left_90");
    params_sim.start_places_enable_track("turn_left_180");
    params_sim.evolution_generation_count = 300;
    params_sim.evolution_population_size = 30;
    test_params_sim(&params_sim, &params_phys, "nn2_default_places_hard_physics");
    params_sim = params_sim_copy.clone();

    return;

    params_sim.nn.pass_dirs_diff = true;
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_dirs_diff");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_dirs_second_layer = true;
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_dirs_second_layer");
    params_sim = params_sim_copy.clone();

    params_sim.start_places_enable = true;
    params_sim.start_places_disable_all_tracks();
    params_sim.start_places_enable_track("complex");
    params_sim.start_places_enable_track("turn_left_90");
    params_sim.start_places_enable_track("turn_left_180");
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_start_places");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_current_track = true;
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_pass_current_track");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_current_segment = true;
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_pass_current_segment");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_dirs_diff = true;
    params_sim.nn.pass_dirs_second_layer = true;
    params_sim.nn.pass_current_segment = true;
    test_params_sim_evolve_simple(&params_sim, &params_phys, "nn2_pass_everything");
    params_sim = params_sim_copy.clone();
}

pub const RUN_EVOLUTION: bool = true;
pub const RUN_FROM_PREV_NN: bool = false;
const ONE_THREADED: bool = false;
const PRINT: bool = false;
const PRINT_EVERY_10: bool = false;
const PRINT_EVERY_10_ONLY_EVALS: bool = true;
const PRINT_EVALS: bool = true;

const RUNS_COUNT: usize = 100;
const POPULATION_SIZE: usize = 30;
const GENERATIONS_COUNT: usize = 100;
pub const OTHER_PARAMS_SIZE: usize = 1;
