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
use differential_evolution::self_adaptive_de;
use differential_evolution::Population;
use egui::emath::RectTransform;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::DragValue;
use egui::Painter;
use egui::Shape;
use egui::Slider;
use egui::Stroke;
use egui::Ui;
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
    pub value: fxx,
    pub start: fxx,
    pub end: fxx,
}

impl Clamped {
    pub fn new(value: fxx, start: fxx, end: fxx) -> Self {
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
    pub hidden_layers: Vec<LayerDescription>,
    pub inv_distance: bool,
    pub inv_distance_coef: fxx,
    pub inv_distance_pow: fxx,
    pub view_angle_ratio: fxx,
    pub pass_current_physics: bool,

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
    pub autoencoder_hidden_layers: Vec<LayerDescription>,

    pub pass_time_mods: Vec<fxx>,

    pub use_ranking_network: bool,
    pub ranking_hidden_layers: Vec<LayerDescription>,
    pub rank_without_physics: bool,
    pub rank_close_to_zero: bool,

    pub output_discrete_action: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, Default)]
pub struct PhysicsPatch {
    pub simple_physics: Option<fxx>,
    pub traction: Option<fxx>,
    pub acceleration_ratio: Option<fxx>,
    pub friction_coef: Option<fxx>,
    pub turn_speed: Option<fxx>,
    pub name: Option<String>,
}

impl PhysicsPatch {
    pub fn get_text(name: &str, value: Option<fxx>) -> String {
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

    pub fn simple_physics(value: fxx) -> Self {
        Self {
            simple_physics: Some(value),
            ..Self::default()
        }
    }

    pub fn simple_physics_ignored(value: fxx) -> Self {
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
    pub simulation_simple_physics: fxx,
    pub simulation_enable_random_nn_output: bool,
    pub simulation_random_output_range: fxx,
    pub simulation_use_output_regularization: bool,
    pub simulation_random_output_second_way: bool,
    pub random_output_probability: fxx,

    pub evolution_sample_mean: bool,
    pub evolution_generation_count: usize,
    pub evolution_population_size: usize,
    pub evolution_learning_rate: f64,
    pub evolution_distance_to_solution: f64,
    pub evolution_simple_add_mid_start: bool,
    pub evolution_start_input_range: fxx,

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
                |elem, ui, _| egui_usize(ui, &mut elem.size),
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
    const CAR_OUTPUT_SIZE_CONVERTED: usize = 4;
    const POSSIBLE_ACTIONS_COUNT: usize = POSSIBLE_ACTIONS.len();
    const PHYSICS_COEFS: usize = 4;

    pub fn get_ranking_input_size(&self) -> usize {
        if self.rank_without_physics {
            Self::CAR_INPUT_SIZE + self.dirs_size * 2 + Self::CAR_OUTPUT_SIZE_CONVERTED
        } else {
            Self::CAR_INPUT_SIZE * 2 + self.dirs_size * 3 + Self::CAR_OUTPUT_SIZE_CONVERTED
        }
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
                + self.pass_current_physics as usize * Self::PHYSICS_COEFS
        }
    }

    pub fn get_total_output_neurons(&self) -> usize {
        if self.use_ranking_network {
            1
        } else {
            if self.output_discrete_action {
                Self::POSSIBLE_ACTIONS_COUNT
            } else {
                Self::CAR_OUTPUT_SIZE + self.pass_next_size
            }
        }
    }

    pub fn get_nn_sizes(&self) -> Vec<LayerDescription> {
        if self.use_ranking_network {
            std::iter::once(LayerDescription::none(self.get_total_input_neurons()))
                .chain(self.ranking_hidden_layers.iter().copied())
                .chain(std::iter::once(LayerDescription::none(
                    self.get_total_output_neurons(),
                )))
                .collect()
        } else {
            std::iter::once(LayerDescription::none(self.get_total_input_neurons()))
                .chain(self.hidden_layers.iter().copied())
                .chain(std::iter::once(LayerDescription::none(
                    self.get_total_output_neurons(),
                )))
                .collect()
        }
    }

    pub fn get_nn_autoencoder_input_sizes(&self) -> Vec<LayerDescription> {
        std::iter::once(LayerDescription::none(self.dirs_size))
            .chain(self.autoencoder_hidden_layers.iter().copied())
            .chain(std::iter::once(LayerDescription::none(
                self.autoencoder_exits,
            )))
            .collect()
    }

    pub fn get_nn_autoencoder_output_sizes(&self) -> Vec<LayerDescription> {
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

        ui.label("Random output probability:");
        egui_0_1(ui, &mut self.random_output_probability);
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
        vecx2(
            rng.gen_range(-pos_range..pos_range),
            rng.gen_range(-pos_range..pos_range),
        ),
        rng.gen_range(-angle_speed_range..angle_speed_range),
        vecx2(
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
    car.change_position(
        current_angle - start_angle,
        a - car_start,
        0.,
        vecx2(0., 0.),
    );
    car
}

#[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct PointsStorage {
    pub is_reward: bool,
    pub points: Vec<Vecx2>,
}

impl PointsStorage {
    pub fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        ui.selectable_value(&mut self.is_reward, false, "Walls");
        ui.selectable_value(&mut self.is_reward, true, "Rewards");
        egui_array_inner(&mut self.points, ui, data_id, egui_pos2, false);
    }
}

pub fn egui_pos2(pos: &mut Vecx2, ui: &mut Ui, data_id: egui::Id) {
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
                    vecx2(-14.23, 17.90),
                    vecx2(5635.54, 246.73),
                    vecx2(5640.00, -98.93),
                    vecx2(6238.29, -76.85),
                    vecx2(6273.62, 485.02),
                    vecx2(5630.75, 488.32),
                    vecx2(-100.47, 409.92),
                    vecx2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![vecx2(255.56, 251.16), vecx2(5484.87, 358.62)],
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
                    vecx2(84.64, -1.89),
                    vecx2(1227.77, -3.24),
                    vecx2(2054.61, 103.97),
                    vecx2(2971.24, 374.67),
                    vecx2(3590.36, 815.56),
                    vecx2(3966.93, 1354.28),
                    vecx2(4023.21, 1911.76),
                    vecx2(4039.29, 2559.02),
                    vecx2(4052.83, 2951.00),
                    vecx2(2972.12, 2941.26),
                    vecx2(2973.34, 2549.38),
                    vecx2(3669.42, 2560.36),
                    vecx2(3660.04, 1903.72),
                    vecx2(3590.36, 1490.97),
                    vecx2(3282.13, 1095.64),
                    vecx2(2814.44, 787.42),
                    vecx2(1999.67, 558.26),
                    vecx2(1249.21, 438.99),
                    vecx2(86.01, 445.69),
                    vecx2(84.67, -1.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(251.92, 250.12),
                    vecx2(1234.09, 215.53),
                    vecx2(2034.77, 349.62),
                    vecx2(2942.04, 604.14),
                    vecx2(3430.01, 958.38),
                    vecx2(3786.43, 1417.46),
                    vecx2(3839.26, 1887.05),
                    vecx2(3851.00, 2485.98),
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
                    vecx2(80.34, -5.05),
                    vecx2(997.84, -6.53),
                    vecx2(997.84, -1646.50),
                    vecx2(994.08, -2093.93),
                    vecx2(1880.17, -2085.06),
                    vecx2(1878.69, -1653.10),
                    vecx2(1457.33, -1649.46),
                    vecx2(1466.19, 534.11),
                    vecx2(87.06, 524.75),
                    vecx2(78.86, -5.05),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(258.24, 250.23),
                    vecx2(785.25, 251.82),
                    vecx2(1063.20, 200.65),
                    vecx2(1211.43, 80.65),
                    vecx2(1231.00, -149.45),
                    vecx2(1220.93, -1508.56),
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
                    vecx2(68.68, -37.37),
                    vecx2(68.39, 495.62),
                    vecx2(1628.53, 506.55),
                    vecx2(1626.82, -437.56),
                    vecx2(70.41, -430.71),
                    vecx2(75.44, -1071.74),
                    vecx2(-334.32, -1068.79),
                    vecx2(-332.84, -36.24),
                    vecx2(68.70, -37.34),
                    vecx2(1219.76, -44.18),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(251.38, 247.87),
                    vecx2(987.99, 246.03),
                    vecx2(1315.29, 175.24),
                    vecx2(1449.97, -39.40),
                    vecx2(1328.72, -223.41),
                    vecx2(953.16, -260.21),
                    vecx2(238.07, -248.15),
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
                    vecx2(550.85, -8.85),
                    vecx2(552.00, 501.00),
                    vecx2(-2048.97, 486.81),
                    vecx2(-2051.63, -17.28),
                    vecx2(521.62, -12.89),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(-200.11, 253.13),
                    vecx2(-959.28, 250.08),
                    vecx2(-1433.05, 248.37),
                    vecx2(-1783.67, 248.37),
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
                    vecx2(110.72, 101.35),
                    vecx2(1391.74, -7.79),
                    vecx2(1805.02, -432.51),
                    vecx2(1927.44, -1337.17),
                    vecx2(2159.11, -1920.08),
                    vecx2(2786.96, -2298.42),
                    vecx2(3761.07, -2298.42),
                    vecx2(4246.98, -1890.44),
                    vecx2(4414.30, -1315.14),
                    vecx2(4428.05, -613.78),
                    vecx2(4150.72, 46.32),
                    vecx2(4778.57, 322.66),
                    vecx2(4638.04, 662.90),
                    vecx2(3654.31, 253.14),
                    vecx2(3804.62, -109.54),
                    vecx2(4024.65, -666.50),
                    vecx2(4017.78, -1223.46),
                    vecx2(3839.00, -1718.54),
                    vecx2(3694.60, -1917.94),
                    vecx2(2876.35, -1911.06),
                    vecx2(2507.34, -1652.07),
                    vecx2(2351.48, -1347.23),
                    vecx2(2285.36, -489.60),
                    vecx2(1992.89, 117.08),
                    vecx2(1633.33, 401.70),
                    vecx2(976.30, 512.67),
                    vecx2(100.88, 529.78),
                    vecx2(110.83, 101.35),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(257.32, 247.26),
                    vecx2(1508.32, 197.59),
                    vecx2(2056.80, -456.76),
                    vecx2(2158.07, -1332.80),
                    vecx2(2369.81, -1773.12),
                    vecx2(2813.37, -2095.70),
                    vecx2(3726.17, -2087.27),
                    vecx2(4042.62, -1794.75),
                    vecx2(4209.94, -1277.47),
                    vecx2(4215.89, -625.99),
                    vecx2(4016.48, -138.96),
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
                    vecx2(-8.88, -9.06),
                    vecx2(2150.00, -1.43),
                    vecx2(2142.86, -1431.43),
                    vecx2(3690.00, -1424.29),
                    vecx2(3687.09, 3202.46),
                    vecx2(-2832.30, 157.16),
                    vecx2(-2834.52, -3251.33),
                    vecx2(755.70, -3253.55),
                    vecx2(751.49, -4097.08),
                    vecx2(1868.51, -4094.22),
                    vecx2(1916.49, -2829.04),
                    vecx2(751.27, -2828.04),
                    vecx2(-2377.98, -2861.28),
                    vecx2(-1792.91, -341.48),
                    vecx2(-11.10, 580.45),
                    vecx2(-8.88, -9.06),
                    vecx2(-11.10, 580.45),
                    vecx2(2883.23, 591.53),
                    vecx2(2883.23, -862.84),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(249.86, 250.44),
                    vecx2(1953.63, 285.30),
                    vecx2(2266.79, 233.48),
                    vecx2(2435.79, 66.79),
                    vecx2(2485.06, -204.70),
                    vecx2(2484.85, -717.33),
                    vecx2(2609.34, -1050.41),
                    vecx2(2874.87, -1191.42),
                    vecx2(3175.06, -1060.41),
                    vecx2(3301.25, -774.23),
                    vecx2(3302.97, 488.77),
                    vecx2(2895.47, 1000.49),
                    vecx2(333.65, 1022.85),
                    vecx2(-1671.56, 191.94),
                    vecx2(-2228.68, -312.10),
                    vecx2(-2595.71, -2323.18),
                    vecx2(-2644.09, -2728.56),
                    vecx2(-2598.97, -2967.43),
                    vecx2(-2372.19, -3075.67),
                    vecx2(393.66, -3040.27),
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
                    vecx2(-77.00, 178.00),
                    vecx2(2558.78, -2103.75),
                    vecx2(2892.96, -1732.53),
                    vecx2(256.05, 574.77),
                    vecx2(-86.66, 186.21),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(430.68, 109.81),
                    vecx2(478.70, 57.43),
                    vecx2(548.55, -1.51),
                    vecx2(642.42, -88.83),
                    vecx2(725.37, -169.59),
                    vecx2(823.60, -256.91),
                    vecx2(1048.43, -459.92),
                    vecx2(1266.72, -647.64),
                    vecx2(1714.21, -1038.38),
                    vecx2(1934.68, -1237.02),
                    vecx2(2083.11, -1354.89),
                    vecx2(2242.46, -1485.87),
                    vecx2(2390.90, -1616.84),
                    vecx2(2506.59, -1719.43),
                    vecx2(2615.74, -1804.57),
                ],
            },
        ],
    )
}

pub fn track_loop() -> (String, Vec<PointsStorage>) {
    (
        "loop".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(86.06, 43.66),
                    vecx2(1392.63, 43.17),
                    vecx2(1793.14, 4.34),
                    vecx2(2145.64, -91.17),
                    vecx2(2416.52, -250.92),
                    vecx2(2413.05, -907.29),
                    vecx2(2459.93, -1110.46),
                    vecx2(2598.85, -1221.59),
                    vecx2(2732.55, -1251.11),
                    vecx2(2890.57, -1249.37),
                    vecx2(3032.96, -1172.97),
                    vecx2(3128.46, -1041.00),
                    vecx2(3168.40, -926.39),
                    vecx2(3178.82, -252.66),
                    vecx2(3177.08, 275.22),
                    vecx2(3135.41, 540.90),
                    vecx2(3018.72, 718.56),
                    vecx2(2577.61, 706.56),
                    vecx2(2618.70, 495.43),
                    vecx2(2577.65, 706.61),
                    vecx2(2262.06, 750.83),
                    vecx2(2137.43, 947.83),
                    vecx2(2022.85, 1120.70),
                    vecx2(1876.11, 1193.06),
                    vecx2(1659.02, 1174.97),
                    vecx2(1421.82, 1132.76),
                    vecx2(1287.14, 1066.42),
                    vecx2(1102.21, 971.95),
                    vecx2(954.83, 726.59),
                    vecx2(167.53, 695.21),
                    vecx2(-492.26, 847.12),
                    vecx2(-614.34, 776.66),
                    vecx2(-663.20, 688.23),
                    vecx2(-693.45, 564.90),
                    vecx2(-688.80, 404.34),
                    vecx2(-646.91, 287.99),
                    vecx2(-567.80, 153.02),
                    vecx2(-421.20, 80.89),
                    vecx2(-232.71, 39.00),
                    vecx2(86.09, 43.65),
                ],
            },
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(-225.21, 440.22),
                    vecx2(91.11, 415.14),
                    vecx2(1383.63, 435.24),
                    vecx2(1866.07, 348.80),
                    vecx2(2282.17, 248.29),
                    vecx2(2613.84, 79.44),
                    vecx2(2794.75, -107.50),
                    vecx2(2784.70, -913.57),
                ],
            },
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(2007.19, 328.93),
                    vecx2(1677.63, 857.40),
                    vecx2(1075.64, 447.04),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(862.96, 227.59),
                    vecx2(1927.17, 162.67),
                    vecx2(2390.26, 5.65),
                    vecx2(2618.69, -284.78),
                    vecx2(2596.21, -688.33),
                    vecx2(2595.15, -923.22),
                    vecx2(2782.57, -1114.72),
                    vecx2(2976.93, -966.99),
                    vecx2(2991.32, -716.79),
                    vecx2(2996.44, 59.44),
                    vecx2(2645.06, 282.04),
                    vecx2(1959.63, 814.33),
                    vecx2(1723.76, 1043.03),
                    vecx2(1225.86, 732.50),
                    vecx2(962.03, 592.78),
                    vecx2(269.51, 562.66),
                    vecx2(-54.91, 590.48),
                    vecx2(-362.20, 610.78),
                    vecx2(-524.86, 475.04),
                    vecx2(-390.01, 273.18),
                    vecx2(-70.39, 226.72),
                ],
            },
        ],
    )
}

pub fn track_straight_turn() -> (String, Vec<PointsStorage>) {
    (
        "straight_turn".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(-14.28, 17.83),
                    vecx2(6230.98, 178.47),
                    vecx2(6226.23, 514.13),
                    vecx2(8279.91, 511.92),
                    vecx2(8284.18, 1346.21),
                    vecx2(7831.66, 1351.39),
                    vecx2(7838.53, 917.79),
                    vecx2(5626.36, 905.14),
                    vecx2(5634.77, 516.46),
                    vecx2(-24.13, 409.14),
                    vecx2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(255.56, 251.16),
                    vecx2(5064.73, 336.38),
                    vecx2(5508.00, 347.74),
                    vecx2(5929.05, 519.46),
                    vecx2(6289.30, 727.45),
                    vecx2(6700.19, 736.97),
                    vecx2(7687.94, 738.33),
                ],
            },
        ],
    )
}

pub fn track_bubble_straight() -> (String, Vec<PointsStorage>) {
    (
        "bubble_straight".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(-14.23, 17.90),
                    vecx2(2599.76, 26.70),
                    vecx2(2966.35, -1636.05),
                    vecx2(3561.60, -2191.07),
                    vecx2(4591.47, -1977.75),
                    vecx2(5247.65, -1495.88),
                    vecx2(5542.16, -616.31),
                    vecx2(5592.24, 824.42),
                    vecx2(6848.82, 78.22),
                    vecx2(6597.50, -422.08),
                    vecx2(6983.78, -647.80),
                    vecx2(7460.51, 160.08),
                    vecx2(5576.49, 1328.17),
                    vecx2(5148.15, 1946.77),
                    vecx2(4013.86, 2054.23),
                    vecx2(3132.76, 1927.19),
                    vecx2(2680.58, 1198.54),
                    vecx2(2601.85, 440.19),
                    vecx2(-22.07, 424.51),
                    vecx2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(255.56, 251.16),
                    vecx2(2325.81, 275.87),
                    vecx2(3057.63, 318.99),
                    vecx2(3704.72, 628.55),
                    vecx2(4753.57, 1008.18),
                    vecx2(5653.04, 1040.02),
                    vecx2(6337.59, 638.05),
                    vecx2(7002.24, 236.07),
                ],
            },
        ],
    )
}

pub fn track_bubble_180() -> (String, Vec<PointsStorage>) {
    (
        "bubble_180".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(-14.23, 17.90),
                    vecx2(2599.76, 26.70),
                    vecx2(2966.35, -1636.05),
                    vecx2(3561.60, -2191.07),
                    vecx2(4591.47, -1977.75),
                    vecx2(5247.65, -1495.88),
                    vecx2(5542.16, -616.31),
                    vecx2(5592.24, 824.42),
                    vecx2(5576.49, 1328.17),
                    vecx2(5148.15, 1946.77),
                    vecx2(4013.86, 2054.23),
                    vecx2(3132.76, 1927.19),
                    vecx2(1337.52, 2671.57),
                    vecx2(867.80, 1516.32),
                    vecx2(1299.44, 1330.97),
                    vecx2(1598.02, 2027.67),
                    vecx2(2844.35, 1480.42),
                    vecx2(2601.85, 440.19),
                    vecx2(-22.07, 424.51),
                    vecx2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(255.56, 251.16),
                    vecx2(2325.81, 275.87),
                    vecx2(2981.46, 318.99),
                    vecx2(3406.82, 462.62),
                    vecx2(3572.69, 897.69),
                    vecx2(3406.82, 1305.58),
                    vecx2(2891.40, 1770.22),
                    vecx2(1692.98, 2278.02),
                ],
            },
        ],
    )
}

pub fn track_separation() -> (String, Vec<PointsStorage>) {
    (
        "separation".to_owned(),
        vec![
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(-14.23, 17.90),
                    vecx2(1526.19, 7.75),
                    vecx2(2181.20, 126.85),
                    vecx2(2935.45, 103.39),
                    vecx2(3657.23, 532.85),
                    vecx2(3918.87, 926.21),
                    vecx2(3852.11, 1494.61),
                    vecx2(3956.76, 2126.16),
                    vecx2(4021.72, 2763.13),
                    vecx2(3954.50, 3156.18),
                    vecx2(3178.59, 3268.06),
                    vecx2(3140.70, 2840.41),
                    vecx2(3595.41, 2802.52),
                    vecx2(3529.11, 2160.45),
                    vecx2(3489.42, 1819.41),
                    vecx2(3166.42, 1536.11),
                    vecx2(2413.97, 1007.41),
                    vecx2(2085.56, 666.37),
                    vecx2(1549.65, 448.04),
                    vecx2(-28.29, 422.55),
                    vecx2(-14.23, 17.90),
                ],
            },
            PointsStorage {
                is_reward: true,
                points: vec![
                    vecx2(255.56, 251.16),
                    vecx2(1368.96, 241.33),
                    vecx2(1919.10, 310.59),
                    vecx2(2586.74, 548.78),
                    vecx2(3025.21, 786.96),
                    vecx2(3331.97, 1048.60),
                    vecx2(3514.22, 1357.16),
                    vecx2(3691.05, 1777.60),
                    vecx2(3752.40, 2160.14),
                    vecx2(3813.75, 2775.45),
                ],
            },
            PointsStorage {
                is_reward: false,
                points: vec![
                    vecx2(2332.59, 437.51),
                    vecx2(2810.95, 439.02),
                    vecx2(3249.43, 666.38),
                    vecx2(3485.81, 955.09),
                    vecx2(3462.35, 1252.82),
                    vecx2(3078.00, 1122.90),
                    vecx2(2747.79, 900.95),
                    vecx2(2332.77, 439.02),
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
        track_loop(),
        track_straight_turn(),
        track_bubble_straight(),
        track_bubble_180(),
        track_separation(),
    ]
}

pub struct RewardPathProcessor {
    rewards: Vec<Reward>,
    max_distance: fxx,
    current_segment_fxx: fxx,
}

impl RewardPathProcessor {
    pub fn new(rewards: Vec<Reward>) -> Self {
        Self {
            rewards,
            max_distance: 0.,
            current_segment_fxx: 0.,
        }
    }

    fn process_point(&mut self, point: Vecx2, params: &SimulationParameters) -> fxx {
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
            (pos as fxx + t, dist)
        } else {
            (0., 0.)
        };
        self.current_segment_fxx = segm;

        if self.max_distance < segm {
            if params.rewards_enable_distance_integral {
                reward_sum += (self.current_segment_fxx - self.max_distance) / (dist + 1.0);
            }
            self.max_distance = self.current_segment_fxx;
        }

        reward_sum
    }

    fn reset(&mut self) {
        self.current_segment_fxx = 0.;
        self.max_distance = 0.;
    }

    fn draw(&self, point: Vecx2, painter: &Painter, to_screen: &RectTransform) {
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
                to_screen.transform_pos(point.into()),
                to_screen.transform_pos(projection1.into()),
            ],
            Stroke::new(1.0, Color32::DARK_RED),
        ));

        for (i, (a, b)) in pairs(self.rewards.iter()).enumerate() {
            painter.add(Shape::line(
                vec![
                    to_screen.transform_pos(a.center.into()),
                    to_screen.transform_pos(b.center.into()),
                ],
                Stroke::new(1.0, Color32::DARK_GREEN),
            ));
        }
    }

    pub fn distance_percent(&self) -> fxx {
        self.max_distance / (self.rewards.len() as fxx - 1.0)
    }

    pub fn all_acquired(&self) -> bool {
        (self.distance_percent() - 1.0).abs() < 1e-6
    }

    pub fn get_current_segment_fxx(&self) -> fxx {
        self.current_segment_fxx
    }

    pub fn get_rewards(&self) -> &[Reward] {
        &self.rewards
    }
}

fn convert_dir(params: &NnParameters, intersection: Option<fxx>) -> fxx {
    if params.inv_distance {
        intersection
            .map(|x| params.inv_distance_coef / (x + 1.).powf(params.inv_distance_pow))
            .unwrap_or(0.)
    } else {
        intersection.map(|x| x.max(1000.)).unwrap_or(1000.)
    }
}

pub struct NnProcessorAutoencoder {
    input: Vec<fxx>,
    output: Vec<fxx>,
    nn_input: NeuralNetwork,
    nn_output: NeuralNetwork,
    autoencoder_loss: fxx,
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

    pub fn new(params: &[fxx], nn_params: NnParameters) -> Self {
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

    pub fn process<'a>(&'a mut self, dirs: &[Option<fxx>]) -> &'a [fxx] {
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
        self.autoencoder_loss += sum / dirs.len() as fxx;

        values
    }

    pub fn get_regularization_loss(&self) -> fxx {
        self.nn_input.get_regularization() + self.nn_output.get_regularization()
    }

    pub fn get_autoencoder_loss(&self) -> fxx {
        if self.calc_counts == 0 {
            0.
        } else {
            self.autoencoder_loss / self.calc_counts as fxx
        }
    }

    pub fn reset(&mut self) {
        self.input.iter_mut().for_each(|x| *x = 0.);
        self.output.iter_mut().for_each(|x| *x = 0.);
        self.autoencoder_loss = 0.;
    }
}

#[rustfmt::skip]
const BEST_AUTOENCODER_21_10_10_10_5: [fxx; 1006] = [0.45737186,-0.17592742,0.41990036,0.3023682,0.46085286,0.09734252,0.20202918,0.18677261,0.15259826,0.08691186,0.16731122,0.10409911,-0.27700168,-0.013539409,-0.09711216,0.14397214,-0.15910262,0.21080758,0.43845347,0.07938459,0.47766662,0.49709564,0.5581285,0.5085441,0.37938422,0.13941869,-0.011706705,0.111483075,-0.3481085,-0.17621183,-0.16999112,0.33578566,0.33830214,0.3392177,0.46204475,0.43641356,0.02911971,-0.24209979,-0.13739654,0.07810422,-0.42370325,0.048519064,0.47701773,0.36765498,0.25073645,0.34227213,0.28530744,0.12449606,0.33620736,0.4206451,0.37811056,0.48892096,0.31235722,-0.019208623,0.28711075,-0.32138065,-0.48255187,-0.073856294,0.21494687,0.2527926,0.25357565,0.06038692,0.21669765,-0.4017394,0.0030092,0.027453631,-0.008625239,-0.12991595,-0.3729226,0.27464026,-0.35417527,-0.32996136,-0.3039164,-0.41292477,-0.008672744,-0.16740495,0.27102596,-0.25378257,-0.09404045,-0.34924185,0.4185815,-0.19595659,0.06775886,-0.40013322,0.0044076815,0.22488806,0.038864266,-0.38992977,-0.3962791,-0.008726857,-0.08023614,0.045806993,0.23833953,0.5205801,0.768315,0.30037403,-0.008285542,0.13156843,0.016080525,-0.41769928,-0.5351447,-0.2394889,0.4227838,0.29043153,0.3176221,-0.42664915,-0.20242606,-0.25436103,-0.005583709,0.16876525,0.41205645,-0.28080022,0.10476436,-0.31362674,-0.2677709,-0.42060456,-0.3070178,0.15683068,0.42874318,0.22397202,0.3479392,-0.08452995,0.15452468,0.3514351,-0.01399497,-0.40478998,-0.2482755,-0.1356977,-0.2107391,0.1617366,-0.24560514,-0.09844121,-0.05664088,0.016249405,-0.20677516,-0.057893697,-0.3120921,0.14034316,0.19313832,0.2763481,0.3531536,0.56920975,0.5262653,0.38728634,-0.030576818,0.6514924,-0.10670456,-0.069721915,0.25045338,-0.14655343,0.35060158,-0.10266501,0.63437945,0.32942742,0.45425716,0.074557275,0.39037,0.0637424,-0.17551796,-0.20605043,0.3715435,0.44256073,-0.0024101275,-0.19201526,-0.24129438,0.39782032,0.5097004,0.1726135,-0.3583182,-0.23892967,0.28712755,-0.21878761,0.21266371,-0.3139548,0.2520895,0.20426053,-0.38272342,-0.13531768,-0.37770674,-0.07767549,-0.079563916,0.076762915,-0.09228301,-0.15359625,0.39501822,-0.32253093,-0.05489093,-0.10004258,0.043926954,-0.21595538,0.42019904,-0.19991599,-0.2796001,-0.3063535,-0.1659193,0.11443508,-0.28578854,0.07319701,-0.2500814,-0.015817225,0.39411527,-0.14725995,0.39196688,-0.25877836,-0.04152623,-0.095975876,-0.15781116,0.028069556,-0.14534119,0.019865453,-0.06348205,-0.038866445,-0.12543958,0.0,-0.049645416,-0.008875614,-0.34252134,-0.02051096,0.0,0.0,0.027617633,0.20035297,-0.35912716,0.33826768,0.24858415,0.13768375,-0.03795594,-0.09491876,-0.5105505,-0.2659443,-0.45389396,0.11748606,0.09288216,0.32547277,-0.030020177,0.24329084,0.08851558,-0.42366457,-0.26695818,0.101017475,-0.49297047,0.36307758,-0.34514493,0.2988848,0.035871148,0.5472412,0.50963855,-0.45673212,0.40956378,-0.1742078,0.31611833,0.5084936,0.40096274,-0.1617916,0.42529443,0.28289756,0.31026813,0.1375107,0.032167792,-0.39572728,0.28746232,-0.04957891,0.20623049,0.15467656,-0.4147029,-0.11097115,0.8231028,0.20070969,0.4504164,-0.1172778,0.43438476,-0.3721964,0.4799559,0.5127816,0.22977056,0.5342281,-0.49618167,0.48291194,-0.54680806,-0.41188288,-0.46225387,0.02290225,-0.30547047,-0.4821162,0.16012192,0.38117933,0.09050703,-0.19624546,-0.15609431,0.07042986,0.3148246,-0.05269588,-0.039140135,-0.27311864,-0.5313749,-0.07278303,-0.28564167,0.40633324,0.27438074,-0.33285016,0.48366016,-0.1868916,-0.11395642,0.41317356,0.6122248,-0.17725272,0.5885771,-0.043597978,-0.12985998,-0.48262614,0.036938548,-0.23215446,-0.4586741,0.40202367,0.39871365,0.40458053,0.11369836,0.047267616,-0.33472955,-0.0736922,0.0,0.0,0.0,-0.04239895,-0.048857134,-0.07255001,0.0,0.02166893,0.003860553,0.0,0.015568614,-0.12189758,0.1656887,0.56778526,0.2573384,-0.14126685,0.26964617,0.38049787,-0.36964574,0.5429429,0.4858694,0.5392759,0.33397192,-0.5292928,0.48138225,-0.3814337,-0.26645464,-0.22139981,0.0393188,0.540257,0.3732344,0.061119914,-0.40323785,-0.027448416,-0.09904677,-0.39385843,0.51572704,0.051013887,0.010567129,-0.45928824,-0.046225548,0.35471177,-0.11607221,0.4036979,0.04851145,0.31053594,-0.20114025,-0.06318814,-0.11971743,0.2558158,0.4767382,0.06389463,0.5359721,0.27281535,0.61154187,-0.17682181,-0.5467057,-0.15335092,0.16274224,-0.2988279,-0.21623209,0.47806942,0.51816785,-0.3264413,0.3658301,0.23505229,-0.064686,0.58192694,0.036728177,-0.19700703,-0.24694583,0.36191887,0.12063205,-0.07626569,-0.3799886,-0.45023346,0.45717847,0.10571009,-0.30130303,0.34269577,-0.20633999,0.0499897,-0.097061306,-0.4296894,-0.44272336,0.5295114,0.40523547,0.47777525,-0.47180068,0.43413532,0.14695245,-0.039705575,0.15221053,-0.10159143,0.1270639,0.17898011,-0.32885066,-0.5786,-0.023145985,0.24564582,0.2326225,-0.0007247329,0.21068501,-0.068891704,-0.2498591,0.38694972,0.25435388,-0.26001695,-0.53119427,0.42076278,-0.020152673,0.0,0.0,-0.04903647,-0.021485018,0.0063240416,0.0,-0.013520969,-0.04661106,0.0,0.25270316,-0.044974983,-0.515672,0.079448774,0.50788134,0.65939474,0.326761,-0.44779658,-0.6086799,-0.48577356,-0.47844335,-0.041437924,-0.6244541,0.6047946,0.60107666,-0.8273604,0.25618458,-0.1575329,0.027038105,-0.23821822,0.5200817,0.5768277,0.2281723,-0.57039213,-0.59204894,-0.4897523,-0.20118606,-0.10484761,-0.14021656,-0.38588452,0.3546,0.61131567,-0.4037295,0.7011073,-0.33956572,0.04620619,-0.32616884,-0.4394969,-0.49657133,-0.33803958,0.40583158,-0.35029602,-0.5258989,-0.026526408,0.27576658,-0.792013,-0.20139068,0.011485338,-0.31658253,0.2183519,-0.018916938,0.0050536706,-0.02281617,-0.038263872,0.0310839,0.56170845,-0.6282362,-0.5168994,-0.15849586,0.096899875,-0.060975116,0.33497098,0.17076254,0.3759598,-0.5072324,0.13570258,-0.1473247,-0.17493798,0.10895595,0.43225223,-0.5622761,0.22041798,0.25107282,-0.3827208,-0.3367378,-0.53930104,0.06395388,-0.21373653,-0.13711393,-0.17852244,-0.014904106,0.7355005,0.25485387,0.20877622,0.59275186,-0.28384012,0.39728564,0.021546245,-0.22174235,-0.5313238,0.29870403,-0.7480616,0.23096626,0.35752147,0.31776556,0.60854894,0.7437414,0.40992495,0.0069172373,-0.6218858,0.4920594,0.07133994,-0.070414774,0.6690848,0.041190106,0.016126594,-0.0058748773,-0.07512411,0.0,0.0,-0.005659065,-0.011941007,-0.006117512,-0.031855706,-0.040702112,0.648664,-0.41656962,0.22611614,0.07048929,0.38365042,0.43846026,-0.4582126,0.52142835,-0.41739795,0.20872425,-0.74500805,-0.5456883,0.29133123,0.0045635104,0.45645988,0.0218608,0.37450957,0.19543046,-0.2703051,0.37894905,0.3440333,0.42775798,0.06840312,-0.14772394,-0.13403201,0.49949837,0.16941537,-0.42480052,0.33693975,0.5180814,-0.23840593,0.34738067,-0.14095815,0.4771434,0.0059478283,0.3157413,-0.4141257,-0.32466415,0.30409428,0.073748946,0.30155033,-0.17202307,0.31254074,0.34113133,-0.4173333,-0.30333576,0.3580578,-0.058534212,0.42712164,0.46393025,0.19257312,0.91011685,0.17239583,-0.01824528,-0.12686616,-0.19433935,-0.3245527,0.43490216,0.22452417,0.1861319,-0.4840148,-0.01780321,0.18180817,0.38608533,-0.09568596,-0.057836026,0.31867123,0.5001768,-0.5310913,-0.23036578,-0.18935914,0.44626456,-0.014953927,-0.005733669,-0.4992405,0.40648514,0.236542,-0.47628355,-0.32074094,0.43714648,0.073094346,0.43527675,0.13248475,0.12132627,0.37790197,-0.17227016,-0.52738965,0.039165292,-0.16698352,0.4481608,-0.061499804,0.26578775,-0.3720556,0.3283339,0.43164974,0.27652317,0.059041668,0.36649668,0.30042675,0.5507893,0.031096091,-0.023072015,0.015341973,0.063868605,0.05314186,-0.10451103,0.012260199,-0.033602644,-0.007871839,-0.028260214,0.4560528,-0.43863538,-0.1972818,-0.54110587,-0.11297603,0.5834811,-0.3068129,0.16640697,0.15583868,0.306662,-0.1322146,-0.25152695,-0.36682713,-0.40586734,-0.52713156,0.08048719,-0.4596381,-0.16805714,0.22507417,0.01995802,0.41539112,0.12115909,0.61171275,0.17476349,-0.17620562,-0.38986272,-0.1986934,-0.07216763,-0.2522651,0.1745536,-0.00065242837,0.33300617,-0.16377176,0.47842458,0.1233035,-0.11198796,-0.5111505,0.44108307,-0.3224152,-0.34009638,-0.19228707,-0.30730093,0.25576028,-0.10244663,-0.31067827,0.37724394,0.49409118,0.0061814627,0.5397924,0.32008857,-0.30164802,-0.50917286,0.17394805,-0.0714896,-0.44997922,0.15607458,-0.53747565,-0.5079738,-0.0361138,0.21564847,-0.57721186,-0.90376776,0.52751344,0.44228637,-0.07307916,-0.33051163,0.03798145,-0.7166991,0.0744902,0.533621,-0.58328587,0.2642905,0.30252588,0.12714164,-0.34574488,0.023801422,-0.13283314,0.2592124,0.11557618,0.27972016,0.22628273,0.4573506,-0.03671455,0.5283449,0.016419219,-0.30363163,-0.51035285,0.22357996,-0.4086221,-0.15721984,0.47282434,-0.09929528,0.5269532,0.14871538,0.47924593,0.12536359,0.06255887,-0.4338604,-0.17016983,0.039393302,-0.021478916,0.0,-0.0011150183,-0.02002353,0.013970958,0.0,0.08725746,-0.043776356,0.14214514,0.026229413,-0.2595059,-0.12774545,-0.0494774,-0.12511805,0.4504645,-0.08772072,-0.73110783,0.125394,0.31506735,0.42245546,0.3609071,-0.2768759,0.36766338,-0.24215254,0.1554749,0.31662637,-0.4220921,0.14024962,0.53150356,-0.052198645,0.24542153,-0.27837205,0.301992,-0.30569372,0.24076378,-0.045823783,-0.43873274,0.102294445,0.28654665,0.004846038,0.00082837255,-0.2729205,-0.24653284,0.11174075,-0.07795748,0.24567664,-0.62129486,0.5075251,-0.5105555,0.5848545,0.38590983,-0.16161081,0.2442681,0.37855762,0.1243355,0.25341237,-0.23978654,0.18581568,-0.5839694,-0.06217745,0.018985085,-0.0029155314,-0.11399212,0.34180018,0.38483477,0.06241387,-0.28890932,0.11825153,-0.3466623,0.17485446,0.20287298,0.3890488,-0.036215372,0.46085256,0.45194423,0.42120856,-0.021849155,0.019964645,0.087934755,-0.08417687,0.2719986,0.105430365,-0.38356945,0.23822773,0.23885334,0.3386371,0.09151632,0.3015638,0.5727032,0.07485627,0.48018882,0.08769888,-0.04850754,-0.36953154,0.1649228,-0.015656859,0.30176848,0.2974766,0.5414424,-0.2573507,-0.0125634745,0.25059485,0.0073877983,0.32782456,0.39748195,-0.09214687,0.08788801,0.24517,0.113038145,-0.12556338,0.14891444,-0.3637699,0.16135764,-0.24549964,-0.048727535,-0.4137956,-0.064132504,0.6118638,-0.39429182,-0.08805484,-0.18773945,0.2982645,0.11784611,-0.25582108,-0.112161316,0.05399874,-0.3052031,0.4420171,-0.3254955,0.26785895,-0.41021463,0.2770481,0.4295652,0.01817906,0.16657665,-0.093443215,-0.3001354,-0.18076026,-0.14202349,0.17395072,-0.115156375,-0.09231439,0.49168733,-0.006198864,-0.27805942,-0.28996944,-0.03537516,-0.21342821,0.054626282,0.1938549,-0.08305472,-0.106129885,0.6057189,-0.08438851,-0.18504211,0.12727177,0.17185001,-0.4829378,0.1772602,0.071630456,-0.11114428,0.41013658,0.26588863,-0.067258045,-0.40872657,-0.32674155,0.24508859,-0.29301828,0.24470176,0.36417535,-0.1254141,-0.3829799,0.3146924,-0.07486214,-0.22775005,-0.41284937,0.38637522,-0.40476888,0.16482165,0.23975624,-0.33793283,0.3607992,0.049563147,-0.07752391,-0.07130672,-0.43954402,0.35074002,-0.267593,0.007313381,0.41151854,-0.33716315,-0.35077953,0.11318766,-0.3380483,0.20592208,0.035967678,0.39766645,-0.21563718,-0.1851213,0.22164433,-0.31974375,-0.4119708,0.09578852,-0.05242302,0.23938423,-0.23844309,0.42013696,-0.14793545,-0.2336527,0.19472612,-0.12992854,0.32164264,0.09721068,0.4162817,0.016214421,0.25102162,0.4798254,-0.012202531,-0.17921817,0.1839505,0.12687603,0.1467754,0.11232976,0.15392108,0.09507806,0.0960064,0.054173872,0.10056898,0.08604917,0.12875709,0.2974497,0.13084707,0.03666326,0.001766446,-0.039353047,-0.016559495,-0.014885214,0.016923485,0.020065885,0.03429328,0.04688359];

pub struct NnProcessor {
    params: NnParameters,
    input_values: Vec<fxx>,
    next_values: Vec<fxx>,
    prev_output: Vec<fxx>,
    prev_dirs: Vec<Option<fxx>>,
    nn: NeuralNetwork,
    simple_physics: fxx,

    calc_counts: usize,

    output_diff_loss: fxx,
    output_regularization_loss: fxx,

    current_track: usize,

    nn_autoencoder: NnProcessorAutoencoder,
}

pub struct SimulationVars<'a> {
    car: &'a Car,
    walls: &'a [Wall],
    params_phys: &'a PhysicsParameters,
    dirs: &'a [Vecx2],
}

pub const POSSIBLE_ACTIONS: [CarInput; 9] = [
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
];

impl NnProcessor {
    pub fn new_zeros(nn_params: NnParameters, simple_physics: fxx, current_track: usize) -> Self {
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
        params: &[fxx],
        nn_params: NnParameters,
        simple_physics: fxx,
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
        time_passed: fxx,
        distance_percent: fxx,
        dpenalty: fxx,
        dirs: &[Option<fxx>],
        dirs_second_layer: &[Option<fxx>],
        current_segment_fxx: fxx,
        internals: &InternalCarValues,
        params_sim: &SimulationParameters,
        params_phys: &PhysicsParameters,
    ) -> &[fxx] {
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
            for y in &internals.to_fxx() {
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
                *input_values_iter.next().unwrap() = (i == self.current_track) as usize as fxx;
            }
        }

        if self.params.pass_current_segment {
            for track in 0..self.params.max_tracks {
                for i in 0..self.params.max_segments {
                    *input_values_iter.next().unwrap() = if track == self.current_track {
                        if i as fxx <= current_segment_fxx && current_segment_fxx < (i + 1) as fxx {
                            1. + (current_segment_fxx - i as fxx)
                        } else {
                            0.
                        }
                    } else {
                        0.
                    };
                }
            }
        }

        if self.params.pass_current_physics {
            *input_values_iter.next().unwrap() = params_phys.traction_coefficient;
            *input_values_iter.next().unwrap() = params_phys.friction_coefficient;
            *input_values_iter.next().unwrap() = params_phys.acceleration_ratio;
            *input_values_iter.next().unwrap() = params_phys.simple_physics_ratio;
        }

        assert!(input_values_iter.next().is_none());

        &self.input_values
    }

    pub fn process(
        &mut self,
        time_passed: fxx,
        distance_percent: fxx,
        dpenalty: fxx,
        dirs: &[Option<fxx>],
        dirs_second_layer: &[Option<fxx>],
        current_segment_fxx: fxx,
        internals: &InternalCarValues,
        params_sim: &SimulationParameters,
        simulation_vars: SimulationVars,
    ) -> CarInput {
        if self.params.use_ranking_network {
            let result = POSSIBLE_ACTIONS
                .iter()
                .map(|action| {
                    let mut input_values_iter = self.input_values.iter_mut();

                    for intersection in dirs {
                        *input_values_iter.next().unwrap() =
                            convert_dir(&self.params, *intersection);
                    }

                    for y in &internals.to_fxx() {
                        *input_values_iter.next().unwrap() = *y;
                    }

                    *input_values_iter.next().unwrap() = action.brake;
                    *input_values_iter.next().unwrap() = action.acceleration;
                    *input_values_iter.next().unwrap() = action.remove_turn;
                    *input_values_iter.next().unwrap() = action.turn;

                    if self.params.rank_without_physics {
                        for (prev, current) in self.prev_dirs.iter().zip(dirs.iter()) {
                            *input_values_iter.next().unwrap() = convert_dir(&self.params, *prev)
                                - convert_dir(&self.params, *current);
                        }
                    } else {
                        let mut car = simulation_vars.car.clone();
                        let walls = &simulation_vars.walls;
                        let params_phys = &simulation_vars.params_phys;
                        let precomputed_dirs = &simulation_vars.dirs;
                        car.process_input(action, params_phys);

                        for i in 0..params_phys.steps_per_time {
                            let time = params_phys.time / params_phys.steps_per_time as fxx;
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

                        for y in &car.get_internal_values().to_fxx() {
                            *input_values_iter.next().unwrap() = *y;
                        }
                    }

                    assert!(input_values_iter.next().is_none());

                    let values = self.nn.calc(&self.input_values);
                    assert!(values.len() == 1);

                    (action, values[0])
                })
                .max_by(|(_, value1), (_, value2)| {
                    if self.params.rank_close_to_zero {
                        value1.abs().partial_cmp(&value2.abs()).unwrap().reverse()
                    } else {
                        value1.partial_cmp(value2).unwrap()
                    }
                })
                .unwrap()
                .0
                .clone();

            if self.params.rank_without_physics {
                for (prev, current) in self.prev_dirs.iter_mut().zip(dirs.iter()) {
                    *prev = *current;
                }
            }

            return result;
        }

        self.calc_counts += 1;

        self.calc_input_vector(
            time_passed,
            distance_percent,
            dpenalty,
            dirs,
            dirs_second_layer,
            current_segment_fxx,
            internals,
            params_sim,
            simulation_vars.params_phys,
        );
        let values = self.nn.calc(&self.input_values);

        if self.params.output_discrete_action {
            POSSIBLE_ACTIONS[values
                .iter()
                .enumerate()
                .max_by(|(_, value1), (_, value2)| value1.partial_cmp(value2).unwrap())
                .unwrap()
                .0]
                .clone()
        } else {
            let mut output_values_iter = values.iter();

            let prev_output_len = self.prev_output.len();
            for y in &mut self.prev_output {
                let new_value = *output_values_iter.next().unwrap();
                self.output_diff_loss += (*y - new_value).abs() / prev_output_len as fxx;
                *y = new_value;
                if new_value.abs() > 10. {
                    self.output_regularization_loss +=
                        (new_value.abs() - 10.) / prev_output_len as fxx;
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

            CarInput::from_fxx(&values[0..NnParameters::CAR_OUTPUT_SIZE])
        }
    }

    pub fn get_autoencoder_loss(&self) -> fxx {
        self.nn_autoencoder.get_autoencoder_loss()
    }

    pub fn get_regularization_loss(&self) -> fxx {
        self.nn.get_regularization() + self.nn_autoencoder.get_regularization_loss()
    }

    pub fn get_output_diff_loss(&self) -> fxx {
        if self.calc_counts == 0 {
            0.
        } else {
            self.output_diff_loss / self.calc_counts as fxx
        }
    }

    pub fn get_output_regularization_loss(&self) -> fxx {
        if self.calc_counts == 0 {
            0.
        } else {
            self.output_regularization_loss / self.calc_counts as fxx
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
    current: (Option<fxx>, Option<fxx>),
    new_value: Option<fxx>,
) -> (Option<fxx>, Option<fxx>) {
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
    pub penalty: fxx,
    prev_penalty: fxx,
    pub reward: fxx,
    pub time_passed: fxx,
    pub dirs: Vec<Vecx2>,
    dirs_values: Vec<Option<fxx>>,
    dirs_values_second_layer: Vec<Option<fxx>>,
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
                    (i as fxx / (params.nn.dirs_size - 1) as fxx - 0.5)
                        * TAUx
                        * params.nn.view_angle_ratio
                })
                .map(|t| rotate_around_origin(vecx2(1., 0.), t))
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
        observe_distance: &mut impl FnMut(Vecx2, Vecx2, fxx, Option<fxx>),
        get_input: &mut impl FnMut(
            fxx,
            fxx,
            fxx,
            &[Option<fxx>],
            &[Option<fxx>],
            &InternalCarValues,
            fxx,
            SimulationVars,
        ) -> CarInput,
        drift_observer: &mut impl FnMut(usize, Vecx2, fxx),
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

        let mut input = get_input(
            self.time_passed,
            self.reward_path_processor.distance_percent(),
            self.penalty - self.prev_penalty,
            &self.dirs_values,
            &self.dirs_values_second_layer,
            &self.car.get_internal_values(),
            self.reward_path_processor.get_current_segment_fxx(),
            SimulationVars {
                car: &self.car,
                walls: &self.walls,
                params_phys,
                dirs: &self.dirs,
            },
        );

        let mut rng = thread_rng();
        if rng.gen_range(0.0..1.0) < params_sim.random_output_probability {
            input = POSSIBLE_ACTIONS[rng.gen_range(0..POSSIBLE_ACTIONS.len())].clone();
        }
        self.car.process_input(&input, params_phys);
        self.prev_penalty = self.penalty;

        for i in 0..params_phys.steps_per_time {
            let time = params_phys.time / params_phys.steps_per_time as fxx;

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
                    .map(|p| to_screen.transform_pos(p.into()))
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
    penalty: fxx,
    reward: fxx,
    early_finish_percent: fxx,
    distance_percent: fxx,
    all_acquired: bool,
    simple_physics: fxx,
    simple_physics_raw: fxx,
    autoencoder_loss: fxx,
    regularization_loss: fxx,
    output_diff_loss: fxx,
    output_regularization_loss: fxx,
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
) -> fxx {
    let evals = evals
        .iter()
        .filter(|x| !x.name.ends_with("_ignore"))
        .collect::<Vec<_>>();
    if params.rewards_second_way {
        let all_acquired = evals.iter().all(|x| x.all_acquired);
        let len = evals.len() as fxx;
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
            .map(|x| x.all_acquired as usize as fxx)
            .sum::<fxx>();
        // let regularization_sum = evals.iter().map(|x| x.regularization_loss).sum::<fxx>();
        let regularization_sum = evals
            .iter()
            .map(|x| x.output_regularization_loss)
            .sum::<fxx>()
            / evals.len() as fxx;
        let penalty_level_1 = evals
            .iter()
            .map(|x| (x.penalty < 15.) as usize as fxx)
            .sum::<fxx>();
        let penalty_level_2 = evals
            .iter()
            .map(|x| (x.penalty < 5.) as usize as fxx)
            .sum::<fxx>();
        let penalty_level_3 = evals
            .iter()
            .map(|x| (x.penalty < 2.) as usize as fxx)
            .sum::<fxx>();
        let penalty_level_4 = evals
            .iter()
            .map(|x| (x.penalty == 0.) as usize as fxx)
            .sum::<fxx>();
        let penalty_levels_len = 4. * len;
        let penalty_levels =
            (penalty_level_1 + penalty_level_2 + penalty_level_3 + penalty_level_4)
                / penalty_levels_len;
        let smooth_metrics = evals.iter().map(|x| x.to_fxx_second_way()).sum::<fxx>();
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
        let len = evals.len() as fxx;
        let tracks_sum = evals
            .iter()
            .map(|x| x.all_acquired as usize as fxx + x.distance_percent)
            .sum::<fxx>();
        let penalty_sum = evals
            .iter()
            .map(|x| (x.penalty == 0.) as usize as fxx + 1. / (1. + x.penalty))
            .sum::<fxx>();
        let early_finish_reward = evals
            .iter()
            .map(|x| {
                if x.penalty == 0. {
                    x.early_finish_percent * 2.
                } else {
                    0.
                }
            })
            .sum::<fxx>();
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
                    .to_fxx(params)
                } else {
                    x.to_fxx(params)
                }
            })
            .sum::<fxx>()
            / evals.len() as fxx
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
    fn to_fxx(&self, params: &SimulationParameters) -> fxx {
        0. + self.reward * params.eval_reward.value
            + self.early_finish_percent * params.eval_early_finish.value
            + self.distance_percent * params.eval_distance.value
            - self.penalty * params.eval_penalty.value
    }

    // from 0 to 1
    fn to_fxx_second_way(&self) -> fxx {
        (1. / (1. + self.penalty)
            // + (1. - 1. / (1. + self.reward))
            + self.early_finish_percent
            + self.distance_percent
            + self.autoencoder_loss)
            / 5.
    }
}

pub fn patch_params_sim(params: &[fxx], params_sim: &SimulationParameters) -> SimulationParameters {
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
    params: &[fxx],
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
                              current_segment_fxx,
                              simulation_vars| {
                            nn_processor.process(
                                time,
                                dist,
                                dpenalty,
                                dirs,
                                dirs_second_layer,
                                current_segment_fxx,
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
                        early_finish_percent = (steps_quota - i) as fxx / steps_quota as fxx;
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

fn from_f64_to_fxx_vec(pos: &[f64]) -> Vec<fxx> {
    pos.iter().copied().map(|x| x as fxx).collect()
}

fn from_dvector_to_fxx_vec(pos: &DVector<f64>) -> Vec<fxx> {
    pos.iter().copied().map(|x| x as fxx).collect()
}

fn to_vec_fxx(pos: &[f32]) -> Vec<fxx> {
    pos.iter().copied().map(|x| x as fxx).collect::<Vec<_>>()
}

pub fn evolve_by_differential_evolution(params_sim: &SimulationParameters) {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let params_phys = params_sim.patch_physics_parameters(PhysicsParameters::default());
    let nn_len = params_sim.nn.get_nns_len();

    let input_done: Vec<(f32, f32)> = if RUN_FROM_PREV_NN {
        // include!("nn.data")
        //     .1
        //     .into_iter()
        //     .map(|x| (x - 0.01, x + 0.01))
        //     .collect()
        todo!()
    } else {
        vec![(-10., 10.); nn_len]
    };

    assert_eq!(input_done.len(), nn_len);

    let now = Instant::now();
    let mut de = self_adaptive_de(input_done, |pos| {
        let evals = eval_nn(&to_vec_fxx(pos), &params_phys, params_sim);
        -sum_evals(&evals, params_sim, false) as f32
    });
    for pos in 0..100_000 {
        let value = de.iter().next().unwrap();
        if pos % 20 == 0 && pos != 0 {
            println!("{pos}. {value}, {:?}", now.elapsed() / pos as u32);
        }
        if pos % 300 == 0 && pos != 0 {
            let (cost, vec) = de.best().unwrap();
            println!("cost: {}", cost);
            print_evals(&eval_nn(&to_vec_fxx(vec), &params_phys, params_sim));
        }
        if pos % 1000 == 0 && pos != 0 {
            let (_, vec) = de.best().unwrap();
            println!("(vec!{:?}, vec!{:?})", nn_sizes, vec);
        }
    }
    // show the result
    let (cost, pos) = de.best().unwrap();
    println!("cost: {}", cost);
    print_evals(&eval_nn(&to_vec_fxx(pos), &params_phys, params_sim));
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
        let evals = eval_nn(&to_vec_fxx(pos), &params_phys, params_sim);
        -sum_evals(&evals, params_sim, false) as f32
    });

    let mut true_params_sim = SimulationParameters::true_metric(&params_sim);

    for pos in 0..generations_count {
        let now = Instant::now();
        for _ in 0..population_size {
            let _ = de.iter().next().unwrap();
        }
        let (value, point) = de.best().unwrap();

        true_params_sim = patch_params_sim(&to_vec_fxx(point), &true_params_sim);

        let evals = eval_nn(&to_vec_fxx(point), params_phys, &params_sim);
        let true_evals = eval_nn(&to_vec_fxx(point), params_phys, &true_params_sim);

        result.push(EvolveOutputEachStep {
            nn: to_vec_fxx(point),
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
        // include!("nn.data").1
        todo!()
    } else {
        vec![1.0; nn_len]
    };

    assert_eq!(input_done.len(), nn_len);

    // while params_sim.simulation_simple_physics >= 0. {
    let mut state = cmaes::options::CMAESOptions::new(input_done, 10.0)
        .population_size(100)
        .sample_mean(params_sim.evolution_sample_mean)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(&from_dvector_to_fxx_vec(&x), &params_phys, &params_sim);
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
                &from_dvector_to_fxx_vec(&point),
                &params_phys,
                &params_sim,
            ));
        }
        println!("{pos}. {value}, {:?}", now.elapsed() / (pos + 1) as u32);
        if pos % 10 == 0 && pos != 0 {
            print_evals(&eval_nn(
                &from_dvector_to_fxx_vec(&point),
                &params_phys,
                &params_sim,
            ));
        }
        if pos % 10 == 0 && pos != 0 {
            println!(
                "(vec!{:?}, vec!{:?})",
                nn_sizes,
                &from_dvector_to_fxx_vec(&point)
            );
        }
    }
    let solution = state.overall_best_individual().unwrap().point.clone();

    let params_phys_clone = params_phys.clone();
    params_phys = params_sim.patch_physics_parameters(params_phys_clone);
    input_done = solution.iter().copied().collect();
    println!("(vec!{:?}, vec!{:?})", nn_sizes, solution.as_slice());
    let evals = eval_nn(
        &from_dvector_to_fxx_vec(&&solution),
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
    nn: Vec<fxx>,
    evals: Vec<TrackEvaluation>,
    evals_cost: fxx,
    true_evals: Vec<TrackEvaluation>,
    true_evals_cost: fxx,
}

pub fn evolve_by_cma_es_custom(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    nn_input: &[fxx],
    population_size: usize,
    generations_count: usize,
    step_size: Option<f64>,
    stop_at: Option<fxx>,
) -> Vec<EvolveOutputEachStep> {
    let nn_sizes = params_sim.nn.get_nn_sizes();
    let nn_len = params_sim.nn.get_nns_len();
    let input_done: Vec<f64> = nn_input.iter().map(|x| *x as f64).collect();
    assert_eq!(input_done.len(), nn_len + OTHER_PARAMS_SIZE);

    let mut result: Vec<EvolveOutputEachStep> = Default::default();

    let mut state = cmaes::options::CMAESOptions::new(input_done, step_size.unwrap_or(1.))
        .population_size(population_size)
        .cm(params_sim.evolution_learning_rate)
        .sample_mean(params_sim.evolution_sample_mean)
        .build(|x: &DVector<f64>| -> f64 {
            let evals = eval_nn(&from_dvector_to_fxx_vec(&x), params_phys, params_sim);
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

        let nn = from_dvector_to_fxx_vec(&point);
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
    x: &mut Vec<fxx>,
    f: &mut dyn FnMut(&Vec<fxx>) -> T,
    idx: usize,
    y: fxx,
) -> T {
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(x);
    x[idx] = xtmp;
    fx1
}

fn forward_diff_vec(x: &Vec<fxx>, f: &mut dyn FnMut(&Vec<fxx>) -> fxx) -> Vec<fxx> {
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
                &from_f64_to_fxx_vec(&p.to_vec()),
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
                    &from_f64_to_fxx_vec(&x.to_vec()),
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
        // include!("nn.data").1.into_iter().collect()
        todo!()
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
        &from_f64_to_fxx_vec(&input_vec.to_vec()),
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
    best_start: Option<Vec<fxx>>,
    stop_at: Option<fxx>,
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
                &from_f64_to_fxx_vec(&p.to_vec()),
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

        let nn = from_f64_to_fxx_vec(point.as_slice().unwrap());
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
    let input_done: Vec<fxx> = todo!(); //include!("nn.data").1;
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
            hidden_layers: vec![LayerDescription::relu_best(6)],
            inv_distance: true,
            inv_distance_coef: 20.,
            inv_distance_pow: 0.5,
            view_angle_ratio: 2. / 6.,
            pass_current_physics: false,

            dirs_size: 21,
            pass_dirs: true,
            pass_dirs_diff: false,
            pass_dirs_second_layer: false,

            use_dirs_autoencoder: false,
            autoencoder_hidden_layers: vec![LayerDescription::relu_best(6)],
            autoencoder_exits: 5,

            pass_current_track: false,
            max_tracks: 12,

            pass_current_segment: false,
            max_segments: 20,

            pass_time_mods: vec![],

            use_ranking_network: false,
            ranking_hidden_layers: vec![LayerDescription::relu_best(6)],
            rank_without_physics: false,
            rank_close_to_zero: false,

            output_discrete_action: false,
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
                    get_all_tracks()
                        .into_iter()
                        .map(|x| x.0)
                        .map(|x| (x.to_owned(), true)),
                )
                .collect(),
            tracks_enable_mirror: true,

            start_places_enable: false,
            start_places_for_tracks: all_tracks
                .into_iter()
                .chain(
                    get_all_tracks()
                        .into_iter()
                        .map(|x| x.0)
                        .into_iter()
                        .flat_map(|x| vec![x.to_owned(), x.to_owned() + "_mirror"])
                        .map(|x| (x.to_owned(), false)),
                )
                .collect(),

            mutate_car_enable: false,
            mutate_car_count: 3,
            mutate_car_angle_range: Clamped::new(0.7, 0., TAUx / 8.),

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
            simulation_random_output_second_way: true,
            random_output_probability: 0.0,

            evolution_sample_mean: false,
            evolution_generation_count: 100,
            evolution_population_size: 30,
            evolution_learning_rate: 1.0,
            evolution_distance_to_solution: 10.,
            evolution_simple_add_mid_start: false,
            evolution_start_input_range: 10.,

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
        result.rewards_second_way = true;
        result.mutate_car_enable = false;
        result.simulation_stop_penalty.value = 100.;

        result.simulation_simple_physics = other.simulation_simple_physics;
        result.evolve_simple_physics = other.evolve_simple_physics;

        result.nn = other.nn.clone();

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
    input: &[fxx],
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

fn random_input_by_len(len: usize, limit: fxx) -> Vec<fxx> {
    let mut rng = thread_rng();
    (0..len)
        .map(|_| rng.gen_range(-limit..limit))
        .collect::<Vec<fxx>>()
}

fn random_input(params_sim: &SimulationParameters) -> Vec<fxx> {
    random_input_by_len(
        params_sim.nn.get_nns_len() + OTHER_PARAMS_SIZE,
        params_sim.evolution_start_input_range,
    )
}

fn test_params_sim_fn<
    F: Fn(
            &SimulationParameters,
            &PhysicsParameters,
            &[fxx],
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
        let start = Instant::now();
        let result = f(
            params_sim,
            params_phys,
            &random_input(params_sim),
            POPULATION_SIZE,
            GENERATIONS_COUNT,
        );
        println!(
            "FINISH! Time: {:?}, score: {}",
            start.elapsed(),
            result.last().unwrap().evals_cost
        );
        result
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
    test_params_sim_fn(
        params_sim,
        params_phys,
        name,
        |params_sim, params_phys, input, population_size, generations_count| {
            evolve_by_cma_es_custom(
                params_sim,
                params_phys,
                input,
                params_sim.evolution_population_size,
                params_sim.evolution_generation_count,
                Some(params_sim.evolution_distance_to_solution),
                None,
            )
        },
    )
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
struct Dirs(Vec<Option<fxx>>);
#[derive(Default, Serialize, Deserialize, Clone)]
struct TrackDirs(Vec<Dirs>);
#[derive(Default, Serialize, Deserialize, Clone)]
struct AllTrackDirs(Vec<TrackDirs>);

fn eval_tracks_dirs(
    params_sim: &SimulationParameters,
    params_phys: &PhysicsParameters,
    input: &[fxx],
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
                      current_segment_fxx,
                      simulation_vars| {
                    result.0.last_mut().unwrap().0.push(Dirs(dirs.to_vec()));
                    nn_processor.process(
                        time,
                        dist,
                        dpenalty,
                        dirs,
                        dirs_second_layer,
                        current_segment_fxx,
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

fn eval_ae(params: &[fxx], all_tracks_dirs: &AllTrackDirs, nn_params: &NnParameters) -> fxx {
    let mut loss = 0.;
    let mut ae_processor = NnProcessorAutoencoder::new(&params, nn_params.clone());
    for TrackDirs(track_dirs) in &all_tracks_dirs.0 {
        ae_processor.reset();
        for Dirs(dirs) in track_dirs {
            ae_processor.process(&dirs);
        }
        loss += ae_processor.get_autoencoder_loss();
    }
    loss / all_tracks_dirs.0.len() as fxx
}

fn evolve_by_cma_es_custom_ae(
    input: &[fxx],
    all_tracks_dirs: &AllTrackDirs,
    nn_params: &NnParameters,
    population_size: usize,
    generations_count: usize,
) -> Vec<fxx> {
    let input_done: Vec<f64> = input.iter().map(|x| *x as f64).collect();

    let mut state = cmaes::options::CMAESOptions::new(input_done, 10.)
        .population_size(population_size)
        .build(|x: &DVector<f64>| -> f64 {
            -eval_ae(&from_dvector_to_fxx_vec(&x), all_tracks_dirs, nn_params) as f64
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

    from_dvector_to_fxx_vec(&state.overall_best_individual().unwrap().point)
}

#[allow(unused_imports, unused_variables)]
fn evolve_by_bfgs_autoencoder(
    input: &[fxx],
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
                &from_f64_to_fxx_vec(&p.to_vec()),
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
                    &from_f64_to_fxx_vec(&p.to_vec()),
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
        &from_f64_to_fxx_vec(&input_vec.to_vec()),
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
                let values: Vec<fxx> = all_nns
                    .iter()
                    .map(|nn| nn.layers[layer_idx].matrix[i][j])
                    .collect();

                // Calculate mean
                let mean: fxx = values.iter().sum::<fxx>() / values.len() as fxx;

                // Calculate std dev
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<fxx>() / values.len() as fxx;
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
            let values: Vec<fxx> = all_nns
                .iter()
                .map(|nn| nn.layers[layer_idx].bias[i])
                .collect();

            // Calculate mean
            let mean: fxx = values.iter().sum::<fxx>() / values.len() as fxx;

            // Calculate std dev
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<fxx>() / values.len() as fxx;
            let std_dev = variance.sqrt();

            println!("Position [{}]: mean = {:.3}, std = {:.3}", i, mean, std_dev);
        }
    }
}

fn interpolate_two_nns(params_sim: &SimulationParameters, params_phys: &PhysicsParameters) {
    let file =
        std::fs::read_to_string("graphs/graphs_to_copy/nn2_dirs_diff_add_simple_mid.json").unwrap();
    let all_runs: Vec<Vec<EvolveOutputEachStep>> = serde_json::from_str(&file).unwrap();
    let all_nns = all_runs
        .iter()
        .map(|x| x.last().unwrap().nn.clone())
        .collect::<Vec<Vec<fxx>>>();

    let nn1 = all_nns[0].clone();
    let evals1 = eval_nn(&nn1, params_phys, params_sim);
    print_evals(&evals1);
    let cost1 = sum_evals(&evals1, params_sim, true);
    println!("cost1 = {:.3}", cost1);

    let mut data: Vec<Vec<(fxx, fxx)>> = vec![];
    for i in 0..10 {
        let nn1 = all_nns[i].clone();
        for j in (i + 1)..10 {
            let nn2 = all_nns[j].clone();
            let mut data_row: Vec<(fxx, fxx)> = vec![];
            for t in (0..=30).map(|x| x as fxx / 30.) {
                let nn_t = nn1
                    .iter()
                    .zip(nn2.iter())
                    .map(|(x, y)| x * t + y * (1. - t))
                    .collect::<Vec<fxx>>();

                let evals = eval_nn(&nn_t, params_phys, params_sim);
                let cost = sum_evals(&evals, params_sim, true);

                println!("t = {:.3}, cost = {:.3}", t, cost);
                data_row.push((t, cost));
            }
            data.push(data_row);
        }
    }
    save_json_to_file(&data, "interpolation_data.json");
}

fn evaluate_noise(params_sim: &SimulationParameters, params_phys: &PhysicsParameters) {
    let mut params_sim = params_sim.clone();

    #[rustfmt::skip]
    let params = vec![];
    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true; // better than using physics 🥲
                                               // params_sim.nn.output_discrete_action = true;

    params_sim.nn.ranking_hidden_layers = vec![
        LayerDescription::relu_best(10),
        LayerDescription::relu_best(10),
    ];
    params_sim.simulation_simple_physics = 0.0;
    params_sim.simulation_stop_penalty.value = 50.;
    params_sim.tracks_enable_mirror = false;
    params_sim.simulation_random_output_second_way = true;

    let runs_amount = 300;
    let t_amount = 50;
    let data = (0..runs_amount)
        .into_par_iter()
        .map(|i| {
            let mut params_sim = params_sim.clone();
            let now = Instant::now();
            let mut data_row: Vec<(fxx, fxx)> = vec![];
            for t in (0..=t_amount).map(|x| x as fxx / t_amount as fxx / 2.) {
                params_sim.random_output_probability = t;
                let evals = eval_nn(&params, &params_phys, &params_sim);
                let cost = sum_evals(&evals, &params_sim, true);

                data_row.push((t, cost));
            }
            println!("{i}, time: {:?}", now.elapsed());
            data_row
        })
        .collect::<Vec<_>>();
    // save_json_to_file(&data, "graphs/graphs_to_copy/interpolation_data_actor_predictor.json");
    save_json_to_file(
        &data,
        "graphs/graphs_to_copy/interpolation_data_ranker.json",
    );
}

fn eval_nn_from_python(params_sim: &SimulationParameters, params_phys: &PhysicsParameters) {
    let mut params_sim = params_sim.clone();

    let mut params = read_nn_from_file("records/neural_network_actor.json");
    params.push(0.);
    params_sim.nn.pass_dirs_diff = true;
    params_sim.nn.output_discrete_action = true;
    params_sim.nn.hidden_layers = vec![LayerDescription::relu_best(16)];
    params_sim.simulation_simple_physics = 0.0;
    params_sim.simulation_stop_penalty.value = 50.;
    let evals = eval_nn(&params, &params_phys, &params_sim);
    print_evals(&evals);
}

fn read_nn_from_file(file_path: &str) -> Vec<fxx> {
    let file = std::fs::read_to_string(file_path).unwrap();
    let nn: NeuralNetworkUnoptimized = serde_json::from_str(&file).unwrap();
    nn.to_optimized().get_values().iter().copied().collect()
}

pub fn evolution() {
    let mut params_sim = SimulationParameters::default();
    let mut params_phys = PhysicsParameters::default();

    params_sim.enable_all_tracks();
    params_sim.disable_track("straight_45");
    params_sim.disable_track("loop");
    params_sim.disable_track("straight_turn");
    params_sim.disable_track("bubble_straight");
    params_sim.disable_track("bubble_180");
    params_sim.disable_track("separation");
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
    params_sim.nn.hidden_layers = vec![LayerDescription::relu_best(10)];
    params_sim.nn.inv_distance_coef = 30.;

    // new settings
    params_sim.simulation_simple_physics = 0.0;
    params_sim.nn.ranking_hidden_layers = vec![LayerDescription::relu_best(10)];
    params_sim.simulation_stop_penalty.value = 50.;
    params_sim.tracks_enable_mirror = false;
    params_sim.nn.pass_dirs_diff = true;
    params_sim.evolution_population_size = 30;
    params_sim.evolution_generation_count = 300;
    params_sim.evolution_distance_to_solution = 1.;
    params_sim.evolution_start_input_range = 1.;

    let mut params_sim_copy = params_sim.clone();

    #[rustfmt::skip]
    let params = vec![-1.0140746,7.969905,3.255649,-0.45773593,-8.814844,5.5313916,2.2310336,-6.857345,0.03863446,-6.8043175,3.8996851,4.1762123,6.342612,8.253213,10.223366,-3.637407,-4.825101,-6.223656,-5.710869,-2.1575553,0.3453375,0.38729206,-0.84449524,4.307528,8.513787,9.111382,5.708905,-5.0000615,5.2223225,-4.882063,-6.8563724,-3.2832754,10.826247,5.1875124,1.0637699,5.1628504,7.831787,1.6877187,2.3491712,10.182303,-2.650052,-5.8988256,2.2731888,-5.933333,-2.4040658,-0.6451996,2.6045537,6.8096733,1.5365217,0.112321965,2.7377155,-6.143689,-0.6136311,2.9335635,4.068984,-2.3936646,4.105215,8.365123,1.4176098,8.74154,7.0931883,-0.4234664,9.790796,-5.3568335,-6.184567,-1.878323,-0.44523206,-4.3636174,-10.265222,11.838661,-3.2664053,-10.227155,-1.1348896,-12.036602,-3.6037822,10.177021,2.8815486,-10.635429,5.409533,1.2621099,6.355703,3.449306,3.7818124,3.4725761,-0.221656,-18.189472,4.8883586,6.5181103,7.976028,2.3536148,-11.689482,-5.211554,-7.1512904,-6.6349115,-3.059728,-8.888863,-3.1679716,5.815808,-6.8705287,-1.28352,-3.3116326,-3.6879137,-3.8458946,4.145627,10.688421,6.8072286,-8.7282295,6.2244353,1.6423159,0.4508205,2.1093874,5.083254,7.8963084,-10.452589,1.5590851,-9.058856,0.2950253,4.867293,-0.22889403,2.3118467,4.435561,1.0126159,1.2630934,0.6034488,-7.261848,-6.456671,-8.173853,-2.5947015,1.6566625,0.8303851,-3.9235418,-3.6188898,-5.6813827,8.366194,2.9604132,-3.100621,-2.343282,7.5686,0.7278174,-12.403346,-7.4190397,-12.375827,-1.6933929,1.2848308,-2.6327226,-6.1778817,-0.69299924,3.7549577,-0.28050584,0.7778191,-3.4546475,-0.70060945,-4.8373885,4.4729676,-3.1355038,7.8780136,3.5531738,8.082677,-2.719097,-1.0375485,5.097316,-1.9870536,-9.435947,5.1853786,0.77007765,0.77070636,-3.4930844,5.965919,3.0085702,-0.71396536,-1.0684042,1.980887,3.3859272,-3.9332833,-0.97807574,0.67047924,-0.77030766,1.1952081,8.525371,-0.27639112,-8.219419,-4.3506627,0.93446237,3.955072,-3.682673,-1.5677782,-6.7918572,4.556543,8.988926,1.6153657,13.157081,-7.3095202,-6.7097244,-1.7330406,-5.400186,1.8234549,10.5188875,-4.4084005,-2.8515608,4.394829,-10.170477,-3.6425822,5.3774214,8.574465,5.125784,-2.592275,0.416314,-1.4990387,-1.3619555,9.500459,-3.345467,1.6097945,-2.4869363,6.2751403,-2.7470694,-1.241509,9.263079,-2.357499,-0.71642643,9.081934,7.098453,13.064735,6.072085,0.86135876,0.42790878,-13.554738,-4.993654,7.828532,-0.26332355,1.3722762,3.2465909,-1.5418559,-3.5693011,-5.8441744,6.4627514,2.8090293,0.31156737,-3.857934,-11.280977,-4.0953608,-7.7492537,0.58041936,8.433039,-4.3115425,0.3458287,2.1145627,4.65377,-4.6333575,2.3745975,-1.8756385,4.223438,6.89845,0.79969174,-1.7829785,1.6288344,-1.8161334,-3.4675262,4.832272,-1.6210185,-1.696128,1.0028566,7.706657,-6.269442,-3.4574616,-4.935856,-0.52170604,9.285264,-1.5701663,0.94492525,-2.3315585,-6.3180594,1.2134569,-1.0206414,-1.8991766,-3.3108075,-1.8519135,-5.114469,-8.711949,5.9418797,5.3772616,-3.3919933,5.016679,-9.09488,15.286478,-3.5477734,-10.717793,-8.667156,3.430417,-9.043924,-6.3868427,3.771201,-9.276051,1.6036136,4.0459547,8.174598,13.675274,3.1494143,13.16244,2.141045,3.4357753,0.46819896,3.7956643,-6.269334,2.6654906,0.88894075,0.89534014,-1.8865765,-5.9851017,2.8273027,6.1478505,-7.5374966,-1.1878409,-2.6607187,1.4505111,-10.753712,-6.662928,-0.49729615,-2.2162392,2.1547415,-1.5608006,-6.2728963,0.29494822,2.0775702,0.94253886,-3.3023574,0.20184654,3.6718824,10.206356,-3.8106627,-0.86780673,-1.8850509,-4.458633,-0.2645319,-5.911129,5.141833,-6.1215625,3.2625692,5.239356,6.819452,6.961727,2.2123358,-0.37622842,3.0404775,-0.5514076,3.5382001,-0.4843125,9.572454,3.317881,-7.289935,8.832711,-7.854359,-3.8881705,1.4729297,-2.0490656,-6.3347015,0.12885739,11.582888,-8.913492,-6.617836,1.5036279,1.3143181,-4.920342,3.9786863,-4.9569116,8.047843,-0.64386404,-7.1422963,6.4997573,-5.8164296,-2.0426564,3.9373107,5.9638896,1.1785966,7.270487,-3.711227,-2.8631806,-3.3163843,-2.6125124,2.950503,-9.781572,1.1625123,7.967913,0.6939382,10.151258,-8.032308,-0.117915586,-1.595022,-7.2094173,5.5032854,1.3340335,1.4497712,0.8595004,3.7436821,-0.55241823,-2.5470426,-6.1092176,-10.614954,-5.7577777,0.35974905,-1.9811981,0.0054431106,-8.374586,6.760215,4.6665254,-2.4265747,3.263907,3.302049,9.178065,2.2093327,-11.06965,0.75237775,-4.508033,3.2366216,8.981704,5.302875,-8.408273,3.1710312,0.5038109,-0.73679346,-1.4566271,-12.260539,0.8673921,-1.9078183,-11.013678,2.750532,3.2524343,9.519965,12.611303,-2.7479165,6.3550787,1.4470017,-0.12573671,-6.489775,7.47961,0.7660119,2.517699,-4.97592,-8.806205,5.183689,0.123767346,-5.8479776,-0.12721643,7.4339147,5.3236895,2.7361069,-3.559886,3.7529047,-5.3707476,3.5633106,3.2847552,7.4842453,4.7480836,5.0248837,-3.7692423,-1.7893031,-3.1387558,1.1278142,-8.203136,2.105451,1.8934679,1.1727809,-3.0505586,-0.3243254,4.8932843,5.556009,-4.2797256,5.2714214,4.9797897,2.9320369,3.3040593,0.21399115,2.205502,2.7676451,4.722789,-2.0099642,5.958663,5.9893565,-14.761563,-2.696921,-1.9571339,-1.3625263,-4.778177,5.32249,-1.0973876,-0.2765338,-4.162304,-7.1115403,2.008183,5.265838,7.005275,0.998941,10.689762,-0.7107461,-2.3121784,1.6911142,0.7218279,-9.3835,-3.8390162,1.7028191,2.462449,0.035071786,-8.322564,2.435498,2.2931461,0.9931594,-0.8946622,3.4547412,-5.8351493,10.893359,-2.5663805,0.97312886,3.7423441,-0.13199523,5.793001,4.774439,-2.5494466,-11.072732,0.14585106,5.24943,-3.491153,-4.736221,-1.0961194,2.4460342,-13.608879,-0.32326534,2.3322601,5.4159718,-6.0713606,-4.9699006,4.9630904,4.758529,-3.9612696,-3.5901847,-7.9037423,4.1176996,5.1404896,3.7284865,3.2361503,0.15753356,-10.480198,-0.27333233,-12.892551,7.2325597,6.4994297,1.8522301,-7.6900454,-5.385723,13.301702,11.036094,2.9045067,-1.4765296,3.3513288,6.0128174,9.808207,0.62630045,-5.2958302,0.9629272,-6.148471,-9.973472,2.5368805,1.2529771,1.9739354,5.3408813,-0.66777575,-5.8822393,-5.032582,-2.2442875,11.496572,5.9201093,-1.322113,6.150977,-5.9573584,-3.654369,-8.140595,1.5725151,6.523251,-1.0965878,11.567554,1.665575,6.663997,1.0801463,6.6025505];

    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true;
    // params_sim.tracks_enable_mirror = true;

    params_sim.nn.pass_current_track = true;
    params_sim.nn.pass_current_physics = true;
    params_sim.nn.max_tracks = 6;

    // params_sim.enable_track("straight_45");
    // params_sim.enable_track("loop");
    // params_sim.enable_track("straight_turn");
    // // params_sim.enable_track("bubble_straight");
    // // params_sim.enable_track("bubble_180");
    // params_sim.enable_track("separation");

    // params_sim.simulation_random_output_second_way = true;
    // params_sim.random_output_probability = 0.05;
    // params_sim.evolution_learning_rate = 0.9;

    params_sim.eval_add_other_physics = vec![
        PhysicsPatch {
            traction: Some(0.15),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            traction: Some(0.25),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            traction: Some(0.5),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            traction: Some(1.0),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            friction_coef: Some(1.0),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            friction_coef: Some(0.0),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            acceleration_ratio: Some(1.0),
            ..PhysicsPatch::default()
        },
        PhysicsPatch {
            acceleration_ratio: Some(0.6),
            ..PhysicsPatch::default()
        },
    ];

    params_sim.nn.ranking_hidden_layers =
        vec![LayerDescription::new(10, ActivationFunction::SqrtSigmoid)];
    let result = evolve_by_cma_es_custom(
        &params_sim,
        &params_phys,
        // &random_input(&params_sim),
        &params,
        30,
        50,
        Some(1.0),
        None,
    );
    println!("{:?}", result.last().unwrap().nn);

    return;

    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true;
    test_params_sim(&params_sim, &params_phys, "hard2_ranker_all_maps");
    params_sim = params_sim_copy.clone();

    return;

    test_params_sim(&params_sim, &params_phys, "hard2_default");
    params_sim = params_sim_copy.clone();

    params_sim.tracks_enable_mirror = true;
    test_params_sim(&params_sim, &params_phys, "hard2_with_mirror");
    params_sim = params_sim_copy.clone();

    params_sim.nn.output_discrete_action = true;
    test_params_sim(&params_sim, &params_phys, "hard2_discrete");
    params_sim = params_sim_copy.clone();

    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true;
    test_params_sim(&params_sim, &params_phys, "hard2_ranker_no_physics");
    params_sim = params_sim_copy.clone();

    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true;
    params_sim.nn.rank_close_to_zero = true;
    test_params_sim(&params_sim, &params_phys, "hard2_ranker_no_physics_to_zero");
    params_sim = params_sim_copy.clone();

    params_sim.tracks_enable_mirror = true;
    params_sim.nn.use_ranking_network = true;
    params_sim.nn.rank_without_physics = true;
    params_sim.nn.rank_close_to_zero = true;
    test_params_sim(
        &params_sim,
        &params_phys,
        "hard2_ranker_no_physics_to_zero_mirror",
    );
    params_sim = params_sim_copy.clone();

    params_sim.nn.use_ranking_network = true;
    test_params_sim(&params_sim, &params_phys, "hard2_ranker_physics");
    params_sim = params_sim_copy.clone();

    return;

    params_sim.evolution_distance_to_solution = 10.;
    test_params_sim(&params_sim, &params_phys, "hard2_default_cmaes_1_10");
    params_sim = params_sim_copy.clone();

    params_sim.evolution_distance_to_solution = 10.;
    params_sim.evolution_start_input_range = 10.;
    test_params_sim(&params_sim, &params_phys, "hard2_default_cmaes_10_10");
    params_sim = params_sim_copy.clone();

    params_sim.evolution_start_input_range = 10.;
    test_params_sim(&params_sim, &params_phys, "hard2_default_cmaes_10_1");
    params_sim = params_sim_copy.clone();

    return;

    test_params_sim(&params_sim, &params_phys, "hard2_default");
    params_sim = params_sim_copy.clone();

    params_sim.tracks_enable_mirror = true;
    test_params_sim(&params_sim, &params_phys, "hard2_with_mirror");
    params_sim = params_sim_copy.clone();

    params_sim.simulation_stop_penalty.value = 20.;
    test_params_sim(&params_sim, &params_phys, "hard2_stop_penalty_20");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_dirs_diff = false;
    test_params_sim(&params_sim, &params_phys, "hard2_no_dirs_diff");
    params_sim = params_sim_copy.clone();

    params_sim.nn.pass_dirs_diff = false;
    params_sim.simulation_stop_penalty.value = 20.;
    params_sim.tracks_enable_mirror = true;
    test_params_sim(&params_sim, &params_phys, "hard2_all_old_settings");
    params_sim = params_sim_copy.clone();

    return;

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::Relu)];
    test_params_sim(&params_sim, &params_phys, "activations_relu");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::Relu10)];
    test_params_sim(&params_sim, &params_phys, "activations_relu_10");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::ReluLeaky)];
    test_params_sim(&params_sim, &params_phys, "activations_relu_leaky");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::ReluLeaky10)];
    test_params_sim(&params_sim, &params_phys, "activations_relu_leaky_10");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(
        10,
        ActivationFunction::ReluLeakySmooth10,
    )];
    test_params_sim(
        &params_sim,
        &params_phys,
        "activations_relu_leaky_smooth_10",
    );
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::Sigmoid)];
    test_params_sim(&params_sim, &params_phys, "activations_sigmoid");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::SqrtSigmoid)];
    test_params_sim(&params_sim, &params_phys, "activations_sqrt_sigmoid");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::Softmax)];
    test_params_sim(&params_sim, &params_phys, "activations_softmax");
    params_sim = params_sim_copy.clone();

    params_sim.nn.hidden_layers = vec![LayerDescription::new(10, ActivationFunction::ArgmaxOneHot)];
    test_params_sim(&params_sim, &params_phys, "activations_argmax_one_hot");
    params_sim = params_sim_copy.clone();
}

pub const RUN_EVOLUTION: bool = false;
pub const RUN_FROM_PREV_NN: bool = false;
const ONE_THREADED: bool = false;
const PRINT: bool = true;
const PRINT_EVERY_10: bool = false;
const PRINT_EVERY_10_ONLY_EVALS: bool = true;
const PRINT_EVALS: bool = true;

const RUNS_COUNT: usize = 1;
const POPULATION_SIZE: usize = 30;
const GENERATIONS_COUNT: usize = 300;
pub const OTHER_PARAMS_SIZE: usize = 1;
