use crate::common::*;
use crate::evolution::*;
use crate::math::project_to_segment;
use crate::math::*;
use crate::nn::*;
use crate::physics::*;
use crate::storage::*;
use egui::containers::Frame;
use egui::pos2;
use egui::vec2;
use egui::Align2;
use egui::Color32;
use egui::DragValue;
use egui::FontFamily;
use egui::FontId;
use egui::PointerButton;
use egui::Shape;
use egui::Stroke;
use egui::Ui;
use egui::*;
use egui::{emath, Rect, Sense};
use egui_plot::Corner;
use egui_plot::Legend;
use egui_plot::Line;
use egui_plot::Plot;
use egui_plot::PlotPoints;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::collections::VecDeque;

#[derive(serde::Deserialize, serde::Serialize, Default, Clone)]
#[serde(default)]
struct Drifts {
    drift_recording: bool,
    drifts: VecDeque<VecDeque<Vecx2>>,
}

impl Drifts {
    fn process_point(&mut self, point: Vecx2, is_drift: bool) {
        if is_drift {
            if !self.drift_recording {
                self.drift_recording = true;
                self.drifts.push_back(Default::default());
            }
            self.drifts.back_mut().unwrap().push_back(point);
        } else {
            self.drift_recording = false;
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Default)]
#[serde(default)]
#[allow(clippy::type_complexity)]
struct Graphs {
    points: BTreeMap<String, BTreeMap<String, (VecDeque<[f64; 2]>, egui::Color32)>>,
}

impl Graphs {
    fn add_point(
        &mut self,
        group: impl Into<String>,
        name: impl Into<String>,
        value: fxx,
        color: egui::Color32,
        params: &InterfaceParameters,
    ) {
        let entry = &mut self.points.entry(group.into()).or_default();
        let entry = &mut entry.entry(name.into()).or_default();
        entry.0.push_back([
            entry.0.back().map(|x| x[0] + 1.).unwrap_or(0.),
            value.into(),
        ]);
        if entry.0.len() > params.graph_points_size_limit {
            entry.0.pop_front();
        }
        entry.1 = color;
    }

    fn clear(&mut self) {
        self.points.clear();
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    // in car local coordinates
    trajectories: VecDeque<Vecx2>,
    points_count: usize,

    graphs: Graphs,
    drifts: Vec<Drifts>,

    params_intr: InterfaceParameters,
    params_phys: PhysicsParameters,
    params_sim: SimulationParameters,

    // walls: Storage2<Wall>,
    // rewards: Storage2<Reward>,
    points: Vec<PointsStorage>,
    current_edit: usize,
    offset: Vecx2,
    drag_pos: Option<usize>,
    enable_edit_track: bool,

    #[serde(skip)]
    rng: StdRng,

    #[serde(skip)]
    simulation: CarSimulation,

    #[serde(skip)]
    nn_processor: NnProcessor,

    override_nn: bool,

    quota: usize,

    #[serde(skip)]
    all_tracks: Vec<(String, Vec<PointsStorage>)>,

    current_track: String,
    current_track_id: usize,

    current_nn: String,

    finish_step: usize,

    current_start_reward: usize,

    record_user_actions: bool,
    records: Vec<(Vec<fxx>, Vec<fxx>)>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let mut params_sim = SimulationParameters::default();

        params_sim.enable_all_tracks();
        params_sim.disable_track("straight_45");
        params_sim.eval_add_min_distance = false;
        params_sim.evolve_simple_physics = false;
        params_sim.mutate_car_enable = false;
        params_sim.nn.hidden_layers = vec![LayerDescription::relu_best(10)];
        params_sim.nn.inv_distance_coef = 30.;
        // params_sim.nn.pass_current_track = true;
        // params_sim.nn.pass_current_segment = true;
        params_sim.nn.pass_dirs_diff = true;
        params_sim.nn.pass_internals = true;
        // params_sim.nn.pass_next_size = 3;
        params_sim.nn.pass_simple_physics_value = false;
        params_sim.nn.view_angle_ratio = 3. / 6.;
        params_sim.rewards_add_each_acquire = true;
        params_sim.rewards_enable_distance_integral = true;
        params_sim.rewards_second_way = true;
        params_sim.rewards_second_way_penalty = true;
        params_sim.simulation_enable_random_nn_output = false;
        params_sim.simulation_simple_physics = 0.;

        // params_sim.nn.pass_dirs_diff = true;
        // params_sim.nn.pass_internals = true;
        // params_sim.nn.hidden_layers = vec![20, 10];

        // // params_sim.simulation_stop_penalty.value = 1.;
        // params_sim.simulation_simple_physics = 0.0;
        // // params_sim.tracks_enable_mirror = false;
        // params_sim.nn.use_ranking_network = true;
        // // params_sim.simulation_stop_penalty.value = 0.1;
        // params_sim.nn.ranking_hidden_layers = vec![10, 5];
        params_sim.eval_add_other_physics = vec![
            PhysicsPatch {
                traction: Some(0.15),
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

        params_sim.simulation_simple_physics = 0.0;
        params_sim.simulation_stop_penalty.value = 50.;
        params_sim.tracks_enable_mirror = false;
        params_sim.nn.pass_dirs_diff = true;
        params_sim.evolution_population_size = 30;
        params_sim.evolution_generation_count = 300;
        params_sim.evolution_distance_to_solution = 1.;
        params_sim.evolution_start_input_range = 1.;

        params_sim.nn.use_ranking_network = true;
        params_sim.nn.rank_without_physics = true;
        params_sim.tracks_enable_mirror = true;

        params_sim.enable_track("straight_45");
        params_sim.enable_track("loop");
        params_sim.enable_track("straight_turn");
        // params_sim.enable_track("bubble_straight");
        // params_sim.enable_track("bubble_180");
        params_sim.enable_track("separation");

        // params_sim.simulation_random_output_second_way = true;
        // params_sim.random_output_probability = 0.05;
        // params_sim.evolution_learning_rate = 0.9;

        params_sim.nn.ranking_hidden_layers =
            vec![LayerDescription::new(10, ActivationFunction::SqrtSigmoid)];

        Self {
            rng: StdRng::seed_from_u64(42),

            trajectories: Default::default(),
            points_count: 0,
            graphs: Default::default(),
            drifts: vec![Default::default(); 2],

            params_intr: Default::default(),
            params_phys: Default::default(),
            params_sim: params_sim.clone(),

            points: track_complex().1,
            current_edit: 0,
            offset: Vecx2::new(0., 0.),
            drag_pos: None,
            enable_edit_track: false,

            override_nn: true,

            quota: 0,

            nn_processor: NnProcessor::new_zeros(params_sim.nn.clone(), 1.0, 0),

            simulation: CarSimulation::new(
                Default::default(),
                points_storage_to_track(track_complex()).walls,
                points_storage_to_track(track_complex()).rewards,
                &params_sim,
            ),

            all_tracks: get_all_tracks()
                .into_iter()
                .filter(|x| x.0 != "straight_45")
                .flat_map(|x| vec![x.clone(), mirror_horizontally_track2(x)])
                .collect(),
            current_track: "complex".to_string(),
            current_track_id: 10,

            current_nn: Default::default(),

            finish_step: 0,

            current_start_reward: 0,

            record_user_actions: false,
            records: vec![],
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut result: TemplateApp = Default::default();
        if let Some(storage) = cc.storage {
            let stored: TemplateApp =
                eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            // result.points = stored.points;
        }
        result
    }
}

impl TemplateApp {
    fn reset_car(&mut self) {
        self.simulation = CarSimulation::new(
            Default::default(),
            points_storage_to_track(self.all_tracks[self.current_track_id].clone())
                .walls
                .clone(),
            points_storage_to_track(self.all_tracks[self.current_track_id].clone())
                .rewards
                .clone(),
            &self.params_sim,
        );
        self.trajectories.clear();
        self.graphs.clear();
        self.points_count = 0;
        self.drifts.iter_mut().for_each(|x| x.drifts.clear());
        self.offset = Vecx2::new(0., 0.);
        self.drag_pos = None;
        self.quota = 0;
        self.nn_processor.reset();
        self.finish_step = 0;
        self.current_start_reward = 0;
        self.records.clear();
    }
}

impl eframe::App for TemplateApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let keys_down = ctx.input(|i| i.keys_down.clone());
        let backspace_pressed = ctx.input(|i| i.key_pressed(egui::Key::Backspace));
        let escape_pressed = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let wheel = ctx.input(|i| i.smooth_scroll_delta);

        if self.enable_edit_track {
            if self.current_edit >= self.points.len() {
                self.current_edit = self.points.len() - 1;
            }
            let track = points_storage_to_track(("edited".to_string(), self.points.clone()));
            self.simulation.walls = track.walls;
            self.simulation.reward_path_processor =
                crate::evolution::RewardPathProcessor::new(track.rewards);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        egui::ComboBox::from_label("track")
                            .selected_text(self.current_track.clone())
                            .show_ui(ui, |ui| {
                                let mut to_reset = false;
                                for (track_id, track) in self.all_tracks.iter().enumerate() {
                                    if ui
                                        .selectable_value(
                                            &mut self.current_track,
                                            track.0.clone(),
                                            track.0.clone(),
                                        )
                                        .changed()
                                    {
                                        self.current_track_id = track_id;
                                        // self.nn_processor.set_current_track(track_id);
                                        self.nn_processor.set_current_track(0);
                                        self.simulation = CarSimulation::new(
                                            Default::default(),
                                            points_storage_to_track(track.clone()).walls.clone(),
                                            points_storage_to_track(track.clone()).rewards.clone(),
                                            &self.params_sim,
                                        );
                                        self.points = track.1.clone();
                                        to_reset = true;
                                    }
                                }
                                if to_reset {
                                    self.reset_car();
                                }
                            });

                        if ui.button("Mutate car").clicked() {
                            self.reset_car();
                            self.simulation.car = mutate_car(Default::default(), &self.params_sim);
                        }

                        let response = ui.add(egui::Slider::new(
                            &mut self.current_start_reward,
                            0..=(self.simulation.reward_path_processor.get_rewards().len() - 2),
                        ));
                        if response.changed() {
                            self.simulation.car = place_car_to_reward(
                                Default::default(),
                                self.simulation.reward_path_processor.get_rewards(),
                                self.current_start_reward,
                            );
                        }

                        if ui
                            .button(if self.override_nn {
                                "Let NN drive!"
                            } else {
                                "Let me drive!"
                            })
                            .clicked()
                        {
                            self.override_nn = !self.override_nn;
                        }

                        if ui
                            .button(if self.record_user_actions {
                                "Stop record actions"
                            } else {
                                "Start record actions"
                            })
                            .clicked()
                        {
                            self.record_user_actions = !self.record_user_actions;
                        }

                        CollapsingHeader::new("Insert NN")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("NN weights json:");
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    ui.text_edit_multiline(&mut self.current_nn);
                                });
                                if ui.button("Set NN").clicked() {
                                    let numbers: Vec<fxx> =
                                        match serde_json::from_str(&self.current_nn) {
                                            Ok(ok) => ok,
                                            Err(err) => {
                                                dbg!(err);
                                                Default::default()
                                            }
                                        };
                                    let nn_len = self.params_sim.nn.get_nns_len();
                                    if numbers.len() == nn_len + OTHER_PARAMS_SIZE {
                                        self.current_nn.clear();
                                        self.params_sim =
                                            patch_params_sim(&numbers, &self.params_sim);
                                        // let true_params_sim = SimulationParameters::true_metric(&self.params_sim);
                                        let true_evals =
                                            eval_nn(&numbers, &self.params_phys, &self.params_sim);
                                        print_evals(&true_evals);
                                        println!(
                                            "Cost: {}",
                                            sum_evals(&true_evals, &self.params_sim, false)
                                        );
                                        println!("-----");
                                        self.nn_processor = NnProcessor::new(
                                            &numbers[..nn_len],
                                            self.params_sim.nn.clone(),
                                            self.params_sim.simulation_simple_physics,
                                            /*self.current_track_id*/ 0,
                                        );
                                        self.reset_car();
                                    } else {
                                        println!(
                                            "Wrong size: expected {}, got: {}",
                                            nn_len + OTHER_PARAMS_SIZE,
                                            numbers.len()
                                        );
                                    }
                                }
                            });

                        CollapsingHeader::new("Simulation parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                self.params_sim.ui(ui);
                            });
                        CollapsingHeader::new("Physics parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                self.params_phys.ui(ui);
                            });
                        CollapsingHeader::new("Interface parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                self.params_intr.ui(ui);
                            });

                        // self.params_phys = self
                        //     .params_sim
                        //     .patch_physics_parameters(self.params_phys.clone());

                        CollapsingHeader::new("Edit track")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.enable_edit_track, " Enable editing");
                                if ui.button("Print track").clicked() {
                                    println!();
                                    for storage in &self.points {
                                        println!(
                                            "PointsStorage {{ is_reward: {}, points: vec![",
                                            storage.is_reward
                                        );
                                        for point in &storage.points {
                                            println!(
                                                "    Vecx2::new{:.2}, {:.2}),",
                                                point.x, point.y
                                            );
                                        }
                                        println!("]}},");
                                    }
                                    println!("----");
                                }

                                ui.add(egui::Slider::new(
                                    &mut self.current_edit,
                                    0..=self.points.len().saturating_sub(1),
                                ));
                                egui_array(&mut self.points, "Points", ui, PointsStorage::egui);
                            });

                        CollapsingHeader::new("Data")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label(format!(
                                    "Distance: {:.2}%",
                                    self.simulation.reward_path_processor.distance_percent() * 100.
                                ));
                                ui.label(format!(
                                    "Current segment: {:.2}",
                                    self.simulation
                                        .reward_path_processor
                                        .get_current_segment_fxx()
                                ));
                                ui.label(format!(
                                    "All acquired: {}",
                                    self.simulation.reward_path_processor.all_acquired()
                                ));
                                ui.label(format!("Penalty: {:.2}", self.simulation.penalty));
                                ui.label(format!("Quota: {}", self.quota));
                                ui.label(format!("Time: {}", self.simulation.time_passed));
                                if self.simulation.reward_path_processor.all_acquired()
                                    && self.finish_step == 0
                                {
                                    self.finish_step = self.quota;
                                }
                                if self.finish_step != 0 {
                                    ui.label(format!("Finished at: {}", self.finish_step));
                                    if self.record_user_actions {
                                        let name = format!(
                                            "records/record_{}_{}.json",
                                            self.current_track,
                                            chrono::Local::now()
                                                .format("%Y_%m_%d__%H_%M_%S")
                                                .to_string()
                                        );
                                        println!("Saved to {name}");
                                        save_json_to_file(&self.records, &name);
                                        self.records.clear();
                                        self.record_user_actions = false;
                                    }
                                }
                            });
                    });
                    Frame::canvas(ui.style()).show(ui, |ui| {
                        let (mut response, painter) = ui.allocate_painter(
                            Vecx2::new(self.params_intr.canvas_size, self.params_intr.canvas_size)
                                .into(),
                            Sense::click_and_drag(),
                        );

                        let from_screen = emath::RectTransform::from_to(
                            response.rect,
                            Rect::from_center_size(
                                (self.simulation.car.get_center() + self.offset).into(),
                                Vecx2::new(self.params_intr.view_size, self.params_intr.view_size)
                                    .into(),
                            ),
                        );
                        let to_screen = from_screen.inverse();

                        let scroll_amount = 0.05;
                        if response.hovered() {
                            if wheel.y > 0. {
                                self.params_intr.view_size /= 1. + scroll_amount;
                            } else if wheel.y < 0. {
                                self.params_intr.view_size *= 1. + scroll_amount;
                            }
                            ui.ctx().input_mut(|input| {
                                input.smooth_scroll_delta[1] = 0.0;
                            });
                        }

                        if response.dragged_by(PointerButton::Middle) {
                            if let Some(pos_screen) = response.interact_pointer_pos() {
                                let pos = from_screen.transform_pos(pos_screen);
                                let delta_screen = response.drag_delta();
                                let pos_delta =
                                    from_screen.transform_pos(pos_screen + delta_screen);
                                let delta = pos_delta - pos;
                                self.offset -= delta.into();
                            }
                        }

                        if self.current_edit < self.points.len() {
                            let storage = &mut self.points[self.current_edit].points;

                            if response.dragged_by(PointerButton::Primary) {
                                if let Some(pos_screen) = response.interact_pointer_pos() {
                                    let pos = from_screen.transform_pos(pos_screen);
                                    if let Some(drag_pos) = self.drag_pos {
                                        let point = &mut storage[drag_pos];
                                        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Move);
                                        let delta_screen = response.drag_delta();
                                        let pos_delta =
                                            from_screen.transform_pos(pos_screen + delta_screen);
                                        let delta = pos_delta - pos;
                                        *point += delta.into();
                                        response.mark_changed();
                                    } else if response.drag_started() {
                                        for (i, point) in storage.iter_mut().enumerate() {
                                            if (*point - pos.into()).length() < 30. * 2. {
                                                self.drag_pos = Some(i);
                                                ui.output_mut(|o| {
                                                    o.cursor_icon = egui::CursorIcon::Move
                                                });
                                                let delta_screen = response.drag_delta();
                                                let pos_delta = from_screen
                                                    .transform_pos(pos_screen + delta_screen);
                                                let delta = pos_delta - pos;
                                                *point += delta.into();
                                                response.mark_changed();
                                                break;
                                            }
                                        }
                                    }
                                }
                            } else {
                                self.drag_pos = None;
                            }

                            if response.clicked_by(PointerButton::Primary) {
                                if let Some(pos_screen) = response.interact_pointer_pos() {
                                    storage.push(from_screen.transform_pos(pos_screen).into());
                                }
                            } else if response.clicked_by(PointerButton::Secondary) {
                                if let Some(pos_screen) = response.interact_pointer_pos() {
                                    let pos = Vecx2::from(from_screen.transform_pos(pos_screen));
                                    let mut to_remove: Option<usize> = None;
                                    for (i, point) in storage.iter().enumerate() {
                                        if (*point - pos.into()).length() < 30. {
                                            to_remove = Some(i);
                                            break;
                                        }
                                    }
                                    if let Some(to_remove) = to_remove {
                                        storage.remove(to_remove);
                                    } else if storage.len() >= 2 {
                                        let x = storage
                                            .windows(2)
                                            .enumerate()
                                            .map(|(i, x)| {
                                                let (a, b) = (x[0], x[1]);
                                                let (res, t) = project_to_segment(pos.into(), a, b);
                                                (i, res, (pos - res).length(), t)
                                            })
                                            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                                            .map(|x| x.0)
                                            .unwrap();
                                        storage.insert(x + 1, pos.into());
                                    }
                                }
                            }
                        }

                        if self.enable_edit_track {
                            for (storage_pos, storage) in self.points.iter().enumerate() {
                                for point in &storage.points {
                                    painter.add(Shape::circle_stroke(
                                        to_screen.transform_pos(point.into()),
                                        (to_screen
                                            .transform_pos(Pos2::from(point) + vec2(30., 0.)))
                                        .to_vec2()
                                        .length(),
                                        Stroke::new(2.0, Color32::from_rgb(0, 0, 0)),
                                    ));
                                }

                                painter.add(Shape::line(
                                    storage
                                        .points
                                        .iter()
                                        .map(|p| to_screen.transform_pos(p.into()))
                                        .collect(),
                                    Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                                ));

                                for (pos, point) in storage.points.iter().enumerate() {
                                    painter.add(painter.fonts(|f| {
                                        Shape::text(
                                            f,
                                            to_screen.transform_pos(point.into()),
                                            Align2::CENTER_CENTER,
                                            format!("{storage_pos}.{pos}"),
                                            FontId::new(10., FontFamily::Monospace),
                                            Color32::BLACK,
                                        )
                                    }));
                                }
                            }
                        }

                        painter.add(Shape::line(
                            self.trajectories
                                .iter()
                                .map(|p| to_screen.transform_pos(p.into()))
                                .collect(),
                            Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                        ));

                        for wheel_drifts in &self.drifts {
                            for drift in &wheel_drifts.drifts {
                                painter.add(Shape::line(
                                    drift
                                        .iter()
                                        .map(|p| to_screen.transform_pos(p.into()))
                                        .collect(),
                                    Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                                ));
                            }
                        }

                        if escape_pressed {
                            self.reset_car();
                        }

                        self.simulation.draw(&painter, &to_screen);

                        self.quota += 1;

                        for simulation_i in 0..self.params_intr.simulations_per_frame {
                            self.simulation.step(
                                &self.params_phys,
                                &self.params_sim,
                                &mut |origin, dir_pos, t, t2| {
                                    if simulation_i == 0 {
                                        painter.add(Shape::line(
                                            vec![
                                                to_screen.transform_pos(origin.into()),
                                                to_screen.transform_pos(
                                                    (origin + (dir_pos - origin) * t).into(),
                                                ),
                                            ],
                                            Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                                        ));
                                        if self.params_sim.nn.pass_dirs_second_layer {
                                            painter.add(Shape::circle_filled(
                                                to_screen.transform_pos(
                                                    (origin + (dir_pos - origin) * t).into(),
                                                ),
                                                3.,
                                                Color32::from_rgb(0, 0, 255),
                                            ));
                                            if let Some(t2) = t2 {
                                                painter.add(Shape::line(
                                                    vec![
                                                        to_screen.transform_pos(
                                                            (origin + (dir_pos - origin) * t)
                                                                .into(),
                                                        ),
                                                        to_screen.transform_pos(
                                                            (origin + (dir_pos - origin) * t2)
                                                                .into(),
                                                        ),
                                                    ],
                                                    Stroke::new(1.0, Color32::from_rgb(0, 0, 255)),
                                                ));
                                                painter.add(Shape::circle_filled(
                                                    to_screen.transform_pos(
                                                        (origin + (dir_pos - origin) * t2).into(),
                                                    ),
                                                    2.,
                                                    Color32::from_rgb(0, 0, 128),
                                                ));
                                            }
                                        }
                                    }
                                },
                                &mut |time_passed,
                                      distance_percent,
                                      dpenalty,
                                      dirs,
                                      dirs_second_layer,
                                      internals,
                                      current_segment_fxx,
                                      simulation_vars| {
                                    self.graphs.add_point(
                                        "input",
                                        "time_passed",
                                        time_passed,
                                        Color32::YELLOW,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "distance_percent",
                                        distance_percent,
                                        Color32::KHAKI,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "dpenalty",
                                        dpenalty,
                                        Color32::DARK_RED,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "acceleration value",
                                        internals.local_acceleration.length(),
                                        Color32::BLUE,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "acceleration angle",
                                        internals.acceleration_angle(),
                                        Color32::BLUE,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "torque",
                                        internals.angle_acceleration,
                                        Color32::RED,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "speed value",
                                        internals.local_speed.length(),
                                        Color32::GREEN,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "speed angle",
                                        internals.speed_angle(),
                                        Color32::GREEN,
                                        &self.params_intr,
                                    );
                                    self.graphs.add_point(
                                        "input",
                                        "wheel angle",
                                        internals.wheel_angle,
                                        Color32::GREEN,
                                        &self.params_intr,
                                    );

                                    self.graphs.add_point(
                                        "input",
                                        "angle speed",
                                        internals.angle_speed,
                                        Color32::GREEN,
                                        &self.params_intr,
                                    );

                                    if self.params_intr.show_dirs {
                                        for (pos, dir) in dirs.iter().enumerate() {
                                            self.graphs.add_point(
                                                "input",
                                                format!("z dir {pos}"),
                                                dir.unwrap_or(-1.),
                                                Color32::BLACK,
                                                &self.params_intr,
                                            );
                                        }
                                    }

                                    if self.override_nn {
                                        let input_arr = [
                                            if keys_down.contains(&egui::Key::Space) {
                                                20.
                                            } else {
                                                0.0
                                            },
                                            if keys_down.contains(&egui::Key::ArrowUp) {
                                                20.
                                            } else {
                                                0.0
                                            },
                                            if keys_down.contains(&egui::Key::ArrowDown) {
                                                20.
                                            } else {
                                                0.0
                                            },
                                            if !keys_down.contains(&egui::Key::ArrowLeft)
                                                && !keys_down.contains(&egui::Key::ArrowRight)
                                            {
                                                20.
                                            } else {
                                                0.0
                                            },
                                            if keys_down.contains(&egui::Key::ArrowLeft) {
                                                20.
                                            } else {
                                                0.0
                                            },
                                            if keys_down.contains(&egui::Key::ArrowRight) {
                                                20.
                                            } else {
                                                0.0
                                            },
                                        ];

                                        if self.record_user_actions {
                                            self.records.push((
                                                self.nn_processor
                                                    .calc_input_vector(
                                                        time_passed,
                                                        distance_percent,
                                                        dpenalty,
                                                        dirs,
                                                        dirs_second_layer,
                                                        current_segment_fxx,
                                                        internals,
                                                        &self.params_sim,
                                                        &self.params_phys,
                                                    )
                                                    .to_owned(),
                                                input_arr.to_vec(),
                                            ));
                                        }

                                        CarInput::from_fxx(&input_arr)
                                    } else {
                                        let result = self.nn_processor.process(
                                            time_passed,
                                            distance_percent,
                                            dpenalty,
                                            dirs,
                                            dirs_second_layer,
                                            current_segment_fxx,
                                            internals,
                                            &self.params_sim,
                                            simulation_vars,
                                        );
                                        self.graphs.add_point(
                                            "output",
                                            "brake",
                                            result.brake as usize as fxx,
                                            Color32::BLUE,
                                            &self.params_intr,
                                        );
                                        self.graphs.add_point(
                                            "output",
                                            "acceleration",
                                            result.acceleration,
                                            Color32::BLUE,
                                            &self.params_intr,
                                        );
                                        self.graphs.add_point(
                                            "output",
                                            "remove_turn",
                                            result.remove_turn as usize as fxx,
                                            Color32::GREEN,
                                            &self.params_intr,
                                        );
                                        self.graphs.add_point(
                                            "output",
                                            "turn",
                                            result.turn,
                                            Color32::GREEN,
                                            &self.params_intr,
                                        );
                                        result
                                    }
                                },
                                &mut |i, pos, value| {
                                    self.drifts[i].process_point(
                                        pos,
                                        value > self.params_intr.drift_starts_at,
                                    );
                                },
                                &mut |car| {
                                    if simulation_i == 0 {
                                        car.draw_forces(
                                            &painter,
                                            &self.params_intr,
                                            &self.params_phys,
                                            &to_screen,
                                        );
                                    }

                                    self.trajectories.push_back(car.get_center());
                                    self.points_count += 1;
                                    if self.points_count > self.params_intr.graph_points_size_limit
                                    {
                                        self.trajectories.pop_front();
                                        self.points_count -= 1;
                                    }
                                },
                            );
                        }
                    });
                });

                CollapsingHeader::new("Graphs")
                    .default_open(false)
                    .show(ui, |ui| {
                        let create_plot =
                            |name: &str, group_name: &str, params: &InterfaceParameters| {
                                Plot::new(format!("items_demo {} {}", name, group_name))
                                    .legend(Legend::default().position(Corner::RightTop))
                                    .show_x(false)
                                    .show_y(false)
                                    .allow_zoom([false, false])
                                    .allow_scroll([false, false])
                                    .allow_boxed_zoom(false)
                                    .width(params.plot_size as f32)
                                    .height(params.plot_size as f32)
                            };
                        for (group_name, graphs) in self.graphs.points.iter() {
                            ui.horizontal_wrapped(|ui| {
                                for (name, (points, color)) in graphs {
                                    ui.allocate_ui(
                                        vec2(
                                            self.params_intr.plot_size as f32 + 5.,
                                            self.params_intr.plot_size as f32 + 5.,
                                        ),
                                        |ui| {
                                            create_plot(name, group_name, &self.params_intr).show(
                                                ui,
                                                |plot_ui| {
                                                    let line = Line::new(PlotPoints::from_iter(
                                                        points.clone(),
                                                    ))
                                                    .fill(0.)
                                                    .color(*color);
                                                    plot_ui.line(line.name(name));
                                                },
                                            );
                                        },
                                    );
                                }
                            });
                        }
                    });
            });
        });

        ctx.request_repaint();
    }
}
