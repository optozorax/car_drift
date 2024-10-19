use crate::nn::*;
use crate::common::*;
use crate::evolution::*;
use crate::physics::*;
use crate::storage::*;
use egui::containers::Frame;
use egui::*;
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
use egui::{emath, Pos2, Rect, Sense, Vec2};
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
    drifts: VecDeque<VecDeque<Pos2>>,
}

impl Drifts {
    fn process_point(&mut self, point: Pos2, is_drift: bool) {
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
        value: f32,
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

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Default)]
struct PosInStorage(Pos2);

impl PosInStorage {
    fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        egui::Grid::new(data_id.with("wall_grid"))
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Pos X:");
                ui.add(
                    DragValue::new(&mut self.0.x)
                        .speed(5.0)
                        .min_decimals(0)
                        .max_decimals(0),
                );
                ui.end_row();

                ui.label("Pos Y:");
                ui.add(
                    DragValue::new(&mut self.0.y)
                        .speed(5.0)
                        .min_decimals(0)
                        .max_decimals(0),
                );
                ui.end_row();
            });
    }
}

#[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
struct PointsStorage {
    pub is_reward: bool,
    pub points: Vec<PosInStorage>,
}

impl PointsStorage {
    fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        ui.selectable_value(&mut self.is_reward, false, "Walls");
        ui.selectable_value(&mut self.is_reward, true, "Rewards");
        egui_array_inner(&mut self.points, ui, data_id, PosInStorage::egui, false);
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    // in car local coordinates
    trajectories: VecDeque<Pos2>,
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
    offset: Pos2,
    drag_pos: Option<usize>,

    #[serde(skip)]
    rng: StdRng,

    #[serde(skip)]
    simulation: CarSimulation,

    #[serde(skip)]
    nn_processor: NnProcessor,

    override_nn: bool,

    quota: usize,

    #[serde(skip)]
    all_tracks: Vec<Track>,

    current_track: String,

    current_nn: String,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let mut params_sim = SimulationParameters::default();
        params_sim.enable_all_tracks();
        params_sim.simulation_enable_random_nn_output = false;
        params_sim.eval_penalty.value = 200.;
        params_sim.rewards_add_each_acquire = true;
        params_sim.rewards_enable_distance_integral = true;
        params_sim.mutate_car_enable = false;
        params_sim.eval_add_min_distance = false;
        params_sim.simulation_stop_penalty.value = 100.;
        params_sim.simulation_simple_physics = 1.;
        params_sim.rewards_second_way = true;
        params_sim.evolve_simple_physics = true;

        params_sim.nn.pass_internals = true;
        params_sim.nn.pass_simple_physics_value = false;
        params_sim.nn.view_angle_ratio = 5. / 6.;

        Self {
            rng: StdRng::seed_from_u64(42),

            trajectories: Default::default(),
            points_count: 0,
            graphs: Default::default(),
            drifts: vec![Default::default(); 2],

            params_intr: Default::default(),
            params_phys: Default::default(),
            params_sim: params_sim.clone(),

            points: Vec::default(),
            current_edit: 0,
            offset: pos2(0., 0.),
            drag_pos: None,

            override_nn: true,

            quota: 0,

            nn_processor: NnProcessor::new_zeros(Default::default(), 1.0, 0),

            simulation: CarSimulation::new(
                Default::default(),
                track_complex().walls,
                track_complex().rewards,
                &params_sim,
            ),

            all_tracks: get_all_tracks().into_iter().flat_map(|x| vec![x.clone(), mirror_horizontally(x)]).collect(),
            current_track: "complex".to_string(),

            current_nn: Default::default(),
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut result: TemplateApp = Default::default();
        if let Some(storage) = cc.storage {
            let stored: TemplateApp =
                eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            result.points = stored.points;
        }
        result
    }
}

impl TemplateApp {
    fn reset_car(&mut self) {
        self.simulation.reset();
        self.trajectories.clear();
        self.graphs.clear();
        self.points_count = 0;
        self.drifts.iter_mut().for_each(|x| x.drifts.clear());
        self.offset = pos2(0., 0.);
        self.drag_pos = None;
        self.quota = 0;
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

        // if self.current_edit >= self.points.len() {
        //     self.current_edit = self.points.len() - 1;
        // }

        let mut walls: Vec<Wall> = Default::default();
        let mut rewards: Vec<Reward> = Default::default();
        for elem in &self.points {
            if elem.is_reward {
                rewards.extend(rewards_from_points(
                    elem.points.iter().map(|PosInStorage(a)| *a),
                ));
            } else {
                walls.extend(walls_from_points(
                    elem.points.iter().map(|PosInStorage(a)| *a),
                ));
            }
        }
        // self.simulation.walls = walls;
        // self.simulation.reward_path_processor = crate::evolution::RewardPathProcessor::new(rewards);

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        egui::ComboBox::from_label("track")
                            .selected_text(self.current_track.clone())
                            .show_ui(ui, |ui| {
                                let mut to_reset = false;
                                for track in &self.all_tracks {
                                    if ui.selectable_value(&mut self.current_track, track.name.clone(), track.name.clone()).changed() {
                                        self.simulation = CarSimulation::new(
                                            Default::default(),
                                            track.walls.clone(),
                                            track.rewards.clone(),
                                            &self.params_sim,
                                        );
                                        to_reset = true;
                                    }
                                }
                                if to_reset {
                                    self.reset_car();
                                }
                            }
                        );

                        ui.label("NN weights json:");
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.text_edit_multiline(&mut self.current_nn);
                        });
                        if ui.button("Set NN").clicked() {
                            let numbers: Vec<f32> = match serde_json::from_str(&self.current_nn) {
                                Ok(ok) => ok,
                                Err(err) => {
                                    dbg!(err);
                                    Default::default()
                                }
                            };
                            let nn_sizes = self.params_sim.nn.get_nn_sizes();
                            let nn_len = self.params_sim.nn.get_nn_len();
                            if numbers.len() == nn_len + OTHER_PARAMS_SIZE {
                                self.current_nn.clear();
                                self.params_sim = patch_params_sim(&numbers, &self.params_sim);
                                let true_params_sim = SimulationParameters::true_metric(&self.params_sim);
                                let true_evals = eval_nn(&numbers, &self.params_phys, &true_params_sim);
                                print_evals(&true_evals);
                                println!("Cost: {}", sum_evals(&true_evals, &true_params_sim));
                                println!("-----");
                                self.nn_processor = NnProcessor::new(&numbers[..nn_len], self.params_sim.nn.clone(), self.params_sim.simulation_simple_physics, 0); // todo(optozorax): pass current track
                                self.reset_car();
                            } else {
                                println!("Wrong size: expected {nn_len}, got: {}", numbers.len());
                            }
                        }

                        if ui.button("Mutate car").clicked() {
                            self.reset_car();
                            self.simulation.car = mutate_car(Default::default(), &self.params_sim);
                        }

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

                        self.params_phys = self
                            .params_sim
                            .patch_physics_parameters(self.params_phys.clone());
                    });
                    Frame::canvas(ui.style()).show(ui, |ui| {
                        let (mut response, painter) = ui.allocate_painter(
                            Vec2::new(self.params_intr.canvas_size, self.params_intr.canvas_size),
                            Sense::click_and_drag(),
                        );

                        let from_screen = emath::RectTransform::from_to(
                            response.rect,
                            Rect::from_center_size(
                                self.simulation.car.get_center() + self.offset.to_vec2(),
                                Vec2::new(self.params_intr.view_size, self.params_intr.view_size),
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
                                self.offset -= delta;
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
                                        point.0 += delta;
                                        response.mark_changed();
                                    } else {
                                        for (i, point) in storage.iter_mut().enumerate() {
                                            if (point.0 - pos).length() < 30. {
                                                self.drag_pos = Some(i);
                                                ui.output_mut(|o| {
                                                    o.cursor_icon = egui::CursorIcon::Move
                                                });
                                                let delta_screen = response.drag_delta();
                                                let pos_delta = from_screen
                                                    .transform_pos(pos_screen + delta_screen);
                                                let delta = pos_delta - pos;
                                                point.0 += delta;
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
                                    storage
                                        .push(PosInStorage(from_screen.transform_pos(pos_screen)));
                                }
                            } else if response.clicked_by(PointerButton::Secondary) {
                                if let Some(pos_screen) = response.interact_pointer_pos() {
                                    let pos = from_screen.transform_pos(pos_screen);
                                    let mut to_remove: Option<usize> = None;
                                    for (i, point) in storage.iter().enumerate() {
                                        if (point.0 - pos).length() < 30. {
                                            to_remove = Some(i);
                                            break;
                                        }
                                    }
                                    if let Some(to_remove) = to_remove {
                                        storage.remove(to_remove);
                                    }
                                }
                            }
                        }

                        for (storage_pos, storage) in self.points.iter().enumerate() {
                            for point in &storage.points {
                                painter.add(Shape::circle_stroke(
                                    to_screen.transform_pos(point.0),
                                    (to_screen.transform_pos(point.0)
                                        - to_screen.transform_pos(point.0 + vec2(30., 0.)))
                                    .length(),
                                    Stroke::new(2.0, Color32::from_rgb(0, 0, 0)),
                                ));
                            }

                            painter.add(Shape::line(
                                storage
                                    .points
                                    .iter()
                                    .map(|p| to_screen.transform_pos(p.0))
                                    .collect(),
                                Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                            ));

                            for (pos, point) in storage.points.iter().enumerate() {
                                painter.add(painter.fonts(|f| {
                                    Shape::text(
                                        f,
                                        to_screen.transform_pos(point.0),
                                        Align2::CENTER_CENTER,
                                        format!("{storage_pos}.{pos}"),
                                        FontId::new(10., FontFamily::Monospace),
                                        Color32::BLACK,
                                    )
                                }));
                            }
                        }

                        painter.add(Shape::line(
                            self.trajectories
                                .iter()
                                .map(|p| to_screen.transform_pos(*p))
                                .collect(),
                            Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                        ));

                        for wheel_drifts in &self.drifts {
                            for drift in &wheel_drifts.drifts {
                                painter.add(Shape::line(
                                    drift.iter().map(|p| to_screen.transform_pos(*p)).collect(),
                                    Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                                ));
                            }
                        }

                        if escape_pressed {
                            self.reset_car();

                            println!();
                            for storage in &self.points {
                                if storage.is_reward {
                                    println!("rewards_from_points(vec![");
                                } else {
                                    println!("walls_from_points(vec![");
                                }
                                for point in &storage.points {
                                    println!("    pos2({:.2}, {:.2}),", point.0.x, point.0.y);
                                }
                                println!("].into_iter()),");
                            }
                            println!("----");
                        }

                        self.simulation.draw(&painter, &to_screen);

                        self.quota += 1;

                        self.simulation.step(
                            &self.params_phys,
                            &self.params_sim,
                            &mut |origin, dir_pos, t| {
                                painter.add(Shape::line(
                                    vec![
                                        to_screen.transform_pos(origin),
                                        to_screen.transform_pos(origin + (dir_pos - origin) * t),
                                    ],
                                    Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                                ));
                            },
                            &mut |time_passed, distance_percent, dpenalty, dirs, internals| {
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

                                if backspace_pressed {
                                    self.override_nn = !self.override_nn;
                                }
                                if self.override_nn {
                                    CarInput {
                                        brake: keys_down.contains(&egui::Key::Space) as usize
                                            as f32,
                                        acceleration: if keys_down.contains(&egui::Key::ArrowUp) {
                                            1.0
                                        } else if keys_down.contains(&egui::Key::ArrowDown) {
                                            -1.0
                                        } else {
                                            0.0
                                        },
                                        remove_turn: (!(keys_down.contains(&egui::Key::ArrowLeft)
                                            || keys_down.contains(&egui::Key::ArrowRight)))
                                            as usize
                                            as f32,
                                        turn: if keys_down.contains(&egui::Key::ArrowLeft) {
                                            1.0
                                        } else if keys_down.contains(&egui::Key::ArrowRight) {
                                            -1.0
                                        } else {
                                            0.0
                                        },
                                    }
                                } else {
                                    let result = self.nn_processor.process(
                                        time_passed,
                                        distance_percent,
                                        dpenalty,
                                        dirs,
                                        internals,
                                        &self.params_sim,
                                    );
                                    self.graphs.add_point(
                                        "output",
                                        "brake",
                                        result.brake as usize as f32,
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
                                        result.remove_turn as usize as f32,
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
                                    pos.to_pos2(),
                                    value > self.params_intr.drift_starts_at,
                                );
                            },
                            &mut |car| {
                                car.draw_forces(
                                    &painter,
                                    &self.params_intr,
                                    &self.params_phys,
                                    &to_screen,
                                );

                                self.trajectories.push_back(car.get_center());
                                self.points_count += 1;
                                if self.points_count > self.params_intr.graph_points_size_limit {
                                    self.trajectories.pop_front();
                                    self.points_count -= 1;
                                }
                            },
                        );
                    });
                });

                let create_plot = |name: &str, group_name: &str, params: &InterfaceParameters| {
                    Plot::new(format!("items_demo {} {}", name, group_name))
                        .legend(Legend::default().position(Corner::RightTop))
                        .show_x(false)
                        .show_y(false)
                        .allow_zoom([false, false])
                        .allow_scroll([false, false])
                        .allow_boxed_zoom(false)
                        .width(params.plot_size)
                        .height(params.plot_size)
                };
                for (group_name, graphs) in self.graphs.points.iter() {
                    ui.horizontal_wrapped(|ui| {
                        for (name, (points, color)) in graphs {
                            ui.allocate_ui(
                                vec2(
                                    self.params_intr.plot_size + 5.,
                                    self.params_intr.plot_size + 5.,
                                ),
                                |ui| {
                                    create_plot(name, group_name, &self.params_intr).show(
                                        ui,
                                        |plot_ui| {
                                            let line =
                                                Line::new(PlotPoints::from_iter(points.clone()))
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

                ui.label(format!(
                    "Distance: {:.2}%",
                    self.simulation.reward_path_processor.distance_percent() * 100.
                ));
                ui.label(format!(
                    "Rewards percent: {:.2}%",
                    self.simulation
                        .reward_path_processor
                        .rewards_acquired_percent(&self.params_sim)
                        * 100.
                ));
                ui.label(format!("Penalty: {:.2}", self.simulation.penalty));
                ui.label(format!("Quota: {}", self.quota));
                ui.label(format!("Time: {}", self.simulation.time_passed));
                ui.add(egui::Slider::new(
                    &mut self.current_edit,
                    0..=self.points.len().saturating_sub(1),
                ));
                egui_array(&mut self.points, "Points", ui, PointsStorage::egui);
            });
        });

        ctx.request_repaint();
    }
}
