use crate::common::*;
use crate::evolution::*;
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
use egui::{emath, Pos2, Rect, Sense, Vec2};
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
        params: &Parameters,
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

impl StorageElem2 for PosInStorage {
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

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct PointsStorage {
    pub is_reward: bool,
    pub points: Storage2<PosInStorage>,
}

impl Default for PointsStorage {
    fn default() -> Self {
        Self {
            is_reward: false,
            points: Storage2::new("storage".to_string()),
        }
    }
}

impl StorageElem2 for PointsStorage {
    fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        ui.selectable_value(&mut self.is_reward, false, "Walls");
        ui.selectable_value(&mut self.is_reward, true, "Rewards");
        self.points.egui_inner(ui, data_id);
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

    params: Parameters,

    // walls: Storage2<Wall>,
    // rewards: Storage2<Reward>,
    points: Storage2<PointsStorage>,
    current_edit: usize,
    offset: Pos2,
    drag_pos: Option<usize>,

    #[serde(skip)]
    rng: StdRng,

    #[serde(skip)]
    simulation: CarSimulation,

    override_nn: bool,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            rng: StdRng::seed_from_u64(42),

            trajectories: Default::default(),
            points_count: 0,
            graphs: Default::default(),
            drifts: vec![Default::default(); 2],

            params: Default::default(),

            points: Storage2::new("Edits".to_string()),
            current_edit: 0,
            offset: pos2(0., 0.),
            drag_pos: None,

            override_nn: true,

            simulation: CarSimulation::new(
                Default::default(),
                {
                    #[allow(clippy::excessive_precision)]
                    let (sizes, values) = include!("nn.data");
                    let mut result = NeuralNetwork::new(sizes);
                    result
                        .get_values_mut()
                        .iter_mut()
                        .zip(values.iter())
                        .for_each(|(x, y)| *x = *y);
                    result
                },
                track_complex().walls,
                track_complex().rewards,
                // mirror_horizontally(track_complex()).0,
                // mirror_horizontally(track_complex()).1,
                // mirror_horizontally(track_turn_right_smooth()).0,
                // mirror_horizontally(track_turn_right_smooth()).1,
            ),
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

        if self.current_edit >= self.points.storage.len() {
            self.current_edit = self.points.storage.len() - 1;
        }

        let mut walls: Vec<Wall> = Default::default();
        let mut rewards: Vec<Reward> = Default::default();
        for elem in &self.points.storage {
            if elem.is_reward {
                rewards.extend(rewards_from_points(
                    elem.points.storage.iter().map(|PosInStorage(a)| *a),
                ));
            } else {
                walls.extend(walls_from_points(
                    elem.points.storage.iter().map(|PosInStorage(a)| *a),
                ));
            }
        }
        // self.simulation.walls = walls;
        // self.simulation.reward_path_processor = crate::evolution::RewardPathProcessor::new(rewards);

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    // self.params.ui(ui);
                    Frame::canvas(ui.style()).show(ui, |ui| {
                        let (mut response, painter) = ui.allocate_painter(
                            Vec2::new(self.params.canvas_size, self.params.canvas_size),
                            Sense::click_and_drag(),
                        );

                        let from_screen = emath::RectTransform::from_to(
                            response.rect,
                            Rect::from_center_size(
                                self.simulation.car.get_center() + self.offset.to_vec2(),
                                Vec2::new(self.params.view_size, self.params.view_size),
                            ),
                        );
                        let to_screen = from_screen.inverse();

                        let scroll_amount = 0.05;
                        if wheel.y > 0. {
                            self.params.view_size /= 1. + scroll_amount;
                        } else if wheel.y < 0. {
                            self.params.view_size *= 1. + scroll_amount;
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

                        if self.current_edit < self.points.storage.len() {
                            let storage =
                                &mut self.points.storage[self.current_edit].points.storage;

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

                        for (storage_pos, storage) in self.points.storage.iter().enumerate() {
                            for point in &storage.points.storage {
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
                                    .storage
                                    .iter()
                                    .map(|p| to_screen.transform_pos(p.0))
                                    .collect(),
                                Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                            ));

                            for (pos, point) in storage.points.storage.iter().enumerate() {
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
                            for storage in &self.points.storage {
                                if storage.is_reward {
                                    println!("rewards_from_points(vec![");
                                } else {
                                    println!("walls_from_points(vec![");
                                }
                                for point in &storage.points.storage {
                                    println!("    pos2({:.2}, {:.2}),", point.0.x, point.0.y);
                                }
                                println!("].into_iter()),");
                            }
                            println!("----");
                        }

                        self.simulation.draw(&painter, &to_screen);

                        self.simulation.step(
                            &self.params.physics,
                            &mut |origin, dir_pos, t| {
                                painter.add(Shape::line(
                                    vec![
                                        to_screen.transform_pos(origin),
                                        to_screen.transform_pos(origin + (dir_pos - origin) * t),
                                    ],
                                    Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                                ));
                            },
                            &mut |car| {
                                if backspace_pressed {
                                    self.override_nn = !self.override_nn;
                                }
                                if !self.override_nn {
                                    return false;
                                }
                                car.process_input(&CarInput {
                                    brake: keys_down.contains(&egui::Key::Space),
                                    acceleration: if keys_down.contains(&egui::Key::ArrowUp) {
                                        1.0
                                    } else if keys_down.contains(&egui::Key::ArrowDown) {
                                        -1.0
                                    } else {
                                        0.0
                                    },
                                    remove_turn: !(keys_down.contains(&egui::Key::ArrowLeft) && keys_down.contains(&egui::Key::ArrowRight)),
                                    turn: if keys_down.contains(&egui::Key::ArrowLeft) {
                                        1.0
                                    } else if keys_down.contains(&egui::Key::ArrowRight) {
                                        -1.0
                                    } else {
                                        0.0
                                    },
                                }, &self.params.physics);
                                true
                            },
                            &mut |i, pos, value| {
                                self.drifts[i].process_point(
                                    pos.to_pos2(),
                                    value > self.params.drift_starts_at,
                                );
                            },
                            &mut |car| {
                                car.draw_forces(&painter, &self.params, &to_screen);

                                self.trajectories.push_back(car.get_center());
                                self.points_count += 1;
                                if self.points_count > self.params.graph_points_size_limit {
                                    self.trajectories.pop_front();
                                    self.points_count -= 1;
                                }

                                let values = car.get_internal_values();

                                self.graphs.add_point(
                                    "g",
                                    "acceleration",
                                    values.local_acceleration.length(),
                                    Color32::GRAY,
                                    &self.params,
                                );

                                self.graphs.add_point(
                                    "g",
                                    "torque",
                                    values.angle_acceleration,
                                    Color32::BROWN,
                                    &self.params,
                                );

                                self.graphs.add_point(
                                    "g",
                                    "speed",
                                    values.local_speed.length(),
                                    Color32::BLACK,
                                    &self.params,
                                );
                            },
                        );
                    });
                });

                // ui.label(format!("Reward: {}", self.simulation.sum_reward));
                // ui.label(format!(
                //     "Distance reward: {}",
                //     self.simulation.total_reward()
                // ));
                ui.label(format!("Time: {}", self.simulation.time_passed));
                ui.add(egui::Slider::new(
                    &mut self.current_edit,
                    0..=self.points.storage.len().saturating_sub(1),
                ));
                self.points.egui(ui);
            });
        });

        ctx.request_repaint();
    }
}
