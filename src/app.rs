use crate::storage::Storage2;
use egui::containers::Frame;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::Shape;
use egui::Stroke;
use crate::common::*;
use crate::physics::*;
use egui::{emath, Pos2, Rect, Sense, Vec2};
use egui_plot::Corner;
use egui_plot::Legend;
use egui_plot::Line;
use egui_plot::Plot;
use egui_plot::PlotPoints;
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

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    car: Car,

    // in car local coordinates
    trajectories: VecDeque<Pos2>,
    points_count: usize,

    graphs: Graphs,
    drifts: Vec<Drifts>,

    params: Parameters,

    walls: Storage2<Wall>,
    rewards: Storage2<Reward>,
    sum_reward: f32,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            car: Self::get_default_cars(),

            trajectories: Default::default(),
            points_count: 0,
            graphs: Default::default(),
            drifts: vec![Default::default(); 2],

            params: Default::default(),

            walls: Storage2::from_vec(vec![
                Wall::new(pos2(1396.0, 600.0), vec2(25.0, 1500.0), 1.6),
                Wall::new(pos2(1065.0, 0.0), vec2(25.0, 1121.2), 1.6),
                Wall::new(pos2(-13.0, 300.0), vec2(25.0, 353.2), 0.0),
                Wall::new(pos2(2882.0, -127.0), vec2(25.0, 750.0), 0.0),
                Wall::new(pos2(2146.0, -721.0), vec2(25.0, 750.0), 0.0),
                Wall::new(pos2(2912.0, -1423.0), vec2(25.0, 828.4), 1.6),
                Wall::new(pos2(3690.0, 861.0), vec2(25.0, 2416.0), 0.0),
                Wall::new(pos2(445.0, 1694.0), vec2(25.0, 3689.0), 5.1),
                Wall::new(pos2(-2830.0, -1535.0), vec2(25.0, 1905.0), 0.0),
                Wall::new(pos2(-913.0, 112.0), vec2(25.0, 1104.0), 2.1),
                Wall::new(pos2(-2062.0, -1517.0), vec2(25.0, 1375.0), 6.1),
                Wall::new(pos2(-1055.0, -3250.0), vec2(25.0, 1905.0), 1.6),
                Wall::new(pos2(-760.0, -2845.0), vec2(25.0, 1625.0), 1.6),
                Wall::new(pos2(750.0, -3050.0), vec2(25.0, 320.0), 0.0),
            ]),

            rewards: Storage2::from_vec(vec![
                Reward::new(pos2(635.0, 290.0), 305.0),
                Reward::new(pos2(1240.0, 290.0), 305.0),
                Reward::new(pos2(1825.0, 310.0), 300.0),
                Reward::new(pos2(2430.0, 180.0), 500.0),
                Reward::new(pos2(2430.0, -515.0), 500.0),
                Reward::new(pos2(2480.0, -1040.0), 520.0),
                Reward::new(pos2(3270.0, -990.0), 500.0),
                Reward::new(pos2(3295.0, 510.0), 475.0),
                Reward::new(pos2(-660.0, 725.0), 500.0),
                Reward::new(pos2(-2465.0, -185.0), 705.0),
                Reward::new(pos2(-2560.0, -2945.0), 500.0),
                Reward::new(pos2(395.0, -3040.0), 500.0),
                Reward::new(pos2(150.0, -3075.0), 500.0),
                Reward::new(pos2(-140.0, -3065.0), 500.0),
                Reward::new(pos2(-560.0, -3030.0), 500.0),
            ]),
            sum_reward: 0.,
        }
    }
}

impl TemplateApp {
    pub fn get_default_cars() -> Car {
        Default::default()
        // Car::new(
        //     /* center = */ pos2(150., 250.),
        //     /* size = */ vec2(100., 50.) / 2.,
        //     /* angle = */ 0.2,
        //     /* mass = */ 100.,
        //     /* speed = */ vec2(0., 0.),
        //     /* angle_speed = */ 0.,
        //     /* wheels = */ vec![
        //         // four wheels
        //         Wheel::new(pos2(35., -12.), 0.),
        //         Wheel::new(pos2(35., 12.), 0.),
        //         Wheel::new(pos2(-35., 12.), 0.),
        //         Wheel::new(pos2(-35., -12.), 0.),
        //     ],

        //     // four wheels
        //     /* up_wheels = */ vec![2, 3],
        //     /* rotated_wheels = */ vec![0, 1],
        // )
    }

    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut result: TemplateApp = Default::default();
        if let Some(storage) = cc.storage {
            let stored: TemplateApp =
                eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            result.params = stored.params;
            result.walls = stored.walls;
            result.rewards = stored.rewards;
        }
        result.reset_car();
        result
    }
}

impl TemplateApp {
    fn reset_car(&mut self) {
        self.rewards.all_elements_mut().for_each(|x| x.acquired = false);
        self.sum_reward = 0.;
        self.car = Self::get_default_cars();
        self.trajectories.clear();
        self.graphs.clear();
        self.points_count = 0;
        self.drifts.iter_mut().for_each(|x| x.drifts.clear());
    }
}

impl eframe::App for TemplateApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let keys_down = ctx.input(|i| i.keys_down.clone());

        let create_plot = |name: &str, params: &Parameters| {
            Plot::new(format!("items_demo {}", name))
                .legend(Legend::default().position(Corner::RightTop))
                .show_x(false)
                .show_y(false)
                .allow_zoom([false, false])
                .allow_scroll([false, false])
                .allow_boxed_zoom(false)
                .width(params.plot_size)
                .height(params.plot_size)
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                self.params.ui(ui);
                Frame::canvas(ui.style()).show(ui, |ui| {
                    let (response, painter) = ui.allocate_painter(
                        Vec2::new(self.params.canvas_size, self.params.canvas_size),
                        Sense::drag(),
                    );

                    let from_screen = emath::RectTransform::from_to(
                        response.rect,
                        Rect::from_center_size(
                            self.car.get_center(),
                            Vec2::new(self.params.view_size, self.params.view_size),
                        ),
                    );
                    let to_screen = from_screen.inverse();

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

                    for wall in self.walls.all_elements() {
                        painter.add(Shape::closed_line(
                            wall.get_points()
                                .into_iter()
                                .map(|p| to_screen.transform_pos(p))
                                .collect(),
                            Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                        ));
                    }

                    for reward in self.rewards.all_elements() {
                        painter.add(Shape::closed_line(
                            reward.get_points()
                                .into_iter()
                                .map(|p| to_screen.transform_pos(p))
                                .collect(),
                            Stroke::new(1.0, if reward.acquired { Color32::RED } else { Color32::GREEN }),
                        ));
                    }

                    self.car.draw_car(&painter, &to_screen);

                    for i in 0..self.params.steps_per_time {
                        let time = self.params.time / self.params.steps_per_time as f32;

                        if keys_down.contains(&egui::Key::Escape) {
                            self.reset_car();
                        }
                        if keys_down.contains(&egui::Key::Space) {
                            self.car.brake();
                        }
                        if keys_down.contains(&egui::Key::ArrowUp) {
                            self.car.move_forward(1.0);
                        }
                        if keys_down.contains(&egui::Key::ArrowDown) {
                            self.car.move_backwards(1.0);
                        }
                        if keys_down.contains(&egui::Key::ArrowLeft) {
                            self.car.turn_left(1.0, &self.params.physics, time);
                        } else if keys_down.contains(&egui::Key::ArrowRight) {
                            self.car.turn_right(1.0, &self.params.physics, time);
                        } else {
                            self.car.remove_turns(&self.params.physics, time * 5.);
                        }

                        self.car.apply_wheels_force(
                            |i, pos, value| {
                                self.drifts[i].process_point(
                                    pos.to_pos2(),
                                    value > self.params.drift_starts_at,
                                );
                            },
                            &self.params.physics,
                        );

                        for wall in self.walls.all_elements() {
                            if self.car.process_collision(wall, &self.params.physics, time) {
                                self.sum_reward -= 0.01;
                            }
                        }

                        if i == 0 {
                            self.car.draw_forces(&painter, &self.params, &to_screen);

                            let values = self.car.get_internal_values();

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
                        }

                        self.car.step(time);
                    }

                    for reward in self.rewards.all_elements_mut() {
                        self.sum_reward += reward.process_pos(self.car.get_center());
                    }
                });
            });

            /*
            for graphs in self.graphs.points.values() {
                ui.horizontal_wrapped(|ui| {
                    for (name, (points, color)) in graphs {
                        ui.allocate_ui(
                            vec2(self.params.plot_size + 5., self.params.plot_size + 5.),
                            |ui| {
                                create_plot(name.to_string(), &self.params).show(ui, |plot_ui| {
                                    let line = Line::new(PlotPoints::from_iter(points.clone()))
                                        .fill(0.)
                                        .color(*color);
                                    plot_ui.line(line.name(name));
                                });
                            },
                        );
                    }
                });
            }
            */

            ui.label(format!("Reward: {}", self.sum_reward));
            ui.horizontal(|ui| {
                drop(self.walls.egui(ui, &mut (), "Walls"));
                drop(self.rewards.egui(ui, &mut (), "Rewards"));
            });

            if keys_down.contains(&egui::Key::Escape) {
                println!();
                for wall in self.walls.all_elements() {
                    println!("Wall::new(pos2({:.1}, {:.1}), vec2({:.1}, {:.1}), {:.1}),", wall.center.x, wall.center.y, wall.size.x, wall.size.y, wall.angle);
                }
                println!("----");
                for reward in self.rewards.all_elements() {
                    println!("Reward::new(pos2({:.1}, {:.1}), {:.1}),", reward.center.x, reward.center.y, reward.size);
                }
                println!();
                
            }
        });
        });

        self.trajectories.push_back(self.car.get_center());
        self.points_count += 1;
        if self.points_count > self.params.graph_points_size_limit {
            self.trajectories.pop_front();
            self.points_count -= 1;
        }

        ctx.request_repaint();
    }
}
