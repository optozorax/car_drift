use egui::containers::Frame;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::DragValue;
use egui::Shape;
use egui::Slider;
use egui::Stroke;
use egui::Ui;
use egui::{emath, Pos2, Rect, Sense, Vec2};
use egui_plot::Corner;
use egui_plot::Legend;
use egui_plot::Line;
use egui_plot::Plot;
use egui_plot::PlotPoints;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::f32::consts::PI;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
#[serde(default)]
struct Parameters {
    traction_coefficient: f32,
    gravity: f32,
    friction_coefficient: f32,
    full_force_on_speed: f32,
    acceleration_ratio: f32,
    rolling_resistance_coefficient: f32,
    drift_starts_at: f32,
    time: f32,
    wheel_turn_per_time: f32,
    angle_limit: f32,
    draw_scale: f32,
    force_draw_multiplier: f32,
    plot_size: f32,
    canvas_size: f32,
    steps_per_time: usize,
    graph_points_size_limit: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            traction_coefficient: 0.7, // 0.7 for drifting, 0.9 for regular race
            gravity: 9.8,
            friction_coefficient: 0.5, // https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
            full_force_on_speed: 30.,
            acceleration_ratio: 0.3,
            rolling_resistance_coefficient: 0.006, // car tire, https://en.wikipedia.org/wiki/Rolling_resistance
            drift_starts_at: 0.1,
            time: 0.5,
            wheel_turn_per_time: 0.05,
            angle_limit: std::f32::consts::PI * 0.2,
            draw_scale: 0.4,
            force_draw_multiplier: 4.4,
            plot_size: 170.,
            canvas_size: 1000.,
            steps_per_time: 3,
            graph_points_size_limit: 1000,
        }
    }
}

impl Parameters {
    fn ui(&mut self, ui: &mut Ui) {
        egui::Grid::new("my_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label("Traction:");
                egui_0_1(ui, &mut self.traction_coefficient);
                ui.end_row();

                ui.label("Gravity:");
                egui_f32_positive(ui, &mut self.gravity);
                ui.end_row();

                ui.label("Friction coef:");
                egui_0_1(ui, &mut self.friction_coefficient);
                ui.end_row();

                ui.label("Full force on speed:");
                egui_f32_positive(ui, &mut self.full_force_on_speed);
                ui.end_row();

                ui.label("Acceleration ratio:");
                egui_0_1(ui, &mut self.acceleration_ratio);
                ui.end_row();

                ui.label("Rolling resistance:");
                ui.add(
                    DragValue::new(&mut self.rolling_resistance_coefficient)
                        .speed(0.001)
                        .range(0.0..=0.1)
                        .min_decimals(0)
                        .max_decimals(3),
                );
                ui.end_row();

                ui.label("Drift at:");
                ui.add(
                    DragValue::new(&mut self.drift_starts_at)
                        .speed(0.001)
                        .range(0.0..=0.3)
                        .min_decimals(0)
                        .max_decimals(3),
                );
                ui.end_row();

                ui.label("Time step:");
                egui_0_1(ui, &mut self.time);
                ui.end_row();

                ui.label("Turn speed:");
                ui.add(
                    DragValue::new(&mut self.wheel_turn_per_time)
                        .speed(0.01)
                        .range(0.0..=0.3)
                        .min_decimals(0)
                        .max_decimals(2),
                );
                ui.end_row();

                ui.label("Wheel limit:");
                egui_angle(ui, &mut self.angle_limit);
                ui.end_row();

                ui.label("Draw scale:");
                egui_0_1(ui, &mut self.draw_scale);
                ui.end_row();

                ui.label("Force draw mul:");
                egui_f32_positive(ui, &mut self.force_draw_multiplier);
                ui.end_row();

                ui.label("Plot size:");
                egui_f32_positive(ui, &mut self.plot_size);
                ui.end_row();

                ui.label("Canvas size:");
                egui_f32_positive(ui, &mut self.canvas_size);
                ui.end_row();

                ui.label("Steps per time:");
                egui_usize(ui, &mut self.steps_per_time);
                ui.end_row();

                ui.label("Graph points size:");
                egui_usize(ui, &mut self.graph_points_size_limit);
                ui.end_row();
            });
    }
}

fn egui_angle(ui: &mut Ui, angle: &mut f32) {
    let mut current = rad2deg(*angle);
    let previous = current;
    ui.add(
        DragValue::from_get_set(|v| {
            if let Some(v) = v {
                if v > 360. {
                    current = (v % 360.) as f32;
                } else if v < 0. {
                    current = (360. + (v % 360.)) as f32;
                } else {
                    current = v as f32;
                }
            }
            current.into()
        })
        .speed(1)
        .suffix("°"),
    );
    if (previous - current).abs() > 1e-6 {
        *angle = deg2rad(current);
    }
}

fn egui_0_1(ui: &mut Ui, value: &mut f32) {
    ui.add(
        Slider::new(value, 0.0..=1.0)
            .clamp_to_range(true)
            .min_decimals(0)
            .max_decimals(2),
    );
}

fn egui_f32_positive(ui: &mut Ui, value: &mut f32) {
    ui.add(
        DragValue::new(value)
            .speed(0.1)
            .range(0.0..=10000.)
            .min_decimals(0)
            .max_decimals(1),
    );
}

fn egui_usize(ui: &mut Ui, value: &mut usize) {
    ui.add(DragValue::new(value));
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
enum WheelAction {
    Nothing,
    AccelerationForward(f32),
    AccelerationBackward(f32),
    Braking(f32),
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
struct Wheel {
    pos: Pos2,
    angle: f32,
    action: WheelAction,
}

impl Wheel {
    fn get_rot1(&self) -> Pos2 {
        rotate_around_origin(pos2(1., 0.), self.angle)
    }
    fn get_rot2(&self) -> Pos2 {
        rotate_around_origin(pos2(-1., 0.), self.angle)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct Car {
    center: Pos2,
    size: Vec2,
    angle: f32,

    mass: f32,
    moment_of_inertia: f32,

    speed: Vec2,
    angle_speed: f32,

    acceleration: Vec2,
    angle_acceleration: f32,

    prev_acceleration: Vec2,
    prev_speed: Vec2,
    prev_angle_acceleration: f32,
    prev_angle_speed: f32,

    wheels: Vec<Wheel>,

    forces: Vec<(Pos2, Vec2)>,
}

fn rotate_around_origin(pos: Pos2, angle: f32) -> Pos2 {
    Pos2::new(
        pos.x * angle.cos() - pos.y * angle.sin(),
        pos.x * angle.sin() + pos.y * angle.cos(),
    )
}

fn deg2rad(deg: f32) -> f32 {
    deg / 180. * PI
}

fn rad2deg(rad: f32) -> f32 {
    rad * 180. / PI
}

fn dot(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.x + a.y * b.y
}

fn inv_len(a: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let res = a.length();
    if res == 0. {
        0.
    } else {
        1. / res
    }
}

fn angle(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let b = b.into();
    let res = (dot(a, b) * inv_len(a) * inv_len(b)).acos();
    if res.is_nan() {
        0.
    } else {
        res
    }
}

fn cross(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.y - a.y * b.x
}

// project a into b
fn proj(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> Vec2 {
    let a = a.into();
    let b = b.into();
    let dot_b = dot(b, b);
    if dot_b != 0. {
        dot(a, b) / dot_b * b
    } else {
        b
    }
}

impl Default for Car {
    fn default() -> Self {
        let mut result = Car {
            center: pos2(250., 250.),
            size: vec2(100., 50.),
            angle: 0.,
            mass: 100.,
            speed: vec2(0., 0.),
            angle_speed: 0.,
            wheels: vec![
                // four wheels
                // Wheel {
                //     pos: pos2(35., -12.),
                //     angle: 0.,
                //     action: WheelAction::Nothing,
                // },
                // Wheel {
                //     pos: pos2(35., 12.),
                //     angle: 0.,
                //     action: WheelAction::Nothing,
                // },
                // Wheel {
                //     pos: pos2(-35., -12.),
                //     angle: 0.,
                //     action: WheelAction::Nothing,
                // },
                // Wheel {
                //     pos: pos2(-35., 12.),
                //     angle: 0.,
                //     action: WheelAction::Nothing,
                // },

                // two wheels
                Wheel {
                    pos: pos2(35., 0.),
                    angle: 0.,
                    action: WheelAction::Nothing,
                },
                Wheel {
                    pos: pos2(-35., 0.),
                    angle: 0.,
                    action: WheelAction::Nothing,
                },
            ],

            moment_of_inertia: 1.,
            acceleration: vec2(0., 0.),
            angle_acceleration: 0.,
            prev_acceleration: vec2(0., 0.),
            prev_speed: vec2(0., 0.),
            prev_angle_acceleration: 0.,
            prev_angle_speed: 0.,
            forces: vec![],
        };
        result.recalc_moment_of_inertia();
        result
    }
}

impl Car {
    fn step(&mut self, dt: f32) {
        // Two-step Adams–Bashforth
        self.speed += (1.5 * self.acceleration - 0.5 * self.prev_acceleration) * dt;
        self.center += (1.5 * self.speed - 0.5 * self.prev_speed) * dt;
        self.angle_speed +=
            (1.5 * self.angle_acceleration - 0.5 * self.prev_angle_acceleration) * dt;
        self.angle += (1.5 * self.angle_speed - 0.5 * self.prev_angle_speed) * dt;

        // regular euler
        // self.speed += self.acceleration * dt;
        // self.center += self.speed * dt;
        // self.angle_speed += self.angle_acceleration * dt;
        // self.angle += self.angle_speed * dt;

        if self.angle > std::f32::consts::TAU {
            self.angle -= std::f32::consts::TAU;
        }
        if self.angle < -std::f32::consts::TAU {
            self.angle += std::f32::consts::TAU;
        }

        self.prev_acceleration = self.acceleration;
        self.prev_speed = self.speed;
        self.prev_angle_acceleration = self.angle_acceleration;
        self.prev_angle_speed = self.angle_speed;

        // these force act only for one moment
        self.acceleration = vec2(0., 0.);
        self.angle_acceleration = 0.;

        self.forces.clear();
    }

    fn recalc_moment_of_inertia(&mut self) {
        self.moment_of_inertia =
            self.mass * (self.size.x * self.size.x + self.size.y * self.size.y) / 12.;
    }

    // we know for certain that point is inside
    fn apply_force(&mut self, pos: Pos2, dir: Vec2) {
        let r = pos - self.center;
        let torque = cross(r, dir);
        let dw_dt = torque / self.moment_of_inertia;
        let dx_dt = dir / self.mass;
        self.acceleration += dx_dt;
        self.angle_acceleration += dw_dt;

        self.forces.push((pos, dir));
    }

    fn to_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin((pos - self.center).to_pos2(), -self.angle)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin(pos, self.angle) + self.center.to_vec2()
    }

    fn is_inside(&self, mut pos: Pos2) -> bool {
        pos = self.to_local_coordinates(pos);
        pos.x.abs() < self.size.x / 2. && pos.y.abs() < self.size.y / 2.
    }

    fn get_points(&self) -> Vec<Pos2> {
        vec![
            pos2(-self.size.x, -self.size.y),
            pos2(self.size.x, -self.size.y),
            pos2(self.size.x, self.size.y),
            pos2(-self.size.x, self.size.y),
        ]
        .into_iter()
        .map(|p| self.from_local_coordinates(pos2(p.x / 2., p.y / 2.)))
        .collect()
    }

    fn get_wheels(&self) -> Vec<(Pos2, Pos2)> {
        let draw_size = 10.;
        self.wheels
            .iter()
            .map(|wheel| {
                (
                    self.from_local_coordinates(wheel.pos + wheel.get_rot1().to_vec2() * draw_size),
                    self.from_local_coordinates(wheel.pos + wheel.get_rot2().to_vec2() * draw_size),
                )
            })
            .collect()
    }

    fn speed_at_wheel(&self, wheel: &Wheel) -> Vec2 {
        let pos = self.wheel_pos(wheel);
        let r = pos - self.center;
        let rotation_speed = vec2(-r.y, r.x) * self.angle_speed;
        rotation_speed + self.speed
    }

    // length always equal to 1
    fn wheel_dir1(&self, wheel: &Wheel) -> Vec2 {
        let pos = self.wheel_pos(wheel);
        self.from_local_coordinates(wheel.pos + wheel.get_rot1().to_vec2()) - pos
    }

    fn wheel_pos(&self, wheel: &Wheel) -> Pos2 {
        self.from_local_coordinates(wheel.pos)
    }

    fn apply_wheels_force(
        &mut self,
        graphs: &mut Graphs,
        dbg: bool,
        drifts: &mut [Drifts],
        params: &Parameters,
        add_to_graphs: bool,
    ) {
        // about traction: https://www.engineeringtoolbox.com/tractive-effort-d_1783.html
        let traction_force = params.traction_coefficient * self.mass * params.gravity;
        let traction_per_wheel = traction_force / self.wheels.len() as f32;
        for (i, wheel) in self.wheels.clone().into_iter().enumerate() {
            let speed_at_wheel = self.speed_at_wheel(&wheel);
            let speed_coefficient =
                (speed_at_wheel.length() / params.full_force_on_speed).clamp(0., 1.);
            let wheel_dir = self.wheel_dir1(&wheel);
            let wheel_speed_parallel_dir =
                proj(speed_at_wheel, wheel_dir) * inv_len(speed_at_wheel);
            let wheel_speed_perpendicular_dir =
                (proj(speed_at_wheel, wheel_dir) - speed_at_wheel) * inv_len(speed_at_wheel);
            let mut slip_angle = rad2deg(angle(speed_at_wheel, wheel_dir));

            // about pajecja magic formula: http://www.racer.nl/reference/pacejka.htm (I don't use it here)
            if slip_angle > 90. {
                // normalize angle for side friction force
                slip_angle = 180. - slip_angle;
            }
            let slip_coefficient = if slip_angle < 5. {
                slip_angle / 5.
            } else {
                1. / slip_angle + 0.8
            }; // emulate racing car tires that is optimal at 5 degrees
            let corner_force = wheel_speed_perpendicular_dir * slip_coefficient * speed_coefficient;
            let side_friction_ratio = slip_angle / 90.;
            let side_friction_force = -speed_at_wheel.normalized()
                * side_friction_ratio
                * params.friction_coefficient
                * speed_coefficient;
            let rolling_resistance = -speed_at_wheel.normalized()
                * params.rolling_resistance_coefficient
                * (1. - side_friction_ratio)
                * speed_coefficient;

            let braking_or_acceleration_force = match wheel.action {
                WheelAction::Nothing => vec2(0., 0.),
                WheelAction::AccelerationForward(ratio) => {
                    wheel_dir * params.acceleration_ratio * ratio
                }
                WheelAction::AccelerationBackward(ratio) => {
                    -wheel_dir * params.acceleration_ratio * ratio
                }
                WheelAction::Braking(ratio) => {
                    -wheel_speed_parallel_dir
                        * params.friction_coefficient
                        * speed_coefficient.sqrt()
                        * ratio
                }
            };

            let mut total_force = corner_force
                + side_friction_force
                + rolling_resistance
                + braking_or_acceleration_force;
            if total_force.length() > 1. {
                total_force = total_force.normalized();
            }

            drifts[i].process_point(
                self.wheel_pos(&wheel),
                side_friction_force.length() > params.drift_starts_at,
            );

            if add_to_graphs {
                graphs.add_point(
                    format!("w{i}"),
                    format!("wheel {i} brake or accel"),
                    braking_or_acceleration_force.length(),
                    Color32::DARK_RED,
                    params,
                );
                graphs.add_point(
                    format!("w{i}"),
                    format!("wheel {i} corner force"),
                    corner_force.length(),
                    Color32::RED,
                    params,
                );
                graphs.add_point(
                    format!("w{i}"),
                    format!("wheel {i} side friction force"),
                    side_friction_force.length(),
                    Color32::BLUE,
                    params,
                );
                graphs.add_point(
                    format!("w{i}"),
                    format!("wheel {i} rolling resistance"),
                    side_friction_force.length(),
                    Color32::DARK_BLUE,
                    params,
                );
                graphs.add_point(
                    format!("w{i}"),
                    format!("wheel {i} total force"),
                    total_force.length(),
                    Color32::GREEN,
                    params,
                );
            }

            if dbg {
                dbg!(speed_at_wheel);
                dbg!(speed_at_wheel.length());
                dbg!(speed_coefficient);
                dbg!(wheel_dir);
                dbg!(wheel_dir.length());
                dbg!(wheel_speed_parallel_dir);
                dbg!(wheel_speed_parallel_dir.length());
                dbg!(wheel_speed_perpendicular_dir);
                dbg!(wheel_speed_perpendicular_dir.length());
                dbg!(slip_angle);
                dbg!(slip_coefficient);
                dbg!(corner_force);
                dbg!(corner_force.length());
                dbg!(side_friction_ratio);
                dbg!(side_friction_force);
                dbg!(side_friction_force.length());
                dbg!(rolling_resistance);
                dbg!(rolling_resistance.length());
                dbg!(braking_or_acceleration_force);
                dbg!(braking_or_acceleration_force.length());
                dbg!(total_force);
                dbg!(total_force.length());
                dbg!("-------------------------------------------------------");
            }

            total_force *= traction_per_wheel;

            self.apply_force(self.wheel_pos(&wheel), total_force);

            /*
               * потом сделать чтобы колесо крутилось, чтобы нельзя было ехать назад сразу после того как едешь вперёд
               * начать делать дорогу
               * затем базовую эволюцию и нейронку

            */
        }
    }
}

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
    up_wheels: Vec<usize>,
    rotated_wheels: Vec<usize>,

    // in car local coordinates
    drag_start: Option<Pos2>,
    trajectories: VecDeque<VecDeque<Pos2>>,
    points_count: usize,

    graphs: Graphs,
    drifts: Vec<Drifts>,

    params: Parameters,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            car: Default::default(),

            // four wheels
            // up_wheels: vec![2, 3],
            // rotated_wheels: vec![0, 1],

            // two wheels
            up_wheels: vec![1],
            rotated_wheels: vec![0],

            drag_start: None,
            trajectories: Default::default(),
            points_count: 0,
            graphs: Default::default(),
            drifts: vec![Default::default(); Car::default().wheels.len()],

            params: Default::default(),
        }
    }
}

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut result: TemplateApp = Default::default();
        if let Some(storage) = cc.storage {
            let stored: TemplateApp =
                eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            result.params = stored.params;
        }
        result.trajectories.push_back(Default::default());
        result.reset_car();
        result
    }
}

impl TemplateApp {
    fn reset_car(&mut self) {
        self.car.speed = vec2(0., 0.);
        self.car.angle_speed = 0.;
        self.car.angle = 0.;
        self.car.center = pos2(100., 600.);
        self.trajectories.clear();
        self.trajectories.push_back(Default::default());
        self.graphs.clear();
        self.points_count = 0;
        for pos in &self.rotated_wheels {
            self.car.wheels[*pos].angle = 0.;
        }
        self.drifts.iter_mut().for_each(|x| x.drifts.clear());
    }
}

impl eframe::App for TemplateApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut drag_pos = None;

        let keys_down = ctx.input(|i| i.keys_down.clone());

        let create_plot = |name, params: &Parameters| {
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
            ui.horizontal(|ui| {
                self.params.ui(ui);
                if self.graphs.points.contains_key("g") {
                    ui.vertical(|ui| {
                        for (name, (points, color)) in &self.graphs.points["g"] {
                            create_plot(name.to_owned(), &self.params).show(ui, |plot_ui| {
                                let line = Line::new(PlotPoints::from_iter(points.clone()))
                                    .fill(0.)
                                    .color(*color);
                                plot_ui.line(line.name(name));
                            });
                        }
                    });
                }
                Frame::canvas(ui.style()).show(ui, |ui| {
                    let (response, painter) = ui.allocate_painter(
                        Vec2::new(self.params.canvas_size, self.params.canvas_size)
                            * self.params.draw_scale,
                        Sense::drag(),
                    );

                    let from_screen = emath::RectTransform::from_to(
                        (response.rect.translate(-response.rect.min.to_vec2())
                            * self.params.draw_scale)
                            .translate(response.rect.min.to_vec2()),
                        Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                    );
                    let to_screen = from_screen.inverse();

                    if response.dragged() {
                        if let Some(mut pos) = response.interact_pointer_pos() {
                            drag_pos = Some(from_screen.transform_pos(pos));
                            pos = from_screen.transform_pos(pos);
                            if self.car.is_inside(pos) && self.drag_start.is_none() {
                                self.drag_start = Some(self.car.to_local_coordinates(pos));
                            }
                        }
                    } else {
                        self.drag_start = None;
                    }

                    for trajectory in &self.trajectories {
                        painter.add(Shape::line(
                            trajectory
                                .iter()
                                .map(|p| to_screen.transform_pos(*p))
                                .collect(),
                            Stroke::new(1.0, Color32::from_rgb(200, 200, 200)),
                        ));
                    }

                    for wheel_drifts in &self.drifts {
                        for drift in &wheel_drifts.drifts {
                            painter.add(Shape::line(
                                drift.iter().map(|p| to_screen.transform_pos(*p)).collect(),
                                Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
                            ));
                        }
                    }

                    painter.add(Shape::closed_line(
                        self.car
                            .get_points()
                            .into_iter()
                            .map(|p| to_screen.transform_pos(p))
                            .collect(),
                        Stroke::new(2.0, Color32::from_rgb(0, 128, 128)),
                    ));
                    painter.extend(
                        self.car
                            .get_wheels()
                            .into_iter()
                            .map(|p| Shape::LineSegment {
                                points: [
                                    to_screen.transform_pos(p.0),
                                    to_screen.transform_pos(p.1),
                                ],
                                stroke: Stroke::new(2.0, Color32::from_rgb(128, 0, 128)).into(),
                            }),
                    );

                    for i in 0..self.params.steps_per_time {
                        let time = self.params.time / self.params.steps_per_time as f32;

                        if keys_down.contains(&egui::Key::Escape) {
                            self.reset_car();
                        }

                        if keys_down.contains(&egui::Key::Space) {
                            for pos in &self.up_wheels {
                                self.car.wheels[*pos].action = WheelAction::Braking(1.);
                            }
                        }

                        if keys_down.contains(&egui::Key::ArrowUp) {
                            for pos in &self.up_wheels {
                                self.car.wheels[*pos].action = WheelAction::AccelerationForward(1.);
                            }
                        }

                        if keys_down.contains(&egui::Key::ArrowDown) {
                            for pos in &self.up_wheels {
                                self.car.wheels[*pos].action =
                                    WheelAction::AccelerationBackward(1.);
                            }
                        }

                        if keys_down.contains(&egui::Key::ArrowLeft) {
                            for pos in &self.rotated_wheels {
                                self.car.wheels[*pos].angle -=
                                    self.params.wheel_turn_per_time * time;
                                if self.car.wheels[*pos].angle < -self.params.angle_limit {
                                    self.car.wheels[*pos].angle = -self.params.angle_limit;
                                }
                            }
                        } else if keys_down.contains(&egui::Key::ArrowRight) {
                            for pos in &self.rotated_wheels {
                                self.car.wheels[*pos].angle +=
                                    self.params.wheel_turn_per_time * time;
                                if self.car.wheels[*pos].angle > self.params.angle_limit {
                                    self.car.wheels[*pos].angle = self.params.angle_limit;
                                }
                            }
                        } else {
                            for pos in &self.rotated_wheels {
                                let angle = &mut self.car.wheels[*pos].angle;
                                let change = self.params.wheel_turn_per_time * time;
                                if angle.abs() > change {
                                    if *angle > 0. {
                                        *angle -= change;
                                    } else {
                                        *angle += change;
                                    }
                                } else {
                                    *angle *= 0.1;
                                }
                            }
                        }

                        if let Some((drag_start, drag_pos)) = self.drag_start.zip(drag_pos) {
                            let start_converted = self.car.from_local_coordinates(drag_start);
                            self.car
                                .apply_force(start_converted, drag_pos - start_converted);
                        }
                        self.car.apply_wheels_force(
                            &mut self.graphs,
                            keys_down.contains(&egui::Key::Backspace),
                            &mut self.drifts,
                            &self.params,
                            i == 0,
                        );
                        if i == 0 {
                            for (pos, dir) in &self.car.forces {
                                painter.add(Shape::LineSegment {
                                    points: [
                                        to_screen.transform_pos(*pos),
                                        to_screen.transform_pos(
                                            *pos + *dir
                                                * self.params.time
                                                * self.params.force_draw_multiplier,
                                        ),
                                    ],
                                    stroke: Stroke::new(1.0, Color32::from_rgb(0, 0, 0)).into(),
                                });
                            }

                            self.graphs.add_point(
                                "g",
                                "acceleration",
                                self.car.acceleration.length(),
                                Color32::GRAY,
                                &self.params,
                            );

                            self.graphs.add_point(
                                "g",
                                "torque",
                                self.car.angle_acceleration,
                                Color32::BROWN,
                                &self.params,
                            );
                        }
                        let mut prev_car = self.car.clone();
                        self.car.step(time);
                        if self.car.center.x.is_nan() {
                            dbg!(&self.car);
                            dbg!(&prev_car);
                            prev_car.apply_wheels_force(
                                &mut self.graphs,
                                true,
                                &mut self.drifts,
                                &self.params,
                                i == 0,
                            );
                            panic!();
                        }
                        for wheel in &mut self.car.wheels {
                            wheel.action = WheelAction::Nothing;
                        }
                    }
                });
            });

            for (group, graphs) in &self.graphs.points {
                if group == "g" {
                    continue;
                }
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
        });

        self.graphs.add_point(
            "g",
            "speed",
            self.car.speed.length(),
            Color32::BLACK,
            &self.params,
        );

        self.trajectories
            .back_mut()
            .unwrap()
            .push_back(self.car.center);
        self.points_count += 1;
        if self.points_count > self.params.graph_points_size_limit {
            self.trajectories.front_mut().unwrap().pop_front();
            if self.trajectories.front().unwrap().is_empty() {
                self.trajectories.pop_front();
            }
            self.points_count -= 1;
        }

        let mut teleported = false;
        if self.car.center.x > self.params.canvas_size {
            self.car.center.x -= self.params.canvas_size;
            teleported = true;
        }
        if self.car.center.y > self.params.canvas_size {
            self.car.center.y -= self.params.canvas_size;
            teleported = true;
        }
        if self.car.center.x < 0. {
            self.car.center.x += self.params.canvas_size;
            teleported = true;
        }
        if self.car.center.y < 0. {
            self.car.center.y += self.params.canvas_size;
            teleported = true;
        }
        if teleported {
            self.trajectories.push_back(Default::default());
            self.drifts
                .iter_mut()
                .for_each(|x| x.drift_recording = false);
        }

        ctx.request_repaint();
    }
}
