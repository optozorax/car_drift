use std::f32::consts::TAU;
use crate::storage::WhatChanged;
use crate::storage::GetHelper;
use crate::storage::InlineHelper;

use crate::storage::StorageElem2;
use serde::Serialize;
use serde::Deserialize;

use crate::storage::UniqueId;
use crate::storage::Wrapper;

use crate::common::*;
use crate::math::*;
use egui::emath::RectTransform;
use egui::pos2;
use egui::vec2;
use egui::Color32;
use egui::DragValue;
use egui::Painter;
use egui::Pos2;
use egui::Shape;
use egui::Stroke;
use egui::Ui;
use egui::Vec2;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
#[serde(default)]
pub struct PhysicsParameters {
    traction_coefficient: f32,
    gravity: f32,
    friction_coefficient: f32,
    full_force_on_speed: f32,
    acceleration_ratio: f32,
    rolling_resistance_coefficient: f32,
    wheel_turn_per_time: f32,
    angle_limit: f32,
    wall_force: f32,
    max_speed: f32,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            traction_coefficient: 0.7, // 0.7 for drifting, 0.9 for regular race
            gravity: 9.8,
            friction_coefficient: 0.5, // https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
            full_force_on_speed: 30.,
            acceleration_ratio: 0.3,
            rolling_resistance_coefficient: 0.006, // car tire, https://en.wikipedia.org/wiki/Rolling_resistance
            wheel_turn_per_time: 0.01,
            angle_limit: std::f32::consts::PI * 0.2,
            wall_force: 1000.,
            max_speed: 100.,
        }
    }
}

impl PhysicsParameters {
    pub fn grid_ui(&mut self, ui: &mut Ui) {
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

        ui.label("Wall force:");
        egui_f32_positive(ui, &mut self.wall_force);
        ui.end_row();

        ui.label("Max speed:");
        egui_f32_positive(ui, &mut self.max_speed);
        ui.end_row();
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        egui::Grid::new("regular params")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
            });
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
enum WheelAction {
    Nothing,
    AccelerationForward(f32),
    AccelerationBackward(f32),
    Braking(f32),
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Wheel {
    pos: Pos2,
    angle: f32,
    action: WheelAction,
}

impl Wheel {
    pub fn new(pos: Pos2, angle: f32) -> Self {
        Self {
            pos,
            angle,
            action: WheelAction::Nothing,
        }
    }

    pub fn get_rot1(&self) -> Pos2 {
        rotate_around_origin(pos2(1., 0.), self.angle)
    }
    pub fn get_rot2(&self) -> Pos2 {
        rotate_around_origin(pos2(-1., 0.), self.angle)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct Car {
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

    up_wheels: Vec<usize>,
    rotated_wheels: Vec<usize>,
}

impl Default for Car {
    fn default() -> Self {
        Car::new(
            /* center = */ pos2(250., 250.),
            /* size = */ vec2(100., 50.) / 2.,
            /* angle = */ 0.,
            /* mass = */ 100.,
            /* speed = */ vec2(0., 0.),
            /* angle_speed = */ 0.,
            /* wheels = */
            vec![
                // // four wheels
                // Wheel::new(pos2(35., -12.), 0.),
                // Wheel::new(pos2(35., 12.), 0.),
                // Wheel::new(pos2(-35., 12.), 0.),
                // Wheel::new(pos2(-35., -12.), 0.),

                // two wheels
                Wheel::new(pos2(35., 0.), 0.),
                Wheel::new(pos2(-35., 0.), 0.),
            ],
            // four wheels
            // /* up_wheels = */ vec![2, 3],
            // /* rotated_wheels = */ vec![0, 1],

            // two wheels
            /* up_wheels = */
            vec![1],
            /* rotated_wheels = */ vec![0],
        )
    }
}

fn box_moment_of_inertia(size: Vec2, mass: f32) -> f32 {
    mass * (size.x * size.x + size.y * size.y) / 12.
}

impl Car {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: Pos2,
        size: Vec2,
        angle: f32,
        mass: f32,
        speed: Vec2,
        angle_speed: f32,
        wheels: Vec<Wheel>,
        up_wheels: Vec<usize>,
        rotated_wheels: Vec<usize>,
    ) -> Self {
        Self {
            center,
            size,
            angle,
            mass,
            speed,
            angle_speed,
            wheels,
            up_wheels,
            rotated_wheels,

            moment_of_inertia: box_moment_of_inertia(size, mass),
            acceleration: vec2(0., 0.),
            angle_acceleration: 0.,
            prev_acceleration: vec2(0., 0.),
            prev_speed: vec2(0., 0.),
            prev_angle_acceleration: 0.,
            prev_angle_speed: 0.,
            forces: vec![],
        }
    }

    pub fn step(&mut self, dt: f32) {
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

        for wheel in &mut self.wheels {
            wheel.action = WheelAction::Nothing;
        }
    }

    // we know for certain that point is inside
    pub fn apply_force(&mut self, pos: Pos2, dir: Vec2) {
        let r = pos - self.center;
        let torque = cross(r, dir);
        let dw_dt = torque / self.moment_of_inertia;
        let dx_dt = dir / self.mass;
        self.acceleration += dx_dt;
        self.angle_acceleration += dw_dt;

        self.forces.push((pos, dir));
    }

    pub fn to_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin((pos - self.center).to_pos2(), -self.angle)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn from_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin(pos, self.angle) + self.center.to_vec2()
    }

    pub fn is_inside(&self, mut pos: Pos2) -> bool {
        pos = self.to_local_coordinates(pos);
        pos.x.abs() < self.size.x && pos.y.abs() < self.size.y
    }

    pub fn get_points(&self) -> Vec<Pos2> {
        vec![
            pos2(-self.size.x, -self.size.y),
            pos2(self.size.x, -self.size.y),
            pos2(self.size.x, self.size.y),
            pos2(-self.size.x, self.size.y),
        ]
        .into_iter()
        .map(|p| self.from_local_coordinates(pos2(p.x, p.y)))
        .collect()
    }

    pub fn get_wheels(&self) -> Vec<(Pos2, Pos2)> {
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

    pub fn speed_at_point(&self, pos: Pos2) -> Vec2 {
        let r = pos - self.center;
        let rotation_speed = vec2(-r.y, r.x) * self.angle_speed;
        rotation_speed + self.speed
    }

    fn speed_at_wheel(&self, wheel: &Wheel) -> Vec2 {
        self.speed_at_point(self.wheel_pos(wheel))
    }

    // length always equal to 1
    fn wheel_dir1(&self, wheel: &Wheel) -> Vec2 {
        let pos = self.wheel_pos(wheel);
        self.from_local_coordinates(wheel.pos + wheel.get_rot1().to_vec2()) - pos
    }

    fn wheel_pos(&self, wheel: &Wheel) -> Pos2 {
        self.from_local_coordinates(wheel.pos)
    }

    pub fn apply_wheels_force(
        &mut self,
        mut drift_observer: impl FnMut(usize, Vec2, f32),
        params: &PhysicsParameters,
    ) {
        // about traction: https://www.engineeringtoolbox.com/tractive-effort-d_1783.html
        let traction_force = params.traction_coefficient * self.mass * params.gravity;
        let traction_per_wheel = traction_force / self.wheels.len() as f32;
        for (i, wheel) in self.wheels.clone().into_iter().enumerate() {
            let speed_at_wheel = self.speed_at_wheel(&wheel);
            let speed_coefficient =
                (speed_at_wheel.length() / params.full_force_on_speed).clamp(0., 1.);
            let anti_speed_coefficient = if speed_at_wheel.length() < params.max_speed {
                1.
            } else {
                0.
            };
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
                    wheel_dir * params.acceleration_ratio * ratio * anti_speed_coefficient
                }
                WheelAction::AccelerationBackward(ratio) => {
                    -wheel_dir * params.acceleration_ratio * ratio * anti_speed_coefficient
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

            drift_observer(
                i,
                self.wheel_pos(&wheel).to_vec2(),
                side_friction_force.length(),
            );

            total_force *= traction_per_wheel;

            self.apply_force(self.wheel_pos(&wheel), total_force);
        }
    }

    pub fn get_internal_values(&self) -> InternalCarValues {
        InternalCarValues {
            local_speed: rotate_around_origin(self.speed.to_pos2(), self.angle).to_vec2(),
            local_acceleration: rotate_around_origin(self.acceleration.to_pos2(), self.angle)
                .to_vec2(),
            angle_acceleration: self.angle_acceleration,
            angle_speed: self.angle_speed,
            wheel_angle: self.wheels[self.rotated_wheels[0]].angle,
        }
    }
}

impl Car {
    pub fn move_forward(&mut self, ratio: f32) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::AccelerationForward(ratio);
        }
    }

    pub fn move_backwards(&mut self, ratio: f32) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::AccelerationBackward(ratio);
        }
    }

    pub fn brake(&mut self) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::Braking(1.);
        }
    }

    pub fn turn_left(&mut self, ratio: f32, params: &PhysicsParameters, time: f32) {
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle -= params.wheel_turn_per_time * time * ratio;
            if self.wheels[*pos].angle < -params.angle_limit {
                self.wheels[*pos].angle = -params.angle_limit;
            }
        }
    }

    pub fn turn_right(&mut self, ratio: f32, params: &PhysicsParameters, time: f32) {
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle += params.wheel_turn_per_time * time * ratio;
            if self.wheels[*pos].angle > params.angle_limit {
                self.wheels[*pos].angle = params.angle_limit;
            }
        }
    }

    pub fn remove_turns(&mut self, params: &PhysicsParameters, time: f32) {
        for pos in &self.rotated_wheels {
            let angle = &mut self.wheels[*pos].angle;
            let change = params.wheel_turn_per_time * time;
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

    pub fn reset(&mut self) {
        self.speed = vec2(0., 0.);
        self.angle_speed = 0.;
        self.angle = 0.;
        self.center = pos2(50., 600.);
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle = 0.;
        }
    }

    pub fn process_collision(&mut self, wall: &Wall, params: &PhysicsParameters, time: f32) {
        for point in self.get_points() {
            if wall.is_inside(point) {
                let speed_at_point = self.speed_at_point(point);
                let outer_force = wall.outer_force(point);
                let outer_speed = proj(speed_at_point, outer_force);
                let dir = dot(outer_speed, outer_force);
                if dir < 0. {
                    self.apply_force(point, outer_force * params.wall_force / time);
                }
            }
        }
    }

    pub fn draw_car(&mut self, painter: &Painter, to_screen: &RectTransform) {
        painter.add(Shape::closed_line(
            self.get_points()
                .into_iter()
                .map(|p| to_screen.transform_pos(p))
                .collect(),
            Stroke::new(2.0, Color32::from_rgb(0, 128, 128)),
        ));
        painter.extend(self.get_wheels().into_iter().map(|p| Shape::LineSegment {
            points: [to_screen.transform_pos(p.0), to_screen.transform_pos(p.1)],
            stroke: Stroke::new(2.0, Color32::from_rgb(128, 0, 128)).into(),
        }));
    }

    pub fn draw_forces(
        &mut self,
        painter: &Painter,
        params: &Parameters,
        to_screen: &RectTransform,
    ) {
        for (pos, dir) in &self.forces {
            painter.add(Shape::LineSegment {
                points: [
                    to_screen.transform_pos(*pos),
                    to_screen
                        .transform_pos(*pos + *dir * params.time * params.force_draw_multiplier),
                ],
                stroke: Stroke::new(1.0, Color32::from_rgb(0, 0, 0)).into(),
            });
        }
    }

    pub fn get_center(&self) -> Pos2 {
        self.center
    }
}

pub struct InternalCarValues {
    pub local_speed: Vec2,
    pub local_acceleration: Vec2,
    pub angle_acceleration: f32,
    pub angle_speed: f32,
    pub wheel_angle: f32,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Wall {
    pub center: Pos2,
    pub size: Vec2,
    pub angle: f32,
}

impl Default for Wall {
    fn default() -> Self {
        Self {
            center: pos2(500., 500.),
            size: vec2(50., 1500.) / 2.,
            angle: 0.,
        }
    }
}

// todo: сделать оптимизацию чтобы не считать синусы и косинусы постоянно
impl Wall {
    pub fn new(center: Pos2, size: Vec2, angle: f32) -> Self {
        Self {
            center,
            size,
            angle,
        }
    }

    pub fn to_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin((pos - self.center).to_pos2(), -self.angle)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn from_local_coordinates(&self, pos: Pos2) -> Pos2 {
        rotate_around_origin(pos, self.angle) + self.center.to_vec2()
    }

    pub fn is_inside(&self, pos: Pos2) -> bool {
        let pos = self.to_local_coordinates(pos);
        pos.x.abs() < self.size.x && pos.y.abs() < self.size.y
    }

    pub fn outer_force(&self, pos: Pos2) -> Vec2 {
        let xf = self.to_local_coordinates(pos).x / self.size.x;
        rotate_around_origin(
            (vec2(1. + (1. - xf.abs()) * 5., 0.) * xf.signum()).to_pos2(),
            self.angle,
        )
        .to_vec2()
    }

    pub fn get_points(&self) -> Vec<Pos2> {
        vec![
            pos2(-self.size.x, -self.size.y),
            pos2(self.size.x, -self.size.y),
            pos2(self.size.x, self.size.y),
            pos2(-self.size.x, self.size.y),
        ]
        .into_iter()
        .map(|p| self.from_local_coordinates(p))
        .collect()
    }

    pub fn grid_ui(&mut self, ui: &mut Ui) {
        ui.label("Pos X:");
        ui.add(
            DragValue::new(&mut self.center.x)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Pos Y:");
        ui.add(
            DragValue::new(&mut self.center.y)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Size:");
        ui.add(
            DragValue::new(&mut self.size.y)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Thickness:");
        egui_f32(ui, &mut self.size.x);
        ui.end_row();

        ui.label("Angle:");
        egui_angle(ui, &mut self.angle);
        ui.end_row();
    }
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WallId(UniqueId);

impl Wrapper for WallId {
    fn wrap(id: UniqueId) -> Self {
        Self(id)
    }
    fn un_wrap(self) -> UniqueId {
        self.0
    }
}

impl StorageElem2 for Wall {
    type IdWrapper = WallId;
    type GetType = Wall;

    const SAFE_TO_RENAME: bool = true;

    type Input = ();
    type GetInput = ();

    fn egui(
        &mut self,
        ui: &mut Ui,
        _input: &mut Self::Input,
        _inline_helper: &mut InlineHelper<'_, Self>,
        data_id: egui::Id,
        _: Self::IdWrapper,
    ) -> WhatChanged {
        egui::Grid::new(data_id.with("wall_grid"))
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
            });
        WhatChanged::default()
    }

    fn get(
        &self,
        _get_helper: &GetHelper<'_, Self>,
        (): &Self::Input,
    ) -> Option<Self::GetType> {
        Some(self.clone())
    }

    fn remove<F: FnMut(Self::IdWrapper, &mut Self::Input)>(
        &self,
        _f: F,
        _input: &mut Self::Input,
    ) {
    }

    fn errors_count<F: FnMut(Self::IdWrapper) -> usize>(
        &self,
        _f: F,
        (): &Self::Input,
        _: Self::IdWrapper,
    ) -> usize {
        0
    }
}


#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Reward {
    pub center: Pos2,
    pub size: f32,
    pub acquired: bool,
}

impl Reward {
    pub fn new(center: Pos2, size: f32) -> Self {
        Self {
            center,
            size,
            acquired: false,
        }
    }

    pub fn process_pos(&mut self, pos: Pos2) -> f32 {
        if !self.acquired {
            if (pos - self.center).length() < self.size {
                self.acquired = true;
                1.
            } else {
                0.
            }
        } else {
            0.
        }
    }

    pub fn grid_ui(&mut self, ui: &mut Ui) {
        ui.label("Pos X:");
        ui.add(
            DragValue::new(&mut self.center.x)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Pos Y:");
        ui.add(
            DragValue::new(&mut self.center.y)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Size:");
        ui.add(
            DragValue::new(&mut self.size)
                .speed(5.0)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();
    }

    pub fn get_points(&self) -> Vec<Pos2> {
        let n = 18;
        (0..n).map(|i| i as f32 / n as f32 * TAU).map(|i| pos2(i.sin(), i.cos()))
        .map(|p| p * self.size + self.center.to_vec2())
        .collect()
    }
}

impl Default for Reward {
    fn default() -> Self {
        Self {
            center: pos2(500., 500.),
            size: 500.,
            acquired: false,
        }
    }
}

impl StorageElem2 for Reward {
    type IdWrapper = WallId;
    type GetType = Reward;

    const SAFE_TO_RENAME: bool = true;

    type Input = ();
    type GetInput = ();

    fn egui(
        &mut self,
        ui: &mut Ui,
        _input: &mut Self::Input,
        _inline_helper: &mut InlineHelper<'_, Self>,
        data_id: egui::Id,
        _: Self::IdWrapper,
    ) -> WhatChanged {
        egui::Grid::new(data_id.with("wall_grid"))
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
            });
        WhatChanged::default()
    }

    fn get(
        &self,
        _get_helper: &GetHelper<'_, Self>,
        (): &Self::Input,
    ) -> Option<Self::GetType> {
        Some(self.clone())
    }

    fn remove<F: FnMut(Self::IdWrapper, &mut Self::Input)>(
        &self,
        _f: F,
        _input: &mut Self::Input,
    ) {
    }

    fn errors_count<F: FnMut(Self::IdWrapper) -> usize>(
        &self,
        _f: F,
        (): &Self::Input,
        _: Self::IdWrapper,
    ) -> usize {
        0
    }
}
