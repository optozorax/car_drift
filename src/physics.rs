use crate::common::*;
use crate::math::*;
use crate::nn::sqrt_sigmoid;
use egui::emath::RectTransform;
use egui::Color32;
use egui::DragValue;
use egui::Painter;
use egui::Shape;
use egui::Stroke;
use egui::Ui;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
#[serde(default)]
pub struct PhysicsParameters {
    pub traction_coefficient: fxx,
    pub gravity: fxx,
    pub friction_coefficient: fxx,
    pub full_force_on_speed: fxx,
    pub acceleration_ratio: fxx,
    pub rolling_resistance_coefficient: fxx,
    pub wheel_turn_per_time: fxx,
    pub angle_limit: fxx,
    pub wall_force: fxx,
    pub max_speed: fxx,
    pub steps_per_time: usize,
    pub time: fxx,
    pub simple_physics_ratio: fxx, // 0 - hard, 1 - simple
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            traction_coefficient: 0.9, // 0.7 for drifting, 0.9 for regular race
            gravity: 9.8,
            friction_coefficient: 0.5, // https://www.engineeringtoolbox.com/friction-coefficients-d_778.html
            full_force_on_speed: 30.,
            acceleration_ratio: 0.4, // 0.6 is good for me, 0.2 is good for normal physics
            rolling_resistance_coefficient: 0.006, // car tire, https://en.wikipedia.org/wiki/Rolling_resistance
            wheel_turn_per_time: 0.1,              // 0.07 is good for me
            angle_limit: PIx * 0.2,
            wall_force: 1000.,
            max_speed: 100.,
            steps_per_time: 3,
            time: 0.5,
            simple_physics_ratio: 0.0,
        }
    }
}

impl PhysicsParameters {
    pub fn grid_ui(&mut self, ui: &mut Ui) {
        ui.label("Traction:");
        egui_0_1(ui, &mut self.traction_coefficient);
        ui.end_row();

        ui.label("Gravity:");
        egui_fxx_positive(ui, &mut self.gravity);
        ui.end_row();

        ui.label("Friction coef:");
        egui_0_1(ui, &mut self.friction_coefficient);
        ui.end_row();

        ui.label("Full force on speed:");
        egui_fxx_positive(ui, &mut self.full_force_on_speed);
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
        egui_fxx_positive(ui, &mut self.wall_force);
        ui.end_row();

        ui.label("Max speed:");
        egui_fxx_positive(ui, &mut self.max_speed);
        ui.end_row();

        ui.label("Steps per time:");
        egui_usize(ui, &mut self.steps_per_time);
        ui.end_row();

        ui.label("Time step:");
        egui_0_1(ui, &mut self.time);
        ui.end_row();

        ui.label("Simple physics:");
        egui_0_1(ui, &mut self.simple_physics_ratio);
        ui.end_row();
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        egui::Grid::new("physics params")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
                ui.separator();
                ui.end_row();
            });
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
enum WheelAction {
    Nothing,
    AccelerationForward(fxx),
    AccelerationBackward(fxx),
    Braking(fxx),
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Wheel {
    pos: Vecx2,
    angle: fxx,
    action: WheelAction,

    angle_speed: fxx,
    remove_turns: fxx,
}

impl Wheel {
    pub fn new(pos: Vecx2, angle: fxx) -> Self {
        Self {
            pos,
            angle,
            action: WheelAction::Nothing,
            angle_speed: 0.,
            remove_turns: 0.,
        }
    }

    pub fn get_rot1(&self) -> Vecx2 {
        rotate_around_origin(vecx2(1., 0.), self.angle)
    }
    pub fn get_rot2(&self) -> Vecx2 {
        rotate_around_origin(vecx2(-1., 0.), self.angle)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct Car {
    center: Vecx2,
    size: Vecx2,
    angle: fxx,

    mass: fxx,
    moment_of_inertia: fxx,

    speed: Vecx2,
    angle_speed: fxx,

    acceleration: Vecx2,
    angle_acceleration: fxx,

    prev_acceleration: Vecx2,
    prev_speed: Vecx2,
    prev_angle_acceleration: fxx,
    prev_angle_speed: fxx,

    wheels: Vec<Wheel>,

    forces: Vec<(Vecx2, Vecx2)>,

    up_wheels: Vec<usize>,
    rotated_wheels: Vec<usize>,

    cached_for_angle: fxx,
    cached_angle_sin: fxx,
    cached_angle_cos: fxx,
}

impl Default for Car {
    fn default() -> Self {
        Car::new(
            /* center = */ vecx2(250., 250.),
            /* size = */ vecx2(100., 50.) / 2.,
            /* angle = */ 0.,
            /* mass = */ 100.,
            /* speed = */ vecx2(0., 0.),
            /* angle_speed = */ 0.,
            /* wheels = */
            vec![
                // // four wheels
                // Wheel::new(vecx2(35., -12.), 0.),
                // Wheel::new(vecx2(35., 12.), 0.),
                // Wheel::new(vecx2(-35., 12.), 0.),
                // Wheel::new(vecx2(-35., -12.), 0.),

                // two wheels
                Wheel::new(vecx2(35., 0.), 0.),
                Wheel::new(vecx2(-35., 0.), 0.),
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

fn box_moment_of_inertia(size: Vecx2, mass: fxx) -> fxx {
    mass * (size.x * size.x + size.y * size.y) / 12.
}

impl Car {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: Vecx2,
        size: Vecx2,
        angle: fxx,
        mass: fxx,
        speed: Vecx2,
        angle_speed: fxx,
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
            acceleration: vecx2(0., 0.),
            angle_acceleration: 0.,
            prev_acceleration: vecx2(0., 0.),
            prev_speed: vecx2(0., 0.),
            prev_angle_acceleration: 0.,
            prev_angle_speed: 0.,
            forces: vec![],
            cached_for_angle: angle,
            cached_angle_sin: angle.sin(),
            cached_angle_cos: angle.cos(),
        }
    }

    pub fn change_position(&mut self, dangle: fxx, dpos: Vecx2, dangle_speed: fxx, dspeed: Vecx2) {
        self.angle += dangle;
        self.center += dpos;
        self.speed += dspeed;
        self.angle_speed += dangle_speed;
        self.update_cache();
    }

    pub fn update_cache(&mut self) {
        if self.cached_for_angle != self.angle {
            self.cached_for_angle = self.angle;
            self.cached_angle_cos = self.angle.cos();
            self.cached_angle_sin = self.angle.sin();
        }
    }

    pub fn step(&mut self, dt: fxx, params: &PhysicsParameters) {
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

        if self.angle > TAUx {
            self.angle -= TAUx;
        }
        if self.angle < -TAUx {
            self.angle += TAUx;
        }

        self.prev_acceleration = self.acceleration;
        self.prev_speed = self.speed;
        self.prev_angle_acceleration = self.angle_acceleration;
        self.prev_angle_speed = self.angle_speed;

        // these force act only for one moment
        self.acceleration = vecx2(0., 0.);
        self.angle_acceleration = 0.;

        self.forces.clear();

        for wheel in &mut self.wheels {
            if wheel.remove_turns != 0. {
                let change = wheel.remove_turns * dt;
                if wheel.angle.abs() > change {
                    wheel.angle += -change * wheel.angle.signum();
                } else {
                    wheel.angle = 0.;
                }
                wheel.remove_turns = 0.;
            }

            wheel.angle += wheel.angle_speed * dt;
            wheel.angle = wheel.angle.clamp(-params.angle_limit, params.angle_limit);
            wheel.angle_speed = 0.;
            wheel.action = WheelAction::Nothing;
        }

        self.update_cache();
    }

    // we know for certain that point is inside
    pub fn apply_force(&mut self, pos: Vecx2, dir: Vecx2, params: &PhysicsParameters) {
        let dir = dir * (1. - params.simple_physics_ratio);
        let r = pos - self.center;
        let torque = cross(r, dir);
        let dw_dt = torque / self.moment_of_inertia;
        let dx_dt = dir / self.mass;
        self.acceleration += dx_dt;
        self.angle_acceleration += dw_dt;

        self.forces.push((pos, dir));
    }

    #[inline(always)]
    pub fn to_local_coordinates(&self, pos: Vecx2) -> Vecx2 {
        debug_assert_eq!(self.cached_for_angle, self.angle);
        rotate_around_origin_optimized(
            pos - self.center,
            -self.cached_angle_sin,
            self.cached_angle_cos,
        )
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline(always)]
    pub fn from_local_coordinates(&self, pos: Vecx2) -> Vecx2 {
        debug_assert_eq!(self.cached_for_angle, self.angle);
        rotate_around_origin_optimized(pos, self.cached_angle_sin, self.cached_angle_cos)
            + self.center
    }

    pub fn is_inside(&self, mut pos: Vecx2) -> bool {
        pos = self.to_local_coordinates(pos);
        pos.x.abs() < self.size.x && pos.y.abs() < self.size.y
    }

    #[inline(always)]
    pub fn get_points(&self) -> [Vecx2; 4] {
        [
            self.from_local_coordinates(vecx2(-self.size.x, -self.size.y)),
            self.from_local_coordinates(vecx2(self.size.x, -self.size.y)),
            self.from_local_coordinates(vecx2(self.size.x, self.size.y)),
            self.from_local_coordinates(vecx2(-self.size.x, self.size.y)),
        ]
    }

    pub fn get_wheels(&self) -> Vec<(Vecx2, Vecx2)> {
        let draw_size = 10.;
        self.wheels
            .iter()
            .map(|wheel| {
                (
                    self.from_local_coordinates(wheel.pos + wheel.get_rot1() * draw_size),
                    self.from_local_coordinates(wheel.pos + wheel.get_rot2() * draw_size),
                )
            })
            .collect()
    }

    pub fn speed_at_point(&self, pos: Vecx2) -> Vecx2 {
        let r = pos - self.center;
        let rotation_speed = vecx2(-r.y, r.x) * self.angle_speed;
        rotation_speed + self.speed
    }

    fn speed_at_wheel(&self, wheel: &Wheel) -> Vecx2 {
        self.speed_at_point(self.wheel_pos(wheel))
    }

    // length always equal to 1
    fn wheel_dir1(&self, wheel: &Wheel) -> Vecx2 {
        let pos = self.wheel_pos(wheel);
        self.from_local_coordinates(wheel.pos + wheel.get_rot1()) - pos
    }

    fn wheel_pos(&self, wheel: &Wheel) -> Vecx2 {
        self.from_local_coordinates(wheel.pos)
    }

    pub fn apply_wheels_force(
        &mut self,
        drift_observer: &mut impl FnMut(usize, Vecx2, fxx),
        params: &PhysicsParameters,
    ) {
        // about traction: https://www.engineeringtoolbox.com/tractive-effort-d_1783.html
        let traction_force = params.traction_coefficient * self.mass * params.gravity;
        let traction_per_wheel = traction_force / self.wheels.len() as fxx;

        // optimization in order to not clone wheels array
        // DO NOT USE WHEEL ARRAY INSIDE!!!
        let mut wheels_temp: Vec<Wheel> = Default::default();
        std::mem::swap(&mut wheels_temp, &mut self.wheels);

        for (i, wheel) in wheels_temp.iter().enumerate() {
            let speed_at_wheel = self.speed_at_wheel(wheel);
            let speed_coefficient =
                (speed_at_wheel.length() / params.full_force_on_speed).clamp(0., 1.);
            let anti_speed_coefficient = if speed_at_wheel.length() < params.max_speed {
                1.
            } else {
                0.
            };
            let wheel_dir = self.wheel_dir1(wheel);
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
                WheelAction::Nothing => vecx2(0., 0.),
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

            drift_observer(i, self.wheel_pos(wheel), side_friction_force.length());

            total_force *= traction_per_wheel;

            self.apply_force(self.wheel_pos(wheel), total_force, params);
        }

        std::mem::swap(&mut wheels_temp, &mut self.wheels);
    }

    pub fn get_internal_values(&self) -> InternalCarValues {
        InternalCarValues {
            local_speed: rotate_around_origin(self.prev_speed, -self.angle),
            local_acceleration: rotate_around_origin(self.prev_acceleration, -self.angle),
            angle_acceleration: self.prev_angle_acceleration,
            angle_speed: self.prev_angle_speed,
            wheel_angle: self.wheels[self.rotated_wheels[0]].angle,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct CarInput {
    pub brake: fxx, // 0..1
    pub acceleration: fxx,
    pub remove_turn: fxx, // 0..1
    pub turn: fxx,
}

fn two_relus_to_ratio(a: fxx, b: fxx) -> fxx {
    if a > b && a > 0. {
        sqrt_sigmoid(a)
    } else if b > a && b > 0. {
        -sqrt_sigmoid(b)
    } else {
        0.
    }
}

impl CarInput {
    pub const SIZE: usize = 6;

    pub fn from_fxx(input: &[fxx]) -> CarInput {
        debug_assert_eq!(input.len(), Self::SIZE);
        CarInput {
            brake: sqrt_sigmoid(input[0]).max(0.),
            acceleration: two_relus_to_ratio(input[1], input[2]),
            remove_turn: sqrt_sigmoid(input[3]).max(0.),
            turn: two_relus_to_ratio(input[4], input[5]),
        }
    }
}

impl Car {
    pub fn process_input(&mut self, input: &CarInput, params: &PhysicsParameters) {
        if input.brake == 0. {
            self.center += (self.from_local_coordinates(vecx2(1., 0.)) - self.center)
                * 10.0
                * input.acceleration
                * params.simple_physics_ratio;
        }

        if input.remove_turn == 0. {
            self.angle -= 0.02 * input.turn * params.simple_physics_ratio;
        }

        self.remove_turns(params.simple_physics_ratio.powf(4.0), params);

        if input.brake > 0. {
            self.brake(input.brake, params);
        } else if input.acceleration > 0. {
            self.move_forward(input.acceleration.abs());
        } else if input.acceleration < 0. {
            self.move_backwards(input.acceleration.abs());
        }

        if input.remove_turn > 0. {
            self.remove_turns(input.remove_turn, params);
        } else if input.turn > 0. {
            self.turn_left(input.turn.abs(), params);
        } else if input.turn < 0. {
            self.turn_right(input.turn.abs(), params);
        }

        self.update_cache();
    }

    pub fn move_forward(&mut self, ratio: fxx) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::AccelerationForward(ratio);
        }
    }

    pub fn move_backwards(&mut self, ratio: fxx) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::AccelerationBackward(ratio);
        }
    }

    pub fn brake(&mut self, ratio: fxx, params: &PhysicsParameters) {
        for pos in &self.up_wheels {
            self.wheels[*pos].action = WheelAction::Braking(ratio);
        }
    }

    pub fn turn_left(&mut self, ratio: fxx, params: &PhysicsParameters) {
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle_speed = -params.wheel_turn_per_time * ratio;
        }
    }

    pub fn turn_right(&mut self, ratio: fxx, params: &PhysicsParameters) {
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle_speed = params.wheel_turn_per_time * ratio;
        }
    }

    pub fn remove_turns(&mut self, ratio: fxx, params: &PhysicsParameters) {
        for pos in &self.rotated_wheels {
            self.wheels[*pos].remove_turns = params.wheel_turn_per_time * 5. * ratio;
        }
    }

    pub fn reset(&mut self) {
        self.speed = vecx2(0., 0.);
        self.angle_speed = 0.;
        self.angle = 0.;
        self.center = vecx2(50., 600.);
        for pos in &self.rotated_wheels {
            self.wheels[*pos].angle = 0.;
        }
    }

    pub fn process_collision(&mut self, wall: &Wall, params: &PhysicsParameters) -> bool {
        let mut have = false;
        for point in self.get_points() {
            if wall.is_inside(point) {
                let outer_force = wall.outer_force(point);
                self.apply_force(point, outer_force * params.wall_force, params);
                self.center += outer_force * 10.0 * params.simple_physics_ratio;
                if outer_force.length() > 1.7 {
                    self.speed *= 0.1;
                    self.angle_speed *= 0.1;
                }
                have = true;
            }
        }
        have
    }

    pub fn draw_car(&mut self, painter: &Painter, to_screen: &RectTransform) {
        painter.add(Shape::closed_line(
            self.get_points()
                .into_iter()
                .map(|p| to_screen.transform_pos(p.into()))
                .collect(),
            Stroke::new(2.0, Color32::from_rgb(0, 128, 128)),
        ));
        painter.extend(self.get_wheels().into_iter().map(|p| Shape::LineSegment {
            points: [
                to_screen.transform_pos(p.0.into()),
                to_screen.transform_pos(p.1.into()),
            ],
            stroke: Stroke::new(2.0, Color32::from_rgb(128, 0, 128)).into(),
        }));
    }

    pub fn draw_forces(
        &self,
        painter: &Painter,
        params: &InterfaceParameters,
        params_phys: &PhysicsParameters,
        to_screen: &RectTransform,
    ) {
        for (pos, dir) in &self.forces {
            painter.add(Shape::LineSegment {
                points: [
                    to_screen.transform_pos(pos.into()),
                    to_screen.transform_pos(
                        (*pos + *dir * params_phys.time * params.force_draw_multiplier).into(),
                    ),
                ],
                stroke: Stroke::new(1.0, Color32::from_rgb(0, 0, 0)).into(),
            });
        }
    }

    pub fn get_center(&self) -> Vecx2 {
        self.center
    }
}

pub struct InternalCarValues {
    pub local_speed: Vecx2,
    pub local_acceleration: Vecx2,
    pub angle_acceleration: fxx,
    pub angle_speed: fxx,
    pub wheel_angle: fxx,
}

impl InternalCarValues {
    pub const SIZE: usize = 10;

    pub fn speed_angle(&self) -> fxx {
        self.local_speed.y.atan2(self.local_speed.x)
    }

    pub fn acceleration_angle(&self) -> fxx {
        self.local_acceleration.y.atan2(self.local_speed.x)
    }

    pub fn to_fxx(&self) -> [fxx; Self::SIZE] {
        [
            self.speed_angle().sin(),
            self.speed_angle().cos(),
            self.local_speed.length(),
            self.acceleration_angle().sin(),
            self.acceleration_angle().cos(),
            self.local_acceleration.length(),
            self.angle_acceleration,
            self.angle_speed,
            self.wheel_angle.sin(),
            self.wheel_angle.cos(),
        ]
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Wall {
    pub center: Vecx2,
    pub size: Vecx2,
    pub angle_sin: fxx,
    pub angle_cos: fxx,
}

impl Default for Wall {
    fn default() -> Self {
        Self::new(vecx2(500., 500.), vecx2(50., 1500.) / 2., 0.)
    }
}

pub fn walls_from_points(points: impl Iterator<Item = Vecx2> + Clone) -> Vec<Wall> {
    pairs(points)
        .map(|(a, b)| {
            let center = (a + b) / 2.;
            let dir = b - a;
            let angle = dir.y.atan2(dir.x) + TAUx / 4.;
            let size = vecx2(25., dir.length() / 2. + 30.);
            Wall::new(center, size, angle)
        })
        .collect()
}

pub fn rewards_from_points(points: impl Iterator<Item = Vecx2> + Clone) -> Vec<Reward> {
    points.map(|a| Reward::new(a)).collect()
}

// todo: сделать оптимизацию чтобы не считать синусы и косинусы постоянно
impl Wall {
    pub fn new(center: Vecx2, size: Vecx2, angle: fxx) -> Self {
        Self {
            center,
            size,
            angle_sin: angle.sin(),
            angle_cos: angle.cos(),
        }
    }

    #[inline(always)]
    pub fn to_local_coordinates(&self, pos: Vecx2) -> Vecx2 {
        rotate_around_origin_optimized((pos - self.center), -self.angle_sin, self.angle_cos)
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline(always)]
    pub fn from_local_coordinates(&self, pos: Vecx2) -> Vecx2 {
        rotate_around_origin_optimized(pos, self.angle_sin, self.angle_cos) + self.center
    }

    pub fn is_inside(&self, pos: Vecx2) -> bool {
        let pos = self.to_local_coordinates(pos);
        pos.x.abs() < self.size.x && pos.y.abs() < self.size.y
    }

    pub fn outer_force(&self, pos: Vecx2) -> Vecx2 {
        let xf = self.to_local_coordinates(pos).x / self.size.x;
        rotate_around_origin_optimized(
            (vecx2(1. + (1. - xf.abs()), 0.) * xf.signum()),
            self.angle_sin,
            self.angle_cos,
        )
    }

    #[inline(always)]
    pub fn get_points(&self) -> [Vecx2; 4] {
        [
            self.from_local_coordinates(vecx2(-self.size.x, -self.size.y)),
            self.from_local_coordinates(vecx2(self.size.x, -self.size.y)),
            self.from_local_coordinates(vecx2(self.size.x, self.size.y)),
            self.from_local_coordinates(vecx2(-self.size.x, self.size.y)),
        ]
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
        egui_fxx(ui, &mut self.size.x);
        ui.end_row();

        ui.label("Angle:");
        // egui_angle(ui, &mut self.angle);
        ui.end_row();
    }

    pub fn intersect_ray(&self, origin: Vecx2, dir_pos: Vecx2) -> Option<fxx> {
        let o = self.to_local_coordinates(origin);
        let d = self.to_local_coordinates(dir_pos) - o;
        let size = self.size;

        let mut t_min = fxx::NEG_INFINITY;
        let mut t_max = fxx::INFINITY;

        {
            if d.x.abs() > fxx::EPSILON {
                let t1 = (-size.x - o.x) / d.x;
                let t2 = (size.x - o.x) / d.x;

                let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };

                t_min = t_min.max(t_near);
                t_max = t_max.min(t_far);
            } else if o.x.abs() > size.x {
                return None;
            }
        }

        {
            if d.y.abs() > fxx::EPSILON {
                let t1 = (-size.y - o.y) / d.y;
                let t2 = (size.y - o.y) / d.y;

                let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };

                t_min = t_min.max(t_near);
                t_max = t_max.min(t_far);
            } else if o.y.abs() > size.y {
                return None;
            }
        }

        if t_min > t_max {
            None
        } else {
            match t_min.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => Some(t_min),
                Some(std::cmp::Ordering::Equal) => Some(0.0),
                _ => None,
            }
        }
    }
}

impl Wall {
    pub fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        egui::Grid::new(data_id.with("wall_grid"))
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
                ui.separator();
                ui.end_row();
            });
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Reward {
    pub center: Vecx2,
}

impl Reward {
    pub fn new(center: Vecx2) -> Self {
        Self { center }
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
    }
}

impl Default for Reward {
    fn default() -> Self {
        Self {
            center: vecx2(500., 500.),
        }
    }
}

impl Reward {
    pub fn egui(&mut self, ui: &mut Ui, data_id: egui::Id) {
        egui::Grid::new(data_id.with("wall_grid"))
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                self.grid_ui(ui);
            });
    }
}
