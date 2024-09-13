use egui::Pos2;
use egui::Vec2;
use std::f32::consts::PI;

pub fn rotate_around_origin(pos: Pos2, angle: f32) -> Pos2 {
    Pos2::new(
        pos.x * angle.cos() - pos.y * angle.sin(),
        pos.x * angle.sin() + pos.y * angle.cos(),
    )
}

pub fn deg2rad(deg: f32) -> f32 {
    deg / 180. * PI
}

pub fn rad2deg(rad: f32) -> f32 {
    rad * 180. / PI
}

pub fn dot(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.x + a.y * b.y
}

pub fn inv_len(a: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let res = a.length();
    if res == 0. {
        0.
    } else {
        1. / res
    }
}

pub fn angle(
    a: impl Into<Vec2> + std::marker::Copy,
    b: impl Into<Vec2> + std::marker::Copy,
) -> f32 {
    let a = a.into();
    let b = b.into();
    let res = (dot(a, b) * inv_len(a) * inv_len(b)).acos();
    if res.is_nan() {
        0.
    } else {
        res
    }
}

pub fn cross(
    a: impl Into<Vec2> + std::marker::Copy,
    b: impl Into<Vec2> + std::marker::Copy,
) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.y - a.y * b.x
}

// project a into b
pub fn proj(
    a: impl Into<Vec2> + std::marker::Copy,
    b: impl Into<Vec2> + std::marker::Copy,
) -> Vec2 {
    let a = a.into();
    let b = b.into();
    let dot_b = dot(b, b);
    if dot_b != 0. {
        dot(a, b) / dot_b * b
    } else {
        b
    }
}
