use egui::Pos2;
use egui::Vec2;
use std::f32::consts::PI;

#[inline(always)]
pub fn rotate_around_origin(pos: Pos2, angle: f32) -> Pos2 {
    Pos2::new(
        pos.x * angle.cos() - pos.y * angle.sin(),
        pos.x * angle.sin() + pos.y * angle.cos(),
    )
}

#[inline(always)]
pub fn rotate_around_origin_optimized(pos: Pos2, angle_sin: f32, angle_cos: f32) -> Pos2 {
    Pos2::new(
        pos.x * angle_cos - pos.y * angle_sin,
        pos.x * angle_sin + pos.y * angle_cos,
    )
}

#[inline(always)]
pub fn deg2rad(deg: f32) -> f32 {
    deg / 180. * PI
}

#[inline(always)]
pub fn rad2deg(rad: f32) -> f32 {
    rad * 180. / PI
}

#[inline(always)]
pub fn dot(a: impl Into<Vec2> + std::marker::Copy, b: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.x + a.y * b.y
}

#[inline(always)]
pub fn inv_len(a: impl Into<Vec2> + std::marker::Copy) -> f32 {
    let a = a.into();
    let res = a.length();
    if res == 0. {
        0.
    } else {
        let res = 1. / res;
        if res.is_infinite() {
            0.
        } else {
            res
        }
    }
}

#[inline(always)]
pub fn project_to_segment(point: Pos2, segment_start: Pos2, segment_end: Pos2) -> (Pos2, f32) {
    let segment_vec = segment_end - segment_start;
    let point_vec = point - segment_start;

    let t = dot(point_vec, segment_vec) / dot(segment_vec, segment_vec);

    if t <= 0.0 {
        (segment_start, 0.)
    } else if t >= 1.0 {
        (segment_end, 1.)
    } else {
        (segment_start + segment_vec * t, t)
    }
}

#[inline(always)]
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

#[inline(always)]
pub fn cross(
    a: impl Into<Vec2> + std::marker::Copy,
    b: impl Into<Vec2> + std::marker::Copy,
) -> f32 {
    let a = a.into();
    let b = b.into();
    a.x * b.y - a.y * b.x
}

// project a into b
#[inline(always)]
pub fn proj(
    a: impl Into<Vec2> + std::marker::Copy,
    b: impl Into<Vec2> + std::marker::Copy,
) -> Vec2 {
    let a = a.into();
    let b = b.into();
    let dot_b = dot(b, b);
    if dot_b != 0. {
        let res = dot(a, b) / dot_b * b;
        if res.x.is_infinite() || res.y.is_infinite() || res.x.is_nan() || res.y.is_nan() {
            b
        } else {
            res
        }
    } else {
        b
    }
}
