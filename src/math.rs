use std::fmt;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

pub type fxx = f32;
pub const PIx: fxx = std::f32::consts::PI;
pub const TAUx: fxx = std::f32::consts::TAU;
pub const Ex: fxx = std::f32::consts::E;

#[derive(Clone, Copy, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Vecx2 {
    pub x: fxx,
    pub y: fxx,
}

pub fn vecx2(x: fxx, y: fxx) -> Vecx2 {
    Vecx2::new(x, y)
}

#[inline(always)]
pub fn Vecx2(x: fxx, y: fxx) -> Vecx2 {
    Vecx2 { x, y }
}

impl From<[fxx; 2]> for Vecx2 {
    #[inline(always)]
    fn from(v: [fxx; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

impl From<&[fxx; 2]> for Vecx2 {
    #[inline(always)]
    fn from(v: &[fxx; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

impl From<Vecx2> for [fxx; 2] {
    #[inline(always)]
    fn from(v: Vecx2) -> Self {
        [v.x, v.y]
    }
}

impl From<&Vecx2> for [fxx; 2] {
    #[inline(always)]
    fn from(v: &Vecx2) -> Self {
        [v.x, v.y]
    }
}

// ----------------------------------------------------------------------------
// Compatibility and convenience conversions to and from (fxx, fxx):

impl From<(fxx, fxx)> for Vecx2 {
    #[inline(always)]
    fn from(v: (fxx, fxx)) -> Self {
        Self { x: v.0, y: v.1 }
    }
}

impl From<egui::Vec2> for Vecx2 {
    #[inline(always)]
    fn from(v: egui::Vec2) -> Self {
        Self {
            x: v.x as fxx,
            y: v.y as fxx,
        }
    }
}

impl From<egui::Pos2> for Vecx2 {
    #[inline(always)]
    fn from(v: egui::Pos2) -> Self {
        Self {
            x: v.x as fxx,
            y: v.y as fxx,
        }
    }
}

impl From<&(fxx, fxx)> for Vecx2 {
    #[inline(always)]
    fn from(v: &(fxx, fxx)) -> Self {
        Self { x: v.0, y: v.1 }
    }
}

impl From<Vecx2> for (fxx, fxx) {
    #[inline(always)]
    fn from(v: Vecx2) -> Self {
        (v.x, v.y)
    }
}

impl From<Vecx2> for egui::Vec2 {
    #[inline(always)]
    fn from(v: Vecx2) -> Self {
        egui::Vec2::new(v.x as f32, v.y as f32)
    }
}

impl From<Vecx2> for egui::Pos2 {
    #[inline(always)]
    fn from(v: Vecx2) -> Self {
        egui::Pos2::new(v.x as f32, v.y as f32)
    }
}

impl From<&Vecx2> for egui::Vec2 {
    #[inline(always)]
    fn from(v: &Vecx2) -> Self {
        egui::Vec2::new(v.x as f32, v.y as f32)
    }
}

impl From<&Vecx2> for egui::Pos2 {
    #[inline(always)]
    fn from(v: &Vecx2) -> Self {
        egui::Pos2::new(v.x as f32, v.y as f32)
    }
}

impl From<&Vecx2> for (fxx, fxx) {
    #[inline(always)]
    fn from(v: &Vecx2) -> Self {
        (v.x, v.y)
    }
}

impl Vecx2 {
    #[inline(always)]
    pub const fn new(x: fxx, y: fxx) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub fn length(self) -> fxx {
        self.x.hypot(self.y)
    }

    #[must_use]
    #[inline(always)]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len <= 0.0 {
            self
        } else {
            self / len
        }
    }
}

impl Eq for Vecx2 {}

impl Neg for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Vecx2(-self.x, -self.y)
    }
}

impl AddAssign for Vecx2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        };
    }
}

impl SubAssign for Vecx2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        };
    }
}

impl Add for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// Element-wise multiplication
impl Mul<Self> for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, vec: Self) -> Self {
        Self {
            x: self.x * vec.x,
            y: self.y * vec.y,
        }
    }
}

/// Element-wise division
impl Div<Self> for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}

impl MulAssign<fxx> for Vecx2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: fxx) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign<fxx> for Vecx2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: fxx) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl Mul<fxx> for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, factor: fxx) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }
}

impl Mul<Vecx2> for fxx {
    type Output = Vecx2;

    #[inline(always)]
    fn mul(self, vec: Vecx2) -> Vecx2 {
        Vecx2 {
            x: self * vec.x,
            y: self * vec.y,
        }
    }
}

impl Div<fxx> for Vecx2 {
    type Output = Self;

    #[inline(always)]
    fn div(self, factor: fxx) -> Self {
        Self {
            x: self.x / factor,
            y: self.y / factor,
        }
    }
}

impl fmt::Debug for Vecx2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(precision) = f.precision() {
            write!(f, "[{1:.0$} {2:.0$}]", precision, self.x, self.y)
        } else {
            write!(f, "[{:.1} {:.1}]", self.x, self.y)
        }
    }
}

impl fmt::Display for Vecx2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        self.x.fmt(f)?;
        f.write_str(" ")?;
        self.y.fmt(f)?;
        f.write_str("]")?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------

#[inline(always)]
pub fn rotate_around_origin(pos: Vecx2, angle: fxx) -> Vecx2 {
    Vecx2::new(
        pos.x * angle.cos() - pos.y * angle.sin(),
        pos.x * angle.sin() + pos.y * angle.cos(),
    )
}

#[inline(always)]
pub fn rotate_around_origin_optimized(pos: Vecx2, angle_sin: fxx, angle_cos: fxx) -> Vecx2 {
    Vecx2::new(
        pos.x * angle_cos - pos.y * angle_sin,
        pos.x * angle_sin + pos.y * angle_cos,
    )
}

#[inline(always)]
pub fn deg2rad(deg: fxx) -> fxx {
    deg / 180. * PIx
}

#[inline(always)]
pub fn rad2deg(rad: fxx) -> fxx {
    rad * 180. / PIx
}

#[inline(always)]
pub fn dot(
    a: impl Into<Vecx2> + std::marker::Copy,
    b: impl Into<Vecx2> + std::marker::Copy,
) -> fxx {
    let a = a.into();
    let b = b.into();
    a.x * b.x + a.y * b.y
}

#[inline(always)]
pub fn inv_len(a: impl Into<Vecx2> + std::marker::Copy) -> fxx {
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
pub fn project_to_segment(point: Vecx2, segment_start: Vecx2, segment_end: Vecx2) -> (Vecx2, fxx) {
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
    a: impl Into<Vecx2> + std::marker::Copy,
    b: impl Into<Vecx2> + std::marker::Copy,
) -> fxx {
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
    a: impl Into<Vecx2> + std::marker::Copy,
    b: impl Into<Vecx2> + std::marker::Copy,
) -> fxx {
    let a = a.into();
    let b = b.into();
    a.x * b.y - a.y * b.x
}

// project a into b
#[inline(always)]
pub fn proj(
    a: impl Into<Vecx2> + std::marker::Copy,
    b: impl Into<Vecx2> + std::marker::Copy,
) -> Vecx2 {
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
