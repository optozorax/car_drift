use crate::math::deg2rad;
use crate::math::rad2deg;

use egui::DragValue;

use egui::Ui;

use egui::Slider;

pub fn pairs<U, T: Iterator<Item = U> + Clone>(t: T) -> impl Iterator<Item = (U, U)> {
    t.clone().zip(t.skip(1))
}

pub trait AnyOrBothWith {
    type Inner;
    fn any_or_both_with<F: FnOnce(Self::Inner, Self::Inner) -> Self::Inner>(
        self,
        b: Option<Self::Inner>,
        f: F,
    ) -> Option<Self::Inner>;
}

impl<T> AnyOrBothWith for Option<T> {
    type Inner = T;
    fn any_or_both_with<F: FnOnce(Self::Inner, Self::Inner) -> Self::Inner>(
        self,
        b: Option<Self::Inner>,
        f: F,
    ) -> Option<Self::Inner> {
        match (self, b) {
            (Some(a), Some(b)) => Some((f)(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

pub fn egui_angle(ui: &mut Ui, angle: &mut f32) {
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

pub fn egui_0_1(ui: &mut Ui, value: &mut f32) {
    ui.add(
        Slider::new(value, 0.0..=1.0)
            .clamp_to_range(true)
            .min_decimals(0)
            .max_decimals(2),
    );
}

pub fn egui_f32_positive(ui: &mut Ui, value: &mut f32) {
    ui.add(
        DragValue::new(value)
            .speed(0.1)
            .range(0.0..=10000.)
            .min_decimals(0)
            .max_decimals(1),
    );
}

pub fn egui_f32(ui: &mut Ui, value: &mut f32) {
    ui.add(
        DragValue::new(value)
            .speed(0.1)
            .min_decimals(0)
            .max_decimals(1),
    );
}

pub fn egui_usize(ui: &mut Ui, value: &mut usize) {
    ui.add(DragValue::new(value));
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
#[serde(default)]
pub struct InterfaceParameters {
    pub drift_starts_at: f32,
    pub force_draw_multiplier: f32,
    pub plot_size: f32,
    pub canvas_size: f32,
    pub view_size: f32,
    pub graph_points_size_limit: usize,
}

impl Default for InterfaceParameters {
    fn default() -> Self {
        Self {
            drift_starts_at: 0.1,
            force_draw_multiplier: 4.4,
            plot_size: 170.,
            canvas_size: 1000.,
            view_size: 1500.,
            graph_points_size_limit: 1000,
        }
    }
}

impl InterfaceParameters {
    pub fn grid_ui(&mut self, ui: &mut Ui) {
        ui.label("Drift at:");
        ui.add(
            DragValue::new(&mut self.drift_starts_at)
                .speed(0.001)
                .range(0.0..=0.3)
                .min_decimals(0)
                .max_decimals(3),
        );
        ui.end_row();

        ui.label("Force draw mul:");
        egui_f32_positive(ui, &mut self.force_draw_multiplier);
        ui.end_row();

        ui.label("Plot size:");
        egui_f32_positive(ui, &mut self.plot_size);
        ui.end_row();

        ui.label("Canvas size:");
        ui.add(
            DragValue::new(&mut self.canvas_size)
                .speed(1.)
                .range(0.0..=10000.)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("View size:");
        ui.add(
            DragValue::new(&mut self.view_size)
                .speed(1.)
                .range(0.0..=10000.)
                .min_decimals(0)
                .max_decimals(0),
        );
        ui.end_row();

        ui.label("Graph points size:");
        egui_usize(ui, &mut self.graph_points_size_limit);
        ui.end_row();
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        egui::Grid::new("regular params")
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
