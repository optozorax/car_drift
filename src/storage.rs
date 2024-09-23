use egui::*;

pub fn egui_array<T: std::default::Default>(
    array: &mut Vec<T>,
    name: &str,
    ui: &mut Ui,
    egui_t: impl FnMut(&mut T, &mut Ui, egui::Id),
) {
    let data_id = ui.make_persistent_id(name);
    egui::CollapsingHeader::new(name)
        .id_source(data_id.with("header"))
        .default_open(false)
        .show(ui, |ui| {
            egui_array_inner(array, ui, data_id, egui_t, false);
        });
}

pub fn egui_array_inner<T: std::default::Default>(
    array: &mut Vec<T>,
    ui: &mut Ui,
    data_id: egui::Id,
    mut egui_t: impl FnMut(&mut T, &mut Ui, egui::Id),
    compact: bool,
) {
    let mut to_delete = None;
    let mut to_move_up = None;
    let mut to_move_down = None;

    let len = array.len();
    let mut buttons_ui = |pos, ui: &mut Ui| {
        if ui
            .add(Button::new(RichText::new("Delete").color(Color32::RED)))
            .clicked()
        {
            to_delete = Some(pos);
        }
        if ui
            .add_enabled(
                pos + 1 != len,
                Button::new(RichText::new("⏷").color(ui.visuals().hyperlink_color)), // down
            )
            .clicked()
        {
            to_move_down = Some(pos);
        }
        if ui
            .add_enabled(
                pos != 0,
                Button::new(RichText::new("⏶").color(ui.visuals().hyperlink_color)), // up
            )
            .clicked()
        {
            to_move_up = Some(pos);
        }
    };

    for (pos, elem) in array.iter_mut().enumerate() {
        if compact {
            ui.horizontal(|ui| {
                egui_t(elem, ui, data_id.with(pos));
                buttons_ui(pos, ui);
            });
        } else {
            egui::CollapsingHeader::new(format!("# {pos}"))
                .id_source(pos)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        buttons_ui(pos, ui);
                    });

                    egui_t(elem, ui, data_id.with(pos));
                });
        }
    }

    if let Some(pos) = to_delete {
        array.remove(pos);
    } else if let Some(pos) = to_move_up {
        array.swap(pos, pos - 1);
    } else if let Some(pos) = to_move_down {
        array.swap(pos, pos + 1);
    }

    if ui
        .add(Button::new(RichText::new("Add").color(Color32::GREEN)))
        .clicked()
    {
        array.push(Default::default());
    }
}
