use egui::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Storage2<T: StorageElem2> {
    pub name: String,
    pub storage: Vec<T>,
}

impl<T: StorageElem2> Storage2<T> {
    pub fn new(name: String) -> Self {
        Self {
            name,
            storage: Default::default(),
        }
    }

    pub fn from_vec(name: String, storage: Vec<T>) -> Self {
        Self { name, storage }
    }

    pub fn egui(&mut self, ui: &mut Ui) {
        let data_id = ui.make_persistent_id(&self.name);
        egui::CollapsingHeader::new(&self.name)
            .id_source(data_id.with("header"))
            .default_open(false)
            .show(ui, |ui| {
                self.egui_inner(ui, data_id);
            });
    }

    pub fn egui_inner(&mut self, ui: &mut Ui, data_id: egui::Id) {
        let mut to_delete = None;
        let mut to_move_up = None;
        let mut to_move_down = None;

        let len = self.storage.len();
        for (pos, elem) in self.storage.iter_mut().enumerate() {
            egui::CollapsingHeader::new(format!("id{pos}"))
                .id_source(pos)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .add(Button::new(RichText::new("Delete").color(Color32::RED)))
                                .clicked()
                            {
                                to_delete = Some(pos);
                            }
                            if ui
                                .add_enabled(
                                    pos + 1 != len,
                                    Button::new(
                                        RichText::new("⏷").color(ui.visuals().hyperlink_color),
                                    ), // down
                                )
                                .clicked()
                            {
                                to_move_down = Some(pos);
                            }
                            if ui
                                .add_enabled(
                                    pos != 0,
                                    Button::new(
                                        RichText::new("⏶").color(ui.visuals().hyperlink_color),
                                    ), // up
                                )
                                .clicked()
                            {
                                to_move_up = Some(pos);
                            }
                        });
                    });

                    elem.egui(ui, data_id.with(pos));
                });
        }

        if let Some(pos) = to_delete {
            self.storage.remove(pos);
        } else if let Some(pos) = to_move_up {
            self.storage.swap(pos, pos - 1);
        } else if let Some(pos) = to_move_down {
            self.storage.swap(pos, pos + 1);
        }

        if ui
            .add(Button::new(RichText::new("Add").color(Color32::GREEN)))
            .clicked()
        {
            self.storage.push(Default::default());
        }
    }
}

pub trait StorageElem2: Sized + Default + Clone + Serialize {
    fn egui(&mut self, ui: &mut Ui, data_id: egui::Id);
}
