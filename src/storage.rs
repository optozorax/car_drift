use egui::*;
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;


#[macro_export]
macro_rules! hlist {
    ($x:ty, $($tail:tt)+) => { ($x, hlist!($($tail)+)) };
    ($x:ty) => { ($x, ()) };
}

#[macro_export]
macro_rules! hval {
    ($x:expr, $($tail:tt)+) => { ($x, hval!($($tail)+)) };
    ($x:expr) => { ($x, ()) };
}

#[macro_export]
macro_rules! hpat {
    ($x:pat, $($tail:tt)+) => { ($x, hpat!($($tail)+)) };
    ($x:pat) => { ($x, ()) };
}

#[macro_export]
macro_rules! with_swapped {
    ($name:ident => ($($values:expr),+); $run:expr) => {{
        let mut $name = (with_swapped!(@defaults $($values),+));
        with_swapped!(@swapped $name, $($values),+);
        let result = $run;
        with_swapped!(@swapped $name, $($values),+);
        result
    }};

    (@swapped $path:expr, $x:expr, $($tail:tt)+) => {
        std::mem::swap(&mut $path.0, &mut $x);
        with_swapped!(@swapped $path.1, $($tail)+);
    };

    (@swapped $path:expr, $x:expr) => {
        std::mem::swap(&mut $path.0, &mut $x);
    };

    (@defaults $x:expr, $($tail:tt)+) => {
        (Default::default(), with_swapped!(@defaults $($tail)+))
    };

    (@defaults $x:expr) => { (Default::default(), ()) };
}

#[cfg(test)]
mod test {
    #[test]
    fn test() {
        fn a(hpat!(a, b): &mut hlist!(Vec<String>, Vec<String>)) {
            a.push("aoeu".to_string());
            b.push("stnh".to_string());
        }

        fn b(hpat!(a, b, c): &mut hlist!(Vec<String>, i64, i8)) {
            a.push("lcrg".to_string());
            *b = 10000;
            *c = 125;
        }

        #[derive(Default, Debug, PartialEq)]
        struct Scene {
            data1: Vec<String>,
            data2: Vec<String>,
            data3: i64,
            data4: i8,
        }

        let mut scene = Scene::default();

        with_swapped!(x => (scene.data1, scene.data2); a(&mut x));
        with_swapped!(x => (scene.data1, scene.data3, scene.data4); b(&mut x));

        assert_eq!(
            scene,
            Scene {
                data1: vec!["aoeu".to_owned(), "lcrg".to_owned()],
                data2: vec!["stnh".to_owned()],
                data3: 10000,
                data4: 125,
            }
        );
    }
}

pub fn egui_label(ui: &mut Ui, label: &str, size: f64) {
    let (rect, _) = ui.allocate_at_least(egui::vec2(size as f32, 0.), Sense::hover());
    ui.painter().text(
        rect.max,
        Align2::RIGHT_CENTER,
        label,
        egui::FontId::monospace(14.0),
        ui.visuals().text_color(),
    );
}

pub fn egui_with_red_field<Res>(
    ui: &mut Ui,
    has_errors: bool,
    f: impl FnOnce(&mut Ui) -> Res,
) -> Res {
    let previous = ui.visuals().clone();
    if has_errors {
        ui.visuals_mut().selection.stroke.color = Color32::RED;
        ui.visuals_mut().widgets.inactive.bg_stroke.color = Color32::from_rgb_additive(128, 0, 0);
        ui.visuals_mut().widgets.inactive.bg_stroke.width = 1.0;
        ui.visuals_mut().widgets.hovered.bg_stroke.color =
            Color32::from_rgb_additive(255, 128, 128);
    }
    let result = f(ui);
    if has_errors {
        *ui.visuals_mut() = previous;
    }
    result
}

#[derive(Debug, Clone, Default)]
#[must_use]
pub struct WhatChanged {
    pub uniform: bool,
    pub shader: bool,
}

impl WhatChanged {
    pub fn from_uniform(uniform: bool) -> Self {
        Self {
            uniform,
            shader: false,
        }
    }

    pub fn from_shader(shader: bool) -> Self {
        Self {
            uniform: false,
            shader,
        }
    }
}

impl std::ops::BitOrAssign for WhatChanged {
    fn bitor_assign(&mut self, rhs: Self) {
        self.uniform |= rhs.uniform;
        self.shader |= rhs.shader;
    }
}

pub fn check_changed<T: PartialEq + Clone, F: FnOnce(&mut T)>(t: &mut T, f: F) -> bool {
    let previous = t.clone();
    f(t);
    previous != *t
}

pub fn egui_bool(ui: &mut Ui, flag: &mut bool) -> bool {
    check_changed(flag, |flag| drop(ui.add(Checkbox::new(flag, ""))))
}

#[derive(Clone, Debug, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct UniqueId(usize);

impl fmt::Display for UniqueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default, Serialize, Deserialize)]
pub struct UniqueIds {
    available: VecDeque<UniqueId>,
    max: usize,
}

impl Wrapper for () {
    fn wrap(_: UniqueId) -> Self {}
    fn un_wrap(self) -> UniqueId {
        UniqueId(0)
    }
}

impl UniqueIds {
    pub fn get_unique(&mut self) -> UniqueId {
        if let Some(result) = self.available.pop_front() {
            result
        } else {
            let result = UniqueId(self.max);
            self.max += 1;
            result
        }
    }

    pub fn remove_existing(&mut self, id: UniqueId) {
        self.available.push_back(id);
        self.available.make_contiguous().sort();
        while self
            .available
            .back()
            .map(|x| x.0 == self.max - 1)
            .unwrap_or(false)
        {
            self.max -= 1;
            self.available.pop_back().unwrap();
        }
    }
}

#[cfg(test)]
mod id_test {
    use super::*;

    #[test]
    fn test() {
        let mut ids = UniqueIds::default();
        assert_eq!(ids.get_unique().0, 0);
        assert_eq!(ids.get_unique().0, 1);
        assert_eq!(ids.get_unique().0, 2);
        assert_eq!(ids.get_unique().0, 3);
        ids.remove_existing(UniqueId(2));
        assert_eq!(
            ids,
            UniqueIds {
                available: vec![UniqueId(2)].into_iter().collect(),
                max: 4,
            }
        );
        ids.remove_existing(UniqueId(3));
        assert_eq!(
            ids,
            UniqueIds {
                available: vec![].into_iter().collect(),
                max: 2,
            }
        );
        ids.remove_existing(UniqueId(1));
        assert_eq!(
            ids,
            UniqueIds {
                available: vec![].into_iter().collect(),
                max: 1,
            }
        );
        assert_eq!(ids.get_unique().0, 1);
        ids.remove_existing(UniqueId(0));
        assert_eq!(ids.get_unique().0, 0);
    }
}


#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageWithNames<T> {
    pub names: Vec<String>,
    pub storage: Vec<T>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum StorageInner<T> {
    Named(T, String),
    Inline(T),
}

impl<T: Default> Default for StorageInner<T> {
    fn default() -> Self {
        StorageInner::Inline(T::default())
    }
}

impl<T> AsRef<T> for StorageInner<T> {
    fn as_ref(&self) -> &T {
        use StorageInner::*;
        match self {
            Named(t, _) => t,
            Inline(t) => t,
        }
    }
}

impl<T> AsMut<T> for StorageInner<T> {
    fn as_mut(&mut self) -> &mut T {
        use StorageInner::*;
        match self {
            Named(t, _) => t,
            Inline(t) => t,
        }
    }
}

impl<T> StorageInner<T> {
    fn is_named_as(&self, name: &str) -> bool {
        use StorageInner::*;
        match self {
            Named(_, n) => n == name,
            Inline(_) => false,
        }
    }

    fn is_inline(&self) -> bool {
        use StorageInner::*;
        match self {
            Named(_, _) => false,
            Inline(_) => true,
        }
    }

    fn name(&self) -> Option<&str> {
        use StorageInner::*;
        match self {
            Named(_, n) => Some(n),
            Inline(_) => None,
        }
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Storage2<T: StorageElem2> {
    ids: UniqueIds,
    storage: BTreeMap<UniqueId, StorageInner<T>>,
    storage_order: Vec<UniqueId>,
}

pub struct GetHelper<'a, T: StorageElem2>(&'a Storage2<T>, &'a T::GetInput);

impl<'a, T: StorageElem2> GetHelper<'a, T> {
    pub fn get(&self, id: T::IdWrapper) -> Option<T::GetType> {
        self.0.get(id, self.1)
    }

    pub fn find_id(&self, name: &str) -> Option<T::IdWrapper> {
        self.0.find_id(name)
    }
}

pub struct InlineHelper<'a, T: StorageElem2>(&'a mut Storage2<T>);

impl<'a, T: StorageElem2> InlineHelper<'a, T> {
    pub fn inline(
        &mut self,
        label: &str,
        label_size: f64,
        id: &mut Option<T::IdWrapper>,
        ui: &mut Ui,
        input: &mut T::Input,
        data_id: egui::Id,
    ) -> WhatChanged {
        self.0.inline(label, label_size, id, ui, input, data_id)
    }
}

impl<T: StorageElem2> Storage2<T> {
    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut result: Self = Default::default();
        for elem in vec {
            let id = result.ids.get_unique();
            result.storage_order.push(id);
            result.storage.insert(
                id,
                StorageInner::Named(elem, format!("id{}", id)),
            );
        }
        result
    }

    pub fn get(&self, id: T::IdWrapper, input: &T::GetInput) -> Option<T::GetType> {
        let mut visited = vec![];
        self.get_inner(id, &mut visited, input)
    }

    pub fn get_original(&self, id: T::IdWrapper) -> Option<&T> {
        self.storage.get(&id.un_wrap()).map(|x| x.as_ref())
    }

    pub fn get_original_mut(&mut self, id: T::IdWrapper) -> Option<&mut T> {
        self.storage.get_mut(&id.un_wrap()).map(|x| x.as_mut())
    }

    /// First Option shows is element presented or not. Second Option is represente is element has a name
    pub fn get_name(&self, id: T::IdWrapper) -> Option<Option<&str>> {
        self.storage.get(&id.un_wrap()).map(|x| x.name())
    }

    pub fn find_id(&self, name: &str) -> Option<T::IdWrapper> {
        self.storage_order
            .iter()
            .find(|id| {
                self.storage
                    .get(id)
                    .map(|elem| elem.is_named_as(name))
                    .unwrap_or(false)
            })
            .map(|id| T::IdWrapper::wrap(*id))
    }

    pub fn all_ids(&self) -> impl Iterator<Item = T::IdWrapper> + '_ {
        self.storage.keys().map(|key| T::IdWrapper::wrap(*key))
    }

    pub fn all_elements(&self) -> impl Iterator<Item = &T> + '_ {
        self.storage.values().map(|x| x.as_ref())
    }

    pub fn all_elements_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.storage.values_mut().map(|x| x.as_mut())
    }

    fn get_inner(
        &self,
        id: T::IdWrapper,
        visited: &mut Vec<T::IdWrapper>,
        input: &T::GetInput,
    ) -> Option<T::GetType> {
        if visited.iter().any(|x| x.un_wrap() == id.un_wrap()) {
            return None;
        }

        visited.push(id);
        let result = self
            .storage
            .get(&id.un_wrap())?
            .as_ref()
            .get(&GetHelper(self, input), input);
        visited.pop().unwrap();
        result
    }

    fn remove(&mut self, id: T::IdWrapper, input: &mut T::Input) {
        let id = id.un_wrap();
        let element = self.storage.remove(&id).unwrap();
        self.ids.remove_existing(id);
        if let Some(pos) = self.storage_order.iter().position(|x| *x == id) {
            self.storage_order.remove(pos);
        }

        // Recursively delete inline elements
        element
            .as_ref()
            .remove(|id, input| self.remove_as_field(id, input), input);
    }

    pub fn set(&mut self, id: T::IdWrapper, value: T) {
        if let Some(t) = self.storage.get_mut(&id.un_wrap()) {
            *t.as_mut() = value;
        } else {
            crate::error!()
        }
    }

    pub fn set_id(&mut self, id: T::IdWrapper, from: T::IdWrapper) {
        let id = id.un_wrap();
        let from = from.un_wrap();

        if let Some(t) = self.storage.get(&from).map(|x| (*x).as_ref().clone()) {
            *self.storage.get_mut(&id).unwrap().as_mut() = t;
        } else {
            crate::error!();
        }
    }

    fn push_default(&mut self) {
        let id = self.ids.get_unique();
        self.storage_order.push(id);
        self.storage.insert(
            id,
            StorageInner::Named(Default::default(), format!("id{}", id)),
        );
    }

    /// Removes field like it is was field of someone, then recursively removes inside content if it's inline.
    pub fn remove_as_field(&mut self, id: T::IdWrapper, input: &mut T::Input) {
        let id = id.un_wrap();
        if self.storage.get(&id).unwrap().is_inline() {
            let element = self.storage.remove(&id).unwrap();
            self.ids.remove_existing(id);

            element
                .as_ref()
                .remove(|id, input| self.remove_as_field(id, input), input);
        }
    }

    fn remove_by_pos(&mut self, pos: usize, input: &mut T::Input) {
        let id = self.storage_order.remove(pos);
        self.remove(T::IdWrapper::wrap(id), input);
    }

    pub fn visible_elements(&self) -> impl Iterator<Item = (T::IdWrapper, &str)> + '_ {
        let storage = &self.storage;
        self.storage_order.iter().map(move |id| {
            (
                T::IdWrapper::wrap(*id),
                storage.get(id).unwrap().name().unwrap(),
            )
        })
    }

    pub fn egui(&mut self, ui: &mut Ui, input: &mut T::Input, name: &str) -> WhatChanged {
        use std::borrow::Cow;

        let data_id = ui.make_persistent_id(name).with("inner");

        let errors_count = self.errors_count_all(input);
        let header = if errors_count > 0 {
            Cow::Owned(format!("{} ({} err)", name, errors_count))
        } else {
            Cow::Borrowed(name)
        };
        let mut changed = WhatChanged::default();
        egui::CollapsingHeader::new(header)
            .id_source(name)
            .default_open(false)
            .show(ui, |ui| {
                changed |= self.egui_inner(ui, input, data_id);
            });
        changed
    }

    fn egui_inner(&mut self, ui: &mut Ui, input: &mut T::Input, data_id: egui::Id) -> WhatChanged {
        let mut changed = WhatChanged::default();
        let mut to_delete = None;
        let mut to_move_up = None;
        let mut to_move_down = None;

        let mut storage_order = Vec::new();
        std::mem::swap(&mut storage_order, &mut self.storage_order);

        let len = storage_order.len();
        for (pos, id) in storage_order.iter().enumerate() {
            let errors_count = self.errors_count_id(T::IdWrapper::wrap(*id), input);

            let mut elem = StorageInner::default();
            std::mem::swap(&mut elem, self.storage.get_mut(id).unwrap());

            if let StorageInner::Named(elem, name) = &mut elem {
                let name_error = self.storage.iter().any(|x| x.1.is_named_as(name));

                let errors_count = errors_count + name_error as usize;

                let header_name = if errors_count > 0 {
                    format!("{} ({} err)", name, errors_count)
                } else {
                    name.clone()
                };

                egui::CollapsingHeader::new(header_name)
                    .id_source(id)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            egui_label(ui, "Name:", 45.);
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
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
                                let mut name_response = egui_with_red_field(ui, name_error, |ui| {
                                    ui.text_edit_singleline(name)
                                });
                                changed.shader |= name_response.changed();
                                if !T::SAFE_TO_RENAME {
                                    name_response = name_response.on_hover_text(
                                        "This name is not safe to rename, you will\n\
                                    need to rename it in other places by yourself",
                                    );
                                }
                                if name_error {
                                    name_response.on_hover_text(format!(
                                        "Error: name '{}' already used",
                                        name
                                    ));
                                }
                            });
                        });

                        changed |= elem.egui(
                            ui,
                            input,
                            &mut InlineHelper(self),
                            data_id.with(pos),
                            T::IdWrapper::wrap(*id),
                        );
                    });
            } else {
                ui.label("Internal error, this is inline element, it shouldn't be here.");
            }

            std::mem::swap(&mut elem, self.storage.get_mut(id).unwrap());
        }
        std::mem::swap(&mut storage_order, &mut self.storage_order);

        if let Some(pos) = to_delete {
            changed.shader = true;
            self.remove_by_pos(pos, input);
        } else if let Some(pos) = to_move_up {
            self.storage_order.swap(pos, pos - 1);
        } else if let Some(pos) = to_move_down {
            self.storage_order.swap(pos, pos + 1);
        }

        if ui
            .add(Button::new(RichText::new("Add").color(Color32::GREEN)))
            .clicked()
        {
            self.push_default();
            changed.shader = true;
        }

        changed
    }

    pub fn inline(
        &mut self,
        label: &str,
        label_size: f64,
        id: &mut Option<T::IdWrapper>,
        ui: &mut Ui,
        input: &mut T::Input,
        data_id: egui::Id,
    ) -> WhatChanged {
        let mut changed = WhatChanged::default();

        ui.vertical(|ui| {
            if let Some(id_inner) = id {
                if self.storage.get(&id_inner.un_wrap()).is_none() {
                    crate::error!(format, "id {:?} transformed to `None`", id_inner);
                    *id = None;
                    changed.uniform = true;
                }
            }

            let mut inline = if let Some(id_inner) = id {
                self.storage
                    .get(&id_inner.un_wrap())
                    .map(|x| x.is_inline())
                    .unwrap() // Because we earlier checked this
            } else {
                false
            };

            ui.horizontal(|ui| {
                egui_label(ui, label, label_size);

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(egui::SelectableLabel::new(inline, "📌"))
                        .on_hover_text(
                            "Toggle inline anonymous element instead\nof referencing to name of the other.",
                        )
                        .clicked()
                    {
                        if inline {
                            if let Some(id) = id {
                                self.remove(*id, input);
                                ui.memory_mut(|memory| memory.data.remove::<String>(data_id));
                            }
                        }

                        inline = !inline;

                        if inline {
                            let new_id = self.ids.get_unique();
                            self.storage
                                .insert(new_id, StorageInner::Inline(Default::default()));
                            *id = Some(T::IdWrapper::wrap(new_id));
                        } else {
                            *id = None;
                        }
                    }

                    if inline {
                        ui.label("inline");
                    } else {
                        // Named
                        changed.uniform |= self.find_named_id(id, ui, data_id);
                    }
                });
            });

            if inline {
                // id now must be correct
                with_swapped!(elem => (*self.storage.get_mut(&id.unwrap().un_wrap()).unwrap()); {
                    ui.group(|ui| {
                        changed |= elem.0.as_mut().egui(ui, input, &mut InlineHelper(self), data_id.with("inline"), id.unwrap());
                    });
                });
            }
        });

        changed
    }

    fn find_named_id(&self, id: &mut Option<T::IdWrapper>, ui: &mut Ui, data_id: egui::Id) -> bool {
        let mut current_name = if let Some(id_inner) = id {
            self.storage
                .get(&id_inner.un_wrap())
                .unwrap()
                .name()
                .unwrap()
                .to_owned()
        } else {
            ui.memory_mut(|memory| {
                memory
                    .data
                    .get_persisted_mut_or_default::<String>(data_id)
                    .clone()
            })
        };

        let changed = ui
            .horizontal(|ui| {
                let mut response = egui_with_red_field(ui, id.is_none(), |ui| {
                    ui.text_edit_singleline(&mut current_name)
                });
                if id.is_none() {
                    response = response.on_hover_text("This name is not found");
                }

                response.changed()
            })
            .inner;
        if changed {
            if let Some((new_id, _)) = self
                .storage
                .iter()
                .find(|(_, elem)| elem.is_named_as(&current_name))
            {
                *id = Some(T::IdWrapper::wrap(*new_id));
                ui.memory_mut(|memory| memory.data.remove::<String>(data_id));
            } else {
                *id = None;
                ui.memory_mut(|memory| memory.data.insert_persisted(data_id, current_name));
            }
        }

        changed
    }

    pub fn inline_only_name(
        &mut self,
        label: &str,
        label_size: f64,
        id: &mut Option<T::IdWrapper>,
        ui: &mut Ui,
        data_id: egui::Id,
    ) -> WhatChanged {
        let mut changed = WhatChanged::default();
        ui.vertical(|ui| {
            if let Some(id_inner) = id {
                if self.storage.get(&id_inner.un_wrap()).is_none() {
                    crate::error!(format, "id {:?} transformed to `None`", id_inner);
                    *id = None;
                    changed.uniform = true;
                }
            }

            ui.horizontal(|ui| {
                egui_label(ui, label, label_size);
                changed.uniform |= self.find_named_id(id, ui, data_id);
            });
        });

        changed
    }

    // This element is inline, so errors counted only for inline elements and inner inline elements
    pub fn errors_inline(&self, id: T::IdWrapper, input: &T::Input) -> usize {
        let mut visited = vec![];
        self.errors_count_inner(id, &mut visited, input, false)
    }

    pub fn errors_count_all(&self, input: &T::Input) -> usize {
        self.storage_order
            .iter()
            .map(|id| self.errors_count_id(T::IdWrapper::wrap(*id), input))
            .sum()
    }

    // Errors count for current id, it must be not inline element.
    pub fn errors_count_id(&self, id: T::IdWrapper, input: &T::Input) -> usize {
        let mut visited = vec![];
        self.errors_count_inner(id, &mut visited, input, true)
    }

    fn errors_count_inner(
        &self,
        id: T::IdWrapper,
        visited: &mut Vec<T::IdWrapper>,
        input: &T::Input,
        allow_not_inline: bool,
    ) -> usize {
        if let Some(elem) = self.storage.get(&id.un_wrap()) {
            if !allow_not_inline && !elem.is_inline() {
                return 0;
            }

            if visited.iter().any(|x| x.un_wrap() == id.un_wrap()) {
                return 0;
            }

            visited.push(id);
            let result = elem.as_ref().errors_count(
                |id| self.errors_count_inner(id, visited, input, false),
                input,
                id,
            );
            visited.pop().unwrap();
            result
        } else {
            1
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.storage_order.len()
    }
}

pub trait Wrapper:
    Clone
    + std::fmt::Debug
    + Copy
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + std::hash::Hash
    + serde::Serialize
    + for<'a> serde::Deserialize<'a>
{
    fn wrap(t: UniqueId) -> Self;
    fn un_wrap(self) -> UniqueId;
}

pub trait StorageElem2: Sized + Default + Clone + Serialize {
    type IdWrapper: Wrapper;
    type GetType;

    const SAFE_TO_RENAME: bool;

    type Input;
    type GetInput;

    fn egui(
        &mut self,
        ui: &mut Ui,
        input: &mut Self::Input,
        inline_helper: &mut InlineHelper<'_, Self>,
        data_id: egui::Id,
        self_id: Self::IdWrapper,
    ) -> WhatChanged;

    fn get(&self, get_helper: &GetHelper<'_, Self>, input: &Self::GetInput) -> Option<Self::GetType>;

    fn remove<F: FnMut(Self::IdWrapper, &mut Self::Input)>(&self, f: F, input: &mut Self::Input);

    fn errors_count<F: FnMut(Self::IdWrapper) -> usize>(
        &self,
        f: F,
        input: &Self::Input,
        self_id: Self::IdWrapper,
    ) -> usize;
}

impl<T: StorageElem2> From<StorageWithNames<T>> for Storage2<T> {
    fn from(_: StorageWithNames<T>) -> Storage2<T> {
        todo!()
    }
}
