// #![warn(clippy::all, rust_2018_idioms)]
#![allow(dead_code)]

mod app;
mod common;
mod math;
mod physics;
mod storage;
pub use app::TemplateApp;

#[macro_export]
macro_rules! error {
	(format, $format_string:literal, $($args:expr),*) => {
		println!("Error on {}:{}. {}", file!(), line!(), format!($format_string, $($args),*))
	};

	(debug, $context:expr) => {
		println!("Error on {}:{}, debug context: {:?}", file!(), line!(), $context)
	};

	() => {
		println!("Error on {}:{}", file!(), line!())
	};
}
