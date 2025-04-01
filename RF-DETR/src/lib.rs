pub mod grpc;
pub mod preprocess;
pub mod model;
pub mod cli;
pub mod mapping;
pub mod postprocess;
pub mod service;

pub use crate::model::OnnxModel;
pub use crate::grpc::{ImageRequest, DetectionResponse};
pub use crate::preprocess::PreProcessor;
pub use crate::cli::Args;
pub use crate::mapping::load_class_mapping;
pub use crate::postprocess::PostProcessor;
pub use crate::service::MyImageProcessor;