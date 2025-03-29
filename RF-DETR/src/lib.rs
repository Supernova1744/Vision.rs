pub mod grpc;
pub mod preprocess;
pub mod model;
pub mod cli;
pub mod mapping;
pub mod postprocess;
pub mod service;

pub use crate::model::OnnxModel;
pub use crate::grpc::{ImageRequest, DetectionResponse};
pub use crate::preprocess::{Processor, PreprocessConfig};
pub use crate::cli::Args;
pub use crate::mapping::load_class_mapping;
pub use crate::postprocess::{softmax_and_filter, non_maximum_suppression};
pub use crate::service::MyImageProcessor;