use clap::Parser;


#[derive(Parser, Clone, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// ONNX model path
    #[arg(long, required = true)]
    pub model: String,

    /// using CUDA EP
    #[arg(long)]
    pub cuda: bool,

    #[arg(long)]
    pub profile: bool,

    #[arg(long)]
    pub deep_profile: bool,

    #[arg(long, default_value_t = String::from(r"output\"))]
    pub output: String,

    #[arg(long, default_value_t = 560)]
    pub img_w: usize,

    #[arg(long, default_value_t = 560)]
    pub img_h: usize,
    
    #[arg(long, default_value_t = 0.5)]
    pub conf_th: f32,
    
    #[arg(long, default_value_t = 0.25)]
    pub iou_th: f32,
    
    #[arg(skip = 3)]
    pub ch: i32,

    #[arg(skip = [0.485, 0.456, 0.406])]
    pub mean: [f32; 3],

    #[arg(skip = [0.229, 0.224, 0.225])]
    pub std: [f32; 3],

}
