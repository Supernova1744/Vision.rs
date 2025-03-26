use clap::Parser;


#[derive(Parser, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// ONNX model path
    #[arg(long, required = true)]
    pub model: String,

    /// image path
    #[arg(long, required = true)]
    pub source: String,

    /// using CUDA EP
    #[arg(long)]
    pub cuda: bool,

    #[arg(long, default_value_t = String::from(r"output\"))]
    pub output: String,

}
