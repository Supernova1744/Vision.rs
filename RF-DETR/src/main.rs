use clap::Parser;

use tonic::transport::Server;
use RF_DETR::{Processor, PreprocessConfig, OnnxModel, Args};

use RF_DETR::service::MyImageProcessor;
use RF_DETR::grpc::image_processor_server::ImageProcessorServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start gRPC server
    let args = Args::parse();
    let addr = "[::1]:50051".parse()?;
    let processor = Processor::new(PreprocessConfig::default());
    let model = OnnxModel::new(args.clone().cuda).load_model(&args.model)?;
    let x = ImageProcessorServer::new(MyImageProcessor::new(model, processor, args));
    println!("ImageProcessor server listening on {}", addr);
    Server::builder()
    .add_service(x)
    .serve(addr)
    .await?;

    Ok(())
}
