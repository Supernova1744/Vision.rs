use clap::Parser;
use tonic::transport::Server;
use RF_DETR::service::MyImageProcessor;
use RF_DETR::{PreProcessor, PostProcessor, OnnxModel, Args};
use RF_DETR::grpc::image_processor_server::ImageProcessorServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the command line arguments
    let args = Args::parse();
    // Define gRPC server address
    let addr = "[::1]:50051".parse()?;
    // Load the model, preprocessors, and postprocessors
    let model = OnnxModel::new(args.clone().cuda).load_model(&args.model)?;
    let preprocessor = PreProcessor::new(args.clone());
    let postprocessor = PostProcessor::new(args.clone());
    
    let server = ImageProcessorServer::new(
        MyImageProcessor::new(
            model,
            preprocessor,
            postprocessor,
            args)
        );
    Server::builder()
    .add_service(server)
    .serve(addr)
    .await?;
    println!("RF-DETR Object Detection server listening on {}", addr);

    Ok(())
}
