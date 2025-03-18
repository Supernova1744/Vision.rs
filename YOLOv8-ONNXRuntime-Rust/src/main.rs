use clap::Parser;

use std::error::Error;
use tonic::transport::Server;

use yolov8_rs::{
    Args, YOLOv8,
    yolo_service_server::YoloServiceServer,
    MyYoloService
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments and initialize the YOLOv8 model once.
    let args = Args::parse();

    let model = YOLOv8::new(args)
        .map_err(|e| format!("Error creating model: {:?}", e))?;
    model.summary();

    // Create the service with the pre-initialized model.
    let yolo_service = MyYoloService::new(model);
    let addr = "[::1]:50051".parse()?;
    println!("YOLOService server listening on {}", addr);

    // Start the gRPC server.
    Server::builder()
        .add_service(YoloServiceServer::new(yolo_service))
        .serve(addr)
        .await?;

    Ok(())
}
