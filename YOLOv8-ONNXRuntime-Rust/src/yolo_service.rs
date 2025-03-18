use tokio::sync::Mutex;

use std::fmt;
use image::DynamicImage;
use tonic::{Request, Response, Status, async_trait};

use crate::{
    YOLOv8,
    ProcessImagesRequest, ProcessImagesResponse,
    yolo_service_server::YoloService,
    convert_yolo_result
};

/// The YOLO gRPC service.
pub struct MyYoloService {
    model: Mutex<YOLOv8>,
}

// Custom Debug implementation that doesn't try to print the inner YOLOv8 model.
impl fmt::Debug for MyYoloService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MyYoloService")
            .field("model", &"YOLOv8 (not Debug)")
            .finish()
    }
}

impl MyYoloService {
    /// Creates a new service instance with the provided model.
    pub fn new(model: YOLOv8) -> Self {
        Self {
            model: Mutex::new(model),
        }
    }
}

#[async_trait]
impl YoloService for MyYoloService {
    async fn process_images(
        &self,
        request: Request<ProcessImagesRequest>,
    ) -> Result<Response<ProcessImagesResponse>, Status> {
        let req = request.into_inner();
        let mut results = Vec::new();

        // Process each image in the request.
        for image_data in req.images {
            let dynamic_image: DynamicImage = image::load_from_memory(&image_data)
                .map_err(|e| Status::invalid_argument(format!("Failed to decode image: {}", e)))?;

            let xs = vec![dynamic_image];

            // Lock the model for exclusive mutable access.
            let ys = {
                let mut model = self.model.lock().await;
                model.run(&xs).map_err(|e| {
                    Status::internal(format!("Model run failed: {}", e))
                })?
            };

            let result = convert_yolo_result(&ys[0]);

            results.push(result);
        }

        let response = ProcessImagesResponse { results };
        Ok(Response::new(response))
    }
}
