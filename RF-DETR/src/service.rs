use ndarray::{Array, ArrayBase, CowArray, IxDynImpl, OwnedRepr};
use tokio::sync::Mutex;

use tonic::{Response, Status};
use crate::cli::Args;
use crate::grpc;
use crate::preprocess::Processor;
use crate::postprocess::{softmax_and_filter, non_maximum_suppression};


#[derive(Debug)]
pub struct MyImageProcessor {
        model: Mutex<ort::session::Session>,
        processor: Mutex<Processor>,
        args: Args,
}

impl Default for MyImageProcessor {
    fn default() -> Self {
        panic!("Default is not implemented for MyImageProcessor because ort::session::Session does not implement Default");
    }
}

impl MyImageProcessor {
    /// Creates a new instance of MyImageProcessor with the provided model and processor.
    pub fn new(model: ort::session::Session, processor: Processor, args: Args) -> Self {
        Self {
            model: Mutex::new(model),
            processor: Mutex::new(processor),
            args: args,
        }
    }
}

#[tonic::async_trait]
impl grpc::image_processor_server::ImageProcessor for MyImageProcessor {
    async fn process_image(
        &self,
        request: tonic::Request<crate::grpc::ImageRequest>,
    ) -> Result<tonic::Response<crate::grpc::DetectionResponse>, tonic::Status> {
        // 1. Decode image bytes
        let image_data = &request.into_inner().image_data;
        let image = image::load_from_memory(image_data)
            .map_err(|e| Status::invalid_argument(format!("Invalid image: {}", e)))?;

        // 2. Run your image processing logic (replace with your actual function)
        let (xs, offset) = self.processor.lock().await.preprocess(&vec![image.clone()], self.args.deep_profile)
            .map_err(|e| Status::internal(format!("Preprocessing error: {}", e)))?;
        let xs = CowArray::from(xs);
        let input_data = ort::inputs![xs.view()].map_err(|e| Status::internal(format!("ORT input error: {}", e)))?;
        let session = self.model.lock().await;
        let ys = session.run(input_data)
            .map_err(|e| Status::internal(format!("Model run error: {}", e)))?;
        let i = ys
            .iter()
            .map(|(_k, v)| {
                let v = v.try_extract_tensor::<f32>().unwrap().into_owned();
                v
            })
            .collect::<Vec<Array<_, _>>>();
        let boxes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &i[0];
        let classes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &i[1];

        let (filtered_classes, filtered_conf, filtered_boxes) = softmax_and_filter(classes, boxes, 0.5)
            .map_err(|e| Status::internal(format!("Softmax and filter error: {}", e)))?;
        let (filtered_conf, filtered_classes, filtered_boxes) = non_maximum_suppression(filtered_conf, filtered_classes, filtered_boxes, 0.5);

        // 3. Prepare response
        Ok(Response::new(crate::grpc::DetectionResponse {
            filtered_conf,
            filtered_classes,
            filtered_boxes: filtered_boxes.into_iter().flat_map(|b| b).collect(),
        }))
    }

    
    

}