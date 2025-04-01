use image::GenericImageView;
use ndarray::{Array, ArrayBase, CowArray, IxDynImpl, OwnedRepr};
use tokio::sync::Mutex;

use tonic::{Response, Status};
use crate::cli::Args;
use crate::grpc;
use crate::preprocess::PreProcessor;
use crate::postprocess::PostProcessor;


#[derive(Debug)]
pub struct MyImageProcessor {
        model: Mutex<ort::session::Session>,
        preprocessor: Mutex<PreProcessor>,
        postprocessor: Mutex<PostProcessor>,
        args: Args,
}

impl Default for MyImageProcessor {
    fn default() -> Self {
        panic!("Default is not implemented for MyImageProcessor because ort::session::Session does not implement Default");
    }
}

impl MyImageProcessor {
    /// Creates a new instance of MyImageProcessor with the provided model and processor.
    pub fn new(model: ort::session::Session, preprocessor: PreProcessor, postprocessor: PostProcessor, args: Args) -> Self {
        Self {
            model: Mutex::new(model),
            preprocessor: Mutex::new(preprocessor),
            postprocessor: Mutex::new(postprocessor),
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
        let t = std::time::Instant::now();
        // 1. Decode image bytes
        let image_data = &request.into_inner().image_data;
        let image = image::load_from_memory(image_data)
        .map_err(|e| Status::invalid_argument(format!("Invalid image: {}", e)))?;
        let (orig_w, orig_h) = image.dimensions();
        if self.args.profile {
            println!("[image loading]: {:?}", t.elapsed());
        }
        
        let t = std::time::Instant::now();
        // 2. Run your image processing logic (replace with your actual function)
        let (xs, offset) = self.preprocessor.lock().await.preprocess(&vec![image.clone()], self.args.deep_profile)
            .map_err(|e| Status::internal(format!("Preprocessing error: {}", e)))?;
        if self.args.profile {
            println!("[preprocessing]: {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        let xs = CowArray::from(xs);
        let input_data = ort::inputs![xs.view()].map_err(|e| Status::internal(format!("ORT input error: {}", e)))?;
        if self.args.profile {
            println!("[input tensor preparation]: {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        let session = self.model.lock().await;
        let ys = session.run(input_data)
            .map_err(|e| Status::internal(format!("Model run error: {}", e)))?;
        if self.args.profile {
            println!("[model run]: {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();
        let i: Vec<ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>>> = ys
            .iter()
            .map(|(_k, v)| {
                let v = v.try_extract_tensor::<f32>().unwrap().into_owned();
                v
            })
            .collect::<Vec<Array<_, _>>>();
        if self.args.profile {
            println!("[model output]: {:?}", t.elapsed());
        }
        let t = std::time::Instant::now();

        let (filtered_boxes, filtered_classes, filtered_conf ) = self.postprocessor.lock().await
            .postprocess(i, orig_w as f32, orig_h as f32, offset)
            .map_err(|e| Status::internal(format!("Postprocessing error: {}", e)))?;
        if self.args.profile {
            println!("[postprocessing]: {:?}", t.elapsed());
        }
        // 3. Prepare response
        Ok(Response::new(crate::grpc::DetectionResponse {
            filtered_conf,
            filtered_classes,
            filtered_boxes: filtered_boxes.into_iter().flat_map(|b| b).collect(),
        }))
    }

    
    

}