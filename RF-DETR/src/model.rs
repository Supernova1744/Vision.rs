use ort::session::builder::SessionBuilder;
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

pub struct OnnxModel {
    provider: [ort::execution_providers::ExecutionProviderDispatch; 1]

}
impl OnnxModel {
    pub fn new(cuda: bool) -> Self {
        let provider = if cuda {
            [CUDAExecutionProvider::default().build().error_on_failure()]
        } else {
            [CPUExecutionProvider::default().build()]
        };
        Self {
            provider
        }
    }
    pub fn load_model(&self, model_path: &str) -> Result<ort::session::Session, Box<dyn std::error::Error>> {
        let session = SessionBuilder::new()?
            .with_execution_providers(self.provider.clone())?
            // .with_intra_threads(4)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(session)
    }
}
