
use burn::{
    module::Param, nn::conv::Conv2dConfig, prelude::*,
    tensor::activation::softmax,
};

// #[derive(Module, Debug)]
pub struct DFL<B: Backend> {
    pub conv: nn::conv::Conv2d<B>,
    pub c1: usize,
}
impl <B: Backend> DFL<B> {
    pub fn new(c1: usize, device: &B::Device) -> Self {
        let mut conv = Conv2dConfig::new([c1, 1], [1, 1])
        .with_bias(false).init(device).no_grad();

        let x = Tensor::<B, 1, Int>::arange(0..c1 as i64, device);
        let x = x.float().reshape(Shape::new([1, c1 as usize, 1, 1])).no_grad();
        conv.weight = Param::from_tensor(x).no_grad();
        Self { conv, c1 }
    }
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = input.dims();
        let b = shape[0];
        let a = shape[2];
        let reshaped_input = input.reshape(Shape::new([b, 4, self.c1 as usize, a]));
        self.conv.forward(
            softmax(
            reshaped_input.movedim(2, 1)
            ,1
        )).reshape(Shape::new([b, 4, a]))
    }
}

