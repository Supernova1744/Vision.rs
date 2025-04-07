use burn::{
    nn::PaddingConfig2d,
    prelude::*,
};
use crate::conv::ConvBlock;


#[derive(Module, Debug)]
pub struct SPPF<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    pool: nn::pool::MaxPool2d
}
impl <B: Backend> SPPF<B> {
    pub fn new(c1: usize, c2: usize, k: usize, device: &B::Device) -> Self {
        let c_ = c1 / 2; // hidden channels
        let conv1 = ConvBlock::new(c1, c_, 1, 1, None, None, None, device);
        let conv2 = ConvBlock::new(c_ * 4, c2, 1, 1, None, None, None, device);
        let pool = nn::pool::MaxPool2dConfig::new([k, k])
        .with_strides([1, 1])
        .with_padding(PaddingConfig2d::Explicit(k/2, k/2))
        .init();
        Self {
            conv1,
            conv2,
            pool,
        }
    }
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut y = vec![self.conv1.forward(input)];
        for i in 0..3 {
            y.push(self.pool.forward(y[i].clone()));
        }
        let y = Tensor::cat(y, 1);
        let y = self.conv2.forward(y);
        y
    }
}