use burn::{
    nn::{BatchNorm, PaddingConfig2d},
    prelude::*
};
use num_integer::Integer;


fn autopad(k: usize, p: Option<usize>, d: usize) -> usize {
    let mut v: usize = k;
    if d > 1 {
        v = d * (k - 1) + 1;
    }
    if p.is_none() {
        return v / 2;
    }
    p.unwrap()
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: nn::SiLU,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(c1: usize, c2: usize, k: usize, s: usize, p: Option<usize>, g: Option<usize>, d: Option<usize>, device: &B::Device) -> Self {
        let g = g.unwrap_or(1);
        let d = d.unwrap_or(1);
        let p = autopad(k, p, d);
        let conv = nn::conv::Conv2dConfig::new([c1, c2], [k, k])
            .with_groups(g)
            .with_stride([s, s])
            .with_dilation([d, d])
            .with_bias(false)
            .with_padding(PaddingConfig2d::Explicit(p, p))
            .init(device);
        let norm = nn::BatchNormConfig::new(c2).init(device);

        Self {
            conv,
            norm,
            activation: <nn::SiLU as burn::module::Module<B>>::to_device(nn::SiLU::new(), device)
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }

    pub fn forward_fuse(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        self.activation.forward(x)
    }
}


#[derive(Module, Debug)]
pub struct DWConv<B: Backend> {
    conv: ConvBlock<B>,
}

impl <B: Backend> DWConv<B> {
    pub fn new(c1: usize, c2: usize, k: usize, s: usize, d: Option<usize>, device: &B::Device) -> Self {
        let g = c1.gcd(&(c2 as usize));
        let conv = ConvBlock::new(c1, c2, k, s, None, Some(g), d, device);
        Self { conv }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(input)
    }
}