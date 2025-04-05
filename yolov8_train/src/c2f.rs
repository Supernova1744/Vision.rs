use burn::prelude::*;
use crate::conv::ConvBlock;
use crate::bottleneck::Bottleneck;


#[derive(Module, Debug)]
pub struct C2f<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    bottlenecks: Vec<Bottleneck<B>>,
    c: usize,
}

impl<B: Backend> C2f<B> {
    pub fn new(
        c1: usize,
        c2: usize,
        n: Option<usize>,
        shortcut: Option<bool>,
        g: Option<usize>,
        e: Option<f32>,
        device: &B::Device,
    ) -> Self {

        let n = n.unwrap_or(1);
        let shortcut = shortcut.unwrap_or(false);
        let g = g.unwrap_or(1);
        let e = e.unwrap_or(0.5);
        let c = (c2 as f32 * e) as usize;

        let conv1 = ConvBlock::new(
            c1,
            2 * c,
            1,
            1,
            None,
            None,
            None,
            device
        );
        let conv2 = ConvBlock::new(
            ((2 + n) * c) as usize,
            c2,
            1,
            1,
            None,
            None,
            None,
            device
        );
        
        let bottlenecks = (0..n)
            .map(|_| Bottleneck::new(
                c,
                c,
                shortcut,
                Some(g),
                Some([3, 3]),
                Some(1.0),
                device)
            ).collect();

        Self {
            conv1,
            conv2,
            bottlenecks,
            c
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = input.detach();
        let mut y = self.conv1.forward(x).split(self.c, 1);
        
        for bottleneck in &self.bottlenecks {
            // Split into separate steps to avoid overlapping borrows
            let last = y.get(y.len().wrapping_sub(1)).unwrap();          // Mutable borrow for pop
            let processed = bottleneck.forward(last.clone()); // No borrow of `y` here
            y.push(processed);                    // New mutable borrow for push
        }
        
        self.conv2.forward(Tensor::cat(y, 1))
    }
}