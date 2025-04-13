use burn::prelude::*;
use crate::conv::ConvBlock;

// #[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    add: bool,
}


impl <B: Backend> Bottleneck<B> {
    /// Create a new Bottleneck module.
    ///
    /// # Arguments
    /// * `channels`: Input and output channels.
    /// * `kernel_size`: Kernel sizes for convolutions.
    /// * `e`: Expansion ratio.
    /// * `g`: Groups for convolutions.
    /// * `shortcut`: Whether to use shortcut connection.
    /// * `device`: Device to initialize the module on.
    ///

    pub fn new(c1: usize, c2: usize, shortcut:bool, g:Option<usize>, k: Option<[usize; 2]>, e: Option<f32>, device: &B::Device) -> Self {
        let g = g.unwrap_or(1);
        let k = k.unwrap_or([3, 3]);
        let e = e.unwrap_or(0.5);
        let c_ = (c2 as f32 * e).round() as usize; // hidden channels

        let conv1 = ConvBlock::new(
            c1,
            c_,
            k[0],
            1,
            None,
            None,
            None,
            device
        );
        let conv2 = ConvBlock::new(
            c_,
            c2,
            k[1],
            1,
            None,
            Some(g),
            None,
            device
        );
        let add = shortcut && c1 == c2;
        Self {
            conv1,
            conv2,
            add,
        }
    }
    /// Apply bottleneck with optional shortcut connection.
    ///
    /// # Arguments
    /// * `input`: Input tensor.
    ///
    /// # Returns
    /// * `Tensor<B, 4>`: Output tensor.
    ///
    /// `x + cv2(cv1(x))` if `(shortcut & c1 == c2)` else `cv2(cv1(x))`
    ///
    /// # Example
    /// ```
    /// let device = burn::device::Device::default();
    /// let bottleneck = Bottleneck::new([3, 3], [3, 3], 0.5, 1, true, &device);
    /// let input = Tensor::randn([1, 3, 26, 26], &device);
    /// let output = bottleneck.forward(input);
    /// assert_eq!(output.dims(), [1, 3, 26, 26]);
    /// ```
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.add {
            let x = self.conv1.forward(input.clone());
            let x = self.conv2.forward(x);
            x.add(input) //+ input
        } else {
            let x = self.conv1.forward(input);
            let x = self.conv2.forward(x);
            x
        }
    }
    
}
