use std::ops::Mul;

use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig, AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
    },
    prelude::*,
};

use crate::bottleneck::Bottleneck;
use crate::c2f::C2f;
use crate::conv::ConvBlock;
use crate::sppf::SPPF;

use burn::{
    backend::{Autodiff, wgpu::Wgpu},
    optim::AdamConfig,
    nn
};
use crate::head::Detect;


type MyBackend = Wgpu<f32, i32>;

#[derive(Module, Debug)]
pub struct Model<B: Backend = burn::backend::wgpu::Wgpu<f32, i32>> {
    // Backbone
    conv1: ConvBlock<B>, //1.  [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    conv2: ConvBlock<B>, //2.  [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    c2f1: C2f<B>,   //3.  [-1, 3, C2f, [128, True]]
    conv3: ConvBlock<B>, //4.  [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    c2f2: C2f<B>,   //5.  [-1, 6, C2f, [256, True]]
    conv4: ConvBlock<B>, //6.  [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    c2f3: C2f<B>,   //7.  [-1, 6, C2f, [512, True]]
    conv5: ConvBlock<B>, //8.  [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    c2f4: C2f<B>,   //9.  [-1, 3, C2f, [1024, True]]
    sppf: SPPF<B>,       //10. [-1, 1, SPPF, [1024, 5]] # 9
    // Head
    upsample1: Interpolate2d, //11. [-1, 1, nn.Upsample, [None, 2, "nearest"]] //12. [[-1, 6], 1, Concat, [1]] # cat backbone P4
    c2f5: C2f<B>, //13. [-1, 3, C2f, [512, 2, 3]] # 12
    upsample2: Interpolate2d, //14. [-1, 1, nn.Upsample, [None, 2, "nearest"]] //15. [[-1, 4], 1, Concat, [1]] # cat backbone P3
    c2f6: C2f<B>, //16. [-1, 3, C2f, [256]] # 15 (P3/8-small)
    conv6: ConvBlock<B>, //17. [-1, 1, Conv, [256, 3, 2]] # 17 //18. [[-1, 12], 1, Concat, [1]] # cat head P4
    c2f7: C2f<B>, //19. [-1, 3, C2f, [512]] # 18 (P4/16-medium)
    conv7: ConvBlock<B>, //20. [-1, 1, Conv, [512, 3, 2]] # 20 //21. [[-1, 9], 1, Concat, [1]] # cat head P5
    c2f8: C2f<B>, //22. [-1, 3, C2f, [1024]] # 21 (P5/32-large)
    pub detect: Detect<B>,
}


fn multw(a: i32) -> usize {
    (a as f32 * 0.25) as usize
}

fn multd(a: i32) -> usize {
    (a as f32 * 0.33) as usize
}

impl <B: Backend> Model<B> {
    /// Create a new Model module.
    ///
    /// # Arguments
    /// * `device`: Device to initialize the module on.
    ///
    pub fn new(device: &B::Device) -> Self {
        // Backbone
        let wscale = 0.25; // 0.25 for YOLOv8n
        let dscale = 0.33; // 0.33 for YOLOv8n
        let conv1 = ConvBlock::new(3, multw(64), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new(multw(64), multw(128), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f1 = C2f::new(multw(128), multw(128), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let conv3 = ConvBlock::new(multw(128), multw(256), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f2 = C2f::new(multw(256), multw(256), Some(multd(6)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let conv4 = ConvBlock::new(multw(256), multw(512), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f3 = C2f::new(multw(512), multw(512), Some(multd(6)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let conv5 = ConvBlock::new(multw(512), multw(1024), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f4 = C2f::new(multw(1024), multw(1024), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let sppf = SPPF::new(multw(1024), multw(1024), 5, device); // out: [Batch,8,26,26]
        // Head
        let upsample1 = Interpolate2dConfig::new().with_scale_factor(Some([2.0, 2.0])).with_mode(burn::nn::interpolate::InterpolateMode::Nearest).init(); // out: [Batch,8,26,26]
        let c2f5 = C2f::new(multw(1536), multw(512), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let upsample2 = Interpolate2dConfig::new().with_scale_factor(Some([2.0, 2.0])).with_mode(burn::nn::interpolate::InterpolateMode::Nearest).init(); // out: [Batch,8,26,26]
        let c2f6 = C2f::new(multw(768), multw(256), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        
        let conv6 = ConvBlock::new(multw(256), multw(256), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f7 = C2f::new(multw(256), multw(512), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        let conv7 = ConvBlock::new(multw(512), multw(512), 3, 2, None, None, None, device); // out: [Batch,8,26,26]
        let c2f8 = C2f::new(multw(512), multw(1024), Some(multd(3)), Some(true), None, None, device); // out: [Batch,8,26,26]
        
        let detect = Detect::new(80, [multw(256), multw(512), multw(1024)].to_vec(), device); // out: [Batch,8,26,26]
        Self {
            conv1,
            conv2,
            c2f1,
            conv3,
            c2f2,
            conv4,
            c2f3,
            conv5,
            c2f4,
            sppf,
            upsample1,
            c2f5,
            upsample2,
            c2f6,
            conv6,
            c2f7,
            conv7,
            c2f8,
            detect
        }

    }
    
}
impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, class_prob]
    pub fn forward(&self, images: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let x = images;
        // let t = std::time::Instant::now();
        let x = self.conv1.forward(x);
        // println!("[*] conv1: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.conv2.forward(x);
        // println!("[*] conv2: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.c2f1.forward(x);
        // println!("[*] c2f1: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x4 = self.conv3.forward(x);
        // println!("[*] conv3: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.c2f2.forward(x4.clone());
        // println!("[*] c2f2: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x6 = self.conv4.forward(x);
        // println!("[*] conv4: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.c2f3.forward(x6.clone());
        // println!("[*] c2f3: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.conv5.forward(x);
        // println!("[*] conv5: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.c2f4.forward(x);
        // println!("[*] c2f4: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.sppf.forward(x);
        // println!("[*] sppf: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.upsample1.forward(x);
        // println!("[*] upsample1: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = Tensor::cat([x, x6].to_vec(), 1); // cat backbone P4
        // println!("[*] cat: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.c2f5.forward(x);
        // println!("[*] c2f5: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.upsample2.forward(x);
        // println!("[*] upsample2: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = Tensor::cat([x, x4].to_vec(), 1); // cat backbone P3
        // println!("[*] cat: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x15 = self.c2f6.forward(x);
        // println!("[*] c2f6: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.conv6.forward(x15.clone());
        // println!("[*] conv6: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x18 = self.c2f7.forward(x);
        // println!("[*] c2f7: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.conv7.forward(x18.clone());
        // println!("[*] conv7: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x21 = self.c2f8.forward(x);
        // println!("[*] c2f8: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = [x15, x18, x21].to_vec();
        // println!("[*] to vec: {:?}", t.elapsed());
        // let t = std::time::Instant::now();
        let x = self.detect.forward(x); // cat head P4, P5, P6
        // println!("[*] detect: {:?}", t.elapsed());
        x
    }
}