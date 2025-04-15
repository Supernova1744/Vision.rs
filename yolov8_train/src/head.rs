use burn::{
    nn::{conv::Conv2d, BatchNorm, PaddingConfig2d},
    prelude::*, tensor
};
use crate::conv::{DWConv, ConvBlock};
use crate::dfl::DFL;
use num_integer::Integer;



// #[derive(Module, Debug)]
pub struct Detect<B: Backend> {
    cv2_seq1: Vec<ConvBlock<B>>,
    cv2_seq2: Vec<ConvBlock<B>>,
    cv2_seq3: Vec<Conv2d<B>>,
    cv3_seq1_b1: Vec<ConvBlock<B>>,
    cv3_seq1_b2: Vec<ConvBlock<B>>,
    cv3_seq2_b1: Vec<ConvBlock<B>>,
    cv3_seq2_b2: Vec<ConvBlock<B>>,
    cv3_seq3_b1: Vec<Conv2d<B>>,
    dfl: DFL<B>,
    stride: Tensor<B, 1>,
    strides: Tensor<B, 3>,
    anchors: Tensor<B, 3>,
    shape: Option<Tensor<B, 1>>,
    pub reg_max: usize,
    no: usize,
    nl: usize,
    pub nc: usize,
    end2end: bool,
    export: bool,
    max_det: usize,
    legacy: bool,
    dynamic: bool,
    training: bool
}

impl <B: Backend> Detect<B> {
    pub fn new(nc: usize, ch: Vec<usize>, device: &B::Device) -> Self {
        let nl = ch.len();
        let reg_max = 16;
        let no = nc + reg_max * 4;
        let c2 = ch[0].max(16).max(ch[0] / 4).max(reg_max * 4);
        let c3 = ch[0].max(nc).min(100);
        let anchors = Tensor::<B, 3>::empty([1,1,1], device);
        let strides = Tensor::<B, 3>::empty([1,1,1], device);

        let cv2_seq1 = (0..nl).map(|i|
        ConvBlock::<B>::new(ch[i], c2, 3, 1, None, None, None, device)
        ).collect::<Vec<ConvBlock<B>>>();

        let cv2_seq2 = (0..nl).map(|i|
            ConvBlock::<B>::new(c2, c2, 3, 1, None, None, None, device)
            ).collect::<Vec<ConvBlock<B>>>();

        let cv2_seq3 = (0..nl).map(|i|
            nn::conv::Conv2dConfig::new([c2, 4 * reg_max], [1, 1]).with_bias(true).init(device)
        ).collect::<Vec<Conv2d<B>>>();


        let cv3_seq1_b1 = (0..nl).map(|i|
            // DWConv::<B>::new(ch[i], ch[i], 3, 1, None, device)
            ConvBlock::<B>::new(ch[i], ch[i], 3, 1, None, Some(ch[i].gcd(&(ch[i] as usize))), None, device)
            
        ).collect::<Vec<ConvBlock<B>>>();
        
        let cv3_seq1_b2 = (0..nl).map(|i|
            ConvBlock::<B>::new(ch[i], c3, 1, 1, None, None, None, device)
        ).collect::<Vec<ConvBlock<B>>>();
        
        let cv3_seq2_b1 = (0..nl).map(|i|
            // DWConv::<B>::new(c3, c3, 3, 1, None, device)
            ConvBlock::<B>::new(c3, c3, 3, 1, None, Some(c3.gcd(&(c3 as usize))), None, device)
        ).collect::<Vec<ConvBlock<B>>>();

        let cv3_seq2_b2 = (0..nl).map(|i|
            ConvBlock::<B>::new(c3, c3, 1, 1, None, None, None, device)
        ).collect::<Vec<ConvBlock<B>>>();

        let cv3_seq3_b1: Vec<Conv2d<B>> = (0..nl).map(|i|
            nn::conv::Conv2dConfig::new([c3, nc], [1, 1]).with_bias(true).init(device)
        ).collect::<Vec<Conv2d<B>>>();
        
        let stride = Tensor::<B, 1>::zeros([nl], device);

        let dfl = DFL::<B>::new(reg_max, device);

        Self {
            cv2_seq1,
            cv2_seq2,
            cv2_seq3,
            cv3_seq1_b1,
            cv3_seq1_b2,
            cv3_seq2_b1,
            cv3_seq2_b2,
            cv3_seq3_b1,
            dfl,
            stride,
            strides,
            anchors,
            shape: None,
            reg_max,
            no,
            nl,
            nc,
            end2end: false,
            export: false,
            max_det: 300,
            legacy: false,
            dynamic: false,
            training: false,
        }
    }

    

    pub fn forward(&self, x: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        // make a mutable copy
        // let x = x.clone();
        let mut out = Vec::with_capacity(self.nl);
        // do each layer i in parallel
        x.iter()
         .enumerate()
         .for_each(|(i, xi)| {
            // let t = std::time::Instant::now();
            let y1 = self.cv2_seq3[i].forward(
                self.cv2_seq2[i].forward(
                    self.cv2_seq1[i].forward(xi.clone())
                )
             );
            // println!("[*] cv2: {:?}", t.elapsed());
            // let t = std::time::Instant::now();
            let y2 = self.cv3_seq3_b1[i].forward(
                self.cv3_seq2_b2[i].forward(
                    self.cv3_seq2_b1[i].forward(
                        self.cv3_seq1_b2[i].forward(
                            self.cv3_seq1_b1[i].forward(xi.clone())
                        )
                    )
                )
             );
            //  println!("[*] cv3: {:?}", t.elapsed());
    
             // stitch them back together
             out.push(Tensor::<B, 4>::cat(vec![y1, y2], 1));
         });
    
        out
    }

    fn dist2bbox(&self, distance: Tensor<B, 3>, anchor_points: Tensor<B, 3>, xywh: bool, dim: usize) -> Tensor<B, 3>
    {   
        let chunks = distance.chunk(2, dim);
        let lt = chunks[0].clone();
        let rb = chunks[1].clone();
        let x1y1 = anchor_points.clone().sub(lt);
        let x2y2 = anchor_points.sub(rb);
        if xywh {
            let c_xy = (x1y1.clone().add(x2y2.clone())).div_scalar(2);
            let wh = x2y2.sub(x1y1);
            Tensor::cat([c_xy, wh].to_vec(), dim)
        }
        else{
            Tensor::cat([x1y1, x2y2].to_vec(), dim)
        }
    }
    
    
    
}



