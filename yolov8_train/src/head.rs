use burn::{
    prelude::nn::conv::{Conv2d, Conv2dConfig},
    prelude::*
};
use crate::conv::ConvBlock;
use crate::dfl::DFL;
use num_integer::Integer;

enum Block<B: Backend> {
    ConvBlock(ConvBlock<B>),
    Conv2d(Conv2d<B>)
}

impl <B: Backend> Block<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Block::ConvBlock(conv_block) => conv_block.forward(input),
            Block::Conv2d(conv2d) => conv2d.forward(input)
        }
    }
    
}

pub struct Detect<B: Backend> {
    cv2: Vec<Vec<Block<B>>>,
    cv3: Vec<Vec<Block<B>>>,
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

        let mut cv2 = Vec::<Vec<Block<B>>>::with_capacity(nl);
        let mut cv3 = Vec::<Vec<Block<B>>>::with_capacity(nl);

        for i in 0..nl {
            cv2.push(vec![
                Block::ConvBlock(ConvBlock::<B>::new(ch[i], c2, 3, 1, None, None, None, device)),
                Block::ConvBlock(ConvBlock::<B>::new(c2, c2, 3, 1, None, None, None, device)),
                Block::Conv2d(Conv2dConfig::new([c2, 4 * reg_max], [1, 1]).with_bias(true).init(device))
            ]);
            
            cv3.push(vec![
                Block::ConvBlock(ConvBlock::<B>::new(ch[i], ch[i], 3, 1, None, Some(ch[i].gcd(&(ch[i] as usize))), None, device)),
                Block::ConvBlock(ConvBlock::<B>::new(ch[i], c3, 1, 1, None, None, None, device)),
                Block::ConvBlock(ConvBlock::<B>::new(c3, c3, 3, 1, None, Some(c3.gcd(&(c3 as usize))), None, device)),
                Block::ConvBlock(ConvBlock::<B>::new(c3, c3, 1, 1, None, None, None, device)),
                Block::Conv2d(Conv2dConfig::new([c3, nc], [1, 1]).with_bias(true).init(device))
            ]);
                    
        }
      
        
        let stride = Tensor::<B, 1>::zeros([nl], device);

        let dfl = DFL::<B>::new(reg_max, device);

        Self {
            cv2,
            cv3,
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
        let mut out = Vec::with_capacity(self.nl);
        for (i, xi) in x.into_iter().enumerate() {
            
            let y1 = self.cv2[i]
                .iter()
                .fold(xi.clone(), |acc, layer| layer.forward(acc));
            
            let y2 = self.cv3[i]
                .iter()
                .fold(xi, |acc, layer| layer.forward(acc));
            
            out.push(Tensor::cat(vec![y1, y2], 1));
        }
        out
    }

    fn dist2bbox(&self, distance: Tensor<B, 3>, anchor_points: Tensor<B, 3>, xywh: bool, dim: usize) -> Tensor<B, 3>{   
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
    
    fn decode_bboxes(&self, bboxes: Tensor<B, 3>, anchors: Tensor<B, 3>, xywh: bool) -> Tensor<B, 3>{
        self.dist2bbox(bboxes, anchors, xywh, 1)
    }

    pub fn inference(&self, x: Vec<Tensor<B, 4>>) -> Tensor<B, 3>{
        let shape = x[0].clone().dims();
        let mut reshaped_x: Vec<Tensor<B, 3>> = vec![];
        x.iter().for_each(|xi| {
            let reshaped_xi = xi.clone().reshape(
                [shape[0] as i32, self.no as i32, -1]
            );
            reshaped_x.push(reshaped_xi);
        });
        let x_cat = Tensor::cat(reshaped_x, 2);
        let splits = x_cat.split_with_sizes([self.reg_max * 4, self.nc].to_vec(), 1);
        let bbox = splits[0].clone();
        let cls = splits[1].clone();
        let dfl_box = self.dfl.forward(bbox);
        let dbox = self.decode_bboxes(dfl_box, self.anchors.clone(), true).mul(self.strides.clone());
        let output = Tensor::cat([dbox, nn::Sigmoid.forward(cls)].to_vec(), 1);
        output
        
    }
    // Convert the following python function to rust
    // def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
    //     """
    //     Post-processes YOLO model predictions.

    //     Args:
    //         preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
    //             format [x, y, w, h, class_probs].
    //         max_det (int): Maximum detections per image.
    //         nc (int, optional): Number of classes. Default: 80.

    //     Returns:
    //         (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
    //             dimension format [x, y, w, h, max_class_prob, class_index].
    //     """
    //     batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)P
    //     boxes, scores = preds.split([4, nc], dim=-1)
    //     index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    //     boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    //     scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
    //     scores, index = scores.flatten(1).topk(min(max_det, anchors))
    //     i = torch.arange(batch_size)[..., None]  # batch indices
    //     return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

    // fn postprocess(&self, preds: Tensor<B, 3>, max_det: usize, nc: usize) -> Tensor<B, 4> {
    //     let [batch_size, anchors, _] = preds.dims();
    //     let splits = preds.split_with_sizes([4, nc].to_vec(), 2);
    //     let (boxes, scores) = (splits[0], splits[1]);
    //     let index = scores.max_dim(2).topk(max_det.min(anchors), 2).slice([1..2, 0..batch_size as usize, 0..anchors as usize]).unsqueeze();
    //     let x = index.clone().repeat(&[1, 1, nc]);
    //     let boxes = boxes.gather(2, index.int().repeat(&[1, 1, 4]));
    //     let scores = scores.gather(2, index.int().repeat(&[1, 1, nc]));
    //     let scores = scores.flatten(1,1).topk(max_det.min(anchors), 0).slice([0..batch_size as usize, 0..max_det as usize]).unsqueeze();
    //     let i = Tensor::<B, 1, Int>::arange(0..batch_size as i64, self.stride.device()).reshape([batch_size as usize, 1]);
    //     Tensor::<B, 4>::cat(vec![boxes.gather(0, i), scores.unsqueeze(), (index.float() % nc).unsqueeze()], -1)
    // }
        
    
    
}



