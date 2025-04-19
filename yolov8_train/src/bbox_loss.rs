use burn::{prelude::*, tensor::Tensor, nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig}};
use std::ops::{Mul, Sub, Add, Div};

// def bbox2dist(anchor_points, bbox, reg_max):
//     """Transform bbox(xyxy) to dist(ltrb)."""
//     x1y1, x2y2 = bbox.chunk(2, -1)
//     return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)

pub fn bbox2dist<B: Backend>(anchor_points: Tensor<B, 4>, bbox: Tensor<B, 4>, reg_max: usize) -> Tensor<B, 4> {
    let xyxy = bbox.chunk(2, 3);
    let x1y1 = &xyxy[0];
    let x2y2 = &xyxy[1];
    Tensor::cat([anchor_points.clone().sub(x1y1.clone()), x2y2.clone().sub(anchor_points)].to_vec(), 3).clamp(0.0, reg_max as f32 - 0.01)
}

