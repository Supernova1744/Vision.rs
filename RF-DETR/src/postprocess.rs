use anyhow::Result;
use tonic::Status;
use std::{collections::HashMap, error::Error};
use ndarray::{Array, Array1, Array3, ArrayBase, Axis, IxDynImpl, OwnedRepr};
use ordered_float::OrderedFloat;
use crate::cli::Args;

#[derive(Debug)]
pub struct PostProcessor {
    pub config: Args
}

impl PostProcessor {
    pub fn new(config: Args) -> Self {
        Self {
            config
        }
    }
    /// Applies softmax to a 1D array (slice) and returns a new Array1<f32>.
    pub fn softmax(&self, slice: &Array1<f32>) -> Array1<f32> {
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Array1<f32> = slice.mapv(|x| (x - max_val).exp());
        let sum_exp: f32 = exp_vals.sum();
        exp_vals.mapv(|v| v / sum_exp)
    }
    /// Computes the argmax and max value of a softmaxed array.
    /// Returns a tuple of (index, value).
    pub fn argmax_and_max(&self, softmaxed: &Array1<f32>) -> (usize, f32) {
        softmaxed
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (i, &val)| {
                if val > max_val { (i, val) } else { (max_idx, max_val) }
            })
    
    }
    pub fn softmax_and_filter(
        &self,
        classes_dyn: &Array<f32, ndarray::IxDyn>,
        boxes_dyn: &Array<f32, ndarray::IxDyn>,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<Array1<f32>>), Box<dyn Error>> {
        // Reshape the dynamic arrays to fixed dimensions.
        // We assume the shape is (1, num_boxes, num_classes) for classes and (1, num_boxes, 4) for boxes.
        let classes_fixed: Array3<f32> = classes_dyn
            .view()
            .into_dimensionality::<ndarray::Ix3>()?
            .into_dimensionality::<ndarray::Ix3>()?.to_owned();
        
            let boxes_fixed: Array3<f32> = boxes_dyn.view()
            .into_dimensionality::<ndarray::Ix3>()?
            .to_owned();
        
        // Get the 2D views from axis 0 (since shape is (1, 300, ...)).
        let classes_2d = classes_fixed.index_axis(Axis(0), 0); // shape (300, 91)
        let boxes_2d = boxes_fixed.index_axis(Axis(0), 0);       // shape (300, 4)
        
        let mut filtered_classes = Vec::new();
        let mut filtered_conf = Vec::new();
        let mut filtered_boxes = Vec::new();
    
        // Iterate over the 300 boxes (each row in axis 0).
        for (class_row, box_row) in classes_2d.axis_iter(Axis(0)).zip(boxes_2d.axis_iter(Axis(0))) {
            // Convert the class row to an Array1<f32>
            let class_row_vec = class_row.to_owned();
            // Apply softmax on the 91 logits.
            let softmaxed = self.softmax(&class_row_vec);
            // Compute confidence as the maximum probability.
            let confidence = softmaxed.iter().cloned().fold(0.0, f32::max);
            if confidence >= self.config.conf_th {
                let ids_and_classes = self.argmax_and_max(&softmaxed);
                filtered_classes.push(ids_and_classes.0 as i32);
                filtered_conf.push(ids_and_classes.1 as f32);
                // Also push the corresponding box. Convert the 1D view to an owned Array1.
                filtered_boxes.push(box_row.to_owned());
            }
        }
        Ok((filtered_classes, filtered_conf, filtered_boxes))
    }
    
    /// Compute the Intersection over Union (IoU) of two boxes in center-format.
    pub fn compute_iou(&self, b1: &Array1<f32>, b2: &Array1<f32>) -> f32 {
        let (x1_1, y1_1, x2_1, y2_1) = (b1[0], b1[1], b1[2], b1[3]);
        let (x1_2, y1_2, x2_2, y2_2) = (b2[0], b2[1], b2[2], b2[3]);
    
        let inter_x1 = x1_1.max(x1_2);
        let inter_y1 = y1_1.max(y1_2);
        let inter_x2 = x2_1.min(x2_2);
        let inter_y2 = y2_1.min(y2_2);
    
        let inter_area = ((inter_x2 - inter_x1).max(0.0)) * ((inter_y2 - inter_y1).max(0.0));
        let area1 = (x2_1 - x1_1).max(0.0) * (y2_1 - y1_1).max(0.0);
        let area2 = (x2_2 - x1_2).max(0.0) * (y2_2 - y1_2).max(0.0);
        let union_area = area1 + area2 - inter_area;
        if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
    }
    pub fn non_maximum_suppression(
        &self,
        class_confs: Vec<f32>,
        class_ids: Vec<i32>,
        boxes: Vec<Array1<f32>>
    ) -> (Vec<f32>, Vec<i32>, Vec<Array1<f32>>) {
        // Group indices by class id.
        let mut by_class: HashMap<OrderedFloat<f32>, Vec<usize>> = HashMap::new();
        for (i, &cid) in class_ids.iter().enumerate() {
            by_class.entry(OrderedFloat(cid as f32)).or_default().push(i);
        }
    
        let mut keep_indices: Vec<usize> = Vec::new();
    
        // Process each class group separately.
        for (_cid, indices) in by_class.iter_mut() {
            // Sort the indices in descending order of confidence.
            indices.sort_by(|&i1, &i2| {
                // Convert confidence to f32 for sorting.
                (class_confs[i2])
                    .partial_cmp(&(class_confs[i1]))
                    .unwrap()
            });
    
            // Standard NMS: iterate and suppress boxes with high overlap.
            let mut suppressed = vec![false; indices.len()];
            for i in 0..indices.len() {
                if suppressed[i] {
                    continue;
                }
                let idx_i = indices[i];
                keep_indices.push(idx_i);
                for j in (i + 1)..indices.len() {
                    if suppressed[j] {
                        continue;
                    }
                    let idx_j = indices[j];
                    let iou = self.compute_iou(&boxes[idx_i], &boxes[idx_j]);
                    if iou > self.config.iou_th {
                        suppressed[j] = true;
                    }
                }
            }
        }
    
        // Optionally, sort the kept indices in ascending order.
        keep_indices.sort_unstable();
    
        let filtered_confs: Vec<f32> = keep_indices.iter().map(|&i| class_confs[i]).collect();
        let filtered_ids: Vec<i32> = keep_indices.iter().map(|&i| class_ids[i]).collect();
        let filtered_boxes: Vec<Array1<f32>> = keep_indices.iter().map(|&i| boxes[i].clone()).collect();
    
        (filtered_confs, filtered_ids, filtered_boxes)
    }

    pub fn denormalize(
        &self,
        orig_w: f32,
        orig_h: f32,
        x_off: u32,
        y_off: u32,
        boxes: Vec<Array1<f32>>,
    ) -> Vec<[i32; 4]>{

        let mut output = Vec::new();
        let scale = (self.config.img_w as f32 / orig_w).min(self.config.img_h as f32 / orig_h);
    
        for box_ in boxes {
            // Denormalize to padded image coordinates
            let x_pad = box_[0] * self.config.img_w as f32;
            let y_pad = box_[1] * self.config.img_h as f32;
            let w_pad = box_[2] * self.config.img_w as f32;
            let h_pad = box_[3] * self.config.img_h as f32;
    
            // Adjust for padding offset
            let x_resized = x_pad - x_off as f32;
            let y_resized = y_pad - y_off as f32;
    
            // Convert to original image coordinates
            let x_center = (x_resized / scale).round() as i32;
            let y_center = (y_resized / scale).round() as i32;
            let width = (w_pad / scale).round() as i32;
            let height = (h_pad / scale).round() as i32;
            output.push([x_center, y_center, width, height]);
        }
        output
    }

    pub fn postprocess(&self, model_output: Vec<ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>>>, orig_w: f32, orig_h: f32, offset: Vec<(u32, u32)>) -> Result<(Vec<[i32; 4]>, Vec<i32>, Vec<f32>), Box<dyn Error>> {
        let boxes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &model_output[0];
        let classes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &model_output[1];
        let (filtered_classes, filtered_conf, filtered_boxes) = self.softmax_and_filter(classes, boxes)
            .map_err(|e| Status::internal(format!("Softmax and filter error: {}", e)))?;
        let (filtered_conf, filtered_classes, filtered_boxes) = self.non_maximum_suppression(filtered_conf, filtered_classes, filtered_boxes);
        
        let filtered_boxes = self.denormalize(
            orig_w, orig_h,
            offset[0].0, offset[0].1, filtered_boxes);
        Ok((filtered_boxes, filtered_classes, filtered_conf))
    }

}




