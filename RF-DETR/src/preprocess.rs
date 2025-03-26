use anyhow::Result;
use rayon::prelude::*;
use image::DynamicImage;
use ndarray::{Array, IxDyn};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const HEIGHT: usize = 560;
const WIDTH: usize = 560;
const CHANNELS: usize = 3;

pub fn preprocess(xs: &Vec<DynamicImage>) -> Result<Array<f32, IxDyn>> {
    let ys_vec: Vec<_> = xs.into_par_iter().enumerate().map(|(_, x)| {
        let img = x.resize_exact(WIDTH as u32, HEIGHT as u32, image::imageops::FilterType::Lanczos3);
        let img = img.to_rgb8();
        let pixels = img.pixels();
        
        let mut img_arr = Array::from_elem((CHANNELS, HEIGHT, WIDTH), 144.0 / 255.0);
        
        for (i, rgb) in pixels.enumerate() {
            let y = i / WIDTH as usize;
            let x = i % WIDTH as usize;
            img_arr[[0, y, x]] = (rgb[0] as f32 / 255.0 - MEAN[0]) / STD[0];
            img_arr[[1, y, x]] = (rgb[1] as f32 / 255.0 - MEAN[1]) / STD[1];
            img_arr[[2, y, x]] = (rgb[2] as f32 / 255.0 - MEAN[2]) / STD[2];
        }
        img_arr
    }).collect();

    let views: Vec<_> = ys_vec.iter().map(|arr| arr.view()).collect();
    let ys = ndarray::stack(ndarray::Axis(0), &views).unwrap().into_dyn();
    Ok(ys)
}
