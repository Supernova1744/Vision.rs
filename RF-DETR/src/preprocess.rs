use anyhow::Result;
use rayon::prelude::*;
use image::{DynamicImage, GenericImageView};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const HEIGHT: usize = 560;
const WIDTH: usize = 560;
const CHANNELS: usize = 3;


pub fn preprocess(xs: &Vec<DynamicImage>) -> Result<(ndarray::Array<f32, ndarray::IxDyn>, Vec<(u32, u32)>), Box<dyn std::error::Error>> {
    let ys_vec: Vec<(ndarray::Array<f32, ndarray::Dim<[usize; 3]>>, (u32, u32))> = xs
        .par_iter()
        .enumerate()
        .map(|(_, x)| {
            let (orig_width, orig_height) = (x.width(), x.height());
            // Determine scale to maintain aspect ratio
            let scale = (WIDTH as f32 / orig_width as f32).min(HEIGHT as f32 / orig_height as f32);
            let new_width = (orig_width as f32 * scale) as u32;
            let new_height = (orig_height as f32 * scale) as u32;

            let resized = x.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
            let resized = resized.to_rgb8();

            let mean_r = (MEAN[0] * 255.0) as u8;
            let mean_g = (MEAN[1] * 255.0) as u8;
            let mean_b = (MEAN[2] * 255.0) as u8;
            let mut padded = image::RgbImage::from_pixel(
                WIDTH as u32,
                HEIGHT as u32,
                image::Rgb([mean_r, mean_g, mean_b])
            );
            // Compute offsets to center the resized image in the padded image
            let x_offset = (WIDTH as u32 - new_width) / 2;
            let y_offset = (HEIGHT as u32 - new_height) / 2;

            // Overlay the resized image onto the padded image at the calculated offsets
            image::imageops::overlay(&mut padded, &resized, x_offset as i64, y_offset as i64);

            let img = DynamicImage::ImageRgb8(padded).to_rgb8();
            let pixels = img.pixels();

            let mut img_arr = ndarray::Array::from_elem((CHANNELS, HEIGHT, WIDTH), 0 as f32); //144.0 / 255.0);

            // Populate the array with normalized pixel values
            for (i, rgb) in pixels.enumerate() {
                let y = i / WIDTH as usize;
                let x = i % WIDTH as usize;
                img_arr[[0, y, x]] = (rgb[0] as f32 / 255.0 - MEAN[0]) / STD[0];
                img_arr[[1, y, x]] = (rgb[1] as f32 / 255.0 - MEAN[1]) / STD[1];
                img_arr[[2, y, x]] = (rgb[2] as f32 / 255.0 - MEAN[2]) / STD[2];
            }
            (img_arr, (x_offset, y_offset))
        })
        .collect();

    // Separate the image arrays and the offsets
    let (img_arrs, offsets): (Vec<_>, Vec<_>) = ys_vec.into_iter().unzip();
    let views: Vec<_> = img_arrs.iter().map(|arr| arr.view()).collect();
    let ys = ndarray::stack(ndarray::Axis(0), &views)?.into_dyn();
    
    Ok((ys, offsets))
}

