use anyhow::Result;
use rayon::prelude::*;
use image::DynamicImage;
use fast_image_resize::images::Image;
use fast_image_resize::{IntoImageView, Resizer};

#[derive(Debug)]
pub struct PreprocessConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

#[derive(Debug)]
pub  struct Processor {
    pub config: PreprocessConfig,
}

impl PreprocessConfig {
    pub fn default() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            height: 560,
            width: 560,
            channels: 3,
        }
    }
}


impl Processor {
    /// Create a new instance of the Processor struct
    pub fn new(config: PreprocessConfig) -> Self {
        Self {
            config,
        }
    }
    pub fn convert_to_dynamic(&self, image: Image<'static>) -> DynamicImage {
        DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(image.width(), image.height(), image.buffer().to_vec())
                .expect("Failed to create ImageBuffer")
        )
    }
    /// Preprocess the input images
    /// Applying image normalization and resizing with padding
    /// Returns a tuple containing the preprocessed images and the offsets
    pub fn preprocess(&self, xs: &Vec<DynamicImage>, deep_profile: bool) -> Result<(ndarray::Array<f32, ndarray::IxDyn>, Vec<(u32, u32)>), Box<dyn std::error::Error>> {
        let ys_vec: Vec<(ndarray::Array<f32, ndarray::Dim<[usize; 3]>>, (u32, u32))> = xs.par_iter().enumerate().map(|(_, x)| {
            let t = std::time::Instant::now();
            let (orig_width, orig_height) = (x.width(), x.height());
            let scale = (self.config.width as f32 / orig_width as f32).min(self.config.height as f32 / orig_height as f32);
            let new_width = (orig_width as f32 * scale) as u32;
            let new_height = (orig_height as f32 * scale) as u32;
            if deep_profile{
                println!("[preprocessing - 1]: {:?}", t.elapsed());
            }
            let t = std::time::Instant::now();
            let mut dst_image = Image::new(
                new_width,
                new_height,
                x.pixel_type().unwrap(),
            );

            // Create Resizer instance and resize source image
            // into buffer of destination image
            let mut resizer = Resizer::new();
            let resize_options = fast_image_resize::ResizeOptions::new();
            resize_options.resize_alg(fast_image_resize::ResizeAlg::Nearest);
            resizer.resize(x, &mut dst_image, Some(&resize_options)).unwrap();
            let resized = self.convert_to_dynamic(dst_image).to_rgb8();
            if deep_profile{
                println!("[preprocessing - 2]: {:?}", t.elapsed());
            }
            let t = std::time::Instant::now();
            let mean_r = (self.config.mean[0] * 255.0) as u8;
            let mean_g = (self.config.mean[1] * 255.0) as u8;
            let mean_b = (self.config.mean[2] * 255.0) as u8;
            let mut padded = image::RgbImage::from_pixel(
                self.config.width as u32,
                self.config.height as u32,
                image::Rgb([mean_r, mean_g, mean_b])
            );
            if deep_profile{
                println!("[preprocessing - 3]: {:?}", t.elapsed());
            }
            let t = std::time::Instant::now();
            // Compute offsets to center the resized image in the padded image
            let x_offset = (self.config.width as u32 - new_width) / 2;
            let y_offset = (self.config.height as u32 - new_height) / 2;
            // Overlay the resized image onto the padded image at the calculated offsets
            image::imageops::overlay(&mut padded, &resized, x_offset as i64, y_offset as i64);
            if deep_profile{
                println!("[preprocessing - 4]: {:?}", t.elapsed());
            }
            let t = std::time::Instant::now();
            let img = DynamicImage::ImageRgb8(padded).to_rgb8();
            let pixels = img.pixels();

            let mut img_arr = ndarray::Array::from_elem((self.config.channels, self.config.height, self.config.width), 0 as f32); //144.0 / 255.0);
            if deep_profile {
                println!("[preprocessing - 5]: {:?}", t.elapsed());
            }
            let t = std::time::Instant::now();
            // Populate the array with normalized pixel values
            for (i, rgb) in pixels.enumerate() {
                let y = i / self.config.width as usize;
                let x = i % self.config.width as usize;
                img_arr[[0, y, x]] = (rgb[0] as f32 / 255.0 - self.config.mean[0]) / self.config.std[0];
                img_arr[[1, y, x]] = (rgb[1] as f32 / 255.0 - self.config.mean[1]) / self.config.std[1];
                img_arr[[2, y, x]] = (rgb[2] as f32 / 255.0 - self.config.mean[2]) / self.config.std[2];
            }
            if deep_profile {
                println!("[preprocessing - 6]: {:?}", t.elapsed());
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
}


