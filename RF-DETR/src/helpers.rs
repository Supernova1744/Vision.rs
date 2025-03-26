use anyhow::Result;
use image::Rgba;
use ndarray::Array1;

use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};


const WIDTH: u32 = 560;
const HEIGHT: u32 = 560;


pub fn draw_boxes(
    img_path: &str,
    boxes: Vec<Array1<f32>>,
    offsets: &[(u32, u32)],
    output_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    // Load original image
    let mut img = image::open(img_path)?.to_rgba8();
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    
    // Get preprocessing parameters
    let (x_off, y_off) = offsets[0];
    let scale = (WIDTH as f32 / orig_w).min(HEIGHT as f32 / orig_h);

    for box_ in boxes {
        // Denormalize to padded image coordinates
        let x_pad = box_[0] * WIDTH as f32;
        let y_pad = box_[1] * HEIGHT as f32;
        let w_pad = box_[2] * WIDTH as f32;
        let h_pad = box_[3] * HEIGHT as f32;

        // Adjust for padding offset
        let x_resized = x_pad - x_off as f32;
        let y_resized = y_pad - y_off as f32;

        // Convert to original image coordinates
        let x_center = (x_resized / scale).round() as i32;
        let y_center = (y_resized / scale).round() as i32;
        let width = (w_pad / scale).round() as i32;
        let height = (h_pad / scale).round() as i32;

        // Calculate bounding box coordinates
        let left = x_center - width / 2;
        let top = y_center - height / 2;
        let rect = Rect::at(left, top).of_size(width as u32, height as u32);

        // Draw red rectangle with 2px thickness
        draw_hollow_rect_mut(&mut img, rect, Rgba([255, 0, 0, 255]));
    }

    img.save(output_path)?;
    Ok(())
}
