use clap::Parser;
use std::path::Path;

use yolov8_rs::{Args, YOLOv8};
use yolov8_rs::utils::{get_all_files, is_valid_image};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse all args
    let args = Args::parse();
    // Define Path
    let path = Path::new(&args.source);
    // Check if the path is single file or dir
    let all_files = if path.exists() && path.is_file() {
        vec![args.source.clone()]
    } else {
        get_all_files(&path)?
    };

    // 1. build yolov8 model
    let mut model = YOLOv8::new(args)?;
    model.summary(); // model info

    for file_path in all_files.iter(){
        if is_valid_image(file_path) == false {
            println!("Invalid Image Path: {:?}", file_path);
            continue;
        }
        println!("Working on Image: {:?}", file_path);
        // 2. load image
        let x = image::ImageReader::open(&file_path)?
            .with_guessed_format()?
            .decode()?;

        // 3. model support dynamic batch inference, so input should be a Vec
        let xs = vec![x];

        // You can test `--batch 2` with this
        // let xs = vec![x.clone(), x];

        // 4. run
        let ys = model.run(&xs)?;
        println!("{:?}", ys);
    }

    Ok(())
}
