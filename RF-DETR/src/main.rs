mod helpers;
mod preprocess;
mod postprocess;
mod cli;
mod mapping;

use crate::cli::Args;
use crate::helpers::draw_boxes;
use crate::preprocess::preprocess;
use crate::mapping::load_class_mapping;
use crate::postprocess::{softmax_and_filter, non_maximum_suppression};

use std::error::Error;
use std::path::Path;
use clap::Parser;
use ndarray::{Array, CowArray, OwnedRepr};
use ort::session::builder::SessionBuilder;
use ndarray::{ArrayBase, IxDynImpl};


fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mapping = load_class_mapping(r"assets\labels\coco-labels-91.txt")?;

    let provider = if args.cuda {
        [ort::execution_providers::CUDAExecutionProvider::default().build().error_on_failure()]
    } else {
        [ort::execution_providers::CPUExecutionProvider::default().build()]
    };

    print!("Loading model with {:?} provider...", provider);
    
    let sessionbuilder = SessionBuilder::new()?.with_execution_providers(provider)?;
    let session = sessionbuilder.commit_from_file(args.model)?;

    let original_img = image::open(&args.source).unwrap();
    let (xs, offset) = preprocess(&vec![original_img.clone()])?;
    
    let xs = CowArray::from(xs);
    let input_data = ort::inputs![xs.view()]?;
    let ys = session.run(input_data)?;
    let i = ys
        .iter()
        .map(|(_k, v)| {
            let v = v.try_extract_tensor::<f32>().unwrap().into_owned();
            v
        })
        .collect::<Vec<Array<_, _>>>();
    let boxes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &i[0];

    let classes: &ArrayBase<OwnedRepr<f32>, ndarray::Dim<IxDynImpl>> = &i[1];
    let (filtered_classes, filtered_conf, filtered_boxes) = softmax_and_filter(classes, boxes, 0.5)?;
    let (filtered_conf, filtered_classes, filtered_boxes) = non_maximum_suppression(filtered_conf, filtered_classes, filtered_boxes, 0.5);
    draw_boxes(original_img.clone(), filtered_boxes, &offset, Path::new(&args.output).join("output.png").to_str().unwrap())?;
    
    println!("Detections:");
    for (class, conf) in filtered_classes.iter().zip(filtered_conf.iter()) {
        match mapping.get(&(*class as usize)) {
            Some(class) => println!("Class: {}, Confidence: {:.2}%", class, conf * 100.0),
            None => println!("Class: {}, Confidence: {:.2}%", class, conf * 100.0),
        }
    }

    Ok(())
}
