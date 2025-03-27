#![allow(clippy::type_complexity)]

mod cli;
mod model;
mod mapping;
mod helpers;
mod preprocess;
mod postprocess;

use crate::cli::Args;
use crate::helpers::draw_boxes;
use crate::preprocess::{Processor, PreprocessConfig};
use crate::mapping::load_class_mapping;
use crate::postprocess::{softmax_and_filter, non_maximum_suppression};
use crate::model::OnnxModel;


use std::error::Error;
use std::path::Path;
use clap::Parser;
use ndarray::{Array, CowArray, OwnedRepr};
use ndarray::{ArrayBase, IxDynImpl};

use tonic::transport::Server;

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mapping = load_class_mapping(r"assets\labels\coco-labels-91.txt")?;
    let processor = Processor::new(PreprocessConfig::default());

    let session = OnnxModel::new(args.cuda).load_model(&args.model)?;
    
    for _ in 0..1000{

        let original_img = image::open(&args.source).unwrap();
        let t = std::time::Instant::now();
        let (xs, offset) = processor.preprocess(&vec![original_img.clone()], args.deep_profile)?;
        let xs = CowArray::from(xs);
        let input_data = ort::inputs![xs.view()]?;
        if args.profile {
            println!("[preprocessing - *]: {:?}", t.elapsed());
        }
        
        let t = std::time::Instant::now();
        let ys = session.run(input_data)?;
        println!("[model]: {:?}", t.elapsed());
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
    }

    Ok(())
}
