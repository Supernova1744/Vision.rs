use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_class_mapping(file_path: &str) -> Result<HashMap<usize, String>, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    
    let mapping: HashMap<usize, String> = reader
        .lines()
        .enumerate()
        .filter_map(|(id, line)| {
            line.ok().map(|name| {
                (id + 1, name.trim().to_string()) // Trim whitespace and convert to String
            })
        })
        .collect();

    Ok(mapping)
}