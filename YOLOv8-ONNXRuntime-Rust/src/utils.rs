use std::fs;
use std::io;
use std::path::Path;

pub fn get_all_files(dir: &Path) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                files.push(path.to_string_lossy().into_owned());
            }
        }
    }
    Ok(files)
}

pub fn is_valid_image(file_path: &String) -> bool {
    let valid_exts = ["jpg", "png", "jpeg"];
    let path = Path::new(file_path);

    // Check that the extension is valid.
    let ext_valid = match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => valid_exts.contains(&ext.to_lowercase().as_str()),
        None => false,
    };

    // Check that the file size is not zero.
    let size_valid = match fs::metadata(path) {
        Ok(metadata) => metadata.len() != 0,
        Err(_) => false,
    };

    ext_valid && size_valid
}