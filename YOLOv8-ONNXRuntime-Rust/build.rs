use std::env;
use std::path::Path;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("PROTOC", r"C:\protoc-30.1-win64\bin\protoc.exe");
    let proto_dir = Path::new("proto");
    let proto_file = proto_dir.join("result.proto");
    tonic_build::compile_protos(proto_file).map_err(|e| {
        eprintln!("Failed to compile protos: {}", e);
        e
    })?;
    Ok(())
}