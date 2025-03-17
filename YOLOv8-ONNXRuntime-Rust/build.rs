use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = Path::new("proto");
    let proto_file = proto_dir.join("result.proto");
    tonic_build::compile_protos(proto_file)?;
    Ok(())
}