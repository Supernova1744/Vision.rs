# Vision.rs
A high-performance library for Computer Vision, leveraging Rust's speed and safety. Optionally supports a gRPC API for building scalable microservices, enabling seamless integration and remote inference.

## Run
```
cargo run --release -- --model assets\weights\yolov8n.onnx --source assets\data\input.jpg
```

You can add ``` --cuda --device_id <id>``` to use cuda