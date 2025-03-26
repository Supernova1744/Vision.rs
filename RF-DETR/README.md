# Vision.rs - ONNX Inference in Rust with ORT

Vision.rs is a Rust-based inference engine for vision models using the ONNX Runtime (ORT). This repository provides a streamlined way to run vision models efficiently on CPU and CUDA.

## Features
- Run ONNX-based vision models with Rust.
- Support for CPU and CUDA execution.

## Getting Started
### Prerequisites
Ensure you have Rust and Cargo installed:
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```


### Installing
Clone the repository:
```sh
$ git clone https://github.com/yourusername/Vision.rs.git
$ cd Vision.rs
```

### Running Inference
Use the following command to run inference with the RF-DETR model:
```sh
$ cd RF-DETR
$ cargo run --release -- --model ./assets/weights/inference_model.onnx --source ./assets/data/input.jpg --cuda
```

### Obtaining Model Weights
To get the RF-DETR model weights, install the `rfdetr` Python package and export the model:
```sh
$ pip install rfdetr
```
Then, in a Python shell:
```python
>>> from rfdetr import RFDETRBase
>>> model = RFDETRBase()
>>> model.export()
```
This will generate an ONNX model that can be used for inference.


## File Structure
```
Vision.rs/
├── src/
│   ├── main.rs
│   └── ...
├── assets/
│   ├── weights/
│   │   └── inference_model.onnx
│   ├── data/
│   │   └── input.jpg
├── Cargo.toml
├── README.md
└── .gitignore
```


## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Contact
For any questions or issues, reach out via GitHub Issues.

