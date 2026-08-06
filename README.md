# Fast_Plate_OCR_Model_Evaluation

This notebooks demonstrates the complete inference pipeline for Fast Plate OCR models, including model conversion from Keras to ONNX (.onnx) and TensorRT (.engine), followed by inference and performance benchmarking across all three model formats.

# Strcture
```bash
├── Engine_file_inference.ipynb
├── onnx_inference.ipynb
├── models
├── README.md
├── requirements.txt
└── test_images
```

# 🛠️ Environment Setup (Docker)

Follow the steps below to set up the environment and run the evaluation notebooks.

## Step 1: Clone the Repository

Clone the repository and navigate to the project directory.

```bash
git clone -b Development_Branch  https://github.com/dhairyashil1012-ease/Fast_Plate_OCR_Model_Evaluation.git

cd Fast_Plate_OCR_Model_Evaluation
```

---

## Step 2: Download the TensorRT Docker Image

TensorRT Docker images are available from the NVIDIA NGC Catalog:

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags

Pull the required TensorRT image (replace the tag with the version you want to use).

```bash

docker pull nvcr.io/nvidia/tensorrt:26.04-py3

```

---

## Step 3:  Sample Input Images

You can use your own images for evaluation or Sample Images are present in test_images directory:


After downloading, extract the images into the appropriate evaluation directory corresponding to your use case.

---

## Step 4: Launch the TensorRT Docker Container

Mount the project directory inside the container and start an interactive TensorRT environment.

```bash
docker run -it \
    --gpus all \
    -p 8888:8888 \
    -v `pwd`:/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:26.06-py3 \
    bash
```

---

## Step 5: Install Python Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

---

## Step 6: Install System Dependencies

Install the required system libraries.

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6  libxext6  libxrender1 libxcb1
```

---

## Step 7: Start Jupyter Notebook

Launch the Jupyter Notebook server inside the Docker container.

```bash
jupyter notebook \
    --port=8888 \
    --no-browser \
    --ip=0.0.0.0 \
    --allow-root
```

Copy the generated URL from the terminal and open it in your web browser.

---

## Re-entering an Active Container:
```Bash

docker exec -it <container_id> bash
```

# 🚀 Model Conversion & Compilation Pipeline

1. Keras to ONNX Export

Export your trained .keras file to ONNX format using your preferred conversion script. Ensure the input layer tensor is explicitly mapped 
Use Fast_Plate_Ocr_ONNX_Model_Evaluation.ipynb the conversion code is present in file just update your keras file name and execute that code that automatically store the onnx model (e.g.,input_layer).

2. Compile TensorRT Engine from ONNX

Navigate to the models directory and run trtexec to build an optimized engine with dynamic batch shapes which is below:


## Create Engine model from ONNX model 
```bash
cd /workspace/models

trtexec --onnx=<onnx_model_name.onnx> \
  --saveEngine=<example.engine> \
  --minShapes=input_layer:1x64x128x3 \
  --optShapes=input_layer:32x64x128x3 \
  --maxShapes=input_layer:64x64x128x3
```
