# YOLOv8 Quantization-Aware Training Tutorial

This tutorial provides a comprehensive guide on using our enhanced `train_qat.py` script to perform Quantization-Aware Training (QAT) on YOLOv8 models. QAT integrates quantization effects during training, allowing models to adapt and maintain accuracy when deployed in int8 precision.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Basic Usage](#3-basic-usage)
4. [Training Workflows](#4-training-workflows)
5. [Command-Line Arguments](#5-command-line-arguments)
6. [Configuration Options](#6-configuration-options)
7. [Advanced Use Cases](#7-advanced-use-cases)
8. [Monitoring and Analysis](#8-monitoring-and-analysis)
9. [Exporting and Deployment](#9-exporting-and-deployment)
10. [Troubleshooting](#10-troubleshooting)

## 1. Prerequisites

Before starting, ensure you have the following:

- Python 3.8 or newer
- PyTorch 1.12 or newer (with CUDA support recommended)
- Ultralytics YOLOv8 package
- A pre-trained YOLOv8 model (n, s, m, l, or x variant)
- A properly formatted dataset (COCO, Pascal VOC, or custom in YOLOv8 format)

## 2. Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yolov8-qat.git
cd yolov8-qat
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Set up the project structure (if not already done):

```
yolov8-qat/
├── src/
│   ├── config.py
│   ├── models/
│   │   └── yolov8_qat.py
│   └── quantization/
│       ├── calibration.py
│       ├── fake_quantize.py
│       ├── fusion.py
│       ├── observers.py
│       ├── qconfig.py
│       └── utils.py
├── configs/
│   └── qat_config.yaml
├── models/
│   ├── pretrained/
│   ├── checkpoints/
│   └── exported/
├── dataset/
├── logs/
└── train_qat.py
```

## 3. Basic Usage

To run QAT with default settings:

```bash
python train_qat.py --model yolov8n.pt --data coco.yaml
```

This will:
1. Load a pretrained YOLOv8-nano model
2. Apply default quantization settings
3. Train for the default number of epochs (2)
4. Save the quantized model to the default location

For a more customized run:

```bash
python train_qat.py \
  --model models/pretrained/yolov8s.pt \
  --data dataset/custom_dataset.yaml \
  --qconfig sensitive \
  --epochs 5 \
  --batch-size 16 \
  --img-size 640 \
  --lr 0.0005 \
  --device 0 \
  --save-dir models/checkpoints/custom_run \
  --output custom_quantized_model.pt \
  --eval \
  --export
```

## 4. Training Workflows

### Standard QAT Workflow

```bash
# 1. Start with a pre-trained model
python train_qat.py --model yolov8n.pt --data coco.yaml --config configs/qat_config.yaml

# 2. Evaluate the quantized model
python train_qat.py --model models/checkpoints/qat/quantized_model.pt --data coco.yaml --eval

# 3. Export for deployment
python train_qat.py --model models/checkpoints/qat/quantized_model.pt --export --export-dir models/exported
```

### QAT with Knowledge Distillation

```bash
# Use knowledge distillation to maintain accuracy during quantization
python train_qat.py \
  --model yolov8n.pt \
  --data coco.yaml \
  --config configs/qat_config.yaml \
  --distillation \
  --teacher-model models/pretrained/yolov8m.pt \
  --epochs 5
```

### Mixed Precision QAT

```bash
# Use different bit-widths for different layers based on sensitivity
python train_qat.py \
  --model yolov8n.pt \
  --data coco.yaml \
  --config configs/qat_config.yaml \
  --mixed-precision
```

## 5. Command-Line Arguments

The script accepts numerous arguments to customize the QAT process:

### Basic Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to model or model name | pretrained YOLOv8n |
| `--data` | Dataset YAML path | coco.yaml |
| `--config` | QAT configuration file | configs/qat_config.yaml |

### QAT Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--qconfig` | Quantization configuration | default |
| `--fuse` | Fuse Conv+BN+ReLU modules | True |
| `--keep-detection-head` | Quantize detection head | False |

### Training Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--epochs` | Number of epochs | 2 |
| `--batch-size` | Batch size | 16 |
| `--img-size` | Input image size | 640 |
| `--lr` | Learning rate | 0.0005 |
| `--device` | Training device | cuda |

### Output Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--save-dir` | Directory to save results | models/checkpoints/qat |
| `--output` | Output model name | quantized_model.pt |
| `--log-dir` | Directory to save logs | logs/qat |

### Advanced Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--distillation` | Use knowledge distillation | False |
| `--teacher-model` | Teacher model path | None |
| `--mixed-precision` | Use mixed precision | False |
| `--analyze` | Analyze quantization effects | False |
| `--seed` | Random seed | 42 |
| `--eval` | Evaluate after training | False |
| `--export` | Export after training | False |
| `--export-dir` | Directory for exports | models/exported |

## 6. Configuration Options

The `qat_config.yaml` file allows for fine-grained control over the QAT process. Here are the main configuration sections:

### Model Settings

```yaml
model:
  architecture: "yolov8"
  variant: "n"  # n, s, m, l, x
```

### Quantization Parameters

```yaml
quantization:
  # Default configuration
  default_qconfig: "default"
  
  # Weight quantization settings
  weight:
    dtype: "qint8"
    scheme: "per_channel"
    observer: "minmax"
    symmetric: true
    bit_width: 8
    
  # Activation quantization settings
  activation:
    dtype: "quint8"
    scheme: "per_tensor"
    observer: "moving_average_minmax"
    symmetric: false
    bit_width: 8
    
  # Fake quantization settings
  fake_quantize:
    type: "custom"
    grad_factor: 1.0
```

### Layer-Specific Configuration

```yaml
layer_configs:
  - pattern: "model.0.conv"  # First layer
    config:
      activation:
        observer: "histogram"
      weight:
        observer: "per_channel_minmax"
  
  - pattern: "model.[0-9]+.detect"  # Detection head
    config:
      activation:
        observer: "histogram"
      weight:
        observer: "per_channel_minmax"
    fake_quantize:
      type: "lsq"
```

### QAT Parameters

```yaml
qat_params:
  skip_detection_head: false
  fuse_modules: true
  use_lsq: true
  use_mixed_precision: true
  use_distillation: true
  post_qat_fine_tuning: true
  fine_tuning_epochs: 1
```

### Training Parameters

```yaml
train_params:
  epochs: 2
  batch_size: 16
  lr: 0.0005
  lr_scheduler: "cosine"
  optimizer: "SGD"
  momentum: 0.937
  weight_decay: 0.0005
```

### Export Settings

```yaml
export:
  formats: ["onnx", "tensorrt"]
  include_metadata: true
  simplify: true
```

## 7. Advanced Use Cases

### Fine-Tuning a QAT Model

After initial QAT, you might want to fine-tune for specific accuracy targets:

```bash
python train_qat.py \
  --model models/checkpoints/qat/quantized_model.pt \
  --data dataset/custom_dataset.yaml \
  --epochs 2 \
  --lr 0.0001
```

### Quantizing a Custom YOLOv8 Model

If you have a custom-trained YOLOv8 model:

```bash
python train_qat.py \
  --model models/custom_trained/custom_yolov8.pt \
  --data dataset/custom_dataset.yaml \
  --config configs/custom_qat_config.yaml
```

### Targeting Specific Hardware

To optimize for specific hardware acceleration:

```yaml
# In qat_config.yaml
quantization:
  weight:
    dtype: "qint8"
    scheme: "per_channel"  # For maximum accuracy
  
  activation:
    dtype: "quint8"
    scheme: "per_tensor"  # For better hardware compatibility
    symmetric: false

# For TensorRT deployment
export:
  formats: ["tensorrt"]
  simplify: true
```

Then run:

```bash
python train_qat.py \
  --model yolov8n.pt \
  --data coco.yaml \
  --config configs/qat_config.yaml \
  --export
```

## 8. Monitoring and Analysis

### Analyzing Quantization Effects

To analyze the impact of quantization:

```bash
python train_qat.py \
  --model yolov8n.pt \
  --data coco.yaml \
  --analyze
```

This generates a report with:
- Quantization ratio (percentage of quantized layers)
- Size comparison (FP32 vs INT8)
- Layer-wise error analysis
- Performance metrics

### Visualizing Training Progress

Training metrics are logged to the specified log directory:

```bash
python train_qat.py \
  --model yolov8n.pt \
  --data coco.yaml \
  --log-dir logs/custom_run
```

You can visualize these using TensorBoard:

```bash
tensorboard --logdir logs/custom_run
```

## 9. Exporting and Deployment

### Exporting to Different Formats

```bash
python train_qat.py \
  --model models/checkpoints/qat/quantized_model.pt \
  --export \
  --export-dir models/exported
```

Supported formats (configured in qat_config.yaml):
- ONNX
- TensorRT
- TFLite
- OpenVINO

### Validating Exported Models

After export, validate the model:

```bash
python validate_exports.py --model models/exported/onnx/quantized_model.onnx --data coco.yaml
```

### Deployment Example

To deploy the quantized model with OpenVINO:

```python
import cv2
import numpy as np
from openvino.runtime import Core

# Load the model
ie = Core()
model = ie.read_model(model="models/exported/openvino/quantized_model.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Prepare input
image = cv2.imread("sample.jpg")
input_data = preprocess_image(image)  # Your preprocessing function

# Run inference
results = compiled_model([input_data])[0]
```

## 10. Troubleshooting

### Common Issues and Solutions

**Issue: "CUDA out of memory" error**

Solution: Reduce batch size or image size
```bash
python train_qat.py --model yolov8n.pt --data coco.yaml --batch-size 8 --img-size 416
```

**Issue: Significant accuracy drop after quantization**

Solutions:
1. Use knowledge distillation
   ```bash
   python train_qat.py --model yolov8n.pt --data coco.yaml --distillation
   ```

2. Try a different QConfig
   ```bash
   python train_qat.py --model yolov8n.pt --data coco.yaml --qconfig sensitive
   ```

3. Train for more epochs
   ```bash
   python train_qat.py --model yolov8n.pt --data coco.yaml --epochs 10
   ```

**Issue: Module fusion errors**

Solution: Disable automatic fusion and use default patterns
```bash
python train_qat.py --model yolov8n.pt --data coco.yaml --no-fuse
```

**Issue: Exported model doesn't match training accuracy**

Solution: Validate the exported model and check for quantization parameter changes
```bash
python train_qat.py --model models/checkpoints/qat/quantized_model.pt --eval --export --validate-exports
```

### Logging and Debugging

To increase log verbosity for debugging:

```bash
python -u train_qat.py --model yolov8n.pt --data coco.yaml --log-level debug
```

## Summary

This tutorial covered the entire workflow of Quantization-Aware Training for YOLOv8 models using the enhanced `train_qat.py` script. By following these guidelines, you can effectively quantize your models while maintaining accuracy, resulting in faster inference and reduced model size for deployment.

For more detailed information on configuring the QAT process, refer to the QAT Configuration Guide document.
