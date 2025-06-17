import torch
import torch.quantization
from ultralytics import YOLO
import yaml
import os

# Kiểm tra GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("GPU not detected. Ensure CUDA is installed.")

# Kiểm tra NumPy
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    raise RuntimeError("NumPy not available. Install with 'pip install numpy==1.24.4'")

# Kiểm tra dataset YAML
dataset_path = "F:/kltn-prj/datasets/vietnam-traffic-sign-detection/dataset.yaml"
try:
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)
    print("YAML content:", data)
    assert os.path.exists(data['train']), f"Train folder {data['train']} not found"
    assert os.path.exists(data['val']), f"Val folder {data['val']} not found"
except Exception as e:
    print(f"Error loading YAML: {e}")
    raise

# Tải mô hình pretrained YOLOv8n
model_path = "F:/kltn-prj/models/pretrained/yolov8n.pt"
model_yolo = YOLO(model_path)
print(model_yolo.info())
model = model_yolo.model

# Cấu hình QAT
qconfig = torch.quantization.QConfig(
    activation=torch.quantization.MovingAverageMinMaxObserver.with_args(quant_min=0, quant_max=255, dtype=torch.qint8, averaging_constant=0.01),
    weight=torch.quantization.MinMaxObserver.with_args(quant_min=-128, quant_max=127, dtype=torch.qint8)
)
model.qconfig = qconfig
try:
    model_prepared = torch.quantization.prepare_qat(model.train(), inplace=True)
    model_prepared.apply(torch.quantization.disable_observer)  # Khóa observer
except Exception as e:
    print(f"Error preparing QAT: {e}")
    model_prepared = model

# Gán mô hình QAT
model_yolo.model = model_prepared

def main():
    global model_yolo
    # Huấn luyện QAT
    finetune_epochs = 1
    try:
        print("Starting QAT training...")
        print("Model structure before training:", model_yolo.model)
        model_yolo.train(data=dataset_path,
                         optimizer="Adam",
                         freeze=10,
                         epochs=finetune_epochs,
                         batch=4,
                         imgsz=640,
                         dropout=0.1,
                         device=0,
                         lr0=0.0005,
                         lrf=0.2,
                         cache='disk',
                         amp=True,
                         mosaic=0.0,
                         hsv_h=0.0,
                         fliplr=0.0,
                         translate=0.0,
                         workers=2,
                         verbose=True)
        print("Model structure after training:", model_yolo.model)
        print("QAT training completed.")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Falling back to pretrained model.")
        return

    # Lưu mô hình QAT
    try:
        qat_model_path = "F:/kltn-prj/models/checkpoints/QAT(2)/qat_best.pt"
        model_yolo.save(qat_model_path)
        print(f"QAT model saved at {qat_model_path}")
    except Exception as e:
        print(f"Error saving QAT model: {e}")

    # Validate mô hình QAT (FP32)
    try:
        print("Starting validation...")
        results = model_yolo.val(data=dataset_path, batch=4, workers=2, device=0)
        print("Validation results for QAT model (FP32):", results)
    except Exception as e:
        print(f"Error during validation: {e}")

    # Xuất ONNX
    try:
        model_yolo.export(format='onnx', opset=13)
        print("FP32 ONNX model exported as best_qat.onnx")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

    # Chuyển đổi INT8
    try:
        model_int8 = torch.quantization.convert(model_yolo.model.eval())
        model_yolo.model = model_int8
        int8_model_path = "F:/kltn-prj/models/checkpoints/QAT(2)/qat_int8_best.pt"
        model_yolo.save(int8_model_path)
        print(f"INT8 model saved at {int8_model_path}")
    except Exception as e:
        print(f"Error converting to INT8: {e}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()