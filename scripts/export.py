import torch
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.yolov8_qat import QuantizedYOLOv8



def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8-QAT model")
    parser.add_argument("--model", type=str, required=True, help="Path to QAT model checkpoint")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "tflite"], help="Export format")
    parser.add_argument("--output", type=str, default="models/exported", help="Output directory")
    return parser.parse_args()

import shutil

def main():
    args = parse_args()
    
    # Load model
    model = QuantizedYOLOv8(args.model)
    
    # Define output directory and final export path
    export_format = args.format.lower()
    output_dir = Path(args.output) / export_format
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"yolov8_qat.{export_format}"

    # Run export (Ultralytics will save in original model folder)
    model_exported_path = model.export(
        export_path=str(final_path),  # This won't be honored directly by Ultralytics
        format=export_format
    )

    # After export, move the file manually if it's in the wrong place
    if export_format == "onnx":
        expected_default = Path(args.model).with_suffix(".onnx")  # e.g., best.pt → best.onnx
        if expected_default.exists():
            shutil.move(str(expected_default), str(final_path))
            print(f"[INFO] Moved ONNX file from {expected_default} to {final_path}")
        else:
            print(f"[WARN] Expected ONNX output not found at {expected_default}")

    print(f"[INFO] Model exported to {final_path}")

if __name__ == "__main__":
    main()