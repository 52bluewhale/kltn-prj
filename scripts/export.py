import torch
import argparse
from pathlib import Path

from src.models.yolov8_qat import YOLOv8QAT

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8-QAT model")
    parser.add_argument("--model", type=str, required=True, help="Path to QAT model checkpoint")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "tflite"], help="Export format")
    parser.add_argument("--output", type=str, default="models/exported", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    model = YOLOv8QAT(args.model)
    model.eval()
    
    # Create output directory
    output_dir = Path(args.output) / args.format
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model
    if args.format == "onnx":
        # Export to ONNX
        model.export_onnx(output_dir / "yolov8_qat.onnx")
    elif args.format == "tflite":
        # Export to TFLite (you'll need to implement this)
        pass
    
    print(f"Model exported to {output_dir}")

if __name__ == "__main__":
    main()