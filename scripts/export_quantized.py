import torch
import argparse
import os
from pathlib import Path
import torch.onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw quantized PyTorch model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to quantized model (.pt)")
    parser.add_argument("--output", type=str, default="models/exported/onnx/yolov8_qat.onnx", help="Output ONNX file path")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model
    export_path = args.output
    img_size = args.img_size

    print(f"[INFO] Loading quantized model from {model_path}")
    model = torch.load(model_path, map_location='cpu')
    # model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    export_dir = os.path.dirname(export_path)
    os.makedirs(export_dir, exist_ok=True)

    print(f"[INFO] Exporting to ONNX at {export_path}")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print("[✅] Export complete!")


if __name__ == "__main__":
    main()
