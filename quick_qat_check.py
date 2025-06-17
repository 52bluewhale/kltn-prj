from ultralytics import YOLO

print("qat model: ")
model = YOLO('models/checkpoints/qat_enhanced_2/best_qat.pt')  # or your custom model
model.info(verbose=True)

print("yolov8n model: ")
model = YOLO('yolov8n.pt')  # or your custom model
model.info(verbose=True)
