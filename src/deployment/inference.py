"""
Inference utilities for YOLOv8 QAT models.

This module provides functions for loading and running inference with 
quantized YOLOv8 models using various inference engines.
"""

import torch
import numpy as np
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2

# Setup logging
logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Base class for inference engines.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file
            device: Device to run inference on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.input_shape = None
        self.initialized = False
    
    def load_model(self):
        """
        Load model for inference.
        
        Returns:
            Success flag
        """
        raise NotImplementedError("Subclasses must implement load_model")
    
    def preprocess(self, image: Union[np.ndarray, str]):
        """
        Preprocess image for inference.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Preprocessed image
        """
        raise NotImplementedError("Subclasses must implement preprocess")
    
    def run_inference(self, input_data):
        """
        Run inference on input data.
        
        Args:
            input_data: Input data
            
        Returns:
            Inference results
        """
        raise NotImplementedError("Subclasses must implement run_inference")
    
    def postprocess(self, outputs, original_size: Tuple[int, int]):
        """
        Postprocess model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed results
        """
        raise NotImplementedError("Subclasses must implement postprocess")


class PyTorchInferenceEngine(InferenceEngine):
    """
    Inference engine for PyTorch models.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", quantized: bool = True):
        """
        Initialize PyTorch inference engine.
        
        Args:
            model_path: Path to model file
            device: Device to run inference on
            quantized: Whether the model is quantized
        """
        super().__init__(model_path, device)
        self.quantized = quantized
        if quantized and device == "cuda":
            logger.warning("Quantized models may not be supported on CUDA. Falling back to CPU.")
            self.device = "cpu"
    
    def load_model(self):
        """
        Load PyTorch model for inference.
        
        Returns:
            Success flag
        """
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location="cpu")
            
            # Extract model from checkpoint
            if "model" in checkpoint:
                self.model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                # Try to initialize model from state dict
                # This requires the model class to be defined
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    self.model = DetectionModel()
                    self.model.load_state_dict(checkpoint["state_dict"])
                except ImportError:
                    logger.error("Ultralytics not found. Cannot load model from state dict.")
                    return False
                except Exception as e:
                    logger.error(f"Failed to load state dict: {e}")
                    return False
            else:
                self.model = checkpoint
            
            # Move model to device
            if self.device == "cpu" and self.quantized:
                # Ensure model is in eval mode for quantized inference
                self.model.eval()
            else:
                self.model = self.model.to(self.device)
                self.model.eval()
            
            # Get input shape from model if available
            if hasattr(self.model, "input_shape"):
                self.input_shape = self.model.input_shape
            else:
                # Default YOLOv8 input shape
                self.input_shape = (1, 3, 640, 640)
            
            self.initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def preprocess(self, image: Union[np.ndarray, str]):
        """
        Preprocess image for PyTorch model.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for postprocessing
        original_size = image.shape[:2]
        
        # Resize image
        height, width = self.input_shape[2:]
        resized = cv2.resize(image, (width, height))
        
        # Normalize and convert to tensor
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Convert to tensor
        tensor = torch.from_numpy(img)
        
        if not self.quantized:
            tensor = tensor.to(self.device)
        
        return tensor, original_size
    
    def run_inference(self, input_data):
        """
        Run inference with PyTorch model.
        
        Args:
            input_data: Tuple of (input_tensor, original_size)
            
        Returns:
            Model outputs
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        input_tensor, original_size = input_data
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        # Postprocess results
        results = self.postprocess(outputs, original_size)
        
        return results, inference_time
    
    def postprocess(self, outputs, original_size: Tuple[int, int]):
        """
        Postprocess PyTorch model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed detection results: list of [x1, y1, x2, y2, conf, class_id]
        """
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Standard output format: [batch_size, num_boxes, 6]
            # where 6 is [x1, y1, x2, y2, confidence, class_id]
            predictions = outputs.detach().cpu().numpy()
        elif isinstance(outputs, (tuple, list)):
            # Some YOLOv8 models output a tuple of tensors
            if len(outputs) >= 1 and isinstance(outputs[0], torch.Tensor):
                predictions = outputs[0].detach().cpu().numpy()
            else:
                predictions = np.array([])
        else:
            # Unsupported output format
            logger.warning(f"Unsupported output format: {type(outputs)}")
            predictions = np.array([])
        
        # Scale bounding boxes to original image size
        if len(predictions) > 0 and predictions.shape[-1] >= 6:
            orig_h, orig_w = original_size
            input_h, input_w = self.input_shape[2:]
            
            # Scale factor
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            # Scale bounding boxes
            predictions[:, 0] *= scale_x  # x1
            predictions[:, 1] *= scale_y  # y1
            predictions[:, 2] *= scale_x  # x2
            predictions[:, 3] *= scale_y  # y2
        
        return predictions


class ONNXInferenceEngine(InferenceEngine):
    """
    Inference engine for ONNX models.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize ONNX inference engine.
        
        Args:
            model_path: Path to ONNX model file
            device: Device to run inference on
        """
        super().__init__(model_path, device)
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = None
        self.input_name = None
        self.output_names = None
    
    def load_model(self):
        """
        Load ONNX model for inference.
        
        Returns:
            Success flag
        """
        try:
            import onnxruntime as ort
            
            # Create inference session
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            
            # Get input and output names
            model_inputs = self.session.get_inputs()
            model_outputs = self.session.get_outputs()
            
            if len(model_inputs) > 0:
                self.input_name = model_inputs[0].name
                self.input_shape = model_inputs[0].shape
                
                # Handle dynamic batch size
                if self.input_shape[0] == -1:
                    self.input_shape = tuple([1] + list(self.input_shape[1:]))
            
            if len(model_outputs) > 0:
                self.output_names = [output.name for output in model_outputs]
            
            self.initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def preprocess(self, image: Union[np.ndarray, str]):
        """
        Preprocess image for ONNX model.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for postprocessing
        original_size = image.shape[:2]
        
        # Resize image
        height, width = self.input_shape[2:]
        resized = cv2.resize(image, (width, height))
        
        # Normalize image
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return {self.input_name: img}, original_size
    
    def run_inference(self, input_data):
        """
        Run inference with ONNX model.
        
        Args:
            input_data: Tuple of (input_dict, original_size)
            
        Returns:
            Model outputs
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        input_dict, original_size = input_data
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, input_dict)
        inference_time = time.time() - start_time
        
        # Postprocess results
        results = self.postprocess(outputs, original_size)
        
        return results, inference_time
    
    def postprocess(self, outputs, original_size: Tuple[int, int]):
        """
        Postprocess ONNX model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed detection results
        """
        # Get output tensor
        if len(outputs) > 0:
            predictions = outputs[0]
        else:
            return np.array([])
        
        # Scale bounding boxes to original image size
        if len(predictions) > 0 and predictions.shape[-1] >= 6:
            orig_h, orig_w = original_size
            input_h, input_w = self.input_shape[2:]
            
            # Scale factor
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            # Scale bounding boxes
            predictions[:, 0] *= scale_x  # x1
            predictions[:, 1] *= scale_y  # y1
            predictions[:, 2] *= scale_x  # x2
            predictions[:, 3] *= scale_y  # y2
        
        return predictions


class TFLiteInferenceEngine(InferenceEngine):
    """
    Inference engine for TFLite models.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize TFLite inference engine.
        
        Args:
            model_path: Path to TFLite model file
            device: Device to run inference on (only CPU supported)
        """
        super().__init__(model_path, "cpu")  # TFLite only supports CPU
        self.interpreter = None
        self.input_details = None
        self.output_details = None
    
    def load_model(self):
        """
        Load TFLite model for inference.
        
        Returns:
            Success flag
        """
        try:
            import tensorflow as tf
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            if len(self.input_details) > 0:
                self.input_shape = tuple(self.input_details[0]["shape"])
            
            self.initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            return False
    
    def preprocess(self, image: Union[np.ndarray, str]):
        """
        Preprocess image for TFLite model.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for postprocessing
        original_size = image.shape[:2]
        
        # Resize image
        height, width = self.input_shape[1:3]
        resized = cv2.resize(image, (width, height))
        
        # Normalize image
        img = resized.astype(np.float32) / 255.0
        
        # TFLite models expect NHWC format
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img, original_size
    
    def run_inference(self, input_data):
        """
        Run inference with TFLite model.
        
        Args:
            input_data: Tuple of (input_tensor, original_size)
            
        Returns:
            Model outputs
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        input_tensor, original_size = input_data
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get output tensor
        outputs = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail["index"])
            outputs.append(output)
        
        # Postprocess results
        results = self.postprocess(outputs, original_size)
        
        return results, inference_time
    
    def postprocess(self, outputs, original_size: Tuple[int, int]):
        """
        Postprocess TFLite model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed detection results
        """
        # Get detection boxes
        if len(outputs) >= 4:
            # TFLite YOLOv8 model returns separate tensors for boxes, scores, classes
            boxes = outputs[0][0]  # [N, 4]
            scores = outputs[2][0]  # [N]
            classes = outputs[1][0]  # [N]
            
            # Filter by confidence
            valid_indices = scores > 0.001
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
            valid_classes = classes[valid_indices]
            
            # Format predictions as [x1, y1, x2, y2, confidence, class_id]
            predictions = np.zeros((len(valid_boxes), 6))
            predictions[:, :4] = valid_boxes
            predictions[:, 4] = valid_scores
            predictions[:, 5] = valid_classes
        elif len(outputs) >= 1:
            # Model returns single tensor with all detections
            predictions = outputs[0]
        else:
            return np.array([])
        
        # Scale bounding boxes to original image size
        if len(predictions) > 0 and predictions.shape[-1] >= 6:
            orig_h, orig_w = original_size
            input_h, input_w = self.input_shape[1:3]
            
            # Scale factor
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            # Scale bounding boxes
            predictions[:, 0] *= scale_x  # x1
            predictions[:, 1] *= scale_y  # y1
            predictions[:, 2] *= scale_x  # x2
            predictions[:, 3] *= scale_y  # y2
        
        return predictions


class OpenVINOInferenceEngine(InferenceEngine):
    """
    Inference engine for OpenVINO models.
    """
    
    def __init__(self, model_path: str, device: str = "CPU"):
        """
        Initialize OpenVINO inference engine.
        
        Args:
            model_path: Path to OpenVINO model file (.xml)
            device: Device to run inference on (CPU, GPU, etc.)
        """
        super().__init__(model_path, device)
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
    
    def load_model(self):
        """
        Load OpenVINO model for inference.
        
        Returns:
            Success flag
        """
        try:
            import openvino as ov
            
            # Create OpenVINO Core
            core = ov.Core()
            
            # Read model
            model = core.read_model(self.model_path)
            
            # Check if model file is an IR file
            if not self.model_path.endswith(".xml"):
                logger.warning(f"Model file {self.model_path} is not an IR file.")
                
                # Check if corresponding .bin file exists
                bin_path = os.path.splitext(self.model_path)[0] + ".bin"
                if not os.path.exists(bin_path):
                    logger.warning(f"Weights file {bin_path} not found.")
            
            # Compile model for device
            self.compiled_model = core.compile_model(model, self.device)
            
            # Get input and output layers
            inputs = model.inputs
            outputs = model.outputs
            
            if len(inputs) > 0:
                self.input_layer = inputs[0]
                self.input_shape = inputs[0].shape
            
            if len(outputs) > 0:
                self.output_layer = outputs[0]
            
            self.initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            return False
    
    def preprocess(self, image: Union[np.ndarray, str]):
        """
        Preprocess image for OpenVINO model.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for postprocessing
        original_size = image.shape[:2]
        
        # Resize image
        height, width = self.input_shape[2:]
        resized = cv2.resize(image, (width, height))
        
        # Normalize image
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img, original_size
    
    def run_inference(self, input_data):
        """
        Run inference with OpenVINO model.
        
        Args:
            input_data: Tuple of (input_tensor, original_size)
            
        Returns:
            Model outputs
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        input_tensor, original_size = input_data
        
        # Create input dict
        inputs = {self.input_layer.any_name: input_tensor}
        
        # Run inference
        start_time = time.time()
        results = self.compiled_model(inputs)
        inference_time = time.time() - start_time
        
        # Get output tensor
        output_key = self.output_layer.any_name
        outputs = results[output_key]
        
        # Postprocess results
        processed_results = self.postprocess(outputs, original_size)
        
        return processed_results, inference_time
    
    def postprocess(self, outputs, original_size: Tuple[int, int]):
        """
        Postprocess OpenVINO model outputs.
        
        Args:
            outputs: Model outputs
            original_size: Original image size (height, width)
            
        Returns:
            Processed detection results
        """
        # Get detection boxes
        predictions = outputs
        
        # Scale bounding boxes to original image size
        if len(predictions) > 0 and predictions.shape[-1] >= 6:
            orig_h, orig_w = original_size
            input_h, input_w = self.input_shape[2:]
            
            # Scale factor
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            # Scale bounding boxes
            predictions[:, 0] *= scale_x  # x1
            predictions[:, 1] *= scale_y  # y1
            predictions[:, 2] *= scale_x  # x2
            predictions[:, 3] *= scale_y  # y2
        
        return predictions


def create_inference_engine(model_path: str, backend: str = "pytorch", device: str = "cuda", **kwargs):
    """
    Create inference engine for the specified backend.
    
    Args:
        model_path: Path to model file
        backend: Inference backend ("pytorch", "onnx", "tflite", "openvino")
        device: Device to run inference on
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Inference engine
    """
    # Create appropriate inference engine
    if backend.lower() == "pytorch":
        quantized = kwargs.get("quantized", True)
        engine = PyTorchInferenceEngine(model_path, device, quantized)
    elif backend.lower() == "onnx":
        engine = ONNXInferenceEngine(model_path, device)
    elif backend.lower() == "tflite":
        engine = TFLiteInferenceEngine(model_path, device)
    elif backend.lower() == "openvino":
        engine = OpenVINOInferenceEngine(model_path, device)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    # Load model
    success = engine.load_model()
    if not success:
        raise RuntimeError(f"Failed to load model with {backend} backend")
    
    return engine


def run_inference(model_path: str, image_path: str, backend: str = "pytorch", device: str = "cuda", 
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45, **kwargs):
    """
    Run inference on a single image.
    
    Args:
        model_path: Path to model file
        image_path: Path to image file
        backend: Inference backend
        device: Device to run inference on
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with inference results
    """
    # Create inference engine
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    
    # Read and preprocess image
    input_data = engine.preprocess(image_path)
    
    # Run inference
    detections, inference_time = engine.run_inference(input_data)
    
    # Filter by confidence
    if len(detections) > 0:
        mask = detections[:, 4] >= conf_threshold
        detections = detections[mask]
    
    # Apply NMS if needed
    # NOTE: For simplicity, NMS is not implemented here as it's backend-specific
    # and would require additional code for each backend
    
    return {
        "detections": detections,
        "inference_time": inference_time,
        "fps": 1.0 / inference_time if inference_time > 0 else 0
    }


def run_batch_inference(model_path: str, image_paths: List[str], backend: str = "pytorch", 
                     device: str = "cuda", batch_size: int = 4, **kwargs):
    """
    Run inference on a batch of images.
    
    Args:
        model_path: Path to model file
        image_paths: List of paths to image files
        backend: Inference backend
        device: Device to run inference on
        batch_size: Batch size for inference
        **kwargs: Additional arguments for specific backends
        
    Returns:
        List of dictionaries with inference results
    """
    # Create inference engine
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    
    # Process images in batches
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_results = []
        
        # Process each image in batch
        for img_path in batch_paths:
            # Read and preprocess image
            input_data = engine.preprocess(img_path)
            
            # Run inference
            detections, inference_time = engine.run_inference(input_data)
            
            # Store results
            batch_results.append({
                "image_path": img_path,
                "detections": detections,
                "inference_time": inference_time
            })
        
        results.extend(batch_results)
    
    return results


def get_inference_profile(model_path: str, image: Union[str, np.ndarray], backend: str = "pytorch", 
                         device: str = "cuda", num_runs: int = 100, **kwargs):
    """
    Get inference profile for a model.
    
    Args:
        model_path: Path to model file
        image: Path to image file or image array
        backend: Inference backend
        device: Device to run inference on
        num_runs: Number of inference runs
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Dictionary with inference profile
    """
    # Create inference engine
    engine = create_inference_engine(model_path, backend, device, **kwargs)
    
    # Read and preprocess image
    input_data = engine.preprocess(image)
    
    # Run warmup inference
    _, _ = engine.run_inference(input_data)
    
    # Run inference multiple times and measure performance
    inference_times = []
    
    for _ in range(num_runs):
        _, inference_time = engine.run_inference(input_data)
        inference_times.append(inference_time)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    return {
        "mean_inference_time": mean_time,
        "median_inference_time": median_time,
        "std_inference_time": std_time,
        "min_inference_time": min_time,
        "max_inference_time": max_time,
        "fps": 1.0 / mean_time if mean_time > 0 else 0,
        "backend": backend,
        "device": device
    }


def load_model_for_inference(model_path: str, backend: str = "pytorch", device: str = "cuda", **kwargs):
    """
    Load model for inference without running inference.
    
    Args:
        model_path: Path to model file
        backend: Inference backend
        device: Device to run inference on
        **kwargs: Additional arguments for specific backends
        
    Returns:
        Loaded inference engine
    """
    return create_inference_engine(model_path, backend, device, **kwargs)