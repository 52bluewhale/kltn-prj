#!/usr/bin/env python
"""
YOLOv8 Quantized Model Export Script

This script exports quantized YOLOv8 models to various formats including:
- ONNX (optimized for inference)
- TensorRT (NVIDIA GPU acceleration)
- TFLite (mobile/edge deployment)
- OpenVINO (Intel optimization)
- CoreML (Apple devices)

Supports both QAT models and standard quantized models.
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from pathlib import Path
import time

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('export')

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics YOLO package not found. Please install with: pip install ultralytics")
    sys.exit(1)

# Import project modules
from src.config import DEVICE
from src.models.yolov8_qat import QuantizedYOLOv8
from src.quantization.utils import (
    load_quantized_model,
    get_model_size,
    analyze_quantization_effects
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export YOLOv8 Quantized Models")
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model file (.pt)')
    parser.add_argument('--model-type', type=str, default='auto',
                      choices=['auto', 'quantized', 'qat', 'standard'],
                      help='Type of model to export')
    
    # Export parameters
    parser.add_argument('--format', type=str, nargs='+', 
                      default=['onnx'],
                      choices=['onnx', 'tensorrt', 'tflite', 'openvino', 'coreml', 'torchscript', 'engine'],
                      help='Export format(s)')
    parser.add_argument('--imgsz', type=int, default=640,
                      help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for export')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='models/exported',
                      help='Output directory for exported models')
    parser.add_argument('--name', type=str, default=None,
                      help='Name for exported model (auto-generated if not specified)')
    
    # Export options
    parser.add_argument('--simplify', action='store_true', default=True,
                      help='Simplify ONNX model')
    parser.add_argument('--optimize', action='store_true', default=True,
                      help='Optimize exported model')
    parser.add_argument('--half', action='store_true',
                      help='Use FP16 precision (not recommended for quantized models)')
    parser.add_argument('--int8', action='store_true',
                      help='Use INT8 quantization during export')
    parser.add_argument('--dynamic', action='store_true',
                      help='Enable dynamic axes for ONNX')
    
    # Validation parameters
    parser.add_argument('--validate', action='store_true', default=True,
                      help='Validate exported model')
    parser.add_argument('--data', type=str, default=None,
                      help='Dataset YAML for validation')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                      help='Confidence threshold for validation')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                      help='IoU threshold for NMS')
    
    # Advanced parameters
    parser.add_argument('--device', type=str, default=DEVICE,
                      help='Device to use for export')
    parser.add_argument('--workspace', type=int, default=4,
                      help='TensorRT workspace size (GB)')
    parser.add_argument('--verbose', action='store_true',
                      help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    parser.add_argument('--config', type=str, default='configs/qat_config.yaml',
                      help='Configuration file')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def detect_model_type(model_path):
    """
    Automatically detect the type of model.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Model type string
    """
    try:
        # Try to load as PyTorch model to inspect
        if model_path.endswith('.pt'):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's a quantized model based on metadata or structure
            if isinstance(checkpoint, dict):
                metadata = checkpoint.get('metadata', {})
                if metadata.get('quantized', False) or metadata.get('format') == 'quantized_int8':
                    return 'quantized'
                if metadata.get('prepared_for_qat', False):
                    return 'qat'
                
                # Check if it contains just a model without YOLO wrapper
                if 'model' in checkpoint and 'metadata' in checkpoint:
                    return 'quantized'
                    
            # Check if it's a raw PyTorch model (common for quantized models)
            if hasattr(checkpoint, '__class__') and 'quantized' in str(type(checkpoint)).lower():
                return 'quantized'
                
            # Try to load with YOLO and check for quantization
            try:
                model = YOLO(model_path)
                
                # Check for quantized layers
                has_quantized_layers = False
                for name, module in model.model.named_modules():
                    if 'quantized' in module.__class__.__name__.lower():
                        has_quantized_layers = True
                        break
                    if hasattr(module, 'weight_fake_quant') or hasattr(module, 'activation_post_process'):
                        return 'qat'
                
                if has_quantized_layers:
                    return 'quantized'
                
                return 'standard'
            except Exception as yolo_error:
                logger.debug(f"YOLO loading failed: {yolo_error}")
                # If YOLO can't load it, it might be a quantized model
                return 'quantized'
            
    except Exception as e:
        logger.warning(f"Could not auto-detect model type: {e}")
        return 'standard'

def create_export_name(model_path, format_name, model_type):
    """Create a name for the exported model."""
    base_name = Path(model_path).stem
    
    # Add model type suffix
    if model_type in ['quantized', 'qat']:
        suffix = f"_{model_type}"
    else:
        suffix = ""
    
    return f"{base_name}{suffix}"

def export_model(model, format_list, output_dir, export_name, args):
    """
    Export model to specified formats.
    
    Args:
        model: YOLO model object
        format_list: List of export formats
        output_dir: Output directory
        export_name: Base name for exported files
        args: Command line arguments
        
    Returns:
        Dictionary of exported file paths
    """
    exported_files = {}
    
    for format_name in format_list:
        logger.info(f"Exporting to {format_name.upper()} format...")
        
        # Create format-specific directory
        format_dir = os.path.join(output_dir, format_name)
        os.makedirs(format_dir, exist_ok=True)
        
        # Set export parameters based on format
        export_kwargs = {
            'format': format_name,
            'imgsz': args.imgsz,
            'optimize': args.optimize,
            'half': args.half and format_name not in ['tensorrt'],  # Avoid half precision for TensorRT with quantized models
            'int8': args.int8,
            'device': args.device,
            'verbose': args.verbose,
            'batch': args.batch_size,
        }
        
        # Format-specific parameters
        if format_name == 'onnx':
            export_kwargs.update({
                'simplify': args.simplify,
                'dynamic': args.dynamic,
                'opset': 12,  # Compatible opset version
            })
        elif format_name == 'tensorrt':
            export_kwargs.update({
                'workspace': args.workspace,
                'int8': args.int8,
                'fp16': not args.int8,  # Use FP16 if not using INT8
            })
        elif format_name == 'tflite':
            export_kwargs.update({
                'int8': args.int8,
                'nms': False,  # TFLite doesn't support NMS
            })
        
        try:
            start_time = time.time()
            
            # Export model
            exported_path = model.export(**export_kwargs)
            
            export_time = time.time() - start_time
            
            # Move to format directory if needed
            if isinstance(exported_path, str):
                exported_filename = os.path.basename(exported_path)
                final_path = os.path.join(format_dir, f"{export_name}.{format_name}")
                
                # Handle different extensions
                if format_name == 'onnx' and not final_path.endswith('.onnx'):
                    final_path += '.onnx'
                elif format_name == 'tensorrt' and not final_path.endswith('.engine'):
                    final_path += '.engine'
                elif format_name == 'tflite' and not final_path.endswith('.tflite'):
                    final_path += '.tflite'
                
                # Move file if it's not already in the right place
                if os.path.abspath(exported_path) != os.path.abspath(final_path):
                    if os.path.exists(exported_path):
                        os.rename(exported_path, final_path)
                        exported_path = final_path
                
                exported_files[format_name] = exported_path
                
                # Get file size
                file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
                logger.info(f"✓ {format_name.upper()} export completed in {export_time:.2f}s")
                logger.info(f"  File: {exported_path}")
                logger.info(f"  Size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Failed to export to {format_name}: {e}")
            if args.verbose:
                import traceback
                logger.error(traceback.format_exc())
    
    return exported_files

def validate_exported_model(model_path, data_yaml, args):
    """
    Validate exported model performance.
    
    Args:
        model_path: Path to exported model
        data_yaml: Dataset YAML file
        args: Command line arguments
        
    Returns:
        Validation results
    """
    if not data_yaml or not os.path.exists(data_yaml):
        logger.warning("No dataset provided for validation, skipping validation")
        return None
    
    try:
        logger.info(f"Validating exported model: {model_path}")
        
        # Load exported model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch_size,
            conf=args.conf_thres,
            iou=args.iou_thres,
            device=args.device,
            verbose=args.verbose
        )
        
        logger.info(f"Validation results for {os.path.basename(model_path)}:")
        logger.info(f"  mAP50: {results.box.map50:.4f}")
        logger.info(f"  mAP50-95: {results.box.map:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed for {model_path}: {e}")
        return None

def analyze_model_info(model_path, model_type):
    """
    Analyze and display model information.
    
    Args:
        model_path: Path to model file
        model_type: Type of model
    """
    try:
        logger.info(f"Analyzing model: {model_path}")
        
        # Try to get basic file info first
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        # Try to load model for analysis
        try:
            if model_type == 'quantized':
                # For quantized models, try multiple loading methods
                try:
                    model = YOLO(model_path)
                except:
                    # Try loading checkpoint directly
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model_data = checkpoint['model']
                        if hasattr(model_data, 'parameters'):
                            # Count parameters from the model object
                            total_params = sum(p.numel() for p in model_data.parameters())
                            logger.info(f"Model parameters: {total_params:,}")
                        logger.info(f"Model type: Quantized ({model_type})")
                        
                        # Check for quantization info in metadata
                        metadata = checkpoint.get('metadata', {})
                        if metadata:
                            logger.info(f"Quantization metadata:")
                            for key, value in metadata.items():
                                logger.info(f"  {key}: {value}")
                        return
                    else:
                        logger.warning("Could not extract model info from quantized checkpoint")
                        return
            else:
                model = YOLO(model_path)
            
            # Get model size if we have a YOLO model
            if hasattr(model, 'model'):
                model_size = get_model_size(model.model)
                logger.info(f"Model memory size: {model_size:.2f} MB")
                
                # Analyze quantization if applicable
                if model_type in ['quantized', 'qat']:
                    quant_analysis = analyze_quantization_effects(model.model)
                    logger.info(f"Quantization analysis:")
                    logger.info(f"  Quantized modules: {quant_analysis['quantized_modules']} / {quant_analysis['total_modules']}")
                    logger.info(f"  Quantization ratio: {quant_analysis['quantized_ratio']:.2f}")
                
                # Model summary
                logger.info(f"Model summary:")
                logger.info(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
                logger.info(f"  Layers: {len(list(model.model.modules()))}")
        
        except Exception as model_error:
            logger.warning(f"Could not analyze model details: {model_error}")
            logger.info(f"Model type: {model_type}")
        
    except Exception as e:
        logger.warning(f"Could not analyze model info: {e}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        logger.info(f"Model type: {model_type}")

def main():
    """Main export function."""
    args = parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = load_config(args.config)
    export_config = config.get('export', {})
    
    # Override args with config if not specified
    if args.format == ['onnx'] and 'formats' in export_config:
        args.format = export_config['formats']
    
    if not args.name:
        args.name = create_export_name(args.model, args.format[0], args.model_type)
    
    # Validate model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Auto-detect model type if needed
    if args.model_type == 'auto':
        args.model_type = detect_model_type(args.model)
        logger.info(f"Detected model type: {args.model_type}")
    
    # Debug: inspect model file contents if debug mode is enabled
    if args.debug:
        logger.debug("Inspecting model file contents...")
        try:
            checkpoint = torch.load(args.model, map_location='cpu')
            logger.debug(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                logger.debug(f"Checkpoint keys: {list(checkpoint.keys())}")
                if 'metadata' in checkpoint:
                    logger.debug(f"Metadata: {checkpoint['metadata']}")
        except Exception as debug_error:
            logger.debug(f"Could not inspect checkpoint: {debug_error}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze model information
    analyze_model_info(args.model, args.model_type)
    
    logger.info(f"Starting export process...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Export formats: {args.format}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Batch size: {args.batch_size}")
    
    try:
        # Load model based on type
        if args.model_type == 'qat':
            # Load as QAT model
            logger.info("Loading model as QAT model...")
            qat_model = QuantizedYOLOv8(
                model_path=args.model,
                qconfig_name='default',
                skip_detection_head=True,
                fuse_modules=True
            )
            model = qat_model.model
        elif args.model_type == 'quantized':
            # Load quantized model - need special handling
            logger.info("Loading quantized model...")
            try:
                # First try to load as standard YOLO model
                model = YOLO(args.model)
                logger.info("Successfully loaded quantized model as YOLO model")
            except Exception as yolo_error:
                logger.warning(f"Failed to load as YOLO model: {yolo_error}")
                
                # Try to load the quantized model using our custom loader
                try:
                    # Check if custom loader exists
                    try:
                        from src.quantization.utils import load_quantized_model
                        quantized_model_data = load_quantized_model(args.model)
                        
                        if isinstance(quantized_model_data, dict) and 'model' in quantized_model_data:
                            # Extract the model from the saved data
                            raw_model = quantized_model_data['model']
                            
                            # Create a temporary file with just the model for YOLO to load
                            import tempfile
                            temp_dir = tempfile.mkdtemp()
                            temp_model_path = os.path.join(temp_dir, "temp_model.pt")
                            
                            # Save as a standard PyTorch model
                            torch.save(raw_model, temp_model_path)
                            
                            # Load with YOLO
                            model = YOLO(temp_model_path)
                            logger.info("Successfully loaded quantized model via custom loader")
                            
                            # Clean up temp file
                            os.remove(temp_model_path)
                            os.rmdir(temp_dir)
                        else:
                            raise Exception("Quantized model format not recognized")
                    except ImportError:
                        logger.debug("Custom quantized model loader not available, trying direct loading")
                        raise Exception("Custom loader not available")
                        
                except Exception as custom_error:
                    logger.error(f"Failed to load with custom loader: {custom_error}")
                    
                    # Handle state_dict format (from your QAT training)
                    try:
                        checkpoint = torch.load(args.model, map_location='cpu')
                        
                        if isinstance(checkpoint, dict):
                            if 'state_dict' in checkpoint:
                                # This is the format from your QAT training: {'state_dict': ..., 'metadata': ...}
                                logger.info("Found state_dict format quantized model")
                                
                                state_dict = checkpoint['state_dict']
                                metadata = checkpoint.get('metadata', {})
                                
                                logger.info(f"Quantized model metadata: {metadata}")
                                
                                # We need to create a base model and load the quantized weights
                                # First, try to find the original model that was used for training
                                
                                # Look for the original model in different locations
                                possible_base_models = [
                                    'yolov8n.pt',  # Default model
                                    'models/pretrained/yolov8n.pt',
                                    'models/checkpoints/qat/phase4_fine_tuning/weights/best.pt',
                                    'models/checkpoints/qat/phase1_weight_only/weights/best.pt'
                                ]
                                
                                base_model = None
                                for model_path in possible_base_models:
                                    if os.path.exists(model_path):
                                        try:
                                            base_model = YOLO(model_path)
                                            logger.info(f"Using base model: {model_path}")
                                            break
                                        except:
                                            continue
                                
                                if base_model is None:
                                    # Create a default YOLOv8n model
                                    logger.info("Creating default YOLOv8n model as base")
                                    base_model = YOLO('yolov8n.pt')
                                
                                # Load the quantized state dict into the base model
                                try:
                                    # Load state dict with strict=False to handle missing/extra keys
                                    missing_keys, unexpected_keys = base_model.model.load_state_dict(state_dict, strict=False)
                                    
                                    if missing_keys:
                                        logger.warning(f"Missing keys when loading state dict: {missing_keys[:5]}...")
                                    if unexpected_keys:
                                        logger.warning(f"Unexpected keys when loading state dict: {unexpected_keys[:5]}...")
                                    
                                    model = base_model
                                    logger.info("Successfully loaded quantized weights into base model")
                                    
                                except Exception as load_error:
                                    logger.error(f"Failed to load state dict: {load_error}")
                                    
                                    # Fallback: create a temporary model file
                                    import tempfile
                                    temp_dir = tempfile.mkdtemp()
                                    temp_model_path = os.path.join(temp_dir, "temp_quantized_model.pt")
                                    
                                    # Save just the state dict as a model file
                                    torch.save(state_dict, temp_model_path)
                                    
                                    try:
                                        model = YOLO(temp_model_path)
                                        logger.info("Successfully loaded via temporary file")
                                    except Exception as temp_error:
                                        logger.error(f"Temporary file method failed: {temp_error}")
                                        
                                        # Clean up and try one more approach
                                        os.remove(temp_model_path)
                                        os.rmdir(temp_dir)
                                        
                                        # Create a simple model structure
                                        temp_model_data = {
                                            'model': state_dict,
                                            'metadata': metadata
                                        }
                                        
                                        temp_model_path2 = os.path.join(tempfile.mkdtemp(), "model.pt")
                                        torch.save(temp_model_data, temp_model_path2)
                                        
                                        try:
                                            model = YOLO(temp_model_path2)
                                            logger.info("Successfully loaded via restructured temporary file")
                                        except:
                                            raise Exception("Could not load quantized model in any format")
                                        finally:
                                            if os.path.exists(temp_model_path2):
                                                os.remove(temp_model_path2)
                                                os.rmdir(os.path.dirname(temp_model_path2))
                                    
                                    # Clean up temp directory
                                    if os.path.exists(temp_dir):
                                        import shutil
                                        shutil.rmtree(temp_dir)
                                        
                            elif 'model' in checkpoint:
                                # Extract the model state dict or model object
                                model_data = checkpoint['model']
                                
                                # Create a temporary YOLO model and load the weights
                                import tempfile
                                temp_dir = tempfile.mkdtemp()
                                temp_model_path = os.path.join(temp_dir, "temp_model.pt")
                                
                                if hasattr(model_data, 'state_dict'):
                                    # It's a model object
                                    torch.save(model_data.state_dict(), temp_model_path)
                                elif isinstance(model_data, dict):
                                    # It's a state dict
                                    torch.save(model_data, temp_model_path)
                                else:
                                    # Try to save as-is
                                    torch.save(model_data, temp_model_path)
                                
                                # Load with YOLO
                                model = YOLO(temp_model_path)
                                logger.info("Successfully loaded quantized model via checkpoint extraction")
                                
                                # Clean up temp file
                                os.remove(temp_model_path)
                                os.rmdir(temp_dir)
                            else:
                                raise Exception(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())}")
                        else:
                            raise Exception("Checkpoint is not a dictionary")
                            
                    except Exception as final_error:
                        logger.error(f"All loading methods failed: {final_error}")
                        raise Exception(f"Cannot load quantized model: {args.model}")
        else:
            # Load as standard YOLO model
            logger.info("Loading model as standard YOLO model...")
            model = YOLO(args.model)
        
        # Verify model has export capability
        if not hasattr(model, 'export'):
            raise Exception(f"Loaded model does not have export capability. Model type: {type(model)}")
        
        logger.info(f"Successfully loaded model: {type(model)}")
        
        # Export to specified formats
        exported_files = export_model(
            model=model,
            format_list=args.format,
            output_dir=args.output_dir,
            export_name=args.name,
            args=args
        )
        
        # Validate exported models if requested
        if args.validate and args.data:
            logger.info("Validating exported models...")
            validation_results = {}
            
            for format_name, file_path in exported_files.items():
                if os.path.exists(file_path):
                    results = validate_exported_model(file_path, args.data, args)
                    if results:
                        validation_results[format_name] = results
        
        # Summary
        logger.info("=" * 80)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Successfully exported to {len(exported_files)} format(s):")
        
        for format_name, file_path in exported_files.items():
            file_size = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
            logger.info(f"  {format_name.upper()}: {file_path} ({file_size:.2f} MB)")
        
        # Usage examples
        logger.info("\nUsage examples:")
        for format_name, file_path in exported_files.items():
            if format_name == 'onnx':
                logger.info(f"  ONNX: yolo predict model={file_path} source=image.jpg")
            elif format_name == 'tensorrt':
                logger.info(f"  TensorRT: yolo predict model={file_path} source=image.jpg")
            elif format_name == 'tflite':
                logger.info(f"  TFLite: Use with TensorFlow Lite interpreter")
        
        logger.info(f"\nExport completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()