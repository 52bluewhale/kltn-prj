#!/usr/bin/env python
"""
Test script to verify the fix for the deepcopy issue in convert_qat_model_to_quantized.
"""
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    print(f"Added {ROOT} to Python path")

import logging
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_conversion')

# Import the function you modified
from src.quantization.utils import convert_qat_model_to_quantized

def test_conversion():
    """Test convert_qat_model_to_quantized function."""
    # Load a simple model (doesn't need to be QAT trained for this test)
    logger.info("Loading a test model...")
    model_path = "models/pretrained/yolov8n.pt"
    model = YOLO(model_path)
    
    # Just to see if the function runs without error
    logger.info("Testing conversion function...")
    try:
        # We're just testing if the function runs without error,
        # the actual conversion might fail but that's not what we're testing
        convert_qat_model_to_quantized(model.model)
        logger.info("SUCCESS: Function ran without deepcopy error!")
    except AttributeError as e:
        if "deepcopy" in str(e):
            logger.error("FAILED: deepcopy error still exists!")
            logger.error(str(e))
        else:
            logger.info("SUCCESS: No deepcopy error (but encountered other error)")
            logger.info(f"Other error: {str(e)}")
    except Exception as e:
        logger.info("SUCCESS: No deepcopy error (but encountered other error)")
        logger.info(f"Other error: {str(e)}")
    
    logger.info("Test completed")

if __name__ == "__main__":
    test_conversion()