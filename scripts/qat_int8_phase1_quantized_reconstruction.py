#!/usr/bin/env python
"""
Phase 1: Quantized YOLOv8 Architecture Reconstruction
Systematic approach to loading INT8 quantized model
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import psutil
from tqdm import tqdm
import copy

# Add these imports at the top, around line 10-15:
from torch.quantization import QConfig
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantizedModelReconstructor:
    """Handles reconstruction of quantized YOLOv8 model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.saved_data = None
        self.state_dict = None
        self.metadata = None
        self.fp32_model = None
        self.quantized_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"🖥️ Using device: {self.device}")
    
    def step_1a_load_and_analyze(self) -> bool:
        """Load saved model and analyze structure"""
        logger.info("=" * 60)
        logger.info("STEP 1A: LOAD AND ANALYZE SAVED MODEL")
        logger.info("=" * 60)
        
        try:
            logger.info(f"📂 Loading model from: {self.model_path}")
            self.saved_data = torch.load(self.model_path, map_location='cpu')
            
            # Extract components
            self.state_dict = self.saved_data.get('state_dict', self.saved_data.get('model_state_dict'))
            self.metadata = self.saved_data.get('metadata', {})
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 State dict parameters: {len(self.state_dict)}")
            logger.info(f"📋 Metadata: {self.metadata}")
            
            # Analyze parameter types
            param_analysis = self._analyze_parameter_types()
            logger.info(f"🔍 Parameter analysis:")
            for param_type, count in param_analysis.items():
                logger.info(f"   {param_type}: {count}")
            
            # Check quantization parameters specifically
            quant_params = [k for k in self.state_dict.keys() 
                          if any(x in k for x in ['scale', 'zero_point', '_packed_params'])]
            logger.info(f"🎯 Quantization parameters found: {len(quant_params)}")
            
            if len(quant_params) == 0:
                logger.error("❌ No quantization parameters found!")
                return False
            
            # Show sample quantization parameters
            logger.info(f"📝 Sample quantization parameters:")
            for i, param_name in enumerate(quant_params[:5]):
                param = self.state_dict[param_name]
                logger.info(f"   {i+1}. {param_name}: {param.shape if hasattr(param, 'shape') else type(param)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _analyze_parameter_types(self) -> Dict[str, int]:
        """Analyze types of parameters in state_dict"""
        analysis = {
            'weight_params': 0,
            'bias_params': 0,
            'bn_params': 0,
            'quantization_params': 0,
            'other_params': 0
        }
        
        for name, param in self.state_dict.items():
            if 'weight' in name and 'fake_quant' not in name:
                analysis['weight_params'] += 1
            elif 'bias' in name:
                analysis['bias_params'] += 1
            elif any(bn_key in name for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):
                analysis['bn_params'] += 1
            elif any(quant_key in name for quant_key in ['scale', 'zero_point', '_packed_params']):
                analysis['quantization_params'] += 1
            else:
                analysis['other_params'] += 1
        
        return analysis
    
    def step_1b_create_base_architecture(self) -> bool:
        """Create base YOLOv8n architecture"""
        logger.info("=" * 60)
        logger.info("STEP 1B: CREATE BASE YOLOV8N ARCHITECTURE")
        logger.info("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            # Create base YOLOv8n model
            logger.info("🏗️ Creating base YOLOv8n model...")
            base_yolo = YOLO('yolov8n.pt')
            self.fp32_model = base_yolo.model
            
            # Verify model structure
            fp32_params = dict(self.fp32_model.named_parameters())
            logger.info(f"📊 FP32 model parameters: {len(fp32_params)}")
            
            # Compare with our saved model structure
            saved_weight_params = [k for k in self.state_dict.keys() if 'weight' in k and 'fake_quant' not in k]
            fp32_weight_params = [k for k in fp32_params.keys() if 'weight' in k]
            
            logger.info(f"🔍 Structure comparison:")
            logger.info(f"   Saved model weights: {len(saved_weight_params)}")
            logger.info(f"   FP32 model weights: {len(fp32_weight_params)}")
            
            # Check if structures match
            if len(saved_weight_params) != len(fp32_weight_params):
                logger.warning("⚠️ Parameter count mismatch - checking shapes...")
                
                # Sample shape comparison
                logger.info("📏 Sample shape comparison:")
                for i, (saved_name, fp32_name) in enumerate(zip(saved_weight_params[:5], fp32_weight_params[:5])):
                    saved_shape = self.state_dict[saved_name].shape
                    fp32_shape = fp32_params[fp32_name].shape
                    match = "✅" if saved_shape == fp32_shape else "❌"
                    logger.info(f"   {i+1}. {match} {saved_name}: {saved_shape} vs {fp32_name}: {fp32_shape}")
            
            # Modify model for correct number of classes if needed
            if 'nc' in self.metadata:
                target_classes = self.metadata['nc']
                logger.info(f"🎯 Adjusting model for {target_classes} classes...")
                # This will be handled in quantization preparation
            
            logger.info("✅ Base architecture created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating base architecture: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step_1c_prepare_quantization(self) -> bool:
        """Prepare model for quantization with LSQ configuration"""
        logger.info("=" * 60)
        logger.info("STEP 1C: PREPARE QUANTIZATION CONFIGURATION")
        logger.info("=" * 60)
        
        try:
            from torch.quantization import QConfig
            from torch.quantization.fake_quantize import FakeQuantize
            from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver
            
            # Extract quantization config from metadata
            qconfig_name = self.metadata.get('qconfig', 'lsq')
            logger.info(f"📋 Using quantization config: {qconfig_name}")
            
            # Create appropriate QConfig based on metadata
            if qconfig_name == 'lsq':
                # LSQ configuration - try to match training setup
                qconfig = self._create_lsq_qconfig()
            else:
                # Default configuration
                qconfig = self._create_default_qconfig()
            
            logger.info(f"⚙️ Applying quantization configuration...")
            
            # Apply qconfig to all quantizable layers
            quantized_layers = 0
            for name, module in self.fp32_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Skip detection head if it wasn't quantized
                    if any(skip_pattern in name for skip_pattern in ['detect', 'model.22']):
                        # Check if detection layers were quantized in our saved model
                        detect_params = [k for k in self.state_dict.keys() if name in k and 'scale' in k]
                        if not detect_params:
                            logger.info(f"⏭️ Skipping detection layer: {name}")
                            module.qconfig = None
                            continue
                    
                    module.qconfig = qconfig
                    quantized_layers += 1
                    logger.debug(f"Applied qconfig to {name}")
            
            logger.info(f"✅ Applied quantization config to {quantized_layers} layers")
            
            # Prepare model for QAT
            logger.info("🔧 Preparing model for quantization-aware training...")
            self.fp32_model.train()  # Must be in training mode for prepare_qat
            
            # Prepare for QAT
            prepared_model = torch.quantization.prepare_qat(self.fp32_model, inplace=False)
            
            # Convert to quantized model
            logger.info("🔄 Converting to quantized model...")
            prepared_model.eval()  # Must be in eval mode for convert
            self.quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            # Verify quantized model structure
            quantized_params = dict(self.quantized_model.named_parameters())
            logger.info(f"📊 Quantized model parameters: {len(quantized_params)}")
            
            # Check for quantization-specific parameters
            quant_specific = [k for k in quantized_params.keys() 
                            if any(x in k for x in ['scale', 'zero_point'])]
            logger.info(f"🎯 Quantization-specific parameters: {len(quant_specific)}")
            
            if len(quant_specific) == 0:
                logger.error("❌ No quantization parameters in converted model!")
                return False
            
            logger.info("✅ Quantization preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in quantization preparation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _create_lsq_qconfig(self) -> QConfig:
        """Create LSQ-style QConfig"""
        try:
            # Try to import LSQ components
            from torch.quantization.fake_quantize import FakeQuantize
            from torch.quantization.observer import MovingAverageMinMaxObserver, PerChannelMinMaxObserver
            
            # LSQ-inspired configuration
            activation_fake_quant = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0, quant_max=255,
                dtype=torch.quint8, qscheme=torch.per_tensor_affine
            )
            
            weight_fake_quant = FakeQuantize.with_args(
                observer=PerChannelMinMaxObserver,
                quant_min=-128, quant_max=127,
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
            )
            
            return QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
            
        except Exception as e:
            logger.warning(f"⚠️ LSQ config creation failed, using default: {e}")
            return self._create_default_qconfig()
    
    def _create_default_qconfig(self) -> QConfig:
        """Create default QConfig"""
        from torch.quantization import get_default_qat_qconfig
        return get_default_qat_qconfig()
    
    def step_1d_load_quantized_state_dict(self) -> bool:
        """Load INT8 state_dict into quantized model"""
        logger.info("=" * 60)
        logger.info("STEP 1D: LOAD QUANTIZED STATE DICT")
        logger.info("=" * 60)
        
        try:
            if self.quantized_model is None:
                logger.error("❌ No quantized model available!")
                return False
            
            logger.info("📥 Loading INT8 state_dict into quantized model...")
            
            # Get current model parameters
            current_params = dict(self.quantized_model.named_parameters())
            current_buffers = dict(self.quantized_model.named_buffers())
            all_current = {**current_params, **current_buffers}
            
            logger.info(f"🔍 Current model state:")
            logger.info(f"   Parameters: {len(current_params)}")
            logger.info(f"   Buffers: {len(current_buffers)}")
            logger.info(f"   Total: {len(all_current)}")
            
            # Check parameter compatibility
            saved_keys = set(self.state_dict.keys())
            current_keys = set(all_current.keys())
            
            missing_keys = current_keys - saved_keys
            extra_keys = saved_keys - current_keys
            
            if missing_keys:
                logger.warning(f"⚠️ Missing keys in saved model: {len(missing_keys)}")
                logger.debug(f"Missing: {list(missing_keys)[:5]}...")
            
            if extra_keys:
                logger.warning(f"⚠️ Extra keys in saved model: {len(extra_keys)}")
                logger.debug(f"Extra: {list(extra_keys)[:5]}...")
            
            # Attempt to load state_dict
            logger.info("🔄 Loading state_dict...")
            missing_keys, unexpected_keys = self.quantized_model.load_state_dict(
                self.state_dict, strict=False
            )
            
            if missing_keys:
                logger.warning(f"⚠️ Missing keys during load: {len(missing_keys)}")
                logger.debug(f"Missing: {missing_keys[:5]}...")
            
            if unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys during load: {len(unexpected_keys)}")
                logger.debug(f"Unexpected: {unexpected_keys[:5]}...")
            
            # Check if we have reasonable parameter overlap
            total_saved = len(self.state_dict)
            total_loaded = total_saved - len(unexpected_keys)
            load_ratio = total_loaded / total_saved if total_saved > 0 else 0
            
            logger.info(f"📊 Loading statistics:")
            logger.info(f"   Total saved parameters: {total_saved}")
            logger.info(f"   Successfully loaded: {total_loaded}")
            logger.info(f"   Load ratio: {load_ratio:.2%}")
            
            if load_ratio < 0.5:
                logger.error(f"❌ Too few parameters loaded ({load_ratio:.2%})")
                return False
            
            logger.info("✅ State dict loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading state dict: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step_1e_test_basic_functionality(self) -> bool:
        """Test basic inference functionality"""
        logger.info("=" * 60)
        logger.info("STEP 1E: TEST BASIC FUNCTIONALITY")
        logger.info("=" * 60)
        
        try:
            if self.quantized_model is None:
                logger.error("❌ No quantized model available!")
                return False
            
            # Move model to device
            self.quantized_model.to(self.device)
            self.quantized_model.eval()
            
            # Create test input
            test_input = torch.randn(1, 3, 640, 640).to(self.device)
            logger.info(f"📥 Created test input: {test_input.shape}")
            
            # Test inference
            logger.info("🧪 Testing inference...")
            with torch.no_grad():
                start_time = time.perf_counter()
                output = self.quantized_model(test_input)
                inference_time = time.perf_counter() - start_time
            
            logger.info(f"✅ Inference successful!")
            logger.info(f"⏱️ Inference time: {inference_time*1000:.2f} ms")
            
            # Analyze output
            if isinstance(output, (list, tuple)):
                logger.info(f"📤 Output: {len(output)} tensors")
                for i, out in enumerate(output):
                    if hasattr(out, 'shape'):
                        logger.info(f"   Output {i}: {out.shape}")
                        logger.info(f"   Range: [{out.min().item():.3f}, {out.max().item():.3f}]")
            else:
                if hasattr(output, 'shape'):
                    logger.info(f"📤 Output shape: {output.shape}")
                    logger.info(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Test multiple runs for stability
            logger.info("🔄 Testing inference stability...")
            times = []
            for i in range(10):
                with torch.no_grad():
                    start_time = time.perf_counter()
                    _ = self.quantized_model(test_input)
                    times.append(time.perf_counter() - start_time)
            
            mean_time = np.mean(times) * 1000
            std_time = np.std(times) * 1000
            logger.info(f"📊 Stability test (10 runs):")
            logger.info(f"   Mean time: {mean_time:.2f} ± {std_time:.2f} ms")
            logger.info(f"   FPS: {1000/mean_time:.1f}")
            
            logger.info("✅ Basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def step_1f_performance_benchmark(self) -> Dict:
        """Comprehensive performance benchmarking"""
        logger.info("=" * 60)
        logger.info("STEP 1F: PERFORMANCE BENCHMARKING")
        logger.info("=" * 60)
        
        try:
            if self.quantized_model is None:
                logger.error("❌ No quantized model available!")
                return {}
            
            # Benchmark parameters
            num_warmup = 20
            num_runs = 100
            batch_sizes = [1, 4]
            input_sizes = [(640, 640), (416, 416)]
            
            results = {}
            
            for batch_size in batch_sizes:
                for input_size in input_sizes:
                    logger.info(f"🏃 Benchmarking batch_size={batch_size}, input_size={input_size}")
                    
                    # Create test input
                    test_input = torch.randn(batch_size, 3, *input_size).to(self.device)
                    
                    # Warmup
                    logger.info(f"🔥 Warming up ({num_warmup} runs)...")
                    for _ in range(num_warmup):
                        with torch.no_grad():
                            _ = self.quantized_model(test_input)
                    
                    # Benchmark
                    logger.info(f"📊 Benchmarking ({num_runs} runs)...")
                    times = []
                    memory_usage = []
                    
                    for i in tqdm(range(num_runs), desc="Benchmarking"):
                        # Memory before
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                            mem_before = torch.cuda.memory_allocated()
                        else:
                            mem_before = psutil.Process().memory_info().rss
                        
                        # Inference
                        with torch.no_grad():
                            start_time = time.perf_counter()
                            output = self.quantized_model(test_input)
                            end_time = time.perf_counter()
                        
                        # Memory after
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                            mem_after = torch.cuda.memory_allocated()
                            mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
                        else:
                            mem_after = psutil.Process().memory_info().rss
                            mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
                        
                        times.append((end_time - start_time) * 1000)  # ms
                        memory_usage.append(mem_used)
                    
                    # Calculate statistics
                    config_key = f"bs{batch_size}_{input_size[0]}x{input_size[1]}"
                    results[config_key] = {
                        'batch_size': batch_size,
                        'input_size': input_size,
                        'mean_time_ms': np.mean(times),
                        'std_time_ms': np.std(times),
                        'min_time_ms': np.min(times),
                        'max_time_ms': np.max(times),
                        'fps': 1000 / np.mean(times) * batch_size,
                        'mean_memory_mb': np.mean(memory_usage),
                        'max_memory_mb': np.max(memory_usage),
                        'device': str(self.device)
                    }
                    
                    logger.info(f"📈 Results for {config_key}:")
                    logger.info(f"   Time: {results[config_key]['mean_time_ms']:.2f} ± {results[config_key]['std_time_ms']:.2f} ms")
                    logger.info(f"   FPS: {results[config_key]['fps']:.1f}")
                    logger.info(f"   Memory: {results[config_key]['mean_memory_mb']:.2f} MB")
            
            logger.info("✅ Performance benchmarking completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarking failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

def main():
    """Main Phase 1 execution"""
    print("="*80)
    print("🚀 PHASE 1: QUANTIZED YOLOV8 RECONSTRUCTION")
    print("="*80)
    
    # Configuration
    model_path = "models/checkpoints/qat_full_features_6/qat_yolov8n_full_int8_final.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return False
    
    # Initialize reconstructor
    reconstructor = QuantizedModelReconstructor(model_path)
    
    # Execute Phase 1 steps
    steps = [
        ("Step 1A: Load and Analyze", reconstructor.step_1a_load_and_analyze),
        ("Step 1B: Create Base Architecture", reconstructor.step_1b_create_base_architecture),
        ("Step 1C: Prepare Quantization", reconstructor.step_1c_prepare_quantization),
        ("Step 1D: Load Quantized State Dict", reconstructor.step_1d_load_quantized_state_dict),
        ("Step 1E: Test Basic Functionality", reconstructor.step_1e_test_basic_functionality),
    ]
    
    # Execute steps sequentially
    for step_name, step_func in steps:
        logger.info(f"\n🔄 Executing: {step_name}")
        success = step_func()
        if not success:
            logger.error(f"❌ {step_name} failed! Stopping execution.")
            return False
        logger.info(f"✅ {step_name} completed successfully")
    
    # Performance benchmarking (final step)
    logger.info(f"\n🔄 Executing: Step 1F: Performance Benchmarking")
    benchmark_results = reconstructor.step_1f_performance_benchmark()
    
    if not benchmark_results:
        logger.error("❌ Performance benchmarking failed!")
        return False
    
    # Phase 1 Summary
    logger.info("\n" + "="*60)
    logger.info("🎯 PHASE 1 SUMMARY")
    logger.info("="*60)
    
    print(f"\n✅ PHASE 1 COMPLETED SUCCESSFULLY!")
    print(f"\n📊 PERFORMANCE RESULTS:")
    
    for config, results in benchmark_results.items():
        print(f"\n🔧 Configuration: {config}")
        print(f"   Inference time: {results['mean_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
        print(f"   FPS: {results['fps']:.1f}")
        print(f"   Memory usage: {results['mean_memory_mb']:.2f} MB")
    
    # Pi 5 assessment
    best_config = min(benchmark_results.values(), key=lambda x: x['mean_time_ms'])
    pi5_estimated_time = best_config['mean_time_ms'] * 3  # Conservative estimate
    pi5_estimated_fps = 1000 / pi5_estimated_time
    
    print(f"\n🍓 RASPBERRY PI 5 ASSESSMENT:")
    print(f"   Estimated inference time: {pi5_estimated_time:.1f} ms")
    print(f"   Estimated FPS: {pi5_estimated_fps:.1f}")
    
    if pi5_estimated_fps >= 10:
        print(f"   Assessment: ✅ EXCELLENT for Pi 5 deployment")
    elif pi5_estimated_fps >= 5:
        print(f"   Assessment: ✅ GOOD for Pi 5 deployment")
    elif pi5_estimated_fps >= 3:
        print(f"   Assessment: ⚠️ MARGINAL for Pi 5 deployment")
    else:
        print(f"   Assessment: ❌ POOR for Pi 5 deployment")
    
    print(f"\n🎉 Phase 1 reconstruction completed successfully!")
    print(f"📝 Ready to proceed to Phase 2: Performance Validation")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n✅ Phase 1 execution completed successfully!")
        else:
            print(f"\n❌ Phase 1 execution failed!")
    except KeyboardInterrupt:
        print(f"\n⏹️ Phase 1 interrupted by user")
    except Exception as e:
        print(f"\n💥 Phase 1 failed with unexpected error: {e}")
        import traceback
        print(traceback.format_exc())