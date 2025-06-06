#!/usr/bin/env python
"""
Comprehensive INT8 Model Validation Script

This script provides complete performance evaluation for your INT8 quantized model:
- mAP50, mAP50-95, F1, Precision, Recall
- Inference speed benchmarks
- Memory usage analysis
- Comparison with baseline
- Detailed per-class metrics
"""

import os
import sys
import torch
import time
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
    from ultralytics.data import build_dataset
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics not available, using basic validation")
    ULTRALYTICS_AVAILABLE = False

class INT8ModelValidator:
    """Comprehensive validator for INT8 quantized YOLOv8 models"""
    
    def __init__(self, int8_model_path, dataset_yaml, baseline_model_path=None):
        self.int8_model_path = int8_model_path
        self.dataset_yaml = dataset_yaml
        self.baseline_model_path = baseline_model_path
        
        self.int8_model = None
        self.baseline_model = None
        self.dataset_info = None
        self.results = {}
        
        # Load dataset info
        self._load_dataset_info()
        
        # Initialize models
        self._load_int8_model()
        if baseline_model_path:
            self._load_baseline_model()
    
    def _load_dataset_info(self):
        """Load dataset configuration"""
        try:
            with open(self.dataset_yaml, 'r') as f:
                self.dataset_info = yaml.safe_load(f)
            
            self.num_classes = self.dataset_info['nc']
            self.class_names = self.dataset_info['names']
            
            print(f"📊 Dataset loaded: {self.num_classes} classes")
            print(f"📁 Classes: {list(self.class_names)[:5]}..." if len(self.class_names) > 5 else f"📁 Classes: {list(self.class_names)}")
            
        except Exception as e:
            print(f"❌ Failed to load dataset info: {e}")
            sys.exit(1)
    
    def _load_int8_model(self):
        """Load and prepare INT8 model for validation"""
        print(f"🔧 Loading INT8 model: {self.int8_model_path}")
        
        try:
            # Load the INT8 model data
            model_data = torch.load(self.int8_model_path, map_location='cpu')
            
            if isinstance(model_data, dict) and 'state_dict' in model_data:
                state_dict = model_data['state_dict']
                metadata = model_data.get('metadata', {})
                
                print(f"✅ INT8 model loaded successfully")
                print(f"📊 Metadata: {metadata}")
                
                # Create YOLOv8 model with correct architecture
                if ULTRALYTICS_AVAILABLE:
                    # Create base YOLOv8 model
                    self.int8_model = YOLO('yolov8n.pt')
                    
                    # Load the quantized state dict
                    try:
                        self.int8_model.model.load_state_dict(state_dict, strict=False)
                        print(f"✅ Quantized weights loaded into YOLOv8 architecture")
                    except Exception as e:
                        print(f"⚠️ Partial weight loading: {e}")
                        # Try loading with strict=False
                        missing_keys, unexpected_keys = self.int8_model.model.load_state_dict(state_dict, strict=False)
                        print(f"⚠️ Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    
                    # Set to evaluation mode
                    self.int8_model.model.eval()
                else:
                    print("❌ Ultralytics not available, cannot create full validation")
                    return False
                
            else:
                print(f"❌ Unexpected model format: {type(model_data)}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load INT8 model: {e}")
            import traceback
            print(traceback.format_exc())
            return False
        
        return True
    
    def _load_baseline_model(self):
        """Load baseline FP32 model for comparison"""
        if not self.baseline_model_path or not os.path.exists(self.baseline_model_path):
            print(f"⚠️ Baseline model not found: {self.baseline_model_path}")
            return False
        
        try:
            print(f"🔧 Loading baseline model: {self.baseline_model_path}")
            
            if ULTRALYTICS_AVAILABLE:
                self.baseline_model = YOLO(self.baseline_model_path)
                print(f"✅ Baseline model loaded successfully")
            else:
                print("❌ Ultralytics not available")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load baseline model: {e}")
            return False
        
        return True
    
    def run_comprehensive_validation(self):
        """Run complete validation suite"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE INT8 MODEL VALIDATION")
        print("="*80)
        
        # 1. Basic model info
        self._analyze_model_structure()
        
        # 2. Inference speed benchmark
        speed_results = self._benchmark_inference_speed()
        self.results['speed'] = speed_results
        
        # 3. Memory usage analysis
        memory_results = self._analyze_memory_usage()
        self.results['memory'] = memory_results
        
        # 4. Accuracy validation
        if ULTRALYTICS_AVAILABLE:
            accuracy_results = self._validate_accuracy()
            self.results['accuracy'] = accuracy_results
        else:
            print("⚠️ Skipping accuracy validation (Ultralytics not available)")
        
        # 5. Compare with baseline
        if self.baseline_model:
            comparison_results = self._compare_with_baseline()
            self.results['comparison'] = comparison_results
        
        # 6. Generate comprehensive report
        self._generate_report()
        
        return self.results
    
    def _analyze_model_structure(self):
        """Analyze model structure and quantization"""
        print(f"\n📊 MODEL STRUCTURE ANALYSIS")
        print("-" * 50)
        
        try:
            # Load model data for analysis
            model_data = torch.load(self.int8_model_path, map_location='cpu')
            state_dict = model_data['state_dict']
            metadata = model_data.get('metadata', {})
            
            # Basic info
            total_params = len(state_dict)
            file_size_mb = os.path.getsize(self.int8_model_path) / (1024**2)
            
            # Count quantized parameters
            quantized_params = 0
            param_types = {}
            
            for name, param in state_dict.items():
                if torch.is_tensor(param):
                    dtype_str = str(param.dtype)
                    param_types[dtype_str] = param_types.get(dtype_str, 0) + 1
                    
                    if param.dtype in [torch.qint8, torch.quint8]:
                        quantized_params += 1
            
            print(f"📦 Total parameters: {total_params}")
            print(f"🔥 Quantized parameters: {quantized_params}")
            print(f"📏 File size: {file_size_mb:.2f} MB")
            print(f"📋 Parameter types: {param_types}")
            
            # Metadata analysis
            if metadata:
                print(f"\n📄 Metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
            
            structure_info = {
                'total_params': total_params,
                'quantized_params': quantized_params,
                'file_size_mb': file_size_mb,
                'param_types': param_types,
                'metadata': metadata
            }
            
            self.results['structure'] = structure_info
            
        except Exception as e:
            print(f"❌ Structure analysis failed: {e}")
    
    def _benchmark_inference_speed(self, num_warmup=20, num_runs=100):
        """Comprehensive inference speed benchmark"""
        print(f"\n⏱️ INFERENCE SPEED BENCHMARK")
        print("-" * 50)
        
        if not self.int8_model:
            print("❌ No model loaded")
            return {}
        
        try:
            # Prepare test input
            test_input = torch.randn(1, 3, 640, 640)
            
            if hasattr(self.int8_model, 'model'):
                model = self.int8_model.model
            else:
                model = self.int8_model
            
            model.eval()
            
            # Warmup
            print(f"🔥 Warming up ({num_warmup} runs)...")
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            print(f"📊 Benchmarking ({num_runs} runs)...")
            times = []
            
            for _ in tqdm(range(num_runs), desc="Benchmarking"):
                start_time = time.perf_counter()
                with torch.no_grad():
                    output = model(test_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times_ms = [t * 1000 for t in times]
            stats = {
                'mean_ms': np.mean(times_ms),
                'std_ms': np.std(times_ms),
                'min_ms': np.min(times_ms),
                'max_ms': np.max(times_ms),
                'median_ms': np.median(times_ms),
                'p95_ms': np.percentile(times_ms, 95),
                'p99_ms': np.percentile(times_ms, 99),
                'fps': 1.0 / np.mean(times),
                'throughput_imgs_per_sec': 1.0 / np.mean(times)
            }
            
            print(f"📊 Speed Results:")
            print(f"   Average: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
            print(f"   Median: {stats['median_ms']:.2f} ms")
            print(f"   Min/Max: {stats['min_ms']:.2f}/{stats['max_ms']:.2f} ms")
            print(f"   95th/99th percentile: {stats['p95_ms']:.2f}/{stats['p99_ms']:.2f} ms")
            print(f"   🚀 FPS: {stats['fps']:.1f}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Speed benchmark failed: {e}")
            return {}
    
    def _analyze_memory_usage(self):
        """Analyze memory usage"""
        print(f"\n💾 MEMORY USAGE ANALYSIS")
        print("-" * 50)
        
        try:
            import psutil
            import gc
            
            # Get current memory
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)
            
            # Model memory
            if self.int8_model:
                if hasattr(self.int8_model, 'model'):
                    model = self.int8_model.model
                else:
                    model = self.int8_model
                
                # Calculate model memory
                model_memory = 0
                for param in model.parameters():
                    model_memory += param.numel() * param.element_size()
                
                model_memory_mb = model_memory / (1024**2)
                
                # Test inference memory
                test_input = torch.randn(1, 3, 640, 640)
                
                gc.collect()
                memory_before_inference = process.memory_info().rss / (1024**2)
                
                with torch.no_grad():
                    output = model(test_input)
                
                memory_after_inference = process.memory_info().rss / (1024**2)
                inference_memory = memory_after_inference - memory_before_inference
                
                memory_stats = {
                    'model_parameters_mb': model_memory_mb,
                    'process_memory_mb': memory_after_inference,
                    'inference_overhead_mb': inference_memory,
                    'file_size_mb': os.path.getsize(self.int8_model_path) / (1024**2)
                }
                
                print(f"📊 Memory Usage:")
                print(f"   Model parameters: {memory_stats['model_parameters_mb']:.2f} MB")
                print(f"   Process memory: {memory_stats['process_memory_mb']:.2f} MB")
                print(f"   Inference overhead: {memory_stats['inference_overhead_mb']:.2f} MB")
                print(f"   File size: {memory_stats['file_size_mb']:.2f} MB")
                
                return memory_stats
            
        except ImportError:
            print("⚠️ psutil not available, skipping memory analysis")
        except Exception as e:
            print(f"❌ Memory analysis failed: {e}")
        
        return {}
    
    def _validate_accuracy(self):
        """Validate model accuracy using ultralytics"""
        print(f"\n🎯 ACCURACY VALIDATION")
        print("-" * 50)
        
        if not ULTRALYTICS_AVAILABLE:
            print("❌ Ultralytics not available")
            return {}
        
        try:
            # Run validation using ultralytics
            print("🔍 Running validation on dataset...")
            
            results = self.int8_model.val(
                data=self.dataset_yaml,
                imgsz=640,
                batch=16,
                device='cpu',
                verbose=False,
                save=False,
                plots=False
            )
            
            # Extract metrics
            metrics = {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr)) if (results.box.mp + results.box.mr) > 0 else 0.0
            }
            
            # Per-class metrics if available
            if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
                per_class_map50 = {}
                per_class_map50_95 = {}
                
                for i, class_idx in enumerate(results.box.ap_class_index):
                    class_name = self.class_names[int(class_idx)]
                    if len(results.box.ap.shape) > 1:
                        per_class_map50[class_name] = float(results.box.ap[i, 0])  # AP@0.5
                        per_class_map50_95[class_name] = float(results.box.ap[i, :].mean())  # AP@0.5:0.95
                
                metrics['per_class_map50'] = per_class_map50
                metrics['per_class_map50_95'] = per_class_map50_95
            
            print(f"📊 Accuracy Results:")
            print(f"   mAP@0.5: {metrics['map50']:.4f}")
            print(f"   mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Accuracy validation failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {}
    
    def _compare_with_baseline(self):
        """Compare INT8 model with baseline FP32 model"""
        print(f"\n⚖️ BASELINE COMPARISON")
        print("-" * 50)
        
        if not self.baseline_model:
            print("❌ No baseline model loaded")
            return {}
        
        try:
            # Get baseline results
            print("🔍 Validating baseline model...")
            baseline_results = self.baseline_model.val(
                data=self.dataset_yaml,
                imgsz=640,
                batch=16,
                device='cpu',
                verbose=False,
                save=False,
                plots=False
            )
            
            baseline_metrics = {
                'map50': float(baseline_results.box.map50),
                'map50_95': float(baseline_results.box.map),
                'precision': float(baseline_results.box.mp),
                'recall': float(baseline_results.box.mr),
            }
            
            # Speed comparison
            print("🏃 Benchmarking baseline speed...")
            baseline_speed = self._benchmark_model_speed(self.baseline_model.model)
            
            # INT8 metrics (should already be in results)
            int8_metrics = self.results.get('accuracy', {})
            int8_speed = self.results.get('speed', {})
            
            # Calculate comparisons
            comparison = {
                'baseline': baseline_metrics,
                'int8': {
                    'map50': int8_metrics.get('map50', 0),
                    'map50_95': int8_metrics.get('map50_95', 0),
                    'precision': int8_metrics.get('precision', 0),
                    'recall': int8_metrics.get('recall', 0),
                },
                'differences': {},
                'speed_comparison': {
                    'baseline_fps': baseline_speed.get('fps', 0),
                    'int8_fps': int8_speed.get('fps', 0),
                    'speedup': int8_speed.get('fps', 0) / baseline_speed.get('fps', 1) if baseline_speed.get('fps', 0) > 0 else 0
                }
            }
            
            # Calculate metric differences
            for metric in ['map50', 'map50_95', 'precision', 'recall']:
                baseline_val = baseline_metrics.get(metric, 0)
                int8_val = int8_metrics.get(metric, 0)
                comparison['differences'][metric] = int8_val - baseline_val
                comparison['differences'][f'{metric}_percent'] = ((int8_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            print(f"📊 Comparison Results:")
            print(f"   Baseline mAP@0.5: {baseline_metrics['map50']:.4f}")
            print(f"   INT8 mAP@0.5: {int8_metrics.get('map50', 0):.4f}")
            print(f"   Difference: {comparison['differences']['map50']:.4f} ({comparison['differences']['map50_percent']:.1f}%)")
            print(f"   Speed improvement: {comparison['speed_comparison']['speedup']:.1f}x")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Baseline comparison failed: {e}")
            return {}
    
    def _benchmark_model_speed(self, model, num_runs=50):
        """Benchmark a specific model's speed"""
        try:
            model.eval()
            test_input = torch.randn(1, 3, 640, 640)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _ = model(test_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            return {
                'mean_ms': np.mean(times) * 1000,
                'fps': 1.0 / np.mean(times)
            }
        except Exception as e:
            print(f"⚠️ Speed benchmark failed: {e}")
            return {}
    
    def _generate_report(self):
        """Generate comprehensive validation report"""
        print(f"\n📋 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)
        
        # Model Overview
        structure = self.results.get('structure', {})
        print(f"🏗️ MODEL STRUCTURE:")
        print(f"   File: {os.path.basename(self.int8_model_path)}")
        print(f"   Size: {structure.get('file_size_mb', 0):.2f} MB")
        print(f"   Quantized params: {structure.get('quantized_params', 0)}/{structure.get('total_params', 0)}")
        
        # Performance Overview
        accuracy = self.results.get('accuracy', {})
        speed = self.results.get('speed', {})
        
        print(f"\n🎯 ACCURACY METRICS:")
        print(f"   mAP@0.5: {accuracy.get('map50', 0):.4f}")
        print(f"   mAP@0.5:0.95: {accuracy.get('map50_95', 0):.4f}")
        print(f"   F1-Score: {accuracy.get('f1', 0):.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Average inference: {speed.get('mean_ms', 0):.2f} ms")
        print(f"   FPS: {speed.get('fps', 0):.1f}")
        print(f"   95th percentile: {speed.get('p95_ms', 0):.2f} ms")
        
        # Comparison with baseline
        comparison = self.results.get('comparison', {})
        if comparison:
            print(f"\n⚖️ BASELINE COMPARISON:")
            differences = comparison.get('differences', {})
            speed_comp = comparison.get('speed_comparison', {})
            
            print(f"   mAP@0.5 change: {differences.get('map50_percent', 0):.1f}%")
            print(f"   Speed improvement: {speed_comp.get('speedup', 0):.1f}x")
        
        # Quality Assessment
        print(f"\n🏆 QUALITY ASSESSMENT:")
        map50 = accuracy.get('map50', 0)
        fps = speed.get('fps', 0)
        
        if map50 > 0.75:
            accuracy_grade = "🟢 Excellent"
        elif map50 > 0.65:
            accuracy_grade = "🟡 Good"
        elif map50 > 0.5:
            accuracy_grade = "🟠 Fair"
        else:
            accuracy_grade = "🔴 Poor"
        
        if fps > 200:
            speed_grade = "🟢 Very Fast"
        elif fps > 100:
            speed_grade = "🟡 Fast"
        elif fps > 50:
            speed_grade = "🟠 Moderate"
        else:
            speed_grade = "🔴 Slow"
        
        print(f"   Accuracy: {accuracy_grade} (mAP@0.5: {map50:.3f})")
        print(f"   Speed: {speed_grade} ({fps:.1f} FPS)")
        
        # Deployment Readiness
        print(f"\n🚀 DEPLOYMENT READINESS:")
        
        deployment_score = 0
        if map50 > 0.7: deployment_score += 1
        if fps > 100: deployment_score += 1
        if structure.get('file_size_mb', 10) < 10: deployment_score += 1
        
        if deployment_score == 3:
            deployment_status = "✅ Ready for Production"
        elif deployment_score == 2:
            deployment_status = "🟡 Ready with Monitoring"
        elif deployment_score == 1:
            deployment_status = "🟠 Needs Optimization"
        else:
            deployment_status = "🔴 Not Ready"
        
        print(f"   Status: {deployment_status}")
        print(f"   Score: {deployment_score}/3")
    
    def save_results(self, output_file):
        """Save validation results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive INT8 Model Validation")
    parser.add_argument('--int8-model', type=str, 
                       default='models/checkpoints/qat_full_features_4/qat_yolov8n_full_int8_final.pt',
                       help='Path to INT8 model')
    parser.add_argument('--dataset', type=str,
                       default='datasets/vietnam-traffic-sign-detection/dataset.yaml',
                       help='Dataset YAML file')
    parser.add_argument('--baseline', type=str,
                       default='models/checkpoints/qat_full_features_4/weights/best.pt',
                       help='Baseline model for comparison')
    parser.add_argument('--output', type=str,
                       default='int8_validation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create validator
    validator = INT8ModelValidator(
        int8_model_path=args.int8_model,
        dataset_yaml=args.dataset,
        baseline_model_path=args.baseline if os.path.exists(args.baseline) else None
    )
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    validator.save_results(args.output)
    
    print(f"\n🎉 Validation completed successfully!")
    return results

if __name__ == "__main__":
    main()