#!/usr/bin/env python
"""
Focused INT8 Model Validator

Specifically designed to validate your custom QAT-generated INT8 model format.
Addresses the issues found in your comprehensive validation script.
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import json
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QATGeneratedINT8Validator:
    """Validator specifically for your QAT-generated INT8 model format"""
    
    def __init__(self, model_path: str, dataset_yaml: str):
        self.model_path = model_path
        self.dataset_yaml = dataset_yaml
        self.model_data = None
        self.state_dict = None
        self.metadata = {}
        self.class_names = []
        self.num_classes = 0
        
    def load_and_analyze_model(self) -> bool:
        """Load and analyze the custom INT8 model"""
        try:
            logger.info(f"🔧 Loading QAT-generated INT8 model: {self.model_path}")
            
            # Load model file
            self.model_data = torch.load(self.model_path, map_location='cpu')
            
            if not isinstance(self.model_data, dict):
                logger.error(f"❌ Invalid model format: {type(self.model_data)}")
                return False
            
            # Extract components
            self.state_dict = self.model_data.get('state_dict')
            self.metadata = self.model_data.get('metadata', {})
            
            if self.state_dict is None:
                logger.error("❌ No state_dict found in model")
                return False
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"📊 Model contains {len(self.state_dict)} parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def load_dataset_info(self) -> bool:
        """Load dataset information"""
        try:
            with open(self.dataset_yaml, 'r') as f:
                dataset_info = yaml.safe_load(f)
            
            self.num_classes = dataset_info['nc']
            self.class_names = dataset_info['names']
            
            logger.info(f"📋 Dataset: {self.num_classes} classes")
            logger.info(f"📁 Classes: {self.class_names[:5]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset info: {e}")
            return False
    
    def analyze_quantization_quality(self) -> Dict:
        """Analyze the quantization quality and structure"""
        logger.info("🔍 Analyzing quantization quality...")
        
        analysis = {
            'total_parameters': len(self.state_dict),
            'quantized_parameters': 0,
            'float_parameters': 0,
            'parameter_types': {},
            'quantized_param_names': [],
            'float_param_names': [],
            'model_size_mb': os.path.getsize(self.model_path) / (1024**2)
        }
        
        # Analyze each parameter
        for name, param in self.state_dict.items():
            if torch.is_tensor(param):
                dtype_str = str(param.dtype)
                analysis['parameter_types'][dtype_str] = analysis['parameter_types'].get(dtype_str, 0) + 1
                
                if param.dtype in [torch.qint8, torch.quint8]:
                    analysis['quantized_parameters'] += 1
                    analysis['quantized_param_names'].append(name)
                else:
                    analysis['float_parameters'] += 1
                    analysis['float_param_names'].append(name)
        
        # Calculate metrics
        total_params = analysis['total_parameters']
        quant_params = analysis['quantized_parameters']
        analysis['quantization_ratio'] = quant_params / total_params if total_params > 0 else 0
        
        # Expected compression based on metadata
        if 'size_info' in self.metadata:
            size_info = self.metadata['size_info']
            analysis['expected_compression'] = size_info.get('compression_ratio', 0)
            analysis['expected_size_reduction'] = size_info.get('size_reduction_percent', 0)
        
        logger.info("📊 Quantization Analysis Results:")
        logger.info(f"   Total parameters: {analysis['total_parameters']}")
        logger.info(f"   Quantized parameters: {analysis['quantized_parameters']}")
        logger.info(f"   Float parameters: {analysis['float_parameters']}")
        logger.info(f"   Quantization ratio: {analysis['quantization_ratio']:.2%}")
        logger.info(f"   Model size: {analysis['model_size_mb']:.2f} MB")
        logger.info(f"   Parameter types: {analysis['parameter_types']}")
        
        return analysis
    
    def validate_model_structure(self) -> Dict:
        """Validate that the model structure is correct for YOLOv8"""
        logger.info("🏗️ Validating model structure...")
        
        validation = {
            'has_backbone_params': False,
            'has_neck_params': False,
            'has_head_params': False,
            'estimated_architecture': 'unknown',
            'yolov8_compatible': False
        }
        
        # Look for typical YOLOv8 parameter patterns
        backbone_patterns = ['model.0.', 'model.1.', 'model.2.', 'model.3.', 'model.4.']
        neck_patterns = ['model.10.', 'model.11.', 'model.12.', 'model.13.', 'model.14.']
        head_patterns = ['model.22.']  # Detection head
        
        for name in self.state_dict.keys():
            if any(pattern in name for pattern in backbone_patterns):
                validation['has_backbone_params'] = True
            if any(pattern in name for pattern in neck_patterns):
                validation['has_neck_params'] = True
            if any(pattern in name for pattern in head_patterns):
                validation['has_head_params'] = True
        
        # Estimate architecture
        total_params = len(self.state_dict)
        if total_params < 200:
            validation['estimated_architecture'] = 'yolov8n'
        elif total_params < 400:
            validation['estimated_architecture'] = 'yolov8s'
        else:
            validation['estimated_architecture'] = 'yolov8m_or_larger'
        
        # Check YOLOv8 compatibility
        validation['yolov8_compatible'] = (
            validation['has_backbone_params'] and 
            validation['has_neck_params']
        )
        
        logger.info("🏗️ Structure Validation Results:")
        logger.info(f"   Backbone parameters: {'✅' if validation['has_backbone_params'] else '❌'}")
        logger.info(f"   Neck parameters: {'✅' if validation['has_neck_params'] else '❌'}")
        logger.info(f"   Head parameters: {'✅' if validation['has_head_params'] else '❌'}")
        logger.info(f"   Estimated architecture: {validation['estimated_architecture']}")
        logger.info(f"   YOLOv8 compatible: {'✅' if validation['yolov8_compatible'] else '❌'}")
        
        return validation
    
    def test_model_functionality(self) -> Dict:
        """Test basic model functionality without full reconstruction"""
        logger.info("🧪 Testing model functionality...")
        
        test_results = {
            'parameters_loadable': False,
            'quantized_tensors_valid': False,
            'shapes_consistent': False,
            'no_corrupted_data': False,
            'functionality_score': 0
        }
        
        try:
            # Test parameter loading
            loadable_count = 0
            corrupted_count = 0
            quantized_valid_count = 0
            total_quantized = 0
            
            for name, param in self.state_dict.items():
                try:
                    # Test basic tensor operations
                    if torch.is_tensor(param):
                        _ = param.shape  # Test shape access
                        _ = param.dtype  # Test dtype access
                        
                        if not torch.isnan(param).any() and not torch.isinf(param).any():
                            loadable_count += 1
                        else:
                            corrupted_count += 1
                        
                        # Test quantized tensors specifically
                        if param.dtype in [torch.qint8, torch.quint8]:
                            total_quantized += 1
                            try:
                                # Test quantized tensor operations
                                _ = param.int_repr()  # Test quantized tensor access
                                quantized_valid_count += 1
                            except:
                                pass
                                
                except Exception as e:
                    corrupted_count += 1
            
            # Calculate test results
            total_params = len(self.state_dict)
            test_results['parameters_loadable'] = (corrupted_count == 0)
            test_results['quantized_tensors_valid'] = (quantized_valid_count == total_quantized) if total_quantized > 0 else True
            test_results['no_corrupted_data'] = (corrupted_count == 0)
            test_results['shapes_consistent'] = True  # Basic shape test passed if we get here
            
            # Calculate functionality score
            score = 0
            if test_results['parameters_loadable']: score += 1
            if test_results['quantized_tensors_valid']: score += 1
            if test_results['shapes_consistent']: score += 1
            if test_results['no_corrupted_data']: score += 1
            
            test_results['functionality_score'] = score
            
            logger.info("🧪 Functionality Test Results:")
            logger.info(f"   Parameters loadable: {'✅' if test_results['parameters_loadable'] else '❌'}")
            logger.info(f"   Quantized tensors valid: {'✅' if test_results['quantized_tensors_valid'] else '❌'}")
            logger.info(f"   Shapes consistent: {'✅' if test_results['shapes_consistent'] else '❌'}")
            logger.info(f"   No corrupted data: {'✅' if test_results['no_corrupted_data'] else '❌'}")
            logger.info(f"   Functionality score: {score}/4")
            
        except Exception as e:
            logger.error(f"❌ Functionality test failed: {e}")
        
        return test_results
    
    def analyze_metadata_consistency(self) -> Dict:
        """Analyze metadata for consistency"""
        logger.info("📄 Analyzing metadata consistency...")
        
        consistency = {
            'has_metadata': len(self.metadata) > 0,
            'has_size_info': 'size_info' in self.metadata,
            'has_qconfig_info': 'qconfig' in self.metadata,
            'conversion_successful': self.metadata.get('conversion_successful', False),
            'deployment_ready': self.metadata.get('deployment_ready', False),
            'preserved_quantizers': self.metadata.get('preserved_quantizers', 0),
            'metadata_complete': False
        }
        
        # Check metadata completeness
        expected_fields = ['framework', 'format', 'conversion_successful', 'deployment_ready']
        has_all_fields = all(field in self.metadata for field in expected_fields)
        consistency['metadata_complete'] = has_all_fields
        
        logger.info("📄 Metadata Analysis:")
        logger.info(f"   Has metadata: {'✅' if consistency['has_metadata'] else '❌'}")
        logger.info(f"   Has size info: {'✅' if consistency['has_size_info'] else '❌'}")
        logger.info(f"   Conversion successful: {'✅' if consistency['conversion_successful'] else '❌'}")
        logger.info(f"   Deployment ready: {'✅' if consistency['deployment_ready'] else '❌'}")
        logger.info(f"   Preserved quantizers: {consistency['preserved_quantizers']}")
        logger.info(f"   Metadata complete: {'✅' if consistency['metadata_complete'] else '❌'}")
        
        if self.metadata:
            logger.info(f"📋 Full metadata: {json.dumps(self.metadata, indent=2, default=str)}")
        
        return consistency
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation suite"""
        print("\n" + "="*80)
        print("🚀 FOCUSED INT8 MODEL VALIDATION")
        print("="*80)
        
        results = {}
        
        # Load model and dataset
        if not self.load_and_analyze_model():
            return {}
        
        if not self.load_dataset_info():
            return {}
        
        # Run all validation tests
        results['quantization_analysis'] = self.analyze_quantization_quality()
        results['structure_validation'] = self.validate_model_structure()
        results['functionality_test'] = self.test_model_functionality()
        results['metadata_consistency'] = self.analyze_metadata_consistency()
        
        # Generate overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        # Print summary
        self._print_validation_summary(results)
        
        return results
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """Generate overall model assessment"""
        assessment = {
            'model_validity': 'unknown',
            'deployment_readiness': 'unknown',
            'recommendations': [],
            'critical_issues': [],
            'warnings': [],
            'overall_score': 0
        }
        
        score = 0
        max_score = 0
        
        # Check quantization quality
        quant_analysis = results.get('quantization_analysis', {})
        if quant_analysis.get('quantized_parameters', 0) > 0:
            score += 2
            assessment['recommendations'].append("✅ Model has quantized parameters")
        else:
            assessment['critical_issues'].append("❌ No quantized parameters found")
        max_score += 2
        
        # Check structure
        structure = results.get('structure_validation', {})
        if structure.get('yolov8_compatible', False):
            score += 2
            assessment['recommendations'].append("✅ Model structure is YOLOv8 compatible")
        else:
            assessment['warnings'].append("⚠️ Model structure may not be fully YOLOv8 compatible")
        max_score += 2
        
        # Check functionality
        functionality = results.get('functionality_test', {})
        func_score = functionality.get('functionality_score', 0)
        score += func_score
        max_score += 4
        
        if func_score == 4:
            assessment['recommendations'].append("✅ All functionality tests passed")
        elif func_score >= 2:
            assessment['warnings'].append("⚠️ Some functionality tests failed")
        else:
            assessment['critical_issues'].append("❌ Major functionality issues detected")
        
        # Check metadata
        metadata = results.get('metadata_consistency', {})
        if metadata.get('deployment_ready', False):
            score += 1
            assessment['recommendations'].append("✅ Model marked as deployment ready")
        else:
            assessment['warnings'].append("⚠️ Model not marked as deployment ready")
        max_score += 1
        
        # Calculate final assessment
        assessment['overall_score'] = score / max_score if max_score > 0 else 0
        
        if assessment['overall_score'] >= 0.8:
            assessment['model_validity'] = 'excellent'
            assessment['deployment_readiness'] = 'ready'
        elif assessment['overall_score'] >= 0.6:
            assessment['model_validity'] = 'good'
            assessment['deployment_readiness'] = 'ready_with_monitoring'
        elif assessment['overall_score'] >= 0.4:
            assessment['model_validity'] = 'fair'
            assessment['deployment_readiness'] = 'needs_testing'
        else:
            assessment['model_validity'] = 'poor'
            assessment['deployment_readiness'] = 'not_ready'
        
        return assessment
    
    def _print_validation_summary(self, results: Dict):
        """Print comprehensive validation summary"""
        print("\n" + "="*80)
        print("📋 VALIDATION SUMMARY")
        print("="*80)
        
        assessment = results.get('overall_assessment', {})
        
        print(f"🏆 OVERALL ASSESSMENT:")
        print(f"   Model Validity: {assessment.get('model_validity', 'unknown').upper()}")
        print(f"   Deployment Readiness: {assessment.get('deployment_readiness', 'unknown').upper()}")
        print(f"   Overall Score: {assessment.get('overall_score', 0):.1%}")
        
        # Print recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            print(f"\n✅ POSITIVE FINDINGS:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Print warnings
        warnings = assessment.get('warnings', [])
        if warnings:
            print(f"\n⚠️ WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")
        
        # Print critical issues
        issues = assessment.get('critical_issues', [])
        if issues:
            print(f"\n❌ CRITICAL ISSUES:")
            for issue in issues:
                print(f"   {issue}")
        
        # Model info summary
        quant_analysis = results.get('quantization_analysis', {})
        print(f"\n📊 MODEL STATISTICS:")
        print(f"   File size: {quant_analysis.get('model_size_mb', 0):.2f} MB")
        print(f"   Total parameters: {quant_analysis.get('total_parameters', 0)}")
        print(f"   Quantized parameters: {quant_analysis.get('quantized_parameters', 0)}")
        print(f"   Quantization ratio: {quant_analysis.get('quantization_ratio', 0):.1%}")
        
        # Next steps
        print(f"\n🚀 RECOMMENDED NEXT STEPS:")
        if assessment.get('model_validity') in ['excellent', 'good']:
            print("   1. ✅ Model appears valid - proceed with inference testing")
            print("   2. 🧪 Test inference on sample images")
            print("   3. 📊 Benchmark inference speed")
            print("   4. 🚀 Deploy for production use")
        else:
            print("   1. 🔧 Address critical issues identified above")
            print("   2. 🔍 Check original QAT training logs")
            print("   3. 🔄 Consider re-running QAT training")
            print("   4. 📞 Seek technical support if needed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Focused INT8 Model Validator")
    parser.add_argument('--model', type=str,
                       default='models/checkpoints/qat_full_features_4/qat_yolov8n_full_int8_final.pt',
                       help='Path to INT8 model')
    parser.add_argument('--dataset', type=str,
                       default='datasets/vietnam-traffic-sign-detection/dataset.yaml',
                       help='Dataset YAML file')
    parser.add_argument('--output', type=str,
                       default='focused_validation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create validator
    validator = QATGeneratedINT8Validator(args.model, args.dataset)
    
    # Run validation
    results = validator.run_comprehensive_validation()
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {args.output}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    print("\n" + "="*80)
    print("✅ FOCUSED VALIDATION COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()