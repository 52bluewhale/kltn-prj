#!/usr/bin/env python
"""
Diagnose Quantization Issues

This script diagnoses problems with your QAT to INT8 conversion process.
"""

import os
import torch
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_qat_model(qat_model_path):
    """Analyze the QAT model to see if it's ready for conversion."""
    logger.info(f"🔍 Analyzing QAT model: {qat_model_path}")
    
    try:
        qat_data = torch.load(qat_model_path, map_location='cpu')
        
        analysis = {
            'file_exists': True,
            'file_size_mb': round(os.path.getsize(qat_model_path) / (1024 * 1024), 2),
            'contains_fake_quantizers': False,
            'fake_quantizer_count': 0,
            'observer_states': {},
            'ready_for_conversion': False
        }
        
        # Check if it's saved with preservation info
        if isinstance(qat_data, dict) and 'fake_quant_count' in qat_data:
            analysis['fake_quantizer_count'] = qat_data['fake_quant_count']
            analysis['contains_fake_quantizers'] = qat_data['fake_quant_count'] > 0
            analysis['preservation_info'] = qat_data.get('qat_info', {})
            
            logger.info(f"✅ Found preservation info: {qat_data['fake_quant_count']} FakeQuantizers")
        
        # Try to extract the model
        if isinstance(qat_data, dict):
            model = None
            for key in ['model', 'model_state_dict', 'state_dict']:
                if key in qat_data:
                    if key == 'model':
                        model = qat_data[key]
                        break
            
            if model is not None:
                # Analyze FakeQuantizers in the model
                fake_quantizers = []
                observer_ready_count = 0
                
                for name, module in model.named_modules():
                    if 'FakeQuantize' in type(module).__name__:
                        fake_quantizers.append(name)
                        
                        # Check if observers are ready
                        if hasattr(module, 'activation_post_process'):
                            observer = module.activation_post_process
                            if hasattr(observer, 'min_val') and hasattr(observer, 'max_val'):
                                if observer.min_val != float('inf') and observer.max_val != float('-inf'):
                                    observer_ready_count += 1
                
                analysis['fake_quantizer_count'] = len(fake_quantizers)
                analysis['contains_fake_quantizers'] = len(fake_quantizers) > 0
                analysis['observer_ready_count'] = observer_ready_count
                analysis['ready_for_conversion'] = observer_ready_count > 0
                
                logger.info(f"📊 Model analysis:")
                logger.info(f"   FakeQuantizers: {len(fake_quantizers)}")
                logger.info(f"   Ready observers: {observer_ready_count}")
                logger.info(f"   Ready for conversion: {analysis['ready_for_conversion']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ QAT analysis failed: {e}")
        return {'file_exists': False, 'error': str(e)}

def analyze_int8_model(int8_model_path):
    """Analyze the INT8 model to see if conversion worked."""
    logger.info(f"🔍 Analyzing INT8 model: {int8_model_path}")
    
    try:
        int8_data = torch.load(int8_model_path, map_location='cpu')
        
        analysis = {
            'file_exists': True,
            'file_size_mb': round(os.path.getsize(int8_model_path) / (1024 * 1024), 2),
            'contains_quantized_tensors': False,
            'quantized_tensor_count': 0,
            'tensor_analysis': {},
            'conversion_successful': False
        }
        
        # Extract state dict
        state_dict = None
        if isinstance(int8_data, dict):
            for key in ['model', 'model_state_dict', 'state_dict']:
                if key in int8_data:
                    if key == 'model':
                        # If it's a full model, get its state dict
                        if hasattr(int8_data[key], 'state_dict'):
                            state_dict = int8_data[key].state_dict()
                        else:
                            state_dict = int8_data[key]
                    else:
                        state_dict = int8_data[key]
                    break
        else:
            if hasattr(int8_data, 'state_dict'):
                state_dict = int8_data.state_dict()
            else:
                state_dict = int8_data
        
        if state_dict is not None:
            quantized_tensors = []
            fp32_tensors = []
            quantized_ops = []
            
            for name, param in state_dict.items():
                if torch.is_tensor(param):
                    if param.is_quantized:
                        quantized_tensors.append({
                            'name': name,
                            'shape': list(param.shape),
                            'dtype': str(param.dtype),
                            'qscheme': str(param.qscheme())
                        })
                    else:
                        fp32_tensors.append({
                            'name': name,
                            'shape': list(param.shape),
                            'dtype': str(param.dtype)
                        })
                elif hasattr(param, '__dict__'):
                    # Check for quantized operations
                    param_type = type(param).__name__
                    if 'Quantized' in param_type or 'Int8' in param_type:
                        quantized_ops.append({
                            'name': name,
                            'type': param_type
                        })
            
            analysis['quantized_tensor_count'] = len(quantized_tensors)
            analysis['contains_quantized_tensors'] = len(quantized_tensors) > 0
            analysis['fp32_tensor_count'] = len(fp32_tensors)
            analysis['quantized_ops_count'] = len(quantized_ops)
            analysis['conversion_successful'] = len(quantized_tensors) > 0 or len(quantized_ops) > 0
            
            analysis['tensor_analysis'] = {
                'quantized_examples': quantized_tensors[:3],
                'fp32_examples': fp32_tensors[:3],
                'quantized_ops': quantized_ops
            }
            
            logger.info(f"📊 INT8 model analysis:")
            logger.info(f"   Quantized tensors: {len(quantized_tensors)}")
            logger.info(f"   FP32 tensors: {len(fp32_tensors)}")
            logger.info(f"   Quantized ops: {len(quantized_ops)}")
            logger.info(f"   Conversion successful: {analysis['conversion_successful']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ INT8 analysis failed: {e}")
        return {'file_exists': False, 'error': str(e)}

def diagnose_conversion_process(qat_path, int8_path):
    """Comprehensive diagnosis of the QAT to INT8 conversion."""
    logger.info("🔬 Running comprehensive quantization diagnosis...")
    
    diagnosis = {
        'qat_analysis': None,
        'int8_analysis': None,
        'issues_found': [],
        'recommendations': [],
        'overall_status': 'UNKNOWN'
    }
    
    # Analyze QAT model
    if os.path.exists(qat_path):
        diagnosis['qat_analysis'] = analyze_qat_model(qat_path)
    else:
        diagnosis['issues_found'].append(f"QAT model not found: {qat_path}")
    
    # Analyze INT8 model
    if os.path.exists(int8_path):
        diagnosis['int8_analysis'] = analyze_int8_model(int8_path)
    else:
        diagnosis['issues_found'].append(f"INT8 model not found: {int8_path}")
    
    # Generate recommendations
    qat_ok = diagnosis['qat_analysis'] and diagnosis['qat_analysis'].get('contains_fake_quantizers', False)
    int8_ok = diagnosis['int8_analysis'] and diagnosis['int8_analysis'].get('contains_quantized_tensors', False)
    
    if not qat_ok:
        diagnosis['issues_found'].append("QAT model doesn't contain FakeQuantizers")
        diagnosis['recommendations'].append("Re-run QAT training with proper quantization preservation")
    
    if not int8_ok:
        diagnosis['issues_found'].append("INT8 model doesn't contain quantized tensors")
        diagnosis['recommendations'].append("Re-run INT8 conversion with proper observer calibration")
    
    # Check for observer warnings (from your log)
    if qat_ok and not int8_ok:
        diagnosis['issues_found'].append("Conversion failed despite good QAT model")
        diagnosis['recommendations'].extend([
            "Observers may not have been properly calibrated during training",
            "Try calibrating the model before conversion",
            "Check if model was in eval() mode during conversion"
        ])
    
    # Overall status
    if qat_ok and int8_ok:
        diagnosis['overall_status'] = 'SUCCESS'
    elif qat_ok and not int8_ok:
        diagnosis['overall_status'] = 'CONVERSION_FAILED'
    elif not qat_ok:
        diagnosis['overall_status'] = 'QAT_FAILED'
    else:
        diagnosis['overall_status'] = 'UNKNOWN_ERROR'
    
    return diagnosis

def generate_fix_recommendations(diagnosis):
    """Generate specific fix recommendations based on diagnosis."""
    logger.info("💡 Generating fix recommendations...")
    
    recommendations = []
    
    status = diagnosis['overall_status']
    
    if status == 'QAT_FAILED':
        recommendations.extend([
            "🔧 Re-run QAT training with quantization preservation enabled",
            "📊 Verify that FakeQuantizers are created during prepare_qat()",
            "🔍 Check that quantization config is properly applied",
            "⚙️ Ensure model.train() is called before prepare_qat()"
        ])
    
    elif status == 'CONVERSION_FAILED':
        recommendations.extend([
            "🔄 Re-run INT8 conversion with proper observer calibration",
            "📈 Ensure observers were updated during training",
            "🎯 Try manual calibration before conversion",
            "⚡ Set model to eval() mode before conversion",
            "🔧 Use torch.quantization.convert() with proper settings"
        ])
    
    elif status == 'SUCCESS':
        recommendations.extend([
            "✅ Both models appear correct",
            "🚀 The issue may be in model loading/inference",
            "🔍 Try the fixed test script for proper INT8 handling",
            "📤 Consider exporting to ONNX for deployment"
        ])
    
    else:
        recommendations.extend([
            "❓ Unknown issue detected",
            "🔬 Check model files manually",
            "📋 Review training and conversion logs",
            "🆘 Consider re-running entire QAT process"
        ])
    
    return recommendations

def main():
    """Main diagnosis function."""
    
    # Configuration
    QAT_MODEL_PATH = "models/checkpoints/qat_full_features_1/qat_model_with_fakequant.pt"
    INT8_MODEL_PATH = "models/checkpoints/qat_full_features_1/qat_yolov8n_full_int8_final.pt"
    OUTPUT_DIR = "analysis_results/quantization_diagnosis"
    
    print("="*80)
    print("🔬 QUANTIZATION DIAGNOSIS")
    print("="*80)
    
    logger.info(f"🎯 QAT Model: {QAT_MODEL_PATH}")
    logger.info(f"🎯 INT8 Model: {INT8_MODEL_PATH}")
    
    # Run diagnosis
    diagnosis = diagnose_conversion_process(QAT_MODEL_PATH, INT8_MODEL_PATH)
    
    # Generate recommendations
    recommendations = generate_fix_recommendations(diagnosis)
    
    # Save diagnosis report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    diagnosis_report = {
        'diagnosis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'qat_model_path': QAT_MODEL_PATH,
        'int8_model_path': INT8_MODEL_PATH,
        'diagnosis': diagnosis,
        'recommendations': recommendations
    }
    
    report_path = os.path.join(OUTPUT_DIR, "diagnosis_report.yaml")
    with open(report_path, 'w') as f:
        yaml.dump(diagnosis_report, f, default_flow_style=False, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("🔬 DIAGNOSIS RESULTS")
    print("="*80)
    print(f"📊 Overall Status: {diagnosis['overall_status']}")
    
    if diagnosis['qat_analysis']:
        qat = diagnosis['qat_analysis']
        print(f"🔧 QAT Model: {qat.get('fake_quantizer_count', 0)} FakeQuantizers")
    
    if diagnosis['int8_analysis']:
        int8 = diagnosis['int8_analysis']
        print(f"⚡ INT8 Model: {int8.get('quantized_tensor_count', 0)} Quantized Tensors")
    
    print("\n🚨 Issues Found:")
    for issue in diagnosis['issues_found']:
        print(f"   • {issue}")
    
    print("\n💡 Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n📋 Full report: {report_path}")
    print("="*80)

if __name__ == "__main__":
    import time
    main()