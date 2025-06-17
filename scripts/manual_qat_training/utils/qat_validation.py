#!/usr/bin/env python3
"""
QAT Validation Utilities
========================

Reusable utilities for validating QAT preservation throughout manual training.
These functions are critical for detecting when/where FakeQuantize modules are lost.
"""

import torch
import time
import logging
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QATValidator:
    """
    Comprehensive QAT validation system.
    Tracks FakeQuantize modules and validates preservation throughout training.
    """
    
    def __init__(self, expected_fake_quant_count: int = 90):
        """
        Initialize QAT validator.
        
        Args:
            expected_fake_quant_count: Expected number of FakeQuantize modules
        """
        self.expected_count = expected_fake_quant_count
        self.validation_history = []
        self.checkpoints = {}
        
    def count_fake_quantize_modules(self, model: torch.nn.Module) -> int:
        """Count FakeQuantize modules in model."""
        count = 0
        for name, module in model.named_modules():
            if 'FakeQuantize' in str(type(module)):
                count += 1
        return count
    
    def get_fake_quantize_modules(self, model: torch.nn.Module) -> List[str]:
        """Get list of FakeQuantize module names."""
        modules = []
        for name, module in model.named_modules():
            if 'FakeQuantize' in str(type(module)):
                modules.append(name)
        return modules
    
    def validate_qat_preservation(self, 
                                 model: torch.nn.Module, 
                                 stage_name: str, 
                                 raise_on_failure: bool = True) -> bool:
        """
        Critical validation function - must be called after every training operation.
        
        Args:
            model: Model to validate
            stage_name: Name of current training stage for logging
            raise_on_failure: Whether to raise exception on validation failure
            
        Returns:
            True if QAT is preserved, False otherwise
            
        Raises:
            RuntimeError: If QAT preservation fails and raise_on_failure=True
        """
        fake_quant_count = self.count_fake_quantize_modules(model)
        
        # Record validation
        validation_record = {
            'timestamp': time.time(),
            'stage': stage_name,
            'fake_quant_count': fake_quant_count,
            'expected_count': self.expected_count,
            'success': fake_quant_count == self.expected_count,
            'model_id': id(model)
        }
        
        self.validation_history.append(validation_record)
        
        if fake_quant_count == self.expected_count:
            logger.info(f"✅ QAT PRESERVED at {stage_name}: {fake_quant_count}/{self.expected_count} modules")
            return True
        else:
            error_msg = f"QAT PRESERVATION FAILURE at {stage_name}! Expected: {self.expected_count}, Found: {fake_quant_count}"
            logger.error(f"🚨 {error_msg}")
            
            if raise_on_failure:
                raise RuntimeError(error_msg)
            return False
    
    def capture_checkpoint(self, model: torch.nn.Module, checkpoint_name: str) -> Dict:
        """
        Capture detailed model state at a checkpoint.
        
        Args:
            model: Model to capture
            checkpoint_name: Name of checkpoint
            
        Returns:
            Dictionary with detailed model state information
        """
        fake_quant_count = self.count_fake_quantize_modules(model)
        fake_quant_modules = self.get_fake_quantize_modules(model)
        
        # Count other relevant modules
        qconfig_count = sum(1 for n, m in model.named_modules() 
                           if hasattr(m, 'qconfig') and m.qconfig is not None)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        checkpoint_data = {
            'checkpoint_name': checkpoint_name,
            'timestamp': time.time(),
            'fake_quant_count': fake_quant_count,
            'fake_quant_modules': fake_quant_modules[:5],  # Sample for logging
            'qconfig_count': qconfig_count,
            'total_parameters': total_params,
            'model_id': id(model),
            'model_type': str(type(model).__name__)
        }
        
        self.checkpoints[checkpoint_name] = checkpoint_data
        
        logger.info(f"📸 CHECKPOINT [{checkpoint_name}]: {fake_quant_count} FakeQuantize, {total_params} params")
        
        return checkpoint_data
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str) -> Optional[Dict]:
        """
        Compare two checkpoints to identify changes.
        
        Args:
            checkpoint1: Name of first checkpoint
            checkpoint2: Name of second checkpoint
            
        Returns:
            Dictionary with comparison results, None if checkpoints don't exist
        """
        if checkpoint1 not in self.checkpoints or checkpoint2 not in self.checkpoints:
            logger.error(f"Cannot compare checkpoints - missing data")
            return None
        
        cp1 = self.checkpoints[checkpoint1]
        cp2 = self.checkpoints[checkpoint2]
        
        fake_quant_diff = cp2['fake_quant_count'] - cp1['fake_quant_count']
        param_diff = cp2['total_parameters'] - cp1['total_parameters']
        model_changed = cp1['model_id'] != cp2['model_id']
        
        comparison = {
            'from_checkpoint': checkpoint1,
            'to_checkpoint': checkpoint2,
            'fake_quant_change': fake_quant_diff,
            'param_change': param_diff,
            'model_object_changed': model_changed,
            'status': '✅ PRESERVED' if fake_quant_diff == 0 else f'❌ LOST {abs(fake_quant_diff)} modules'
        }
        
        logger.info(f"🔄 COMPARISON [{checkpoint1} → {checkpoint2}]:")
        logger.info(f"   FakeQuantize: {cp1['fake_quant_count']} → {cp2['fake_quant_count']} ({fake_quant_diff:+d})")
        logger.info(f"   Parameters: {cp1['total_parameters']} → {cp2['total_parameters']} ({param_diff:+d})")
        logger.info(f"   Model ID: {'CHANGED' if model_changed else 'SAME'}")
        logger.info(f"   Status: {comparison['status']}")
        
        return comparison
    
    def generate_validation_report(self) -> Dict:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation statistics and results
        """
        if not self.validation_history:
            return {'error': 'No validation history available'}
        
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v['success'])
        success_rate = successful_validations / total_validations * 100
        
        # Find failure points
        failures = [v for v in self.validation_history if not v['success']]
        first_failure = failures[0] if failures else None
        
        report = {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': len(failures),
            'success_rate': success_rate,
            'first_failure': first_failure,
            'expected_count': self.expected_count,
            'final_count': self.validation_history[-1]['fake_quant_count'] if self.validation_history else 0,
            'overall_success': success_rate == 100.0
        }
        
        return report
    
    def print_validation_summary(self):
        """Print comprehensive validation summary."""
        report = self.generate_validation_report()
        
        print("\n" + "=" * 80)
        print("🔍 QAT VALIDATION COMPREHENSIVE REPORT")
        print("=" * 80)
        
        print(f"📊 Validation Statistics:")
        print(f"   Total validations: {report['total_validations']}")
        print(f"   Successful: {report['successful_validations']}")
        print(f"   Failed: {report['failed_validations']}")
        print(f"   Success rate: {report['success_rate']:.1f}%")
        print(f"   Expected FakeQuantize count: {report['expected_count']}")
        print(f"   Final FakeQuantize count: {report['final_count']}")
        
        if report['overall_success']:
            print("\n🎉 OVERALL RESULT: ✅ COMPLETE SUCCESS")
            print("   All QAT validations passed throughout training")
        else:
            print("\n💥 OVERALL RESULT: ❌ QAT PRESERVATION FAILED")
            if report['first_failure']:
                failure = report['first_failure']
                print(f"   First failure at: {failure['stage']}")
                print(f"   FakeQuantize count dropped to: {failure['fake_quant_count']}")
        
        # Print validation history (recent entries)
        print(f"\n📋 Recent Validation History:")
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        for i, validation in enumerate(recent_validations):
            status = "✅" if validation['success'] else "❌"
            stage = validation['stage'][:30]  # Truncate long stage names
            count = validation['fake_quant_count']
            print(f"   {i+1:2d}. {status} {stage:32s} - {count:2d} FakeQuantize modules")
        
        print("\n" + "=" * 80)

def validate_qat_quick(model: torch.nn.Module, stage_name: str, expected_count: int = 90) -> bool:
    """
    Quick QAT validation function for simple use cases.
    
    Args:
        model: Model to validate
        stage_name: Stage name for logging
        expected_count: Expected FakeQuantize module count
        
    Returns:
        True if QAT is preserved, False otherwise
        
    Raises:
        RuntimeError: If QAT preservation fails
    """
    fake_quant_count = sum(1 for n, m in model.named_modules() 
                          if 'FakeQuantize' in str(type(m)))
    
    if fake_quant_count != expected_count:
        error_msg = f"QAT FAILURE at {stage_name}! Expected: {expected_count}, Found: {fake_quant_count}"
        logger.error(f"🚨 {error_msg}")
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ QAT OK at {stage_name}: {fake_quant_count}/{expected_count} modules")
    return True

def analyze_qat_model_structure(model: torch.nn.Module) -> Dict:
    """
    Analyze the structure of a QAT model for debugging.
    
    Args:
        model: QAT model to analyze
        
    Returns:
        Dictionary with detailed analysis
    """
    analysis = {
        'fake_quantize_modules': [],
        'qconfig_modules': [],
        'quantized_modules': [],
        'regular_modules': [],
        'total_parameters': 0,
        'fake_quantize_count': 0,
        'qconfig_count': 0,
        'quantized_count': 0
    }
    
    for name, module in model.named_modules():
        module_type = str(type(module).__name__)
        
        if 'FakeQuantize' in module_type:
            analysis['fake_quantize_modules'].append((name, module_type))
            analysis['fake_quantize_count'] += 1
        elif 'Quantized' in module_type:
            analysis['quantized_modules'].append((name, module_type))
            analysis['quantized_count'] += 1
        elif hasattr(module, 'qconfig') and module.qconfig is not None:
            analysis['qconfig_modules'].append((name, module_type))
            analysis['qconfig_count'] += 1
        else:
            analysis['regular_modules'].append((name, module_type))
    
    analysis['total_parameters'] = sum(p.numel() for p in model.parameters())
    
    return analysis

def print_qat_model_analysis(model: torch.nn.Module):
    """Print detailed QAT model analysis."""
    analysis = analyze_qat_model_structure(model)
    
    print("\n🔍 QAT MODEL STRUCTURE ANALYSIS")
    print("=" * 50)
    print(f"📊 Module Statistics:")
    print(f"   FakeQuantize modules: {analysis['fake_quantize_count']}")
    print(f"   QConfig modules: {analysis['qconfig_count']}")
    print(f"   Quantized modules: {analysis['quantized_count']}")
    print(f"   Regular modules: {len(analysis['regular_modules'])}")
    print(f"   Total parameters: {analysis['total_parameters']:,}")
    
    # Show sample modules
    if analysis['fake_quantize_modules']:
        print(f"\n📋 Sample FakeQuantize modules:")
        for name, module_type in analysis['fake_quantize_modules'][:5]:
            print(f"   - {name} ({module_type})")
    
    if analysis['quantized_modules']:
        print(f"\n📋 Sample Quantized modules:")
        for name, module_type in analysis['quantized_modules'][:5]:
            print(f"   - {name} ({module_type})")
    
    print("=" * 50)

# Convenience functions for integration
def create_qat_validator(expected_count: int = 90) -> QATValidator:
    """Create QAT validator with standard settings."""
    return QATValidator(expected_fake_quant_count=expected_count)

def validate_qat_preservation_safe(model: torch.nn.Module, 
                                  stage_name: str, 
                                  validator: Optional[QATValidator] = None) -> bool:
    """
    Safe QAT validation that doesn't raise exceptions.
    
    Args:
        model: Model to validate
        stage_name: Stage name for logging
        validator: Optional validator instance
        
    Returns:
        True if QAT is preserved, False otherwise
    """
    if validator is None:
        return validate_qat_quick(model, stage_name, expected_count=90)
    
    return validator.validate_qat_preservation(model, stage_name, raise_on_failure=False)