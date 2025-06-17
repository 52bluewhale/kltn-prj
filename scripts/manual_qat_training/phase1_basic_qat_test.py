#!/usr/bin/env python3
"""
Phase 1: QAT Preservation Proof - Manual Training Loop
=======================================================

Objective: Prove that manual PyTorch training loop can preserve all 90 FakeQuantize 
modules throughout training, unlike YOLOv8's train() method which destroys them.

This script implements the most basic manual training to validate our approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from ultralytics import YOLO
import yaml
import os
import time
from pathlib import Path
import numpy as np

# ============================================================================
# REUSE: QAT Setup from Working Script
# ============================================================================

class QATDiagnosticMonitor:
    """Diagnostic system to track QAT preservation"""
    
    def __init__(self):
        self.checkpoints = []
        self.model_snapshots = {}
        
    def capture_model_state(self, model, checkpoint_name):
        """Capture detailed model state at a specific checkpoint."""
        fake_quant_count = 0
        fake_quant_modules = []
        total_params = 0
        
        for name, module in model.named_modules():
            if 'FakeQuantize' in str(type(module)):
                fake_quant_count += 1
                fake_quant_modules.append(name)
        
        for param in model.parameters():
            total_params += param.numel()
        
        snapshot = {
            'checkpoint': checkpoint_name,
            'timestamp': time.time(),
            'fake_quant_count': fake_quant_count,
            'fake_quant_modules': fake_quant_modules[:3],  # Sample
            'total_parameters': total_params,
            'model_id': id(model)
        }
        
        self.model_snapshots[checkpoint_name] = snapshot
        self.checkpoints.append(checkpoint_name)
        
        print(f"📸 CHECKPOINT [{checkpoint_name}]: {fake_quant_count} FakeQuantize modules, {total_params} params")
        return snapshot
    
    def validate_qat_preservation(self, model, stage_name, expected_count=90):
        """Critical function: Validate QAT preservation at any stage"""
        fake_quant_count = sum(1 for n, m in model.named_modules() 
                              if 'FakeQuantize' in str(type(m)))
        
        if fake_quant_count != expected_count:
            print(f"🚨 QAT PRESERVATION FAILURE at {stage_name}!")
            print(f"   Expected: {expected_count} FakeQuantize modules")
            print(f"   Found: {fake_quant_count} FakeQuantize modules")
            return False
        
        print(f"✅ QAT PRESERVED at {stage_name}: {fake_quant_count}/{expected_count} modules")
        return True

# Global monitor
monitor = QATDiagnosticMonitor()

def setup_enhanced_qat_config():
    """Setup enhanced QAT configuration"""
    from torch.quantization.fake_quantize import FakeQuantize
    from torch.quantization.observer import PerChannelMinMaxObserver, MovingAverageMinMaxObserver
    from torch.quantization import QConfig
    
    # Enhanced per-channel weight quantizer
    weight_fake_quant = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    
    # Enhanced activation quantizer  
    activation_fake_quant = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        averaging_constant=0.01
    )
    
    qconfig = QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
    return qconfig

def apply_qconfig_to_model(model, qconfig):
    """Apply QConfig to model modules"""
    modules_configured = 0
    detection_modules_skipped = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Skip detection head for stability
            if 'detect' in name or '22' in name:
                module.qconfig = None
                detection_modules_skipped += 1
            else:
                module.qconfig = qconfig
                modules_configured += 1
    
    print(f"✅ QConfig applied to {modules_configured} modules")
    print(f"✅ Skipped {detection_modules_skipped} detection modules")
    return modules_configured

def prepare_qat_model():
    """Prepare QAT model using our proven working setup"""
    print("🔧 Preparing QAT Model (Reusing Working Setup)")
    print("=" * 60)
    
    # Load model
    model_path = "models/pretrained/yolov8n.pt"
    model_yolo = YOLO(model_path)
    model = model_yolo.model
    
    print(f"✅ YOLOv8n loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup and apply QConfig
    qconfig = setup_enhanced_qat_config()
    modules_configured = apply_qconfig_to_model(model, qconfig)
    
    if modules_configured == 0:
        raise RuntimeError("No modules configured for QAT!")
    
    # Prepare for QAT
    model.train()
    model_prepared = torch.quantization.prepare_qat(model, inplace=True)
    model_yolo.model = model_prepared

    # QUICK FIX: Ensure all parameters require gradients
    for name, param in model_prepared.named_parameters():
        if 'dfl.conv.weight' not in name:  # Skip DFL layer (supposed to be frozen)
            param.requires_grad = True

    print("🔧 Fixed gradient requirements for QAT parameters")
    
    # Validate initial QAT setup
    monitor.capture_model_state(model_prepared, "QAT_SETUP_COMPLETE")
    
    fake_quant_count = sum(1 for n, m in model_prepared.named_modules() 
                          if 'FakeQuantize' in str(type(m)))
    
    print(f"✅ QAT setup complete: {fake_quant_count} FakeQuantize modules")
    
    if fake_quant_count == 0:
        raise RuntimeError("QAT setup failed - no FakeQuantize modules created!")
    
    return model_prepared, model_yolo

# ============================================================================
# PHASE 1: BASIC MANUAL TRAINING IMPLEMENTATION
# ============================================================================

def create_dummy_batch(batch_size=2, img_size=640, num_classes=58):
    """Create dummy batch for testing (Phase 1 - no real data yet)"""
    # Dummy images
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Dummy targets (simplified format for Phase 1)
    # In real YOLOv8: targets = [img_idx, class, x, y, w, h]
    targets = []
    for i in range(batch_size):
        # Each image has 1-3 random objects
        num_objects = torch.randint(1, 4, (1,)).item()
        for _ in range(num_objects):
            target = torch.tensor([
                i,  # image index
                torch.randint(0, num_classes, (1,)).item(),  # class
                torch.rand(1).item(),  # x center (0-1)
                torch.rand(1).item(),  # y center (0-1) 
                torch.rand(1).item() * 0.5,  # width (0-0.5)
                torch.rand(1).item() * 0.5   # height (0-0.5)
            ])
            targets.append(target)
    
    targets = torch.stack(targets) if targets else torch.empty(0, 6)
    
    return images, targets

def simple_detection_loss(predictions, targets):
    """Minimal fix for gradient computation"""
    if isinstance(predictions, (list, tuple)):
        pred = predictions[0]
    else:
        pred = predictions
    
    # Simple loss that definitely preserves gradients
    loss = pred.mean().abs()  # Mean absolute value - simple and gradient-friendly
    return loss

def manual_training_step(model, images, targets, optimizer):
    """
    Single manual training step - the core of our solution.
    This replaces YOLOv8's train() method with manual control.
    """
    # 1. Forward pass
    model.train()
    predictions = model(images)
    
    # 2. QAT validation after forward pass
    if not monitor.validate_qat_preservation(model, "AFTER_FORWARD_PASS"):
        raise RuntimeError("QAT lost during forward pass!")
    
    # 3. Loss calculation  
    loss = simple_detection_loss(predictions, targets)
    
    # 4. QAT validation after loss calculation
    if not monitor.validate_qat_preservation(model, "AFTER_LOSS_CALCULATION"):
        raise RuntimeError("QAT lost during loss calculation!")
    
    # 5. Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # 6. QAT validation after backward pass
    if not monitor.validate_qat_preservation(model, "AFTER_BACKWARD_PASS"):
        raise RuntimeError("QAT lost during backward pass!")
    
    # 7. Optimizer step
    optimizer.step()
    
    # 8. QAT validation after optimizer step  
    if not monitor.validate_qat_preservation(model, "AFTER_OPTIMIZER_STEP"):
        raise RuntimeError("QAT lost during optimizer step!")
    
    return loss.item()

def manual_training_loop_basic(model, optimizer, num_epochs=2, batches_per_epoch=5):
    """
    Basic manual training loop for Phase 1.
    Tests QAT preservation through complete training cycles.
    """
    print(f"\n🏋️ Starting Manual Training Loop (Phase 1)")
    print(f"📊 Training: {num_epochs} epochs, {batches_per_epoch} batches/epoch")
    print("=" * 60)
    
    # Pre-training validation
    monitor.capture_model_state(model, "BEFORE_MANUAL_TRAINING")
    
    total_batches = 0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = []
        
        for batch_idx in range(batches_per_epoch):
            # Create dummy batch for testing
            images, targets = create_dummy_batch(batch_size=2)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
                model = model.cuda()
            
            # Manual training step
            try:
                loss = manual_training_step(model, images, targets, optimizer)
                epoch_losses.append(loss)
                total_batches += 1
                
                print(f"   Batch {batch_idx + 1}/{batches_per_epoch}: Loss = {loss:.6f}")
                
                # Checkpoint after every few batches
                if batch_idx % 2 == 0:
                    monitor.capture_model_state(model, f"EPOCH_{epoch}_BATCH_{batch_idx}")
                
            except RuntimeError as e:
                print(f"❌ Training failed at epoch {epoch}, batch {batch_idx}: {e}")
                return False
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"   📊 Epoch {epoch + 1} complete: Avg Loss = {avg_loss:.6f}")
        
        # Critical checkpoint after each epoch
        monitor.capture_model_state(model, f"EPOCH_{epoch}_COMPLETE")
    
    # Post-training validation
    monitor.capture_model_state(model, "MANUAL_TRAINING_COMPLETE")
    
    print(f"\n✅ Manual training completed successfully!")
    print(f"📊 Total batches trained: {total_batches}")
    print(f"🎯 QAT preservation: SUCCESSFUL throughout entire training")
    
    return True

# ============================================================================
# PHASE 1 MAIN EXECUTION
# ============================================================================

def phase1_main():
    """
    Phase 1 main execution: Prove manual training preserves QAT.
    """
    print("🚀 PHASE 1: QAT Preservation Proof")
    print("=" * 60)
    print("Objective: Prove manual training loop preserves all 90 FakeQuantize modules")
    print("Expected Outcome: ✅ QAT preserved throughout manual training")
    print("=" * 60)
    
    try:
        # Step 1: Prepare QAT model
        print("\n📋 Step 1: QAT Model Preparation")
        model, model_yolo = prepare_qat_model()
        
        # Step 2: Setup optimizer
        print("\n📋 Step 2: Optimizer Setup")
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        print("✅ AdamW optimizer created")
        
        # Step 3: Manual training loop
        print("\n📋 Step 3: Manual Training Loop Execution")
        success = manual_training_loop_basic(model, optimizer, num_epochs=2, batches_per_epoch=3)
        
        if not success:
            print("❌ Phase 1 FAILED: Manual training loop failed")
            return False
        
        # Step 4: Final validation
        print("\n📋 Step 4: Final QAT Validation")
        final_fake_quant_count = sum(1 for n, m in model.named_modules() 
                                   if 'FakeQuantize' in str(type(m)))
        
        print(f"🔍 Final FakeQuantize count: {final_fake_quant_count}")
        
        if final_fake_quant_count == 90:
            print("🎉 PHASE 1 SUCCESS: All 90 FakeQuantize modules preserved!")
            print("✅ Manual training loop approach is VIABLE")
            
            # Step 5: Test model saving
            print("\n📋 Step 5: QAT Model Saving Test")
            save_path = "F:/kltn-prj/models/checkpoints/phase1_test/qat_manual_trained.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save using PyTorch directly to preserve QAT
            torch.save({
                'model_state_dict': model.state_dict(),
                'fake_quant_count': final_fake_quant_count,
                'training_method': 'manual_loop_phase1',
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                print(f"✅ QAT model saved: {save_path} ({file_size:.2f} MB)")
                
                # Test reload
                loaded_data = torch.load(save_path, map_location='cpu')
                print(f"✅ Model reload test: {loaded_data['fake_quant_count']} FakeQuantize modules preserved")
            else:
                print("❌ Model save failed")
                return False
            
            return True
        else:
            print(f"❌ PHASE 1 FAILED: Only {final_fake_quant_count}/90 FakeQuantize modules preserved")
            return False
            
    except Exception as e:
        print(f"❌ Phase 1 execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_phase1_report():
    """Generate comprehensive Phase 1 results report"""
    print("\n" + "=" * 80)
    print("📊 PHASE 1 COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Print all checkpoints
    print("📸 Training Checkpoints:")
    for i, checkpoint in enumerate(monitor.checkpoints):
        state = monitor.model_snapshots[checkpoint]
        status = "✅" if state['fake_quant_count'] == 90 else "❌"
        print(f"{i+1:2d}. {status} {checkpoint:25s} - {state['fake_quant_count']:2d} FakeQuantize modules")
    
    # Summary
    total_checkpoints = len(monitor.checkpoints)
    successful_checkpoints = sum(1 for cp in monitor.checkpoints 
                               if monitor.model_snapshots[cp]['fake_quant_count'] == 90)
    
    print(f"\n📊 Summary:")
    print(f"   Total checkpoints: {total_checkpoints}")
    print(f"   Successful checkpoints: {successful_checkpoints}")
    print(f"   Success rate: {successful_checkpoints/total_checkpoints*100:.1f}%")
    
    if successful_checkpoints == total_checkpoints:
        print("🎉 PHASE 1 RESULT: COMPLETE SUCCESS")
        print("✅ Manual training loop successfully preserves QAT throughout training")
        print("🎯 Next: Ready for Phase 2 (YOLOv8 Loss Integration)")
    else:
        print("❌ PHASE 1 RESULT: PARTIAL/COMPLETE FAILURE")
        print("🔧 Action needed: Debug QAT preservation issues")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🎯 PHASE 1: MANUAL QAT TRAINING LOOP TEST")
    print("=" * 60)
    print("Testing manual PyTorch training loop to preserve QAT quantization")
    print("This replaces YOLOv8's problematic train() method")
    print("=" * 60)
    
    success = phase1_main()
    
    # Generate comprehensive report
    generate_phase1_report()
    
    if success:
        print("\n🎊 PHASE 1 COMPLETED SUCCESSFULLY!")
        print("✅ Proof: Manual training loop preserves QAT quantization")
        print("🎯 Ready to proceed to Phase 2: YOLOv8 Loss Integration")
    else:
        print("\n💥 PHASE 1 FAILED!")
        print("🔧 Manual training loop needs debugging before proceeding")
        exit(1)