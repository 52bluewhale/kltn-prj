#!/usr/bin/env python
"""
FIXED: YOLOv8 QAT Training Script Following Your Phased Training Algorithm
Implements the exact algorithm from your diagram with proper phase transitions
"""
import os
import sys
import logging
import argparse
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_qat_fixed_phased')

# Import fixed implementation
from src.models.yolov8_qat_fixed import FixedQuantizedYOLOv8

class PhasedQATTrainer:
    """
    SOLUTION: Implements your exact phased training algorithm from the diagram
    """
    
    def __init__(self, qat_model, total_epochs):
        """
        Initialize phased training following your algorithm.
        
        Args:
            qat_model: FixedQuantizedYOLOv8 instance
            total_epochs: Total number of training epochs
        """
        self.qat_model = qat_model
        self.total_epochs = total_epochs
        
        # Phase boundaries following your algorithm
        self.phase1_end = max(1, int(total_epochs * 0.3))      # 30% - Weights Only
        self.phase2_end = max(2, int(total_epochs * 0.7))      # 40% - Add Activations  
        self.phase3_end = max(3, int(total_epochs * 0.9))      # 20% - All Quantizers
        # phase4_end = total_epochs                             # 10% - Fine-tuning
        
        # Initialize quantizer state manager for phased control
        self._initialize_quantizer_manager()
        
        logger.info("🔄 Phased QAT Training Algorithm Initialized:")
        logger.info(f"  Phase 1 (Weights Only): Epochs 1-{self.phase1_end}")
        logger.info(f"  Phase 2 (Add Activations): Epochs {self.phase1_end+1}-{self.phase2_end}")
        logger.info(f"  Phase 3 (All Quantizers): Epochs {self.phase2_end+1}-{self.phase3_end}")
        logger.info(f"  Phase 4 (Fine-tuning): Epochs {self.phase3_end+1}-{total_epochs}")
    
    def _initialize_quantizer_manager(self):
        """Initialize the enhanced quantizer state manager."""
        try:
            from src.quantization.enhanced_quantizer_manager import EnhancedQuantizerStateManager
            self.quantizer_manager = EnhancedQuantizerStateManager(self.qat_model.model.model)
            logger.info("✅ Enhanced quantizer state manager initialized")
        except ImportError:
            logger.warning("⚠️ Enhanced quantizer manager not available, using basic state management")
            self.quantizer_manager = None
    
    def configure_phase_1_weights_only(self):
        """
        PHASE 1: Weights Only
        Initial epochs with partial quantization to ease training difficulty
        """
        logger.info("🔧 Configuring Phase 1: Weights Only")
        
        if self.quantizer_manager:
            success = self.quantizer_manager.set_phase_state(
                phase_name="phase1_weight_only",
                weights_enabled=True,
                activations_enabled=False
            )
            if success:
                logger.info("✅ Phase 1 configured via enhanced manager")
            else:
                logger.warning("⚠️ Enhanced manager failed, using fallback")
                self._fallback_configure_weights_only()
        else:
            self._fallback_configure_weights_only()
    
    def configure_phase_2_add_activations(self):
        """
        PHASE 2: Add Activations
        Enable more quantizers gradually
        """
        logger.info("🔧 Configuring Phase 2: Add Activations")
        
        if self.quantizer_manager:
            success = self.quantizer_manager.set_phase_state(
                phase_name="phase2_activations",
                weights_enabled=True,
                activations_enabled=True
            )
            if success:
                logger.info("✅ Phase 2 configured via enhanced manager")
            else:
                logger.warning("⚠️ Enhanced manager failed, using fallback")
                self._fallback_configure_all_quantizers()
        else:
            self._fallback_configure_all_quantizers()
    
    def configure_phase_3_all_quantizers(self):
        """
        PHASE 3: All Quantizers
        Full network quantization enabled
        """
        logger.info("🔧 Configuring Phase 3: All Quantizers")
        
        if self.quantizer_manager:
            success = self.quantizer_manager.set_phase_state(
                phase_name="phase3_full_quant",
                weights_enabled=True,
                activations_enabled=True
            )
            if success:
                logger.info("✅ Phase 3 configured via enhanced manager")
            else:
                logger.warning("⚠️ Enhanced manager failed, using fallback")
                self._fallback_configure_all_quantizers()
        else:
            self._fallback_configure_all_quantizers()
    
    def configure_phase_4_fine_tuning(self, trainer):
        """
        PHASE 4: Fine-tuning
        Lower LR to recover accuracy
        """
        logger.info("🔧 Configuring Phase 4: Fine-tuning")
        
        # Keep all quantizers enabled (same as Phase 3)
        if self.quantizer_manager:
            success = self.quantizer_manager.set_phase_state(
                phase_name="phase4_fine_tuning",
                weights_enabled=True,
                activations_enabled=True
            )
            if success:
                logger.info("✅ Phase 4 quantization configured")
            else:
                logger.warning("⚠️ Using fallback for Phase 4")
        
        # Reduce learning rate for fine-tuning (following your algorithm)
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            original_lr = trainer.optimizer.param_groups[0]['lr']
            new_lr = original_lr * 0.1  # Reduce by 10x for fine-tuning
            
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.info(f"🔽 Learning rate reduced for fine-tuning: {original_lr:.6f} → {new_lr:.6f}")
        else:
            logger.warning("⚠️ Could not access optimizer for LR reduction")
    
    def _fallback_configure_weights_only(self):
        """Fallback method to configure weights-only quantization."""
        logger.info("🔄 Using fallback method for weights-only configuration")
        
        weight_count = 0
        activation_count = 0
        
        for name, module in self.qat_model.model.model.named_modules():
            # Keep weight quantizers active
            if hasattr(module, 'weight_fake_quant'):
                # Weight quantizers remain active
                weight_count += 1
            
            # Disable activation quantizers
            if hasattr(module, 'activation_post_process'):
                if not hasattr(module, '_original_activation_post_process'):
                    module._original_activation_post_process = module.activation_post_process
                module.activation_post_process = torch.nn.Identity()
                activation_count += 1
        
        logger.info(f"✅ Fallback Phase 1: {weight_count} weight quantizers active, {activation_count} activation quantizers disabled")
    
    def _fallback_configure_all_quantizers(self):
        """Fallback method to enable all quantizers."""
        logger.info("🔄 Using fallback method for all quantizers configuration")
        
        weight_count = 0
        activation_count = 0
        
        for name, module in self.qat_model.model.model.named_modules():
            # Weight quantizers remain active
            if hasattr(module, 'weight_fake_quant'):
                weight_count += 1
            
            # Restore activation quantizers
            if hasattr(module, 'activation_post_process') and hasattr(module, '_original_activation_post_process'):
                module.activation_post_process = module._original_activation_post_process
                activation_count += 1
        
        logger.info(f"✅ Fallback configuration: {weight_count} weight quantizers, {activation_count} activation quantizers active")
    
    def get_current_phase(self, epoch):
        """Determine current phase based on epoch."""
        if epoch <= self.phase1_end:
            return "phase1_weight_only"
        elif epoch <= self.phase2_end:
            return "phase2_activations"
        elif epoch <= self.phase3_end:
            return "phase3_full_quant"
        else:
            return "phase4_fine_tuning"
    
    def should_transition_phase(self, epoch, current_phase):
        """Check if we should transition to next phase."""
        transitions = {
            self.phase1_end: ("phase1_weight_only", "phase2_activations"),
            self.phase2_end: ("phase2_activations", "phase3_full_quant"),
            self.phase3_end: ("phase3_full_quant", "phase4_fine_tuning")
        }
        
        if epoch in transitions:
            from_phase, to_phase = transitions[epoch]
            return current_phase == from_phase, to_phase
        
        return False, None
    
    def verify_phase_configuration(self, phase_name):
        """Verify that the phase configuration is correct."""
        fake_quant_count = sum(1 for n, m in self.qat_model.model.model.named_modules() 
                              if 'FakeQuantize' in type(m).__name__)
        
        # Count active vs disabled quantizers
        weight_active = 0
        activation_active = 0
        
        for name, module in self.qat_model.model.model.named_modules():
            if hasattr(module, 'weight_fake_quant') and not isinstance(module.weight_fake_quant, torch.nn.Identity):
                weight_active += 1
            if hasattr(module, 'activation_post_process') and not isinstance(module.activation_post_process, torch.nn.Identity):
                activation_active += 1
        
        logger.info(f"📊 Phase {phase_name} verification:")
        logger.info(f"  - Total FakeQuantize modules: {fake_quant_count}")
        logger.info(f"  - Active weight quantizers: {weight_active}")
        logger.info(f"  - Active activation quantizers: {activation_active}")
        
        # Verify phase expectations
        if phase_name == "phase1_weight_only":
            expected_activations = 0
            if activation_active > expected_activations:
                logger.warning(f"⚠️ Phase 1 should have 0 active activation quantizers, found {activation_active}")
        elif phase_name in ["phase2_activations", "phase3_full_quant", "phase4_fine_tuning"]:
            if activation_active == 0:
                logger.warning(f"⚠️ Phase {phase_name} should have active activation quantizers, found 0")
        
        return fake_quant_count > 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FIXED: YOLOv8 Phased QAT Training")
    
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--data', type=str, default='datasets/vietnam-traffic-sign-detection/dataset.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=str, default='models/checkpoints/qat_phased')
    parser.add_argument('--qconfig', type=str, default='default')
    parser.add_argument('--use-penalty-loss', action='store_true', default=True,
                       help='Use quantization penalty loss')
    parser.add_argument('--penalty-alpha', type=float, default=0.01,
                       help='Penalty loss weight')
    
    return parser.parse_args()

def main():
    """
    SOLUTION: Main training function implementing your phased training algorithm
    """
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info("🚀 Starting PHASED YOLOv8 QAT Training (Following Your Algorithm)")
    logger.info(f"📁 Model: {args.model}")
    logger.info(f"📊 Dataset: {args.data}")
    logger.info(f"📈 Total Epochs: {args.epochs}")
    logger.info(f"⚙️ Penalty Loss: {args.use_penalty_loss} (α={args.penalty_alpha})")
    
    # STEP 1: Initialize QAT Model
    try:
        logger.info("📦 Initializing QAT model...")
        qat_model = FixedQuantizedYOLOv8(
            model_path=args.model,
            qconfig_name=args.qconfig,
            skip_detection_head=True,
            fuse_modules=True
        )
        
        logger.info("⚙️ Preparing model for QAT...")
        qat_model.prepare_for_qat()
        
        # Setup penalty loss if requested
        if args.use_penalty_loss:
            logger.info(f"🎯 Setting up penalty loss (α={args.penalty_alpha})...")
            qat_model.setup_penalty_loss_integration(alpha=args.penalty_alpha, warmup_epochs=5)
        
        logger.info("✅ QAT model preparation completed")
        
    except Exception as e:
        logger.error(f"❌ QAT initialization failed: {e}")
        return None
    
    # STEP 2: Initialize Phased Trainer
    try:
        logger.info("🔧 Initializing phased training algorithm...")
        phased_trainer = PhasedQATTrainer(qat_model, args.epochs)
        logger.info("✅ Phased trainer initialized")
        
    except Exception as e:
        logger.error(f"❌ Phased trainer initialization failed: {e}")
        return None
    
    # STEP 3: Configure Initial Phase (Phase 1: Weights Only)
    try:
        logger.info("🎯 Configuring initial phase...")
        phased_trainer.configure_phase_1_weights_only()
        phased_trainer.verify_phase_configuration("phase1_weight_only")
        logger.info("✅ Initial phase configured")
        
    except Exception as e:
        logger.error(f"❌ Initial phase configuration failed: {e}")
        return None
    
    # STEP 4: Setup Training Callbacks for Phase Transitions
    current_phase = "phase1_weight_only"
    
    def on_train_start(trainer):
        """Pre-training setup and verification."""
        logger.info("🏁 Training started - Verifying initial state...")
        phased_trainer.verify_phase_configuration(current_phase)
    
    def on_epoch_start(trainer):
        """Handle phase transitions at epoch start."""
        nonlocal current_phase
        
        epoch = trainer.epoch
        logger.info(f"\n📅 Starting Epoch {epoch}/{args.epochs}")
        
        # Check for phase transitions
        should_transition, next_phase = phased_trainer.should_transition_phase(epoch, current_phase)
        
        if should_transition and next_phase:
            logger.info(f"🔄 Phase Transition: {current_phase} → {next_phase}")
            
            # Configure new phase
            if next_phase == "phase2_activations":
                phased_trainer.configure_phase_2_add_activations()
            elif next_phase == "phase3_full_quant":
                phased_trainer.configure_phase_3_all_quantizers()
            elif next_phase == "phase4_fine_tuning":
                phased_trainer.configure_phase_4_fine_tuning(trainer)
            
            # Update current phase
            current_phase = next_phase
            
            # Verify new phase configuration
            phased_trainer.verify_phase_configuration(current_phase)
            
            logger.info(f"✅ Phase transition completed: Now in {current_phase}")
    
    def on_epoch_end(trainer):
        """Monitor quantization preservation and log phase status."""
        epoch = trainer.epoch
        
        # Check quantization preservation
        fake_quant_count = sum(1 for n, m in trainer.model.named_modules() 
                              if 'FakeQuantize' in type(m).__name__)
        
        if fake_quant_count == 0:
            logger.warning(f"⚠️ Quantization lost at epoch {epoch}, restoring...")
            qat_model._restore_quantization_modules()
            logger.info("✅ Quantization restored")
        
        # Log phase status
        logger.info(f"📊 Epoch {epoch} Complete - Phase: {current_phase}")
        
        # Update penalty handler epoch if available
        if hasattr(qat_model, 'penalty_handler'):
            qat_model.penalty_handler.update_epoch(epoch)
    
    def on_train_end(trainer):
        """Final verification and cleanup."""
        logger.info("🏁 Training completed - Final verification...")
        final_verification = phased_trainer.verify_phase_configuration(current_phase)
        
        if not final_verification:
            logger.warning("⚠️ Final quantization restoration needed...")
            qat_model._restore_quantization_modules()
        
        logger.info(f"✅ Training completed in final phase: {current_phase}")
    
    # Register callbacks
    qat_model.model.add_callback('on_train_start', on_train_start)
    qat_model.model.add_callback('on_train_epoch_start', on_epoch_start)
    qat_model.model.add_callback('on_train_epoch_end', on_epoch_end)
    qat_model.model.add_callback('on_train_end', on_train_end)
    
    # STEP 5: Execute Phased Training
    try:
        logger.info("🚀 Starting phased QAT training...")
        
        results = qat_model.model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=640,
            lr0=args.lr,
            device=args.device,
            project=os.path.dirname(args.save_dir),
            name=os.path.basename(args.save_dir),
            exist_ok=True,
            pretrained=False,
            val=True,
            verbose=True,
            save_period=-1  # Prevent intermediate saves
        )
        
        logger.info("✅ Phased training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None
    
    # STEP 6: Save Models
    try:
        # Save QAT model
        qat_path = os.path.join(args.save_dir, "qat_model_phased.pt")
        logger.info("💾 Saving QAT model...")
        
        if qat_model.save_qat_model(qat_path):
            logger.info(f"✅ QAT model saved: {qat_path}")
        else:
            logger.error("❌ Failed to save QAT model")
            return None
        
        # Convert to INT8
        int8_path = os.path.join(args.save_dir, "int8_model_phased.pt")
        logger.info("🔄 Converting to INT8...")
        
        quantized_model = qat_model.convert_to_quantized(int8_path)
        if quantized_model is not None:
            logger.info(f"✅ INT8 model saved: {int8_path}")
        else:
            logger.error("❌ INT8 conversion failed")
            
    except Exception as e:
        logger.error(f"❌ Model saving failed: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("🎉 PHASED QAT TRAINING COMPLETED!")
    print("="*80)
    print(f"✅ Training Algorithm: Your Phased QAT Algorithm")
    print(f"✅ Final Phase: {current_phase}")
    print(f"✅ QAT Model: {qat_path}")
    print(f"✅ INT8 Model: {int8_path}")
    print(f"📁 All files saved to: {args.save_dir}")
    print("="*80)
    
    return results

if __name__ == '__main__':
    main()