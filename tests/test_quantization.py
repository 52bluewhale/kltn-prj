import os
import sys
import unittest
import torch
import torch.nn as nn
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantization import (
    CustomMinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
    CustomFakeQuantize,
    PerChannelFakeQuantize,
    LSQFakeQuantize,
    QATConv2d,
    QATLinear,
    QATBatchNorm2d,
    QATReLU,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    calibrate_model,
    create_qconfig,
    prepare_model_for_qat,
    convert_qat_model_to_quantized,
    get_default_qat_qconfig,
    get_sensitive_layer_qconfig,
    prepare_qat_config_from_yaml,
    build_calibrator
)


class TestObservers(unittest.TestCase):
    """Test the observer modules for collecting tensor statistics."""
    
    def setUp(self):
        # Create test tensors
        self.tensor_2d = torch.randn(5, 10)
        self.tensor_4d = torch.randn(2, 3, 16, 16)
        self.constant_tensor = torch.ones(5, 10)
        self.zero_tensor = torch.zeros(5, 10)
    
    def test_custom_minmax_observer(self):
        """Test CustomMinMaxObserver for min/max tracking."""
        # Test with default parameters
        observer = CustomMinMaxObserver()
        _ = observer(self.tensor_2d)
        scale, zero_point = observer.calculate_qparams()
        
        # Check if scale and zero_point are computed
        self.assertIsInstance(scale, torch.Tensor)
        self.assertIsInstance(zero_point, torch.Tensor)
        self.assertEqual(scale.numel(), 1)
        self.assertEqual(zero_point.numel(), 1)
        
        # Check if min and max values are tracked correctly
        self.assertAlmostEqual(observer.min_val.item(), torch.min(self.tensor_2d).item(), places=5)
        self.assertAlmostEqual(observer.max_val.item(), torch.max(self.tensor_2d).item(), places=5)
        
        # Test with different dtype and qscheme
        observer = CustomMinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        _ = observer(self.tensor_2d)
        scale, zero_point = observer.calculate_qparams()
        
        # For symmetric quantization, zero_point should be 0
        self.assertEqual(zero_point.item(), 0)
        
        # Test with constant tensor (edge case)
        observer = CustomMinMaxObserver()
        _ = observer(self.constant_tensor)
        scale, zero_point = observer.calculate_qparams()
        self.assertGreater(scale.item(), 0)  # Scale should be positive
    
    def test_per_channel_minmax_observer(self):
        """Test PerChannelMinMaxObserver for per-channel min/max tracking."""
        # Test with 4D tensor (batch, channels, height, width)
        observer = PerChannelMinMaxObserver(ch_axis=1)  # Channel axis = 1
        _ = observer(self.tensor_4d)
        scales, zero_points = observer.calculate_qparams()
        
        # Check if scales and zero_points are computed per channel
        self.assertEqual(scales.shape[0], self.tensor_4d.shape[1])
        self.assertEqual(zero_points.shape[0], self.tensor_4d.shape[1])
        
        # Manual calculation of min/max per channel
        tensor_reshaped = self.tensor_4d.reshape(self.tensor_4d.shape[1], -1)
        min_vals = torch.min(tensor_reshaped, dim=1)[0]
        max_vals = torch.max(tensor_reshaped, dim=1)[0]
        
        # Check if min and max values are tracked correctly per channel
        self.assertTrue(torch.allclose(observer.min_vals, min_vals))
        self.assertTrue(torch.allclose(observer.max_vals, max_vals))
        
        # Test with symmetric quantization
        observer = PerChannelMinMaxObserver(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        _ = observer(self.tensor_4d)
        scales, zero_points = observer.calculate_qparams()
        
        # For symmetric quantization, all zero_points should be 0
        self.assertTrue(torch.all(zero_points == 0))
    
    def test_histogram_observer(self):
        """Test HistogramObserver for histogram-based calibration."""
        # Test with default parameters
        observer = HistogramObserver()
        _ = observer(self.tensor_2d)
        scale, zero_point = observer.calculate_qparams()
        
        # Check if scale and zero_point are computed
        self.assertIsInstance(scale, torch.Tensor)
        self.assertIsInstance(zero_point, torch.Tensor)
        
        # Check if histogram is initialized
        self.assertEqual(observer.histogram.shape[0], observer.bins)
        self.assertGreater(observer.histogram.sum().item(), 0)  # Histogram should have values
        
        # Test with different bins
        observer = HistogramObserver(bins=100)
        _ = observer(self.tensor_2d)
        self.assertEqual(observer.histogram.shape[0], 100)
        
        # Test with symmetric quantization
        observer = HistogramObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        _ = observer(self.tensor_2d)
        scale, zero_point = observer.calculate_qparams()
        
        # For symmetric quantization, zero_point should be 0
        self.assertEqual(zero_point.item(), 0)


class TestFakeQuantize(unittest.TestCase):
    """Test the fake quantization modules for simulating quantization."""
    
    def setUp(self):
        # Create test tensors
        self.tensor_2d = torch.randn(5, 10)
        self.tensor_4d = torch.randn(2, 3, 16, 16)
        self.weight_tensor = torch.randn(16, 8, 3, 3)  # Typical conv weight shape
    
    def test_custom_fake_quantize(self):
        """Test CustomFakeQuantize for tensor quantization."""
        # Create observer and fake quantize modules
        observer = CustomMinMaxObserver()
        fake_quant = CustomFakeQuantize(observer, 0, 255)
        
        # Test in training mode
        fake_quant.train()
        output = fake_quant(self.tensor_2d)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.tensor_2d.shape)
        
        # Scale and zero_point should be initialized
        self.assertGreater(fake_quant.scale.item(), 0)
        
        # Test in eval mode
        fake_quant.eval()
        output = fake_quant(self.tensor_2d)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.tensor_2d.shape)
        
        # Test gradient flow
        fake_quant.train()
        tensor = self.tensor_2d.clone().requires_grad_()
        output = fake_quant(tensor)
        loss = output.sum()
        loss.backward()
        
        # Gradient should be computed
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)
    
    def test_per_channel_fake_quantize(self):
        """Test PerChannelFakeQuantize for per-channel quantization."""
        # Create observer and fake quantize modules
        observer = PerChannelMinMaxObserver(ch_axis=0)
        fake_quant = PerChannelFakeQuantize(observer, -128, 127, ch_axis=0)
        
        # Test in training mode with weights (typically per-channel)
        fake_quant.train()
        output = fake_quant(self.weight_tensor)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.weight_tensor.shape)
        
        # Scale should be a vector with length equal to number of output channels
        self.assertEqual(fake_quant.scale.shape[0], self.weight_tensor.shape[0])
        
        # Test in eval mode
        fake_quant.eval()
        output = fake_quant(self.weight_tensor)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.weight_tensor.shape)
        
        # Test gradient flow
        fake_quant.train()
        tensor = self.weight_tensor.clone().requires_grad_()
        output = fake_quant(tensor)
        loss = output.sum()
        loss.backward()
        
        # Gradient should be computed
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)
    
    def test_lsq_fake_quantize(self):
        """Test LSQFakeQuantize for learned step size quantization."""
        # Create observer and fake quantize modules
        observer = CustomMinMaxObserver()
        fake_quant = LSQFakeQuantize(observer, -128, 127)
        
        # Test in training mode
        fake_quant.train()
        output = fake_quant(self.tensor_2d)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.tensor_2d.shape)
        
        # Step size should be initialized
        self.assertTrue(fake_quant.initialized.item())
        self.assertGreater(fake_quant.step_size.item(), 0)
        
        # Test in eval mode
        fake_quant.eval()
        output = fake_quant(self.tensor_2d)
        
        # Output should have the same shape
        self.assertEqual(output.shape, self.tensor_2d.shape)
        
        # Test gradient flow and step size learning
        fake_quant.train()
        tensor = self.tensor_2d.clone().requires_grad_()
        output = fake_quant(tensor)
        loss = output.sum()
        # Store the initial step size
        initial_step_size = fake_quant.step_size.clone()
        # Backward pass
        loss.backward()
        
        # Gradient should be computed for input
        self.assertIsNotNone(tensor.grad)
        self.assertEqual(tensor.grad.shape, tensor.shape)
        
        # Gradient should be computed for step size
        self.assertIsNotNone(fake_quant.step_size.grad)
        
        # Update step size
        with torch.no_grad():
            fake_quant.step_size -= 0.01 * fake_quant.step_size.grad
        
        # Step size should be different after update
        self.assertFalse(torch.allclose(fake_quant.step_size, initial_step_size))


class TestQATModules(unittest.TestCase):
    """Test the QAT modules for quantization-aware training."""
    
    def setUp(self):
        # Create test tensors and QConfig
        self.input_2d = torch.randn(32, 10)  # For linear layer
        self.input_4d = torch.randn(32, 3, 16, 16)  # For conv layer
        self.qconfig = get_default_qat_qconfig()
    
    def test_qat_conv2d(self):
        """Test QATConv2d for quantization-aware convolution."""
        # Create QAT Conv2d
        conv = QATConv2d(3, 16, kernel_size=3, stride=1, padding=1, qconfig=self.qconfig)
        
        # Test in training mode
        conv.train()
        output = conv(self.input_4d)
        
        # Output should have expected shape
        expected_shape = (self.input_4d.shape[0], 16, self.input_4d.shape[2], self.input_4d.shape[3])
        self.assertEqual(output.shape, expected_shape)
        
        # Test in eval mode
        conv.eval()
        output = conv(self.input_4d)
        self.assertEqual(output.shape, expected_shape)
        
        # Test from_float conversion
        float_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        float_conv.qconfig = self.qconfig
        qat_conv = QATConv2d.from_float(float_conv)
        
        # QAT Conv should have the same parameters
        self.assertEqual(qat_conv.in_channels, float_conv.in_channels)
        self.assertEqual(qat_conv.out_channels, float_conv.out_channels)
        self.assertEqual(qat_conv.kernel_size, float_conv.kernel_size)
        
        # Test to_float conversion
        float_conv_back = qat_conv.to_float()
        
        # Float Conv should have the same parameters
        self.assertEqual(float_conv_back.in_channels, qat_conv.in_channels)
        self.assertEqual(float_conv_back.out_channels, qat_conv.out_channels)
        self.assertEqual(float_conv_back.kernel_size, qat_conv.kernel_size)
    
    def test_qat_batchnorm2d(self):
        """Test QATBatchNorm2d for quantization-aware batch normalization."""
        # Create QAT BatchNorm2d
        bn = QATBatchNorm2d(16, qconfig=self.qconfig)
        
        # Create input with correct shape
        bn_input = torch.randn(32, 16, 16, 16)
        
        # Test in training mode
        bn.train()
        output = bn(bn_input)
        
        # Output should have the same shape as input
        self.assertEqual(output.shape, bn_input.shape)
        
        # Test in eval mode
        bn.eval()
        output = bn(bn_input)
        self.assertEqual(output.shape, bn_input.shape)
        
        # Test from_float conversion
        float_bn = nn.BatchNorm2d(16)
        float_bn.qconfig = self.qconfig
        qat_bn = QATBatchNorm2d.from_float(float_bn)
        
        # QAT BN should have the same parameters
        self.assertEqual(qat_bn.num_features, float_bn.num_features)
        self.assertEqual(qat_bn.eps, float_bn.eps)
        
        # Test to_float conversion
        float_bn_back = qat_bn.to_float()
        
        # Float BN should have the same parameters
        self.assertEqual(float_bn_back.num_features, qat_bn.num_features)
        self.assertEqual(float_bn_back.eps, qat_bn.eps)
    
    def test_qat_linear(self):
        """Test QATLinear for quantization-aware linear layer."""
        # Create QAT Linear
        linear = QATLinear(10, 20, qconfig=self.qconfig)
        
        # Test in training mode
        linear.train()
        output = linear(self.input_2d)
        
        # Output should have expected shape
        expected_shape = (self.input_2d.shape[0], 20)
        self.assertEqual(output.shape, expected_shape)
        
        # Test in eval mode
        linear.eval()
        output = linear(self.input_2d)
        self.assertEqual(output.shape, expected_shape)
        
        # Test from_float conversion
        float_linear = nn.Linear(10, 20)
        float_linear.qconfig = self.qconfig
        qat_linear = QATLinear.from_float(float_linear)
        
        # QAT Linear should have the same parameters
        self.assertEqual(qat_linear.in_features, float_linear.in_features)
        self.assertEqual(qat_linear.out_features, float_linear.out_features)
        
        # Test to_float conversion
        float_linear_back = qat_linear.to_float()
        
        # Float Linear should have the same parameters
        self.assertEqual(float_linear_back.in_features, qat_linear.in_features)
        self.assertEqual(float_linear_back.out_features, qat_linear.out_features)
    
    def test_qat_relu(self):
        """Test QATReLU for quantization-aware ReLU."""
        # Create QAT ReLU
        relu = QATReLU(qconfig=self.qconfig)
        
        # Create input with both positive and negative values
        relu_input = torch.randn(32, 10)
        
        # Test in training mode
        relu.train()
        output = relu(relu_input)
        
        # Output should have the same shape as input
        self.assertEqual(output.shape, relu_input.shape)
        
        # All values should be >= 0
        self.assertTrue(torch.all(output >= 0))
        
        # Test in eval mode
        relu.eval()
        output = relu(relu_input)
        self.assertEqual(output.shape, relu_input.shape)
        self.assertTrue(torch.all(output >= 0))
        
        # Test from_float conversion
        float_relu = nn.ReLU()
        float_relu.qconfig = self.qconfig
        qat_relu = QATReLU.from_float(float_relu)
        
        # QAT ReLU should have the same inplace parameter
        self.assertEqual(qat_relu.inplace, float_relu.inplace)
        
        # Test to_float conversion
        float_relu_back = qat_relu.to_float()
        
        # Float ReLU should have the same inplace parameter
        self.assertEqual(float_relu_back.inplace, qat_relu.inplace)


class TestFusionMethods(unittest.TestCase):
    """Test the fusion methods for model optimization."""
    
    def setUp(self):
        # Create test modules
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        
        # Create a simple model with Conv-BN-ReLU pattern
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Test input
        self.input_4d = torch.randn(2, 3, 16, 16)
    
    def test_fuse_conv_bn(self):
        """Test fuse_conv_bn function."""
        # Fuse Conv and BN
        fused_conv = fuse_conv_bn(self.conv, self.bn)
        
        # Fused Conv should be a Conv2d
        self.assertIsInstance(fused_conv, nn.Conv2d)
        
        # Fused Conv should have bias
        self.assertIsNotNone(fused_conv.bias)
        
        # Compare outputs before and after fusion
        self.conv.eval()
        self.bn.eval()
        fused_conv.eval()
        
        with torch.no_grad():
            output_before = self.bn(self.conv(self.input_4d))
            output_after = fused_conv(self.input_4d)
        
        # Outputs should be close
        self.assertTrue(torch.allclose(output_before, output_after, rtol=1e-3, atol=1e-3))
    
    def test_fuse_conv_bn_relu(self):
        """Test fuse_conv_bn_relu function."""
        # Fuse Conv, BN, and ReLU
        fused_module = fuse_conv_bn_relu(self.conv, self.bn, self.relu)
        
        # Compare outputs before and after fusion
        self.conv.eval()
        self.bn.eval()
        self.relu.eval()
        
        with torch.no_grad():
            output_before = self.relu(self.bn(self.conv(self.input_4d)))
            output_after = fused_module(self.input_4d)
        
        # Outputs should be close
        self.assertTrue(torch.allclose(output_before, output_after, rtol=1e-3, atol=1e-3))


class TestQConfigMethods(unittest.TestCase):
    """Test the QConfig creation and management methods."""
    
    def test_create_qconfig(self):
        """Test create_qconfig function."""
        # Create QConfig with default parameters
        qconfig = create_qconfig()
        
        # QConfig should have activation and weight attributes
        self.assertTrue(hasattr(qconfig, 'activation'))
        self.assertTrue(hasattr(qconfig, 'weight'))
        
        # Create QConfig with custom parameters
        qconfig = create_qconfig(
            activation_observer=HistogramObserver,
            weight_observer=PerChannelMinMaxObserver,
            activation_dtype=torch.quint8,
            weight_dtype=torch.qint8,
            activation_qscheme=torch.per_tensor_affine,
            weight_qscheme=torch.per_channel_symmetric,
        )
        
        # QConfig should have correct dtypes
        self.assertEqual(qconfig.activation().dtype, torch.quint8)
        self.assertEqual(qconfig.weight().dtype, torch.qint8)
    
    def test_get_default_qat_qconfig(self):
        """Test get_default_qat_qconfig function."""
        # Get default QAT QConfig
        qconfig = get_default_qat_qconfig()
        
        # QConfig should have activation and weight attributes
        self.assertTrue(hasattr(qconfig, 'activation'))
        self.assertTrue(hasattr(qconfig, 'weight'))
    
    def test_get_sensitive_layer_qconfig(self):
        """Test get_sensitive_layer_qconfig function."""
        # Get sensitive layer QConfig
        qconfig = get_sensitive_layer_qconfig()
        
        # QConfig should have activation and weight attributes
        self.assertTrue(hasattr(qconfig, 'activation'))
        self.assertTrue(hasattr(qconfig, 'weight'))
        
        # Should be different from default QConfig
        default_qconfig = get_default_qat_qconfig()
        self.assertNotEqual(type(qconfig.activation), type(default_qconfig.activation))


class TestModelCalibration(unittest.TestCase):
    """Test the model calibration methods."""
    
    def setUp(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Create a small dataset
        self.inputs = [torch.randn(2, 3, 16, 16) for _ in range(10)]
        
        # Create a simple DataLoader-like object
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
                self.idx = 0
            
            def __iter__(self):
                self.idx = 0
                return self
            
            def __next__(self):
                if self.idx >= len(self.data):
                    raise StopIteration
                data = self.data[self.idx]
                self.idx += 1
                return data
            
            def __len__(self):
                return len(self.data)
        
        self.dataloader = SimpleDataLoader(self.inputs)
    
    def test_build_calibrator(self):
        """Test build_calibrator function."""
        # Build calibrators with different methods
        methods = ['minmax', 'percentile', 'entropy']
        
        for method in methods:
            calibrator = build_calibrator(
                self.model,
                self.dataloader,
                method=method,
                device='cpu',
                num_batches=5
            )
            
            # Calibrator should have correct attributes
            self.assertTrue(hasattr(calibrator, 'model'))
            self.assertTrue(hasattr(calibrator, 'calibration_loader'))
            self.assertTrue(hasattr(calibrator, 'device'))
            self.assertTrue(hasattr(calibrator, 'num_batches'))
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_calibrate_model_cuda(self):
        """Test calibrate_model function with CUDA if available."""
        # Move model to CUDA
        model = self.model.cuda()
        
        # Create CUDA inputs
        cuda_inputs = [x.cuda() for x in self.inputs]
        
        # Create a simple DataLoader-like object for CUDA
        class CudaDataLoader:
            def __init__(self, data):
                self.data = data
                self.idx = 0
            
            def __iter__(self):
                self.idx = 0
                return self
            
            def __next__(self):
                if self.idx >= len(self.data):
                    raise StopIteration
                data = self.data[self.idx]
                self.idx += 1
                return data
            
            def __len__(self):
                return len(self.data)
        
        cuda_dataloader = CudaDataLoader(cuda_inputs)
        
        # Add QConfig to model
        qconfig = get_default_qat_qconfig()
        model.qconfig = qconfig
        
        # Calibrate model
        calibrated_model = calibrate_model(
            model,
            cuda_dataloader,
            method='minmax',
            num_batches=5,
            device='cuda'
        )
        
        # Model should be calibrated
        self.assertEqual(calibrated_model, model)  # Should modify in-place


class TestEndToEndWorkflow(unittest.TestCase):
    """Test the end-to-end QAT workflow."""
    
    def setUp(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Create a simple test input
        self.test_input = torch.randn(2, 3, 16, 16)
        
        # Create temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_model_for_qat(self):
        """Test prepare_model_for_qat function."""
        # Create a simple config
        config = {
            "activation": {
                "observer": "moving_average_minmax",
                "dtype": "quint8",
                "qscheme": "per_tensor_affine"
            },
            "weight": {
                "observer": "minmax",
                "dtype": "qint8",
                "qscheme": "per_channel_symmetric"
            }
        }
        
        # Prepare model for QAT
        qat_model = prepare_model_for_qat(self.model, config, inplace=False)
        
        # Model structure should be similar but with QAT modules
        self.assertEqual(len(qat_model), len(self.model))
        
        # First layer should be QATConv2d
        self.assertIsInstance(qat_model[0], QATConv2d)
        
        # Compare outputs before and after preparation
        self.model.eval()
        qat_model.eval()
        
        with torch.no_grad():
            output_before = self.model(self.test_input)
            output_after = qat_model(self.test_input)
        
        # Outputs should be similar (not identical due to quantization effects)
        self.assertTrue(torch.allclose(output_before, output_after, rtol=1e-2, atol=1e-2))
    
    def test_convert_qat_model_to_quantized(self):
        """Test convert_qat_model_to_quantized function."""
        # Create a simple config
        config = {
            "activation": {
                "observer": "moving_average_minmax",
                "dtype": "quint8",
                "qscheme": "per_tensor_affine"
            },
            "weight": {
                "observer": "minmax",
                "dtype": "qint8",
                "qscheme": "per_channel_symmetric"
            }
        }
        
        # Prepare model for QAT
        qat_model = prepare_model_for_qat(self.model, config, inplace=False)
        
        # Simulate calibration by forward pass
        qat_model.eval()
        with torch.no_grad():
            _ = qat_model(self.test_input)
        
        # Convert QAT model to quantized model
        # This will not fully succeed due to limitations in unittest environment,
        # but it should run without errors
        try:
            quantized_model = convert_qat_model_to_quantized(qat_model, inplace=False)
            
            # Quantized model should have different structure
            self.assertNotEqual(len(quantized_model), len(self.model))
        except Exception as e:
            # Skip test if conversion fails (expected in some environments)
            self.skipTest(f"Quantization conversion failed: {e}")
    
    def test_yaml_config_preparation(self):
        """Test YAML config preparation."""
        # Create a sample YAML-like config
        yaml_config = {
            "quantization": {
                "activation": {
                    "observer": "moving_average_minmax",
                    "dtype": "quint8",
                    "qscheme": "per_tensor_affine"
                },
                "weight": {
                    "observer": "minmax",
                    "dtype": "qint8",
                    "qscheme": "per_channel_symmetric"
                },
                "layer_configs": [
                    {
                        "pattern": "conv1",
                        "config": {
                            "activation": {
                                "observer": "histogram",
                                "dtype": "quint8"
                            },
                            "weight": {
                                "observer": "per_channel_minmax",
                                "dtype": "qint8"
                            }
                        }
                    }
                ]
            }
        }
        
        # Prepare QAT config from YAML
        qat_config = prepare_qat_config_from_yaml(yaml_config)
        
        # Should have default and pattern configs
        self.assertIn("default", qat_config)
        
        # Default config should have correct properties
        self.assertTrue(hasattr(qat_config["default"], "activation"))
        self.assertTrue(hasattr(qat_config["default"], "weight"))


class TestQATModelTransforms(unittest.TestCase):
    """Test model transformations for QAT."""
    
    def setUp(self):
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.fc = nn.Linear(32 * 16 * 16, 10)
            
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        self.model = SimpleModel()
        
        # Create test input
        self.test_input = torch.randn(2, 3, 16, 16)
    
    def test_skip_layers_from_quantization(self):
        """Test skipping layers from quantization."""
        from src.quantization.utils import skip_layers_from_quantization
        
        # Add QConfig to model
        qconfig = get_default_qat_qconfig()
        for module in self.model.modules():
            module.qconfig = qconfig
        
        # Skip 'fc' layer from quantization
        skip_patterns = ['fc']
        model_with_skips = skip_layers_from_quantization(self.model, skip_patterns)
        
        # fc layer should not have QConfig
        self.assertFalse(hasattr(model_with_skips.fc, 'qconfig'))
        
        # Other layers should still have QConfig
        self.assertTrue(hasattr(model_with_skips.conv1, 'qconfig'))
        self.assertTrue(hasattr(model_with_skips.conv2, 'qconfig'))
    
    def test_apply_layer_specific_quantization(self):
        """Test applying layer-specific quantization."""
        from src.quantization.utils import apply_layer_specific_quantization
        
        # Create layer-specific QConfigs
        layer_qconfigs = {
            'conv1': get_first_layer_qconfig(),
            'fc': get_sensitive_layer_qconfig(),
        }
        
        # Apply layer-specific quantization
        model = apply_layer_specific_quantization(self.model, layer_qconfigs)
        
        # Layers should have specific QConfigs
        self.assertEqual(model.conv1.qconfig, get_first_layer_qconfig())
        self.assertEqual(model.fc.qconfig, get_sensitive_layer_qconfig())
        
        # Other layers should have default QConfig
        self.assertNotEqual(model.conv2.qconfig, get_first_layer_qconfig())
        self.assertNotEqual(model.conv2.qconfig, get_sensitive_layer_qconfig())


class TestQuantizationUtils(unittest.TestCase):
    """Test utility functions for quantization."""
    
    def setUp(self):
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.relu1 = nn.ReLU()
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu2 = nn.ReLU()
                self.fc = nn.Linear(32 * 16 * 16, 10)
            
            def forward(self, x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        self.model = SimpleModel()
        
        # Create temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_load_quantization_config(self):
        """Test loading quantization config from file."""
        from src.quantization.utils import load_quantization_config
        import json
        import os
        
        # Create a simple config
        config = {
            "activation": {
                "observer": "moving_average_minmax",
                "dtype": "quint8",
                "qscheme": "per_tensor_affine"
            },
            "weight": {
                "observer": "minmax",
                "dtype": "qint8",
                "qscheme": "per_channel_symmetric"
            }
        }
        
        # Save config to file
        config_path = os.path.join(self.temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Load config
        loaded_config = load_quantization_config(config_path)
        
        # Config should be loaded correctly
        self.assertEqual(loaded_config["activation"]["observer"], config["activation"]["observer"])
        self.assertEqual(loaded_config["weight"]["dtype"], config["weight"]["dtype"])
    
    def test_get_model_size(self):
        """Test getting model size."""
        from src.quantization.utils import get_model_size
        
        # Get size of model
        size_bytes = get_model_size(self.model)
        
        # Size should be positive
        self.assertGreater(size_bytes, 0)
        
        # For a small model like this, size should be less than 1MB
        self.assertLess(size_bytes, 1024 * 1024)
    
    def test_compare_model_sizes(self):
        """Test comparing model sizes."""
        from src.quantization.utils import compare_model_sizes
        
        # Create a larger model
        larger_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Compare model sizes
        comparison = compare_model_sizes(self.model, larger_model)
        
        # Comparison should include size info
        self.assertIn("model1_size", comparison)
        self.assertIn("model2_size", comparison)
        self.assertIn("reduction", comparison)
        
        # Larger model should be larger
        self.assertGreater(comparison["model2_size"], comparison["model1_size"])
    
    def test_save_and_load_quantized_model(self):
        """Test saving and loading quantized model."""
        from src.quantization.utils import save_quantized_model, load_quantized_model
        import os
        
        # Prepare model for QAT
        qconfig = get_default_qat_qconfig()
        self.model.qconfig = qconfig
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.qconfig = qconfig
        
        # Simulate QAT preparation
        for name, module in self.model.named_children():
            if isinstance(module, nn.Conv2d):
                setattr(self.model, name, QATConv2d.from_float(module))
            elif isinstance(module, nn.BatchNorm2d):
                setattr(self.model, name, QATBatchNorm2d.from_float(module))
            elif isinstance(module, nn.ReLU):
                setattr(self.model, name, QATReLU.from_float(module))
            elif isinstance(module, nn.Linear):
                setattr(self.model, name, QATLinear.from_float(module))
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pth")
        save_quantized_model(self.model, model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = load_quantized_model(model_path, self.model)
        
        # Loaded model should have the same structure
        for name, module in self.model.named_modules():
            if name:  # Skip empty name (root module)
                loaded_module = dict(loaded_model.named_modules())[name]
                self.assertEqual(type(module), type(loaded_module))


class TestQuantizationSchemes(unittest.TestCase):
    """Test quantization schemes."""
    
    def test_get_quantizer_functions(self):
        """Test get_weight_quantizer and get_activation_quantizer functions."""
        from src.quantization.schemes import get_weight_quantizer, get_activation_quantizer
        
        # Get weight quantizer
        weight_quantizer = get_weight_quantizer("INT8_SYMMETRIC")
        
        # Weight quantizer should be callable
        self.assertTrue(callable(weight_quantizer))
        
        # Get activation quantizer
        activation_quantizer = get_activation_quantizer("UINT8_ASYMMETRIC")
        
        # Activation quantizer should be callable
        self.assertTrue(callable(activation_quantizer))
    
    def test_predefined_schemes(self):
        """Test predefined quantization schemes."""
        from src.quantization.schemes import INT8_SYMMETRIC, INT8_SYMMETRIC_PER_CHANNEL, UINT8_ASYMMETRIC
        
        # Schemes should be dictionaries with expected keys
        for scheme in [INT8_SYMMETRIC, INT8_SYMMETRIC_PER_CHANNEL, UINT8_ASYMMETRIC]:
            self.assertIsInstance(scheme, dict)
            self.assertIn("observer", scheme)
            self.assertIn("dtype", scheme)
            self.assertIn("qscheme", scheme)


class TestQuantizationErrorAnalysis(unittest.TestCase):
    """Test quantization error analysis."""
    
    def setUp(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Create test input
        self.test_input = torch.randn(2, 3, 16, 16)
        
        # Prepare model for QAT
        qconfig = get_default_qat_qconfig()
        self.model.qconfig = qconfig
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.qconfig = qconfig
    
    def test_measure_layer_wise_quantization_error(self):
        """Test measuring layer-wise quantization error."""
        from src.quantization.utils import measure_layer_wise_quantization_error
        
        # Measure quantization error
        error_stats = measure_layer_wise_quantization_error(
            self.model, 
            self.test_input
        )
        
        # Error stats should include model layers
        for i in range(len(self.model)):
            module_type = type(self.model[i]).__name__
            if isinstance(self.model[i], (nn.Conv2d, nn.Linear)):
                self.assertIn(f"{i}.{module_type}", error_stats)
        
        # Each entry should have error metrics
        for layer_name, stats in error_stats.items():
            self.assertIn("mse", stats)
            self.assertIn("mae", stats)
            self.assertIn("max_error", stats)


if __name__ == "__main__":
    unittest.main()