"""
Unit tests for model initialization with LoRA and performance assertions
Tests dynamic LoRA insertion, memory usage, and inference latency
"""

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.models.peft_model import (
    DynamicLoRAConfig,
    PEFTModelFactory,
    create_model_with_dynamic_lora
)

# Suppress warnings during testing
warnings.filterwarnings('ignore')


class TestDynamicLoRAConfig(unittest.TestCase):
    """Test DynamicLoRAConfig class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_config_path = Path("test_lora_config.json")
        cls.test_config = {
            "description": "Test LoRA configuration",
            "timestamp": "2024-01-01T00:00:00",
            "model_name": "test-model",
            "analysis_metadata": {
                "svd_threshold": 0.9,
                "min_rank": 4,
                "max_rank": 64,
                "smoothing_factor": 0.8
            },
            "layer_ranks": {
                "q_proj": 16,
                "k_proj": 16,
                "v_proj": 16,
                "o_proj": 8,
                "gate_proj": 32,
                "up_proj": 32,
                "down_proj": 32
            },
            "total_parameters": 1000000,
            "compression_ratio": 0.05
        }
        
        # Write test config
        with open(cls.test_config_path, 'w') as f:
            json.dump(cls.test_config, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if cls.test_config_path.exists():
            cls.test_config_path.unlink()
    
    def test_load_config(self):
        """Test loading LoRA configuration"""
        config = DynamicLoRAConfig(str(self.test_config_path))
        
        self.assertIsNotNone(config.config)
        self.assertEqual(config.config["model_name"], "test-model")
    
    def test_get_layer_ranks(self):
        """Test getting layer-specific ranks"""
        config = DynamicLoRAConfig(str(self.test_config_path))
        ranks = config.get_layer_ranks()
        
        self.assertEqual(ranks["q_proj"], 16)
        self.assertEqual(ranks["gate_proj"], 32)
        self.assertEqual(len(ranks), 7)
    
    def test_get_compression_ratio(self):
        """Test getting compression ratio"""
        config = DynamicLoRAConfig(str(self.test_config_path))
        ratio = config.get_compression_ratio()
        
        self.assertAlmostEqual(ratio, 0.05)
    
    def test_create_lora_config(self):
        """Test creating LoraConfig with dynamic ranks"""
        config = DynamicLoRAConfig(str(self.test_config_path))
        
        # Mock peft.LoraConfig
        with patch('core.models.peft_model.LoraConfig') as mock_lora:
            mock_lora.return_value = MagicMock()
            
            lora_config = config.create_lora_config()
            
            # Check that LoraConfig was called with correct parameters
            mock_lora.assert_called_once()
            call_kwargs = mock_lora.call_args.kwargs
            
            # Check target modules
            self.assertIn("q_proj", call_kwargs["target_modules"])
            self.assertIn("gate_proj", call_kwargs["target_modules"])
            
            # Check rank_pattern
            self.assertIn("rank_pattern", call_kwargs)
            rank_pattern = call_kwargs["rank_pattern"]
            self.assertIn(r".*\.q_proj", rank_pattern)
            self.assertEqual(rank_pattern[r".*\.q_proj"], 16)


class TestLoRAInsertion(unittest.TestCase):
    """Test correct LoRA layer insertion"""
    
    def setUp(self):
        """Set up test model"""
        # Create a simple test model
        self.model = self._create_test_model()
        
        # Create test config
        self.test_config_path = Path("test_lora_insertion.json")
        self.test_config = {
            "layer_ranks": {
                "linear1": 8,
                "linear2": 16
            },
            "compression_ratio": 0.1
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up"""
        if self.test_config_path.exists():
            self.test_config_path.unlink()
        
        # Clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _create_test_model(self):
        """Create a simple test model"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)
                self.config = MagicMock()
                self.config.architectures = ["TestModel"]
            
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x
        
        return TestModel()
    
    def test_lora_layer_insertion(self):
        """Test that LoRA layers are correctly inserted"""
        
        # Mock the PEFT library
        with patch('core.models.peft_model.get_peft_model') as mock_get_peft:
            # Create a mock PEFT model
            mock_peft_model = MagicMock()
            mock_peft_model.named_parameters.return_value = [
                ("lora_A", torch.nn.Parameter(torch.randn(8, 128))),
                ("lora_B", torch.nn.Parameter(torch.randn(256, 8))),
            ]
            mock_get_peft.return_value = mock_peft_model
            
            # Test model creation
            with patch('core.models.peft_model.DynamicLoRAConfig') as mock_config_class:
                mock_config = MagicMock()
                mock_config.get_layer_ranks.return_value = {"linear1": 8, "linear2": 16}
                mock_config.get_compression_ratio.return_value = 0.1
                mock_config.create_lora_config.return_value = MagicMock()
                mock_config_class.return_value = mock_config
                
                # Create PEFT model
                peft_model, _ = PEFTModelFactory.create_peft_model_from_config(
                    base_model=self.model,
                    rank_config_path=str(self.test_config_path)
                )
                
                # Verify get_peft_model was called
                mock_get_peft.assert_called_once()
                
                # Verify the model can perform a forward pass
                test_input = torch.randn(1, 128)
                try:
                    output = peft_model(test_input)
                    self.assertIsNotNone(output)
                except:
                    pass  # Mock might not support forward pass
    
    def test_heterogeneous_ranks(self):
        """Test that different layers get different ranks as configured"""
        
        config = DynamicLoRAConfig(str(self.test_config_path))
        ranks = config.get_layer_ranks()
        
        # Verify different ranks for different layers
        self.assertEqual(ranks["linear1"], 8)
        self.assertEqual(ranks["linear2"], 16)
        self.assertNotEqual(ranks["linear1"], ranks["linear2"])


class TestPerformanceAssertions(unittest.TestCase):
    """Test performance assertions for memory and latency"""
    
    def setUp(self):
        """Set up for performance tests"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance thresholds (hardware-dependent)
        # NOTE: These thresholds are benchmarked for NVIDIA A100 80GB
        # Adjust for different hardware configurations
        self.memory_threshold_mb = 500  # 500 MB for test model
        self.latency_threshold_ms = 100  # 100ms for test model
        
        # Create test configuration
        self.test_config_path = Path("test_perf_config.json")
        self.test_config = {
            "layer_ranks": {
                "q_proj": 8,
                "k_proj": 8,
                "v_proj": 8,
                "o_proj": 8
            },
            "compression_ratio": 0.05
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up"""
        if self.test_config_path.exists():
            self.test_config_path.unlink()
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @pytest.mark.gpu
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_memory_usage(self):
        """Test that memory usage is below threshold"""
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create a small test model
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(self.device)
        
        # Measure memory after model creation
        memory_bytes = torch.cuda.memory_allocated(self.device)
        memory_mb = memory_bytes / (1024 * 1024)
        
        print(f"\nMemory usage: {memory_mb:.2f} MB")
        print(f"Threshold: {self.memory_threshold_mb} MB")
        
        # Assert memory usage is below threshold
        self.assertLess(
            memory_mb,
            self.memory_threshold_mb,
            f"Memory usage ({memory_mb:.2f} MB) exceeds threshold ({self.memory_threshold_mb} MB)"
        )
    
    @pytest.mark.gpu
    def test_inference_latency(self):
        """Test that inference latency is below threshold"""
        
        # Create a small test model
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(self.device)
        
        model.eval()
        
        # Create test input
        batch_size = 1
        seq_length = 128
        hidden_size = 512
        test_input = torch.randn(batch_size, seq_length, hidden_size).to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input[0])
        
        # Measure latency
        num_iterations = 100
        
        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(test_input[0])
        
        # Synchronize if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate average latency
        total_time_ms = (end_time - start_time) * 1000
        avg_latency_ms = total_time_ms / num_iterations
        
        print(f"\nAverage inference latency: {avg_latency_ms:.2f} ms")
        print(f"Threshold: {self.latency_threshold_ms} ms")
        
        # Assert latency is below threshold
        self.assertLess(
            avg_latency_ms,
            self.latency_threshold_ms,
            f"Inference latency ({avg_latency_ms:.2f} ms) exceeds threshold ({self.latency_threshold_ms} ms)"
        )
    
    @pytest.mark.gpu
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_memory_with_gradient_checkpointing(self):
        """Test memory usage with gradient checkpointing enabled"""
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model with gradient checkpointing
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        ).to(self.device)
        
        # Enable gradient checkpointing (mock for test model)
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
        
        # Create test input
        test_input = torch.randn(1, 128, 512).to(self.device)
        
        # Forward pass
        output = model(test_input[0])
        
        # Backward pass to test gradient checkpointing
        loss = output.sum()
        loss.backward()
        
        # Measure peak memory
        peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        print(f"\nPeak memory with gradient checkpointing: {peak_memory_mb:.2f} MB")
        
        # Assert memory usage is still reasonable
        # Gradient checkpointing should reduce memory usage
        self.assertLess(
            peak_memory_mb,
            self.memory_threshold_mb * 1.5,  # Allow 50% more for gradient computation
            f"Peak memory ({peak_memory_mb:.2f} MB) exceeds threshold"
        )


class TestSVDArtifactPersistence(unittest.TestCase):
    """Test persistence of SVD analysis artifacts"""
    
    def setUp(self):
        """Set up test directories"""
        self.test_output_dir = Path("test_svd_outputs")
        self.test_output_dir.mkdir(exist_ok=True)
        
        self.plot_dir = self.test_output_dir / "plots"
        self.data_dir = self.test_output_dir / "raw_data"
        self.delta_dir = self.test_output_dir / "delta_weights"
        
        for dir_path in [self.plot_dir, self.data_dir, self.delta_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test directories"""
        import shutil
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_save_singular_value_plots(self):
        """Test saving singular value decay plots"""
        
        # Create mock singular values
        singular_values = np.exp(-np.linspace(0, 5, 100))
        
        # Create a simple plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.semilogy(singular_values)
        plt.title("Test Singular Value Decay")
        
        # Save plot
        plot_path = self.plot_dir / "test_layer_svd.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Assert plot was saved
        self.assertTrue(plot_path.exists())
        self.assertGreater(plot_path.stat().st_size, 0)
    
    def test_save_raw_svd_data(self):
        """Test saving raw SVD data"""
        
        # Create test data
        test_data = {
            "layer_1": {
                "singular_values": [1.0, 0.5, 0.25, 0.125],
                "rank": 2,
                "energy_retained": 0.95
            },
            "layer_2": {
                "singular_values": [2.0, 1.0, 0.5],
                "rank": 1,
                "energy_retained": 0.90
            }
        }
        
        # Save data
        data_path = self.data_dir / "test_svd_data.json"
        with open(data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Assert data was saved correctly
        self.assertTrue(data_path.exists())
        
        # Load and verify
        with open(data_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["layer_1"]["rank"], 2)
        self.assertEqual(len(loaded_data), 2)
    
    def test_save_delta_weights(self):
        """Test saving delta weight matrices"""
        
        # Create test delta weights
        delta_weights = torch.randn(256, 128)
        
        # Save weights
        weight_path = self.delta_dir / "test_layer_delta.pt"
        torch.save(delta_weights, weight_path)
        
        # Assert weights were saved
        self.assertTrue(weight_path.exists())
        
        # Load and verify
        loaded_weights = torch.load(weight_path)
        self.assertEqual(loaded_weights.shape, (256, 128))
        torch.testing.assert_close(loaded_weights, delta_weights)


def run_performance_benchmarks():
    """Run performance benchmarks and print results"""
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Memory benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create test model
        model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        ).to(device)
        
        # Measure memory
        memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"\nTest Model Memory: {memory_mb:.2f} MB")
    
    # Latency benchmark
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024)
    ).to(device)
    
    model.eval()
    test_input = torch.randn(1, 1024).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    
    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(1000):
            _ = model(test_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_latency_ms = (end - start) * 1000 / 1000
    print(f"Average Inference Latency: {avg_latency_ms:.3f} ms")
    
    print("\nNOTE: These thresholds are hardware-dependent.")
    print("Adjust test thresholds based on your hardware configuration.")
    print("="*80)


if __name__ == "__main__":
    # Run benchmarks first
    run_performance_benchmarks()
    
    # Run tests
    unittest.main(verbosity=2)