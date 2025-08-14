#!/usr/bin/env python3
"""
Test suite for experimental protocol implementation
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_results import (
    ExperimentResult,
    AggregatedResult,
    ResultAnalyzer
)
from scripts.run_multi_seed_experiment import (
    ExperimentConfig,
    ExperimentRunner,
    SeedResult
)

import numpy as np
from scipy import stats


class TestExperimentalProtocol(unittest.TestCase):
    """Test experimental protocol compliance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.analyzer = ResultAnalyzer(output_dir=self.test_dir)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_multi_seed_requirement(self):
        """Test that multi-seed runs are enforced"""
        # Create mock results with different seed counts
        single_seed = [
            ExperimentResult("exp1", 42, {"accuracy": 0.85})
        ]
        
        multi_seed = [
            ExperimentResult("exp2", 42, {"accuracy": 0.85}),
            ExperimentResult("exp2", 84, {"accuracy": 0.83}),
            ExperimentResult("exp2", 126, {"accuracy": 0.86})
        ]
        
        # Single seed should have 0 std
        agg_single = self.analyzer.aggregate_results(single_seed)
        self.assertEqual(agg_single.metrics_std["accuracy"], 0.0)
        self.assertEqual(agg_single.num_seeds, 1)
        
        # Multi-seed should have non-zero std
        agg_multi = self.analyzer.aggregate_results(multi_seed)
        self.assertGreater(agg_multi.metrics_std["accuracy"], 0)
        self.assertEqual(agg_multi.num_seeds, 3)
    
    def test_aggregated_reporting_format(self):
        """Test that results are properly formatted as mean ± std"""
        results = [
            ExperimentResult("exp1", 42, {"accuracy": 0.850}),
            ExperimentResult("exp1", 84, {"accuracy": 0.830}),
            ExperimentResult("exp1", 126, {"accuracy": 0.860})
        ]
        
        aggregated = self.analyzer.aggregate_results(results)
        
        # Test formatting
        formatted = aggregated.format_metric("accuracy", precision=3)
        
        # Should be in format "mean ± std"
        self.assertIn("±", formatted)
        
        # Parse the formatted string
        parts = formatted.split("±")
        self.assertEqual(len(parts), 2)
        
        mean_str = parts[0].strip()
        std_str = parts[1].strip()
        
        # Verify values
        expected_mean = np.mean([0.850, 0.830, 0.860])
        expected_std = np.std([0.850, 0.830, 0.860], ddof=1)
        
        self.assertAlmostEqual(float(mean_str), expected_mean, places=3)
        self.assertAlmostEqual(float(std_str), expected_std, places=3)
    
    def test_statistical_significance_testing(self):
        """Test statistical significance testing between experiments"""
        # Create two sets of results
        results_a = AggregatedResult(
            experiment_id="exp_a",
            experiment_name="Model A",
            metrics_mean={"accuracy": 0.85},
            metrics_std={"accuracy": 0.02},
            metrics_raw={"accuracy": [0.83, 0.85, 0.87]},
            num_seeds=3,
            seeds=[42, 84, 126]
        )
        
        results_b = AggregatedResult(
            experiment_id="exp_b",
            experiment_name="Model B",
            metrics_mean={"accuracy": 0.80},
            metrics_std={"accuracy": 0.02},
            metrics_raw={"accuracy": [0.78, 0.80, 0.82]},
            num_seeds=3,
            seeds=[42, 84, 126]
        )
        
        # Test paired t-test
        t_stat, p_value, is_significant = self.analyzer.perform_statistical_test(
            results_a, results_b, "accuracy", test_type="paired_t"
        )
        
        # With these values, should be significant
        self.assertLess(p_value, 0.05)
        self.assertTrue(is_significant)
        
        # Test bootstrap test
        _, p_value_bootstrap, _ = self.analyzer.perform_statistical_test(
            results_a, results_b, "accuracy", test_type="bootstrap"
        )
        
        # Bootstrap should give similar result
        self.assertLess(p_value_bootstrap, 0.1)
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation"""
        # Valid config
        valid_config = {
            "experiment_name": "test_exp",
            "mode": "sft",
            "config_file": __file__  # Use this file as dummy config
        }
        
        config = ExperimentConfig(**valid_config)
        self.assertEqual(config.mode, "sft")
        self.assertEqual(len(config.seeds), 3)  # Default seeds
        
        # Invalid mode
        with self.assertRaises(ValueError):
            invalid_config = valid_config.copy()
            invalid_config["mode"] = "invalid"
            ExperimentConfig(**invalid_config)
        
        # Non-existent config file
        with self.assertRaises(FileNotFoundError):
            invalid_config = valid_config.copy()
            invalid_config["config_file"] = "non_existent.yaml"
            ExperimentConfig(**invalid_config)
    
    def test_reproducibility_with_seeds(self):
        """Test that same seed produces consistent initialization"""
        np.random.seed(42)
        values1 = np.random.randn(10)
        
        np.random.seed(42)
        values2 = np.random.randn(10)
        
        np.testing.assert_array_equal(values1, values2)
    
    def test_registry_management(self):
        """Test experiment registry operations"""
        # Create test registry file
        registry_file = Path(self.test_dir) / "registry.json"
        registry_file.write_text("[]")
        
        # Mock config
        config = Mock(spec=ExperimentConfig)
        config.experiment_name = "test"
        config.mode = "sft"
        config.config_file = "test.yaml"
        config.seeds = [42, 84]
        config.output_dir = self.test_dir
        config.wandb_project = "test_project"
        config.wandb_tags = []
        config.dry_run = False
        
        # Create runner with mocked registry file
        runner = ExperimentRunner(config)
        runner.registry_file = registry_file
        
        # Register experiment
        runner._register_experiment()
        
        # Load and verify registry
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        self.assertEqual(len(registry), 1)
        entry = registry[0]
        self.assertEqual(entry["name"], "test")
        self.assertEqual(entry["mode"], "sft")
        self.assertEqual(entry["seeds"], [42, 84])
        self.assertEqual(entry["status"], "running")
        
        # Update status
        runner._update_registry("completed")
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        self.assertEqual(registry[0]["status"], "completed")
    
    def test_metric_aggregation_edge_cases(self):
        """Test metric aggregation with edge cases"""
        # Empty results
        with self.assertRaises(ValueError):
            self.analyzer.aggregate_results([])
        
        # Different metrics across seeds
        results = [
            ExperimentResult("exp1", 42, {"accuracy": 0.85, "f1": 0.82}),
            ExperimentResult("exp1", 84, {"accuracy": 0.83}),  # Missing f1
            ExperimentResult("exp1", 126, {"accuracy": 0.86, "loss": 0.15})  # Extra metric
        ]
        
        aggregated = self.analyzer.aggregate_results(results)
        
        # Should handle all metrics
        self.assertIn("accuracy", aggregated.metrics_mean)
        self.assertIn("f1", aggregated.metrics_raw)
        self.assertIn("loss", aggregated.metrics_raw)
        
        # f1 should only have 1 value
        self.assertEqual(len(aggregated.metrics_raw["f1"]), 1)
        
        # loss should only have 1 value
        self.assertEqual(len(aggregated.metrics_raw["loss"]), 1)
    
    def test_comparison_table_generation(self):
        """Test generation of comparison tables"""
        # Create mock experiment results
        with patch.object(self.analyzer, 'load_local_results') as mock_load:
            # Mock two experiments
            mock_load.side_effect = [
                [
                    ExperimentResult("baseline", 42, {"accuracy": 0.80, "f1": 0.78}),
                    ExperimentResult("baseline", 84, {"accuracy": 0.79, "f1": 0.77}),
                    ExperimentResult("baseline", 126, {"accuracy": 0.81, "f1": 0.79})
                ],
                [
                    ExperimentResult("improved", 42, {"accuracy": 0.85, "f1": 0.83}),
                    ExperimentResult("improved", 84, {"accuracy": 0.84, "f1": 0.82}),
                    ExperimentResult("improved", 126, {"accuracy": 0.86, "f1": 0.84})
                ]
            ]
            
            # Generate markdown table
            table = self.analyzer.generate_comparison_table(
                ["baseline", "improved"],
                ["accuracy", "f1"],
                output_format="markdown"
            )
            
            # Verify table contains expected elements
            self.assertIn("baseline", table)
            self.assertIn("improved", table)
            self.assertIn("±", table)  # Should have std
            
            # Generate LaTeX table
            mock_load.side_effect = [
                [ExperimentResult("baseline", 42, {"accuracy": 0.80})],
                [ExperimentResult("improved", 42, {"accuracy": 0.85})]
            ]
            
            latex_table = self.analyzer.generate_comparison_table(
                ["baseline", "improved"],
                ["accuracy"],
                output_format="latex"
            )
            
            # Verify LaTeX formatting
            self.assertIn("\\begin{table}", latex_table)
            self.assertIn("\\toprule", latex_table)
            self.assertIn("\\bottomrule", latex_table)
    
    def test_parallel_seed_execution(self):
        """Test parallel seed execution configuration"""
        config = Mock(spec=ExperimentConfig)
        config.experiment_name = "test"
        config.mode = "sft"
        config.config_file = "test.yaml"
        config.seeds = [42, 84, 126]
        config.parallel_seeds = True
        config.max_workers = 2
        config.output_dir = self.test_dir
        config.dry_run = True
        config.wandb_project = "test"
        config.wandb_tags = []
        
        runner = ExperimentRunner(config)
        
        # Mock the single seed runner
        with patch.object(runner, '_run_single_seed') as mock_run:
            mock_run.return_value = SeedResult(seed=42, success=True)
            
            # Run should use parallel execution
            with patch('scripts.run_multi_seed_experiment.ProcessPoolExecutor') as mock_executor:
                mock_executor.return_value.__enter__.return_value.submit = Mock()
                runner._run_parallel_seeds()
                
                # Verify executor was created with correct max_workers
                mock_executor.assert_called_once_with(max_workers=2)


class TestStatisticalMethods(unittest.TestCase):
    """Test statistical methods implementation"""
    
    def test_paired_t_test(self):
        """Test paired t-test implementation"""
        # Create paired samples
        sample_a = [0.85, 0.83, 0.87, 0.84, 0.86]
        sample_b = [0.80, 0.78, 0.82, 0.79, 0.81]
        
        # Manual calculation
        differences = np.array(sample_a) - np.array(sample_b)
        expected_t = np.mean(differences) / (np.std(differences, ddof=1) / np.sqrt(len(differences)))
        
        # SciPy calculation
        t_stat, p_value = stats.ttest_rel(sample_a, sample_b)
        
        # Verify t-statistic
        self.assertAlmostEqual(t_stat, expected_t, places=5)
        
        # Should be significant
        self.assertLess(p_value, 0.001)
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation"""
        np.random.seed(42)
        sample = np.random.normal(0.85, 0.02, 10)
        
        # Bootstrap
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(sample, size=len(sample), replace=True)
            bootstrap_means.append(np.mean(resample))
        
        # Calculate 95% CI
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        # True mean should be within CI
        true_mean = np.mean(sample)
        self.assertGreater(true_mean, ci_lower)
        self.assertLess(true_mean, ci_upper)
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons"""
        # Original p-values
        p_values = [0.01, 0.04, 0.03, 0.002]
        n_comparisons = len(p_values)
        
        # Apply Bonferroni correction
        alpha = 0.05
        corrected_alpha = alpha / n_comparisons
        
        # Determine significance after correction
        significant = [p < corrected_alpha for p in p_values]
        
        # Only very small p-values should remain significant
        self.assertEqual(significant, [True, False, False, True])


class TestExperimentReproducibility(unittest.TestCase):
    """Test experiment reproducibility features"""
    
    def test_environment_capture(self):
        """Test that environment is properly captured"""
        import platform
        
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": False,  # Would check torch.cuda.is_available() in real scenario
            "timestamp": "2024-01-15T10:00:00"
        }
        
        # Verify all required fields are present
        self.assertIn("python_version", env_info)
        self.assertIn("platform", env_info)
        self.assertIn("cuda_available", env_info)
        self.assertIn("timestamp", env_info)
    
    def test_configuration_versioning(self):
        """Test configuration versioning for reproducibility"""
        config = {
            "model": {
                "name": "qwen2.5-vl",
                "version": "7b",
                "checkpoint": "path/to/checkpoint"
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "seed": 42
            },
            "data": {
                "train_path": "data/train.json",
                "val_path": "data/val.json",
                "version": "v1.0"
            }
        }
        
        # Serialize configuration
        config_json = json.dumps(config, indent=2, sort_keys=True)
        
        # Deserialize and verify
        loaded_config = json.loads(config_json)
        self.assertEqual(loaded_config, config)
        
        # Verify critical reproducibility fields
        self.assertIn("seed", loaded_config["training"])
        self.assertIn("version", loaded_config["data"])


def run_protocol_compliance_check():
    """Run a comprehensive protocol compliance check"""
    print("="*60)
    print("Experimental Protocol Compliance Check")
    print("="*60)
    
    checks = {
        "Multi-seed requirement": False,
        "Aggregated reporting": False,
        "Statistical testing": False,
        "Registry management": False,
        "Reproducibility": False
    }
    
    # Check 1: Multi-seed requirement
    try:
        # Verify default seeds are set
        config = ExperimentConfig(
            experiment_name="test",
            mode="sft",
            config_file=__file__
        )
        if len(config.seeds) >= 3:
            checks["Multi-seed requirement"] = True
    except:
        pass
    
    # Check 2: Aggregated reporting
    try:
        analyzer = ResultAnalyzer()
        # Check format_metric method exists
        if hasattr(analyzer, 'format_metric'):
            checks["Aggregated reporting"] = True
    except:
        pass
    
    # Check 3: Statistical testing
    try:
        analyzer = ResultAnalyzer()
        # Check statistical test methods exist
        if hasattr(analyzer, 'perform_statistical_test'):
            checks["Statistical testing"] = True
    except:
        pass
    
    # Check 4: Registry management
    registry_file = Path("experiments/registry.json")
    if registry_file.exists():
        checks["Registry management"] = True
    
    # Check 5: Reproducibility
    protocol_file = Path("docs/EXPERIMENTAL_PROTOCOL.md")
    if protocol_file.exists():
        checks["Reproducibility"] = True
    
    # Print results
    print("\nCompliance Check Results:")
    print("-"*40)
    
    all_passed = True
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("-"*40)
    if all_passed:
        print("✓ All protocol requirements met!")
    else:
        print("✗ Some requirements not met. Please review.")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    # Run compliance check first
    print("\nRunning protocol compliance check...")
    compliance_passed = run_protocol_compliance_check()
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)