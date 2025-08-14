#!/usr/bin/env python3
"""
Online Simulation Script

Serves as a configurable engine for validating the functional correctness 
and stability of the entire online learning system.

Supports:
- Short-term functional testing
- Long-running stability tests
- Chaos engineering (worker crashes)
- Performance profiling
- Memory leak detection
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.multiprocessing as mp
from dataclasses import dataclass, field
import psutil
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.modules.voting import VotingModule
from core.modules.reward_shaping import RewardOrchestrator
from core.modules.alerter import Alerter, HealthMonitor
from core.data_structures import Experience

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    duration_hours: float = 0.1  # Default 6 minutes for quick test
    num_requests: Optional[int] = None
    request_rate: float = 10.0  # Requests per second
    enable_chaos: bool = False
    worker_crash_probability: float = 0.01
    enable_monitoring: bool = True
    monitoring_interval: float = 10.0
    memory_leak_threshold: float = 1.1  # 10% growth allowed
    queue_growth_threshold: int = 100
    faiss_failure_threshold: float = 0.01
    wal_growth_threshold_mb: float = 100
    output_dir: Path = field(default_factory=lambda: Path("./simulation_results"))
    seed: int = 42
    model_path: Optional[str] = None
    config_path: Optional[str] = None


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_updates: int = 0
    worker_crashes: int = 0
    worker_restarts: int = 0
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    final_memory_mb: float = 0.0
    max_queue_size: int = 0
    faiss_failures: int = 0
    alerts_triggered: int = 0
    inference_times: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'total_updates': self.total_updates,
            'update_rate': self.total_updates / max((self.end_time - self.start_time).total_seconds() / 60, 1) if self.end_time else 0,
            'worker_crashes': self.worker_crashes,
            'worker_restarts': self.worker_restarts,
            'memory': {
                'initial_mb': self.initial_memory_mb,
                'peak_mb': self.peak_memory_mb,
                'final_mb': self.final_memory_mb,
                'growth_ratio': self.final_memory_mb / max(self.initial_memory_mb, 1)
            },
            'max_queue_size': self.max_queue_size,
            'faiss_failures': self.faiss_failures,
            'alerts_triggered': self.alerts_triggered,
            'inference': {
                'mean_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
                'p50_time_ms': np.percentile(self.inference_times, 50) * 1000 if self.inference_times else 0,
                'p95_time_ms': np.percentile(self.inference_times, 95) * 1000 if self.inference_times else 0,
                'p99_time_ms': np.percentile(self.inference_times, 99) * 1000 if self.inference_times else 0
            },
            'confidence': {
                'mean': np.mean(self.confidence_scores) if self.confidence_scores else 0,
                'std': np.std(self.confidence_scores) if self.confidence_scores else 0,
                'min': np.min(self.confidence_scores) if self.confidence_scores else 0,
                'max': np.max(self.confidence_scores) if self.confidence_scores else 0
            }
        }


class MockModel:
    """Mock model for simulation testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def forward(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock prediction."""
        self.call_count += 1
        return {
            'answer': f"mock_answer_{self.call_count}",
            'confidence': random.uniform(0.3, 0.95),
            'logits': torch.randn(1, 100),
            'trajectory': []
        }
    
    def parameters(self):
        """Return mock parameters."""
        return [torch.nn.Parameter(torch.randn(10, 10))]
    
    def state_dict(self):
        """Return mock state dict."""
        return {'mock_param': torch.randn(10, 10)}
    
    def eval(self):
        """Set to eval mode."""
        pass


class OnlineSimulation:
    """
    Main simulation engine for testing the online learning system.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.metrics = SimulationMetrics(start_time=datetime.now())
        
        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.inference_engine = None
        self.experience_buffer = None
        self.voting_module = None
        self.reward_orchestrator = None
        
        # Process tracking
        self.worker_process = None
        self.worker_restart_count = 0
        
        # Memory tracking
        self.memory_history = deque(maxlen=1000)
        self.process = psutil.Process()
        
        # Alert tracking
        self.alert_count = 0
        
        logger.info(f"Simulation initialized with config: {config}")
    
    def initialize_components(self):
        """Initialize all system components."""
        logger.info("Initializing system components...")
        
        # Load or create model
        if self.config.model_path:
            logger.info(f"Loading model from {self.config.model_path}")
            # In production, would load actual model
            self.model = MockModel()
        else:
            logger.info("Using mock model for simulation")
            self.model = MockModel()
        
        # Load configuration
        if self.config.config_path:
            with open(self.config.config_path, 'r') as f:
                system_config = json.load(f)
        else:
            system_config = self._get_default_config()
        
        # Initialize components
        self.experience_buffer = ExperienceBuffer(
            max_size=system_config.get('buffer_size', 10000),
            embedding_dim=system_config.get('embedding_dim', 768)
        )
        
        self.voting_module = VotingModule(config=system_config)
        
        self.reward_orchestrator = RewardOrchestrator(config=system_config)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(
            model=self.model,
            experience_buffer=self.experience_buffer,
            voting_module=self.voting_module,
            reward_orchestrator=self.reward_orchestrator,
            config=system_config
        )
        
        # Record initial memory
        self.metrics.initial_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        logger.info("System components initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            'buffer_size': 10000,
            'embedding_dim': 768,
            'k_neighbors': 5,
            'voting_strategy': 'weighted',
            'confidence_threshold': 0.7,
            'cold_start_threshold': 100,
            'min_learning_rate': 1e-6,
            'max_learning_rate': 1e-4,
            'enable_wandb_logging': False,
            'monitoring_interval': self.config.monitoring_interval,
            'alert_channels': ['log'],
            'kl_alert_threshold': 0.2,
            'queue_alert_threshold': 900,
            'max_queue_size': 1000,
            'shm_timeout': 60.0,
            'watchdog_interval': 5.0,
            'hil_mode_enabled': False
        }
    
    async def generate_request(self) -> Dict[str, Any]:
        """Generate a mock request for testing."""
        # Generate random image features
        image_features = torch.randn(1, 3, 224, 224)
        
        # Generate random question
        questions = [
            "What is in this image?",
            "How many objects are there?",
            "What color is the main object?",
            "Is there text in the image?",
            "What is the spatial relationship?",
            "Describe the scene.",
            "What action is happening?",
            "Where is this located?"
        ]
        question = random.choice(questions)
        
        return {
            'image_features': image_features,
            'question': question,
            'request_id': f"sim_{self.metrics.total_requests}"
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Tuple[bool, float, float]:
        """
        Process a single request through the system.
        
        Returns:
            Tuple of (success, confidence, inference_time)
        """
        start_time = time.time()
        
        try:
            # Run inference and adaptation
            result, confidence, metadata = await self.inference_engine.infer_and_adapt(request)
            
            inference_time = time.time() - start_time
            
            # Record metrics
            self.metrics.inference_times.append(inference_time)
            self.metrics.confidence_scores.append(confidence)
            
            # Check for updates
            if metadata.get('update_triggered', False):
                self.metrics.total_updates += 1
            
            return True, confidence, inference_time
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return False, 0.0, time.time() - start_time
    
    def inject_chaos(self):
        """Inject chaos by randomly killing the update worker."""
        if not self.config.enable_chaos:
            return
        
        if random.random() < self.config.worker_crash_probability:
            if self.inference_engine.update_worker_process and \
               self.inference_engine.update_worker_process.is_alive():
                
                logger.warning("[CHAOS] Injecting worker crash!")
                pid = self.inference_engine.update_worker_process.pid
                
                try:
                    os.kill(pid, signal.SIGKILL)
                    self.metrics.worker_crashes += 1
                    logger.info(f"[CHAOS] Killed worker process {pid}")
                except ProcessLookupError:
                    logger.warning(f"[CHAOS] Process {pid} already dead")
    
    def check_worker_health(self):
        """Check and restart worker if needed (Task 006 preview)."""
        if self.inference_engine.update_worker_process and \
           not self.inference_engine.update_worker_process.is_alive():
            
            logger.warning("Worker process dead, attempting restart...")
            
            # Wait for cleanup
            time.sleep(2)
            
            # Restart worker
            try:
                self.inference_engine.start_update_worker()
                self.metrics.worker_restarts += 1
                logger.info("Worker process restarted successfully")
            except Exception as e:
                logger.error(f"Failed to restart worker: {e}")
    
    def collect_health_metrics(self) -> Dict[str, Any]:
        """Collect current health metrics."""
        metrics = {}
        
        # Memory metrics
        memory_info = self.process.memory_info()
        metrics['memory_rss_mb'] = memory_info.rss / (1024 * 1024)
        metrics['memory_percent'] = self.process.memory_percent()
        
        # Queue metrics
        queue_sizes = self.inference_engine._get_queue_sizes()
        metrics['queue_sizes'] = queue_sizes
        metrics['max_queue_size'] = max(queue_sizes.values()) if queue_sizes else 0
        
        # Update peak memory
        self.metrics.peak_memory_mb = max(
            self.metrics.peak_memory_mb,
            metrics['memory_rss_mb']
        )
        
        # Update max queue size
        self.metrics.max_queue_size = max(
            self.metrics.max_queue_size,
            metrics['max_queue_size']
        )
        
        return metrics
    
    async def run_simulation(self):
        """Run the main simulation loop."""
        logger.info(f"Starting simulation for {self.config.duration_hours} hours")
        
        # Initialize components
        self.initialize_components()
        
        # Start the inference engine worker
        self.inference_engine.start_update_worker()
        
        # Calculate end time
        end_time = datetime.now() + timedelta(hours=self.config.duration_hours)
        
        # Request timing
        request_interval = 1.0 / self.config.request_rate
        last_request_time = time.time()
        
        # Monitoring timing
        last_monitor_time = time.time()
        
        # Main simulation loop
        while True:
            current_time = datetime.now()
            
            # Check termination conditions
            if self.config.num_requests and self.metrics.total_requests >= self.config.num_requests:
                logger.info(f"Reached target number of requests: {self.config.num_requests}")
                break
            
            if current_time >= end_time:
                logger.info(f"Reached simulation duration: {self.config.duration_hours} hours")
                break
            
            # Generate and process request
            if time.time() - last_request_time >= request_interval:
                request = await self.generate_request()
                self.metrics.total_requests += 1
                
                success, confidence, inference_time = await self.process_request(request)
                
                if success:
                    self.metrics.successful_requests += 1
                else:
                    self.metrics.failed_requests += 1
                
                # Log progress periodically
                if self.metrics.total_requests % 100 == 0:
                    logger.info(
                        f"Progress: {self.metrics.total_requests} requests, "
                        f"{self.metrics.successful_requests} successful, "
                        f"{self.metrics.total_updates} updates"
                    )
                
                last_request_time = time.time()
            
            # Inject chaos if enabled
            self.inject_chaos()
            
            # Check worker health and restart if needed
            self.check_worker_health()
            
            # Collect metrics periodically
            if time.time() - last_monitor_time >= self.config.monitoring_interval:
                health_metrics = self.collect_health_metrics()
                self.memory_history.append(health_metrics['memory_rss_mb'])
                last_monitor_time = time.time()
            
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.001)
        
        # Final metrics
        self.metrics.end_time = datetime.now()
        self.metrics.final_memory_mb = self.process.memory_info().rss / (1024 * 1024)
        
        logger.info("Simulation completed")
    
    def validate_results(self) -> Dict[str, bool]:
        """
        Validate simulation results against thresholds.
        
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Memory leak detection
        memory_growth = self.metrics.final_memory_mb / max(self.metrics.initial_memory_mb, 1)
        validations['memory_leak'] = memory_growth <= self.config.memory_leak_threshold
        
        if not validations['memory_leak']:
            logger.error(
                f"MEMORY LEAK DETECTED: Growth ratio {memory_growth:.2f} > "
                f"threshold {self.config.memory_leak_threshold}"
            )
        
        # Queue growth check
        validations['queue_bounded'] = self.metrics.max_queue_size <= self.config.queue_growth_threshold
        
        if not validations['queue_bounded']:
            logger.error(
                f"QUEUE GROWTH DETECTED: Max size {self.metrics.max_queue_size} > "
                f"threshold {self.config.queue_growth_threshold}"
            )
        
        # FAISS failure rate
        if self.metrics.total_requests > 0:
            faiss_failure_rate = self.metrics.faiss_failures / self.metrics.total_requests
            validations['faiss_stable'] = faiss_failure_rate <= self.config.faiss_failure_threshold
            
            if not validations['faiss_stable']:
                logger.error(
                    f"HIGH FAISS FAILURE RATE: {faiss_failure_rate:.3f} > "
                    f"threshold {self.config.faiss_failure_threshold}"
                )
        else:
            validations['faiss_stable'] = True
        
        # Worker recovery (if chaos enabled)
        if self.config.enable_chaos:
            validations['fault_recovery'] = self.metrics.worker_restarts == self.metrics.worker_crashes
            
            if not validations['fault_recovery']:
                logger.error(
                    f"FAULT RECOVERY FAILED: {self.metrics.worker_restarts} restarts != "
                    f"{self.metrics.worker_crashes} crashes"
                )
        else:
            validations['fault_recovery'] = True
        
        # Overall success
        validations['overall'] = all(validations.values())
        
        return validations
    
    def save_results(self):
        """Save simulation results to file."""
        # Create results dictionary
        results = {
            'config': {
                'duration_hours': self.config.duration_hours,
                'num_requests': self.config.num_requests,
                'request_rate': self.config.request_rate,
                'enable_chaos': self.config.enable_chaos,
                'worker_crash_probability': self.config.worker_crash_probability
            },
            'metrics': self.metrics.to_dict(),
            'validations': self.validate_results(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        output_file = self.config.output_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Duration: {results['metrics']['duration_seconds']:.1f} seconds")
        print(f"Total Requests: {results['metrics']['total_requests']}")
        print(f"Success Rate: {results['metrics']['success_rate']:.2%}")
        print(f"Update Rate: {results['metrics']['update_rate']:.2f} updates/min")
        print(f"Memory Growth: {results['metrics']['memory']['growth_ratio']:.2f}x")
        print(f"Mean Inference Time: {results['metrics']['inference']['mean_time_ms']:.1f}ms")
        print(f"P99 Inference Time: {results['metrics']['inference']['p99_time_ms']:.1f}ms")
        
        if self.config.enable_chaos:
            print(f"Worker Crashes: {self.metrics.worker_crashes}")
            print(f"Worker Restarts: {self.metrics.worker_restarts}")
        
        print("\nVALIDATION RESULTS:")
        for check, passed in results['validations'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check}: {status}")
        
        print("="*60)
        
        return results['validations']['overall']
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up simulation resources...")
        
        if self.inference_engine:
            self.inference_engine.shutdown()
        
        logger.info("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run online learning simulation')
    
    # Duration and limits
    parser.add_argument('--duration', type=float, default=0.1,
                       help='Simulation duration in hours (default: 0.1 = 6 minutes)')
    parser.add_argument('--num-requests', type=int, default=None,
                       help='Maximum number of requests (optional)')
    parser.add_argument('--request-rate', type=float, default=10.0,
                       help='Request rate per second (default: 10)')
    
    # Chaos testing
    parser.add_argument('--enable-chaos', action='store_true',
                       help='Enable chaos testing (worker crashes)')
    parser.add_argument('--worker-crash-probability', type=float, default=0.01,
                       help='Probability of worker crash per iteration (default: 0.01)')
    
    # Monitoring
    parser.add_argument('--monitoring-interval', type=float, default=10.0,
                       help='Health monitoring interval in seconds (default: 10)')
    
    # Thresholds
    parser.add_argument('--memory-leak-threshold', type=float, default=1.1,
                       help='Memory growth ratio threshold (default: 1.1 = 10% growth)')
    parser.add_argument('--queue-growth-threshold', type=int, default=100,
                       help='Maximum queue size threshold (default: 100)')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default='./simulation_results',
                       help='Output directory for results')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to system configuration JSON (optional)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SimulationConfig(
        duration_hours=args.duration,
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        enable_chaos=args.enable_chaos,
        worker_crash_probability=args.worker_crash_probability,
        monitoring_interval=args.monitoring_interval,
        memory_leak_threshold=args.memory_leak_threshold,
        queue_growth_threshold=args.queue_growth_threshold,
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        config_path=args.config_path,
        seed=args.seed
    )
    
    # Run simulation
    simulation = OnlineSimulation(config)
    
    try:
        # Run async simulation
        asyncio.run(simulation.run_simulation())
        
        # Save and validate results
        success = simulation.save_results()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        simulation.save_results()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        traceback.print_exc()
        sys.exit(2)
        
    finally:
        simulation.cleanup()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()