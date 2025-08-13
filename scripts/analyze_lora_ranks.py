#!/usr/bin/env python3
"""
SVD Analysis Script for Dynamic LoRA Rank Configuration
This script performs Singular Value Decomposition (SVD) on weight deltas
to determine optimal LoRA ranks for parameter-efficient fine-tuning.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd as scipy_svd
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class SVDAnalysisConfig:
    """Configuration for SVD analysis"""
    pretrained_model_path: str
    finetuned_model_path: str
    output_dir: str = "analysis_outputs/svd"
    svd_threshold: float = 0.9  # Energy retention threshold
    min_rank: int = 4
    max_rank: int = 128
    smoothing_factor: float = 0.8
    use_randomized_svd: bool = True
    n_oversamples: int = 10  # For randomized SVD
    save_plots: bool = True
    save_raw_data: bool = True
    save_delta_weights: bool = False  # Can be memory intensive
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LayerSVDResult:
    """Results of SVD analysis for a single layer"""
    layer_name: str
    original_shape: Tuple[int, ...]
    singular_values: np.ndarray
    energy_retained: float
    r_raw: int  # Raw rank from SVD
    r_final: int  # Final rank after constraints
    decay_rate: float  # Singular value decay rate
    compression_ratio: float


class SVDAnalyzer:
    """Performs SVD analysis on model weight deltas for LoRA rank determination"""
    
    def __init__(self, config: SVDAnalysisConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.save_plots:
            self.plot_dir = self.output_dir / "plots"
            self.plot_dir.mkdir(exist_ok=True)
            
        if config.save_raw_data:
            self.data_dir = self.output_dir / "raw_data"
            self.data_dir.mkdir(exist_ok=True)
            
        if config.save_delta_weights:
            self.delta_dir = self.output_dir / "delta_weights"
            self.delta_dir.mkdir(exist_ok=True)
    
    def load_model_weights(self, model_path: str) -> Dict[str, torch.Tensor]:
        """Load model weights from checkpoint"""
        logger.info(f"Loading model weights from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        return state_dict
    
    def compute_weight_delta(
        self, 
        pretrained_weights: Dict[str, torch.Tensor],
        finetuned_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute weight delta: W_finetuned - W_pretrained"""
        logger.info("Computing weight deltas")
        
        delta_weights = {}
        
        # Find common keys
        common_keys = set(pretrained_weights.keys()) & set(finetuned_weights.keys())
        logger.info(f"Found {len(common_keys)} common layers")
        
        for key in tqdm(common_keys, desc="Computing deltas"):
            if pretrained_weights[key].shape == finetuned_weights[key].shape:
                delta = finetuned_weights[key] - pretrained_weights[key]
                
                # Only store non-zero deltas
                if torch.abs(delta).max() > 1e-8:
                    delta_weights[key] = delta
                    
        logger.info(f"Computed {len(delta_weights)} non-zero weight deltas")
        return delta_weights
    
    def perform_svd(
        self, 
        weight_matrix: torch.Tensor,
        use_randomized: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform SVD on a weight matrix"""
        
        # Convert to numpy and handle different shapes
        matrix = weight_matrix.detach().cpu().numpy()
        
        # Reshape if needed (e.g., for conv layers)
        original_shape = matrix.shape
        if len(matrix.shape) > 2:
            matrix = matrix.reshape(matrix.shape[0], -1)
        
        if use_randomized and min(matrix.shape) > 100:
            # Use randomized SVD for efficiency on large matrices
            n_components = min(min(matrix.shape) - 1, 512)
            U, S, Vt = randomized_svd(
                matrix, 
                n_components=n_components,
                n_oversamples=self.config.n_oversamples,
                random_state=42
            )
        else:
            # Use full SVD for smaller matrices
            U, S, Vt = scipy_svd(matrix, full_matrices=False)
            
        return U, S, Vt
    
    def analyze_singular_values(
        self, 
        singular_values: np.ndarray,
        layer_name: str = ""
    ) -> Tuple[int, float, float]:
        """Analyze singular values to determine optimal rank"""
        
        # Normalize singular values
        normalized_sv = singular_values / singular_values[0] if singular_values[0] > 0 else singular_values
        
        # Calculate cumulative energy
        cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        
        # Find rank that retains threshold energy
        r_raw = np.argmax(cumulative_energy >= self.config.svd_threshold) + 1
        
        # Calculate decay rate (how quickly singular values drop)
        if len(singular_values) > 1:
            decay_rate = -np.log(normalized_sv[min(10, len(normalized_sv)-1)])
        else:
            decay_rate = 0.0
            
        energy_retained = cumulative_energy[r_raw-1] if r_raw > 0 else 0.0
        
        return r_raw, energy_retained, decay_rate
    
    def apply_rank_constraints(
        self, 
        r_raw: int,
        layer_name: str = ""
    ) -> int:
        """Apply bounding and smoothing constraints to raw rank"""
        
        # Apply min/max bounds
        r_bounded = np.clip(r_raw, self.config.min_rank, self.config.max_rank)
        
        # Round to nearest power of 2 for efficiency (optional)
        if r_bounded > 16:
            r_final = int(2 ** np.round(np.log2(r_bounded)))
        else:
            r_final = int(r_bounded)
            
        # Ensure it's still within bounds after rounding
        r_final = np.clip(r_final, self.config.min_rank, self.config.max_rank)
        
        return r_final
    
    def smooth_ranks_across_layers(
        self, 
        layer_results: Dict[str, LayerSVDResult]
    ) -> Dict[str, LayerSVDResult]:
        """Apply smoothing across similar layers to prevent extreme variance"""
        
        if self.config.smoothing_factor <= 0:
            return layer_results
            
        # Group layers by type
        layer_groups = {}
        for name, result in layer_results.items():
            # Extract layer type (e.g., q_proj, k_proj, etc.)
            layer_type = name.split('.')[-1] if '.' in name else name
            
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append((name, result))
        
        # Apply smoothing within each group
        smoothed_results = {}
        for layer_type, group in layer_groups.items():
            if len(group) > 1:
                # Calculate mean rank for the group
                mean_rank = np.mean([r.r_final for _, r in group])
                
                # Apply smoothing
                for name, result in group:
                    smoothed_rank = int(
                        self.config.smoothing_factor * mean_rank + 
                        (1 - self.config.smoothing_factor) * result.r_final
                    )
                    smoothed_rank = np.clip(smoothed_rank, self.config.min_rank, self.config.max_rank)
                    
                    # Update result
                    result.r_final = smoothed_rank
                    smoothed_results[name] = result
            else:
                name, result = group[0]
                smoothed_results[name] = result
                
        return smoothed_results
    
    def plot_singular_values(
        self, 
        singular_values: np.ndarray,
        layer_name: str,
        r_final: int,
        energy_retained: float
    ):
        """Plot singular value decay curve"""
        
        if not self.config.save_plots:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot singular values
        plt.subplot(1, 2, 1)
        plt.semilogy(singular_values[:min(100, len(singular_values))], 'b-', linewidth=2)
        plt.axvline(x=r_final, color='r', linestyle='--', label=f'Rank={r_final}')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title(f'{layer_name} - Singular Value Decay')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot cumulative energy
        plt.subplot(1, 2, 2)
        cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        plt.plot(cumulative_energy[:min(100, len(cumulative_energy))], 'g-', linewidth=2)
        plt.axhline(y=self.config.svd_threshold, color='orange', linestyle='--', 
                   label=f'Threshold={self.config.svd_threshold}')
        plt.axvline(x=r_final, color='r', linestyle='--', label=f'Rank={r_final}')
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Energy')
        plt.title(f'Energy Retention: {energy_retained:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_name = layer_name.replace('/', '_').replace('.', '_')
        plt.savefig(self.plot_dir / f"{safe_name}_svd.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(
        self,
        layer_results: Dict[str, LayerSVDResult],
        model_name: str = "Qwen/Qwen2.5-VL-7B"
    ):
        """Save analysis results to JSON and other formats"""
        
        # Prepare LoRA rank configuration
        lora_config = {
            "description": "LoRA rank configuration determined by SVD analysis",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "analysis_metadata": {
                "num_samples": "full_model",
                "svd_threshold": self.config.svd_threshold,
                "min_rank": self.config.min_rank,
                "max_rank": self.config.max_rank,
                "smoothing_factor": self.config.smoothing_factor
            },
            "layer_ranks": {},
            "layer_metadata": {},
            "total_parameters": 0,
            "compression_ratio": 0.0
        }
        
        # Extract ranks and metadata for each layer type
        layer_type_ranks = {}
        layer_type_metadata = {}
        total_lora_params = 0
        total_original_params = 0
        
        for name, result in layer_results.items():
            # Extract layer type
            layer_type = name.split('.')[-1] if '.' in name else name
            
            # Store rank
            if layer_type not in layer_type_ranks:
                layer_type_ranks[layer_type] = []
                layer_type_metadata[layer_type] = []
                
            layer_type_ranks[layer_type].append(result.r_final)
            layer_type_metadata[layer_type].append({
                "layer": name,
                "r_raw": result.r_raw,
                "r_final": result.r_final,
                "energy_retained": float(result.energy_retained),
                "decay_rate": float(result.decay_rate),
                "compression_ratio": float(result.compression_ratio)
            })
            
            # Calculate parameters
            orig_params = np.prod(result.original_shape)
            lora_params = result.r_final * sum(result.original_shape[:2])
            total_original_params += orig_params
            total_lora_params += lora_params
        
        # Average ranks for each layer type
        for layer_type in layer_type_ranks:
            ranks = layer_type_ranks[layer_type]
            lora_config["layer_ranks"][layer_type] = int(np.median(ranks))
            lora_config["layer_metadata"][layer_type] = layer_type_metadata[layer_type]
        
        # Calculate overall compression
        lora_config["total_parameters"] = int(total_lora_params)
        lora_config["compression_ratio"] = float(total_lora_params / total_original_params) if total_original_params > 0 else 0.0
        
        # Save main configuration
        config_path = Path("configs/lora_rank_config.json")
        with open(config_path, 'w') as f:
            json.dump(lora_config, f, indent=2)
        logger.info(f"Saved LoRA configuration to {config_path}")
        
        # Save detailed analysis results
        if self.config.save_raw_data:
            detailed_results = {}
            for name, result in layer_results.items():
                detailed_results[name] = {
                    "original_shape": result.original_shape,
                    "r_raw": result.r_raw,
                    "r_final": result.r_final,
                    "energy_retained": float(result.energy_retained),
                    "decay_rate": float(result.decay_rate),
                    "compression_ratio": float(result.compression_ratio),
                    "singular_values": result.singular_values[:50].tolist()  # Save top 50
                }
                
            detailed_path = self.data_dir / "detailed_svd_analysis.json"
            with open(detailed_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"Saved detailed analysis to {detailed_path}")
    
    def run_analysis(self) -> Dict[str, LayerSVDResult]:
        """Run complete SVD analysis pipeline"""
        
        logger.info("Starting SVD analysis for LoRA rank determination")
        
        # Load model weights
        pretrained_weights = self.load_model_weights(self.config.pretrained_model_path)
        finetuned_weights = self.load_model_weights(self.config.finetuned_model_path)
        
        # Compute weight deltas
        delta_weights = self.compute_weight_delta(pretrained_weights, finetuned_weights)
        
        # Analyze each layer
        layer_results = {}
        target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                        'gate_proj', 'up_proj', 'down_proj']
        
        for layer_name, delta in tqdm(delta_weights.items(), desc="Analyzing layers"):
            # Filter to only relevant layers
            if not any(target in layer_name for target in target_layers):
                continue
                
            # Perform SVD
            U, S, Vt = self.perform_svd(delta, use_randomized=self.config.use_randomized_svd)
            
            # Analyze singular values
            r_raw, energy_retained, decay_rate = self.analyze_singular_values(S, layer_name)
            
            # Apply constraints
            r_final = self.apply_rank_constraints(r_raw, layer_name)
            
            # Calculate compression ratio
            original_params = np.prod(delta.shape)
            lora_params = r_final * sum(delta.shape[:2])
            compression_ratio = lora_params / original_params
            
            # Store results
            result = LayerSVDResult(
                layer_name=layer_name,
                original_shape=tuple(delta.shape),
                singular_values=S,
                energy_retained=energy_retained,
                r_raw=r_raw,
                r_final=r_final,
                decay_rate=decay_rate,
                compression_ratio=compression_ratio
            )
            layer_results[layer_name] = result
            
            # Plot if enabled (only for selected layers to avoid too many plots)
            if self.config.save_plots and len(layer_results) <= 10:
                self.plot_singular_values(S, layer_name, r_final, energy_retained)
            
            # Save delta weights if requested (only for small layers)
            if self.config.save_delta_weights and delta.numel() < 1e6:
                safe_name = layer_name.replace('/', '_').replace('.', '_')
                torch.save(delta, self.delta_dir / f"{safe_name}_delta.pt")
        
        # Apply smoothing across layers
        layer_results = self.smooth_ranks_across_layers(layer_results)
        
        # Save results
        self.save_analysis_results(layer_results)
        
        # Print summary
        self.print_summary(layer_results)
        
        return layer_results
    
    def print_summary(self, layer_results: Dict[str, LayerSVDResult]):
        """Print analysis summary"""
        
        print("\n" + "="*80)
        print("SVD ANALYSIS SUMMARY")
        print("="*80)
        
        # Group by layer type
        layer_types = {}
        for name, result in layer_results.items():
            layer_type = name.split('.')[-1] if '.' in name else name
            if layer_type not in layer_types:
                layer_types[layer_type] = []
            layer_types[layer_type].append(result)
        
        # Print summary for each layer type
        total_original = 0
        total_lora = 0
        
        for layer_type, results in layer_types.items():
            ranks = [r.r_final for r in results]
            compressions = [r.compression_ratio for r in results]
            
            print(f"\n{layer_type}:")
            print(f"  Median Rank: {int(np.median(ranks))}")
            print(f"  Rank Range: {min(ranks)} - {max(ranks)}")
            print(f"  Avg Compression: {np.mean(compressions):.2%}")
            
            # Calculate total parameters
            for r in results:
                total_original += np.prod(r.original_shape)
                total_lora += r.r_final * sum(r.original_shape[:2])
        
        print(f"\nOverall Statistics:")
        print(f"  Total Original Parameters: {total_original:,}")
        print(f"  Total LoRA Parameters: {total_lora:,}")
        print(f"  Overall Compression Ratio: {total_lora/total_original:.2%}")
        print("="*80)


def main():
    """Main entry point for the script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze model weight deltas to determine optimal LoRA ranks"
    )
    parser.add_argument(
        "--pretrained", 
        type=str, 
        required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--finetuned", 
        type=str, 
        required=True,
        help="Path to finetuned model checkpoint"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="analysis_outputs/svd",
        help="Directory to save analysis outputs"
    )
    parser.add_argument(
        "--svd-threshold", 
        type=float, 
        default=0.9,
        help="Energy retention threshold for rank selection (0-1)"
    )
    parser.add_argument(
        "--min-rank", 
        type=int, 
        default=4,
        help="Minimum allowed rank"
    )
    parser.add_argument(
        "--max-rank", 
        type=int, 
        default=128,
        help="Maximum allowed rank"
    )
    parser.add_argument(
        "--smoothing-factor", 
        type=float, 
        default=0.8,
        help="Smoothing factor for rank regularization across layers (0-1)"
    )
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Disable saving plots"
    )
    parser.add_argument(
        "--save-delta-weights", 
        action="store_true",
        help="Save delta weight matrices (can be memory intensive)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SVDAnalysisConfig(
        pretrained_model_path=args.pretrained,
        finetuned_model_path=args.finetuned,
        output_dir=args.output_dir,
        svd_threshold=args.svd_threshold,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        smoothing_factor=args.smoothing_factor,
        save_plots=not args.no_plots,
        save_delta_weights=args.save_delta_weights
    )
    
    # Run analysis
    analyzer = SVDAnalyzer(config)
    results = analyzer.run_analysis()
    
    logger.info("SVD analysis completed successfully!")
    

if __name__ == "__main__":
    main()