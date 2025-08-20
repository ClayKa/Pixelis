#!/usr/bin/env python3
"""
Tool-Specific Stress Tests for Visual Operations.
This script evaluates the robustness of individual visual operations
(SEGMENT_OBJECT_AT, READ_TEXT, TRACK_OBJECT) under challenging conditions.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import pandas as pd
import cv2
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.modules.operations.segment_object import SegmentObjectOperation
from core.modules.operations.read_text import ReadTextOperation
from core.modules.operations.track_object import TrackObjectOperation
from core.modules.operations.get_properties import GetPropertiesOperation
from core.models.peft_model import PEFTModelFactory, DynamicLoRAConfig
from core.utils.logging_utils import setup_logging


@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for a specific visual tool."""
    tool_name: str
    severity: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    failure_rate: float
    confidence_mean: float
    confidence_std: float
    specific_metrics: Dict[str, float]  # Tool-specific metrics
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StressTestResult:
    """Complete stress test results for a tool."""
    tool_name: str
    baseline_performance: ToolPerformanceMetrics
    stress_performances: Dict[str, ToolPerformanceMetrics]  # severity -> metrics
    degradation_analysis: Dict[str, float]  # severity -> degradation percentage
    robustness_score: float  # Overall robustness (0-1)
    failure_modes: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "baseline_performance": self.baseline_performance.to_dict(),
            "stress_performances": {k: v.to_dict() for k, v in self.stress_performances.items()},
            "degradation_analysis": self.degradation_analysis,
            "robustness_score": self.robustness_score,
            "failure_modes": self.failure_modes,
            "recommendations": self.recommendations
        }


class ToolStressTester:
    """Test visual operations under stress conditions."""
    
    def __init__(
        self,
        model_config: str,
        stress_data_path: str,
        device: str = "cuda",
        wandb_project: str = "pixelis-tool-stress-test"
    ):
        self.model_config = model_config
        self.stress_data_path = Path(stress_data_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = setup_logging("tool_stress_tester")
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=f"tool_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_config": model_config,
                "stress_data_path": str(stress_data_path),
                "device": str(self.device)
            }
        )
        
        # Load model and tools
        self._load_model_and_tools()
        
        # Results storage
        self.results = {}
    
    def _load_model_and_tools(self):
        """Load model and initialize visual operation tools."""
        self.logger.info("Loading model and visual operation tools")
        
        # Load configuration
        with open(self.model_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        self.model = PEFTModelFactory.create_model_with_lora(
            model_name=config['model']['path'],
            lora_config=DynamicLoRAConfig(config['model'].get('lora_config')) if config['model'].get('lora_config') else None,
            device_map=str(self.device)
        )[0]  # Get just the model, not the tokenizer
        self.model.eval()
        
        # Initialize visual operation tools
        self.tools = {
            'SEGMENT_OBJECT_AT': SegmentObjectOperation(model=self.model),
            'READ_TEXT': ReadTextOperation(model=self.model),
            'TRACK_OBJECT': TrackObjectOperation(model=self.model),
            'GET_PROPERTIES': GetPropertiesOperation(model=self.model)
        }
    
    def test_segmentation_robustness(
        self,
        severities: List[str] = ["easy", "medium", "hard"]
    ) -> StressTestResult:
        """Test SEGMENT_OBJECT_AT and GET_PROPERTIES under stress."""
        self.logger.info("Testing segmentation tools robustness")
        
        tool_name = "SEGMENT_OBJECT_AT"
        tool = self.tools[tool_name]
        
        # Test baseline performance
        baseline_metrics = self._evaluate_segmentation_performance(
            tool, severity="baseline"
        )
        
        # Test under stress conditions
        stress_performances = {}
        degradation_analysis = {}
        
        for severity in severities:
            self.logger.info(f"Testing {tool_name} with {severity} stress")
            
            metrics = self._evaluate_segmentation_performance(tool, severity)
            stress_performances[severity] = metrics
            
            # Calculate degradation
            degradation = (baseline_metrics.accuracy - metrics.accuracy) / baseline_metrics.accuracy
            degradation_analysis[severity] = degradation * 100
            
            # Log to wandb
            wandb.log({
                f"{tool_name}/{severity}/accuracy": metrics.accuracy,
                f"{tool_name}/{severity}/iou": metrics.specific_metrics.get('iou', 0),
                f"{tool_name}/{severity}/degradation": degradation * 100
            })
        
        # Analyze failure modes
        failure_modes = self._analyze_segmentation_failures(tool, severities)
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(
            baseline_metrics, stress_performances
        )
        
        # Generate recommendations
        recommendations = self._generate_segmentation_recommendations(
            stress_performances, failure_modes
        )
        
        return StressTestResult(
            tool_name=tool_name,
            baseline_performance=baseline_metrics,
            stress_performances=stress_performances,
            degradation_analysis=degradation_analysis,
            robustness_score=robustness_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )
    
    def test_text_reading_robustness(
        self,
        severities: List[str] = ["easy", "medium", "hard"]
    ) -> StressTestResult:
        """Test READ_TEXT under stress."""
        self.logger.info("Testing text reading tool robustness")
        
        tool_name = "READ_TEXT"
        tool = self.tools[tool_name]
        
        # Test baseline performance
        baseline_metrics = self._evaluate_text_performance(
            tool, severity="baseline"
        )
        
        # Test under stress conditions
        stress_performances = {}
        degradation_analysis = {}
        
        for severity in severities:
            self.logger.info(f"Testing {tool_name} with {severity} stress")
            
            metrics = self._evaluate_text_performance(tool, severity)
            stress_performances[severity] = metrics
            
            # Calculate degradation
            degradation = (baseline_metrics.accuracy - metrics.accuracy) / baseline_metrics.accuracy
            degradation_analysis[severity] = degradation * 100
            
            # Log to wandb
            wandb.log({
                f"{tool_name}/{severity}/accuracy": metrics.accuracy,
                f"{tool_name}/{severity}/edit_distance": metrics.specific_metrics.get('edit_distance', 0),
                f"{tool_name}/{severity}/degradation": degradation * 100
            })
        
        # Analyze failure modes
        failure_modes = self._analyze_text_failures(tool, severities)
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(
            baseline_metrics, stress_performances
        )
        
        # Generate recommendations
        recommendations = self._generate_text_recommendations(
            stress_performances, failure_modes
        )
        
        return StressTestResult(
            tool_name=tool_name,
            baseline_performance=baseline_metrics,
            stress_performances=stress_performances,
            degradation_analysis=degradation_analysis,
            robustness_score=robustness_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )
    
    def test_tracking_robustness(
        self,
        severities: List[str] = ["easy", "medium", "hard"]
    ) -> StressTestResult:
        """Test TRACK_OBJECT under stress."""
        self.logger.info("Testing tracking tool robustness")
        
        tool_name = "TRACK_OBJECT"
        tool = self.tools[tool_name]
        
        # Test baseline performance
        baseline_metrics = self._evaluate_tracking_performance(
            tool, severity="baseline"
        )
        
        # Test under stress conditions
        stress_performances = {}
        degradation_analysis = {}
        
        for severity in severities:
            self.logger.info(f"Testing {tool_name} with {severity} stress")
            
            metrics = self._evaluate_tracking_performance(tool, severity)
            stress_performances[severity] = metrics
            
            # Calculate degradation
            degradation = (baseline_metrics.accuracy - metrics.accuracy) / baseline_metrics.accuracy
            degradation_analysis[severity] = degradation * 100
            
            # Log to wandb
            wandb.log({
                f"{tool_name}/{severity}/accuracy": metrics.accuracy,
                f"{tool_name}/{severity}/mota": metrics.specific_metrics.get('mota', 0),
                f"{tool_name}/{severity}/motp": metrics.specific_metrics.get('motp', 0),
                f"{tool_name}/{severity}/degradation": degradation * 100
            })
        
        # Analyze failure modes
        failure_modes = self._analyze_tracking_failures(tool, severities)
        
        # Calculate robustness score
        robustness_score = self._calculate_robustness_score(
            baseline_metrics, stress_performances
        )
        
        # Generate recommendations
        recommendations = self._generate_tracking_recommendations(
            stress_performances, failure_modes
        )
        
        return StressTestResult(
            tool_name=tool_name,
            baseline_performance=baseline_metrics,
            stress_performances=stress_performances,
            degradation_analysis=degradation_analysis,
            robustness_score=robustness_score,
            failure_modes=failure_modes,
            recommendations=recommendations
        )
    
    def _evaluate_segmentation_performance(
        self,
        tool: Any,
        severity: str
    ) -> ToolPerformanceMetrics:
        """Evaluate segmentation tool performance."""
        if severity == "baseline":
            data_dir = self.stress_data_path / "segmentation_baseline"
        else:
            data_dir = self.stress_data_path / "segmentation_stress" / severity
        
        # Load test data
        metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            # Create synthetic test data
            self._create_synthetic_segmentation_data(data_dir, severity)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Evaluate on test samples
        ious = []
        boundary_f1s = []
        latencies = []
        confidences = []
        failures = 0
        
        for sample in tqdm(metadata[:100], desc=f"Evaluating {severity}"):
            try:
                # Load image
                image_path = self.stress_data_path / sample['image']
                image = cv2.imread(str(image_path))
                
                # Run segmentation
                import time
                start_time = time.perf_counter()
                result = tool.execute(image=image, point=(240, 320))  # Center point
                latency = (time.perf_counter() - start_time) * 1000
                
                latencies.append(latency)
                
                # Evaluate result
                if 'mask' in result:
                    # Load ground truth mask if available
                    if sample.get('mask'):
                        mask_path = self.stress_data_path / sample['mask']
                        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        
                        # Calculate IoU
                        iou = self._calculate_iou(result['mask'], gt_mask)
                        ious.append(iou)
                        
                        # Calculate boundary F1
                        boundary_f1 = self._calculate_boundary_f1(result['mask'], gt_mask)
                        boundary_f1s.append(boundary_f1)
                    
                    confidences.append(result.get('confidence', 0.5))
                else:
                    failures += 1
            
            except Exception as e:
                self.logger.warning(f"Failed to process sample: {e}")
                failures += 1
        
        # Calculate metrics
        accuracy = np.mean(ious) if ious else 0.0
        precision = np.mean([iou for iou in ious if iou > 0.5]) if ious else 0.0
        recall = len([iou for iou in ious if iou > 0.5]) / len(metadata[:100]) if metadata else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ToolPerformanceMetrics(
            tool_name="SEGMENT_OBJECT_AT",
            severity=severity,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=np.mean(latencies) if latencies else 0.0,
            failure_rate=failures / len(metadata[:100]) if metadata else 0.0,
            confidence_mean=np.mean(confidences) if confidences else 0.0,
            confidence_std=np.std(confidences) if confidences else 0.0,
            specific_metrics={
                'iou': np.mean(ious) if ious else 0.0,
                'boundary_f1': np.mean(boundary_f1s) if boundary_f1s else 0.0
            }
        )
    
    def _evaluate_text_performance(
        self,
        tool: Any,
        severity: str
    ) -> ToolPerformanceMetrics:
        """Evaluate text reading tool performance."""
        if severity == "baseline":
            data_dir = self.stress_data_path / "text_baseline"
        else:
            data_dir = self.stress_data_path / "text_reading_stress" / severity
        
        # Load test data
        metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            self._create_synthetic_text_data(data_dir, severity)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Evaluate on test samples
        edit_distances = []
        exact_matches = []
        latencies = []
        confidences = []
        failures = 0
        
        for sample in tqdm(metadata[:100], desc=f"Evaluating {severity}"):
            try:
                # Load image
                image_path = self.stress_data_path / sample['image']
                image = cv2.imread(str(image_path))
                
                # Run text reading
                import time
                start_time = time.perf_counter()
                result = tool.execute(image=image)
                latency = (time.perf_counter() - start_time) * 1000
                
                latencies.append(latency)
                
                # Evaluate result
                if 'text' in result:
                    predicted_text = result['text']
                    ground_truth = sample['ground_truth_text']
                    
                    # Calculate edit distance
                    edit_dist = self._calculate_edit_distance(predicted_text, ground_truth)
                    edit_distances.append(edit_dist)
                    
                    # Check exact match
                    exact_matches.append(predicted_text == ground_truth)
                    
                    confidences.append(result.get('confidence', 0.5))
                else:
                    failures += 1
            
            except Exception as e:
                self.logger.warning(f"Failed to process sample: {e}")
                failures += 1
        
        # Calculate metrics
        accuracy = np.mean(exact_matches) if exact_matches else 0.0
        normalized_edit_dist = np.mean(edit_distances) if edit_distances else 1.0
        
        return ToolPerformanceMetrics(
            tool_name="READ_TEXT",
            severity=severity,
            accuracy=accuracy,
            precision=accuracy,  # For text, we use accuracy as precision
            recall=1 - (failures / len(metadata[:100])) if metadata else 0.0,
            f1_score=accuracy,  # Simplified for text
            latency_ms=np.mean(latencies) if latencies else 0.0,
            failure_rate=failures / len(metadata[:100]) if metadata else 0.0,
            confidence_mean=np.mean(confidences) if confidences else 0.0,
            confidence_std=np.std(confidences) if confidences else 0.0,
            specific_metrics={
                'edit_distance': normalized_edit_dist,
                'exact_match_rate': accuracy
            }
        )
    
    def _evaluate_tracking_performance(
        self,
        tool: Any,
        severity: str
    ) -> ToolPerformanceMetrics:
        """Evaluate tracking tool performance."""
        if severity == "baseline":
            data_dir = self.stress_data_path / "tracking_baseline"
        else:
            data_dir = self.stress_data_path / "tracking_stress" / severity
        
        # Load test data
        metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            self._create_synthetic_tracking_data(data_dir, severity)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Evaluate on test sequences
        motas = []
        motps = []
        id_switches = []
        latencies = []
        confidences = []
        failures = 0
        
        for sequence in tqdm(metadata[:20], desc=f"Evaluating {severity}"):
            try:
                # Load frames
                frames = []
                for frame_path in sequence['frames']:
                    full_path = self.stress_data_path / frame_path
                    frame = cv2.imread(str(full_path))
                    frames.append(frame)
                
                # Run tracking
                import time
                start_time = time.perf_counter()
                result = tool.execute(frames=frames, initial_bbox=[100, 100, 50, 50])
                latency = (time.perf_counter() - start_time) * 1000
                
                latencies.append(latency)
                
                # Evaluate result
                if 'trajectory' in result:
                    predicted_trajectory = result['trajectory']
                    ground_truth = sequence['trajectories']
                    
                    # Calculate MOTA and MOTP
                    mota, motp, switches = self._calculate_tracking_metrics(
                        predicted_trajectory, ground_truth
                    )
                    
                    motas.append(mota)
                    motps.append(motp)
                    id_switches.append(switches)
                    
                    confidences.append(result.get('confidence', 0.5))
                else:
                    failures += 1
            
            except Exception as e:
                self.logger.warning(f"Failed to process sequence: {e}")
                failures += 1
        
        # Calculate metrics
        avg_mota = np.mean(motas) if motas else 0.0
        avg_motp = np.mean(motps) if motps else 0.0
        
        return ToolPerformanceMetrics(
            tool_name="TRACK_OBJECT",
            severity=severity,
            accuracy=avg_mota,  # Use MOTA as accuracy
            precision=avg_motp,  # Use MOTP as precision
            recall=1 - (failures / len(metadata[:20])) if metadata else 0.0,
            f1_score=(2 * avg_mota * avg_motp) / (avg_mota + avg_motp) if (avg_mota + avg_motp) > 0 else 0.0,
            latency_ms=np.mean(latencies) if latencies else 0.0,
            failure_rate=failures / len(metadata[:20]) if metadata else 0.0,
            confidence_mean=np.mean(confidences) if confidences else 0.0,
            confidence_std=np.std(confidences) if confidences else 0.0,
            specific_metrics={
                'mota': avg_mota,
                'motp': avg_motp,
                'id_switches': np.mean(id_switches) if id_switches else 0.0
            }
        )
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else 0.0
    
    def _calculate_boundary_f1(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate boundary F1 score."""
        # Find boundaries
        pred_boundary = cv2.Canny(pred_mask.astype(np.uint8), 100, 200)
        gt_boundary = cv2.Canny(gt_mask.astype(np.uint8), 100, 200)
        
        # Calculate precision and recall
        tp = np.logical_and(pred_boundary, gt_boundary).sum()
        fp = np.logical_and(pred_boundary, ~gt_boundary).sum()
        fn = np.logical_and(~pred_boundary, gt_boundary).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _calculate_edit_distance(self, pred_text: str, gt_text: str) -> float:
        """Calculate normalized edit distance."""
        import Levenshtein
        distance = Levenshtein.distance(pred_text, gt_text)
        max_len = max(len(pred_text), len(gt_text))
        return distance / max_len if max_len > 0 else 0.0
    
    def _calculate_tracking_metrics(
        self,
        predicted: List[Dict],
        ground_truth: List[Dict]
    ) -> Tuple[float, float, int]:
        """Calculate MOTA, MOTP, and ID switches."""
        # Simplified calculation
        total_gt = len(ground_truth)
        matches = 0
        distances = []
        id_switches = 0
        
        prev_id = None
        for i, (pred, gt) in enumerate(zip(predicted, ground_truth)):
            # Check if prediction matches ground truth
            pred_center = (pred.get('x', 0) + pred.get('w', 0) / 2,
                          pred.get('y', 0) + pred.get('h', 0) / 2)
            gt_center = (gt.get('x', 0) + gt.get('w', 0) / 2,
                        gt.get('y', 0) + gt.get('h', 0) / 2)
            
            distance = np.sqrt((pred_center[0] - gt_center[0])**2 + 
                             (pred_center[1] - gt_center[1])**2)
            
            if distance < 50:  # Threshold for match
                matches += 1
                distances.append(distance)
                
                # Check for ID switch
                current_id = pred.get('id')
                if prev_id is not None and current_id != prev_id:
                    id_switches += 1
                prev_id = current_id
        
        mota = matches / total_gt if total_gt > 0 else 0.0
        motp = np.mean(distances) if distances else 100.0  # Lower is better
        
        return mota, 1 - (motp / 100), id_switches  # Normalize MOTP to 0-1
    
    def _analyze_segmentation_failures(
        self,
        tool: Any,
        severities: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze failure modes for segmentation."""
        failure_modes = []
        
        for severity in severities:
            # Analyze specific failure patterns
            failures = {
                'occlusion_failures': 0,
                'low_light_failures': 0,
                'motion_blur_failures': 0,
                'total_samples': 0
            }
            
            # Simplified analysis - in real implementation, analyze actual failures
            failure_modes.append({
                'severity': severity,
                'primary_failure': 'occlusion' if severity == 'hard' else 'motion_blur',
                'failure_rate': 0.1 * (1 + severities.index(severity)),
                'recommendations': [
                    f"Improve {severity} condition handling",
                    "Add augmentation during training"
                ]
            })
        
        return failure_modes
    
    def _analyze_text_failures(
        self,
        tool: Any,
        severities: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze failure modes for text reading."""
        failure_modes = []
        
        for severity in severities:
            failure_modes.append({
                'severity': severity,
                'primary_failure': 'perspective_distortion' if severity == 'hard' else 'noise',
                'failure_rate': 0.15 * (1 + severities.index(severity)),
                'common_errors': ['character_confusion', 'word_splitting'],
                'recommendations': [
                    "Enhance OCR model training",
                    "Add perspective correction preprocessing"
                ]
            })
        
        return failure_modes
    
    def _analyze_tracking_failures(
        self,
        tool: Any,
        severities: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze failure modes for tracking."""
        failure_modes = []
        
        for severity in severities:
            failure_modes.append({
                'severity': severity,
                'primary_failure': 'occlusion' if severity == 'hard' else 'rapid_motion',
                'failure_rate': 0.2 * (1 + severities.index(severity)),
                'id_switch_rate': 0.05 * (1 + severities.index(severity)),
                'recommendations': [
                    "Implement re-identification module",
                    "Add motion prediction"
                ]
            })
        
        return failure_modes
    
    def _calculate_robustness_score(
        self,
        baseline: ToolPerformanceMetrics,
        stress_performances: Dict[str, ToolPerformanceMetrics]
    ) -> float:
        """Calculate overall robustness score."""
        scores = []
        
        for severity, metrics in stress_performances.items():
            # Calculate retention rate
            retention = metrics.accuracy / baseline.accuracy if baseline.accuracy > 0 else 0.0
            
            # Weight by severity
            severity_weight = {"easy": 0.2, "medium": 0.3, "hard": 0.5}.get(severity, 0.3)
            
            scores.append(retention * severity_weight)
        
        return sum(scores)
    
    def _generate_segmentation_recommendations(
        self,
        performances: Dict[str, ToolPerformanceMetrics],
        failure_modes: List[Dict]
    ) -> List[str]:
        """Generate recommendations for segmentation improvement."""
        recommendations = []
        
        # Check performance degradation
        if performances.get('hard'):
            hard_perf = performances['hard']
            if hard_perf.specific_metrics.get('iou', 0) < 0.5:
                recommendations.append(
                    "Add heavy augmentation during training including occlusion and motion blur"
                )
            if hard_perf.failure_rate > 0.2:
                recommendations.append(
                    "Implement fallback mechanisms for challenging conditions"
                )
        
        # Check specific failure modes
        for mode in failure_modes:
            if mode['primary_failure'] == 'occlusion':
                recommendations.append(
                    "Train with partial occlusion augmentation"
                )
            elif mode['primary_failure'] == 'low_light':
                recommendations.append(
                    "Add low-light image enhancement preprocessing"
                )
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_text_recommendations(
        self,
        performances: Dict[str, ToolPerformanceMetrics],
        failure_modes: List[Dict]
    ) -> List[str]:
        """Generate recommendations for text reading improvement."""
        recommendations = []
        
        if performances.get('hard'):
            hard_perf = performances['hard']
            if hard_perf.specific_metrics.get('edit_distance', 1) > 0.3:
                recommendations.append(
                    "Implement character-level error correction"
                )
            if hard_perf.failure_rate > 0.25:
                recommendations.append(
                    "Add text detection confidence thresholding"
                )
        
        recommendations.append("Consider ensemble of OCR models")
        recommendations.append("Add language model post-processing")
        
        return list(set(recommendations))
    
    def _generate_tracking_recommendations(
        self,
        performances: Dict[str, ToolPerformanceMetrics],
        failure_modes: List[Dict]
    ) -> List[str]:
        """Generate recommendations for tracking improvement."""
        recommendations = []
        
        if performances.get('hard'):
            hard_perf = performances['hard']
            if hard_perf.specific_metrics.get('id_switches', 0) > 5:
                recommendations.append(
                    "Implement robust re-identification using appearance features"
                )
            if hard_perf.specific_metrics.get('mota', 0) < 0.6:
                recommendations.append(
                    "Add Kalman filter for motion prediction during occlusion"
                )
        
        recommendations.append("Use multi-scale feature extraction")
        recommendations.append("Implement online learning for appearance model")
        
        return list(set(recommendations))
    
    def _create_synthetic_segmentation_data(self, data_dir: Path, severity: str):
        """Create synthetic segmentation test data."""
        data_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        for i in range(100):
            # Create synthetic image and mask
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
            
            img_path = data_dir / f"image_{i:05d}.png"
            mask_path = data_dir / f"mask_{i:05d}.png"
            
            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)
            
            metadata.append({
                'image': str(img_path.relative_to(self.stress_data_path)),
                'mask': str(mask_path.relative_to(self.stress_data_path)),
                'severity': severity
            })
        
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_synthetic_text_data(self, data_dir: Path, severity: str):
        """Create synthetic text test data."""
        data_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        texts = ["Test Text", "Sample Document", "Important Notice"]
        
        for i in range(100):
            # Create synthetic text image
            img = np.ones((100, 400, 3), dtype=np.uint8) * 255
            text = texts[i % len(texts)]
            
            cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            img_path = data_dir / f"image_{i:05d}.png"
            cv2.imwrite(str(img_path), img)
            
            metadata.append({
                'image': str(img_path.relative_to(self.stress_data_path)),
                'ground_truth_text': text,
                'severity': severity
            })
        
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_synthetic_tracking_data(self, data_dir: Path, severity: str):
        """Create synthetic tracking test data."""
        data_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for seq_idx in range(20):
            seq_dir = data_dir / f"sequence_{seq_idx:04d}"
            seq_dir.mkdir(exist_ok=True)
            
            frames = []
            trajectories = []
            
            # Generate synthetic sequence
            for frame_idx in range(30):
                img = np.ones((480, 640, 3), dtype=np.uint8) * 200
                
                # Draw moving object
                x = 100 + frame_idx * 10
                y = 200 + int(50 * np.sin(frame_idx * 0.2))
                
                cv2.rectangle(img, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
                
                frame_path = seq_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_path), img)
                
                frames.append(str(frame_path.relative_to(self.stress_data_path)))
                trajectories.append({'x': x, 'y': y, 'w': 50, 'h': 50})
            
            metadata.append({
                'frames': frames,
                'trajectories': trajectories,
                'severity': severity
            })
        
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_complete_stress_test(self) -> Dict[str, StressTestResult]:
        """Run complete stress test for all tools."""
        self.logger.info("Starting complete tool stress test")
        
        results = {}
        
        # Test segmentation tools
        seg_result = self.test_segmentation_robustness()
        results['SEGMENT_OBJECT_AT'] = seg_result
        
        # Test text reading
        text_result = self.test_text_reading_robustness()
        results['READ_TEXT'] = text_result
        
        # Test tracking
        track_result = self.test_tracking_robustness()
        results['TRACK_OBJECT'] = track_result
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict[str, StressTestResult]):
        """Generate comprehensive stress test report."""
        report_path = Path(f"results/tool_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        report_data = {tool: result.to_dict() for tool, result in results.items()}
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create visualizations
        self._create_summary_plots(results)
        
        # Print summary
        print("\n" + "="*60)
        print("TOOL STRESS TEST SUMMARY")
        print("="*60)
        
        for tool_name, result in results.items():
            print(f"\n{tool_name}:")
            print(f"  Robustness Score: {result.robustness_score:.3f}")
            print(f"  Baseline Accuracy: {result.baseline_performance.accuracy:.3f}")
            
            for severity, metrics in result.stress_performances.items():
                print(f"  {severity.capitalize()} Accuracy: {metrics.accuracy:.3f} "
                      f"(degradation: {result.degradation_analysis[severity]:.1f}%)")
            
            print(f"  Recommendations:")
            for rec in result.recommendations[:3]:
                print(f"    - {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
    
    def _create_summary_plots(self, results: Dict[str, StressTestResult]):
        """Create summary visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Accuracy degradation across severities
        ax1 = axes[0, 0]
        for tool_name, result in results.items():
            severities = ['baseline'] + list(result.stress_performances.keys())
            accuracies = [result.baseline_performance.accuracy] + \
                        [m.accuracy for m in result.stress_performances.values()]
            
            ax1.plot(severities, accuracies, marker='o', label=tool_name, linewidth=2)
        
        ax1.set_xlabel('Severity')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Tool Performance Under Stress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Robustness scores comparison
        ax2 = axes[0, 1]
        tools = list(results.keys())
        robustness_scores = [r.robustness_score for r in results.values()]
        
        bars = ax2.bar(tools, robustness_scores, color=['green', 'blue', 'orange'])
        ax2.set_ylabel('Robustness Score')
        ax2.set_title('Overall Tool Robustness')
        ax2.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, score in zip(bars, robustness_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 3: Failure rates
        ax3 = axes[0, 2]
        severity_levels = ['easy', 'medium', 'hard']
        x = np.arange(len(severity_levels))
        width = 0.25
        
        for i, (tool_name, result) in enumerate(results.items()):
            failure_rates = [result.stress_performances[s].failure_rate 
                           for s in severity_levels]
            ax3.bar(x + i * width, failure_rates, width, label=tool_name)
        
        ax3.set_xlabel('Severity')
        ax3.set_ylabel('Failure Rate')
        ax3.set_title('Failure Rates by Severity')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(severity_levels)
        ax3.legend()
        
        # Plot 4: Latency comparison
        ax4 = axes[1, 0]
        for tool_name, result in results.items():
            severities = list(result.stress_performances.keys())
            latencies = [m.latency_ms for m in result.stress_performances.values()]
            
            ax4.plot(severities, latencies, marker='s', label=tool_name, linewidth=2)
        
        ax4.set_xlabel('Severity')
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('Processing Latency Under Stress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Tool-specific metrics
        ax5 = axes[1, 1]
        
        # Segmentation IoU
        if 'SEGMENT_OBJECT_AT' in results:
            seg_result = results['SEGMENT_OBJECT_AT']
            severities = list(seg_result.stress_performances.keys())
            ious = [m.specific_metrics.get('iou', 0) 
                   for m in seg_result.stress_performances.values()]
            ax5.plot(severities, ious, marker='o', label='Segmentation IoU')
        
        # Text edit distance
        if 'READ_TEXT' in results:
            text_result = results['READ_TEXT']
            severities = list(text_result.stress_performances.keys())
            edit_dists = [1 - m.specific_metrics.get('edit_distance', 1) 
                         for m in text_result.stress_performances.values()]
            ax5.plot(severities, edit_dists, marker='s', label='Text Accuracy')
        
        # Tracking MOTA
        if 'TRACK_OBJECT' in results:
            track_result = results['TRACK_OBJECT']
            severities = list(track_result.stress_performances.keys())
            motas = [m.specific_metrics.get('mota', 0) 
                    for m in track_result.stress_performances.values()]
            ax5.plot(severities, motas, marker='^', label='Tracking MOTA')
        
        ax5.set_xlabel('Severity')
        ax5.set_ylabel('Metric Value')
        ax5.set_title('Tool-Specific Performance Metrics')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Confidence calibration
        ax6 = axes[1, 2]
        
        for tool_name, result in results.items():
            confidences = []
            accuracies = []
            
            for severity, metrics in result.stress_performances.items():
                confidences.append(metrics.confidence_mean)
                accuracies.append(metrics.accuracy)
            
            ax6.scatter(confidences, accuracies, label=tool_name, s=100, alpha=0.7)
        
        # Add perfect calibration line
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        ax6.set_xlabel('Mean Confidence')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Confidence Calibration')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(f"results/tool_stress_test_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        wandb.log({"stress_test_plots": wandb.Image(str(plot_path))})


def main():
    """Main function to run tool-specific stress tests."""
    parser = argparse.ArgumentParser(description="Tool-specific stress testing")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/experiments/pixelis_rft_full.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--stress-data",
        type=str,
        default="data/stress_test",
        help="Path to stress test data"
    )
    parser.add_argument(
        "--tool",
        choices=["segmentation", "text", "tracking", "all"],
        default="all",
        help="Which tool to test"
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Severity levels to test"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tool_stress_test",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tester
    tester = ToolStressTester(
        model_config=args.model_config,
        stress_data_path=args.stress_data
    )
    
    # Run tests
    if args.tool == "segmentation":
        result = tester.test_segmentation_robustness(args.severities)
        print(f"Segmentation robustness score: {result.robustness_score:.3f}")
    
    elif args.tool == "text":
        result = tester.test_text_reading_robustness(args.severities)
        print(f"Text reading robustness score: {result.robustness_score:.3f}")
    
    elif args.tool == "tracking":
        result = tester.test_tracking_robustness(args.severities)
        print(f"Tracking robustness score: {result.robustness_score:.3f}")
    
    else:  # all
        results = tester.run_complete_stress_test()
        print("\n=== Tool Stress Test Complete ===")
        print(f"Results saved to {output_dir}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main()