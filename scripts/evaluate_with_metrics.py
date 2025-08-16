#!/usr/bin/env python3
"""
Enhanced evaluation script with comprehensive tool-specific metrics.
Evaluates models on standard and custom benchmarks with detailed metric tracking.
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment
import editdistance

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.reproducibility import (
    ArtifactManager,
    ArtifactType,
    ExperimentContext,
    EnvironmentCaptureLevel,
    track_artifacts,
)
from core.utils.logging_utils import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation evaluation."""
    iou: float
    boundary_f1: float
    pixel_accuracy: float
    dice_coefficient: float
    hausdorff_distance: float


@dataclass
class OCRMetrics:
    """Metrics for OCR evaluation."""
    character_error_rate: float
    word_error_rate: float
    edit_distance: int
    accuracy: float
    case_sensitive_accuracy: float


@dataclass
class TrackingMetrics:
    """Metrics for object tracking evaluation."""
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    id_switches: int
    false_positives: int
    false_negatives: int
    fragmentations: int
    recall: float
    precision: float


@dataclass
class PropertyMetrics:
    """Metrics for property extraction evaluation."""
    accuracy: float
    attribute_precision: float
    attribute_recall: float
    attribute_f1: float
    completeness: float


@dataclass
class ReasoningMetrics:
    """Metrics for reasoning quality evaluation."""
    coherence_score: float
    exploration_efficiency: float
    tool_usage_efficiency: float
    trajectory_length: float
    success_rate: float
    self_correction_rate: float


class ToolSpecificEvaluator:
    """Evaluator for tool-specific metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with configuration."""
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== Segmentation Metrics ==========
    
    def calculate_iou(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> float:
        """Calculate Intersection over Union."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def calculate_boundary_f1(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        threshold: int = 3
    ) -> float:
        """Calculate boundary F1 score."""
        from scipy import ndimage
        
        # Extract boundaries
        pred_boundary = ndimage.binary_erosion(pred_mask) ^ pred_mask
        gt_boundary = ndimage.binary_erosion(gt_mask) ^ gt_mask
        
        # Dilate boundaries for threshold matching
        pred_dilated = ndimage.binary_dilation(pred_boundary, iterations=threshold)
        gt_dilated = ndimage.binary_dilation(gt_boundary, iterations=threshold)
        
        # Calculate precision and recall
        tp = np.logical_and(pred_boundary, gt_dilated).sum()
        fp = np.logical_and(pred_boundary, ~gt_dilated).sum()
        fn = np.logical_and(gt_boundary, ~pred_dilated).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def calculate_dice_coefficient(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> float:
        """Calculate Dice coefficient."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum())
        return dice
    
    def evaluate_segmentation(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> SegmentationMetrics:
        """Evaluate segmentation performance."""
        pred_mask = np.array(prediction.get("mask", []))
        gt_mask = np.array(ground_truth.get("mask", []))
        
        # Ensure masks have same shape
        if pred_mask.shape != gt_mask.shape:
            # Resize to match ground truth
            pred_mask = self._resize_mask(pred_mask, gt_mask.shape)
        
        # Calculate metrics
        iou = self.calculate_iou(pred_mask, gt_mask)
        boundary_f1 = self.calculate_boundary_f1(pred_mask, gt_mask)
        pixel_accuracy = np.mean(pred_mask == gt_mask)
        dice = self.calculate_dice_coefficient(pred_mask, gt_mask)
        
        # Hausdorff distance (for boundary quality)
        pred_points = np.argwhere(pred_mask)
        gt_points = np.argwhere(gt_mask)
        
        if len(pred_points) > 0 and len(gt_points) > 0:
            hausdorff = max(
                directed_hausdorff(pred_points, gt_points)[0],
                directed_hausdorff(gt_points, pred_points)[0]
            )
        else:
            hausdorff = float('inf')
        
        return SegmentationMetrics(
            iou=iou,
            boundary_f1=boundary_f1,
            pixel_accuracy=pixel_accuracy,
            dice_coefficient=dice,
            hausdorff_distance=hausdorff
        )
    
    def _resize_mask(
        self,
        mask: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Resize mask to target shape."""
        from scipy import ndimage
        zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
        resized = ndimage.zoom(mask.astype(float), zoom_factors, order=1)
        return (resized > 0.5).astype(bool)
    
    # ========== OCR Metrics ==========
    
    def calculate_character_error_rate(
        self,
        pred_text: str,
        gt_text: str
    ) -> float:
        """Calculate Character Error Rate (CER)."""
        if len(gt_text) == 0:
            return 0.0 if len(pred_text) == 0 else 1.0
        
        distance = editdistance.eval(pred_text, gt_text)
        return distance / len(gt_text)
    
    def calculate_word_error_rate(
        self,
        pred_text: str,
        gt_text: str
    ) -> float:
        """Calculate Word Error Rate (WER)."""
        pred_words = pred_text.split()
        gt_words = gt_text.split()
        
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        
        distance = editdistance.eval(pred_words, gt_words)
        return distance / len(gt_words)
    
    def evaluate_ocr(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> OCRMetrics:
        """Evaluate OCR performance."""
        pred_text = prediction.get("text", "")
        gt_text = ground_truth.get("text", "")
        
        # Calculate metrics
        cer = self.calculate_character_error_rate(pred_text, gt_text)
        wer = self.calculate_word_error_rate(pred_text, gt_text)
        edit_dist = editdistance.eval(pred_text, gt_text)
        
        # Accuracy (exact match)
        accuracy = 1.0 if pred_text == gt_text else 0.0
        
        # Case-sensitive accuracy
        case_sensitive_acc = 1.0 if pred_text == gt_text else 0.0
        if pred_text.lower() == gt_text.lower() and case_sensitive_acc == 0:
            case_sensitive_acc = 0.5  # Partial credit for case mismatch
        
        return OCRMetrics(
            character_error_rate=cer,
            word_error_rate=wer,
            edit_distance=edit_dist,
            accuracy=accuracy,
            case_sensitive_accuracy=case_sensitive_acc
        )
    
    # ========== Tracking Metrics ==========
    
    def calculate_mota(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Tuple[float, Dict[str, int]]:
        """Calculate Multiple Object Tracking Accuracy."""
        total_gt = 0
        false_positives = 0
        false_negatives = 0
        id_switches = 0
        
        # Track ID mappings across frames
        id_mapping = {}
        
        for frame_idx, (pred_frame, gt_frame) in enumerate(
            zip(predictions, ground_truths)
        ):
            pred_boxes = pred_frame.get("boxes", [])
            gt_boxes = gt_frame.get("boxes", [])
            pred_ids = pred_frame.get("ids", [])
            gt_ids = gt_frame.get("ids", [])
            
            total_gt += len(gt_boxes)
            
            if len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            
            if len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue
            
            # Hungarian matching
            cost_matrix = self._compute_cost_matrix(pred_boxes, gt_boxes)
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
            
            # Count matches and misses
            matched_preds = set(pred_indices)
            matched_gts = set(gt_indices)
            
            false_positives += len(pred_boxes) - len(matched_preds)
            false_negatives += len(gt_boxes) - len(matched_gts)
            
            # Check for ID switches
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                pred_id = pred_ids[pred_idx]
                gt_id = gt_ids[gt_idx]
                
                if gt_id in id_mapping:
                    if id_mapping[gt_id] != pred_id:
                        id_switches += 1
                        id_mapping[gt_id] = pred_id
                else:
                    id_mapping[gt_id] = pred_id
        
        # Calculate MOTA
        if total_gt == 0:
            mota = 1.0 if false_positives == 0 else 0.0
        else:
            mota = 1 - (false_positives + false_negatives + id_switches) / total_gt
        
        components = {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_switches": id_switches,
            "total_gt": total_gt
        }
        
        return mota, components
    
    def calculate_motp(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> float:
        """Calculate Multiple Object Tracking Precision."""
        total_distance = 0
        total_matches = 0
        
        for pred_frame, gt_frame in zip(predictions, ground_truths):
            pred_boxes = pred_frame.get("boxes", [])
            gt_boxes = gt_frame.get("boxes", [])
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Hungarian matching
            cost_matrix = self._compute_cost_matrix(pred_boxes, gt_boxes)
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
            
            # Sum IoU for matches
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                iou = self._compute_box_iou(
                    pred_boxes[pred_idx],
                    gt_boxes[gt_idx]
                )
                total_distance += iou
                total_matches += 1
        
        if total_matches == 0:
            return 0.0
        
        return total_distance / total_matches
    
    def _compute_cost_matrix(
        self,
        pred_boxes: List,
        gt_boxes: List
    ) -> np.ndarray:
        """Compute cost matrix for Hungarian matching."""
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)
        cost_matrix = np.zeros((n_pred, n_gt))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                # Use 1 - IoU as cost
                iou = self._compute_box_iou(pred_box, gt_box)
                cost_matrix[i, j] = 1 - iou
        
        return cost_matrix
    
    def _compute_box_iou(
        self,
        box1: Union[List, np.ndarray],
        box2: Union[List, np.ndarray]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Compute intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Compute union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_tracking(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> TrackingMetrics:
        """Evaluate tracking performance."""
        # Calculate MOTA and components
        mota, components = self.calculate_mota(predictions, ground_truths)
        
        # Calculate MOTP
        motp = self.calculate_motp(predictions, ground_truths)
        
        # Calculate precision and recall
        tp = components["total_gt"] - components["false_negatives"]
        fp = components["false_positives"]
        fn = components["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Count fragmentations (simplified)
        fragmentations = components["id_switches"] // 2
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            id_switches=components["id_switches"],
            false_positives=components["false_positives"],
            false_negatives=components["false_negatives"],
            fragmentations=fragmentations,
            recall=recall,
            precision=precision
        )
    
    # ========== Property Extraction Metrics ==========
    
    def evaluate_properties(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> PropertyMetrics:
        """Evaluate property extraction performance."""
        pred_props = prediction.get("properties", {})
        gt_props = ground_truth.get("properties", {})
        
        # Get all unique property keys
        all_keys = set(pred_props.keys()) | set(gt_props.keys())
        
        if len(all_keys) == 0:
            return PropertyMetrics(
                accuracy=1.0,
                attribute_precision=1.0,
                attribute_recall=1.0,
                attribute_f1=1.0,
                completeness=1.0
            )
        
        # Calculate per-attribute metrics
        correct = 0
        pred_count = 0
        gt_count = 0
        
        for key in all_keys:
            if key in pred_props:
                pred_count += 1
            if key in gt_props:
                gt_count += 1
            
            if key in pred_props and key in gt_props:
                # Check if values match (with some tolerance for numerical values)
                if self._properties_match(pred_props[key], gt_props[key]):
                    correct += 1
        
        # Calculate metrics
        accuracy = correct / len(all_keys)
        precision = correct / pred_count if pred_count > 0 else 0
        recall = correct / gt_count if gt_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        completeness = len(pred_props) / len(gt_props) if len(gt_props) > 0 else 1.0
        
        return PropertyMetrics(
            accuracy=accuracy,
            attribute_precision=precision,
            attribute_recall=recall,
            attribute_f1=f1,
            completeness=completeness
        )
    
    def _properties_match(
        self,
        pred_value: Any,
        gt_value: Any,
        tolerance: float = 0.1
    ) -> bool:
        """Check if two property values match."""
        if isinstance(pred_value, (int, float)) and isinstance(gt_value, (int, float)):
            # Numerical comparison with tolerance
            return abs(pred_value - gt_value) / (abs(gt_value) + 1e-6) < tolerance
        elif isinstance(pred_value, str) and isinstance(gt_value, str):
            # String comparison (case-insensitive)
            return pred_value.lower().strip() == gt_value.lower().strip()
        else:
            # Direct comparison
            return pred_value == gt_value
    
    # ========== Reasoning Quality Metrics ==========
    
    def calculate_coherence_score(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> float:
        """Calculate trajectory coherence score."""
        if len(trajectory) < 2:
            return 1.0
        
        # Get embeddings for each step (simplified version)
        embeddings = []
        for step in trajectory:
            # Create a simple embedding based on action and parameters
            action = step.get("action", "")
            params = str(step.get("parameters", {}))
            text = f"{action} {params}"
            
            # Simple hash-based embedding (in practice, use proper embeddings)
            embedding = np.array([hash(text + str(i)) % 100 for i in range(10)])
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Check for repetitions
        repetition_penalty = 0
        for i in range(1, len(trajectory)):
            if trajectory[i].get("action") == trajectory[i-1].get("action"):
                if trajectory[i].get("parameters") == trajectory[i-1].get("parameters"):
                    repetition_penalty += 0.1
        
        coherence = np.mean(similarities) - repetition_penalty
        return max(0, min(1, coherence))
    
    def calculate_exploration_efficiency(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> float:
        """Calculate exploration efficiency."""
        if len(trajectory) == 0:
            return 0.0
        
        # Count unique states
        unique_states = set()
        for step in trajectory:
            state = (
                step.get("action", ""),
                str(step.get("parameters", {}))
            )
            unique_states.add(state)
        
        efficiency = len(unique_states) / len(trajectory)
        return efficiency
    
    def calculate_tool_usage_efficiency(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> float:
        """Calculate tool usage efficiency."""
        if len(trajectory) == 0:
            return 0.0
        
        successful_tools = 0
        total_tools = 0
        
        for step in trajectory:
            if "action" in step and step["action"] != "THINK":
                total_tools += 1
                if step.get("success", True):
                    successful_tools += 1
        
        if total_tools == 0:
            return 1.0
        
        return successful_tools / total_tools
    
    def evaluate_reasoning(
        self,
        trajectory: List[Dict[str, Any]],
        success: bool = False
    ) -> ReasoningMetrics:
        """Evaluate reasoning quality."""
        coherence = self.calculate_coherence_score(trajectory)
        exploration = self.calculate_exploration_efficiency(trajectory)
        tool_efficiency = self.calculate_tool_usage_efficiency(trajectory)
        
        # Calculate self-correction rate
        corrections = 0
        for i, step in enumerate(trajectory):
            if i > 0 and "correction" in step.get("action", "").lower():
                corrections += 1
        
        self_correction_rate = corrections / len(trajectory) if len(trajectory) > 0 else 0
        
        return ReasoningMetrics(
            coherence_score=coherence,
            exploration_efficiency=exploration,
            tool_usage_efficiency=tool_efficiency,
            trajectory_length=float(len(trajectory)),
            success_rate=1.0 if success else 0.0,
            self_correction_rate=self_correction_rate
        )


class ComprehensiveEvaluator:
    """Main evaluator combining all metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize comprehensive evaluator."""
        self.config = config or {}
        self.tool_evaluator = ToolSpecificEvaluator(config)
        self.results = defaultdict(list)
    
    def evaluate_sample(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
        category: str
    ) -> Dict[str, Any]:
        """Evaluate a single sample based on its category."""
        metrics = {}
        
        # Get trajectory if available
        trajectory = prediction.get("trajectory", [])
        
        # Evaluate based on category
        if category == "precise_segmentation":
            seg_metrics = self.tool_evaluator.evaluate_segmentation(
                prediction, ground_truth
            )
            metrics.update(asdict(seg_metrics))
            
        elif category == "text_extraction":
            ocr_metrics = self.tool_evaluator.evaluate_ocr(
                prediction, ground_truth
            )
            metrics.update(asdict(ocr_metrics))
            
        elif category == "object_tracking":
            # Tracking requires frame sequences
            pred_frames = prediction.get("frames", [])
            gt_frames = ground_truth.get("frames", [])
            
            if pred_frames and gt_frames:
                track_metrics = self.tool_evaluator.evaluate_tracking(
                    pred_frames, gt_frames
                )
                metrics.update(asdict(track_metrics))
            
        elif category == "property_analysis":
            prop_metrics = self.tool_evaluator.evaluate_properties(
                prediction, ground_truth
            )
            metrics.update(asdict(prop_metrics))
            
        elif category == "complex_reasoning":
            # Evaluate multiple aspects
            if "mask" in ground_truth:
                seg_metrics = self.tool_evaluator.evaluate_segmentation(
                    prediction, ground_truth
                )
                metrics.update({f"seg_{k}": v for k, v in asdict(seg_metrics).items()})
            
            if "text" in ground_truth:
                ocr_metrics = self.tool_evaluator.evaluate_ocr(
                    prediction, ground_truth
                )
                metrics.update({f"ocr_{k}": v for k, v in asdict(ocr_metrics).items()})
            
            if "properties" in ground_truth:
                prop_metrics = self.tool_evaluator.evaluate_properties(
                    prediction, ground_truth
                )
                metrics.update({f"prop_{k}": v for k, v in asdict(prop_metrics).items()})
        
        # Always evaluate reasoning quality if trajectory available
        if trajectory:
            success = prediction.get("success", False)
            reason_metrics = self.tool_evaluator.evaluate_reasoning(
                trajectory, success
            )
            metrics.update({f"reason_{k}": v for k, v in asdict(reason_metrics).items()})
        
        return metrics
    
    def evaluate_benchmark(
        self,
        predictions: List[Dict[str, Any]],
        benchmark_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate on complete benchmark."""
        logger.info("Evaluating on custom capabilities benchmark")
        
        category_results = defaultdict(lambda: defaultdict(list))
        overall_results = defaultdict(list)
        
        # Process each category
        for category, category_data in benchmark_data["categories"].items():
            logger.info(f"Evaluating {category}...")
            
            samples = category_data["samples"]
            category_predictions = predictions.get(category, [])
            
            if len(category_predictions) != len(samples):
                logger.warning(f"Prediction count mismatch for {category}: "
                              f"{len(category_predictions)} vs {len(samples)}")
            
            # Evaluate each sample
            for i, (sample, pred) in enumerate(zip(samples, category_predictions)):
                gt = sample.get("ground_truth", {})
                
                metrics = self.evaluate_sample(pred, gt, category)
                
                # Store results
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        category_results[category][metric_name].append(value)
                        overall_results[metric_name].append(value)
        
        # Aggregate results
        aggregated = {
            "overall": {},
            "by_category": {}
        }
        
        # Overall metrics
        for metric_name, values in overall_results.items():
            aggregated["overall"][metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
        
        # Category-specific metrics
        for category, metrics in category_results.items():
            aggregated["by_category"][category] = {}
            for metric_name, values in metrics.items():
                aggregated["by_category"][category][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        
        # Add success criteria evaluation
        aggregated["success_criteria"] = self._evaluate_success_criteria(
            aggregated, benchmark_data
        )
        
        return aggregated
    
    def _evaluate_success_criteria(
        self,
        results: Dict[str, Any],
        benchmark_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check if success criteria are met."""
        criteria_met = {}
        
        # Check each category's criteria
        for category, config in benchmark_data["metadata"]["task_categories"].items():
            if category not in results["by_category"]:
                continue
            
            category_results = results["by_category"][category]
            
            # Check primary metric threshold
            if category == "precise_segmentation":
                met = category_results.get("iou", {}).get("mean", 0) > 0.7
                criteria_met[f"{category}_iou"] = met
                
            elif category == "text_extraction":
                met = category_results.get("character_error_rate", {}).get("mean", 1) < 0.2
                criteria_met[f"{category}_cer"] = met
                
            elif category == "object_tracking":
                met = category_results.get("mota", {}).get("mean", 0) > 0.6
                criteria_met[f"{category}_mota"] = met
                
            elif category == "property_analysis":
                met = category_results.get("accuracy", {}).get("mean", 0) > 0.8
                criteria_met[f"{category}_accuracy"] = met
        
        # Overall success
        criteria_met["overall"] = all(criteria_met.values()) if criteria_met else False
        
        return criteria_met


def main():
    """Main entry point for enhanced evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate models with comprehensive metrics"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions file"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Output path for results"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        help="Evaluate specific category only"
    )
    
    args = parser.parse_args()
    
    # Load data
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    with open(args.benchmark, 'r') as f:
        benchmark_data = json.load(f)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Evaluate
    results = evaluator.evaluate_benchmark(predictions, benchmark_data)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)
    
    # Overall metrics
    logger.info("\nOverall Metrics:")
    for metric, stats in results["overall"].items():
        if isinstance(stats, dict) and "mean" in stats:
            logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Category results
    logger.info("\nCategory Results:")
    for category, metrics in results["by_category"].items():
        logger.info(f"\n  {category}:")
        for metric, stats in metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                logger.info(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Success criteria
    logger.info("\nSuccess Criteria:")
    for criterion, met in results["success_criteria"].items():
        status = "✓" if met else "✗"
        logger.info(f"  {criterion}: {status}")
    
    logger.info(f"\n✓ Results saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())