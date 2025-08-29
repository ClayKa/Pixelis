"""
Comprehensive Validation and Quality Scoring System for CoTA Data
Implements multi-dimensional quality assessment with SOLID principles
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


# ======================== QUALITY DIMENSIONS ========================

class QualityDimension(Enum):
    """Quality dimensions for trajectory evaluation"""
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    DIVERSITY = "diversity"
    COMPLEXITY = "complexity"
    CONSISTENCY = "consistency"


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # All checks must pass
    STANDARD = "standard"  # Most checks must pass
    LENIENT = "lenient"   # Basic checks must pass


class IssueType(Enum):
    """Types of validation issues"""
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"
    LOGICAL_ERROR = "logical_error"
    INCONSISTENCY = "inconsistency"
    QUALITY_ISSUE = "quality_issue"
    DUPLICATE = "duplicate"


# ======================== VALIDATION RESULTS ========================

@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    issue_type: IssueType
    severity: str  # "error", "warning", "info"
    field: str
    message: str
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion
        }


@dataclass
class ValidationResult:
    """Complete validation result for a trajectory"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    quality_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add an issue to the result"""
        if issue.severity == "error":
            self.issues.append(issue)
            self.is_valid = False
        else:
            self.warnings.append(issue)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.is_valid,
            "errors": [i.to_dict() for i in self.issues],
            "warnings": [w.to_dict() for w in self.warnings],
            "quality_scores": {dim.value: score for dim, score in self.quality_scores.items()},
            "overall_score": self.overall_score,
            "metadata": self.metadata
        }


# ======================== ABSTRACT VALIDATORS ========================

class Validator(ABC):
    """Abstract base class for validators"""
    
    @abstractmethod
    def validate(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """Validate a trajectory"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get validator name"""
        pass


class QualityScorer(ABC):
    """Abstract base class for quality scorers"""
    
    @abstractmethod
    def score(self, trajectory: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Score trajectory on multiple dimensions"""
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[QualityDimension, float]:
        """Get dimension weights for overall score"""
        pass


# ======================== CONCRETE VALIDATORS ========================

class StructuralValidator(Validator):
    """Validates the structural integrity of trajectories"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.required_fields = {
            "id", "task", "chain_of_thought", "visual_operations", "answer"
        }
        self.optional_fields = {
            "type", "difficulty", "quality_score", "metadata", "provenance"
        }
    
    def validate(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """Validate trajectory structure"""
        result = ValidationResult(is_valid=True)
        
        # Check required fields
        for field in self.required_fields:
            if field not in trajectory:
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.MISSING_FIELD,
                    severity="error",
                    field=field,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Add '{field}' field to trajectory"
                ))
            elif trajectory[field] is None:
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity="error",
                    field=field,
                    message=f"Required field '{field}' is null",
                    suggestion=f"Provide non-null value for '{field}'"
                ))
        
        # Validate chain of thought structure
        if "chain_of_thought" in trajectory:
            cot = trajectory["chain_of_thought"]
            if not isinstance(cot, list):
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity="error",
                    field="chain_of_thought",
                    message="Chain of thought must be a list",
                    suggestion="Convert to list of thought steps"
                ))
            elif len(cot) == 0:
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity="warning" if self.level == ValidationLevel.LENIENT else "error",
                    field="chain_of_thought",
                    message="Chain of thought is empty",
                    suggestion="Add reasoning steps"
                ))
            else:
                # Validate each thought step
                for i, thought in enumerate(cot):
                    if not isinstance(thought, dict):
                        result.add_issue(ValidationIssue(
                            issue_type=IssueType.INVALID_FORMAT,
                            severity="error",
                            field=f"chain_of_thought[{i}]",
                            message=f"Thought step {i} is not a dictionary",
                            suggestion="Convert to dict with 'thought' and 'type' fields"
                        ))
                    elif "thought" not in thought:
                        result.add_issue(ValidationIssue(
                            issue_type=IssueType.MISSING_FIELD,
                            severity="warning",
                            field=f"chain_of_thought[{i}]",
                            message=f"Thought step {i} missing 'thought' field",
                            suggestion="Add thought content"
                        ))
        
        # Validate visual operations
        if "visual_operations" in trajectory:
            ops = trajectory["visual_operations"]
            if not isinstance(ops, list):
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity="error",
                    field="visual_operations",
                    message="Visual operations must be a list",
                    suggestion="Convert to list of operations"
                ))
            else:
                valid_ops = {
                    "ZOOM_IN", "SELECT_FRAME", "SEGMENT_OBJECT_AT",
                    "GET_PROPERTIES", "READ_TEXT", "TRACK_OBJECT"
                }
                for i, op in enumerate(ops):
                    if not isinstance(op, dict):
                        result.add_issue(ValidationIssue(
                            issue_type=IssueType.INVALID_FORMAT,
                            severity="error",
                            field=f"visual_operations[{i}]",
                            message=f"Operation {i} is not a dictionary",
                            suggestion="Convert to dict with 'operation' and 'params'"
                        ))
                    elif "operation" not in op:
                        result.add_issue(ValidationIssue(
                            issue_type=IssueType.MISSING_FIELD,
                            severity="error",
                            field=f"visual_operations[{i}]",
                            message=f"Operation {i} missing 'operation' field",
                            suggestion="Add operation name"
                        ))
                    elif op["operation"] not in valid_ops:
                        result.add_issue(ValidationIssue(
                            issue_type=IssueType.INVALID_FORMAT,
                            severity="warning",
                            field=f"visual_operations[{i}]",
                            message=f"Unknown operation: {op['operation']}",
                            suggestion=f"Use one of: {valid_ops}"
                        ))
        
        # Check ID uniqueness format
        if "id" in trajectory:
            traj_id = trajectory["id"]
            if not isinstance(traj_id, str) or len(traj_id) < 8:
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity="warning",
                    field="id",
                    message="ID should be a string of at least 8 characters",
                    suggestion="Use a proper unique identifier"
                ))
        
        return result
    
    def get_name(self) -> str:
        return "StructuralValidator"


class LogicalValidator(Validator):
    """Validates logical consistency in trajectories"""
    
    def __init__(self):
        self.operation_dependencies = {
            "GET_PROPERTIES": ["SEGMENT_OBJECT_AT"],  # Needs segmentation first
            "TRACK_OBJECT": ["SELECT_FRAME"],  # Needs frame selection
        }
    
    def validate(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """Validate logical consistency"""
        result = ValidationResult(is_valid=True)
        
        # Check operation dependencies
        if "visual_operations" in trajectory:
            ops = trajectory.get("visual_operations", [])
            executed_ops = set()
            
            for i, op in enumerate(ops):
                if isinstance(op, dict) and "operation" in op:
                    op_name = op["operation"]
                    
                    # Check dependencies
                    if op_name in self.operation_dependencies:
                        required = self.operation_dependencies[op_name]
                        missing = [r for r in required if r not in executed_ops]
                        if missing:
                            result.add_issue(ValidationIssue(
                                issue_type=IssueType.LOGICAL_ERROR,
                                severity="warning",
                                field=f"visual_operations[{i}]",
                                message=f"{op_name} requires {missing} to be executed first",
                                suggestion=f"Add {missing} operation before {op_name}"
                            ))
                    
                    executed_ops.add(op_name)
        
        # Check thought-action alignment
        thoughts = trajectory.get("chain_of_thought", [])
        operations = trajectory.get("visual_operations", [])
        
        if len(thoughts) > 0 and len(operations) > 0:
            # Basic check: should have thoughts between operations
            if len(thoughts) < len(operations):
                result.add_issue(ValidationIssue(
                    issue_type=IssueType.INCONSISTENCY,
                    severity="info",
                    field="chain_of_thought",
                    message="Fewer thoughts than operations",
                    suggestion="Add reasoning for each operation"
                ))
        
        # Check for redundant operations
        if operations:
            op_sequence = [op.get("operation") for op in operations if isinstance(op, dict)]
            for i in range(1, len(op_sequence)):
                if op_sequence[i] == op_sequence[i-1]:
                    result.add_issue(ValidationIssue(
                        issue_type=IssueType.LOGICAL_ERROR,
                        severity="info",
                        field=f"visual_operations[{i}]",
                        message=f"Redundant consecutive {op_sequence[i]} operations",
                        suggestion="Consider combining or removing redundant operations"
                    ))
        
        # Check answer consistency
        answer = trajectory.get("answer", "")
        if answer and len(answer) < 5:
            result.add_issue(ValidationIssue(
                issue_type=IssueType.QUALITY_ISSUE,
                severity="warning",
                field="answer",
                message="Answer is too short",
                suggestion="Provide more detailed answer"
            ))
        
        return result
    
    def get_name(self) -> str:
        return "LogicalValidator"


class DuplicateDetector(Validator):
    """Detects duplicate or near-duplicate trajectories"""
    
    def __init__(self):
        self.seen_hashes = set()
        self.similarity_threshold = 0.95
    
    def _compute_hash(self, trajectory: Dict[str, Any]) -> str:
        """Compute hash for trajectory"""
        # Create normalized representation
        normalized = {
            "task": trajectory.get("task", ""),
            "operations": [
                op.get("operation", "")
                for op in trajectory.get("visual_operations", [])
                if isinstance(op, dict)
            ],
            "answer": trajectory.get("answer", "")
        }
        
        # Compute hash
        content = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _compute_similarity(self, traj1: Dict, traj2: Dict) -> float:
        """Compute similarity between trajectories"""
        # Simple similarity based on operations and answer
        ops1 = set(op.get("operation", "") for op in traj1.get("visual_operations", []))
        ops2 = set(op.get("operation", "") for op in traj2.get("visual_operations", []))
        
        if not ops1 and not ops2:
            return 1.0
        if not ops1 or not ops2:
            return 0.0
        
        intersection = len(ops1 & ops2)
        union = len(ops1 | ops2)
        
        return intersection / union if union > 0 else 0.0
    
    def validate(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """Check for duplicates"""
        result = ValidationResult(is_valid=True)
        
        # Compute hash
        traj_hash = self._compute_hash(trajectory)
        
        if traj_hash in self.seen_hashes:
            result.add_issue(ValidationIssue(
                issue_type=IssueType.DUPLICATE,
                severity="warning",
                field="trajectory",
                message="Potential duplicate trajectory detected",
                suggestion="Ensure trajectory is unique or add variation"
            ))
        
        self.seen_hashes.add(traj_hash)
        
        return result
    
    def get_name(self) -> str:
        return "DuplicateDetector"


# ======================== QUALITY SCORERS ========================

class ComprehensiveQualityScorer(QualityScorer):
    """Comprehensive quality scoring across multiple dimensions"""
    
    def __init__(self):
        self.weights = {
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.COHERENCE: 0.25,
            QualityDimension.CORRECTNESS: 0.25,
            QualityDimension.EFFICIENCY: 0.15,
            QualityDimension.COMPLEXITY: 0.10,
            QualityDimension.CONSISTENCY: 0.05
        }
    
    def score(self, trajectory: Dict[str, Any]) -> Dict[QualityDimension, float]:
        """Score trajectory on all dimensions"""
        scores = {}
        
        scores[QualityDimension.COMPLETENESS] = self._score_completeness(trajectory)
        scores[QualityDimension.COHERENCE] = self._score_coherence(trajectory)
        scores[QualityDimension.CORRECTNESS] = self._score_correctness(trajectory)
        scores[QualityDimension.EFFICIENCY] = self._score_efficiency(trajectory)
        scores[QualityDimension.COMPLEXITY] = self._score_complexity(trajectory)
        scores[QualityDimension.CONSISTENCY] = self._score_consistency(trajectory)
        
        return scores
    
    def get_weights(self) -> Dict[QualityDimension, float]:
        """Get scoring weights"""
        return self.weights
    
    def _score_completeness(self, trajectory: Dict[str, Any]) -> float:
        """Score based on completeness of information"""
        score = 1.0
        penalties = 0.0
        
        # Check required fields
        required_fields = ["id", "task", "chain_of_thought", "visual_operations", "answer"]
        for field in required_fields:
            if field not in trajectory or not trajectory[field]:
                penalties += 0.2
        
        # Check thought steps
        thoughts = trajectory.get("chain_of_thought", [])
        if len(thoughts) < 2:
            penalties += 0.1
        
        # Check operations
        operations = trajectory.get("visual_operations", [])
        if len(operations) < 1:
            penalties += 0.1
        
        # Check metadata
        if "metadata" not in trajectory:
            penalties += 0.05
        
        return max(0.0, score - penalties)
    
    def _score_coherence(self, trajectory: Dict[str, Any]) -> float:
        """Score logical flow and coherence"""
        score = 1.0
        
        thoughts = trajectory.get("chain_of_thought", [])
        operations = trajectory.get("visual_operations", [])
        
        if not thoughts or not operations:
            return 0.5
        
        # Check thought progression
        thought_types = [t.get("type", "unknown") for t in thoughts if isinstance(t, dict)]
        
        # Good pattern: analysis -> planning -> execution -> reflection
        good_patterns = [
            ["analysis", "planning"],
            ["planning", "execution"],
            ["reflection", "correction"]
        ]
        
        pattern_score = 0.0
        for i in range(len(thought_types) - 1):
            pair = [thought_types[i], thought_types[i+1]]
            if pair in good_patterns:
                pattern_score += 0.2
        
        # Check for corrections after high confidence
        for i, thought in enumerate(thoughts):
            if isinstance(thought, dict):
                if thought.get("type") == "correction" and i > 0:
                    prev_confidence = thoughts[i-1].get("confidence", 1.0) if isinstance(thoughts[i-1], dict) else 1.0
                    if prev_confidence > 0.8:
                        score -= 0.1  # Correction after high confidence is suspicious
        
        return min(1.0, max(0.0, score + pattern_score))
    
    def _score_correctness(self, trajectory: Dict[str, Any]) -> float:
        """Score based on correctness indicators"""
        score = 0.8  # Base score
        
        # Check answer quality
        answer = trajectory.get("answer", "")
        if not answer:
            return 0.0
        
        # Length heuristics
        if len(answer) < 10:
            score -= 0.2
        elif len(answer) > 500:
            score -= 0.1
        
        # Check for error indicators
        error_indicators = ["error", "failed", "incorrect", "wrong", "mistake"]
        answer_lower = answer.lower()
        for indicator in error_indicators:
            if indicator in answer_lower:
                score -= 0.15
        
        # Check for confidence indicators
        confidence_indicators = ["correct", "accurate", "precisely", "exactly"]
        for indicator in confidence_indicators:
            if indicator in answer_lower:
                score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _score_efficiency(self, trajectory: Dict[str, Any]) -> float:
        """Score based on operation efficiency"""
        operations = trajectory.get("visual_operations", [])
        
        if not operations:
            return 0.5
        
        score = 1.0
        
        # Penalize redundant operations
        op_sequence = [op.get("operation") for op in operations if isinstance(op, dict)]
        unique_ops = set(op_sequence)
        
        if len(op_sequence) > len(unique_ops) * 1.5:
            score -= 0.2  # Too many repeated operations
        
        # Check for optimal operation count
        task_type = trajectory.get("metadata", {}).get("task_type", "unknown")
        expected_ops = {
            "segmentation": 2,  # SEGMENT + GET_PROPERTIES
            "tracking": 3,      # SELECT_FRAME + TRACK + analysis
            "reading": 2,        # ZOOM_IN + READ_TEXT
        }
        
        if task_type in expected_ops:
            expected = expected_ops[task_type]
            actual = len(operations)
            if actual > expected * 2:
                score -= 0.15
        
        # Penalize very long sequences
        if len(operations) > 10:
            score -= 0.1
        
        return max(0.0, score)
    
    def _score_complexity(self, trajectory: Dict[str, Any]) -> float:
        """Score based on appropriate complexity"""
        difficulty = trajectory.get("difficulty", "medium")
        operations = trajectory.get("visual_operations", [])
        thoughts = trajectory.get("chain_of_thought", [])
        
        # Expected complexity ranges
        expected_ranges = {
            "easy": (2, 4),
            "medium": (3, 6),
            "hard": (5, 10)
        }
        
        if difficulty not in expected_ranges:
            return 0.5
        
        min_ops, max_ops = expected_ranges[difficulty]
        actual_ops = len(operations)
        
        if min_ops <= actual_ops <= max_ops:
            score = 1.0
        elif actual_ops < min_ops:
            score = 0.7  # Too simple
        else:
            score = 0.8  # Too complex
        
        # Adjust for thought complexity
        thought_diversity = len(set(t.get("type", "unknown") for t in thoughts if isinstance(t, dict)))
        if thought_diversity >= 3:
            score = min(1.0, score + 0.1)
        
        return score
    
    def _score_consistency(self, trajectory: Dict[str, Any]) -> float:
        """Score internal consistency"""
        score = 1.0
        
        # Check type consistency
        traj_type = trajectory.get("type", "")
        if traj_type == "trap_sample":
            # Should have some error indicators
            has_error = any(
                "error" in str(op.get("output", {}))
                for op in trajectory.get("visual_operations", [])
                if isinstance(op, dict)
            )
            if not has_error:
                score -= 0.2
        
        elif traj_type == "self_correction":
            # Should have correction thoughts
            has_correction = any(
                t.get("type") == "correction"
                for t in trajectory.get("chain_of_thought", [])
                if isinstance(t, dict)
            )
            if not has_correction:
                score -= 0.2
        
        return max(0.0, score)


# ======================== VALIDATION PIPELINE ========================

class ValidationPipeline:
    """Complete validation and scoring pipeline"""
    
    def __init__(
        self,
        validators: Optional[List[Validator]] = None,
        scorer: Optional[QualityScorer] = None,
        level: ValidationLevel = ValidationLevel.STANDARD
    ):
        self.validators = validators or [
            StructuralValidator(level),
            LogicalValidator(),
            DuplicateDetector()
        ]
        self.scorer = scorer or ComprehensiveQualityScorer()
        self.level = level
        
        # Statistics
        self.stats = {
            "total_validated": 0,
            "total_valid": 0,
            "total_invalid": 0,
            "issue_counts": Counter(),
            "quality_distribution": []
        }
    
    def validate_trajectory(self, trajectory: Dict[str, Any]) -> ValidationResult:
        """Validate a single trajectory"""
        # Combine results from all validators
        combined_result = ValidationResult(is_valid=True)
        
        for validator in self.validators:
            result = validator.validate(trajectory)
            
            # Merge issues
            combined_result.issues.extend(result.issues)
            combined_result.warnings.extend(result.warnings)
            
            if not result.is_valid:
                combined_result.is_valid = False
        
        # Score quality if valid
        if combined_result.is_valid or self.level == ValidationLevel.LENIENT:
            quality_scores = self.scorer.score(trajectory)
            combined_result.quality_scores = quality_scores
            
            # Calculate overall score
            weights = self.scorer.get_weights()
            overall = sum(
                score * weights.get(dim, 0)
                for dim, score in quality_scores.items()
            )
            combined_result.overall_score = overall
            
            # Add quality issues if score is low
            if overall < 0.5:
                combined_result.add_issue(ValidationIssue(
                    issue_type=IssueType.QUALITY_ISSUE,
                    severity="warning",
                    field="overall",
                    message=f"Low quality score: {overall:.2f}",
                    suggestion="Review and improve trajectory quality"
                ))
        
        # Update statistics
        self._update_stats(combined_result)
        
        return combined_result
    
    def validate_batch(
        self,
        trajectories: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[ValidationResult]:
        """Validate a batch of trajectories"""
        results = []
        
        for trajectory in trajectories:
            result = self.validate_trajectory(trajectory)
            results.append(result)
        
        return results
    
    def _update_stats(self, result: ValidationResult):
        """Update validation statistics"""
        self.stats["total_validated"] += 1
        
        if result.is_valid:
            self.stats["total_valid"] += 1
        else:
            self.stats["total_invalid"] += 1
        
        # Count issues
        for issue in result.issues:
            self.stats["issue_counts"][issue.issue_type.value] += 1
        
        # Track quality distribution
        if result.overall_score > 0:
            self.stats["quality_distribution"].append(result.overall_score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        stats = self.stats.copy()
        
        # Calculate quality statistics
        if stats["quality_distribution"]:
            scores = stats["quality_distribution"]
            stats["quality_stats"] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "percentiles": {
                    "25": np.percentile(scores, 25),
                    "50": np.percentile(scores, 50),
                    "75": np.percentile(scores, 75)
                }
            }
        
        return stats
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate validation report"""
        stats = self.get_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Validated: {stats['total_validated']}")
        report.append(f"Valid: {stats['total_valid']} ({stats['total_valid']/max(1, stats['total_validated'])*100:.1f}%)")
        report.append(f"Invalid: {stats['total_invalid']} ({stats['total_invalid']/max(1, stats['total_validated'])*100:.1f}%)")
        report.append("")
        
        if stats["issue_counts"]:
            report.append("Issue Distribution:")
            for issue_type, count in stats["issue_counts"].most_common():
                report.append(f"  - {issue_type}: {count}")
            report.append("")
        
        if "quality_stats" in stats:
            qs = stats["quality_stats"]
            report.append("Quality Statistics:")
            report.append(f"  Mean: {qs['mean']:.3f}")
            report.append(f"  Std: {qs['std']:.3f}")
            report.append(f"  Min: {qs['min']:.3f}")
            report.append(f"  Max: {qs['max']:.3f}")
            report.append(f"  Median: {qs['percentiles']['50']:.3f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# ======================== USAGE EXAMPLES ========================

def example_usage():
    """Example usage of validation and scoring system"""
    
    # Example trajectory
    trajectory = {
        "id": "traj_12345678",
        "task": "Identify and count the cats in the image",
        "chain_of_thought": [
            {"thought": "I need to scan the image for cats", "type": "analysis", "confidence": 0.9},
            {"thought": "I'll segment potential cat regions", "type": "planning", "confidence": 0.85},
            {"thought": "Found 2 cats in the image", "type": "execution", "confidence": 0.95}
        ],
        "visual_operations": [
            {"operation": "SEGMENT_OBJECT_AT", "params": {"x": 100, "y": 200}},
            {"operation": "GET_PROPERTIES", "params": {"mask": "mask_1"}},
            {"operation": "SEGMENT_OBJECT_AT", "params": {"x": 300, "y": 250}},
            {"operation": "GET_PROPERTIES", "params": {"mask": "mask_2"}}
        ],
        "answer": "There are 2 cats in the image. One is in the left portion and another in the right.",
        "type": "golden_positive",
        "difficulty": "medium",
        "metadata": {"task_type": "segmentation", "source": "test"}
    }
    
    # Create validation pipeline
    pipeline = ValidationPipeline(level=ValidationLevel.STANDARD)
    
    # Validate single trajectory
    result = pipeline.validate_trajectory(trajectory)
    
    print("Validation Result:")
    print(f"Valid: {result.is_valid}")
    print(f"Issues: {len(result.issues)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print("\nQuality Scores:")
    for dim, score in result.quality_scores.items():
        print(f"  {dim.value}: {score:.3f}")
    
    # Example with invalid trajectory
    invalid_trajectory = {
        "task": "Find objects",
        "answer": "Found them"
        # Missing required fields
    }
    
    result2 = pipeline.validate_trajectory(invalid_trajectory)
    print("\n" + "=" * 40)
    print("Invalid Trajectory Result:")
    print(f"Valid: {result2.is_valid}")
    print("Issues:")
    for issue in result2.issues:
        print(f"  - [{issue.severity}] {issue.field}: {issue.message}")
    
    # Generate report
    print("\n" + "=" * 40)
    report = pipeline.generate_report()
    print(report)


if __name__ == "__main__":
    print("Validation and Quality Scoring System")
    print("=" * 60)
    example_usage()