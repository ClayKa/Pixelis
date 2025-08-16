"""
Reproducibility and Artifact Management System for Pixelis.

This module provides comprehensive experiment tracking, artifact versioning,
and reproducibility guarantees for all training and evaluation pipelines.
"""

from .artifact_manager import ArtifactManager, ArtifactType
from .experiment_context import (
    ExperimentContext,
    TTRLContext,
    EnvironmentCaptureLevel,
)
from .decorators import track_artifacts, reproducible
from .config_capture import ConfigCapture
from .lineage_tracker import LineageTracker

__all__ = [
    "ArtifactManager",
    "ArtifactType",
    "ExperimentContext",
    "TTRLContext",
    "EnvironmentCaptureLevel",
    "track_artifacts",
    "reproducible",
    "ConfigCapture",
    "LineageTracker",
]

# Version of the reproducibility module
__version__ = "1.0.0"