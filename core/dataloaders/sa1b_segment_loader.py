"""SA-1B loader optimized for segmentation and geometry tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class Sa1bSegmentLoader(BaseLoader):
    """Load SA-1B images with filtered, task-ready instance masks."""

    def __init__(self, config: Dict[str, Any]):
        for key in ("path", "annotations_path"):
            if key not in config:
                raise ValueError(f"Sa1bSegmentLoader config must include '{key}'")

        self.images_path = Path(config["path"])
        self.annotations_path = Path(config["annotations_path"])
        if not self.images_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.images_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotations_path}")

        self.min_pixel_area = config.get("min_pixel_area", 100)
        self.min_stability_score = config.get("min_stability_score", 0.5)
        self.min_predicted_iou = config.get("min_predicted_iou", 0.5)

        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        index: List[Dict[str, Any]] = []

        for annotation_path in sorted(self.annotations_path.glob("*.json")):
            try:
                with annotation_path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed SA-1B annotation %s: %s", annotation_path, exc)
                continue

            image_info = payload.get("image", {})
            annotations = payload.get("annotations", [])
            file_name = image_info.get("file_name")
            if not file_name:
                logger.warning("Skipping annotation without image.file_name: %s", annotation_path)
                continue

            image_path = self.images_path / file_name
            if not image_path.exists():
                logger.warning("Skipping annotation without matching image: %s", image_path)
                continue

            usable_annotations = [
                annotation
                for annotation in annotations
                if self._passes_quality_filter(annotation)
            ]
            if not usable_annotations:
                continue

            index.append(
                {
                    "image_id": image_info.get("image_id", annotation_path.stem),
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "width": image_info.get("width", 0),
                    "height": image_info.get("height", 0),
                    "file_name": file_name,
                    "annotations": usable_annotations,
                    "raw_annotation_count": len(annotations),
                    "num_usable_annotations": len(usable_annotations),
                }
            )

        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        return self._build_sample(self._index[index], self._index[index]["annotations"])

    def get_high_quality_instances(
        self,
        min_stability_score: float = 0.95,
        min_predicted_iou: float = 0.9,
    ) -> List[Dict[str, Any]]:
        return self._filtered_samples(
            lambda ann: ann.get("stability_score", 0.0) >= min_stability_score
            and ann.get("predicted_iou", 0.0) >= min_predicted_iou
        )

    def get_samples_by_area_range(
        self,
        min_area: float = 0,
        max_area: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self._filtered_samples(
            lambda ann: ann.get("area", 0) >= min_area
            and (max_area is None or ann.get("area", 0) <= max_area)
        )

    def get_samples_suitable_for_geometric_comparison(
        self,
        min_instances: int = 2,
        max_instances: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        samples = []
        for entry in self._index:
            count = len(entry["annotations"])
            if count >= min_instances and (max_instances is None or count <= max_instances):
                samples.append(self._build_sample(entry, entry["annotations"]))
        return samples

    def get_geometric_analysis_statistics(self) -> Dict[str, Any]:
        if not self._index:
            return {"error": "No samples available"}

        all_annotations = [
            annotation
            for entry in self._index
            for annotation in entry["annotations"]
        ]
        areas = [annotation.get("area", 0) for annotation in all_annotations]
        aspect_ratios = [
            self._calculate_aspect_ratio(annotation.get("bbox", []))
            for annotation in all_annotations
        ]
        relative_areas = [
            self._relative_area(annotation, self._entry_for_annotation(annotation))
            for annotation in all_annotations
        ]
        stability_scores = [annotation.get("stability_score", 0.0) for annotation in all_annotations]
        predicted_ious = [annotation.get("predicted_iou", 0.0) for annotation in all_annotations]
        coverage_values = [
            self._coverage_ratio(entry["annotations"], entry["width"], entry["height"])
            for entry in self._index
        ]

        return {
            "total_samples_analyzed": len(self._index),
            "total_instances": len(all_annotations),
            "area_statistics": self._numeric_stats(areas, prefix_pixels=True),
            "aspect_ratio_statistics": self._numeric_stats(aspect_ratios),
            "relative_area_statistics": self._numeric_stats(relative_areas),
            "quality_statistics": {
                "avg_stability_score": mean(stability_scores) if stability_scores else 0,
                "avg_predicted_iou": mean(predicted_ious) if predicted_ious else 0,
            },
            "coverage_statistics": self._numeric_stats(coverage_values),
            "filtering_impact": {
                "min_pixel_area_applied": self.min_pixel_area,
                "min_stability_score_applied": self.min_stability_score,
                "min_predicted_iou_applied": self.min_predicted_iou,
            },
        }

    def _build_sample(
        self,
        entry: Dict[str, Any],
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample = self._get_standardized_base(
            sample_id=str(entry["image_id"]),
            media_path=Path(entry["image_path"]),
            media_type="image",
        )

        instances = [self._build_instance(annotation, entry) for annotation in annotations]
        total_area = sum(instance["area_pixels"] for instance in instances)
        coverage_ratio = self._coverage_ratio(annotations, entry["width"], entry["height"])

        sample["annotations"].update(
            {
                "instance_segmentation": instances,
                "num_instances": len(instances),
                "total_segmented_area": total_area,
                "coverage_ratio": coverage_ratio,
                "quality_statistics": self._quality_statistics(annotations, entry),
                "image_metadata": {
                    "sa_image_id": entry["image_id"],
                    "original_width": entry["width"],
                    "original_height": entry["height"],
                    "file_name": entry["file_name"],
                },
                "dataset_info": {
                    "task_type": "instance_segmentation_optimized",
                    "source": "SA-1B",
                    "suitable_for_segment_object_at": True,
                    "suitable_for_get_properties": True,
                    "has_center_points": True,
                    "has_quality_metrics": True,
                    "mask_format": "rle",
                },
            }
        )
        return sample

    def _build_instance(self, annotation: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        bbox = annotation.get("bbox", [])
        area = annotation.get("area", 0)
        center = self._center_point(bbox)
        return {
            "instance_id": annotation.get("id"),
            "bbox": bbox,
            "area_pixels": area,
            "center_point": center,
            "point_coords": annotation.get("point_coords", [center]),
            "crop_box": annotation.get("crop_box", []),
            "segmentation_mask_rle": annotation.get("segmentation", {}),
            "quality_metrics": {
                "stability_score": annotation.get("stability_score", 0.0),
                "predicted_iou": annotation.get("predicted_iou", 0.0),
            },
            "geometric_properties": {
                "aspect_ratio": self._calculate_aspect_ratio(bbox),
                "relative_area": self._relative_area(annotation, entry),
            },
        }

    def _passes_quality_filter(self, annotation: Dict[str, Any]) -> bool:
        return (
            annotation.get("area", 0) >= self.min_pixel_area
            and annotation.get("stability_score", 0.0) >= self.min_stability_score
            and annotation.get("predicted_iou", 0.0) >= self.min_predicted_iou
        )

    def _filtered_samples(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        samples = []
        for entry in self._index:
            annotations = [annotation for annotation in entry["annotations"] if predicate(annotation)]
            if annotations:
                samples.append(self._build_sample(entry, annotations))
        return samples

    @staticmethod
    def _calculate_aspect_ratio(bbox: List[float]) -> float:
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            return 0.0
        return bbox[2] / bbox[3]

    @staticmethod
    def _center_point(bbox: List[float]) -> List[float]:
        if len(bbox) < 4:
            return [0.0, 0.0]
        return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]

    @staticmethod
    def _coverage_ratio(
        annotations: List[Dict[str, Any]],
        width: float,
        height: float,
    ) -> float:
        image_area = width * height
        if image_area <= 0:
            return 0.0
        return min(1.0, sum(annotation.get("area", 0) for annotation in annotations) / image_area)

    @staticmethod
    def _relative_area(annotation: Dict[str, Any], entry: Dict[str, Any]) -> float:
        image_area = entry.get("width", 0) * entry.get("height", 0)
        return annotation.get("area", 0) / image_area if image_area > 0 else 0.0

    def _entry_for_annotation(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        for entry in self._index:
            if annotation in entry["annotations"]:
                return entry
        return {"width": 0, "height": 0}

    @staticmethod
    def _quality_statistics(
        annotations: List[Dict[str, Any]],
        entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        stability_scores = [annotation.get("stability_score", 0.0) for annotation in annotations]
        predicted_ious = [annotation.get("predicted_iou", 0.0) for annotation in annotations]
        raw_count = entry.get("raw_annotation_count", len(annotations))
        return {
            "avg_stability_score": mean(stability_scores) if stability_scores else 0.0,
            "min_stability_score": min(stability_scores) if stability_scores else 0.0,
            "max_stability_score": max(stability_scores) if stability_scores else 0.0,
            "avg_predicted_iou": mean(predicted_ious) if predicted_ious else 0.0,
            "usable_mask_ratio": len(annotations) / raw_count if raw_count else 0.0,
        }

    @staticmethod
    def _numeric_stats(values: List[float], prefix_pixels: bool = False) -> Dict[str, float]:
        if not values:
            if prefix_pixels:
                return {"min_pixels": 0, "max_pixels": 0, "avg_pixels": 0}
            return {"min": 0, "max": 0, "avg": 0}

        if prefix_pixels:
            return {
                "min_pixels": min(values),
                "max_pixels": max(values),
                "avg_pixels": mean(values),
            }
        return {
            "min": min(values),
            "max": max(values),
            "avg": mean(values),
        }
