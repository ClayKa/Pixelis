"""MOT loader with optional sliding-window clip sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import pandas as pd

from .base_loader import BaseLoader


class MotSlidingWindowLoader(BaseLoader):
    """Load MOT sequences either as full videos or fixed-length clips."""

    def __init__(self, config: Dict[str, Any]):
        if "path" not in config:
            raise ValueError("MotSlidingWindowLoader config must include 'path'")

        self.base_path = Path(config["path"])
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_path}")
        if not self.base_path.is_dir():
            raise FileNotFoundError(f"Base path is not a directory: {self.base_path}")

        self.sampling_strategy = config.get("sampling_strategy", {"type": "full_sequence"})
        super().__init__(config)

    def _build_index(self) -> List[Dict[str, Any]]:
        sequences = []
        for sequence_path in sorted(self.base_path.iterdir()):
            img_dir = sequence_path / "img1"
            gt_file = sequence_path / "gt" / "gt.txt"
            if sequence_path.is_dir() and img_dir.is_dir() and gt_file.is_file():
                frame_count = self._count_frames(img_dir)
                if frame_count > 0:
                    sequences.append((sequence_path, frame_count))

        strategy_type = self.sampling_strategy.get("type", "full_sequence")
        if strategy_type == "full_sequence":
            return [
                self._make_clip_entry(sequence_path, frame_count, 1, frame_count, 0, True)
                for sequence_path, frame_count in sequences
            ]

        if strategy_type != "sliding_window":
            raise ValueError(f"Unsupported MOT sampling strategy: {strategy_type}")

        duration = int(self.sampling_strategy.get("clip_duration_frames", 100))
        stride = int(self.sampling_strategy.get("stride_frames", duration))
        min_frames = int(self.sampling_strategy.get("min_clip_frames", 1))
        if duration <= 0 or stride <= 0 or min_frames <= 0:
            raise ValueError("clip_duration_frames, stride_frames, and min_clip_frames must be positive")

        clips: List[Dict[str, Any]] = []
        for sequence_path, frame_count in sequences:
            clip_index = 0
            start_frame = 1
            while start_frame <= frame_count:
                end_frame = min(start_frame + duration - 1, frame_count)
                if end_frame - start_frame + 1 >= min_frames:
                    clips.append(
                        self._make_clip_entry(
                            sequence_path,
                            frame_count,
                            start_frame,
                            end_frame,
                            clip_index,
                            False,
                        )
                    )
                    clip_index += 1
                start_frame += stride

        return clips

    def get_item(self, index: int) -> Dict[str, Any]:
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")

        entry = self._index[index]
        img_dir = entry["sequence_path"] / "img1"
        gt_file = entry["sequence_path"] / "gt" / "gt.txt"
        trajectories = self._parse_clip_trajectories(
            gt_file=gt_file,
            start_frame=entry["start_frame"],
            end_frame=entry["end_frame"],
        )

        return {
            "source_dataset": self.source_name,
            "sample_id": entry["sample_id"],
            "media_type": "video",
            "media_path": str(img_dir.resolve()),
            "width": None,
            "height": None,
            "clip_info": {
                "sequence_id": entry["sequence_id"],
                "clip_index": entry["clip_index"],
                "start_frame": entry["start_frame"],
                "end_frame": entry["end_frame"],
                "duration_frames": entry["duration_frames"],
                "source_sequence_frames": entry["source_sequence_frames"],
                "is_full_sequence": entry["is_full_sequence"],
            },
            "annotations": {
                "tracking": {
                    "trajectories": trajectories,
                    "num_objects": len(trajectories),
                },
                "dataset_info": {
                    "task_type": "multi_object_tracking",
                    "source": "MOT",
                    "sampling_type": self.sampling_strategy.get("type", "full_sequence"),
                },
            },
        }

    def get_clip_statistics(self) -> Dict[str, Any]:
        if not self._index:
            return {
                "total_clips": 0,
                "total_sequences": 0,
                "sampling_type": self.sampling_strategy.get("type", "full_sequence"),
                "clips_per_sequence": {"min": 0, "max": 0, "mean": 0},
                "config": self._strategy_stats(),
            }

        counts = Counter(entry["sequence_id"] for entry in self._index)
        values = list(counts.values())
        return {
            "total_clips": len(self._index),
            "total_sequences": len(counts),
            "sampling_type": self.sampling_strategy.get("type", "full_sequence"),
            "clips_per_sequence": {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
            },
            "config": self._strategy_stats(),
        }

    def _make_clip_entry(
        self,
        sequence_path: Path,
        frame_count: int,
        start_frame: int,
        end_frame: int,
        clip_index: int,
        is_full_sequence: bool,
    ) -> Dict[str, Any]:
        sequence_id = sequence_path.name
        sample_id = sequence_id if is_full_sequence else f"{sequence_id}_clip_{clip_index:04d}"
        return {
            "sequence_path": sequence_path,
            "sequence_id": sequence_id,
            "sample_id": sample_id,
            "clip_index": clip_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": end_frame - start_frame + 1,
            "source_sequence_frames": frame_count,
            "is_full_sequence": is_full_sequence,
        }

    @staticmethod
    def _count_frames(img_dir: Path) -> int:
        return len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))

    def _parse_clip_trajectories(
        self,
        gt_file: Path,
        start_frame: int,
        end_frame: int,
    ) -> List[Dict[str, Any]]:
        df = pd.read_csv(
            gt_file,
            header=None,
            names=[
                "frame_id",
                "object_id",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "confidence",
                "class",
                "visibility",
                "unused",
            ],
        )
        df = df[(df["frame_id"] >= start_frame) & (df["frame_id"] <= end_frame)]

        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in df.iterrows():
            grouped[int(row["object_id"])].append(
                {
                    "frame": int(row["frame_id"]) - start_frame,
                    "absolute_frame": int(row["frame_id"]),
                    "bbox": [
                        float(row["bb_left"]),
                        float(row["bb_top"]),
                        float(row["bb_width"]),
                        float(row["bb_height"]),
                    ],
                    "confidence": float(row["confidence"]),
                }
            )

        return [
            {"object_id": object_id, "trajectory": points}
            for object_id, points in sorted(grouped.items())
        ]

    def _strategy_stats(self) -> Dict[str, Any]:
        strategy_type = self.sampling_strategy.get("type", "full_sequence")
        if strategy_type != "sliding_window":
            return {"type": strategy_type, "overlap_ratio": 0.0}

        duration = float(self.sampling_strategy.get("clip_duration_frames", 100))
        stride = float(self.sampling_strategy.get("stride_frames", duration))
        overlap_ratio = max(0.0, 1.0 - stride / duration) if duration else 0.0
        return {
            **self.sampling_strategy,
            "overlap_ratio": overlap_ratio,
        }


def create_mot_loader(config: Dict[str, Any]) -> MotSlidingWindowLoader:
    """Factory used by tests and manifests."""

    return MotSlidingWindowLoader(config)
