#!/usr/bin/env python3
"""
Augment Data for Stress Testing Visual Operations.
This script creates challenging augmented versions of test data to evaluate
the robustness of individual visual operations under difficult conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import argparse
from tqdm import tqdm
import yaml
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imgaug.augmenters as iaa
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    # For SEGMENT_OBJECT_AT / GET_PROPERTIES tasks
    occlusion_min: float = 0.1
    occlusion_max: float = 0.5
    low_light_factor_min: float = 0.2
    low_light_factor_max: float = 0.5
    motion_blur_kernel_min: int = 5
    motion_blur_kernel_max: int = 21
    
    # For READ_TEXT tasks
    perspective_distortion_scale: float = 0.3
    text_noise_level: float = 0.2
    font_variations: List[str] = None
    text_blur_sigma: float = 2.0
    
    # For TRACK_OBJECT tasks
    rapid_motion_speed: float = 20.0  # pixels per frame
    camera_shake_amplitude: float = 10.0
    occlusion_duration_frames: int = 5
    
    # General augmentations
    gaussian_noise_var: float = 0.05
    salt_pepper_prob: float = 0.02
    jpeg_quality_min: int = 20
    jpeg_quality_max: int = 50
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.1
    
    def __post_init__(self):
        if self.font_variations is None:
            self.font_variations = [
                "arial.ttf", "times.ttf", "courier.ttf", 
                "comic.ttf", "impact.ttf"
            ]


class StressTestAugmenter:
    """Apply challenging augmentations for stress testing."""
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize augmentation pipelines
        self._init_augmentation_pipelines()
    
    def _init_augmentation_pipelines(self):
        """Initialize various augmentation pipelines."""
        
        # Albumentations pipeline for segmentation tasks
        self.segmentation_augment = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.5, -0.2),  # Darken for low-light
                    contrast_limit=(-0.3, 0.3),
                    p=1.0
                ),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0),
                A.RandomRain(slant_lower=-10, slant_upper=10, p=1.0),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=1.0),
            ], p=0.8),
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 21), p=1.0),
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
                A.MedianBlur(blur_limit=9, p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=5,
                max_height=50,
                max_width=50,
                fill_value=0,
                p=0.5
            ),
        ])
        
        # ImgAug pipeline for text tasks
        self.text_augment = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.15))),
            iaa.Sometimes(0.3, iaa.Affine(
                rotate=(-15, 15),
                shear=(-10, 10)
            )),
            iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
            iaa.Sometimes(0.3, iaa.SaltAndPepper(0.02)),
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.5, 2.0)),
                iaa.MotionBlur(k=(3, 11), angle=(-45, 45)),
                iaa.MedianBlur(k=(3, 7))
            ])),
            iaa.Sometimes(0.3, iaa.JpegCompression(compression=(20, 50))),
        ])
        
        # Video augmentation for tracking tasks
        self.video_augment = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=50, sigma=5)),
            iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            iaa.Sometimes(0.4, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
        ])
    
    def augment_for_segmentation(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        severity: str = "medium"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations for SEGMENT_OBJECT_AT and GET_PROPERTIES tasks."""
        
        severity_multiplier = {"easy": 0.3, "medium": 0.6, "hard": 1.0}.get(severity, 0.6)
        
        # Apply partial occlusion
        if random.random() < 0.7 * severity_multiplier:
            image, mask = self._add_partial_occlusion(image, mask)
        
        # Apply low-light conditions
        if random.random() < 0.6 * severity_multiplier:
            image = self._simulate_low_light(image)
        
        # Apply motion blur
        if random.random() < 0.5 * severity_multiplier:
            kernel_size = random.randint(
                self.config.motion_blur_kernel_min,
                self.config.motion_blur_kernel_max
            )
            image = self._add_motion_blur(image, kernel_size)
        
        # Apply albumentations pipeline
        if mask is not None:
            augmented = self.segmentation_augment(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.segmentation_augment(image=image)
            return augmented['image'], None
    
    def augment_for_text_reading(
        self,
        image: np.ndarray,
        text_regions: Optional[List[Dict]] = None,
        severity: str = "medium"
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Apply augmentations for READ_TEXT tasks."""
        
        severity_multiplier = {"easy": 0.3, "medium": 0.6, "hard": 1.0}.get(severity, 0.6)
        
        # Apply perspective distortion
        if random.random() < 0.7 * severity_multiplier:
            image = self._apply_perspective_distortion(image)
        
        # Add text-specific noise
        if random.random() < 0.6 * severity_multiplier:
            image = self._add_text_noise(image, text_regions)
        
        # Vary fonts if we're generating synthetic text
        if text_regions and random.random() < 0.5:
            image = self._vary_text_fonts(image, text_regions)
        
        # Apply imgaug pipeline
        image_aug = self.text_augment(image=image)
        
        # Apply blur specifically to text regions
        if text_regions and random.random() < 0.4 * severity_multiplier:
            image_aug = self._blur_text_regions(image_aug, text_regions)
        
        return image_aug, text_regions
    
    def augment_for_tracking(
        self,
        video_frames: List[np.ndarray],
        object_trajectories: Optional[List[Dict]] = None,
        severity: str = "medium"
    ) -> Tuple[List[np.ndarray], Optional[List[Dict]]]:
        """Apply augmentations for TRACK_OBJECT tasks."""
        
        severity_multiplier = {"easy": 0.3, "medium": 0.6, "hard": 1.0}.get(severity, 0.6)
        
        augmented_frames = []
        augmented_trajectories = object_trajectories.copy() if object_trajectories else None
        
        # Add rapid motion
        if random.random() < 0.6 * severity_multiplier:
            video_frames, augmented_trajectories = self._add_rapid_motion(
                video_frames, augmented_trajectories
            )
        
        # Add camera shake
        if random.random() < 0.7 * severity_multiplier:
            video_frames = self._add_camera_shake(video_frames)
        
        # Add temporary occlusions
        if random.random() < 0.5 * severity_multiplier:
            occlusion_start = random.randint(0, len(video_frames) - self.config.occlusion_duration_frames)
            video_frames = self._add_temporal_occlusion(
                video_frames, occlusion_start, self.config.occlusion_duration_frames
            )
        
        # Apply frame-wise augmentations
        for frame in video_frames:
            augmented_frame = self.video_augment(image=frame)
            
            # Add frame-specific noise
            if random.random() < 0.3 * severity_multiplier:
                augmented_frame = self._add_frame_noise(augmented_frame)
            
            augmented_frames.append(augmented_frame)
        
        return augmented_frames, augmented_trajectories
    
    def _add_partial_occlusion(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Add partial occlusion to image."""
        h, w = image.shape[:2]
        
        # Random occlusion size
        occlusion_ratio = random.uniform(self.config.occlusion_min, self.config.occlusion_max)
        occ_h = int(h * occlusion_ratio)
        occ_w = int(w * occlusion_ratio)
        
        # Random position
        x = random.randint(0, w - occ_w)
        y = random.randint(0, h - occ_h)
        
        # Apply occlusion
        occlusion_color = random.choice([
            [0, 0, 0],  # Black
            [128, 128, 128],  # Gray
            [255, 255, 255],  # White
        ])
        
        image_occluded = image.copy()
        image_occluded[y:y+occ_h, x:x+occ_w] = occlusion_color
        
        # Update mask if provided
        if mask is not None:
            mask_occluded = mask.copy()
            mask_occluded[y:y+occ_h, x:x+occ_w] = 0
            return image_occluded, mask_occluded
        
        return image_occluded, None
    
    def _simulate_low_light(self, image: np.ndarray) -> np.ndarray:
        """Simulate low-light conditions."""
        # Reduce brightness
        factor = random.uniform(self.config.low_light_factor_min, self.config.low_light_factor_max)
        darkened = (image * factor).astype(np.uint8)
        
        # Add noise (more prominent in low light)
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy = cv2.add(darkened, noise)
        
        # Reduce contrast
        alpha = 0.7  # Contrast control
        beta = 30    # Brightness control
        adjusted = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)
        
        return adjusted
    
    def _add_motion_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Add motion blur to image."""
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Random direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        elif direction == 'vertical':
            kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        else:  # diagonal
            np.fill_diagonal(kernel, 1)
        
        kernel = kernel / kernel.sum()
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def _apply_perspective_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective distortion to image."""
        h, w = image.shape[:2]
        
        # Define source points (corners of original image)
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points (distorted corners)
        scale = self.config.perspective_distortion_scale
        dst_points = np.float32([
            [random.uniform(0, w * scale), random.uniform(0, h * scale)],
            [w - random.uniform(0, w * scale), random.uniform(0, h * scale)],
            [w - random.uniform(0, w * scale), h - random.uniform(0, h * scale)],
            [random.uniform(0, w * scale), h - random.uniform(0, h * scale)]
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        distorted = cv2.warpPerspective(image, matrix, (w, h))
        
        return distorted
    
    def _add_text_noise(
        self,
        image: np.ndarray,
        text_regions: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """Add noise specifically to text regions."""
        noisy_image = image.copy()
        
        if text_regions:
            for region in text_regions:
                x, y, w, h = region.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                
                # Extract region
                roi = noisy_image[y:y+h, x:x+w]
                
                # Add salt and pepper noise
                noise = np.random.random(roi.shape)
                roi[noise < self.config.text_noise_level / 2] = 0
                roi[noise > 1 - self.config.text_noise_level / 2] = 255
                
                # Put back
                noisy_image[y:y+h, x:x+w] = roi
        else:
            # Add noise to entire image
            noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
            noisy_image = cv2.add(noisy_image, noise)
        
        return noisy_image
    
    def _vary_text_fonts(
        self,
        image: np.ndarray,
        text_regions: List[Dict]
    ) -> np.ndarray:
        """Vary fonts in text regions (for synthetic text)."""
        # This would require re-rendering text with different fonts
        # For now, we'll apply different distortions to simulate font variation
        
        varied_image = image.copy()
        
        for region in text_regions:
            x, y, w, h = region.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            roi = varied_image[y:y+h, x:x+w]
            
            # Random transformation to simulate font variation
            transform = random.choice([
                lambda img: cv2.resize(img, (w, int(h * 0.8)), interpolation=cv2.INTER_LINEAR),
                lambda img: cv2.resize(img, (w, int(h * 1.2)), interpolation=cv2.INTER_LINEAR),
                lambda img: cv2.GaussianBlur(img, (3, 3), 0),
                lambda img: cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1),
                lambda img: cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1),
            ])
            
            transformed_roi = transform(roi)
            
            # Resize back if needed
            if transformed_roi.shape[:2] != (h, w):
                transformed_roi = cv2.resize(transformed_roi, (w, h))
            
            varied_image[y:y+h, x:x+w] = transformed_roi
        
        return varied_image
    
    def _blur_text_regions(
        self,
        image: np.ndarray,
        text_regions: List[Dict]
    ) -> np.ndarray:
        """Apply blur specifically to text regions."""
        blurred_image = image.copy()
        
        for region in text_regions:
            x, y, w, h = region.get('bbox', [0, 0, image.shape[1], image.shape[0]])
            roi = blurred_image[y:y+h, x:x+w]
            
            # Apply Gaussian blur
            blurred_roi = cv2.GaussianBlur(roi, (5, 5), self.config.text_blur_sigma)
            
            blurred_image[y:y+h, x:x+w] = blurred_roi
        
        return blurred_image
    
    def _add_rapid_motion(
        self,
        frames: List[np.ndarray],
        trajectories: Optional[List[Dict]] = None
    ) -> Tuple[List[np.ndarray], Optional[List[Dict]]]:
        """Add rapid motion to video frames."""
        augmented_frames = []
        augmented_trajectories = trajectories.copy() if trajectories else None
        
        # Random motion vector
        motion_x = random.uniform(-self.config.rapid_motion_speed, self.config.rapid_motion_speed)
        motion_y = random.uniform(-self.config.rapid_motion_speed, self.config.rapid_motion_speed)
        
        for i, frame in enumerate(frames):
            # Calculate translation for this frame
            tx = int(motion_x * i)
            ty = int(motion_y * i)
            
            # Create translation matrix
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # Apply translation
            h, w = frame.shape[:2]
            translated = cv2.warpAffine(frame, M, (w, h))
            
            augmented_frames.append(translated)
            
            # Update trajectories if provided
            if augmented_trajectories and i < len(augmented_trajectories):
                augmented_trajectories[i]['x'] += tx
                augmented_trajectories[i]['y'] += ty
        
        return augmented_frames, augmented_trajectories
    
    def _add_camera_shake(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Add camera shake to video frames."""
        augmented_frames = []
        
        for i, frame in enumerate(frames):
            # Random shake for each frame
            dx = random.uniform(-self.config.camera_shake_amplitude, self.config.camera_shake_amplitude)
            dy = random.uniform(-self.config.camera_shake_amplitude, self.config.camera_shake_amplitude)
            
            # Apply shake
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            h, w = frame.shape[:2]
            shaken = cv2.warpAffine(frame, M, (w, h))
            
            augmented_frames.append(shaken)
        
        return augmented_frames
    
    def _add_temporal_occlusion(
        self,
        frames: List[np.ndarray],
        start_frame: int,
        duration: int
    ) -> List[np.ndarray]:
        """Add temporal occlusion to specific frames."""
        augmented_frames = frames.copy()
        
        for i in range(start_frame, min(start_frame + duration, len(frames))):
            h, w = frames[i].shape[:2]
            
            # Create occlusion
            occ_h = random.randint(h // 3, h // 2)
            occ_w = random.randint(w // 3, w // 2)
            occ_x = random.randint(0, w - occ_w)
            occ_y = random.randint(0, h - occ_h)
            
            # Apply occlusion
            augmented_frames[i][occ_y:occ_y+occ_h, occ_x:occ_x+occ_w] = 0
        
        return augmented_frames
    
    def _add_frame_noise(self, frame: np.ndarray) -> np.ndarray:
        """Add frame-specific noise."""
        noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson'])
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 20, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, noise)
        elif noise_type == 'salt_pepper':
            noise = np.random.random(frame.shape)
            noisy_frame = frame.copy()
            noisy_frame[noise < 0.01] = 0
            noisy_frame[noise > 0.99] = 255
        else:  # poisson
            noisy_frame = np.random.poisson(frame).astype(np.uint8)
        
        return noisy_frame


class StressTestDatasetCreator:
    """Create stress test datasets for different visual operations."""
    
    def __init__(
        self,
        source_data_path: str,
        output_path: str,
        augmenter: StressTestAugmenter = None
    ):
        self.source_data_path = Path(source_data_path)
        self.output_path = Path(output_path)
        self.augmenter = augmenter or StressTestAugmenter()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_segmentation_stress_test(
        self,
        num_samples: int = 1000,
        severities: List[str] = ["easy", "medium", "hard"]
    ):
        """Create stress test dataset for SEGMENT_OBJECT_AT and GET_PROPERTIES."""
        self.logger.info("Creating segmentation stress test dataset")
        
        output_dir = self.output_path / "segmentation_stress"
        output_dir.mkdir(exist_ok=True)
        
        # Load source data
        source_images = self._load_source_images("segmentation")
        
        for severity in severities:
            severity_dir = output_dir / severity
            severity_dir.mkdir(exist_ok=True)
            
            metadata = []
            
            for i in tqdm(range(num_samples), desc=f"Creating {severity} samples"):
                # Select random source image
                source_img, source_mask = random.choice(source_images)
                
                # Apply augmentations
                aug_img, aug_mask = self.augmenter.augment_for_segmentation(
                    source_img, source_mask, severity
                )
                
                # Save augmented data
                img_path = severity_dir / f"image_{i:05d}.png"
                cv2.imwrite(str(img_path), aug_img)
                
                if aug_mask is not None:
                    mask_path = severity_dir / f"mask_{i:05d}.png"
                    cv2.imwrite(str(mask_path), aug_mask)
                else:
                    mask_path = None
                
                # Create metadata entry
                metadata.append({
                    "id": f"{severity}_{i:05d}",
                    "image": str(img_path.relative_to(self.output_path)),
                    "mask": str(mask_path.relative_to(self.output_path)) if mask_path else None,
                    "severity": severity,
                    "source_hash": hashlib.md5(source_img.tobytes()).hexdigest()[:8]
                })
            
            # Save metadata
            metadata_path = severity_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created {num_samples} {severity} segmentation samples")
    
    def create_text_reading_stress_test(
        self,
        num_samples: int = 1000,
        severities: List[str] = ["easy", "medium", "hard"]
    ):
        """Create stress test dataset for READ_TEXT."""
        self.logger.info("Creating text reading stress test dataset")
        
        output_dir = self.output_path / "text_reading_stress"
        output_dir.mkdir(exist_ok=True)
        
        # Load or generate source text images
        source_images = self._load_or_generate_text_images()
        
        for severity in severities:
            severity_dir = output_dir / severity
            severity_dir.mkdir(exist_ok=True)
            
            metadata = []
            
            for i in tqdm(range(num_samples), desc=f"Creating {severity} samples"):
                # Select random source image
                source_img, text_regions, ground_truth_text = random.choice(source_images)
                
                # Apply augmentations
                aug_img, aug_regions = self.augmenter.augment_for_text_reading(
                    source_img, text_regions, severity
                )
                
                # Save augmented data
                img_path = severity_dir / f"image_{i:05d}.png"
                cv2.imwrite(str(img_path), aug_img)
                
                # Create metadata entry
                metadata.append({
                    "id": f"{severity}_{i:05d}",
                    "image": str(img_path.relative_to(self.output_path)),
                    "text_regions": aug_regions,
                    "ground_truth_text": ground_truth_text,
                    "severity": severity
                })
            
            # Save metadata
            metadata_path = severity_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created {num_samples} {severity} text reading samples")
    
    def create_tracking_stress_test(
        self,
        num_sequences: int = 100,
        frames_per_sequence: int = 30,
        severities: List[str] = ["easy", "medium", "hard"]
    ):
        """Create stress test dataset for TRACK_OBJECT."""
        self.logger.info("Creating tracking stress test dataset")
        
        output_dir = self.output_path / "tracking_stress"
        output_dir.mkdir(exist_ok=True)
        
        # Load or generate source video sequences
        source_sequences = self._load_or_generate_video_sequences(
            num_sequences, frames_per_sequence
        )
        
        for severity in severities:
            severity_dir = output_dir / severity
            severity_dir.mkdir(exist_ok=True)
            
            metadata = []
            
            for seq_idx in tqdm(range(num_sequences), desc=f"Creating {severity} sequences"):
                # Select random source sequence
                source_frames, trajectories = source_sequences[seq_idx]
                
                # Apply augmentations
                aug_frames, aug_trajectories = self.augmenter.augment_for_tracking(
                    source_frames, trajectories, severity
                )
                
                # Save augmented sequence
                seq_dir = severity_dir / f"sequence_{seq_idx:04d}"
                seq_dir.mkdir(exist_ok=True)
                
                frame_paths = []
                for frame_idx, frame in enumerate(aug_frames):
                    frame_path = seq_dir / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(str(frame_path.relative_to(self.output_path)))
                
                # Create metadata entry
                metadata.append({
                    "id": f"{severity}_seq_{seq_idx:04d}",
                    "frames": frame_paths,
                    "trajectories": aug_trajectories,
                    "severity": severity,
                    "num_frames": len(aug_frames)
                })
            
            # Save metadata
            metadata_path = severity_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Created {num_sequences} {severity} tracking sequences")
    
    def _load_source_images(self, task_type: str) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Load source images for augmentation."""
        # Placeholder - in real implementation, load from actual dataset
        images = []
        
        for _ in range(100):  # Generate synthetic images
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            if task_type == "segmentation":
                mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
                images.append((img, mask))
            else:
                images.append((img, None))
        
        return images
    
    def _load_or_generate_text_images(self) -> List[Tuple[np.ndarray, List[Dict], str]]:
        """Load or generate text images."""
        images = []
        
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine Learning and Computer Vision",
            "IMPORTANT: Read Instructions Carefully",
            "Error Code: 0x80070057",
            "今日は良い天気ですね",  # Japanese
            "Привет, мир!",  # Russian
            "مرحبا بالعالم",  # Arabic
        ]
        
        for text in texts:
            # Create synthetic text image
            img = np.ones((100, 600, 3), dtype=np.uint8) * 255
            
            # Add text (simplified - in real implementation use proper text rendering)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text[:30], (10, 50), font, 1, (0, 0, 0), 2)
            
            # Create text region
            text_regions = [{
                'bbox': [10, 20, 580, 60],
                'text': text
            }]
            
            images.append((img, text_regions, text))
        
        return images * 20  # Repeat to have more samples
    
    def _load_or_generate_video_sequences(
        self,
        num_sequences: int,
        frames_per_sequence: int
    ) -> List[Tuple[List[np.ndarray], List[Dict]]]:
        """Load or generate video sequences."""
        sequences = []
        
        for _ in range(num_sequences):
            frames = []
            trajectories = []
            
            # Generate synthetic moving object
            h, w = 480, 640
            obj_x, obj_y = 100, 100
            vel_x, vel_y = random.uniform(5, 15), random.uniform(-5, 5)
            
            for frame_idx in range(frames_per_sequence):
                # Create frame
                frame = np.ones((h, w, 3), dtype=np.uint8) * 200
                
                # Update object position
                obj_x += vel_x
                obj_y += vel_y
                
                # Bounce off edges
                if obj_x < 0 or obj_x > w - 50:
                    vel_x = -vel_x
                if obj_y < 0 or obj_y > h - 50:
                    vel_y = -vel_y
                
                obj_x = max(0, min(w - 50, obj_x))
                obj_y = max(0, min(h - 50, obj_y))
                
                # Draw object
                cv2.rectangle(frame, (int(obj_x), int(obj_y)), 
                            (int(obj_x + 50), int(obj_y + 50)), (0, 255, 0), -1)
                
                frames.append(frame)
                trajectories.append({
                    'frame': frame_idx,
                    'x': obj_x,
                    'y': obj_y,
                    'w': 50,
                    'h': 50
                })
            
            sequences.append((frames, trajectories))
        
        return sequences


def main():
    """Main function to create stress test datasets."""
    parser = argparse.ArgumentParser(description="Create stress test datasets")
    parser.add_argument(
        "--source-data",
        type=str,
        default="data/custom_benchmark",
        help="Path to source data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stress_test",
        help="Output directory for stress test data"
    )
    parser.add_argument(
        "--task",
        choices=["segmentation", "text", "tracking", "all"],
        default="all",
        help="Which task to create stress test for"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples per severity level"
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Severity levels to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize augmenter
    config = AugmentationConfig()
    augmenter = StressTestAugmenter(config)
    
    # Initialize dataset creator
    creator = StressTestDatasetCreator(
        args.source_data,
        args.output_dir,
        augmenter
    )
    
    # Create stress test datasets
    if args.task in ["segmentation", "all"]:
        creator.create_segmentation_stress_test(
            num_samples=args.num_samples,
            severities=args.severities
        )
    
    if args.task in ["text", "all"]:
        creator.create_text_reading_stress_test(
            num_samples=args.num_samples,
            severities=args.severities
        )
    
    if args.task in ["tracking", "all"]:
        creator.create_tracking_stress_test(
            num_sequences=args.num_samples // 10,  # Fewer sequences
            frames_per_sequence=30,
            severities=args.severities
        )
    
    print(f"\nStress test datasets created in {args.output_dir}")


if __name__ == "__main__":
    main()