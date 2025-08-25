# tests/dataloaders/test_mot_sliding_window_loader.py

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from core.dataloaders.mot_sliding_window_loader import (
    MotSlidingWindowLoader,
    create_mot_loader
)


class TestMotSlidingWindowLoader:
    """Test suite for MOT sliding window loader."""
    
    @pytest.fixture
    def create_mot_dataset(self):
        """Create a mock MOT dataset structure."""
        def _create(num_sequences=2, frames_per_sequence=500):
            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = Path(tmpdir)
                
                sequences = []
                for seq_idx in range(num_sequences):
                    seq_name = f"MOT20-{seq_idx+1:02d}"
                    seq_path = base_path / seq_name
                    seq_path.mkdir()
                    
                    # Create img1 directory with mock images
                    img_dir = seq_path / "img1"
                    img_dir.mkdir()
                    
                    for frame_idx in range(1, frames_per_sequence + 1):
                        img_file = img_dir / f"{frame_idx:06d}.jpg"
                        img_file.write_bytes(b"fake image")
                    
                    # Create gt directory and ground truth file
                    gt_dir = seq_path / "gt"
                    gt_dir.mkdir()
                    
                    # Generate mock tracking data
                    gt_data = []
                    for frame_idx in range(1, frames_per_sequence + 1):
                        # Create 3 objects per frame
                        for obj_id in range(1, 4):
                            gt_data.append([
                                frame_idx,  # frame_id
                                obj_id,     # object_id
                                100 * obj_id,  # bb_left
                                100 * obj_id,  # bb_top
                                50,         # bb_width
                                50,         # bb_height
                                0.9,        # conf
                                -1,         # x
                                -1,         # y
                                -1          # z
                            ])
                    
                    # Write ground truth
                    gt_df = pd.DataFrame(gt_data)
                    gt_df.to_csv(gt_dir / "gt.txt", header=False, index=False)
                    
                    sequences.append(seq_path)
                
                yield base_path, sequences
        
        return _create
    
    def test_full_sequence_mode(self, create_mot_dataset):
        """Test traditional full sequence loading."""
        with create_mot_dataset(2, 300) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'full_sequence'
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            
            # Should have one sample per sequence
            assert len(loader) == 2
            
            # Get a sample
            sample = loader.get_item(0)
            assert 'clip_info' in sample
            assert sample['clip_info']['duration_frames'] == 300
    
    def test_sliding_window_mode(self, create_mot_dataset):
        """Test sliding window sampling."""
        with create_mot_dataset(2, 500) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 100,
                    'stride_frames': 50,  # 50% overlap
                    'min_clip_frames': 50
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            
            # Should have multiple clips per sequence
            # For 500 frames with 100-frame clips and 50-frame stride:
            # Clips start at: 1, 51, 101, 151, 201, 251, 301, 351, 401, 451
            # Last full clip starts at 401 (frames 401-500)
            # So we expect about 9 clips per sequence
            assert len(loader) > 2  # More than original sequences
            assert len(loader) <= 20  # But reasonable number
            
            # Check clip statistics
            stats = loader.get_clip_statistics()
            assert stats['sampling_type'] == 'sliding_window'
            assert stats['total_sequences'] == 2
            assert stats['config']['overlap_ratio'] == 0.5
    
    def test_clip_content_filtering(self, create_mot_dataset):
        """Test that clips only contain annotations from their frame range."""
        with create_mot_dataset(1, 300) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 100,
                    'stride_frames': 100,  # No overlap
                    'min_clip_frames': 50
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            
            # Get first clip (frames 1-100)
            clip1 = loader.get_item(0)
            
            # Get second clip (frames 101-200)
            clip2 = loader.get_item(1)
            
            # Check that clips have different frame ranges
            assert clip1['clip_info']['start_frame'] == 1
            assert clip1['clip_info']['end_frame'] == 100
            assert clip2['clip_info']['start_frame'] == 101
            assert clip2['clip_info']['end_frame'] == 200
            
            # Check that trajectories are properly filtered
            trajectories1 = clip1['annotations']['tracking']['trajectories']
            trajectories2 = clip2['annotations']['tracking']['trajectories']
            
            # All frame indices in clip1 should be 0-99 (normalized)
            for traj in trajectories1:
                for point in traj['trajectory']:
                    assert 0 <= point['frame'] < 100
            
            # All frame indices in clip2 should also be 0-99 (normalized from 101-200)
            for traj in trajectories2:
                for point in traj['trajectory']:
                    assert 0 <= point['frame'] < 100
    
    def test_minimum_clip_length(self, create_mot_dataset):
        """Test that clips shorter than minimum are excluded."""
        with create_mot_dataset(1, 150) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 100,
                    'stride_frames': 100,
                    'min_clip_frames': 80  # High minimum
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            
            # With 150 frames total and 100-frame stride:
            # Clip 1: frames 1-100 (100 frames) ✓
            # Clip 2: frames 101-150 (50 frames) ✗ (below minimum)
            # So only 1 clip should be created
            assert len(loader) == 1
    
    def test_sample_id_generation(self, create_mot_dataset):
        """Test that sample IDs include clip information."""
        with create_mot_dataset(1, 200) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 100,
                    'stride_frames': 100
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            
            # Get samples
            sample1 = loader.get_item(0)
            sample2 = loader.get_item(1)
            
            # Check sample IDs are unique and informative
            assert sample1['sample_id'] != sample2['sample_id']
            assert 'clip' in sample1['sample_id']
            assert 'clip' in sample2['sample_id']
    
    def test_factory_function(self, create_mot_dataset):
        """Test the factory function for creating loaders."""
        with create_mot_dataset(1, 100) as (base_path, sequences):
            # Test sliding window config
            config_sliding = {
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 50
                }
            }
            
            loader = create_mot_loader(config_sliding)
            assert isinstance(loader, MotSlidingWindowLoader)
            
            # Test full sequence config (default)
            config_full = {
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'full_sequence'
                }
            }
            
            loader = create_mot_loader(config_full)
            # Should return appropriate loader type
            assert loader is not None
    
    def test_statistics_reporting(self, create_mot_dataset):
        """Test clip statistics reporting."""
        with create_mot_dataset(3, 400) as (base_path, sequences):
            config = {
                'name': 'mot_test',
                'path': str(base_path),
                'sampling_strategy': {
                    'type': 'sliding_window',
                    'clip_duration_frames': 150,
                    'stride_frames': 75  # 50% overlap
                }
            }
            
            loader = MotSlidingWindowLoader(config)
            stats = loader.get_clip_statistics()
            
            assert 'total_clips' in stats
            assert 'clips_per_sequence' in stats
            assert 'mean' in stats['clips_per_sequence']
            assert stats['clips_per_sequence']['mean'] > 1  # Multiple clips per sequence
            assert stats['config']['overlap_ratio'] == 0.5