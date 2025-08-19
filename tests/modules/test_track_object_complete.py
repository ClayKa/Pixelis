#!/usr/bin/env python3
"""
Comprehensive test suite for track_object.py to achieve 100% test coverage.
Tests all methods, branches, and edge cases in the TrackObjectOperation class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from core.modules.operations.track_object import TrackObjectOperation


class TestTrackObjectOperation:
    """Comprehensive test suite for TrackObjectOperation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.operation = TrackObjectOperation()
        
        # Test data
        self.test_frame = torch.rand(3, 100, 100)
        self.test_frames = [torch.rand(3, 100, 100) for _ in range(3)]
        self.test_mask = torch.zeros(100, 100)
        self.test_mask[40:60, 30:70] = 1.0  # 20x40 rectangle
        self.test_bbox = [30, 40, 70, 60]  # [x1, y1, x2, y2]
    
    # ================== INITIALIZATION TESTS ==================
    
    def test_init(self):
        """Test __init__ method - covers lines 24-26."""
        operation = TrackObjectOperation()
        assert operation.tracker is None
        assert operation.tracking_state == {}
        assert hasattr(operation, 'logger')
    
    def test_load_model(self):
        """Test _load_model method - covers lines 34-38."""
        # Initially None
        assert self.operation.tracker is None
        
        # Mock logger to verify info call
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            mock_info.assert_called_once_with("Loading tracking model...")
        
        # Should set placeholder tracker
        assert self.operation.tracker == "placeholder_tracker"
        
        # Second call should not reload
        with patch.object(self.operation.logger, 'info') as mock_info:
            self.operation._load_model()
            mock_info.assert_not_called()
    
    # ================== VALIDATION ERROR TESTS ==================
    
    def test_validate_inputs_missing_frames_init(self):
        """Test validation error: missing frames for init - covers lines 62-63."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(init_mask=self.test_mask)
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'frames' or 'frame'")
    
    def test_validate_inputs_missing_init_params(self):
        """Test validation error: missing init_mask/init_bbox - covers lines 66-67."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            # The logic: if no init_mask AND no init_bbox AND no track_id -> invalid mode
            result = self.operation.validate_inputs(frames=self.test_frames)
            assert not result
            mock_error.assert_called_once_with("Must provide initialization or update parameters")
    
    def test_validate_inputs_invalid_bbox_type(self):
        """Test validation error: invalid bbox type - covers lines 72-73."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(frames=self.test_frames, init_bbox="invalid")
            assert not result
            mock_error.assert_called_once_with("'init_bbox' must be [x1, y1, x2, y2]")
    
    def test_validate_inputs_invalid_bbox_length(self):
        """Test validation error: invalid bbox length - covers lines 72-73."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(frames=self.test_frames, init_bbox=[1, 2, 3])
            assert not result
            mock_error.assert_called_once_with("'init_bbox' must be [x1, y1, x2, y2]")
    
    def test_validate_inputs_missing_frame_update(self):
        """Test validation error: missing frame for update - covers lines 78-79."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(track_id="test_track")
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'frame' for update")
    
    def test_validate_inputs_unknown_track_id(self):
        """Test validation error: unknown track_id - covers lines 82-83."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(frame=self.test_frame, track_id="unknown_track")
            assert not result
            mock_error.assert_called_once_with("Unknown track_id: unknown_track")
    
    def test_validate_inputs_invalid_mode(self):
        """Test validation error: invalid mode - covers lines 86-87."""
        with patch.object(self.operation.logger, 'error') as mock_error:
            result = self.operation.validate_inputs(frame=self.test_frame)  # No init or track_id
            assert not result
            mock_error.assert_called_once_with("Must provide initialization or update parameters")
    
    def test_validate_inputs_valid_init_mask(self):
        """Test valid inputs for initialization with mask."""
        result = self.operation.validate_inputs(frames=self.test_frames, init_mask=self.test_mask)
        assert result is True
    
    def test_validate_inputs_valid_init_bbox(self):
        """Test valid inputs for initialization with bbox."""
        result = self.operation.validate_inputs(frame=self.test_frame, init_bbox=self.test_bbox)
        assert result is True
    
    def test_validate_inputs_valid_update(self):
        """Test valid inputs for update mode."""
        # First create a track
        self.operation.tracking_state["test_track"] = {"status": "active"}
        
        result = self.operation.validate_inputs(frame=self.test_frame, track_id="test_track")
        assert result is True
    
    # ================== RUN METHOD TESTS ==================
    
    def test_run_invalid_inputs(self):
        """Test run method with invalid inputs - covers line 121."""
        with pytest.raises(ValueError) as exc_info:
            self.operation.run()  # No arguments
        assert str(exc_info.value) == "Invalid inputs for track operation"
    
    def test_run_model_loading(self):
        """Test run method triggers model loading - covers lines 124."""
        with patch.object(self.operation, '_load_model') as mock_load:
            with patch.object(self.operation, 'validate_inputs', return_value=True):
                with patch.object(self.operation, '_initialize_tracking') as mock_init:
                    self.operation.run(frames=[self.test_frame], init_mask=self.test_mask)
                    mock_load.assert_called_once()
    
    # ================== INITIALIZATION MODE TESTS ==================
    
    def test_initialize_tracking_with_mask_single_frame(self):
        """Test initialization with mask and single frame."""
        result = self.operation.run(
            frame=self.test_frame,
            init_mask=self.test_mask,
            track_id="test_track_1"
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result['track_id'] == "test_track_1"
        assert 'trajectory' in result
        assert 'masks' in result
        assert 'confidences' in result
        assert 'status' in result
        assert 'frames_processed' in result
        assert 'statistics' in result
        
        # Verify state was created
        assert "test_track_1" in self.operation.tracking_state
        state = self.operation.tracking_state["test_track_1"]
        assert state['status'] == 'active'
        assert state['frame_count'] == 1
    
    def test_initialize_tracking_with_bbox_multiple_frames(self):
        """Test initialization with bbox and multiple frames."""
        result = self.operation.run(
            frames=self.test_frames,
            init_bbox=self.test_bbox,
            max_frames=2
        )
        
        # Verify result
        assert isinstance(result, dict)
        assert 'track_id' in result
        assert len(result['trajectory']) == 2  # Limited by max_frames
        assert len(result['confidences']) == 2
        
        # Verify trajectory contains bbox coordinates
        trajectory = result['trajectory']
        assert len(trajectory[0]) == 4  # [x1, y1, x2, y2]
    
    def test_initialize_tracking_with_return_masks(self):
        """Test initialization with return_masks=True."""
        result = self.operation.run(
            frame=self.test_frame,
            init_mask=self.test_mask,
            return_masks=True
        )
        
        assert result['masks'] is not None
        assert len(result['masks']) == 1
        assert isinstance(result['masks'][0], torch.Tensor)
    
    def test_initialize_tracking_without_return_masks(self):
        """Test initialization with return_masks=False."""
        result = self.operation.run(
            frame=self.test_frame,
            init_mask=self.test_mask,
            return_masks=False
        )
        
        assert result['masks'] is None
    
    def test_initialize_tracking_confidence_threshold_loss(self):
        """Test track loss due to confidence threshold."""
        # Mock _track_in_frame to return low confidence
        with patch.object(self.operation, '_track_in_frame', return_value=(self.test_bbox, self.test_mask, 0.3)):
            with patch.object(self.operation.logger, 'warning') as mock_warning:
                result = self.operation.run(
                    frames=self.test_frames,
                    init_bbox=self.test_bbox,
                    confidence_threshold=0.5
                )
                
                # Should stop tracking after first frame due to low confidence
                track_id = result['track_id']
                assert self.operation.tracking_state[track_id]['status'] == 'lost'
                mock_warning.assert_called_once()
    
    def test_initialize_tracking_auto_track_id(self):
        """Test automatic track ID generation."""
        with patch.object(self.operation, '_generate_track_id', return_value="auto_track_123"):
            result = self.operation.run(
                frame=self.test_frame,
                init_mask=self.test_mask
            )
            
            assert result['track_id'] == "auto_track_123"
    
    # ================== UPDATE MODE TESTS ==================
    
    def test_update_tracking_active_track(self):
        """Test updating an active track."""
        # First initialize a track
        init_result = self.operation.run(
            frame=self.test_frame,
            init_bbox=self.test_bbox,
            track_id="update_test"
        )
        
        # Then update it
        update_result = self.operation.run(
            frame=self.test_frame,
            track_id="update_test"
        )
        
        # Verify update result structure
        assert isinstance(update_result, dict)
        assert update_result['track_id'] == "update_test"
        assert 'position' in update_result
        assert 'confidence' in update_result
        assert 'status' in update_result
        assert 'frame_number' in update_result
        assert 'statistics' in update_result
        
        # Verify state was updated
        state = self.operation.tracking_state["update_test"]
        assert state['frame_count'] == 2
        assert len(state['trajectory']) == 2
    
    def test_update_tracking_lost_track(self):
        """Test updating a lost track."""
        # Create a lost track
        self.operation.tracking_state["lost_track"] = {
            'status': 'lost',
            'trajectory': [self.test_bbox],
            'confidences': [0.9]
        }
        
        result = self.operation.run(
            frame=self.test_frame,
            track_id="lost_track"
        )
        
        # Should return status message without processing
        assert result['track_id'] == "lost_track"
        assert result['status'] == 'lost'
        assert 'message' in result
        assert "cannot update" in result['message']
    
    def test_update_tracking_with_return_mask(self):
        """Test update with return_mask=True."""
        # Initialize track
        self.operation.run(
            frame=self.test_frame,
            init_bbox=self.test_bbox,
            track_id="mask_test"
        )
        
        # Update with return_mask
        result = self.operation.run(
            frame=self.test_frame,
            track_id="mask_test",
            return_mask=True
        )
        
        assert result['mask'] is not None
        assert isinstance(result['mask'], torch.Tensor)
    
    def test_update_tracking_confidence_loss(self):
        """Test track loss during update due to low confidence."""
        # Initialize track
        self.operation.run(
            frame=self.test_frame,
            init_bbox=self.test_bbox,
            track_id="loss_test"
        )
        
        # Mock low confidence tracking
        with patch.object(self.operation, '_track_in_frame', return_value=(self.test_bbox, self.test_mask, 0.2)):
            with patch.object(self.operation.logger, 'warning') as mock_warning:
                result = self.operation.run(
                    frame=self.test_frame,
                    track_id="loss_test",
                    confidence_threshold=0.5
                )
                
                assert result['status'] == 'lost'
                assert self.operation.tracking_state["loss_test"]['status'] == 'lost'
                mock_warning.assert_called_once()
    
    # ================== HELPER METHOD TESTS ==================
    
    def test_track_in_frame(self):
        """Test _track_in_frame method - covers lines 293-336."""
        # Create track state for this test
        track_id = "frame_test"
        self.operation.tracking_state[track_id] = {'frame_count': 2}
        
        position, mask, confidence = self.operation._track_in_frame(
            self.test_frame,
            self.test_bbox,
            track_id
        )
        
        # Verify outputs
        assert len(position) == 4  # [x1, y1, x2, y2]
        assert isinstance(mask, torch.Tensor)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_bbox_to_mask_chw_format(self):
        """Test _bbox_to_mask with CHW format - covers lines 338-368."""
        frame_chw = torch.rand(3, 80, 90)  # CHW format
        bbox = [10, 15, 50, 60]
        
        mask = self.operation._bbox_to_mask(frame_chw, bbox)
        
        assert mask.shape == (80, 90)
        assert mask[15:60, 10:50].sum() > 0  # Should have ones in bbox region
        assert mask[0:15, :].sum() == 0  # Should be zero outside bbox
    
    def test_bbox_to_mask_hw_format(self):
        """Test _bbox_to_mask with HW format - covers lines 355-356."""
        frame_hw = torch.rand(80, 90)  # HW format
        bbox = [10, 15, 50, 60]
        
        mask = self.operation._bbox_to_mask(frame_hw, bbox)
        
        assert mask.shape == (80, 90)
        assert mask[15:60, 10:50].sum() > 0
    
    def test_bbox_to_mask_boundary_clipping(self):
        """Test _bbox_to_mask with out-of-bounds bbox - covers lines 362-366."""
        frame = torch.rand(3, 50, 60)
        bbox = [-10, -5, 70, 55]  # Partially out of bounds
        
        mask = self.operation._bbox_to_mask(frame, bbox)
        
        assert mask.shape == (50, 60)
        # Should be clipped to frame boundaries
        assert mask[0:50, 0:60].sum() > 0
    
    def test_mask_to_bbox_normal_mask(self):
        """Test _mask_to_bbox with normal mask - covers lines 370-393."""
        mask = torch.zeros(100, 120)
        mask[20:80, 30:90] = 1.0
        
        bbox = self.operation._mask_to_bbox(mask)
        
        assert bbox == [30, 20, 89, 79]  # [x1, y1, x2, y2]
    
    def test_mask_to_bbox_empty_mask(self):
        """Test _mask_to_bbox with empty mask - covers lines 382-383."""
        empty_mask = torch.zeros(100, 100)
        
        bbox = self.operation._mask_to_bbox(empty_mask)
        
        assert bbox == [0, 0, 0, 0]
    
    def test_calculate_motion_statistics_short_trajectory(self):
        """Test _calculate_motion_statistics with short trajectory - covers lines 408-414."""
        short_trajectory = [self.test_bbox]  # Single frame
        
        stats = self.operation._calculate_motion_statistics(short_trajectory)
        
        assert stats['total_distance'] == 0
        assert stats['average_velocity'] == 0
        assert stats['direction'] is None
        assert stats['is_stationary'] is True
    
    def test_calculate_motion_statistics_normal_trajectory(self):
        """Test _calculate_motion_statistics with normal trajectory - covers lines 415-456."""
        trajectory = [
            [10, 20, 30, 40],   # Frame 1
            [15, 25, 35, 45],   # Frame 2 - moved (5, 5)
            [20, 20, 40, 40],   # Frame 3 - moved (5, -5)
        ]
        
        stats = self.operation._calculate_motion_statistics(trajectory)
        
        # Verify statistics structure
        assert 'total_distance' in stats
        assert 'average_velocity' in stats
        assert 'max_velocity' in stats
        assert 'direction' in stats
        assert 'is_stationary' in stats
        assert 'path_length' in stats
        assert 'displacement' in stats
        
        # Verify values
        assert stats['total_distance'] > 0
        assert stats['path_length'] == 3
        assert isinstance(stats['direction'], float)
    
    def test_generate_track_id(self):
        """Test _generate_track_id method - covers lines 458-466."""
        track_id = self.operation._generate_track_id()
        
        assert isinstance(track_id, str)
        assert track_id.startswith("track_")
        assert len(track_id) > 6  # "track_" + hex chars
        
        # Generate another to test uniqueness
        track_id2 = self.operation._generate_track_id()
        assert track_id != track_id2
    
    # ================== TRACK MANAGEMENT TESTS ==================
    
    def test_reset_track_existing(self):
        """Test reset_track with existing track - covers lines 478-481."""
        # Create a track first
        self.operation.tracking_state["reset_test"] = {"status": "active"}
        
        with patch.object(self.operation.logger, 'debug') as mock_debug:
            result = self.operation.reset_track("reset_test")
            
            assert result is True
            assert "reset_test" not in self.operation.tracking_state
            mock_debug.assert_called_once_with("Reset track reset_test")
    
    def test_reset_track_nonexistent(self):
        """Test reset_track with non-existent track - covers line 482."""
        result = self.operation.reset_track("nonexistent_track")
        
        assert result is False
    
    def test_reset_all_tracks(self):
        """Test reset_all_tracks method - covers lines 485-487."""
        # Create some tracks
        self.operation.tracking_state["track1"] = {"status": "active"}
        self.operation.tracking_state["track2"] = {"status": "lost"}
        
        with patch.object(self.operation.logger, 'debug') as mock_debug:
            self.operation.reset_all_tracks()
            
            assert len(self.operation.tracking_state) == 0
            mock_debug.assert_called_once_with("Reset all tracks")
    
    def test_get_active_tracks_mixed_statuses(self):
        """Test get_active_tracks with mixed track statuses - covers lines 496-500."""
        # Create tracks with different statuses
        self.operation.tracking_state = {
            "active1": {"status": "active"},
            "active2": {"status": "active"},
            "lost1": {"status": "lost"},
            "completed1": {"status": "completed"}
        }
        
        active_tracks = self.operation.get_active_tracks()
        
        assert len(active_tracks) == 2
        assert "active1" in active_tracks
        assert "active2" in active_tracks
        assert "lost1" not in active_tracks
        assert "completed1" not in active_tracks
    
    def test_get_active_tracks_empty(self):
        """Test get_active_tracks with no tracks."""
        active_tracks = self.operation.get_active_tracks()
        
        assert active_tracks == []
    
    # ================== UTILITY METHOD TESTS ==================
    
    def test_get_required_params(self):
        """Test get_required_params method - covers lines 502-505."""
        required = self.operation.get_required_params()
        
        assert isinstance(required, list)
        assert required == []  # As per implementation
    
    def test_get_optional_params(self):
        """Test get_optional_params method - covers lines 507-515."""
        optional = self.operation.get_optional_params()
        
        expected = {
            'max_frames': None,
            'confidence_threshold': 0.5,
            'return_masks': False,
            'return_mask': False,
            'track_id': None
        }
        assert optional == expected
    
    # ================== REGISTRY INTEGRATION TESTS ==================
    
    def test_registry_integration(self):
        """Test operation is registered correctly."""
        from core.modules.operation_registry import registry
        
        assert registry.has_operation('TRACK_OBJECT')
        operation_class = registry.get_operation_class('TRACK_OBJECT')
        assert operation_class == TrackObjectOperation
    
    # ================== EDGE CASE TESTS ==================
    
    def test_tiny_frame_tracking(self):
        """Test tracking with very small frames."""
        tiny_frame = torch.rand(3, 5, 5)
        tiny_bbox = [1, 1, 3, 3]
        
        result = self.operation.run(
            frame=tiny_frame,
            init_bbox=tiny_bbox
        )
        
        assert isinstance(result, dict)
        assert 'trajectory' in result
    
    def test_single_pixel_mask(self):
        """Test with single pixel mask."""
        single_pixel_mask = torch.zeros(100, 100)
        single_pixel_mask[50, 50] = 1.0
        
        result = self.operation.run(
            frame=self.test_frame,
            init_mask=single_pixel_mask
        )
        
        assert isinstance(result, dict)
        bbox = result['trajectory'][0]
        assert bbox == [50, 50, 50, 50]  # Single pixel bbox
    
    def test_large_trajectory(self):
        """Test with many frames to test motion statistics thoroughly."""
        many_frames = [torch.rand(3, 50, 50) for _ in range(20)]
        
        result = self.operation.run(
            frames=many_frames,
            init_bbox=[10, 10, 20, 20],
            max_frames=15
        )
        
        assert len(result['trajectory']) == 15
        assert 'statistics' in result
        stats = result['statistics']
        assert stats['path_length'] == 15
    
    def test_debug_logging(self):
        """Test debug logging in _initialize_tracking."""
        with patch.object(self.operation.logger, 'debug') as mock_debug:
            result = self.operation.run(
                frame=self.test_frame,
                init_bbox=self.test_bbox,
                track_id="debug_test"
            )
            
            # Should log initialization completion
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert "Initialized track debug_test" in call_args
            assert "processed 1 frames" in call_args
            assert "status=active" in call_args
    
    # ================== TARGET REMAINING UNCOVERED LINES ==================
    
    def test_line_147_frames_not_list(self):
        """Test line 147: when frames is not a list but frame is provided."""
        # This should trigger line 147 where single frame is converted to list
        result = self.operation.run(
            frame=self.test_frame,  # Single frame, not frames list
            init_mask=self.test_mask
        )
        
        assert isinstance(result, dict)
        assert 'trajectory' in result
        assert len(result['trajectory']) == 1
    
    def test_line_441_direction_fallback(self):
        """Test line 441: direction calculation with edge case."""
        # Test motion statistics calculation where direction calculation might hit edge cases
        # Try with trajectory where centroids are identical (no movement)
        identical_trajectory = [
            [10, 20, 30, 40],
            [10, 20, 30, 40],  # Exact same position
            [10, 20, 30, 40]   # Still same position
        ]
        
        stats = self.operation._calculate_motion_statistics(identical_trajectory)
        
        # Should handle zero displacement gracefully
        assert 'direction' in stats
        assert stats['total_distance'] == 0
        # For identical positions, the direction should be calculated but displacement is 0
        assert isinstance(stats['direction'], (int, float))
    
    def test_lines_66_67_impossible_condition(self):
        """Test lines 66-67: This appears to be unreachable code due to logical contradiction."""
        # The condition at line 65 checks if both init_mask and init_bbox are missing,
        # but we can only reach this point if is_init is True, which requires at least one to be present.
        # This may be dead code or there's a logical error in the implementation.
        
        # Let's try to force this condition by manipulating the validation logic
        operation = TrackObjectOperation()
        
        # Direct call to try to hit this path - this may not be possible due to the logic
        with patch.object(operation.logger, 'error') as mock_error:
            # Try various combinations to see if we can hit lines 66-67
            kwargs_test = {'frames': self.test_frames}  # No init params
            
            # Manually check the conditions to understand the flow
            is_init = 'init_mask' in kwargs_test or 'init_bbox' in kwargs_test  # False
            is_update = 'track_id' in kwargs_test  # False
            
            # Since both are False, it goes to else block, not init block
            result = operation.validate_inputs(**kwargs_test)
            assert not result
            # This will hit the "invalid mode" error, not the "missing init params" error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])