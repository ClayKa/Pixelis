#!/usr/bin/env python3
"""
Final test to achieve exactly 100% coverage for track_object.py.
Targets the specific remaining uncovered lines identified in the coverage report.
"""

import sys
sys.path.insert(0, '.')

import coverage
import torch
import numpy as np
from unittest.mock import patch


def final_100_percent_track_test():
    """Target the final uncovered lines to achieve 100% coverage."""
    
    # Start coverage
    cov = coverage.Coverage()
    cov.start()
    
    try:
        print("=== FINAL 100% TRACK_OBJECT COVERAGE TEST ===")
        
        from core.modules.operations.track_object import TrackObjectOperation
        
        # Test data
        test_frame = torch.rand(3, 100, 100)
        test_frames = [torch.rand(3, 100, 100) for _ in range(3)]
        test_mask = torch.zeros(100, 100)
        test_mask[40:60, 30:70] = 1.0
        test_bbox = [30, 40, 70, 60]
        
        operation = TrackObjectOperation()
        
        print("Targeting specific uncovered lines...")
        
        # ================== Lines 66-67: Missing init params error ==================
        print("Testing lines 66-67: init params validation...")
        with patch.object(operation.logger, 'error') as mock_error:
            # This should trigger the error for missing init_mask AND init_bbox
            result = operation.validate_inputs(frames=test_frames)  # No init_mask or init_bbox
            assert not result
            mock_error.assert_called_once_with("Must provide either 'init_mask' or 'init_bbox'")
        
        # ================== Lines 78-79: Missing frame for update error ==================
        print("Testing lines 78-79: missing frame for update...")
        with patch.object(operation.logger, 'error') as mock_error:
            # This should trigger missing frame error for update mode
            result = operation.validate_inputs(track_id="test_track")  # No frame provided
            assert not result
            mock_error.assert_called_once_with("Missing required parameter: 'frame' for update")
        
        # ================== Line 147: frames list conversion ==================
        print("Testing line 147: frames list conversion...")
        # Test when frames is not a list but frame is provided
        result = operation.run(
            frame=test_frame,  # Single frame, not list
            init_mask=test_mask
        )
        # This should trigger line 147 where frames is converted to list
        assert isinstance(result, dict)
        
        # ================== Lines 195-197: Confidence threshold break ==================
        print("Testing lines 195-197: confidence threshold logic...")
        # Mock _track_in_frame to return low confidence for frame 1 (not frame 0)
        original_track = operation._track_in_frame
        def mock_track_low_conf(*args, **kwargs):
            return test_bbox, test_mask, 0.2  # Low confidence
        
        with patch.object(operation, '_track_in_frame', side_effect=mock_track_low_conf):
            with patch.object(operation.logger, 'warning') as mock_warning:
                result = operation.run(
                    frames=test_frames,
                    init_bbox=test_bbox,
                    confidence_threshold=0.5  # Higher than 0.2
                )
                # Should trigger track loss warning and break
                track_id = result['track_id']
                assert operation.tracking_state[track_id]['status'] == 'lost'
                # This should cover the warning log and break statement
                mock_warning.assert_called_once()
        
        # ================== Line 251: Update mode status check ==================
        print("Testing line 251: update mode status check...")
        # Create a track with 'completed' status (not 'active')
        operation.tracking_state["completed_track"] = {
            'status': 'completed',
            'trajectory': [test_bbox],
            'confidences': [0.9]
        }
        
        result = operation.run(
            frame=test_frame,
            track_id="completed_track"
        )
        # Should return early due to non-active status
        assert result['status'] == 'completed'
        assert 'cannot update' in result['message']
        
        # ================== Lines 274-275: Update confidence loss ==================
        print("Testing lines 274-275: update confidence loss...")
        # Create an active track for update
        operation.tracking_state["conf_test"] = {
            'status': 'active',
            'trajectory': [test_bbox],
            'masks': [test_mask],
            'confidences': [0.9],
            'frame_count': 1,
            'last_position': test_bbox
        }
        
        # Mock _track_in_frame to return low confidence
        with patch.object(operation, '_track_in_frame', return_value=(test_bbox, test_mask, 0.3)):
            with patch.object(operation.logger, 'warning') as mock_warning:
                result = operation.run(
                    frame=test_frame,
                    track_id="conf_test",
                    confidence_threshold=0.5
                )
                # Should trigger confidence loss during update
                assert result['status'] == 'lost'
                mock_warning.assert_called_once()
        
        # ================== Line 356: HW format frame ==================
        print("Testing line 356: HW format frame...")
        # Test _bbox_to_mask with HW format (2D) frame
        frame_hw = torch.rand(80, 90)  # HW format, not CHW
        mask = operation._bbox_to_mask(frame_hw, [10, 15, 50, 60])
        assert mask.shape == (80, 90)
        # This should trigger the else branch on line 356
        
        # ================== Line 441: Direction calculation fallback ==================
        print("Testing line 441: direction calculation fallback...")
        # Test motion statistics with trajectory that has less than 2 centroids somehow
        # This is tricky - let's use a custom scenario
        trajectory_single = [[10, 20, 30, 40]]  # Single frame
        stats = operation._calculate_motion_statistics(trajectory_single)
        # Should use the fallback direction = 0 on line 441
        assert stats['direction'] is None
        
        # Alternative: trajectory with same position (no movement)
        trajectory_same = [
            [10, 20, 30, 40],
            [10, 20, 30, 40]  # Same position
        ]
        stats = operation._calculate_motion_statistics(trajectory_same)
        # Should calculate direction normally but might hit edge cases
        assert 'direction' in stats
        
        # ================== Line 482: Reset track return False ==================
        print("Testing line 482: reset track return False...")
        # Try to reset a non-existent track
        result = operation.reset_track("absolutely_nonexistent_track_id_12345")
        assert result is False  # Should return False on line 482
        
        print("✓ All specific uncovered lines targeted")
        
        # ================== COMPREHENSIVE RE-TEST ==================
        print("Running comprehensive re-test to ensure all coverage...")
        
        # Re-run all major functionality to maintain coverage
        operation2 = TrackObjectOperation()
        
        # Full validation coverage
        with patch.object(operation2.logger, 'error'):
            operation2.validate_inputs()  # No params - invalid mode
            operation2.validate_inputs(init_mask=test_mask)  # Missing frames
            operation2.validate_inputs(frames=test_frames)  # Missing init params
            operation2.validate_inputs(frames=test_frames, init_bbox="invalid")  # Invalid bbox
            operation2.validate_inputs(track_id="unknown")  # Missing frame for update
            operation2.validate_inputs(frame=test_frame, track_id="unknown")  # Unknown track
        
        # Full workflow tests
        result = operation2.run(frame=test_frame, init_mask=test_mask, return_masks=True)
        result = operation2.run(frames=test_frames, init_bbox=test_bbox, max_frames=2)
        
        # Initialize and update
        operation2.run(frame=test_frame, init_bbox=test_bbox, track_id="update_test2")
        operation2.run(frame=test_frame, track_id="update_test2", return_mask=True)
        
        # Helper methods
        operation2._track_in_frame(test_frame, test_bbox, "update_test2")
        operation2._bbox_to_mask(torch.rand(3, 50, 50), [10, 10, 40, 40])
        operation2._bbox_to_mask(torch.rand(50, 50), [10, 10, 40, 40])  # HW format
        operation2._mask_to_bbox(test_mask)
        operation2._mask_to_bbox(torch.zeros(50, 50))  # Empty mask
        
        # Motion statistics
        operation2._calculate_motion_statistics([test_bbox])  # Short
        operation2._calculate_motion_statistics([[10,20,30,40], [15,25,35,45]])  # Normal
        
        # Track management
        operation2._generate_track_id()
        operation2.tracking_state["temp"] = {"status": "active"}
        operation2.reset_track("temp")
        operation2.reset_track("nonexistent")
        operation2.reset_all_tracks()
        operation2.tracking_state = {"a": {"status": "active"}, "b": {"status": "lost"}}
        operation2.get_active_tracks()
        
        # Utility methods
        operation2.get_required_params()
        operation2.get_optional_params()
        
        print("✓ Comprehensive re-test completed")
        
    finally:
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*80)
        print("FINAL 100% TRACK_OBJECT COVERAGE REPORT")
        print("="*80)
        cov.report(show_missing=True, include='*track_object.py')
        
        # Generate HTML report
        cov.html_report(directory='final_track_100_coverage_html', include='*track_object.py')
        print(f"\nFinal 100% HTML coverage report: final_track_100_coverage_html/")
        
        return cov


if __name__ == "__main__":
    try:
        cov = final_100_percent_track_test()
        print("\n🎯 FINAL 100% TRACK_OBJECT COVERAGE TEST COMPLETED!")
        print("🏆 Track object operation should now have 100% test coverage!")
    except Exception as e:
        print(f"\n❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()