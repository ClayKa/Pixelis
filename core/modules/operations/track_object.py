"""
Track Object Operation

Implements the TRACK_OBJECT visual operation for object tracking in videos.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from ..operation_registry import BaseOperation, registry


class TrackObjectOperation(BaseOperation):
    """
    Tracks an object across multiple frames in a video sequence.
    
    This operation performs object tracking given an initial object specification
    (mask or bounding box) and tracks it through subsequent frames, returning
    trajectories and tracking confidence.
    """
    
    def __init__(self):
        """Initialize the track object operation."""
        super().__init__()
        self.tracker = None  # Will be loaded lazily
        self.tracking_state = {}  # Stores state between frames
    
    def _load_model(self):
        """
        Lazily load the tracking model.
        
        This could be SAM-Track, ByteTrack, or another tracking model.
        """
        if self.tracker is None:
            self.logger.info("Loading tracking model...")
            # Placeholder for actual model loading
            # Example: self.tracker = load_sam_track_model()
            self.tracker = "placeholder_tracker"
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input arguments.
        
        Required for initialization:
            - frames: List of frame tensors or single frame
            - init_mask or init_bbox: Initial object specification
            
        Required for update:
            - frame: New frame to process
            - track_id: ID of track to update
            
        Returns:
            True if inputs are valid, False otherwise
        """
        # Check if this is initialization or update
        is_init = 'init_mask' in kwargs or 'init_bbox' in kwargs
        is_update = 'track_id' in kwargs
        
        if is_init:
            # Initialization mode
            if 'frames' not in kwargs and 'frame' not in kwargs:
                self.logger.error("Missing required parameter: 'frames' or 'frame'")
                return False
            
            if 'init_mask' not in kwargs and 'init_bbox' not in kwargs:
                self.logger.error("Must provide either 'init_mask' or 'init_bbox'")
                return False
            
            if 'init_bbox' in kwargs:
                bbox = kwargs['init_bbox']
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    self.logger.error("'init_bbox' must be [x1, y1, x2, y2]")
                    return False
        
        elif is_update:
            # Update mode
            if 'frame' not in kwargs:
                self.logger.error("Missing required parameter: 'frame' for update")
                return False
            
            if kwargs['track_id'] not in self.tracking_state:
                self.logger.error(f"Unknown track_id: {kwargs['track_id']}")
                return False
        
        else:
            self.logger.error("Must provide initialization or update parameters")
            return False
        
        return True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tracking operation.
        
        Args:
            For initialization:
                frames: List of video frames or single frame
                init_mask: Initial object mask
                init_bbox: Initial bounding box [x1, y1, x2, y2]
                track_id: Optional ID for this track
                
            For update:
                frame: New frame to process
                track_id: ID of track to update
                
            Common:
                max_frames: Maximum number of frames to process
                confidence_threshold: Minimum confidence to continue tracking
                
        Returns:
            Dictionary containing:
                - track_id: Unique identifier for this track
                - trajectory: List of positions/boxes across frames
                - masks: List of segmentation masks (if available)
                - confidences: Tracking confidence for each frame
                - status: 'active', 'lost', or 'completed'
                - statistics: Motion statistics (velocity, direction, etc.)
        """
        # Validate inputs
        if not self.validate_inputs(**kwargs):
            raise ValueError("Invalid inputs for track operation")
        
        # Load model if needed
        self._load_model()
        
        # Determine mode
        is_init = 'init_mask' in kwargs or 'init_bbox' in kwargs
        
        if is_init:
            return self._initialize_tracking(**kwargs)
        else:
            return self._update_tracking(**kwargs)
    
    def _initialize_tracking(self, **kwargs) -> Dict[str, Any]:
        """
        Initialize a new object track.
        
        Args:
            **kwargs: Initialization parameters
            
        Returns:
            Initial tracking result
        """
        # Extract parameters
        frames = kwargs.get('frames', [kwargs.get('frame')])
        if not isinstance(frames, list):
            frames = [frames]
        
        init_mask = kwargs.get('init_mask')
        init_bbox = kwargs.get('init_bbox')
        track_id = kwargs.get('track_id', self._generate_track_id())
        max_frames = kwargs.get('max_frames', len(frames))
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        # Initialize tracking state
        self.tracking_state[track_id] = {
            'frame_count': 0,
            'trajectory': [],
            'masks': [],
            'confidences': [],
            'last_position': None,
            'last_size': None,
            'status': 'active'
        }
        
        # Process frames
        trajectory = []
        masks = []
        confidences = []
        
        for i, frame in enumerate(frames[:max_frames]):
            if i == 0:
                # First frame - use initialization
                if init_bbox:
                    position = init_bbox
                    mask = self._bbox_to_mask(frame, init_bbox)
                else:
                    mask = init_mask
                    position = self._mask_to_bbox(init_mask)
                
                confidence = 1.0  # Perfect confidence for initialization
                
            else:
                # Subsequent frames - track object
                # Placeholder tracking logic
                # In production, this would use the actual tracker
                position, mask, confidence = self._track_in_frame(
                    frame,
                    trajectory[-1] if trajectory else init_bbox,
                    track_id
                )
                
                # Check if tracking is lost
                if confidence < confidence_threshold:
                    self.tracking_state[track_id]['status'] = 'lost'
                    self.logger.warning(f"Track {track_id} lost at frame {i}")
                    break
            
            trajectory.append(position)
            masks.append(mask)
            confidences.append(confidence)
            
            # Update state
            self.tracking_state[track_id]['frame_count'] = i + 1
            self.tracking_state[track_id]['last_position'] = position
        
        # Store results in state
        self.tracking_state[track_id]['trajectory'] = trajectory
        self.tracking_state[track_id]['masks'] = masks
        self.tracking_state[track_id]['confidences'] = confidences
        
        # Calculate motion statistics
        statistics = self._calculate_motion_statistics(trajectory)
        
        # Prepare result
        result = {
            'track_id': track_id,
            'trajectory': trajectory,
            'masks': masks if kwargs.get('return_masks', False) else None,
            'confidences': confidences,
            'status': self.tracking_state[track_id]['status'],
            'frames_processed': len(trajectory),
            'statistics': statistics
        }
        
        self.logger.debug(
            f"Initialized track {track_id}: "
            f"processed {len(trajectory)} frames, "
            f"status={result['status']}"
        )
        
        return result
    
    def _update_tracking(self, **kwargs) -> Dict[str, Any]:
        """
        Update an existing track with a new frame.
        
        Args:
            **kwargs: Update parameters
            
        Returns:
            Updated tracking result
        """
        frame = kwargs['frame']
        track_id = kwargs['track_id']
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        
        state = self.tracking_state[track_id]
        
        if state['status'] != 'active':
            return {
                'track_id': track_id,
                'status': state['status'],
                'message': f"Track is {state['status']}, cannot update"
            }
        
        # Track in new frame
        last_position = state['last_position']
        position, mask, confidence = self._track_in_frame(
            frame,
            last_position,
            track_id
        )
        
        # Update state
        state['trajectory'].append(position)
        state['masks'].append(mask)
        state['confidences'].append(confidence)
        state['frame_count'] += 1
        state['last_position'] = position
        
        # Check if tracking is lost
        if confidence < confidence_threshold:
            state['status'] = 'lost'
            self.logger.warning(f"Track {track_id} lost at frame {state['frame_count']}")
        
        # Calculate incremental statistics
        recent_trajectory = state['trajectory'][-10:]  # Last 10 frames
        statistics = self._calculate_motion_statistics(recent_trajectory)
        
        result = {
            'track_id': track_id,
            'position': position,
            'mask': mask if kwargs.get('return_mask', False) else None,
            'confidence': confidence,
            'status': state['status'],
            'frame_number': state['frame_count'],
            'statistics': statistics
        }
        
        return result
    
    def _track_in_frame(
        self,
        frame: torch.Tensor,
        prev_position: List[int],
        track_id: str
    ) -> Tuple[List[int], torch.Tensor, float]:
        """
        Track object in a single frame.
        
        Args:
            frame: Current frame
            prev_position: Previous position [x1, y1, x2, y2]
            track_id: Track identifier
            
        Returns:
            Tuple of (new_position, mask, confidence)
        """
        # Placeholder tracking logic
        # In production, this would use the actual tracking model
        
        # Simulate object motion (random walk for demo)
        x1, y1, x2, y2 = prev_position
        
        # Add small random motion
        dx = np.random.randint(-5, 6)
        dy = np.random.randint(-5, 6)
        
        new_position = [
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy
        ]
        
        # Create mask
        mask = self._bbox_to_mask(frame, new_position)
        
        # Simulate confidence decay
        state = self.tracking_state[track_id]
        base_confidence = 0.95
        decay_factor = 0.98 ** state['frame_count']
        confidence = base_confidence * decay_factor
        
        return new_position, mask, confidence
    
    def _bbox_to_mask(
        self,
        frame: torch.Tensor,
        bbox: List[int]
    ) -> torch.Tensor:
        """
        Convert bounding box to mask.
        
        Args:
            frame: Frame tensor
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Binary mask
        """
        if len(frame.shape) == 3:  # CHW
            _, h, w = frame.shape
        else:  # HW
            h, w = frame.shape
        
        mask = torch.zeros((h, w))
        x1, y1, x2, y2 = bbox
        
        # Clip to frame boundaries
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        mask[y1:y2, x1:x2] = 1.0
        return mask
    
    def _mask_to_bbox(self, mask: torch.Tensor) -> List[int]:
        """
        Convert mask to bounding box.
        
        Args:
            mask: Binary mask
            
        Returns:
            [x1, y1, x2, y2]
        """
        nonzero = torch.nonzero(mask)
        
        if len(nonzero) == 0:
            return [0, 0, 0, 0]
        
        y_coords = nonzero[:, 0]
        x_coords = nonzero[:, 1]
        
        x1 = x_coords.min().item()
        y1 = y_coords.min().item()
        x2 = x_coords.max().item()
        y2 = y_coords.max().item()
        
        return [x1, y1, x2, y2]
    
    def _calculate_motion_statistics(
        self,
        trajectory: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Calculate motion statistics from trajectory.
        
        Args:
            trajectory: List of positions
            
        Returns:
            Motion statistics dictionary
        """
        if len(trajectory) < 2:
            return {
                'total_distance': 0,
                'average_velocity': 0,
                'direction': None,
                'is_stationary': True
            }
        
        # Calculate centroids
        centroids = []
        for bbox in trajectory:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append([cx, cy])
        
        # Calculate distances
        distances = []
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
        
        # Calculate velocities
        velocities = distances  # Assuming 1 frame = 1 time unit
        
        # Calculate overall direction
        if len(centroids) >= 2:
            total_dx = centroids[-1][0] - centroids[0][0]
            total_dy = centroids[-1][1] - centroids[0][1]
            direction = np.degrees(np.arctan2(total_dy, total_dx))
        else:
            direction = 0
        
        statistics = {
            'total_distance': sum(distances),
            'average_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': max(velocities) if velocities else 0,
            'direction': direction,
            'is_stationary': sum(distances) < 10,  # Threshold for stationary
            'path_length': len(trajectory),
            'displacement': np.sqrt(
                (centroids[-1][0] - centroids[0][0])**2 +
                (centroids[-1][1] - centroids[0][1])**2
            ) if len(centroids) >= 2 else 0
        }
        
        return statistics
    
    def _generate_track_id(self) -> str:
        """
        Generate a unique track ID.
        
        Returns:
            Unique track identifier
        """
        import uuid
        return f"track_{uuid.uuid4().hex[:8]}"
    
    def reset_track(self, track_id: str) -> bool:
        """
        Reset a specific track.
        
        Args:
            track_id: Track to reset
            
        Returns:
            True if track was reset, False if not found
        """
        if track_id in self.tracking_state:
            del self.tracking_state[track_id]
            self.logger.debug(f"Reset track {track_id}")
            return True
        return False
    
    def reset_all_tracks(self):
        """Reset all tracking states."""
        self.tracking_state.clear()
        self.logger.debug("Reset all tracks")
    
    def get_active_tracks(self) -> List[str]:
        """
        Get list of active track IDs.
        
        Returns:
            List of active track IDs
        """
        return [
            track_id
            for track_id, state in self.tracking_state.items()
            if state['status'] == 'active'
        ]
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameters."""
        # Requirements depend on mode (init vs update)
        return []  # Validated in validate_inputs
    
    def get_optional_params(self) -> Dict[str, Any]:
        """Get dictionary of optional parameters with defaults."""
        return {
            'max_frames': None,
            'confidence_threshold': 0.5,
            'return_masks': False,
            'return_mask': False,
            'track_id': None
        }


# Register the operation with the global registry
registry.register(
    'TRACK_OBJECT',
    TrackObjectOperation,
    metadata={
        'description': 'Track an object across video frames',
        'category': 'tracking',
        'input_types': {
            'frames': 'List[torch.Tensor] or torch.Tensor',
            'init_mask': 'Optional[torch.Tensor]',
            'init_bbox': 'Optional[List[int]]',
            'track_id': 'Optional[str]'
        },
        'output_types': {
            'track_id': 'str',
            'trajectory': 'List[List[int]]',
            'masks': 'Optional[List[torch.Tensor]]',
            'confidences': 'List[float]',
            'status': 'str',
            'statistics': 'Dict'
        }
    }
)