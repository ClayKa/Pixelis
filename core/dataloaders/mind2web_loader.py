# core/dataloaders/mind2web_loader.py

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base_loader import BaseLoader


class Mind2WebLoader(BaseLoader):
    """
    A concrete data loader for the Mind2Web dataset.

    This loader is specifically designed to handle Mind2Web's structure, where
    the entire dataset is sharded across multiple large JSON files. It is
    responsible for:
    1.  Discovering and validating all data shards.
    2.  Building a unified index that maps a global sample index to the
        correct file and the sample's position within that file.
    3.  Parsing the complex, nested structure of a Mind2Web trajectory.
    4.  Adapting a raw sample into the project's standard format.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Mind2WebLoader.
        
        Args:
            config: Configuration dictionary containing 'path' (directory with JSON shards)
        """
        # Validate required config keys before calling super().__init__
        if 'path' not in config:
            raise ValueError("Mind2WebLoader config must include 'path'")
        
        # Set up paths before calling super().__init__
        self.data_path = Path(config['path'])
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # Initialize shard cache for performance
        self._shard_cache = {}
        
        # Now call super().__init__ which will call _build_index()
        super().__init__(config)

    def _build_index(self) -> List[Tuple[Path, int]]:
        """
        Build a lightweight pointer index for all samples across JSON shards.
        
        Returns:
            List of tuples (shard_path, index_within_shard) for each sample
        """
        index = []
        
        # Discover all JSON shard files
        shard_files = list(self.data_path.glob('*.json'))
        if not shard_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_path}")
        
        # Sort files for consistent ordering
        shard_files.sort()
        
        # Build pointer index without loading full content
        for shard_path in shard_files:
            try:
                # Open and count trajectories in this shard
                with open(shard_path, 'r', encoding='utf-8') as f:
                    shard_data = json.load(f)
                
                # Validate shard structure
                if not isinstance(shard_data, list):
                    print(f"Warning: Skipping {shard_path} - not a list")
                    continue
                
                # Create pointer tuples for each trajectory in this shard
                num_trajectories = len(shard_data)
                for idx in range(num_trajectories):
                    index.append((shard_path, idx))
                
                print(f"Added {num_trajectories} samples from {shard_path.name}")
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Error reading {shard_path}: {e}")
                continue
        
        print(f"Built index with {len(index)} total samples from {len(shard_files)} shards")
        return index

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a single trajectory sample using the pointer index.
        
        Args:
            index: Global sample index
            
        Returns:
            Standardized sample dictionary
        """
        if index >= len(self._index):
            raise IndexError(f"Index {index} out of range (max: {len(self._index) - 1})")
        
        # Get pointer to the specific trajectory
        shard_path, sample_idx_in_shard = self._index[index]
        
        # Load shard with caching
        if shard_path not in self._shard_cache:
            with open(shard_path, 'r', encoding='utf-8') as f:
                self._shard_cache[shard_path] = json.load(f)
        
        # Get the specific trajectory
        trajectories = self._shard_cache[shard_path]
        raw_trajectory = trajectories[sample_idx_in_shard]
        
        # Extract key information
        annotation_id = raw_trajectory.get('annotation_id', f"unknown_{index}")
        confirmed_task = raw_trajectory.get('confirmed_task', '')
        website = raw_trajectory.get('website', '')
        
        # Create base standardized structure
        # For Mind2Web, we don't have traditional media files, so we create a custom structure
        sample = {
            'source_dataset': self.source_name,
            'sample_id': str(annotation_id),
            'media_path': str(shard_path),  # Point to the shard file
            'media_type': 'webpage_trace',
            'width': None,
            'height': None,
            'annotations': {}
        }
        
        # Add Mind2Web-specific annotations
        sample['annotations'].update({
            'goal': confirmed_task,
            'web_domain': website,
            'action_trace': self._parse_action_trace(raw_trajectory.get('actions', [])),
            'initial_dom': raw_trajectory.get('initial_dom', ''),
            'task_id': raw_trajectory.get('task_id', ''),
            'subdomain': raw_trajectory.get('subdomain', ''),
            'action_reprs': raw_trajectory.get('action_reprs', []),
            'pos_candidates': raw_trajectory.get('pos_candidates', []),
            'neg_candidates': raw_trajectory.get('neg_candidates', []),
            'dataset_info': {
                'task_type': 'web_automation',
                'suitable_for_zoom': True,
                'trajectory_length': len(raw_trajectory.get('actions', [])),
                'has_screenshots': any('screenshot' in str(action) for action in raw_trajectory.get('actions', [])),
                'complexity_score': self._calculate_complexity_score(raw_trajectory)
            }
        })
        
        return sample

    def _parse_action_trace(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse and standardize the action trace from raw Mind2Web format.
        
        Args:
            actions: List of raw action dictionaries
            
        Returns:
            List of standardized action dictionaries
        """
        parsed_actions = []
        
        for action in actions:
            if not isinstance(action, dict):
                continue
                
            parsed_action = {
                'action_type': action.get('action_type', 'unknown'),
                'coordinate': action.get('coordinate', []),
                'text': action.get('text', ''),
                'element_html': action.get('element_html', ''),
                'element_properties': action.get('element_properties', {}),
                'screenshot_path': action.get('screenshot', ''),
                'success': action.get('success', True)
            }
            parsed_actions.append(parsed_action)
        
        return parsed_actions

    def _calculate_complexity_score(self, trajectory: Dict[str, Any]) -> float:
        """
        Calculate a complexity score for the trajectory.
        
        Args:
            trajectory: Raw trajectory dictionary
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        actions = trajectory.get('actions', [])
        if not actions:
            return 0.0
        
        # Basic complexity factors
        num_actions = len(actions)
        unique_action_types = len(set(action.get('action_type', '') for action in actions))
        has_text_input = any(action.get('text', '') for action in actions)
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (num_actions * 0.1) + (unique_action_types * 0.2) + (0.3 if has_text_input else 0))
        return round(complexity, 3)

    def get_trajectory_by_annotation_id(self, annotation_id: str) -> Dict[str, Any]:
        """
        Utility method to get a trajectory by its annotation ID.
        
        Args:
            annotation_id: The annotation ID to search for
            
        Returns:
            Sample dictionary if found
            
        Raises:
            ValueError: If annotation ID not found
        """
        for i in range(len(self)):
            sample = self.get_item(i)
            if sample['sample_id'] == annotation_id:
                return sample
        
        raise ValueError(f"Annotation ID '{annotation_id}' not found")

    def get_trajectories_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all trajectories for a specific web domain.
        
        Args:
            domain: Web domain to filter by
            
        Returns:
            List of sample dictionaries from that domain
        """
        domain_trajectories = []
        
        for i in range(len(self)):
            sample = self.get_item(i)
            if sample['annotations']['web_domain'] == domain:
                domain_trajectories.append(sample)
        
        return domain_trajectories

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self._index:
            return {
                'total_trajectories': 0,
                'total_shards': 0,
                'domains': {},
                'avg_actions_per_trajectory': 0,
                'complexity_distribution': {}
            }
        
        # Sample a subset for statistics (to avoid loading everything)
        sample_size = min(100, len(self._index))
        sample_indices = range(0, len(self._index), len(self._index) // sample_size)
        
        domains = {}
        action_counts = []
        complexities = []
        
        for i in sample_indices:
            try:
                sample = self.get_item(i)
                domain = sample['annotations']['web_domain']
                domains[domain] = domains.get(domain, 0) + 1
                
                action_count = sample['annotations']['dataset_info']['trajectory_length']
                action_counts.append(action_count)
                
                complexity = sample['annotations']['dataset_info']['complexity_score']
                complexities.append(complexity)
                
            except Exception:
                continue
        
        return {
            'total_trajectories': len(self._index),
            'total_shards': len(set(path for path, _ in self._index)),
            'domains': domains,
            'avg_actions_per_trajectory': sum(action_counts) / len(action_counts) if action_counts else 0,
            'complexity_distribution': {
                'mean': sum(complexities) / len(complexities) if complexities else 0,
                'min': min(complexities) if complexities else 0,
                'max': max(complexities) if complexities else 0
            },
            'sample_size_used': len(sample_indices)
        }