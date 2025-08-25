# tests/dataloaders/test_mind2web_loader.py

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.dataloaders.mind2web_loader import Mind2WebLoader


class TestMind2WebLoader:
    """Test suite for Mind2WebLoader."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return {
            'name': 'test_mind2web',
            'path': '/fake/data/path'
        }

    @pytest.fixture
    def sample_trajectory_data(self):
        """Create sample Mind2Web trajectory data."""
        return [
            {
                'annotation_id': 'task_001',
                'confirmed_task': 'Search for a product on an e-commerce website',
                'website': 'amazon.com',
                'task_id': 'shopping_001',
                'subdomain': 'www.amazon.com',
                'initial_dom': '<html><body>Amazon homepage</body></html>',
                'actions': [
                    {
                        'action_type': 'CLICK',
                        'coordinate': [100, 200],
                        'text': '',
                        'element_html': '<input type="text" id="search-box">',
                        'element_properties': {'id': 'search-box', 'type': 'text'},
                        'screenshot': 'screenshot_1.png',
                        'success': True
                    },
                    {
                        'action_type': 'TYPE',
                        'coordinate': [100, 200],
                        'text': 'laptop',
                        'element_html': '<input type="text" id="search-box">',
                        'element_properties': {'id': 'search-box', 'type': 'text'},
                        'screenshot': 'screenshot_2.png',
                        'success': True
                    }
                ],
                'action_reprs': ['click search box', 'type laptop'],
                'pos_candidates': [{'element': 'search-box'}],
                'neg_candidates': [{'element': 'random-button'}]
            },
            {
                'annotation_id': 'task_002',
                'confirmed_task': 'Login to social media platform',
                'website': 'facebook.com',
                'task_id': 'social_001',
                'subdomain': 'www.facebook.com',
                'initial_dom': '<html><body>Facebook login</body></html>',
                'actions': [
                    {
                        'action_type': 'TYPE',
                        'coordinate': [50, 100],
                        'text': 'user@example.com',
                        'element_html': '<input type="email" name="email">',
                        'element_properties': {'name': 'email', 'type': 'email'},
                        'screenshot': 'screenshot_3.png',
                        'success': True
                    }
                ],
                'action_reprs': ['type email'],
                'pos_candidates': [{'element': 'email-input'}],
                'neg_candidates': []
            }
        ]

    @pytest.fixture
    def second_shard_data(self):
        """Create second shard data for multi-shard testing."""
        return [
            {
                'annotation_id': 'task_003',
                'confirmed_task': 'Navigate to settings page',
                'website': 'google.com',
                'task_id': 'nav_001',
                'subdomain': 'settings.google.com',
                'initial_dom': '<html><body>Google settings</body></html>',
                'actions': [
                    {
                        'action_type': 'CLICK',
                        'coordinate': [200, 300],
                        'text': '',
                        'element_html': '<a href="/settings">Settings</a>',
                        'element_properties': {'href': '/settings'},
                        'screenshot': 'screenshot_4.png',
                        'success': True
                    }
                ],
                'action_reprs': ['click settings'],
                'pos_candidates': [{'element': 'settings-link'}],
                'neg_candidates': []
            }
        ]

    def test_init_missing_required_config_keys(self, sample_config):
        """Test that initialization fails when required config keys are missing."""
        # Test missing 'path'
        config_no_path = sample_config.copy()
        del config_no_path['path']
        
        with pytest.raises(ValueError, match="Mind2WebLoader config must include 'path'"):
            Mind2WebLoader(config_no_path)

    def test_init_nonexistent_path(self, sample_config):
        """Test that initialization fails when data path doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            Mind2WebLoader(sample_config)

    def test_build_index_no_json_files(self, sample_config):
        """Test that initialization fails when no JSON files are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty directory
            config = sample_config.copy()
            config['path'] = temp_dir
            
            with pytest.raises(FileNotFoundError, match="No JSON files found"):
                Mind2WebLoader(config)

    def test_build_index_success_single_shard(self, sample_config, sample_trajectory_data):
        """Test successful index building with single shard."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Verify index was built correctly
            assert len(loader) == 2
            assert len(loader._index) == 2
            assert loader._index[0] == (shard_file, 0)
            assert loader._index[1] == (shard_file, 1)

    def test_build_index_success_multiple_shards(self, sample_config, sample_trajectory_data, second_shard_data):
        """Test successful index building with multiple shards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple JSON shard files
            shard1 = temp_path / "shard_001.json"
            shard2 = temp_path / "shard_002.json"
            
            with open(shard1, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            with open(shard2, 'w', encoding='utf-8') as f:
                json.dump(second_shard_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Verify index was built correctly (3 total samples)
            assert len(loader) == 3
            assert len(loader._index) == 3
            
            # Check that all shards are represented
            shard_paths = set(path for path, _ in loader._index)
            assert len(shard_paths) == 2
            assert shard1 in shard_paths
            assert shard2 in shard_paths

    def test_build_index_invalid_json(self, sample_config):
        """Test handling of invalid JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid JSON file
            invalid_file = temp_path / "invalid.json"
            with open(invalid_file, 'w', encoding='utf-8') as f:
                f.write("invalid json content")
            
            # Create valid JSON file
            valid_file = temp_path / "valid.json"
            with open(valid_file, 'w', encoding='utf-8') as f:
                json.dump([{"annotation_id": "test"}], f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader (should skip invalid file)
            loader = Mind2WebLoader(config)
            
            # Should only have valid file's data
            assert len(loader) == 1

    def test_get_item_success(self, sample_config, sample_trajectory_data):
        """Test successful sample retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Test get_item for first sample
            sample = loader.get_item(0)
            
            # Verify basic structure
            assert sample['source_dataset'] == 'test_mind2web'
            assert sample['sample_id'] == 'task_001'
            assert sample['media_type'] == 'webpage_trace'
            assert sample['media_path'] == str(shard_file)
            
            # Verify annotations structure
            assert 'goal' in sample['annotations']
            assert 'web_domain' in sample['annotations']
            assert 'action_trace' in sample['annotations']
            assert 'initial_dom' in sample['annotations']
            assert 'task_id' in sample['annotations']
            assert 'subdomain' in sample['annotations']
            assert 'action_reprs' in sample['annotations']
            assert 'pos_candidates' in sample['annotations']
            assert 'neg_candidates' in sample['annotations']
            assert 'dataset_info' in sample['annotations']
            
            # Check specific values
            assert sample['annotations']['goal'] == 'Search for a product on an e-commerce website'
            assert sample['annotations']['web_domain'] == 'amazon.com'
            assert len(sample['annotations']['action_trace']) == 2
            
            # Check dataset info
            dataset_info = sample['annotations']['dataset_info']
            assert dataset_info['task_type'] == 'web_automation'
            assert dataset_info['suitable_for_zoom'] == True
            assert dataset_info['trajectory_length'] == 2
            assert isinstance(dataset_info['complexity_score'], float)

    def test_get_item_index_out_of_range(self, sample_config, sample_trajectory_data):
        """Test get_item with invalid index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Test invalid index
            with pytest.raises(IndexError, match="Index 10 out of range"):
                loader.get_item(10)

    def test_parse_action_trace(self, sample_config, sample_trajectory_data):
        """Test action trace parsing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Get sample and check action trace
            sample = loader.get_item(0)
            action_trace = sample['annotations']['action_trace']
            
            assert len(action_trace) == 2
            
            # Check first action
            first_action = action_trace[0]
            assert first_action['action_type'] == 'CLICK'
            assert first_action['coordinate'] == [100, 200]
            assert first_action['text'] == ''
            assert first_action['element_html'] == '<input type="text" id="search-box">'
            assert first_action['success'] == True
            
            # Check second action
            second_action = action_trace[1]
            assert second_action['action_type'] == 'TYPE'
            assert second_action['text'] == 'laptop'

    def test_calculate_complexity_score(self, sample_config, sample_trajectory_data):
        """Test complexity score calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Test complexity calculation
            complexity = loader._calculate_complexity_score(sample_trajectory_data[0])
            
            # Should be > 0 since it has actions and text input
            assert complexity > 0
            assert complexity <= 1.0
            assert isinstance(complexity, float)

    def test_shard_caching(self, sample_config, sample_trajectory_data):
        """Test shard caching mechanism."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # First access should load and cache
            sample1 = loader.get_item(0)
            assert shard_file in loader._shard_cache
            
            # Second access should use cache
            sample2 = loader.get_item(1)
            
            # Both samples should be valid
            assert sample1['sample_id'] == 'task_001'
            assert sample2['sample_id'] == 'task_002'

    def test_utility_methods(self, sample_config, sample_trajectory_data):
        """Test utility methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Test get_trajectory_by_annotation_id
            sample = loader.get_trajectory_by_annotation_id('task_001')
            assert sample['sample_id'] == 'task_001'
            assert sample['annotations']['goal'] == 'Search for a product on an e-commerce website'
            
            # Test with non-existent annotation ID
            with pytest.raises(ValueError, match="Annotation ID 'nonexistent' not found"):
                loader.get_trajectory_by_annotation_id('nonexistent')
            
            # Test get_trajectories_by_domain
            amazon_trajectories = loader.get_trajectories_by_domain('amazon.com')
            assert len(amazon_trajectories) == 1
            assert amazon_trajectories[0]['annotations']['web_domain'] == 'amazon.com'
            
            facebook_trajectories = loader.get_trajectories_by_domain('facebook.com')
            assert len(facebook_trajectories) == 1
            assert facebook_trajectories[0]['annotations']['web_domain'] == 'facebook.com'
            
            nonexistent_trajectories = loader.get_trajectories_by_domain('nonexistent.com')
            assert len(nonexistent_trajectories) == 0

    def test_get_dataset_statistics(self, sample_config, sample_trajectory_data):
        """Test dataset statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create JSON shard file
            shard_file = temp_path / "shard_001.json"
            with open(shard_file, 'w', encoding='utf-8') as f:
                json.dump(sample_trajectory_data, f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Get statistics
            stats = loader.get_dataset_statistics()
            
            # Check basic stats
            assert stats['total_trajectories'] == 2
            assert stats['total_shards'] == 1
            assert 'domains' in stats
            assert 'avg_actions_per_trajectory' in stats
            assert 'complexity_distribution' in stats
            assert 'sample_size_used' in stats
            
            # Check that avg_actions_per_trajectory is reasonable
            assert stats['avg_actions_per_trajectory'] > 0

    def test_empty_shard_handling(self, sample_config):
        """Test handling of empty JSON shards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create empty JSON shard file
            empty_shard = temp_path / "empty.json"
            with open(empty_shard, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            # Update config
            config = sample_config.copy()
            config['path'] = str(temp_path)
            
            # Initialize loader
            loader = Mind2WebLoader(config)
            
            # Should handle empty shard gracefully
            assert len(loader) == 0
            assert len(loader._index) == 0