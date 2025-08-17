"""
Unit tests for the ArtifactManager class.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.reproducibility import ArtifactManager, ArtifactType


class TestArtifactManager:
    """Test suite for ArtifactManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton instance
        ArtifactManager._instance = None
        # Use offline mode for tests
        import os
        os.environ["PIXELIS_OFFLINE_MODE"] = "true"
    
    def teardown_method(self):
        """Clean up after tests."""
        # Reset singleton
        ArtifactManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that ArtifactManager follows singleton pattern."""
        manager1 = ArtifactManager()
        manager2 = ArtifactManager()
        assert manager1 is manager2
    
    def test_init_run(self):
        """Test run initialization."""
        manager = ArtifactManager()
        run_name = "test_run"
        
        manager.init_run(run_name, project="test_project", tags=["test"])
        
        assert manager.run_id is not None
        assert run_name in manager.run_id
    
    def test_log_artifact(self):
        """Test artifact logging."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        artifact_data = {"key": "value"}
        artifact = manager.log_artifact(
            name="test_artifact",
            type=ArtifactType.CONFIG,
            data=artifact_data,
            metadata={"description": "Test artifact"}
        )
        
        assert artifact is not None
        assert artifact["name"] == "test_artifact"
        assert artifact["type"] == ArtifactType.CONFIG.value
        assert artifact["version"] == "v1"
    
    def test_log_large_artifact(self, mocker, tmp_path): # <-- Add tmp_path fixture
        """
        Test that logging a large file artifact works correctly
        WITHOUT performing real, slow I/O operations.
        """
        manager = ArtifactManager()
    
        # Mock the slow I/O methods
        mock_hash_compute = mocker.patch.object(manager, '_compute_file_hash', return_value='mock_hash_123')
        mock_storage_upload = mocker.patch.object(manager.storage_backend, 'upload')
    
        # 1. [THE FIX] Create a REAL, EMPTY temporary file.
        #    tmp_path is a pytest fixture that provides a temporary Path object.
        large_file = tmp_path / "large_dataset.bin"
        large_file.touch() # Create the empty file

        # We can still mock its stat() method if we need to simulate a large size
        # without actually writing data.
        mocker.patch.object(large_file, 'stat', return_value=MagicMock(st_size=1 * 1024**3))
    
        # 2. Call the function under test with the REAL Path object
        artifact_meta = manager.log_artifact(
            name="large_dataset",
            type="dataset",
            file_path=large_file # <-- Pass the real, temporary Path object
        )

        # 3. Assert that the slow methods were called with the correct path
        mock_hash_compute.assert_called_once_with(large_file)
        mock_storage_upload.assert_called_once_with(large_file, 'mock_hash_123')

        # 4. Assert metadata is correct
        assert artifact_meta.name == "large_dataset"
        assert artifact_meta.hash == "mock_hash_123"
        assert artifact_meta.size_bytes == 1 * 1024**3
    
    def test_use_artifact(self):
        """Test artifact retrieval."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        # First log an artifact
        artifact_data = {"data": "test"}
        logged_artifact = manager.log_artifact(
            name="test_artifact",
            type=ArtifactType.DATASET,
            data=artifact_data
        )
        
        # Then retrieve it
        retrieved = manager.use_artifact(
            name="test_artifact",
            version="v1"
        )
        
        assert retrieved is not None
        assert retrieved["name"] == "test_artifact"
        assert retrieved["version"] == "v1"
    
    def test_artifact_versioning(self):
        """Test that artifacts get versioned correctly."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        # Log same artifact multiple times
        for i in range(3):
            artifact = manager.log_artifact(
                name="versioned_artifact",
                type=ArtifactType.METRICS,
                data={"iteration": i}
            )
            assert artifact["version"] == f"v{i+1}"
    
    def test_artifact_lineage(self):
        """Test artifact lineage tracking."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        # Create parent artifact
        parent = manager.log_artifact(
            name="parent_artifact",
            type=ArtifactType.DATASET,
            data={"parent": True}
        )
        
        # Create child artifact with lineage
        child = manager.log_artifact(
            name="child_artifact",
            type=ArtifactType.MODEL,
            data={"child": True},
            parent_artifacts=[f"{parent['name']}:{parent['version']}"]
        )
        
        assert "parent_artifacts" in child
        assert f"{parent['name']}:{parent['version']}" in child["parent_artifacts"]
    
    def test_content_addressable_storage(self):
        """Test that identical content produces same hash."""
        manager = ArtifactManager()
        
        content1 = {"data": "test", "value": 123}
        content2 = {"data": "test", "value": 123}
        content3 = {"data": "different", "value": 456}
        
        hash1 = manager._compute_content_hash(content1)
        hash2 = manager._compute_content_hash(content2)
        hash3 = manager._compute_content_hash(content3)
        
        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
    
    @patch("wandb.init")
    @patch("wandb.Artifact")
    def test_wandb_integration(self, mock_artifact_class, mock_init):
        """Test WandB integration when online."""
        import os
        os.environ.pop("PIXELIS_OFFLINE_MODE", None)
        
        # Reset singleton
        ArtifactManager._instance = None
        
        # Set up mocks
        mock_run = MagicMock()
        mock_run.id = "wandb_run_123"
        mock_init.return_value = mock_run
        
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact
        
        manager = ArtifactManager()
        manager.init_run("test_run", project="test_project")
        
        # Verify WandB was initialized
        mock_init.assert_called_once()
        assert manager.wandb_run is not None
    
    def test_list_artifacts(self):
        """Test listing all artifacts."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        # Log several artifacts
        artifacts = []
        for i in range(3):
            artifact = manager.log_artifact(
                name=f"artifact_{i}",
                type=ArtifactType.METRICS,
                data={"index": i}
            )
            artifacts.append(artifact)
        
        # List artifacts
        all_artifacts = manager.list_artifacts()
        
        assert len(all_artifacts) >= 3
        for artifact in artifacts:
            assert any(
                a["name"] == artifact["name"] and a["version"] == artifact["version"]
                for a in all_artifacts
            )
    
    def test_thread_safety(self):
        """Test thread-safe singleton access."""
        import threading
        
        managers = []
        
        def get_manager():
            managers.append(ArtifactManager())
        
        threads = [threading.Thread(target=get_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should be the same instance
        assert all(m is managers[0] for m in managers)
    
    @pytest.mark.parametrize("artifact_type", list(ArtifactType))
    def test_all_artifact_types(self, artifact_type):
        """Test logging all artifact types."""
        manager = ArtifactManager()
        manager.init_run("test_run")
        
        artifact = manager.log_artifact(
            name=f"test_{artifact_type.value}",
            type=artifact_type,
            data={"type": artifact_type.value}
        )
        
        assert artifact["type"] == artifact_type.value