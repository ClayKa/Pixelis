"""
Complete Test Coverage for lineage_tracker.py

This test file ensures 100% coverage of all lines, branches, and edge cases
in the core/reproducibility/lineage_tracker.py module.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, List, Set
import sys

sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.reproducibility.lineage_tracker import (
    LineageNode,
    LineageTracker
)


class TestLineageNode(unittest.TestCase):
    """Test LineageNode dataclass."""
    
    def test_default_creation(self):
        """Test creating LineageNode with required fields."""
        node = LineageNode(
            artifact_id="test-001",
            artifact_name="test_artifact",
            artifact_type="dataset",
            version="v1.0",
            created_at="2024-01-01T00:00:00"
        )
        
        self.assertEqual(node.artifact_id, "test-001")
        self.assertEqual(node.artifact_name, "test_artifact")
        self.assertEqual(node.artifact_type, "dataset")
        self.assertEqual(node.version, "v1.0")
        self.assertEqual(node.created_at, "2024-01-01T00:00:00")
        self.assertIsNone(node.run_id)
        self.assertEqual(node.metadata, {})
    
    def test_full_creation(self):
        """Test creating LineageNode with all fields."""
        metadata = {"size": 1024, "format": "csv"}
        node = LineageNode(
            artifact_id="test-002",
            artifact_name="full_artifact",
            artifact_type="model",
            version="v2.0",
            created_at="2024-01-01T12:00:00",
            run_id="run-123",
            metadata=metadata
        )
        
        self.assertEqual(node.artifact_id, "test-002")
        self.assertEqual(node.artifact_name, "full_artifact")
        self.assertEqual(node.artifact_type, "model")
        self.assertEqual(node.version, "v2.0")
        self.assertEqual(node.created_at, "2024-01-01T12:00:00")
        self.assertEqual(node.run_id, "run-123")
        self.assertEqual(node.metadata, metadata)
    
    def test_post_init_none_metadata(self):
        """Test __post_init__ with None metadata."""
        node = LineageNode(
            artifact_id="test-003",
            artifact_name="test",
            artifact_type="config",
            version="v1.0",
            created_at="2024-01-01T00:00:00",
            metadata=None
        )
        
        self.assertEqual(node.metadata, {})
    
    def test_post_init_existing_metadata(self):
        """Test __post_init__ with existing metadata."""
        metadata = {"key": "value"}
        node = LineageNode(
            artifact_id="test-004",
            artifact_name="test",
            artifact_type="config",
            version="v1.0",
            created_at="2024-01-01T00:00:00",
            metadata=metadata
        )
        
        # Should not modify existing metadata
        self.assertEqual(node.metadata, metadata)


class TestLineageTracker(unittest.TestCase):
    """Test LineageTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LineageTracker initialization."""
        tracker = LineageTracker()
        
        self.assertIsInstance(tracker.nodes, dict)
        self.assertEqual(len(tracker.nodes), 0)
        
        self.assertIsInstance(tracker.parents, dict)
        self.assertIsInstance(tracker.children, dict)
        self.assertIsInstance(tracker.run_artifacts, dict)
        self.assertIsInstance(tracker.type_artifacts, dict)
    
    def test_add_artifact_basic(self):
        """Test adding basic artifact without parents."""
        with patch('core.reproducibility.lineage_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"
            
            node = self.tracker.add_artifact(
                artifact_id="artifact-001",
                name="test_artifact",
                artifact_type="dataset",
                version="v1.0"
            )
        
        # Check node creation
        self.assertIsInstance(node, LineageNode)
        self.assertEqual(node.artifact_id, "artifact-001")
        self.assertEqual(node.artifact_name, "test_artifact")
        self.assertEqual(node.artifact_type, "dataset")
        self.assertEqual(node.version, "v1.0")
        self.assertEqual(node.created_at, "2024-01-01T00:00:00")
        self.assertIsNone(node.run_id)
        self.assertEqual(node.metadata, {})
        
        # Check it was added to tracker
        self.assertIn("artifact-001", self.tracker.nodes)
        self.assertEqual(self.tracker.nodes["artifact-001"], node)
        
        # Check type tracking
        self.assertIn("dataset", self.tracker.type_artifacts)
        self.assertIn("artifact-001", self.tracker.type_artifacts["dataset"])
        
        # No run tracking
        self.assertEqual(len(self.tracker.run_artifacts), 0)
        
        # No parent/child relationships
        self.assertEqual(len(self.tracker.parents["artifact-001"]), 0)
        self.assertEqual(len(self.tracker.children["artifact-001"]), 0)
    
    def test_add_artifact_with_run(self):
        """Test adding artifact with run_id."""
        node = self.tracker.add_artifact(
            artifact_id="artifact-002",
            name="test_artifact",
            artifact_type="model",
            version="v1.0",
            run_id="run-123"
        )
        
        self.assertEqual(node.run_id, "run-123")
        
        # Check run tracking
        self.assertIn("run-123", self.tracker.run_artifacts)
        self.assertIn("artifact-002", self.tracker.run_artifacts["run-123"])
    
    def test_add_artifact_with_metadata(self):
        """Test adding artifact with metadata."""
        metadata = {"size": 1024, "format": "csv"}
        node = self.tracker.add_artifact(
            artifact_id="artifact-003",
            name="test_artifact",
            artifact_type="dataset",
            version="v1.0",
            metadata=metadata
        )
        
        self.assertEqual(node.metadata, metadata)
    
    def test_add_artifact_with_parents(self):
        """Test adding artifact with parent relationships."""
        # Add parent first
        parent_node = self.tracker.add_artifact(
            artifact_id="parent-001",
            name="parent_artifact",
            artifact_type="dataset",
            version="v1.0"
        )
        
        # Add child with parent
        child_node = self.tracker.add_artifact(
            artifact_id="child-001",
            name="child_artifact",
            artifact_type="model",
            version="v1.0",
            parent_ids=["parent-001"]
        )
        
        # Check relationships
        self.assertIn("parent-001", self.tracker.parents["child-001"])
        self.assertIn("child-001", self.tracker.children["parent-001"])
    
    def test_add_artifact_with_nonexistent_parent(self):
        """Test adding artifact with non-existent parent."""
        with patch('core.reproducibility.lineage_tracker.logger') as mock_logger:
            node = self.tracker.add_artifact(
                artifact_id="child-002",
                name="child_artifact",
                artifact_type="model",
                version="v1.0",
                parent_ids=["nonexistent-parent"]
            )
            
            # Should log warning
            mock_logger.warning.assert_called_with("Parent artifact not found: nonexistent-parent")
            
            # Should not create relationship
            self.assertEqual(len(self.tracker.parents["child-002"]), 0)
    
    def test_add_artifact_with_multiple_parents(self):
        """Test adding artifact with multiple parents."""
        # Add parents
        parent1 = self.tracker.add_artifact("parent-1", "parent1", "dataset", "v1.0")
        parent2 = self.tracker.add_artifact("parent-2", "parent2", "config", "v1.0")
        
        # Add child with both parents
        child = self.tracker.add_artifact(
            artifact_id="child-multi",
            name="child",
            artifact_type="model",
            version="v1.0",
            parent_ids=["parent-1", "parent-2"]
        )
        
        # Check relationships
        self.assertEqual(len(self.tracker.parents["child-multi"]), 2)
        self.assertIn("parent-1", self.tracker.parents["child-multi"])
        self.assertIn("parent-2", self.tracker.parents["child-multi"])
        self.assertIn("child-multi", self.tracker.children["parent-1"])
        self.assertIn("child-multi", self.tracker.children["parent-2"])
    
    def test_get_ancestors_no_parents(self):
        """Test getting ancestors for artifact with no parents."""
        self.tracker.add_artifact("root", "root", "dataset", "v1.0")
        
        ancestors = self.tracker.get_ancestors("root")
        self.assertEqual(len(ancestors), 0)
    
    def test_get_ancestors_single_level(self):
        """Test getting ancestors with single level."""
        self.tracker.add_artifact("parent", "parent", "dataset", "v1.0")
        self.tracker.add_artifact("child", "child", "model", "v1.0", parent_ids=["parent"])
        
        ancestors = self.tracker.get_ancestors("child")
        self.assertEqual(ancestors, {"parent"})
    
    def test_get_ancestors_multi_level(self):
        """Test getting ancestors with multiple levels."""
        # Create chain: grandparent -> parent -> child
        self.tracker.add_artifact("grandparent", "gp", "dataset", "v1.0")
        self.tracker.add_artifact("parent", "p", "config", "v1.0", parent_ids=["grandparent"])
        self.tracker.add_artifact("child", "c", "model", "v1.0", parent_ids=["parent"])
        
        ancestors = self.tracker.get_ancestors("child")
        self.assertEqual(ancestors, {"parent", "grandparent"})
    
    def test_get_ancestors_with_max_depth(self):
        """Test getting ancestors with depth limit."""
        # Create long chain
        self.tracker.add_artifact("root", "root", "dataset", "v1.0")
        self.tracker.add_artifact("level1", "l1", "config", "v1.0", parent_ids=["root"])
        self.tracker.add_artifact("level2", "l2", "model", "v1.0", parent_ids=["level1"])
        self.tracker.add_artifact("level3", "l3", "metrics", "v1.0", parent_ids=["level2"])
        
        # Get ancestors with depth limit
        ancestors = self.tracker.get_ancestors("level3", max_depth=2)
        self.assertEqual(ancestors, {"level2", "level1"})
        
        # Should not include root (depth 3)
        self.assertNotIn("root", ancestors)
    
    def test_get_ancestors_with_cycle(self):
        """Test getting ancestors with cycles in graph."""
        # Create cycle: A -> B -> C -> A
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "config", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "model", "v1.0", parent_ids=["B"])
        
        # Manually create cycle (normally prevented)
        self.tracker.parents["A"].append("C")
        self.tracker.children["C"].append("A")
        
        ancestors = self.tracker.get_ancestors("C")
        # Should handle cycle gracefully due to visited set
        self.assertTrue(len(ancestors) >= 2)
    
    def test_get_descendants_no_children(self):
        """Test getting descendants for artifact with no children."""
        self.tracker.add_artifact("leaf", "leaf", "metrics", "v1.0")
        
        descendants = self.tracker.get_descendants("leaf")
        self.assertEqual(len(descendants), 0)
    
    def test_get_descendants_single_level(self):
        """Test getting descendants with single level."""
        self.tracker.add_artifact("parent", "parent", "dataset", "v1.0")
        self.tracker.add_artifact("child", "child", "model", "v1.0", parent_ids=["parent"])
        
        descendants = self.tracker.get_descendants("parent")
        self.assertEqual(descendants, {"child"})
    
    def test_get_descendants_multi_level(self):
        """Test getting descendants with multiple levels."""
        # Create chain: parent -> child -> grandchild
        self.tracker.add_artifact("parent", "p", "dataset", "v1.0")
        self.tracker.add_artifact("child", "c", "model", "v1.0", parent_ids=["parent"])
        self.tracker.add_artifact("grandchild", "gc", "metrics", "v1.0", parent_ids=["child"])
        
        descendants = self.tracker.get_descendants("parent")
        self.assertEqual(descendants, {"child", "grandchild"})
    
    def test_get_descendants_with_max_depth(self):
        """Test getting descendants with depth limit."""
        # Create chain
        self.tracker.add_artifact("root", "root", "dataset", "v1.0")
        self.tracker.add_artifact("level1", "l1", "config", "v1.0", parent_ids=["root"])
        self.tracker.add_artifact("level2", "l2", "model", "v1.0", parent_ids=["level1"])
        self.tracker.add_artifact("level3", "l3", "metrics", "v1.0", parent_ids=["level2"])
        
        # Get descendants with depth limit
        descendants = self.tracker.get_descendants("root", max_depth=2)
        self.assertEqual(descendants, {"level1", "level2"})
        
        # Should not include level3 (depth 3)
        self.assertNotIn("level3", descendants)
    
    def test_get_lineage_path_direct(self):
        """Test getting path between directly connected artifacts."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        # Parent to child
        path = self.tracker.get_lineage_path("A", "B")
        self.assertEqual(path, ["A", "B"])
        
        # Child to parent
        path = self.tracker.get_lineage_path("B", "A")
        self.assertEqual(path, ["B", "A"])
    
    def test_get_lineage_path_multi_hop(self):
        """Test getting path with multiple hops."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "config", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "model", "v1.0", parent_ids=["B"])
        
        path = self.tracker.get_lineage_path("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
    
    def test_get_lineage_path_no_path(self):
        """Test getting path when no path exists."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0")  # Disconnected
        
        path = self.tracker.get_lineage_path("A", "B")
        self.assertIsNone(path)
    
    def test_get_lineage_path_nonexistent_nodes(self):
        """Test getting path with non-existent nodes."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        
        # From non-existent
        path = self.tracker.get_lineage_path("nonexistent", "A")
        self.assertIsNone(path)
        
        # To non-existent
        path = self.tracker.get_lineage_path("A", "nonexistent")
        self.assertIsNone(path)
    
    def test_get_lineage_path_same_node(self):
        """Test getting path from node to itself."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        
        path = self.tracker.get_lineage_path("A", "A")
        self.assertEqual(path, ["A"])
    
    def test_get_run_lineage_no_artifacts(self):
        """Test getting run lineage for non-existent run."""
        result = self.tracker.get_run_lineage("nonexistent-run")
        
        expected = {
            "run_id": "nonexistent-run",
            "artifacts": [],
            "edges": []
        }
        self.assertEqual(result, expected)
    
    def test_get_run_lineage_simple(self):
        """Test getting run lineage for simple run."""
        # Add artifact for run
        node = self.tracker.add_artifact(
            artifact_id="run-artifact",
            name="artifact",
            artifact_type="model",
            version="v1.0",
            run_id="test-run"
        )
        
        result = self.tracker.get_run_lineage("test-run")
        
        self.assertEqual(result["run_id"], "test-run")
        self.assertEqual(len(result["artifacts"]), 1)
        self.assertEqual(result["artifacts"][0]["artifact_id"], "run-artifact")
        self.assertEqual(len(result["edges"]), 0)
    
    def test_get_run_lineage_with_dependencies(self):
        """Test getting run lineage with dependencies."""
        # Add parent artifact (different run)
        parent = self.tracker.add_artifact(
            artifact_id="parent",
            name="parent",
            artifact_type="dataset",
            version="v1.0",
            run_id="parent-run"
        )
        
        # Add child artifact in target run
        child = self.tracker.add_artifact(
            artifact_id="child",
            name="child",
            artifact_type="model",
            version="v1.0",
            run_id="target-run",
            parent_ids=["parent"]
        )
        
        result = self.tracker.get_run_lineage("target-run")
        
        self.assertEqual(result["run_id"], "target-run")
        self.assertEqual(len(result["artifacts"]), 2)  # Both parent and child
        self.assertEqual(len(result["edges"]), 1)
        
        # Check edge
        edge = result["edges"][0]
        self.assertEqual(edge["from"], "parent")
        self.assertEqual(edge["to"], "child")
    
    def test_detect_cycles_no_cycles(self):
        """Test cycle detection with no cycles."""
        # Create linear chain
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "metrics", "v1.0", parent_ids=["B"])
        
        cycles = self.tracker.detect_cycles()
        self.assertEqual(len(cycles), 0)
    
    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with cycles."""
        # Create cycle: A -> B -> C -> A
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "config", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "model", "v1.0", parent_ids=["B"])
        
        # Manually create cycle
        self.tracker.parents["A"].append("C")
        self.tracker.children["C"].append("A")
        
        cycles = self.tracker.detect_cycles()
        self.assertGreater(len(cycles), 0)
        
        # Check cycle contains expected nodes
        cycle = cycles[0]
        self.assertIn("A", cycle)
        self.assertIn("B", cycle)
        self.assertIn("C", cycle)
    
    def test_detect_cycles_self_loop(self):
        """Test cycle detection with self-loop."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        
        # Create self-loop
        self.tracker.parents["A"].append("A")
        self.tracker.children["A"].append("A")
        
        cycles = self.tracker.detect_cycles()
        self.assertGreater(len(cycles), 0)
    
    def test_validate_lineage_valid(self):
        """Test lineage validation with valid graph."""
        # Create simple valid graph
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        validation = self.tracker.validate_lineage()
        
        self.assertTrue(validation["is_valid"])
        self.assertEqual(len(validation["issues"]), 0)
        
        # Check stats
        stats = validation["stats"]
        self.assertEqual(stats["total_artifacts"], 2)
        self.assertEqual(stats["total_edges"], 1)
        self.assertIn("dataset", stats["artifacts_by_type"])
        self.assertEqual(stats["artifacts_by_type"]["dataset"], 1)
    
    def test_validate_lineage_with_cycles(self):
        """Test lineage validation with cycles."""
        # Create cycle
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        # Manual cycle
        self.tracker.parents["A"].append("B")
        self.tracker.children["B"].append("A")
        
        validation = self.tracker.validate_lineage()
        
        self.assertFalse(validation["is_valid"])
        
        # Check for cycle issue
        cycle_issues = [issue for issue in validation["issues"] if issue["type"] == "cycles"]
        self.assertEqual(len(cycle_issues), 1)
        self.assertIn("cycles", cycle_issues[0]["message"])
    
    def test_validate_lineage_with_orphans(self):
        """Test lineage validation with orphaned artifacts."""
        # Create orphaned artifact (no parents or children)
        self.tracker.add_artifact("orphan", "orphan", "dataset", "v1.0")
        
        validation = self.tracker.validate_lineage()
        
        # Orphans don't make graph invalid, just noted
        self.assertTrue(validation["is_valid"])
        
        # Check for orphan issue
        orphan_issues = [issue for issue in validation["issues"] if issue["type"] == "orphaned"]
        self.assertEqual(len(orphan_issues), 1)
        self.assertIn("orphaned", orphan_issues[0]["message"])
        self.assertIn("orphan", orphan_issues[0]["details"])
    
    def test_validate_lineage_with_missing_references(self):
        """Test lineage validation with missing references."""
        # Create artifact with missing parent reference
        self.tracker.add_artifact("child", "child", "model", "v1.0")
        
        # Manually add missing reference
        self.tracker.parents["child"].append("missing-parent")
        
        validation = self.tracker.validate_lineage()
        
        self.assertFalse(validation["is_valid"])
        
        # Check for missing reference issue
        missing_issues = [issue for issue in validation["issues"] if issue["type"] == "missing_references"]
        self.assertEqual(len(missing_issues), 1)
        self.assertIn("missing", missing_issues[0]["message"])
    
    def test_validate_lineage_orphan_limit(self):
        """Test orphan details are limited to first 10."""
        # Create many orphaned artifacts
        for i in range(15):
            self.tracker.add_artifact(f"orphan-{i}", f"orphan{i}", "dataset", "v1.0")
        
        validation = self.tracker.validate_lineage()
        
        orphan_issues = [issue for issue in validation["issues"] if issue["type"] == "orphaned"]
        self.assertEqual(len(orphan_issues), 1)
        
        # Should limit to 10
        self.assertEqual(len(orphan_issues[0]["details"]), 10)
    
    def test_validate_lineage_missing_reference_limit(self):
        """Test missing reference details are limited to first 10."""
        # Create many artifacts with missing parents
        for i in range(15):
            artifact_id = f"child-{i}"
            self.tracker.add_artifact(artifact_id, f"child{i}", "model", "v1.0")
            self.tracker.parents[artifact_id].append(f"missing-{i}")
        
        validation = self.tracker.validate_lineage()
        
        missing_issues = [issue for issue in validation["issues"] if issue["type"] == "missing_references"]
        self.assertEqual(len(missing_issues), 1)
        
        # Should limit to 10
        self.assertEqual(len(missing_issues[0]["details"]), 10)
    
    def test_export_to_dot_basic(self):
        """Test basic DOT export."""
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0")
        self.tracker.add_artifact("B", "ArtifactB", "model", "v2.0", parent_ids=["A"])
        
        dot_content = self.tracker.export_to_dot()
        
        # Check DOT structure
        self.assertIn("digraph lineage {", dot_content)
        self.assertIn("rankdir=LR;", dot_content)
        self.assertIn("node [shape=box];", dot_content)
        
        # Check nodes
        self.assertIn('"A"', dot_content)
        self.assertIn("ArtifactA", dot_content)
        self.assertIn("v1.0", dot_content)
        self.assertIn("dataset", dot_content)
        
        # Check edges
        self.assertIn('"A" -> "B"', dot_content)
        
        # Check closing
        self.assertIn("}", dot_content)
    
    def test_export_to_dot_with_file(self):
        """Test DOT export with file output."""
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0")
        
        output_path = Path(self.temp_dir) / "lineage.dot"
        
        with patch('core.reproducibility.lineage_tracker.logger') as mock_logger:
            dot_content = self.tracker.export_to_dot(output_path)
            
            # Check file was created
            self.assertTrue(output_path.exists())
            
            # Check content
            with open(output_path, "r") as f:
                file_content = f.read()
                self.assertEqual(file_content, dot_content)
            
            # Check logging
            mock_logger.info.assert_called_with(f"Exported lineage graph to {output_path}")
    
    def test_export_to_dot_nested_directory(self):
        """Test DOT export with nested directory creation."""
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0")
        
        output_path = Path(self.temp_dir) / "nested" / "dir" / "lineage.dot"
        
        self.tracker.export_to_dot(output_path)
        
        # Check nested directories were created
        self.assertTrue(output_path.parent.exists())
        self.assertTrue(output_path.exists())
    
    def test_export_to_mermaid(self):
        """Test Mermaid export."""
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0")
        self.tracker.add_artifact("B", "ArtifactB", "model", "v2.0", parent_ids=["A"])
        
        mermaid_content = self.tracker.export_to_mermaid()
        
        # Check Mermaid structure
        self.assertIn("graph TD", mermaid_content)
        
        # Check nodes
        self.assertIn("A", mermaid_content)
        self.assertIn("ArtifactA", mermaid_content)
        self.assertIn("v1.0", mermaid_content)
        
        # Check edges
        self.assertIn("A --> B", mermaid_content)
    
    def test_save_and_load(self):
        """Test saving and loading lineage graph."""
        # Create graph
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0", run_id="run-1", metadata={"key": "value"})
        self.tracker.add_artifact("B", "ArtifactB", "model", "v2.0", run_id="run-1", parent_ids=["A"])
        
        save_path = Path(self.temp_dir) / "lineage.json"
        
        # Save
        with patch('core.reproducibility.lineage_tracker.logger') as mock_logger:
            self.tracker.save(save_path)
            mock_logger.info.assert_called_with(f"Saved lineage graph to {save_path}")
        
        # Check file exists
        self.assertTrue(save_path.exists())
        
        # Load into new tracker
        new_tracker = LineageTracker()
        
        with patch('core.reproducibility.lineage_tracker.logger') as mock_logger:
            new_tracker.load(save_path)
            mock_logger.info.assert_called_with(f"Loaded lineage graph from {save_path}")
        
        # Verify loaded content
        self.assertEqual(len(new_tracker.nodes), 2)
        self.assertIn("A", new_tracker.nodes)
        self.assertIn("B", new_tracker.nodes)
        
        # Check node details
        node_a = new_tracker.nodes["A"]
        self.assertEqual(node_a.artifact_name, "ArtifactA")
        self.assertEqual(node_a.artifact_type, "dataset")
        self.assertEqual(node_a.version, "v1.0")
        self.assertEqual(node_a.run_id, "run-1")
        self.assertEqual(node_a.metadata, {"key": "value"})
        
        # Check relationships
        self.assertIn("A", new_tracker.parents["B"])
        self.assertIn("B", new_tracker.children["A"])
        
        # Check tracking
        self.assertIn("run-1", new_tracker.run_artifacts)
        self.assertEqual(len(new_tracker.run_artifacts["run-1"]), 2)
        self.assertIn("dataset", new_tracker.type_artifacts)
        self.assertIn("model", new_tracker.type_artifacts)
    
    def test_save_nested_directory(self):
        """Test saving with nested directory creation."""
        self.tracker.add_artifact("A", "ArtifactA", "dataset", "v1.0")
        
        save_path = Path(self.temp_dir) / "nested" / "dir" / "lineage.json"
        
        self.tracker.save(save_path)
        
        # Check nested directories were created
        self.assertTrue(save_path.parent.exists())
        self.assertTrue(save_path.exists())
    
    def test_calculate_max_depth_no_nodes(self):
        """Test max depth calculation with no nodes."""
        depth = self.tracker._calculate_max_depth()
        self.assertEqual(depth, 0)
    
    def test_calculate_max_depth_single_node(self):
        """Test max depth calculation with single node."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        
        depth = self.tracker._calculate_max_depth()
        self.assertEqual(depth, 0)
    
    def test_calculate_max_depth_linear_chain(self):
        """Test max depth calculation with linear chain."""
        # Create chain: A -> B -> C -> D
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "config", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "model", "v1.0", parent_ids=["B"])
        self.tracker.add_artifact("D", "D", "metrics", "v1.0", parent_ids=["C"])
        
        depth = self.tracker._calculate_max_depth()
        self.assertEqual(depth, 3)  # A(0) -> B(1) -> C(2) -> D(3)
    
    def test_calculate_max_depth_multiple_roots(self):
        """Test max depth calculation with multiple root nodes."""
        # Create two separate chains
        # Chain 1: A -> B (depth 1)
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        # Chain 2: X -> Y -> Z (depth 2)
        self.tracker.add_artifact("X", "X", "dataset", "v1.0")
        self.tracker.add_artifact("Y", "Y", "config", "v1.0", parent_ids=["X"])
        self.tracker.add_artifact("Z", "Z", "model", "v1.0", parent_ids=["Y"])
        
        depth = self.tracker._calculate_max_depth()
        self.assertEqual(depth, 2)  # Maximum of both chains
    
    def test_dfs_depth_with_cycle(self):
        """Test DFS depth calculation with cycles."""
        # Create simple cycle
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        # Create cycle manually
        self.tracker.parents["A"].append("B")
        self.tracker.children["B"].append("A")
        
        # Should handle cycle gracefully
        visited = set()
        depth = self.tracker._dfs_depth("A", 0, visited)
        self.assertGreaterEqual(depth, 0)
    
    def test_count_connected_components_single(self):
        """Test connected components count with single component."""
        # Create connected graph
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        self.tracker.add_artifact("C", "C", "metrics", "v1.0", parent_ids=["B"])
        
        components = self.tracker._count_connected_components()
        self.assertEqual(components, 1)
    
    def test_count_connected_components_multiple(self):
        """Test connected components count with multiple components."""
        # Create two disconnected graphs
        # Component 1
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        self.tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        
        # Component 2
        self.tracker.add_artifact("X", "X", "dataset", "v1.0")
        self.tracker.add_artifact("Y", "Y", "model", "v1.0", parent_ids=["X"])
        
        # Component 3 (isolated)
        self.tracker.add_artifact("Z", "Z", "config", "v1.0")
        
        components = self.tracker._count_connected_components()
        self.assertEqual(components, 3)
    
    def test_dfs_component_already_visited(self):
        """Test DFS component traversal with already visited node."""
        self.tracker.add_artifact("A", "A", "dataset", "v1.0")
        
        visited = {"A"}
        
        # Should return immediately
        self.tracker._dfs_component("A", visited)
        
        # Visited set should not change
        self.assertEqual(visited, {"A"})
    
    def test_get_node_color_known_types(self):
        """Test node color mapping for known artifact types."""
        test_cases = [
            ("dataset", "lightblue"),
            ("model", "lightgreen"),
            ("checkpoint", "lightgreen"),
            ("config", "lightyellow"),
            ("metrics", "lightcoral"),
            ("evaluation", "lightpink"),
            ("environment", "lightgray"),
            ("code", "lavender"),
            ("experience", "lightcyan"),
        ]
        
        for artifact_type, expected_color in test_cases:
            color = self.tracker._get_node_color(artifact_type)
            self.assertEqual(color, expected_color)
    
    def test_get_node_color_case_insensitive(self):
        """Test node color mapping is case insensitive."""
        color_upper = self.tracker._get_node_color("DATASET")
        color_lower = self.tracker._get_node_color("dataset")
        color_mixed = self.tracker._get_node_color("DataSet")
        
        self.assertEqual(color_upper, "lightblue")
        self.assertEqual(color_lower, "lightblue")
        self.assertEqual(color_mixed, "lightblue")
    
    def test_get_node_color_unknown_type(self):
        """Test node color for unknown artifact type."""
        color = self.tracker._get_node_color("unknown_type")
        self.assertEqual(color, "white")
    
    def test_get_mermaid_shape_known_types(self):
        """Test Mermaid shape mapping for known artifact types."""
        test_cases = [
            ("dataset", "([|])"),
            ("model", "{{|}}"),
            ("checkpoint", "{{|}}"),
            ("config", "[|]"),
            ("metrics", "(|)"),
            ("evaluation", "(|)"),
            ("environment", "[|]"),
            ("code", "[|]"),
            ("experience", "([|])"),
        ]
        
        for artifact_type, expected_shape in test_cases:
            shape = self.tracker._get_mermaid_shape(artifact_type)
            self.assertEqual(shape, expected_shape)
    
    def test_get_mermaid_shape_case_insensitive(self):
        """Test Mermaid shape mapping is case insensitive."""
        shape_upper = self.tracker._get_mermaid_shape("DATASET")
        shape_lower = self.tracker._get_mermaid_shape("dataset")
        shape_mixed = self.tracker._get_mermaid_shape("DataSet")
        
        self.assertEqual(shape_upper, "([|])")
        self.assertEqual(shape_lower, "([|])")
        self.assertEqual(shape_mixed, "([|])")
    
    def test_get_mermaid_shape_unknown_type(self):
        """Test Mermaid shape for unknown artifact type."""
        shape = self.tracker._get_mermaid_shape("unknown_type")
        self.assertEqual(shape, "[|]")


class TestIntegration(unittest.TestCase):
    """Integration tests for lineage tracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = LineageTracker()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complex_lineage_workflow(self):
        """Test complex lineage tracking workflow."""
        # Create a realistic ML pipeline lineage
        
        # Raw data
        raw_data = self.tracker.add_artifact(
            "raw-data-001",
            "raw_customer_data.csv",
            "dataset",
            "v1.0",
            run_id="ingestion-run-001",
            metadata={"size": 10000, "format": "csv"}
        )
        
        # Processed data
        processed_data = self.tracker.add_artifact(
            "processed-data-001",
            "cleaned_customer_data.parquet",
            "dataset",
            "v1.1",
            run_id="preprocessing-run-001",
            parent_ids=["raw-data-001"],
            metadata={"size": 9500, "format": "parquet"}
        )
        
        # Training config
        config = self.tracker.add_artifact(
            "config-001",
            "training_config.yaml",
            "config",
            "v1.0",
            run_id="training-run-001",
            metadata={"learning_rate": 0.001, "epochs": 100}
        )
        
        # Model
        model = self.tracker.add_artifact(
            "model-001",
            "customer_classifier.pkl",
            "model",
            "v1.0",
            run_id="training-run-001",
            parent_ids=["processed-data-001", "config-001"],
            metadata={"accuracy": 0.95, "f1_score": 0.92}
        )
        
        # Metrics
        metrics = self.tracker.add_artifact(
            "metrics-001",
            "training_metrics.json",
            "metrics",
            "v1.0",
            run_id="training-run-001",
            parent_ids=["model-001"],
            metadata={"loss": 0.05, "val_loss": 0.07}
        )
        
        # Test lineage queries
        
        # Get ancestors of model
        model_ancestors = self.tracker.get_ancestors("model-001")
        self.assertEqual(model_ancestors, {"processed-data-001", "config-001", "raw-data-001"})
        
        # Get descendants of raw data
        raw_descendants = self.tracker.get_descendants("raw-data-001")
        self.assertEqual(raw_descendants, {"processed-data-001", "model-001", "metrics-001"})
        
        # Get path from raw data to metrics
        path = self.tracker.get_lineage_path("raw-data-001", "metrics-001")
        self.assertEqual(path, ["raw-data-001", "processed-data-001", "model-001", "metrics-001"])
        
        # Get run lineage for training run
        training_lineage = self.tracker.get_run_lineage("training-run-001")
        self.assertEqual(training_lineage["run_id"], "training-run-001")
        self.assertEqual(len(training_lineage["artifacts"]), 5)  # All artifacts involved
        self.assertGreater(len(training_lineage["edges"]), 0)
        
        # Validate lineage
        validation = self.tracker.validate_lineage()
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["stats"]["total_artifacts"], 5)
        self.assertGreater(validation["stats"]["total_edges"], 0)
        
        # Export and verify formats
        dot_content = self.tracker.export_to_dot()
        self.assertIn("digraph lineage", dot_content)
        self.assertIn("raw_customer_data.csv", dot_content)
        
        mermaid_content = self.tracker.export_to_mermaid()
        self.assertIn("graph TD", mermaid_content)
        self.assertIn("customer_classifier.pkl", mermaid_content)
        
        # Test persistence
        save_path = Path(self.temp_dir) / "complex_lineage.json"
        self.tracker.save(save_path)
        
        # Load into new tracker and verify
        new_tracker = LineageTracker()
        new_tracker.load(save_path)
        
        self.assertEqual(len(new_tracker.nodes), 5)
        new_validation = new_tracker.validate_lineage()
        self.assertEqual(new_validation, validation)
    
    def test_error_handling_edge_cases(self):
        """Test error handling and edge cases."""
        # Empty tracker operations
        self.assertEqual(len(self.tracker.get_ancestors("nonexistent")), 0)
        self.assertEqual(len(self.tracker.get_descendants("nonexistent")), 0)
        self.assertIsNone(self.tracker.get_lineage_path("a", "b"))
        
        cycles = self.tracker.detect_cycles()
        self.assertEqual(len(cycles), 0)
        
        # Validation of empty tracker
        validation = self.tracker.validate_lineage()
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["stats"]["total_artifacts"], 0)
        
        # Export of empty tracker
        dot_content = self.tracker.export_to_dot()
        self.assertIn("digraph lineage", dot_content)
        
        mermaid_content = self.tracker.export_to_mermaid()
        self.assertIn("graph TD", mermaid_content)
    
    def test_coverage_edge_cases(self):
        """Test edge cases to achieve 100% coverage."""
        # Test revisiting visited nodes in BFS (covers continue statements)
        # Create diamond pattern: root -> A, B -> leaf (both A and B point to leaf)
        self.tracker.add_artifact("root", "root", "dataset", "v1.0")
        self.tracker.add_artifact("A", "A", "config", "v1.0", parent_ids=["root"])
        self.tracker.add_artifact("B", "B", "config", "v1.0", parent_ids=["root"]) 
        self.tracker.add_artifact("leaf", "leaf", "model", "v1.0", parent_ids=["A", "B"])
        
        # This should trigger the continue statements in BFS when nodes are revisited
        ancestors = self.tracker.get_ancestors("leaf")
        self.assertEqual(ancestors, {"A", "B", "root"})
        
        descendants = self.tracker.get_descendants("root")
        self.assertEqual(descendants, {"A", "B", "leaf"})
        
        # Test path finding with visited nodes
        path = self.tracker.get_lineage_path("root", "leaf")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "root")
        self.assertEqual(path[-1], "leaf")
        
        # Test run lineage with external artifacts not in all_artifacts set
        # This triggers the condition in line 258
        external_tracker = LineageTracker()
        external_tracker.add_artifact("external", "external", "dataset", "v1.0", run_id="external-run")
        external_tracker.add_artifact("internal", "internal", "model", "v1.0", run_id="target-run", parent_ids=["external"])
        
        # Mock the condition where parent is not in all_artifacts
        lineage = external_tracker.get_run_lineage("target-run")
        self.assertEqual(lineage["run_id"], "target-run")
        
        # Test cycle detection that finds a cycle and returns True
        cycle_tracker = LineageTracker()
        cycle_tracker.add_artifact("A", "A", "model", "v1.0")
        cycle_tracker.add_artifact("B", "B", "model", "v1.0", parent_ids=["A"])
        cycle_tracker.add_artifact("C", "C", "model", "v1.0", parent_ids=["B"])
        
        # Create cycle manually to trigger return True in DFS
        cycle_tracker.parents["A"].append("C")
        cycle_tracker.children["C"].append("A")
        
        cycles = cycle_tracker.detect_cycles()
        self.assertGreater(len(cycles), 0)
    
    def test_final_coverage_edge_cases(self):
        """Test final edge cases to achieve 100% coverage."""
        # Test specific branch conditions that need to be triggered
        
        # Test get_descendants with a node that has children not yet visited
        # This should trigger the branch condition in line 180->179
        tracker1 = LineageTracker()
        tracker1.add_artifact("parent", "parent", "dataset", "v1.0")
        tracker1.add_artifact("child1", "child1", "model", "v1.0", parent_ids=["parent"])
        tracker1.add_artifact("child2", "child2", "model", "v1.0", parent_ids=["parent"])
        
        # Get descendants - should visit both children
        descendants = tracker1.get_descendants("parent")
        self.assertEqual(descendants, {"child1", "child2"})
        
        # Test get_lineage_path where we encounter already visited nodes
        # This should trigger the continue in line 215
        tracker2 = LineageTracker()
        tracker2.add_artifact("start", "start", "dataset", "v1.0")
        tracker2.add_artifact("mid1", "mid1", "config", "v1.0", parent_ids=["start"])
        tracker2.add_artifact("mid2", "mid2", "config", "v1.0", parent_ids=["start"])
        tracker2.add_artifact("end", "end", "model", "v1.0", parent_ids=["mid1", "mid2"])
        
        # This creates multiple paths - should encounter visited nodes during BFS
        path = tracker2.get_lineage_path("start", "end")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "start")
        self.assertEqual(path[-1], "end")
        
        # Test get_run_lineage where parent is not in all_artifacts set
        # This should trigger the condition in line 258->257
        tracker3 = LineageTracker()
        tracker3.add_artifact("external_parent", "external", "dataset", "v1.0", run_id="other_run")
        tracker3.add_artifact("target_child", "target", "model", "v1.0", run_id="target_run", parent_ids=["external_parent"])
        
        # The external_parent should not be in the run artifacts for target_run
        lineage = tracker3.get_run_lineage("target_run")
        self.assertEqual(lineage["run_id"], "target_run")
        # Should include both artifacts due to dependency traversal
        self.assertEqual(len(lineage["artifacts"]), 2)
        
        # Test cycle detection return True in DFS (line 289)
        # Need a specific cycle that triggers the recursive return True
        tracker4 = LineageTracker()
        tracker4.add_artifact("A", "A", "model", "v1.0")
        tracker4.add_artifact("B", "B", "model", "v1.0") 
        tracker4.add_artifact("C", "C", "model", "v1.0")
        tracker4.add_artifact("D", "D", "model", "v1.0")
        
        # Create a complex cycle that requires recursive DFS
        # A -> B -> C -> D -> B (cycle in the middle)
        tracker4.parents["B"].extend(["A", "D"])
        tracker4.children["A"].append("B")
        tracker4.children["D"].append("B")
        
        tracker4.parents["C"].append("B")
        tracker4.children["B"].append("C")
        
        tracker4.parents["D"].append("C")
        tracker4.children["C"].append("D")
        
        cycles = tracker4.detect_cycles()
        self.assertGreater(len(cycles), 0)


if __name__ == "__main__":
    unittest.main()