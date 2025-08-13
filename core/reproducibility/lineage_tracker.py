"""
Lineage tracking for artifact dependencies and relationships.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LineageNode:
    """Node in the lineage graph representing an artifact."""
    artifact_id: str
    artifact_name: str
    artifact_type: str
    version: str
    created_at: str
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LineageTracker:
    """
    Track and visualize artifact lineage and dependencies.
    """
    
    def __init__(self):
        """Initialize lineage tracker."""
        # Graph structure: artifact_id -> LineageNode
        self.nodes: Dict[str, LineageNode] = {}
        
        # Edge lists: artifact_id -> List[artifact_id]
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self.children: Dict[str, List[str]] = defaultdict(list)
        
        # Run tracking: run_id -> List[artifact_id]
        self.run_artifacts: Dict[str, List[str]] = defaultdict(list)
        
        # Type tracking: type -> List[artifact_id]
        self.type_artifacts: Dict[str, List[str]] = defaultdict(list)
    
    def add_artifact(
        self,
        artifact_id: str,
        name: str,
        artifact_type: str,
        version: str,
        run_id: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """
        Add an artifact to the lineage graph.
        
        Args:
            artifact_id: Unique identifier for the artifact
            name: Artifact name
            artifact_type: Type of artifact
            version: Artifact version
            run_id: Run that created this artifact
            parent_ids: List of parent artifact IDs
            metadata: Additional metadata
        
        Returns:
            Created LineageNode
        """
        # Create node
        node = LineageNode(
            artifact_id=artifact_id,
            artifact_name=name,
            artifact_type=artifact_type,
            version=version,
            created_at=datetime.now().isoformat(),
            run_id=run_id,
            metadata=metadata or {},
        )
        
        # Add to graph
        self.nodes[artifact_id] = node
        
        # Track by type
        self.type_artifacts[artifact_type].append(artifact_id)
        
        # Track by run
        if run_id:
            self.run_artifacts[run_id].append(artifact_id)
        
        # Add edges
        if parent_ids:
            for parent_id in parent_ids:
                if parent_id in self.nodes:
                    self.parents[artifact_id].append(parent_id)
                    self.children[parent_id].append(artifact_id)
                else:
                    logger.warning(f"Parent artifact not found: {parent_id}")
        
        logger.debug(f"Added artifact to lineage: {artifact_id}")
        
        return node
    
    def get_ancestors(
        self,
        artifact_id: str,
        max_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        Get all ancestor artifacts (transitively).
        
        Args:
            artifact_id: Starting artifact ID
            max_depth: Maximum depth to traverse
        
        Returns:
            Set of ancestor artifact IDs
        """
        ancestors = set()
        visited = set()
        queue = [(artifact_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            for parent_id in self.parents.get(current_id, []):
                if parent_id not in visited:
                    ancestors.add(parent_id)
                    queue.append((parent_id, depth + 1))
        
        return ancestors
    
    def get_descendants(
        self,
        artifact_id: str,
        max_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        Get all descendant artifacts (transitively).
        
        Args:
            artifact_id: Starting artifact ID
            max_depth: Maximum depth to traverse
        
        Returns:
            Set of descendant artifact IDs
        """
        descendants = set()
        visited = set()
        queue = [(artifact_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            for child_id in self.children.get(current_id, []):
                if child_id not in visited:
                    descendants.add(child_id)
                    queue.append((child_id, depth + 1))
        
        return descendants
    
    def get_lineage_path(
        self,
        from_id: str,
        to_id: str,
    ) -> Optional[List[str]]:
        """
        Find path between two artifacts if one exists.
        
        Args:
            from_id: Starting artifact ID
            to_id: Target artifact ID
        
        Returns:
            List of artifact IDs forming the path, or None if no path exists
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return None
        
        # BFS to find path
        visited = set()
        queue = [(from_id, [from_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == to_id:
                return path
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Check both parents and children
            for next_id in self.parents.get(current_id, []) + self.children.get(current_id, []):
                if next_id not in visited:
                    queue.append((next_id, path + [next_id]))
        
        return None
    
    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for all artifacts in a run.
        
        Args:
            run_id: Run identifier
        
        Returns:
            Dictionary containing lineage information
        """
        artifacts = self.run_artifacts.get(run_id, [])
        
        if not artifacts:
            return {"run_id": run_id, "artifacts": [], "edges": []}
        
        # Collect all related artifacts
        all_artifacts = set(artifacts)
        
        for artifact_id in artifacts:
            all_artifacts.update(self.get_ancestors(artifact_id))
            all_artifacts.update(self.get_descendants(artifact_id))
        
        # Build subgraph
        subgraph_nodes = {
            aid: asdict(self.nodes[aid])
            for aid in all_artifacts
            if aid in self.nodes
        }
        
        subgraph_edges = []
        for artifact_id in all_artifacts:
            for parent_id in self.parents.get(artifact_id, []):
                if parent_id in all_artifacts:
                    subgraph_edges.append({
                        "from": parent_id,
                        "to": artifact_id,
                    })
        
        return {
            "run_id": run_id,
            "artifacts": list(subgraph_nodes.values()),
            "edges": subgraph_edges,
        }
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect cycles in the lineage graph.
        
        Returns:
            List of cycles (each cycle is a list of artifact IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str, path: List[str]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for child_id in self.children.get(node_id, []):
                if child_id not in visited:
                    if dfs(child_id, path.copy()):
                        return True
                elif child_id in rec_stack:
                    # Found cycle
                    cycle_start = path.index(child_id)
                    cycle = path[cycle_start:] + [child_id]
                    cycles.append(cycle)
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles
    
    def validate_lineage(self) -> Dict[str, Any]:
        """
        Validate the lineage graph for issues.
        
        Returns:
            Dictionary containing validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "stats": {},
        }
        
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            validation["is_valid"] = False
            validation["issues"].append({
                "type": "cycles",
                "message": f"Found {len(cycles)} cycles in lineage graph",
                "details": cycles,
            })
        
        # Check for orphaned nodes (no parents or children)
        orphaned = []
        for node_id in self.nodes:
            if not self.parents.get(node_id) and not self.children.get(node_id):
                orphaned.append(node_id)
        
        if orphaned:
            validation["issues"].append({
                "type": "orphaned",
                "message": f"Found {len(orphaned)} orphaned artifacts",
                "details": orphaned[:10],  # Limit to first 10
            })
        
        # Check for missing references
        missing = []
        for node_id in self.nodes:
            for parent_id in self.parents.get(node_id, []):
                if parent_id not in self.nodes:
                    missing.append((node_id, parent_id))
        
        if missing:
            validation["is_valid"] = False
            validation["issues"].append({
                "type": "missing_references",
                "message": f"Found {len(missing)} missing parent references",
                "details": missing[:10],
            })
        
        # Collect statistics
        validation["stats"] = {
            "total_artifacts": len(self.nodes),
            "total_edges": sum(len(children) for children in self.children.values()),
            "artifacts_by_type": {
                type_name: len(artifacts)
                for type_name, artifacts in self.type_artifacts.items()
            },
            "artifacts_by_run": {
                run_id: len(artifacts)
                for run_id, artifacts in self.run_artifacts.items()
            },
            "max_depth": self._calculate_max_depth(),
            "connected_components": self._count_connected_components(),
        }
        
        return validation
    
    def export_to_dot(self, output_path: Optional[Path] = None) -> str:
        """
        Export lineage graph to DOT format for visualization.
        
        Args:
            output_path: Optional path to save DOT file
        
        Returns:
            DOT format string
        """
        lines = ["digraph lineage {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box];')
        
        # Add nodes with labels
        for node_id, node in self.nodes.items():
            label = f"{node.artifact_name}\\nv{node.version}\\n{node.artifact_type}"
            color = self._get_node_color(node.artifact_type)
            lines.append(
                f'  "{node_id}" [label="{label}", fillcolor="{color}", style="filled"];'
            )
        
        # Add edges
        for parent_id, children_ids in self.children.items():
            for child_id in children_ids:
                lines.append(f'  "{parent_id}" -> "{child_id}";')
        
        lines.append("}")
        
        dot_content = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(dot_content)
            logger.info(f"Exported lineage graph to {output_path}")
        
        return dot_content
    
    def export_to_mermaid(self) -> str:
        """
        Export lineage graph to Mermaid format for Markdown embedding.
        
        Returns:
            Mermaid format string
        """
        lines = ["graph TD"]
        
        # Add nodes
        for node_id, node in self.nodes.items():
            label = f"{node.artifact_name}<br/>v{node.version}"
            shape = self._get_mermaid_shape(node.artifact_type)
            lines.append(f'  {node_id}{shape}["{label}"]')
        
        # Add edges
        for parent_id, children_ids in self.children.items():
            for child_id in children_ids:
                lines.append(f'  {parent_id} --> {child_id}')
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        """
        Save lineage graph to JSON file.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "nodes": {
                node_id: asdict(node)
                for node_id, node in self.nodes.items()
            },
            "parents": dict(self.parents),
            "children": dict(self.children),
            "run_artifacts": dict(self.run_artifacts),
            "type_artifacts": dict(self.type_artifacts),
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved lineage graph to {path}")
    
    def load(self, path: Path):
        """
        Load lineage graph from JSON file.
        
        Args:
            path: Path to load file
        """
        path = Path(path)
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct nodes
        self.nodes = {
            node_id: LineageNode(**node_data)
            for node_id, node_data in data["nodes"].items()
        }
        
        # Reconstruct edges
        self.parents = defaultdict(list, data["parents"])
        self.children = defaultdict(list, data["children"])
        
        # Reconstruct tracking
        self.run_artifacts = defaultdict(list, data["run_artifacts"])
        self.type_artifacts = defaultdict(list, data["type_artifacts"])
        
        logger.info(f"Loaded lineage graph from {path}")
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the lineage graph."""
        max_depth = 0
        
        # Find root nodes (no parents)
        roots = [
            node_id for node_id in self.nodes
            if not self.parents.get(node_id)
        ]
        
        for root in roots:
            depth = self._dfs_depth(root, 0, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _dfs_depth(self, node_id: str, current_depth: int, visited: Set[str]) -> int:
        """DFS to calculate depth from a node."""
        if node_id in visited:
            return current_depth
        
        visited.add(node_id)
        max_child_depth = current_depth
        
        for child_id in self.children.get(node_id, []):
            child_depth = self._dfs_depth(child_id, current_depth + 1, visited)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _count_connected_components(self) -> int:
        """Count number of connected components in the graph."""
        visited = set()
        components = 0
        
        for node_id in self.nodes:
            if node_id not in visited:
                components += 1
                self._dfs_component(node_id, visited)
        
        return components
    
    def _dfs_component(self, node_id: str, visited: Set[str]):
        """DFS to mark all nodes in a connected component."""
        if node_id in visited:
            return
        
        visited.add(node_id)
        
        # Visit all connected nodes
        for next_id in self.parents.get(node_id, []) + self.children.get(node_id, []):
            self._dfs_component(next_id, visited)
    
    def _get_node_color(self, artifact_type: str) -> str:
        """Get color for node based on artifact type."""
        colors = {
            "dataset": "lightblue",
            "model": "lightgreen",
            "checkpoint": "lightgreen",
            "config": "lightyellow",
            "metrics": "lightcoral",
            "evaluation": "lightpink",
            "environment": "lightgray",
            "code": "lavender",
            "experience": "lightcyan",
        }
        return colors.get(artifact_type.lower(), "white")
    
    def _get_mermaid_shape(self, artifact_type: str) -> str:
        """Get Mermaid shape syntax for artifact type."""
        shapes = {
            "dataset": "([",  # Stadium shape
            "model": "{{",    # Hexagon
            "checkpoint": "{{",
            "config": "[",    # Rectangle
            "metrics": "(",   # Rounded
            "evaluation": "(",
            "environment": "[",
            "code": "[",
            "experience": "([",
        }
        
        opening = shapes.get(artifact_type.lower(), "[")
        
        # Get closing bracket
        closing_map = {"([": "])", "{{": "}}", "[": "]", "(": ")"}
        closing = closing_map.get(opening, "]")
        
        return opening + "|" + closing