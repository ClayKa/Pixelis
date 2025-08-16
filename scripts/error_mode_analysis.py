#!/usr/bin/env python3
"""
Pixelis: Systematic Error Mode Analysis Script

This script performs comprehensive error analysis using:
1. Automated discovery via clustering
2. Manual interpretation and taxonomy creation
3. Comprehensive error analysis report generation

Key Features:
- Extract embeddings from failure cases
- Apply multiple clustering algorithms (K-means, DBSCAN, hierarchical)
- Interactive visualization for pattern exploration
- Generate detailed error taxonomy
- Create publication-quality report
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Project imports
from core.data_structures import Experience
from core.modules.operation_registry import VisualOperationRegistry
from core.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class ErrorCase:
    """Represents a single error/failure case."""
    id: str
    task_type: str
    input_data: Dict[str, Any]
    expected_output: str
    actual_output: str
    error_type: str  # Predicted error category
    trajectory: List[Dict[str, Any]]
    embedding: Optional[np.ndarray] = None
    cluster_id: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorCluster:
    """Represents a cluster of similar errors."""
    cluster_id: int
    error_cases: List[ErrorCase]
    centroid: np.ndarray
    representative_samples: List[ErrorCase]
    common_patterns: Dict[str, Any]
    label: str = ""
    description: str = ""

@dataclass
class ErrorTaxonomy:
    """Manual taxonomy for error classification."""
    perception_failures: List[str] = field(default_factory=lambda: [
        "Low contrast detection", "Occlusion handling", "Small object recognition",
        "Complex scene understanding", "Lighting variation"
    ])
    reasoning_failures: List[str] = field(default_factory=lambda: [
        "Logical inconsistency", "Context misunderstanding", "Causal inference error",
        "Spatial reasoning failure", "Temporal reasoning failure"
    ])
    tool_usage_failures: List[str] = field(default_factory=lambda: [
        "Incorrect tool selection", "Tool parameter error", "Tool chaining failure",
        "Tool output misinterpretation", "Tool timeout"
    ])
    language_failures: List[str] = field(default_factory=lambda: [
        "Instruction misinterpretation", "Ambiguity resolution", "Multi-step instruction",
        "Negation handling", "Quantifier understanding"
    ])

class ErrorModeAnalyzer:
    """Main class for systematic error mode analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the error analyzer."""
        self.config = OmegaConf.load(config_path)
        self.output_dir = Path("analysis/error_modes")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize taxonomy
        self.taxonomy = ErrorTaxonomy()
        
        # Storage
        self.error_cases: List[ErrorCase] = []
        self.clusters: List[ErrorCluster] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def load_failure_cases(self, data_path: str) -> List[ErrorCase]:
        """Load failure cases from evaluation results."""
        logger.info(f"Loading failure cases from {data_path}")
        error_cases = []
        
        # Load evaluation results
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Extract failure cases
        for item in data:
            if not item.get('success', True):  # Only failures
                error_case = ErrorCase(
                    id=item.get('id', str(len(error_cases))),
                    task_type=item.get('task_type', 'unknown'),
                    input_data=item.get('input', {}),
                    expected_output=item.get('expected', ''),
                    actual_output=item.get('actual', ''),
                    error_type='unclassified',
                    trajectory=item.get('trajectory', []),
                    confidence=item.get('confidence', 0.0),
                    metadata=item.get('metadata', {})
                )
                error_cases.append(error_case)
        
        logger.info(f"Loaded {len(error_cases)} failure cases")
        return error_cases
    
    def extract_embeddings(self, error_cases: List[ErrorCase]) -> np.ndarray:
        """Extract embeddings for each error case."""
        logger.info("Extracting embeddings for error cases")
        embeddings = []
        
        for case in tqdm(error_cases, desc="Extracting embeddings"):
            # Create text representation of the error
            text_repr = self._create_text_representation(case)
            
            # Get embedding
            embedding = self.embed_model.encode(text_repr, convert_to_numpy=True)
            case.embedding = embedding
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _create_text_representation(self, case: ErrorCase) -> str:
        """Create text representation for embedding."""
        parts = []
        
        # Add task type
        parts.append(f"Task: {case.task_type}")
        
        # Add input description
        if isinstance(case.input_data, dict):
            if 'question' in case.input_data:
                parts.append(f"Question: {case.input_data['question']}")
            if 'context' in case.input_data:
                parts.append(f"Context: {case.input_data['context'][:200]}")
        
        # Add expected vs actual
        parts.append(f"Expected: {case.expected_output}")
        parts.append(f"Actual: {case.actual_output}")
        
        # Add trajectory summary
        if case.trajectory:
            tools_used = [step.get('tool', '') for step in case.trajectory if 'tool' in step]
            if tools_used:
                parts.append(f"Tools used: {', '.join(tools_used[:5])}")
        
        return " ".join(parts)
    
    def cluster_errors(self, embeddings: np.ndarray, method: str = 'kmeans') -> List[int]:
        """Apply clustering to discover error patterns."""
        logger.info(f"Clustering errors using {method}")
        
        if method == 'kmeans':
            # Determine optimal k using elbow method
            k = self._find_optimal_k(embeddings)
            clusterer = KMeans(n_clusters=k, random_state=42)
            
        elif method == 'dbscan':
            # Use DBSCAN for density-based clustering
            eps = self._find_optimal_eps(embeddings)
            clusterer = DBSCAN(eps=eps, min_samples=5)
            
        elif method == 'hierarchical':
            # Use hierarchical clustering
            n_clusters = min(10, len(embeddings) // 10)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Perform clustering
        labels = clusterer.fit_predict(embeddings)
        
        # Evaluate clustering quality
        if len(set(labels)) > 1:
            silhouette = silhouette_score(embeddings, labels)
            calinski = calinski_harabasz_score(embeddings, labels)
            logger.info(f"Clustering quality - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.3f}")
        
        return labels
    
    def _find_optimal_k(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        silhouettes = []
        
        k_range = range(2, min(max_k, len(embeddings) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(embeddings, labels))
        
        # Find elbow point
        if len(inertias) > 2:
            # Calculate second derivative
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            # Find elbow (maximum curvature)
            elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
            optimal_k = list(k_range)[elbow_idx]
        else:
            optimal_k = 5  # Default
        
        logger.info(f"Optimal k determined: {optimal_k}")
        return optimal_k
    
    def _find_optimal_eps(self, embeddings: np.ndarray) -> float:
        """Find optimal eps for DBSCAN using k-distance graph."""
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distance
        k = min(5, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find elbow point
        # Simple heuristic: use 90th percentile
        eps = np.percentile(distances, 90)
        
        logger.info(f"Optimal eps determined: {eps:.3f}")
        return eps
    
    def analyze_clusters(self, error_cases: List[ErrorCase], labels: np.ndarray) -> List[ErrorCluster]:
        """Analyze discovered clusters and extract patterns."""
        logger.info("Analyzing discovered clusters")
        clusters = []
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get cases in this cluster
            cluster_cases = [case for case, l in zip(error_cases, labels) if l == label]
            cluster_embeddings = np.array([case.embedding for case in cluster_cases])
            
            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find representative samples (closest to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            rep_indices = np.argsort(distances)[:min(3, len(cluster_cases))]
            representative_samples = [cluster_cases[i] for i in rep_indices]
            
            # Extract common patterns
            common_patterns = self._extract_common_patterns(cluster_cases)
            
            cluster = ErrorCluster(
                cluster_id=int(label),
                error_cases=cluster_cases,
                centroid=centroid,
                representative_samples=representative_samples,
                common_patterns=common_patterns
            )
            clusters.append(cluster)
        
        logger.info(f"Analyzed {len(clusters)} clusters")
        return clusters
    
    def _extract_common_patterns(self, cases: List[ErrorCase]) -> Dict[str, Any]:
        """Extract common patterns from a cluster of error cases."""
        patterns = {
            'task_types': Counter([c.task_type for c in cases]),
            'avg_confidence': np.mean([c.confidence for c in cases]),
            'trajectory_lengths': [len(c.trajectory) for c in cases],
            'common_tools': self._get_common_tools(cases),
            'error_keywords': self._extract_error_keywords(cases)
        }
        
        # Find most common failure mode
        if patterns['task_types']:
            patterns['dominant_task'] = patterns['task_types'].most_common(1)[0][0]
        
        return patterns
    
    def _get_common_tools(self, cases: List[ErrorCase]) -> List[str]:
        """Get commonly used tools in error cases."""
        tool_counter = Counter()
        
        for case in cases:
            for step in case.trajectory:
                if 'tool' in step:
                    tool_counter[step['tool']] += 1
        
        return [tool for tool, _ in tool_counter.most_common(5)]
    
    def _extract_error_keywords(self, cases: List[ErrorCase]) -> List[str]:
        """Extract common keywords from error outputs."""
        from collections import Counter
        import re
        
        all_words = []
        for case in cases:
            # Tokenize actual output
            words = re.findall(r'\b\w+\b', case.actual_output.lower())
            all_words.extend(words)
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 2]
        
        # Get most common
        word_counter = Counter(filtered_words)
        return [word for word, _ in word_counter.most_common(10)]
    
    def manual_interpretation(self, clusters: List[ErrorCluster]) -> None:
        """Interactive manual interpretation of clusters."""
        logger.info("Starting manual interpretation interface")
        
        for cluster in clusters:
            print(f"\n{'='*60}")
            print(f"Cluster {cluster.cluster_id} - {len(cluster.error_cases)} cases")
            print(f"{'='*60}")
            
            # Show patterns
            print("\nCommon Patterns:")
            print(f"  Dominant task: {cluster.common_patterns.get('dominant_task', 'N/A')}")
            print(f"  Avg confidence: {cluster.common_patterns['avg_confidence']:.3f}")
            print(f"  Common tools: {', '.join(cluster.common_patterns['common_tools'][:3])}")
            print(f"  Error keywords: {', '.join(cluster.common_patterns['error_keywords'][:5])}")
            
            # Show representative samples
            print("\nRepresentative Samples:")
            for i, sample in enumerate(cluster.representative_samples, 1):
                print(f"\n  Sample {i}:")
                print(f"    Task: {sample.task_type}")
                print(f"    Expected: {sample.expected_output[:100]}...")
                print(f"    Actual: {sample.actual_output[:100]}...")
            
            # Get manual label
            print("\n" + "-"*40)
            print("Suggested categories from taxonomy:")
            all_categories = (
                self.taxonomy.perception_failures +
                self.taxonomy.reasoning_failures +
                self.taxonomy.tool_usage_failures +
                self.taxonomy.language_failures
            )
            
            for i, cat in enumerate(all_categories, 1):
                print(f"  {i}. {cat}")
            
            label = input("\nEnter cluster label (or press Enter to skip): ").strip()
            description = input("Enter cluster description: ").strip()
            
            if label:
                cluster.label = label
                cluster.description = description
                
                # Update error cases with the label
                for case in cluster.error_cases:
                    case.error_type = label
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, 
                          method: str = 'tsne') -> None:
        """Create visualizations of error clusters."""
        logger.info(f"Creating cluster visualization using {method}")
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create interactive plot
        fig = go.Figure()
        
        unique_labels = set(labels)
        colors = px.colors.qualitative.Plotly
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = reduced_embeddings[mask]
            
            cluster_name = f"Cluster {label}"
            if hasattr(self, 'clusters') and label < len(self.clusters):
                cluster = self.clusters[label]
                if cluster.label:
                    cluster_name = cluster.label
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8,
                    opacity=0.7
                ),
                text=[f"Case {j}" for j in range(len(cluster_points))],
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}'
            ))
        
        fig.update_layout(
            title='Error Cluster Visualization',
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2',
            width=1000,
            height=700,
            hovermode='closest'
        )
        
        # Save interactive plot
        output_path = self.output_dir / f'cluster_visualization_{method}.html'
        fig.write_html(str(output_path))
        logger.info(f"Saved visualization to {output_path}")
        
        # Create static plot for report
        self._create_static_visualization(reduced_embeddings, labels, method)
    
    def _create_static_visualization(self, embeddings: np.ndarray, labels: np.ndarray, 
                                    method: str) -> None:
        """Create static visualization for report."""
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            cluster_points = embeddings[mask]
            
            cluster_name = f"Cluster {label}"
            if hasattr(self, 'clusters') and 0 <= label < len(self.clusters):
                cluster = self.clusters[label]
                if cluster.label:
                    cluster_name = cluster.label[:20]  # Truncate for legend
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[color], label=cluster_name, alpha=0.6, s=50)
        
        plt.title(f'Error Clusters ({method.upper()})', fontsize=16)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = self.output_dir / f'cluster_visualization_{method}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved static visualization to {output_path}")
    
    def generate_report(self) -> None:
        """Generate comprehensive error analysis report."""
        logger.info("Generating comprehensive error analysis report")
        
        report_path = self.output_dir / 'error_analysis_report.md'
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# Pixelis Error Mode Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- Total error cases analyzed: {len(self.error_cases)}\n")
            f.write(f"- Number of clusters discovered: {len(self.clusters)}\n")
            f.write(f"- Average cluster size: {np.mean([len(c.error_cases) for c in self.clusters]):.1f}\n\n")
            
            # Error distribution
            f.write("## Error Distribution\n\n")
            
            task_dist = Counter([case.task_type for case in self.error_cases])
            f.write("### By Task Type\n")
            for task, count in task_dist.most_common():
                percentage = (count / len(self.error_cases)) * 100
                f.write(f"- {task}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Cluster analysis
            f.write("## Cluster Analysis\n\n")
            
            for cluster in sorted(self.clusters, key=lambda c: len(c.error_cases), reverse=True):
                f.write(f"### Cluster {cluster.cluster_id}: {cluster.label or 'Unlabeled'}\n\n")
                
                if cluster.description:
                    f.write(f"**Description:** {cluster.description}\n\n")
                
                f.write(f"**Size:** {len(cluster.error_cases)} cases\n\n")
                
                f.write("**Characteristics:**\n")
                patterns = cluster.common_patterns
                f.write(f"- Dominant task type: {patterns.get('dominant_task', 'N/A')}\n")
                f.write(f"- Average confidence: {patterns['avg_confidence']:.3f}\n")
                f.write(f"- Common tools: {', '.join(patterns['common_tools'][:5])}\n")
                f.write(f"- Key error terms: {', '.join(patterns['error_keywords'][:5])}\n\n")
                
                f.write("**Representative Examples:**\n\n")
                for i, sample in enumerate(cluster.representative_samples[:2], 1):
                    f.write(f"*Example {i}:*\n")
                    f.write(f"- Task: {sample.task_type}\n")
                    f.write(f"- Expected: `{sample.expected_output[:100]}...`\n")
                    f.write(f"- Actual: `{sample.actual_output[:100]}...`\n\n")
            
            # Taxonomy mapping
            f.write("## Error Taxonomy Mapping\n\n")
            
            taxonomy_counts = defaultdict(list)
            for cluster in self.clusters:
                if cluster.label:
                    # Find which taxonomy category it belongs to
                    if cluster.label in self.taxonomy.perception_failures:
                        taxonomy_counts['Perception Failures'].append(cluster)
                    elif cluster.label in self.taxonomy.reasoning_failures:
                        taxonomy_counts['Reasoning Failures'].append(cluster)
                    elif cluster.label in self.taxonomy.tool_usage_failures:
                        taxonomy_counts['Tool Usage Failures'].append(cluster)
                    elif cluster.label in self.taxonomy.language_failures:
                        taxonomy_counts['Language Failures'].append(cluster)
                    else:
                        taxonomy_counts['Other'].append(cluster)
            
            for category, clusters in taxonomy_counts.items():
                total_cases = sum(len(c.error_cases) for c in clusters)
                f.write(f"### {category}\n")
                f.write(f"- Clusters: {len(clusters)}\n")
                f.write(f"- Total cases: {total_cases}\n")
                f.write(f"- Percentage: {(total_cases/len(self.error_cases))*100:.1f}%\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the error analysis, the following improvements are recommended:\n\n")
            
            # Generate recommendations based on largest clusters
            for cluster in self.clusters[:5]:
                if cluster.label:
                    f.write(f"1. **{cluster.label}** ({len(cluster.error_cases)} cases)\n")
                    f.write(f"   - Consider additional training data for this error mode\n")
                    f.write(f"   - Implement specific handling for {cluster.common_patterns.get('dominant_task', 'this task type')}\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("The following visualizations have been generated:\n\n")
            f.write("- `cluster_visualization_tsne.html`: Interactive t-SNE visualization\n")
            f.write("- `cluster_visualization_pca.html`: Interactive PCA visualization\n")
            f.write("- `error_distribution.png`: Error distribution charts\n")
            f.write("- `cluster_sizes.png`: Cluster size distribution\n\n")
        
        logger.info(f"Report saved to {report_path}")
        
        # Generate additional charts
        self._generate_distribution_charts()
    
    def _generate_distribution_charts(self) -> None:
        """Generate distribution charts for the report."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Task type distribution
        task_counts = Counter([case.task_type for case in self.error_cases])
        axes[0, 0].bar(range(len(task_counts)), list(task_counts.values()))
        axes[0, 0].set_xticks(range(len(task_counts)))
        axes[0, 0].set_xticklabels(list(task_counts.keys()), rotation=45, ha='right')
        axes[0, 0].set_title('Error Distribution by Task Type')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Cluster size distribution
        cluster_sizes = [len(c.error_cases) for c in self.clusters]
        axes[0, 1].hist(cluster_sizes, bins=min(20, len(self.clusters)))
        axes[0, 1].set_title('Cluster Size Distribution')
        axes[0, 1].set_xlabel('Cluster Size')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Confidence distribution
        confidences = [case.confidence for case in self.error_cases]
        axes[1, 0].hist(confidences, bins=30, alpha=0.7)
        axes[1, 0].set_title('Error Case Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.2f}')
        axes[1, 0].legend()
        
        # 4. Trajectory length distribution
        traj_lengths = [len(case.trajectory) for case in self.error_cases]
        axes[1, 1].hist(traj_lengths, bins=30, alpha=0.7)
        axes[1, 1].set_title('Trajectory Length Distribution')
        axes[1, 1].set_xlabel('Number of Steps')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(traj_lengths), color='red', linestyle='--',
                          label=f'Mean: {np.mean(traj_lengths):.1f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'error_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved distribution charts to {output_path}")
    
    def run_analysis(self, data_path: str, clustering_method: str = 'kmeans',
                    interactive: bool = True) -> None:
        """Run complete error mode analysis pipeline."""
        logger.info("Starting error mode analysis pipeline")
        
        # Step 1: Load failure cases
        self.error_cases = self.load_failure_cases(data_path)
        
        if not self.error_cases:
            logger.warning("No failure cases found to analyze")
            return
        
        # Step 2: Extract embeddings
        self.embeddings = self.extract_embeddings(self.error_cases)
        
        # Step 3: Cluster errors
        labels = self.cluster_errors(self.embeddings, method=clustering_method)
        
        # Step 4: Analyze clusters
        self.clusters = self.analyze_clusters(self.error_cases, labels)
        
        # Step 5: Manual interpretation (optional)
        if interactive:
            self.manual_interpretation(self.clusters)
        
        # Step 6: Visualize clusters
        self.visualize_clusters(self.embeddings, labels, method='tsne')
        self.visualize_clusters(self.embeddings, labels, method='pca')
        
        # Step 7: Generate report
        self.generate_report()
        
        # Step 8: Save results
        self.save_results()
        
        logger.info("Error mode analysis complete")
    
    def save_results(self) -> None:
        """Save analysis results to disk."""
        # Save clusters
        clusters_data = []
        for cluster in self.clusters:
            cluster_dict = {
                'cluster_id': cluster.cluster_id,
                'label': cluster.label,
                'description': cluster.description,
                'size': len(cluster.error_cases),
                'common_patterns': cluster.common_patterns,
                'case_ids': [case.id for case in cluster.error_cases]
            }
            clusters_data.append(cluster_dict)
        
        clusters_path = self.output_dir / 'clusters.json'
        with open(clusters_path, 'w') as f:
            json.dump(clusters_data, f, indent=2, default=str)
        
        # Save error cases with cluster assignments
        cases_data = []
        for case in self.error_cases:
            case_dict = asdict(case)
            case_dict['embedding'] = None  # Don't save embeddings
            cases_data.append(case_dict)
        
        cases_path = self.output_dir / 'error_cases_clustered.json'
        with open(cases_path, 'w') as f:
            json.dump(cases_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pixelis Error Mode Analysis')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to evaluation results with failure cases')
    parser.add_argument('--method', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical'],
                       help='Clustering method to use')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive manual interpretation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ErrorModeAnalyzer(config_path=args.config)
    analyzer.run_analysis(
        data_path=args.data,
        clustering_method=args.method,
        interactive=args.interactive
    )


if __name__ == '__main__':
    main()