#!/usr/bin/env python3
"""
Analyze Human Evaluation Results

This script analyzes the results from human evaluation of reasoning trajectories,
calculating inter-annotator agreement, performing statistical tests, and generating
comprehensive reports on model performance.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HumanEvalAnalyzer:
    """
    Comprehensive analyzer for human evaluation results.
    
    Calculates inter-annotator agreement, performs statistical tests,
    and generates detailed reports on model performance.
    """
    
    def __init__(
        self,
        results_dir: Path,
        output_dir: Path,
        min_annotations_per_sample: int = 3
    ):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing annotation result files
            output_dir: Directory to save analysis outputs
            min_annotations_per_sample: Minimum annotations required per sample
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_annotations = min_annotations_per_sample
        
        # Load all annotations
        self.annotations = self._load_annotations()
        
        # Group annotations by sample
        self.grouped_annotations = self._group_by_sample()
        
        # Extract model names
        self.model_names = self._extract_model_names()
        
        logger.info(f"Loaded {len(self.annotations)} annotations")
        logger.info(f"Covering {len(self.grouped_annotations)} unique samples")
        logger.info(f"Models evaluated: {self.model_names}")
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """
        Load all annotation files from the results directory.
        
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # Load individual annotation files
        for file_path in self.results_dir.glob("*.json"):
            if file_path.name.startswith("session_"):
                # Skip session files (we'll handle them separately)
                continue
            
            try:
                with open(file_path, 'r') as f:
                    annotation = json.load(f)
                    annotations.append(annotation)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Also load session files (JSONL format)
        for file_path in self.results_dir.glob("session_*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        annotation = json.loads(line.strip())
                        annotations.append(annotation)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        return annotations
    
    def _group_by_sample(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group annotations by sample ID.
        
        Returns:
            Dictionary mapping sample_id to list of annotations
        """
        grouped = defaultdict(list)
        
        for annotation in self.annotations:
            sample_id = annotation['sample_id']
            grouped[sample_id].append(annotation)
        
        return dict(grouped)
    
    def _extract_model_names(self) -> List[str]:
        """
        Extract unique model names from annotations.
        
        Returns:
            List of unique model names
        """
        models = set()
        
        for annotation in self.annotations:
            models.add(annotation.get('left_model', 'unknown'))
            models.add(annotation.get('right_model', 'unknown'))
        
        models.discard('unknown')
        return sorted(list(models))
    
    def calculate_fleiss_kappa(
        self,
        ratings: np.ndarray,
        n_categories: int
    ) -> float:
        """
        Calculate Fleiss' Kappa for inter-annotator agreement.
        
        Args:
            ratings: Array of shape (n_samples, n_categories) with rating counts
            n_categories: Number of rating categories
            
        Returns:
            Fleiss' Kappa value
        """
        n_samples = ratings.shape[0]
        n_raters = ratings.sum(axis=1)[0]
        
        # Calculate P_i (proportion of agreement for each sample)
        P_i = (ratings ** 2).sum(axis=1) - n_raters
        P_i = P_i / (n_raters * (n_raters - 1))
        
        # Calculate P_bar (mean of P_i)
        P_bar = P_i.mean()
        
        # Calculate P_e (chance agreement)
        P_j = ratings.sum(axis=0) / (n_samples * n_raters)
        P_e = (P_j ** 2).sum()
        
        # Calculate Kappa
        if P_e == 1:
            return 1.0  # Perfect agreement
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa
    
    def calculate_icc(self, ratings_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate Intraclass Correlation Coefficient (ICC) for ordinal ratings.
        
        Args:
            ratings_matrix: Matrix of ratings (samples x raters)
            
        Returns:
            Tuple of (ICC value, lower CI, upper CI)
        """
        from scipy import stats
        
        n_samples, n_raters = ratings_matrix.shape
        
        # Calculate mean squares
        row_means = np.nanmean(ratings_matrix, axis=1)
        grand_mean = np.nanmean(ratings_matrix)
        
        # Between-subject variance
        MS_between = n_raters * np.sum((row_means - grand_mean) ** 2) / (n_samples - 1)
        
        # Within-subject variance
        MS_within = np.nansum((ratings_matrix - row_means[:, np.newaxis]) ** 2) / (n_samples * (n_raters - 1))
        
        # ICC(2,1) - Two-way random effects, single measurement, absolute agreement
        icc = (MS_between - MS_within) / (MS_between + (n_raters - 1) * MS_within)
        
        # Calculate confidence intervals using F-distribution
        alpha = 0.05
        df1 = n_samples - 1
        df2 = n_samples * (n_raters - 1)
        
        F_value = MS_between / MS_within
        F_lower = F_value / stats.f.ppf(1 - alpha/2, df1, df2)
        F_upper = F_value * stats.f.ppf(1 - alpha/2, df2, df1)
        
        icc_lower = (F_lower - 1) / (F_lower + n_raters - 1)
        icc_upper = (F_upper - 1) / (F_upper + n_raters - 1)
        
        return icc, icc_lower, icc_upper
    
    def analyze_inter_annotator_agreement(self) -> Dict[str, Any]:
        """
        Analyze inter-annotator agreement across all metrics.
        
        Returns:
            Dictionary containing agreement statistics
        """
        results = {}
        
        # Prepare data for agreement calculation
        metrics = ['coherence', 'efficiency', 'thoroughness']
        
        for metric in metrics:
            logger.info(f"Analyzing agreement for {metric}")
            
            # Collect ratings for samples with multiple annotations
            sample_ratings = []
            
            for sample_id, annotations in self.grouped_annotations.items():
                if len(annotations) < self.min_annotations:
                    continue
                
                # Get ratings for left and right models
                left_ratings = [a[f'left_{metric}'] for a in annotations]
                right_ratings = [a[f'right_{metric}'] for a in annotations]
                
                sample_ratings.append(left_ratings[:self.min_annotations])
                sample_ratings.append(right_ratings[:self.min_annotations])
            
            if not sample_ratings:
                logger.warning(f"No samples with sufficient annotations for {metric}")
                continue
            
            # Convert to matrix for ICC calculation
            ratings_matrix = np.array(sample_ratings)
            
            # Calculate ICC
            icc, icc_lower, icc_upper = self.calculate_icc(ratings_matrix)
            
            # Calculate Fleiss' Kappa (need to convert to count matrix)
            n_categories = 5  # 1-5 scale
            count_matrix = np.zeros((len(sample_ratings), n_categories))
            
            for i, ratings in enumerate(sample_ratings):
                for rating in ratings:
                    count_matrix[i, rating - 1] += 1
            
            fleiss_kappa = self.calculate_fleiss_kappa(count_matrix, n_categories)
            
            # Calculate pairwise Cohen's Kappa for each annotator pair
            pairwise_kappas = []
            if ratings_matrix.shape[1] >= 2:
                for i in range(ratings_matrix.shape[1]):
                    for j in range(i + 1, ratings_matrix.shape[1]):
                        try:
                            kappa = cohen_kappa_score(
                                ratings_matrix[:, i],
                                ratings_matrix[:, j],
                                weights='linear'  # Use weighted kappa for ordinal data
                            )
                            pairwise_kappas.append(kappa)
                        except:
                            pass
            
            results[metric] = {
                'icc': icc,
                'icc_ci': (icc_lower, icc_upper),
                'fleiss_kappa': fleiss_kappa,
                'mean_pairwise_kappa': np.mean(pairwise_kappas) if pairwise_kappas else None,
                'interpretation': self._interpret_agreement(icc, fleiss_kappa)
            }
        
        # Analyze preference agreement
        preference_agreement = self._analyze_preference_agreement()
        results['preference'] = preference_agreement
        
        return results
    
    def _interpret_agreement(self, icc: float, kappa: float) -> str:
        """
        Interpret agreement scores.
        
        Args:
            icc: ICC value
            kappa: Kappa value
            
        Returns:
            String interpretation
        """
        # Use ICC as primary measure for ordinal data
        if icc < 0.4:
            return "Poor agreement"
        elif icc < 0.6:
            return "Fair agreement"
        elif icc < 0.75:
            return "Good agreement"
        else:
            return "Excellent agreement"
    
    def _analyze_preference_agreement(self) -> Dict[str, Any]:
        """
        Analyze agreement on overall preference judgments.
        
        Returns:
            Dictionary with preference agreement statistics
        """
        preference_data = []
        
        for sample_id, annotations in self.grouped_annotations.items():
            if len(annotations) < self.min_annotations:
                continue
            
            preferences = [a['preference'] for a in annotations[:self.min_annotations]]
            preference_data.append(preferences)
        
        if not preference_data:
            return {'error': 'Insufficient data for preference analysis'}
        
        # Calculate agreement percentage
        agreement_count = 0
        for preferences in preference_data:
            if len(set(preferences)) == 1:  # All annotators agree
                agreement_count += 1
        
        agreement_percentage = (agreement_count / len(preference_data)) * 100
        
        # Calculate Fleiss' Kappa for preference
        n_categories = 3  # left, right, equal
        count_matrix = np.zeros((len(preference_data), n_categories))
        
        preference_map = {'left': 0, 'right': 1, 'equal': 2}
        for i, preferences in enumerate(preference_data):
            for pref in preferences:
                count_matrix[i, preference_map[pref]] += 1
        
        fleiss_kappa = self.calculate_fleiss_kappa(count_matrix, n_categories)
        
        return {
            'complete_agreement_percentage': agreement_percentage,
            'fleiss_kappa': fleiss_kappa,
            'interpretation': self._interpret_agreement(0, fleiss_kappa)
        }
    
    def identify_disagreement_cases(self, threshold: float = 1.5) -> List[Dict[str, Any]]:
        """
        Identify samples with high annotator disagreement.
        
        Args:
            threshold: Standard deviation threshold for flagging disagreement
            
        Returns:
            List of disagreement cases requiring adjudication
        """
        disagreement_cases = []
        
        for sample_id, annotations in self.grouped_annotations.items():
            if len(annotations) < self.min_annotations:
                continue
            
            # Check each metric for high variance
            for metric in ['coherence', 'efficiency', 'thoroughness']:
                left_ratings = [a[f'left_{metric}'] for a in annotations]
                right_ratings = [a[f'right_{metric}'] for a in annotations]
                
                left_std = np.std(left_ratings)
                right_std = np.std(right_ratings)
                
                if left_std > threshold or right_std > threshold:
                    disagreement_cases.append({
                        'sample_id': sample_id,
                        'metric': metric,
                        'left_std': left_std,
                        'right_std': right_std,
                        'left_ratings': left_ratings,
                        'right_ratings': right_ratings
                    })
        
        return disagreement_cases
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare model performance across all metrics.
        
        Returns:
            DataFrame with model comparison results
        """
        model_scores = defaultdict(lambda: defaultdict(list))
        
        for annotations in self.grouped_annotations.values():
            if len(annotations) < self.min_annotations:
                continue
            
            # Aggregate scores for this sample
            for metric in ['coherence', 'efficiency', 'thoroughness']:
                # Calculate mean scores
                left_scores = [a[f'left_{metric}'] for a in annotations]
                right_scores = [a[f'right_{metric}'] for a in annotations]
                
                left_model = annotations[0]['left_model']
                right_model = annotations[0]['right_model']
                
                model_scores[left_model][metric].extend(left_scores)
                model_scores[right_model][metric].extend(right_scores)
        
        # Calculate summary statistics
        results = []
        for model in self.model_names:
            if model not in model_scores:
                continue
            
            row = {'model': model}
            for metric in ['coherence', 'efficiency', 'thoroughness']:
                scores = model_scores[model][metric]
                if scores:
                    row[f'{metric}_mean'] = np.mean(scores)
                    row[f'{metric}_std'] = np.std(scores)
                    row[f'{metric}_median'] = np.median(scores)
                    row[f'{metric}_n'] = len(scores)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical tests to compare models.
        
        Returns:
            Dictionary with test results
        """
        test_results = {}
        
        # Prepare paired data for key comparisons
        key_comparisons = [
            ('Pixelis-RFT-Base', 'Pixelis-RFT-Coherence', 'H1: Coherence Hypothesis'),
            ('Pixelis-RFT-Full', 'Pixelis-Online', 'H2: Efficiency Hypothesis'),
            ('Pixelis-RFT-Base', 'Pixelis-RFT-Curiosity', 'H3: Curiosity Hypothesis'),
        ]
        
        for model_a, model_b, hypothesis in key_comparisons:
            logger.info(f"Testing: {hypothesis}")
            
            # Collect paired scores
            paired_scores = defaultdict(lambda: {'a': [], 'b': []})
            
            for sample_id, annotations in self.grouped_annotations.items():
                if len(annotations) < self.min_annotations:
                    continue
                
                # Check if this sample compares the models we're interested in
                left_model = annotations[0]['left_model']
                right_model = annotations[0]['right_model']
                
                if {left_model, right_model} != {model_a, model_b}:
                    continue
                
                # Aggregate scores
                for metric in ['coherence', 'efficiency', 'thoroughness']:
                    if left_model == model_a:
                        a_scores = [a[f'left_{metric}'] for a in annotations]
                        b_scores = [a[f'right_{metric}'] for a in annotations]
                    else:
                        a_scores = [a[f'right_{metric}'] for a in annotations]
                        b_scores = [a[f'left_{metric}'] for a in annotations]
                    
                    paired_scores[metric]['a'].append(np.mean(a_scores))
                    paired_scores[metric]['b'].append(np.mean(b_scores))
            
            # Perform Wilcoxon signed-rank test for each metric
            hypothesis_results = {}
            
            for metric, scores in paired_scores.items():
                if len(scores['a']) < 5:  # Need minimum samples
                    hypothesis_results[metric] = {
                        'error': 'Insufficient paired samples',
                        'n_samples': len(scores['a'])
                    }
                    continue
                
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(scores['a'], scores['b'])
                
                # Calculate effect size (rank-biserial correlation)
                n = len(scores['a'])
                r = 1 - (2 * statistic) / (n * (n + 1))
                
                # Cohen's d for paired samples
                diff = np.array(scores['b']) - np.array(scores['a'])
                cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                
                hypothesis_results[metric] = {
                    'n_samples': n,
                    'mean_a': np.mean(scores['a']),
                    'mean_b': np.mean(scores['b']),
                    'mean_difference': np.mean(scores['b']) - np.mean(scores['a']),
                    'wilcoxon_statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'rank_biserial_r': r,
                    'cohen_d': cohen_d,
                    'interpretation': self._interpret_effect_size(cohen_d)
                }
            
            test_results[hypothesis] = hypothesis_results
        
        return test_results
    
    def _interpret_effect_size(self, cohen_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohen_d: Cohen's d value
            
        Returns:
            String interpretation
        """
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "Negligible effect"
        elif abs_d < 0.5:
            return "Small effect"
        elif abs_d < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def generate_visualizations(self):
        """Generate visualization plots for the analysis."""
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Model comparison box plots
        model_df = self.compare_models()
        
        for i, metric in enumerate(['coherence', 'efficiency', 'thoroughness'], 1):
            ax = plt.subplot(3, 3, i)
            
            # Collect data for box plot
            plot_data = []
            labels = []
            
            for model in self.model_names:
                model_scores = []
                for annotations in self.grouped_annotations.values():
                    if len(annotations) < self.min_annotations:
                        continue
                    
                    for ann in annotations:
                        if ann['left_model'] == model:
                            model_scores.append(ann[f'left_{metric}'])
                        if ann['right_model'] == model:
                            model_scores.append(ann[f'right_{metric}'])
                
                if model_scores:
                    plot_data.append(model_scores)
                    labels.append(model.replace('Pixelis-', '').replace('Pixel-Reasoner-', 'PR-'))
            
            if plot_data:
                bp = ax.boxplot(plot_data, labels=labels)
                ax.set_title(f'{metric.capitalize()} Scores by Model')
                ax.set_ylabel('Score (1-5)')
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
        
        # 2. Inter-annotator agreement visualization
        agreement_results = self.analyze_inter_annotator_agreement()
        
        ax = plt.subplot(3, 3, 4)
        metrics = []
        icc_values = []
        kappa_values = []
        
        for metric in ['coherence', 'efficiency', 'thoroughness']:
            if metric in agreement_results:
                metrics.append(metric.capitalize())
                icc_values.append(agreement_results[metric]['icc'])
                kappa_values.append(agreement_results[metric]['fleiss_kappa'])
        
        if metrics:
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, icc_values, width, label='ICC', color='skyblue')
            ax.bar(x + width/2, kappa_values, width, label="Fleiss' κ", color='lightcoral')
            
            ax.set_xlabel('Metric')
            ax.set_ylabel('Agreement Score')
            ax.set_title('Inter-Annotator Agreement')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add interpretation lines
            ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Fair')
            ax.axhline(y=0.6, color='y', linestyle='--', alpha=0.5, label='Good')
            ax.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='Excellent')
        
        # 3. Preference distribution
        ax = plt.subplot(3, 3, 5)
        preference_counts = defaultdict(int)
        
        for annotations in self.grouped_annotations.values():
            for ann in annotations:
                preference_counts[ann['preference']] += 1
        
        if preference_counts:
            labels = list(preference_counts.keys())
            sizes = list(preference_counts.values())
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Overall Preference Distribution')
        
        # 4. Confidence distribution
        ax = plt.subplot(3, 3, 6)
        confidence_scores = []
        
        for annotations in self.grouped_annotations.values():
            for ann in annotations:
                confidence_scores.append(ann['confidence'])
        
        if confidence_scores:
            ax.hist(confidence_scores, bins=5, range=(1, 6), edgecolor='black', alpha=0.7)
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Annotator Confidence Distribution')
            ax.grid(True, alpha=0.3)
        
        # 5. Time spent distribution
        ax = plt.subplot(3, 3, 7)
        time_spent = []
        
        for annotations in self.grouped_annotations.values():
            for ann in annotations:
                if ann.get('time_spent_seconds'):
                    time_spent.append(ann['time_spent_seconds'] / 60)  # Convert to minutes
        
        if time_spent:
            ax.hist(time_spent, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Time Spent (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title('Time per Annotation Distribution')
            ax.axvline(x=np.median(time_spent), color='r', linestyle='--', 
                      label=f'Median: {np.median(time_spent):.1f} min')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Model win rates
        ax = plt.subplot(3, 3, 8)
        model_wins = defaultdict(int)
        model_comparisons = defaultdict(int)
        
        for annotations in self.grouped_annotations.values():
            for ann in annotations:
                left_model = ann['left_model']
                right_model = ann['right_model']
                
                model_comparisons[left_model] += 1
                model_comparisons[right_model] += 1
                
                if ann['preference'] == 'left':
                    model_wins[left_model] += 1
                elif ann['preference'] == 'right':
                    model_wins[right_model] += 1
                else:  # equal
                    model_wins[left_model] += 0.5
                    model_wins[right_model] += 0.5
        
        if model_wins:
            models = list(model_wins.keys())
            win_rates = [model_wins[m] / model_comparisons[m] * 100 for m in models]
            models = [m.replace('Pixelis-', '').replace('Pixel-Reasoner-', 'PR-') for m in models]
            
            bars = ax.bar(models, win_rates)
            ax.set_xlabel('Model')
            ax.set_ylabel('Win Rate (%)')
            ax.set_title('Model Preference Win Rates')
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Color bars based on performance
            for bar, rate in zip(bars, win_rates):
                if rate > 60:
                    bar.set_color('green')
                elif rate > 40:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        # 7. Correlation matrix
        ax = plt.subplot(3, 3, 9)
        
        # Collect all scores for correlation
        all_scores = []
        for annotations in self.grouped_annotations.values():
            for ann in annotations:
                all_scores.append([
                    ann['left_coherence'],
                    ann['left_efficiency'],
                    ann['left_thoroughness'],
                    ann['right_coherence'],
                    ann['right_efficiency'],
                    ann['right_thoroughness']
                ])
        
        if all_scores:
            scores_df = pd.DataFrame(all_scores, columns=[
                'L_Coherence', 'L_Efficiency', 'L_Thoroughness',
                'R_Coherence', 'R_Efficiency', 'R_Thoroughness'
            ])
            
            corr_matrix = scores_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Metric Correlation Matrix')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / 'evaluation_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
        
        plt.close()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Generating comprehensive analysis report...")
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_annotations': len(self.annotations),
                'unique_samples': len(self.grouped_annotations),
                'models_evaluated': self.model_names,
                'min_annotations_per_sample': self.min_annotations
            }
        }
        
        # Inter-annotator agreement
        logger.info("Calculating inter-annotator agreement...")
        report['inter_annotator_agreement'] = self.analyze_inter_annotator_agreement()
        
        # Model comparison
        logger.info("Comparing models...")
        model_comparison = self.compare_models()
        report['model_comparison'] = model_comparison.to_dict('records')
        
        # Statistical tests
        logger.info("Performing statistical tests...")
        report['hypothesis_tests'] = self.perform_statistical_tests()
        
        # Disagreement cases
        logger.info("Identifying disagreement cases...")
        disagreement_cases = self.identify_disagreement_cases()
        report['disagreement_cases'] = {
            'total_cases': len(disagreement_cases),
            'cases': disagreement_cases[:10]  # Include top 10 for review
        }
        
        # Annotator statistics
        annotator_stats = self._calculate_annotator_statistics()
        report['annotator_statistics'] = annotator_stats
        
        # Generate summary
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _calculate_annotator_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics for individual annotators.
        
        Returns:
            Dictionary with annotator statistics
        """
        annotator_data = defaultdict(lambda: {
            'n_annotations': 0,
            'total_time': 0,
            'mean_confidence': [],
            'rating_distribution': defaultdict(int)
        })
        
        for ann in self.annotations:
            annotator_id = ann['annotator_id']
            annotator_data[annotator_id]['n_annotations'] += 1
            
            if ann.get('time_spent_seconds'):
                annotator_data[annotator_id]['total_time'] += ann['time_spent_seconds']
            
            annotator_data[annotator_id]['mean_confidence'].append(ann['confidence'])
            
            # Track rating distribution
            for metric in ['coherence', 'efficiency', 'thoroughness']:
                for side in ['left', 'right']:
                    rating = ann[f'{side}_{metric}']
                    annotator_data[annotator_id]['rating_distribution'][rating] += 1
        
        # Calculate summary statistics
        summary = {}
        for annotator_id, data in annotator_data.items():
            summary[annotator_id] = {
                'n_annotations': data['n_annotations'],
                'mean_time_minutes': data['total_time'] / (60 * data['n_annotations']) if data['n_annotations'] > 0 else 0,
                'mean_confidence': np.mean(data['mean_confidence']) if data['mean_confidence'] else 0,
                'rating_std': np.std(list(data['rating_distribution'].keys())) if data['rating_distribution'] else 0
            }
        
        return summary
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary of findings.
        
        Args:
            report: Full analysis report
            
        Returns:
            Dictionary with summary findings
        """
        summary = {
            'key_findings': [],
            'hypotheses_results': {},
            'recommendations': []
        }
        
        # Check inter-annotator agreement
        agreement = report['inter_annotator_agreement']
        for metric in ['coherence', 'efficiency', 'thoroughness']:
            if metric in agreement:
                icc = agreement[metric]['icc']
                interpretation = agreement[metric]['interpretation']
                summary['key_findings'].append(
                    f"{metric.capitalize()} shows {interpretation} (ICC={icc:.3f})"
                )
        
        # Check hypothesis tests
        for hypothesis, results in report['hypothesis_tests'].items():
            hypothesis_summary = {}
            for metric, test_result in results.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    hypothesis_summary[metric] = {
                        'significant': test_result['significant'],
                        'p_value': test_result['p_value'],
                        'effect_size': test_result.get('interpretation', 'Unknown')
                    }
            summary['hypotheses_results'][hypothesis] = hypothesis_summary
        
        # Generate recommendations
        if report['disagreement_cases']['total_cases'] > 0:
            summary['recommendations'].append(
                f"Review {report['disagreement_cases']['total_cases']} high-disagreement cases for adjudication"
            )
        
        # Check if any model significantly outperforms others
        if report.get('model_comparison'):
            best_models = {}
            for metric in ['coherence', 'efficiency', 'thoroughness']:
                metric_key = f'{metric}_mean'
                best_score = 0
                best_model = None
                
                for model_data in report['model_comparison']:
                    if metric_key in model_data and model_data[metric_key] > best_score:
                        best_score = model_data[metric_key]
                        best_model = model_data['model']
                
                if best_model:
                    best_models[metric] = (best_model, best_score)
            
            for metric, (model, score) in best_models.items():
                summary['key_findings'].append(
                    f"{model} achieves highest {metric} score ({score:.2f}/5.0)"
                )
        
        return summary
    
    def save_report(self, report: Dict[str, Any]):
        """
        Save the analysis report to multiple formats.
        
        Args:
            report: Analysis report dictionary
        """
        # Save as JSON
        json_path = self.output_dir / 'analysis_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved JSON report to {json_path}")
        
        # Save as human-readable text
        text_path = self.output_dir / 'analysis_report.txt'
        with open(text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HUMAN EVALUATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-" * 40 + "\n")
            for key, value in report['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Inter-annotator agreement
            f.write("INTER-ANNOTATOR AGREEMENT\n")
            f.write("-" * 40 + "\n")
            for metric, results in report['inter_annotator_agreement'].items():
                if isinstance(results, dict) and 'icc' in results:
                    f.write(f"\n{metric.upper()}:\n")
                    f.write(f"  ICC: {results['icc']:.3f} (CI: {results['icc_ci'][0]:.3f}-{results['icc_ci'][1]:.3f})\n")
                    f.write(f"  Fleiss' Kappa: {results['fleiss_kappa']:.3f}\n")
                    f.write(f"  Interpretation: {results['interpretation']}\n")
            f.write("\n")
            
            # Hypothesis tests
            f.write("HYPOTHESIS TEST RESULTS\n")
            f.write("-" * 40 + "\n")
            for hypothesis, metrics in report['hypothesis_tests'].items():
                f.write(f"\n{hypothesis}:\n")
                for metric, results in metrics.items():
                    if isinstance(results, dict) and 'p_value' in results:
                        f.write(f"  {metric}:\n")
                        f.write(f"    p-value: {results['p_value']:.4f} {'(SIGNIFICANT)' if results['significant'] else '(not significant)'}\n")
                        f.write(f"    Effect size: {results['interpretation']}\n")
                        f.write(f"    Mean difference: {results['mean_difference']:.3f}\n")
            f.write("\n")
            
            # Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            summary = report['summary']
            
            f.write("\nKey Findings:\n")
            for finding in summary['key_findings']:
                f.write(f"  • {finding}\n")
            
            f.write("\nRecommendations:\n")
            for rec in summary['recommendations']:
                f.write(f"  • {rec}\n")
        
        logger.info(f"Saved text report to {text_path}")
        
        # Save model comparison as CSV
        if 'model_comparison' in report:
            csv_path = self.output_dir / 'model_comparison.csv'
            df = pd.DataFrame(report['model_comparison'])
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved model comparison to {csv_path}")


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(description="Analyze human evaluation results")
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing annotation result files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--min-annotations',
        type=int,
        default=3,
        help='Minimum annotations required per sample'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = HumanEvalAnalyzer(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        min_annotations_per_sample=args.min_annotations
    )
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report
    analyzer.save_report(report)
    
    # Generate visualizations if requested
    if args.generate_plots:
        logger.info("Generating visualizations...")
        analyzer.generate_visualizations()
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal annotations analyzed: {len(analyzer.annotations)}")
    print(f"Unique samples: {len(analyzer.grouped_annotations)}")
    print(f"Models evaluated: {', '.join(analyzer.model_names)}")
    
    # Print key findings
    if 'summary' in report:
        print("\nKEY FINDINGS:")
        for finding in report['summary']['key_findings'][:5]:
            print(f"  • {finding}")
    
    print(f"\nFull report saved to: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())