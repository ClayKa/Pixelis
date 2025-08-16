#!/usr/bin/env python3
"""
Prepare Data for Human Evaluation

This script takes trajectory logs from model evaluations and prepares
blinded A/B pairs for human annotation. It samples questions, creates
model comparisons, and formats the data for the evaluation interface.
"""

import json
import random
import uuid
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations
from collections import defaultdict
import hashlib
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HumanEvalDataPreparator:
    """
    Prepare evaluation data for human annotation.
    
    Creates balanced comparisons between different model outputs,
    ensuring proper blinding and diverse sample selection.
    """
    
    def __init__(
        self,
        trajectory_dir: Path,
        output_path: Path,
        models_to_compare: List[str],
        sample_size: int = 100,
        random_seed: int = 42
    ):
        """
        Initialize the data preparator.
        
        Args:
            trajectory_dir: Directory containing model trajectory files
            output_path: Path to save the prepared evaluation data
            models_to_compare: List of model names to include in comparison
            sample_size: Number of questions to sample
            random_seed: Random seed for reproducibility
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.output_path = Path(output_path)
        self.models_to_compare = models_to_compare
        self.sample_size = sample_size
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load trajectories for each model
        self.model_trajectories = self._load_all_trajectories()
        
        # Get common questions across all models
        self.common_questions = self._find_common_questions()
        
        logger.info(f"Loaded trajectories for {len(self.model_trajectories)} models")
        logger.info(f"Found {len(self.common_questions)} common questions")
    
    def _load_all_trajectories(self) -> Dict[str, Dict[str, Any]]:
        """
        Load trajectory files for all specified models.
        
        Returns:
            Dictionary mapping model_name -> question_id -> trajectory_data
        """
        model_trajectories = {}
        
        for model_name in self.models_to_compare:
            # Look for trajectory file for this model
            trajectory_file = self.trajectory_dir / f"{model_name}_trajectories.json"
            
            if not trajectory_file.exists():
                logger.warning(f"Trajectory file not found for {model_name}: {trajectory_file}")
                continue
            
            with open(trajectory_file, 'r') as f:
                data = json.load(f)
            
            # Index by question ID
            trajectories_by_question = {}
            for item in data.get('trajectories', []):
                question_id = item.get('question_id')
                if question_id:
                    trajectories_by_question[question_id] = item
            
            model_trajectories[model_name] = trajectories_by_question
            logger.info(f"Loaded {len(trajectories_by_question)} trajectories for {model_name}")
        
        return model_trajectories
    
    def _find_common_questions(self) -> List[str]:
        """
        Find questions that have trajectories from all models.
        
        Returns:
            List of question IDs present in all model outputs
        """
        if not self.model_trajectories:
            return []
        
        # Get question sets for each model
        question_sets = [
            set(trajectories.keys())
            for trajectories in self.model_trajectories.values()
        ]
        
        # Find intersection
        common = set.intersection(*question_sets) if question_sets else set()
        return list(common)
    
    def _create_diverse_sample(
        self,
        questions: List[str],
        sample_size: int
    ) -> List[str]:
        """
        Create a diverse sample of questions.
        
        Attempts to balance different question types, difficulties,
        and visual operations required.
        
        Args:
            questions: Pool of available questions
            sample_size: Number of questions to sample
            
        Returns:
            List of selected question IDs
        """
        if len(questions) <= sample_size:
            return questions
        
        # Categorize questions based on metadata
        categories = defaultdict(list)
        
        for q_id in questions:
            # Get a sample trajectory to analyze question type
            sample_trajectory = None
            for model_data in self.model_trajectories.values():
                if q_id in model_data:
                    sample_trajectory = model_data[q_id]
                    break
            
            if sample_trajectory:
                # Categorize by operations used
                operations = set()
                for action in sample_trajectory.get('trajectory', []):
                    op = action.get('operation', 'unknown')
                    operations.add(op)
                
                # Simple categorization
                if 'SEGMENT_OBJECT_AT' in operations:
                    categories['segmentation'].append(q_id)
                elif 'READ_TEXT' in operations:
                    categories['ocr'].append(q_id)
                elif 'TRACK_OBJECT' in operations:
                    categories['tracking'].append(q_id)
                else:
                    categories['general'].append(q_id)
        
        # Sample proportionally from each category
        selected = []
        for category, items in categories.items():
            # Calculate proportion
            proportion = len(items) / len(questions)
            n_samples = max(1, int(sample_size * proportion))
            
            # Sample from this category
            sampled = random.sample(items, min(n_samples, len(items)))
            selected.extend(sampled)
        
        # If we need more samples, add random ones
        remaining = set(questions) - set(selected)
        if len(selected) < sample_size and remaining:
            additional = random.sample(
                list(remaining),
                min(sample_size - len(selected), len(remaining))
            )
            selected.extend(additional)
        
        # Shuffle final selection
        random.shuffle(selected)
        return selected[:sample_size]
    
    def _create_model_pairs(self, models: List[str]) -> List[Tuple[str, str]]:
        """
        Create pairs of models for comparison.
        
        Args:
            models: List of model names
            
        Returns:
            List of (model_a, model_b) tuples
        """
        # Generate all possible pairs
        all_pairs = list(combinations(models, 2))
        
        # For key comparisons, ensure certain pairs are included
        priority_pairs = []
        
        # Always compare against baseline
        if 'Pixel-Reasoner-Base' in models:
            for model in models:
                if model != 'Pixel-Reasoner-Base':
                    priority_pairs.append(('Pixel-Reasoner-Base', model))
        
        # Key ablation comparisons
        key_comparisons = [
            ('Pixelis-RFT-Base', 'Pixelis-RFT-Coherence'),
            ('Pixelis-RFT-Base', 'Pixelis-RFT-Curiosity'),
            ('Pixelis-RFT-Full', 'Pixelis-Online'),
            ('Pixelis-SFT-Only', 'Pixelis-RFT-Full'),
        ]
        
        for pair in key_comparisons:
            if pair[0] in models and pair[1] in models:
                priority_pairs.append(pair)
        
        # Combine priority and other pairs
        other_pairs = [p for p in all_pairs if p not in priority_pairs]
        
        return priority_pairs + other_pairs
    
    def _format_trajectory(self, trajectory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format a trajectory for the evaluation interface.
        
        Args:
            trajectory_data: Raw trajectory data from evaluation
            
        Returns:
            Formatted list of action dictionaries
        """
        formatted = []
        
        trajectory = trajectory_data.get('trajectory', [])
        for action in trajectory:
            formatted_action = {
                'operation': action.get('operation', 'Unknown'),
                'type': action.get('type', ''),
                'arguments': action.get('arguments', {}),
                'result': action.get('result', '')
            }
            
            # Add reasoning if present
            if 'reasoning' in action:
                formatted_action['reasoning'] = action['reasoning']
            
            formatted.append(formatted_action)
        
        return formatted
    
    def prepare_evaluation_data(self) -> Dict[str, Any]:
        """
        Prepare the complete evaluation dataset.
        
        Returns:
            Dictionary containing evaluation samples and metadata
        """
        # Select diverse sample of questions
        selected_questions = self._create_diverse_sample(
            self.common_questions,
            self.sample_size
        )
        
        logger.info(f"Selected {len(selected_questions)} questions for evaluation")
        
        # Create model pairs for comparison
        model_pairs = self._create_model_pairs(list(self.model_trajectories.keys()))
        
        # Create evaluation samples
        samples = []
        sample_id_counter = 0
        
        # Track which pairs have been used for each question
        used_pairs = defaultdict(set)
        
        for question_id in selected_questions:
            # Select model pairs for this question
            # Ensure diversity in comparisons
            available_pairs = [
                p for p in model_pairs
                if p not in used_pairs[question_id]
            ]
            
            if not available_pairs:
                available_pairs = model_pairs  # Reset if all pairs used
            
            # Select a pair for this question
            pair = random.choice(available_pairs)
            used_pairs[question_id].add(pair)
            
            model_a, model_b = pair
            
            # Get trajectories for both models
            traj_a = self.model_trajectories[model_a].get(question_id)
            traj_b = self.model_trajectories[model_b].get(question_id)
            
            if not traj_a or not traj_b:
                logger.warning(f"Missing trajectory for question {question_id}")
                continue
            
            # Create sample
            sample = {
                'sample_id': f"sample_{sample_id_counter:04d}",
                'question_id': question_id,
                'image_path': traj_a.get('image_path', ''),
                'question_text': traj_a.get('question_text', ''),
                'model_a_name': model_a,
                'model_b_name': model_b,
                'trajectory_a': self._format_trajectory(traj_a),
                'trajectory_b': self._format_trajectory(traj_b),
                'final_answer_a': traj_a.get('final_answer', ''),
                'final_answer_b': traj_b.get('final_answer', ''),
                'ground_truth': traj_a.get('ground_truth'),
                'metadata': {
                    'dataset': traj_a.get('dataset', 'unknown'),
                    'difficulty': traj_a.get('difficulty', 'unknown'),
                    'comparison_type': self._categorize_comparison(model_a, model_b)
                }
            }
            
            samples.append(sample)
            sample_id_counter += 1
        
        # Create metadata
        metadata = {
            'total_samples': len(samples),
            'total_questions': len(selected_questions),
            'models_compared': self.models_to_compare,
            'model_pairs': [list(p) for p in model_pairs],
            'random_seed': 42,
            'preparation_timestamp': str(Path.ctime(Path(__file__)))
        }
        
        # Calculate statistics
        comparison_stats = defaultdict(int)
        for sample in samples:
            comp_type = sample['metadata']['comparison_type']
            comparison_stats[comp_type] += 1
        
        metadata['comparison_statistics'] = dict(comparison_stats)
        
        logger.info(f"Prepared {len(samples)} evaluation samples")
        logger.info(f"Comparison statistics: {dict(comparison_stats)}")
        
        return {
            'samples': samples,
            'metadata': metadata
        }
    
    def _categorize_comparison(self, model_a: str, model_b: str) -> str:
        """
        Categorize the type of comparison being made.
        
        Args:
            model_a: First model name
            model_b: Second model name
            
        Returns:
            Category string
        """
        models = {model_a, model_b}
        
        # Baseline comparison
        if 'Pixel-Reasoner-Base' in models:
            return 'baseline_comparison'
        
        # SFT vs RL comparison
        if 'SFT' in model_a and 'RFT' in model_b:
            return 'sft_vs_rl'
        if 'RFT' in model_a and 'SFT' in model_b:
            return 'sft_vs_rl'
        
        # Ablation study
        if 'RFT-Base' in models and ('RFT-Coherence' in models or 'RFT-Curiosity' in models):
            return 'reward_ablation'
        
        # Online vs offline
        if 'Online' in model_a or 'Online' in model_b:
            return 'online_comparison'
        
        # Full model comparison
        if 'RFT-Full' in model_a and 'RFT-Full' in model_b:
            return 'full_model_comparison'
        
        return 'other'
    
    def save_evaluation_data(self, data: Dict[str, Any]):
        """
        Save the prepared evaluation data to file.
        
        Args:
            data: Evaluation data dictionary
        """
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved evaluation data to {self.output_path}")
        
        # Also create a summary file
        summary_path = self.output_path.parent / f"{self.output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Human Evaluation Data Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {data['metadata']['total_samples']}\n")
            f.write(f"Total unique questions: {data['metadata']['total_questions']}\n")
            f.write(f"Models compared: {', '.join(data['metadata']['models_compared'])}\n\n")
            f.write("Comparison Statistics:\n")
            for comp_type, count in data['metadata']['comparison_statistics'].items():
                f.write(f"  {comp_type}: {count}\n")
            f.write("\n")
            f.write("Model Pairs:\n")
            for pair in data['metadata']['model_pairs']:
                f.write(f"  {pair[0]} vs {pair[1]}\n")
        
        logger.info(f"Saved summary to {summary_path}")


def validate_trajectories(trajectory_dir: Path, models: List[str]) -> bool:
    """
    Validate that trajectory files exist for specified models.
    
    Args:
        trajectory_dir: Directory containing trajectory files
        models: List of model names
        
    Returns:
        True if all files exist, False otherwise
    """
    all_exist = True
    
    for model in models:
        trajectory_file = trajectory_dir / f"{model}_trajectories.json"
        if not trajectory_file.exists():
            logger.error(f"Missing trajectory file for {model}: {trajectory_file}")
            all_exist = False
        else:
            logger.info(f"Found trajectory file for {model}")
    
    return all_exist


def generate_mock_trajectories(output_dir: Path, models: List[str], n_questions: int = 50):
    """
    Generate mock trajectory data for testing the evaluation interface.
    
    Args:
        output_dir: Directory to save mock trajectories
        models: List of model names
        n_questions: Number of questions to generate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample operations
    operations = [
        'SEGMENT_OBJECT_AT',
        'READ_TEXT',
        'GET_PROPERTIES',
        'ZOOM_IN',
        'TRACK_OBJECT',
        'ANALYZE',
        'COMPARE',
        'FINAL_ANSWER'
    ]
    
    # Generate questions
    questions = []
    for i in range(n_questions):
        questions.append({
            'question_id': f"q_{i:04d}",
            'question_text': f"Sample question {i}: What is shown in this image?",
            'image_path': f"/path/to/image_{i:04d}.jpg",
            'ground_truth': f"answer_{i}",
            'dataset': random.choice(['mm-vet', 'mmmu', 'custom']),
            'difficulty': random.choice(['easy', 'medium', 'hard'])
        })
    
    # Generate trajectories for each model
    for model_name in models:
        trajectories = []
        
        for q in questions:
            # Generate random trajectory
            n_steps = random.randint(3, 8)
            trajectory = []
            
            for step in range(n_steps):
                if step == n_steps - 1:
                    # Last step is always final answer
                    operation = 'FINAL_ANSWER'
                else:
                    operation = random.choice(operations[:-1])
                
                action = {
                    'operation': operation,
                    'type': 'visual_operation' if operation in operations[:5] else 'reasoning',
                    'arguments': {
                        'x': random.randint(0, 500),
                        'y': random.randint(0, 500)
                    } if operation in ['SEGMENT_OBJECT_AT', 'ZOOM_IN'] else {},
                    'result': f"Result of {operation} operation",
                    'reasoning': f"Reasoning for {operation} step"
                }
                trajectory.append(action)
            
            # Create trajectory entry
            traj_entry = {
                'question_id': q['question_id'],
                'question_text': q['question_text'],
                'image_path': q['image_path'],
                'ground_truth': q['ground_truth'],
                'dataset': q['dataset'],
                'difficulty': q['difficulty'],
                'trajectory': trajectory,
                'final_answer': f"{model_name}_answer_{q['question_id']}",
                'model_confidence': random.uniform(0.6, 0.95)
            }
            trajectories.append(traj_entry)
        
        # Save trajectories
        output_file = output_dir / f"{model_name}_trajectories.json"
        with open(output_file, 'w') as f:
            json.dump({'trajectories': trajectories}, f, indent=2)
        
        logger.info(f"Generated mock trajectories for {model_name}: {output_file}")


def main():
    """Main entry point for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data for human evaluation")
    
    parser.add_argument(
        '--trajectory-dir',
        type=str,
        required=True,
        help='Directory containing model trajectory files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='human_eval_data.json',
        help='Output file for prepared evaluation data'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=[
            'Pixel-Reasoner-Base',
            'Pixelis-SFT-Only',
            'Pixelis-RFT-Base',
            'Pixelis-RFT-Coherence',
            'Pixelis-RFT-Curiosity',
            'Pixelis-RFT-Full',
            'Pixelis-Online'
        ],
        help='List of model names to compare'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of questions to sample for evaluation'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--generate-mock',
        action='store_true',
        help='Generate mock trajectory data for testing'
    )
    
    args = parser.parse_args()
    
    trajectory_dir = Path(args.trajectory_dir)
    
    # Generate mock data if requested
    if args.generate_mock:
        logger.info("Generating mock trajectory data for testing...")
        generate_mock_trajectories(trajectory_dir, args.models)
    
    # Validate trajectory files exist
    if not validate_trajectories(trajectory_dir, args.models):
        logger.error("Some trajectory files are missing. Please run evaluation first or use --generate-mock")
        return 1
    
    # Prepare evaluation data
    preparator = HumanEvalDataPreparator(
        trajectory_dir=trajectory_dir,
        output_path=Path(args.output),
        models_to_compare=args.models,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )
    
    # Prepare and save data
    evaluation_data = preparator.prepare_evaluation_data()
    preparator.save_evaluation_data(evaluation_data)
    
    logger.info("Data preparation complete!")
    return 0


if __name__ == "__main__":
    exit(main())