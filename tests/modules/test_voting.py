"""
Unit Tests for Voting Module

Tests the temporal ensemble voting module including all voting strategies,
confidence calculations, and provenance tracking.
"""

import unittest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from typing import List

from core.data_structures import (
    Experience, VotingResult, Trajectory, Action, ActionType, ExperienceStatus
)
from core.modules.voting import TemporalEnsembleVoting


class TestTemporalEnsembleVoting(unittest.TestCase):
    """Test suite for TemporalEnsembleVoting class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.voting = TemporalEnsembleVoting(
            strategy="weighted",
            min_votes_required=3,
            confidence_threshold=0.5
        )
        
        # Create mock experiences
        self.mock_neighbors = self._create_mock_neighbors()
        self.mock_initial_prediction = {
            'answer': 'cat',
            'confidence': 0.8,
            'trajectory': []
        }
    
    def _create_mock_neighbors(self) -> List[Experience]:
        """Create mock neighbor experiences for testing."""
        neighbors = []
        
        # Create 5 neighbors with varying answers and confidences
        answers = ['cat', 'cat', 'dog', 'cat', 'bird']
        confidences = [0.9, 0.85, 0.6, 0.7, 0.5]
        
        for i, (answer, confidence) in enumerate(zip(answers, confidences)):
            trajectory = Trajectory(
                actions=[
                    Action(
                        type=ActionType.ANSWER,
                        operation="FINAL_ANSWER",
                        result=answer
                    )
                ],
                final_answer=answer,
                total_reward=confidence
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                image_features=torch.randn(1, 512),
                question_text="What animal is this?",
                trajectory=trajectory,
                model_confidence=confidence,
                timestamp=datetime.now() - timedelta(hours=i),
                priority=1.0 + i * 0.1,
                retrieval_count=10 - i,
                success_count=8 - i
            )
            
            neighbors.append(experience)
        
        return neighbors
    
    def test_initialization(self):
        """Test voting module initialization."""
        # Test valid initialization
        voting = TemporalEnsembleVoting(
            strategy="majority",
            min_votes_required=2,
            confidence_threshold=0.6
        )
        self.assertEqual(voting.strategy, "majority")
        self.assertEqual(voting.min_votes_required, 2)
        self.assertEqual(voting.confidence_threshold, 0.6)
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            TemporalEnsembleVoting(strategy="invalid_strategy")
    
    def test_majority_voting(self):
        """Test majority voting strategy."""
        voting = TemporalEnsembleVoting(strategy="majority")
        
        result = voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors
        )
        
        # Cat appears 4 times (including initial), dog 1, bird 1
        self.assertEqual(result.final_answer, 'cat')
        self.assertAlmostEqual(result.confidence, 4/6, places=2)  # 4 out of 6 votes
        
        # Check provenance
        self.assertIn('model_self_answer', result.provenance)
        self.assertEqual(result.provenance['model_self_answer'], 'cat')
        self.assertEqual(result.provenance['voting_strategy'], 'majority')
        self.assertEqual(result.provenance['retrieved_neighbors_count'], 5)
        self.assertEqual(len(result.provenance['neighbor_answers']), 5)
    
    def test_weighted_voting(self):
        """Test weighted voting strategy."""
        voting = TemporalEnsembleVoting(strategy="weighted")
        
        result = voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors
        )
        
        # Weighted voting should favor 'cat' due to higher confidence scores
        self.assertEqual(result.final_answer, 'cat')
        self.assertGreater(result.confidence, 0.5)
        
        # Check provenance contains weights
        self.assertIn('weights', result.provenance)
        self.assertIn('answer_weights', result.provenance)
    
    def test_confidence_voting(self):
        """Test confidence-based voting strategy."""
        voting = TemporalEnsembleVoting(
            strategy="confidence",
            confidence_threshold=0.7
        )
        
        result = voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors
        )
        
        # Only votes with confidence >= 0.7 should be considered
        # That's initial (0.8), neighbor 0 (0.9), neighbor 1 (0.85), neighbor 3 (0.7)
        self.assertEqual(result.final_answer, 'cat')
        self.assertIn('neighbor_answers', result.provenance)
    
    def test_ensemble_voting(self):
        """Test ensemble voting strategy."""
        voting = TemporalEnsembleVoting(strategy="ensemble")
        
        result = voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors
        )
        
        # Ensemble should combine multiple strategies
        self.assertIsNotNone(result.final_answer)
        self.assertIn('sub_strategies', result.provenance)
        self.assertIn('majority', result.provenance['sub_strategies'])
        self.assertIn('weighted', result.provenance['sub_strategies'])
        self.assertIn('confidence', result.provenance['sub_strategies'])
    
    def test_insufficient_votes(self):
        """Test handling of insufficient votes."""
        voting = TemporalEnsembleVoting(
            strategy="weighted",
            min_votes_required=10  # More than we have
        )
        
        result = voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors[:1]  # Only 1 neighbor
        )
        
        # Should return initial prediction with reduced confidence
        self.assertEqual(result.final_answer, 'cat')
        self.assertLess(result.confidence, self.mock_initial_prediction['confidence'])
        self.assertEqual(result.provenance['reason'], 'insufficient_votes')
    
    def test_calculate_vote_weight(self):
        """Test vote weight calculation."""
        neighbor = self.mock_neighbors[0]
        weight = self.voting._calculate_vote_weight(
            neighbor,
            self.mock_initial_prediction
        )
        
        # Weight should be positive and consider multiple factors
        self.assertGreater(weight, 0)
        self.assertLessEqual(weight, 2.0)  # Reasonable upper bound
    
    def test_calculate_agreement_factor(self):
        """Test agreement factor calculation."""
        votes = [
            {'answer': 'cat'},
            {'answer': 'cat'},
            {'answer': 'cat'},
            {'answer': 'dog'}
        ]
        
        # High agreement (75% for cat)
        factor = self.voting._calculate_agreement_factor(votes, 'cat')
        self.assertGreater(factor, 0.7)
        self.assertLessEqual(factor, 1.0)
        
        # Low agreement (25% for dog)
        factor = self.voting._calculate_agreement_factor(votes, 'dog')
        self.assertLess(factor, 0.7)
        self.assertGreaterEqual(factor, 0.5)
    
    def test_voting_result_validation(self):
        """Test VotingResult validation and methods."""
        result = VotingResult(
            final_answer='cat',
            confidence=0.85,
            provenance={
                'model_self_answer': 'cat',
                'retrieved_neighbors_count': 5,
                'neighbor_answers': [
                    {'answer': 'cat', 'confidence': 0.9},
                    {'answer': 'dog', 'confidence': 0.6}
                ],
                'voting_strategy': 'weighted'
            }
        )
        
        # Test get_vote_distribution
        distribution = result.get_vote_distribution()
        self.assertIn('cat', distribution)
        self.assertEqual(distribution['cat'], 2)  # model + 1 neighbor
        self.assertEqual(distribution['dog'], 1)
        
        # Test get_consensus_strength
        strength = result.get_consensus_strength()
        self.assertAlmostEqual(strength, 2/3, places=2)  # 2 cats out of 3 total
        
        # Test to_dict
        result_dict = result.to_dict()
        self.assertEqual(result_dict['final_answer'], 'cat')
        self.assertEqual(result_dict['confidence'], 0.85)
        self.assertIn('consensus_strength', result_dict)
    
    def test_analyze_voting_consistency(self):
        """Test voting consistency analysis."""
        # Create voting history
        history = []
        for i in range(5):
            result = VotingResult(
                final_answer='cat',
                confidence=0.7 + i * 0.05,
                provenance={
                    'model_self_answer': 'cat',
                    'retrieved_neighbors_count': 5,
                    'neighbor_answers': [],
                    'voting_strategy': 'weighted'
                }
            )
            history.append(result)
        
        analysis = self.voting.analyze_voting_consistency(history)
        
        # Check analysis results
        self.assertEqual(analysis['num_votes'], 5)
        self.assertIn('avg_confidence', analysis)
        self.assertIn('std_confidence', analysis)
        self.assertIn('confidence_trend', analysis)
        self.assertEqual(analysis['confidence_trend'], 'increasing')
    
    def test_empty_neighbors(self):
        """Test voting with no neighbors."""
        result = self.voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=[]
        )
        
        # Should use only initial prediction
        self.assertEqual(result.final_answer, 'cat')
        self.assertEqual(result.provenance['retrieved_neighbors_count'], 0)
        self.assertEqual(len(result.provenance['neighbor_answers']), 0)
    
    def test_missing_trajectory_data(self):
        """Test handling of neighbors with missing trajectory data."""
        # Create neighbors with None trajectory
        bad_neighbor = Experience(
            experience_id="bad_exp",
            image_features=torch.randn(1, 512),
            question_text="Test",
            trajectory=Trajectory(),  # Empty trajectory
            model_confidence=0.5
        )
        
        result = self.voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=[bad_neighbor]
        )
        
        # Should handle gracefully
        self.assertIsNotNone(result)
        self.assertEqual(result.provenance['retrieved_neighbors_count'], 1)
    
    def test_provenance_required_fields(self):
        """Test that all required provenance fields are present."""
        result = self.voting.vote(
            initial_prediction=self.mock_initial_prediction,
            neighbors=self.mock_neighbors
        )
        
        required_fields = [
            'model_self_answer',
            'retrieved_neighbors_count',
            'neighbor_answers',
            'voting_strategy'
        ]
        
        for field in required_fields:
            self.assertIn(field, result.provenance)
            self.assertIsNotNone(result.provenance[field])
    
    def test_confidence_bounds(self):
        """Test that confidence values are within valid bounds."""
        strategies = ['majority', 'weighted', 'confidence', 'ensemble']
        
        for strategy in strategies:
            voting = TemporalEnsembleVoting(strategy=strategy)
            result = voting.vote(
                initial_prediction=self.mock_initial_prediction,
                neighbors=self.mock_neighbors
            )
            
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()