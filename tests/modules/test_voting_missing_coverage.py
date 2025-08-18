"""
Tests for voting.py to achieve 100% coverage

This test file targets the 8 missing statements in voting.py
to achieve complete code coverage, including edge cases and error conditions.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, '/Users/clayka7/Documents/Pixelis')

from core.modules.voting import TemporalEnsembleVoting
from core.data_structures import Experience, VotingResult, Trajectory


class TestMissingCoverageWeightedVoting(unittest.TestCase):
    """Test weighted voting edge cases for missing coverage."""
    
    def test_weighted_voting_zero_total_weight(self):
        """Test line 255: normalized_weights when total_weight is 0."""
        voting = TemporalEnsembleVoting(strategy="weighted")
        
        # Create votes with zero weights
        votes = [
            {'answer': 'A', 'confidence': 0.8, 'source': 'test1', 'trajectory': []},
            {'answer': 'B', 'confidence': 0.7, 'source': 'test2', 'trajectory': []},
            {'answer': 'A', 'confidence': 0.9, 'source': 'test3', 'trajectory': []}
        ]
        weights = [0.0, 0.0, 0.0]  # All weights are zero
        
        initial_prediction = {'answer': 'A', 'confidence': 0.8}
        neighbors = []
        
        # This should hit line 255 (else branch)
        result = voting._weighted_voting(votes, weights, initial_prediction, neighbors)
        
        # Verify the result
        self.assertIsInstance(result, VotingResult)
        self.assertIsNotNone(result.final_answer)
        # Check that equal weights were used (1/3 each)
        self.assertEqual(len(result.provenance['weights']), 3)
        for weight in result.provenance['weights']:
            self.assertAlmostEqual(weight, 1.0 / 3)


class TestMissingCoverageConfidenceVoting(unittest.TestCase):
    """Test confidence voting edge cases for missing coverage."""
    
    def test_confidence_voting_no_confident_votes(self):
        """Test lines 314-315: no confident votes fallback."""
        voting = TemporalEnsembleVoting(
            strategy="confidence",
            confidence_threshold=0.9  # High threshold
        )
        
        # Create votes with low confidence (below threshold)
        votes = [
            {'answer': 'A', 'confidence': 0.3, 'source': 'test1', 'trajectory': []},
            {'answer': 'B', 'confidence': 0.5, 'source': 'test2', 'trajectory': []},
            {'answer': 'C', 'confidence': 0.4, 'source': 'test3', 'trajectory': []}
        ]
        weights = [1.0, 1.0, 1.0]
        
        initial_prediction = {'answer': 'B', 'confidence': 0.5}
        neighbors = []
        
        # This should hit lines 314-315 (no confident votes branch)
        result = voting._confidence_voting(votes, weights, initial_prediction, neighbors)
        
        # Verify the result
        self.assertIsInstance(result, VotingResult)
        self.assertEqual(result.final_answer, 'B')  # Should pick highest confidence vote
        self.assertLess(result.confidence, 0.5)  # Confidence should be reduced
        self.assertEqual(result.provenance['reason'], 'no_confident_votes')
        self.assertTrue(result.provenance['fallback'])


class TestMissingCoverageAgreementFactor(unittest.TestCase):
    """Test agreement factor calculation edge cases."""
    
    def test_calculate_agreement_factor_empty_votes(self):
        """Test line 410: return 0.0 when votes is empty."""
        voting = TemporalEnsembleVoting()
        
        # Empty votes list
        votes = []
        final_answer = "A"
        
        # This should hit line 410
        factor = voting._calculate_agreement_factor(votes, final_answer)
        
        # Should return 0.0 for empty votes
        self.assertEqual(factor, 0.0)


class TestMissingCoverageVotingConsistency(unittest.TestCase):
    """Test voting consistency analysis edge cases."""
    
    def test_analyze_voting_consistency_empty_history(self):
        """Test line 444: return error when voting_history is empty."""
        voting = TemporalEnsembleVoting()
        
        # Empty voting history
        voting_history = []
        
        # This should hit line 444
        result = voting.analyze_voting_consistency(voting_history)
        
        # Should return error dictionary
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No voting history available')
    
    def test_analyze_voting_consistency_low_confidence_warning(self):
        """Test line 463: warning for high rate of low confidence votes."""
        voting = TemporalEnsembleVoting()
        
        # Create voting history with many low confidence results
        voting_history = []
        for i in range(10):
            # 4 out of 10 have low confidence (40% > 30% threshold)
            confidence = 0.3 if i < 4 else 0.8
            result = VotingResult(
                final_answer=f"answer_{i}",
                confidence=confidence,
                provenance={
                    'test': True,
                    'model_self_answer': f"answer_{i}",
                    'neighbor_answers': [],
                    'votes': [{'answer': f"answer_{i}", 'confidence': confidence}],
                    'weights': [1.0]
                }
            )
            voting_history.append(result)
        
        # This should hit line 463 (warning case)
        analysis = voting.analyze_voting_consistency(voting_history)
        
        # Should have warning
        self.assertIn('warning', analysis)
        self.assertIn('High rate of low confidence votes', analysis['warning'])
        self.assertIn('40.00%', analysis['warning'])


class TestMissingCoverageTrendCalculation(unittest.TestCase):
    """Test trend calculation edge cases."""
    
    def test_calculate_trend_insufficient_data(self):
        """Test line 478: return 'insufficient_data' when values length < 2."""
        voting = TemporalEnsembleVoting()
        
        # Single value (insufficient for trend)
        values = [0.5]
        
        # This should hit line 478
        trend = voting._calculate_trend(values)
        
        # Should return insufficient_data
        self.assertEqual(trend, "insufficient_data")
        
        # Test with empty list
        values = []
        trend = voting._calculate_trend(values)
        self.assertEqual(trend, "insufficient_data")
    
    def test_calculate_trend_decreasing(self):
        """Test line 489: return 'decreasing' for negative slope."""
        voting = TemporalEnsembleVoting()
        
        # Decreasing values
        values = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        # This should hit line 489 (decreasing branch)
        trend = voting._calculate_trend(values)
        
        # Should return decreasing
        self.assertEqual(trend, "decreasing")
    
    def test_calculate_trend_stable(self):
        """Test stable trend (small slope)."""
        voting = TemporalEnsembleVoting()
        
        # Nearly constant values (small variations)
        values = [0.5, 0.501, 0.499, 0.502, 0.498]
        
        # Should return stable
        trend = voting._calculate_trend(values)
        self.assertEqual(trend, "stable")
    
    def test_calculate_trend_increasing(self):
        """Test increasing trend."""
        voting = TemporalEnsembleVoting()
        
        # Increasing values
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Should return increasing
        trend = voting._calculate_trend(values)
        self.assertEqual(trend, "increasing")


class TestMissingCoverageFullIntegration(unittest.TestCase):
    """Integration tests to ensure all paths are covered."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.voting = TemporalEnsembleVoting(
            strategy="weighted",
            min_votes_required=2,
            confidence_threshold=0.5
        )
    
    def test_vote_with_zero_weight_neighbors(self):
        """Test voting with neighbors that produce zero weights."""
        # Create initial prediction
        initial_prediction = {
            'answer': 'A',
            'confidence': 0.0  # Zero confidence
        }
        
        # Create neighbors with conditions that lead to zero weights
        neighbors = []
        for i in range(3):
            trajectory = Trajectory(
                total_reward=1.0,
                actions=[],
                final_answer=f'answer_{i}'
            )
            
            # Create experience with zero confidence
            experience = Experience(
                experience_id=f"exp_{i}",
                image_features=np.zeros((1, 768)),  # Mock features
                question_text="test question",
                trajectory=trajectory,
                model_confidence=0.0,  # Zero confidence
                timestamp=datetime.now() - timedelta(days=365)  # Very old
            )
            experience.priority = 0.0  # Zero priority
            # success_rate is a computed property, set success_count to 0
            experience.success_count = 0
            experience.retrieval_count = 1  # To avoid division by zero
            
            neighbors.append(experience)
        
        # Perform voting - should handle zero weights gracefully
        result = self.voting.vote(initial_prediction, neighbors)
        
        # Verify result is valid
        self.assertIsInstance(result, VotingResult)
        self.assertIsNotNone(result.final_answer)
    
    def test_confidence_voting_strategy_edge_cases(self):
        """Test confidence voting strategy with various edge cases."""
        # Test with confidence strategy
        voting = TemporalEnsembleVoting(
            strategy="confidence",
            min_votes_required=2,
            confidence_threshold=0.8  # High threshold
        )
        
        # Initial prediction with low confidence
        initial_prediction = {
            'answer': 'X',
            'confidence': 0.2
        }
        
        # Neighbors with low confidence
        neighbors = []
        for i in range(2):
            trajectory = Trajectory(
                total_reward=1.0,
                actions=[],
                final_answer=f'Y{i}'
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                image_features=np.zeros((1, 768)),  # Mock features
                question_text="test question",
                trajectory=trajectory,
                model_confidence=0.3  # Below threshold
            )
            neighbors.append(experience)
        
        # Should trigger no confident votes path
        result = voting.vote(initial_prediction, neighbors)
        
        # Verify fallback was used
        self.assertIsInstance(result, VotingResult)
        self.assertEqual(result.provenance.get('reason'), 'no_confident_votes')
    
    def test_ensemble_voting_complete_coverage(self):
        """Test ensemble voting to ensure all sub-strategies are covered."""
        voting = TemporalEnsembleVoting(
            strategy="ensemble",
            min_votes_required=2,
            confidence_threshold=0.5
        )
        
        # Create diverse votes to exercise all paths
        initial_prediction = {
            'answer': 'A',
            'confidence': 0.6
        }
        
        neighbors = []
        # Add neighbors with varying characteristics
        answers = ['A', 'B', 'A', 'C', 'A']
        confidences = [0.9, 0.3, 0.7, 0.4, 0.8]
        
        for i, (ans, conf) in enumerate(zip(answers, confidences)):
            trajectory = Trajectory(
                total_reward=1.0,
                actions=[],
                final_answer=ans
            )
            
            experience = Experience(
                experience_id=f"exp_{i}",
                image_features=np.zeros((1, 768)),  # Mock features
                question_text="test question",
                trajectory=trajectory,
                model_confidence=conf
            )
            neighbors.append(experience)
        
        # Perform ensemble voting
        result = voting.vote(initial_prediction, neighbors)
        
        # Verify ensemble result
        self.assertIsInstance(result, VotingResult)
        self.assertIn('sub_strategies', result.provenance)
        self.assertIn('answer_scores', result.provenance)
        self.assertIn('agreement', result.provenance)


if __name__ == '__main__':
    # Run with verbose output to see all test coverage
    unittest.main(verbosity=2)