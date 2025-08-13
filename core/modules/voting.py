"""
Voting Module

Implements temporal ensemble voting with multiple strategies for
confidence-based decision making in the online learning system.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import Counter
from ..data_structures import Experience, VotingResult

logger = logging.getLogger(__name__)


class TemporalEnsembleVoting:
    """
    Temporal ensemble voting module for aggregating predictions.
    
    Supports multiple voting strategies:
    - Majority voting
    - Weighted voting based on confidence/similarity
    - Confidence-based voting
    - Ensemble methods
    """
    
    def __init__(
        self,
        strategy: str = "weighted",
        min_votes_required: int = 3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize voting module.
        
        Args:
            strategy: Voting strategy ('majority', 'weighted', 'confidence', 'ensemble')
            min_votes_required: Minimum number of votes needed
            confidence_threshold: Minimum confidence for vote validity
        """
        self.strategy = strategy
        self.min_votes_required = min_votes_required
        self.confidence_threshold = confidence_threshold
        
        # Strategy implementations
        self.strategies = {
            'majority': self._majority_voting,
            'weighted': self._weighted_voting,
            'confidence': self._confidence_voting,
            'ensemble': self._ensemble_voting
        }
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown voting strategy: {strategy}")
        
        logger.info(f"Voting module initialized with strategy: {strategy}")
    
    def vote(
        self,
        initial_prediction: Dict[str, Any],
        neighbors: List[Experience],
        **kwargs
    ) -> VotingResult:
        """
        Perform voting based on initial prediction and neighbors.
        
        Args:
            initial_prediction: Model's initial prediction
            neighbors: List of neighbor experiences
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            VotingResult with final answer and metadata
        """
        # Collect all votes
        votes = []
        weights = []
        
        # Add initial prediction as a vote
        votes.append({
            'answer': initial_prediction.get('answer'),
            'confidence': initial_prediction.get('confidence', 1.0),
            'source': 'initial_prediction',
            'trajectory': initial_prediction.get('trajectory', [])
        })
        weights.append(initial_prediction.get('confidence', 1.0))
        
        # Add neighbor votes
        for neighbor in neighbors:
            if neighbor.trajectory and neighbor.trajectory.final_answer is not None:
                vote = {
                    'answer': neighbor.trajectory.final_answer,
                    'confidence': neighbor.model_confidence,
                    'source': f'neighbor_{neighbor.experience_id}',
                    'trajectory': neighbor.trajectory
                }
                votes.append(vote)
                
                # Calculate weight based on various factors
                weight = self._calculate_vote_weight(neighbor, initial_prediction)
                weights.append(weight)
        
        # Check if we have enough votes
        if len(votes) < self.min_votes_required:
            logger.warning(f"Insufficient votes: {len(votes)} < {self.min_votes_required}")
            # Return initial prediction with low confidence
            return VotingResult(
                final_answer=initial_prediction.get('answer'),
                confidence=initial_prediction.get('confidence', 0.5) * 0.5,
                votes=votes,
                weights=weights,
                provenance={
                    'strategy': self.strategy,
                    'reason': 'insufficient_votes',
                    'num_votes': len(votes)
                }
            )
        
        # Apply voting strategy
        strategy_func = self.strategies[self.strategy]
        result = strategy_func(votes, weights, **kwargs)
        
        return result
    
    def _calculate_vote_weight(
        self,
        neighbor: Experience,
        initial_prediction: Dict[str, Any]
    ) -> float:
        """
        Calculate weight for a neighbor's vote.
        
        Args:
            neighbor: Neighbor experience
            initial_prediction: Initial prediction for reference
            
        Returns:
            Weight value
        """
        weight = 1.0
        
        # Factor 1: Model confidence
        weight *= neighbor.model_confidence
        
        # Factor 2: Success rate (if available)
        if neighbor.success_rate > 0:
            weight *= (0.5 + 0.5 * neighbor.success_rate)
        
        # Factor 3: Recency (newer experiences get higher weight)
        from datetime import datetime
        age_hours = (datetime.now() - neighbor.timestamp).total_seconds() / 3600
        recency_factor = np.exp(-age_hours / 24)  # Decay over 24 hours
        weight *= recency_factor
        
        # Factor 4: Priority score
        weight *= (0.5 + 0.5 * min(neighbor.priority, 2.0) / 2.0)
        
        return weight
    
    def _majority_voting(
        self,
        votes: List[Dict[str, Any]],
        weights: List[float],
        **kwargs
    ) -> VotingResult:
        """
        Simple majority voting.
        
        Args:
            votes: List of votes
            weights: Vote weights (ignored for majority)
            
        Returns:
            VotingResult
        """
        # Count votes
        answer_counts = Counter()
        for vote in votes:
            answer = str(vote['answer'])
            answer_counts[answer] += 1
        
        # Get most common answer
        most_common = answer_counts.most_common(1)[0]
        final_answer = most_common[0]
        vote_count = most_common[1]
        
        # Calculate confidence
        confidence = vote_count / len(votes)
        
        # Create result
        return VotingResult(
            final_answer=final_answer,
            confidence=confidence,
            votes=votes,
            weights=[1.0] * len(votes),  # Equal weights for majority voting
            provenance={
                'strategy': 'majority',
                'vote_distribution': dict(answer_counts),
                'winning_votes': vote_count,
                'total_votes': len(votes)
            }
        )
    
    def _weighted_voting(
        self,
        votes: List[Dict[str, Any]],
        weights: List[float],
        **kwargs
    ) -> VotingResult:
        """
        Weighted voting based on confidence and similarity.
        
        Args:
            votes: List of votes
            weights: Vote weights
            
        Returns:
            VotingResult
        """
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(votes)] * len(votes)
        
        # Aggregate weighted votes
        answer_weights = {}
        for vote, weight in zip(votes, normalized_weights):
            answer = str(vote['answer'])
            answer_weights[answer] = answer_weights.get(answer, 0.0) + weight
        
        # Get answer with highest weight
        final_answer = max(answer_weights, key=answer_weights.get)
        confidence = answer_weights[final_answer]
        
        # Additional confidence adjustment based on agreement
        agreement_factor = self._calculate_agreement_factor(votes, final_answer)
        confidence *= agreement_factor
        
        return VotingResult(
            final_answer=final_answer,
            confidence=confidence,
            votes=votes,
            weights=normalized_weights,
            provenance={
                'strategy': 'weighted',
                'answer_weights': answer_weights,
                'agreement_factor': agreement_factor,
                'effective_votes': sum(1 for v in votes if str(v['answer']) == final_answer)
            }
        )
    
    def _confidence_voting(
        self,
        votes: List[Dict[str, Any]],
        weights: List[float],
        **kwargs
    ) -> VotingResult:
        """
        Voting based primarily on confidence scores.
        
        Args:
            votes: List of votes
            weights: Vote weights
            
        Returns:
            VotingResult
        """
        # Filter votes by confidence threshold
        confident_votes = []
        confident_weights = []
        
        for vote, weight in zip(votes, weights):
            if vote.get('confidence', 0) >= self.confidence_threshold:
                confident_votes.append(vote)
                confident_weights.append(weight * vote['confidence'])
        
        if not confident_votes:
            # No confident votes, fall back to highest confidence
            best_vote = max(votes, key=lambda v: v.get('confidence', 0))
            return VotingResult(
                final_answer=best_vote['answer'],
                confidence=best_vote.get('confidence', 0.5) * 0.5,
                votes=votes,
                weights=weights,
                provenance={
                    'strategy': 'confidence',
                    'reason': 'no_confident_votes',
                    'fallback': True
                }
            )
        
        # Apply weighted voting on confident votes
        return self._weighted_voting(confident_votes, confident_weights)
    
    def _ensemble_voting(
        self,
        votes: List[Dict[str, Any]],
        weights: List[float],
        **kwargs
    ) -> VotingResult:
        """
        Ensemble voting combining multiple strategies.
        
        Args:
            votes: List of votes
            weights: Vote weights
            
        Returns:
            VotingResult
        """
        # Get results from different strategies
        majority_result = self._majority_voting(votes, weights)
        weighted_result = self._weighted_voting(votes, weights)
        confidence_result = self._confidence_voting(votes, weights)
        
        # Combine results
        ensemble_votes = [
            {'answer': majority_result.final_answer, 'confidence': majority_result.confidence},
            {'answer': weighted_result.final_answer, 'confidence': weighted_result.confidence},
            {'answer': confidence_result.final_answer, 'confidence': confidence_result.confidence}
        ]
        
        # Meta-voting on ensemble results
        answer_scores = {}
        for vote in ensemble_votes:
            answer = str(vote['answer'])
            score = vote['confidence']
            answer_scores[answer] = answer_scores.get(answer, 0.0) + score
        
        # Get best answer
        final_answer = max(answer_scores, key=answer_scores.get)
        
        # Calculate ensemble confidence
        confidence = answer_scores[final_answer] / len(ensemble_votes)
        
        # Boost confidence if all strategies agree
        if len(set(v['answer'] for v in ensemble_votes)) == 1:
            confidence = min(confidence * 1.2, 1.0)
        
        return VotingResult(
            final_answer=final_answer,
            confidence=confidence,
            votes=votes,
            weights=weights,
            provenance={
                'strategy': 'ensemble',
                'sub_strategies': {
                    'majority': majority_result.final_answer,
                    'weighted': weighted_result.final_answer,
                    'confidence': confidence_result.final_answer
                },
                'answer_scores': answer_scores,
                'agreement': len(set(v['answer'] for v in ensemble_votes)) == 1
            }
        )
    
    def _calculate_agreement_factor(
        self,
        votes: List[Dict[str, Any]],
        final_answer: str
    ) -> float:
        """
        Calculate agreement factor for confidence adjustment.
        
        Args:
            votes: List of votes
            final_answer: The chosen answer
            
        Returns:
            Agreement factor (0 to 1)
        """
        if not votes:
            return 0.0
        
        # Count how many votes agree with final answer
        agreements = sum(1 for v in votes if str(v['answer']) == final_answer)
        
        # Calculate agreement ratio
        agreement_ratio = agreements / len(votes)
        
        # Apply non-linear scaling
        # High agreement (>0.7) gets boosted
        # Low agreement (<0.3) gets penalized
        if agreement_ratio > 0.7:
            factor = 0.8 + 0.2 * (agreement_ratio - 0.7) / 0.3
        elif agreement_ratio < 0.3:
            factor = 0.6 + 0.2 * agreement_ratio / 0.3
        else:
            factor = 0.6 + 0.2 * (agreement_ratio - 0.3) / 0.4
        
        return min(max(factor, 0.5), 1.0)
    
    def analyze_voting_consistency(
        self,
        voting_history: List[VotingResult]
    ) -> Dict[str, Any]:
        """
        Analyze consistency of voting over time.
        
        Args:
            voting_history: List of past voting results
            
        Returns:
            Analysis dictionary
        """
        if not voting_history:
            return {'error': 'No voting history available'}
        
        # Extract metrics
        confidences = [r.confidence for r in voting_history]
        consensus_strengths = [r.get_consensus_strength() for r in voting_history]
        
        # Calculate statistics
        analysis = {
            'num_votes': len(voting_history),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'avg_consensus_strength': np.mean(consensus_strengths),
            'confidence_trend': self._calculate_trend(confidences),
            'consensus_trend': self._calculate_trend(consensus_strengths)
        }
        
        # Identify problematic patterns
        low_confidence_rate = sum(1 for c in confidences if c < 0.5) / len(confidences)
        if low_confidence_rate > 0.3:
            analysis['warning'] = f"High rate of low confidence votes: {low_confidence_rate:.2%}"
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend in values.
        
        Args:
            values: List of values
            
        Returns:
            Trend description
        """
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"