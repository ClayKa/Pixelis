#!/usr/bin/env python3
"""
Human Review Interface for TTRL

Provides a simple web interface for human experts to review and approve/reject
potential learning updates during the initial deployment phase.

This Human-in-the-Loop (HIL) safety valve ensures the system doesn't learn
from incorrect pseudo-labels before the experience buffer is mature.
"""

import gradio as gr
import torch
import torch.multiprocessing as mp
from typing import Dict, Any, Optional, List, Tuple
import json
import logging
from datetime import datetime
from dataclasses import dataclass
import queue
import threading
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReviewTask:
    """Task awaiting human review."""
    task_id: str
    experience_id: str
    question_text: str
    trajectory: List[Dict[str, Any]]
    model_confidence: float
    consensus_answer: Any
    voting_provenance: Dict[str, Any]
    timestamp: datetime
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'Task ID': self.task_id,
            'Question': self.question_text,
            'Model Confidence': f"{self.model_confidence:.3f}",
            'Consensus Answer': str(self.consensus_answer),
            'Voting Strategy': self.voting_provenance.get('voting_strategy', 'unknown'),
            'Retrieved Neighbors': self.voting_provenance.get('retrieved_neighbors_count', 0),
            'Timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_trajectory_text(self) -> str:
        """Get formatted trajectory text."""
        if not self.trajectory:
            return "No trajectory available"
        
        lines = []
        for i, action in enumerate(self.trajectory):
            lines.append(f"Step {i+1}: {action.get('operation', 'unknown')}")
            if 'arguments' in action:
                lines.append(f"  Args: {json.dumps(action['arguments'], indent=2)}")
            if 'result' in action:
                lines.append(f"  Result: {action['result']}")
        
        return "\n".join(lines)
    
    def get_provenance_text(self) -> str:
        """Get formatted provenance text."""
        return json.dumps(self.voting_provenance, indent=2)


class HumanReviewInterface:
    """
    Human review interface for TTRL online learning.
    
    Fetches tasks from the human_review_queue and presents them
    for expert approval/rejection.
    """
    
    def __init__(self, review_queue: mp.Queue, response_queue: mp.Queue):
        """
        Initialize the review interface.
        
        Args:
            review_queue: Queue to fetch tasks from
            response_queue: Queue to send decisions to
        """
        self.review_queue = review_queue
        self.response_queue = response_queue
        self.pending_tasks: List[ReviewTask] = []
        self.current_task: Optional[ReviewTask] = None
        
        # Statistics
        self.stats = {
            'total_reviewed': 0,
            'approved': 0,
            'rejected': 0,
            'average_confidence': 0.0
        }
        
        # Start background thread to fetch tasks
        self.running = True
        self.fetch_thread = threading.Thread(target=self._fetch_tasks_loop, daemon=True)
        self.fetch_thread.start()
        
        logger.info("Human Review Interface initialized")
    
    def _fetch_tasks_loop(self):
        """Background thread to fetch tasks from queue."""
        while self.running:
            try:
                # Try to get a task (non-blocking)
                raw_task = self.review_queue.get(timeout=1.0)
                
                if raw_task is None:  # Shutdown signal
                    break
                
                # Convert to ReviewTask
                review_task = self._convert_to_review_task(raw_task)
                self.pending_tasks.append(review_task)
                
                logger.info(f"Fetched task {review_task.task_id} for review")
                
            except queue.Empty:
                # No tasks available - expected
                continue
            except Exception as e:
                logger.error(f"Error fetching task: {e}")
    
    def _convert_to_review_task(self, raw_task: Any) -> ReviewTask:
        """Convert raw UpdateTask to ReviewTask for display."""
        # Extract relevant fields from UpdateTask
        return ReviewTask(
            task_id=raw_task.task_id,
            experience_id=raw_task.experience.experience_id,
            question_text=raw_task.experience.question_text,
            trajectory=raw_task.experience.trajectory.actions if hasattr(raw_task.experience.trajectory, 'actions') else [],
            model_confidence=raw_task.experience.model_confidence,
            consensus_answer=raw_task.experience.trajectory.final_answer if hasattr(raw_task.experience.trajectory, 'final_answer') else None,
            voting_provenance=raw_task.metadata.get('voting_provenance', {}),
            timestamp=raw_task.created_at
        )
    
    def get_next_task(self) -> Optional[ReviewTask]:
        """Get the next task for review."""
        if self.pending_tasks:
            self.current_task = self.pending_tasks.pop(0)
            return self.current_task
        return None
    
    def submit_decision(
        self,
        approved: bool,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Submit a review decision.
        
        Args:
            approved: Whether the task is approved
            notes: Optional reviewer notes
            
        Returns:
            Status dictionary
        """
        if self.current_task is None:
            return {'error': 'No task currently under review'}
        
        # Send decision to response queue
        decision = {
            'task_id': self.current_task.task_id,
            'approved': approved,
            'notes': notes,
            'reviewer_timestamp': datetime.now().isoformat()
        }
        
        try:
            self.response_queue.put(decision, timeout=1.0)
            
            # Update statistics
            self.stats['total_reviewed'] += 1
            if approved:
                self.stats['approved'] += 1
            else:
                self.stats['rejected'] += 1
            
            # Update average confidence
            n = self.stats['total_reviewed']
            old_avg = self.stats['average_confidence']
            new_conf = self.current_task.model_confidence
            self.stats['average_confidence'] = (old_avg * (n-1) + new_conf) / n
            
            logger.info(
                f"Review decision: task_id={self.current_task.task_id}, "
                f"approved={approved}, notes='{notes[:50]}...'"
            )
            
            # Clear current task
            self.current_task = None
            
            return {
                'status': 'success',
                'message': f"Decision submitted: {'APPROVED' if approved else 'REJECTED'}"
            }
            
        except Exception as e:
            logger.error(f"Failed to submit decision: {e}")
            return {'error': f'Failed to submit: {str(e)}'}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get review statistics."""
        stats = self.stats.copy()
        stats['pending_tasks'] = len(self.pending_tasks)
        stats['approval_rate'] = (
            self.stats['approved'] / self.stats['total_reviewed']
            if self.stats['total_reviewed'] > 0 else 0.0
        )
        return stats
    
    def shutdown(self):
        """Shutdown the interface."""
        self.running = False
        if self.fetch_thread:
            self.fetch_thread.join(timeout=5.0)
        logger.info("Human Review Interface shutdown")


def create_gradio_app(interface: HumanReviewInterface) -> gr.Blocks:
    """
    Create the Gradio web application.
    
    Args:
        interface: The human review interface
        
    Returns:
        Gradio Blocks app
    """
    
    with gr.Blocks(title="TTRL Human Review Interface") as app:
        gr.Markdown("# TTRL Human-in-the-Loop Review Interface")
        gr.Markdown(
            "Review and approve/reject potential learning updates during the initial "
            "deployment phase. This ensures the system learns from high-quality pseudo-labels."
        )
        
        with gr.Row():
            # Left column: Task details
            with gr.Column(scale=2):
                gr.Markdown("## Current Task")
                
                task_info = gr.JSON(label="Task Information", interactive=False)
                trajectory_text = gr.Textbox(
                    label="Reasoning Trajectory",
                    lines=10,
                    interactive=False
                )
                provenance_text = gr.Textbox(
                    label="Voting Provenance",
                    lines=8,
                    interactive=False
                )
                
                with gr.Row():
                    load_btn = gr.Button("Load Next Task", variant="primary")
                    pending_count = gr.Number(label="Pending Tasks", value=0, interactive=False)
            
            # Right column: Review controls
            with gr.Column(scale=1):
                gr.Markdown("## Review Decision")
                
                reviewer_notes = gr.Textbox(
                    label="Reviewer Notes",
                    placeholder="Optional: Add notes about your decision...",
                    lines=4
                )
                
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve", variant="primary")
                    reject_btn = gr.Button("❌ Reject", variant="stop")
                
                decision_status = gr.Textbox(
                    label="Decision Status",
                    interactive=False
                )
                
                gr.Markdown("## Statistics")
                stats_display = gr.JSON(label="Review Statistics", interactive=False)
                refresh_stats_btn = gr.Button("Refresh Stats")
        
        # Event handlers
        def load_next_task():
            """Load the next task for review."""
            task = interface.get_next_task()
            if task:
                return (
                    task.to_display_dict(),
                    task.get_trajectory_text(),
                    task.get_provenance_text(),
                    len(interface.pending_tasks),
                    ""  # Clear status
                )
            else:
                return (
                    {"message": "No tasks available"},
                    "No trajectory",
                    "No provenance",
                    0,
                    "No tasks in queue"
                )
        
        def approve_task(notes):
            """Approve the current task."""
            result = interface.submit_decision(approved=True, notes=notes)
            status = result.get('message', result.get('error', 'Unknown status'))
            
            # Auto-load next task
            next_data = load_next_task()
            
            return (
                next_data[0],  # task_info
                next_data[1],  # trajectory
                next_data[2],  # provenance
                next_data[3],  # pending_count
                status,        # decision_status
                "",           # Clear notes
                interface.get_stats()  # Update stats
            )
        
        def reject_task(notes):
            """Reject the current task."""
            result = interface.submit_decision(approved=False, notes=notes)
            status = result.get('message', result.get('error', 'Unknown status'))
            
            # Auto-load next task
            next_data = load_next_task()
            
            return (
                next_data[0],  # task_info
                next_data[1],  # trajectory
                next_data[2],  # provenance
                next_data[3],  # pending_count
                status,        # decision_status
                "",           # Clear notes
                interface.get_stats()  # Update stats
            )
        
        # Wire up events
        load_btn.click(
            fn=load_next_task,
            outputs=[task_info, trajectory_text, provenance_text, pending_count, decision_status]
        )
        
        approve_btn.click(
            fn=approve_task,
            inputs=[reviewer_notes],
            outputs=[
                task_info, trajectory_text, provenance_text, 
                pending_count, decision_status, reviewer_notes, stats_display
            ]
        )
        
        reject_btn.click(
            fn=reject_task,
            inputs=[reviewer_notes],
            outputs=[
                task_info, trajectory_text, provenance_text,
                pending_count, decision_status, reviewer_notes, stats_display
            ]
        )
        
        refresh_stats_btn.click(
            fn=lambda: interface.get_stats(),
            outputs=[stats_display]
        )
        
        # Load initial task on startup
        app.load(
            fn=lambda: (load_next_task()[0:5] + (interface.get_stats(),)),
            outputs=[
                task_info, trajectory_text, provenance_text,
                pending_count, decision_status, stats_display
            ]
        )
    
    return app


def main():
    """Main entry point for the human review application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTRL Human Review Interface")
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the interface on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the interface on'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio link'
    )
    
    args = parser.parse_args()
    
    # Create queues (in production, these would connect to the main system)
    review_queue = mp.Queue()
    response_queue = mp.Queue()
    
    # For demo purposes, add some sample tasks
    logger.info("Adding sample tasks for demonstration...")
    for i in range(3):
        from core.data_structures import UpdateTask, Experience, Trajectory, Action, ActionType
        import uuid
        
        # Create sample experience
        sample_trajectory = Trajectory(
            actions=[
                Action(
                    type=ActionType.VISUAL_OPERATION,
                    operation="SEGMENT_OBJECT_AT",
                    arguments={"x": 100, "y": 200},
                    result="Found object: cat"
                ),
                Action(
                    type=ActionType.REASONING,
                    operation="ANALYZE",
                    arguments={},
                    result="The object appears to be a domestic cat"
                ),
                Action(
                    type=ActionType.ANSWER,
                    operation="FINAL_ANSWER",
                    arguments={},
                    result="cat"
                )
            ],
            final_answer="cat",
            total_reward=0.8
        )
        
        sample_experience = Experience(
            experience_id=str(uuid.uuid4()),
            image_features=torch.randn(1, 512),  # Dummy features
            question_text=f"What animal is in the image? (Sample {i+1})",
            trajectory=sample_trajectory,
            model_confidence=0.75 + i * 0.05
        )
        
        sample_task = UpdateTask(
            task_id=str(uuid.uuid4()),
            experience=sample_experience,
            reward_tensor=torch.tensor(0.8),
            learning_rate=1e-4,
            metadata={
                'voting_provenance': {
                    'model_self_answer': 'cat',
                    'retrieved_neighbors_count': 5,
                    'neighbor_answers': [
                        {'answer': 'cat', 'confidence': 0.9},
                        {'answer': 'cat', 'confidence': 0.8},
                        {'answer': 'dog', 'confidence': 0.6},
                        {'answer': 'cat', 'confidence': 0.85},
                        {'answer': 'cat', 'confidence': 0.7}
                    ],
                    'voting_strategy': 'weighted'
                }
            }
        )
        
        review_queue.put(sample_task)
    
    # Create interface
    interface = HumanReviewInterface(review_queue, response_queue)
    
    # Create and launch Gradio app
    app = create_gradio_app(interface)
    
    logger.info(f"Launching Human Review Interface on {args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_api=False
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        interface.shutdown()
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        interface.shutdown()
        raise


if __name__ == "__main__":
    main()