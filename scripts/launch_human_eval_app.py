#!/usr/bin/env python3
"""
Human Evaluation Interface for Reasoning Quality Assessment

This application provides a blind A/B comparison interface for human annotators
to evaluate the quality of reasoning trajectories from different models.
Annotators assess trajectories on coherence, efficiency, and intelligence metrics.
"""

import gradio as gr
import json
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample with two model trajectories."""
    sample_id: str
    question_id: str
    image_path: str
    question_text: str
    model_a_name: str  # Hidden from annotator
    model_b_name: str  # Hidden from annotator
    trajectory_a: List[Dict[str, Any]]
    trajectory_b: List[Dict[str, Any]]
    final_answer_a: str
    final_answer_b: str
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_blinded_pair(self, shuffle: bool = True) -> Tuple[List[Dict], List[Dict], str, str]:
        """
        Get trajectories in randomized order for blind evaluation.
        
        Returns:
            Tuple of (left_trajectory, right_trajectory, left_label, right_label)
        """
        if shuffle and random.random() > 0.5:
            # Swap A and B randomly
            return self.trajectory_b, self.trajectory_a, "Model B", "Model A"
        else:
            return self.trajectory_a, self.trajectory_b, "Model A", "Model B"


@dataclass
class AnnotationResult:
    """Results from a single annotation."""
    annotation_id: str
    sample_id: str
    annotator_id: str
    timestamp: datetime
    
    # Which model was shown on left/right (for unblinding later)
    left_model: str
    right_model: str
    
    # Ratings for left model
    left_correctness: bool
    left_coherence: int  # 1-5 scale
    left_efficiency: int  # 1-5 scale
    left_thoroughness: int  # 1-5 scale
    
    # Ratings for right model
    right_correctness: bool
    right_coherence: int  # 1-5 scale
    right_efficiency: int  # 1-5 scale
    right_thoroughness: int  # 1-5 scale
    
    # Preference
    preference: str  # "left", "right", "equal"
    confidence: int  # 1-5 scale
    
    # Optional comments
    comments: Optional[str] = None
    
    # Time tracking
    time_spent_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class HumanEvaluationInterface:
    """
    Main interface for human evaluation of reasoning trajectories.
    Implements blind A/B testing with comprehensive quality metrics.
    """
    
    def __init__(
        self,
        samples_path: Path,
        output_dir: Path,
        annotator_id: Optional[str] = None
    ):
        """
        Initialize the evaluation interface.
        
        Args:
            samples_path: Path to JSON file with evaluation samples
            output_dir: Directory to save annotation results
            annotator_id: Optional ID for the current annotator
        """
        self.samples_path = Path(samples_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.annotator_id = annotator_id or self._generate_annotator_id()
        self.samples = self._load_samples()
        self.current_index = 0
        self.annotations = []
        self.start_time = None
        
        # Track annotation session
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        
        logger.info(f"Initialized evaluation interface for annotator {self.annotator_id}")
        logger.info(f"Loaded {len(self.samples)} samples for evaluation")
    
    def _generate_annotator_id(self) -> str:
        """Generate a unique annotator ID."""
        return f"annotator_{uuid.uuid4().hex[:8]}"
    
    def _load_samples(self) -> List[EvaluationSample]:
        """Load evaluation samples from JSON file."""
        if not self.samples_path.exists():
            logger.warning(f"Samples file not found: {self.samples_path}")
            return []
        
        with open(self.samples_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data['samples']:
            sample = EvaluationSample(
                sample_id=item['sample_id'],
                question_id=item['question_id'],
                image_path=item['image_path'],
                question_text=item['question_text'],
                model_a_name=item['model_a_name'],
                model_b_name=item['model_b_name'],
                trajectory_a=item['trajectory_a'],
                trajectory_b=item['trajectory_b'],
                final_answer_a=item['final_answer_a'],
                final_answer_b=item['final_answer_b'],
                ground_truth=item.get('ground_truth'),
                metadata=item.get('metadata', {})
            )
            samples.append(sample)
        
        # Shuffle samples for each annotator
        random.shuffle(samples)
        return samples
    
    def get_current_sample(self) -> Optional[EvaluationSample]:
        """Get the current sample for evaluation."""
        if 0 <= self.current_index < len(self.samples):
            return self.samples[self.current_index]
        return None
    
    def format_trajectory(self, trajectory: List[Dict[str, Any]]) -> str:
        """
        Format a trajectory for display.
        
        Args:
            trajectory: List of action dictionaries
            
        Returns:
            Formatted string representation
        """
        if not trajectory:
            return "No trajectory available"
        
        lines = []
        for i, action in enumerate(trajectory, 1):
            operation = action.get('operation', 'Unknown')
            action_type = action.get('type', '')
            
            # Format header
            lines.append(f"**Step {i}: {operation}**")
            
            # Add action type if available
            if action_type:
                lines.append(f"*Type: {action_type}*")
            
            # Format arguments
            if 'arguments' in action and action['arguments']:
                args_str = json.dumps(action['arguments'], indent=2)
                lines.append(f"Arguments:\n```json\n{args_str}\n```")
            
            # Format result
            if 'result' in action:
                result = action['result']
                if isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)
                    lines.append(f"Result:\n```json\n{result_str}\n```")
                else:
                    lines.append(f"Result: {result}")
            
            # Add reasoning if present
            if 'reasoning' in action:
                lines.append(f"Reasoning: {action['reasoning']}")
            
            lines.append("")  # Empty line between steps
        
        return "\n".join(lines)
    
    def save_annotation(self, annotation: AnnotationResult):
        """Save annotation result to file."""
        # Add to session annotations
        self.annotations.append(annotation)
        
        # Save to individual file
        filename = f"{self.annotator_id}_{annotation.annotation_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(annotation.to_dict(), f, indent=2)
        
        # Also append to session log
        session_file = self.output_dir / f"session_{self.session_id}.jsonl"
        with open(session_file, 'a') as f:
            f.write(json.dumps(annotation.to_dict()) + '\n')
        
        logger.info(f"Saved annotation {annotation.annotation_id}")
    
    def calculate_progress(self) -> Dict[str, Any]:
        """Calculate annotation progress statistics."""
        total = len(self.samples)
        completed = self.current_index
        remaining = total - completed
        
        # Calculate time statistics
        if self.annotations:
            avg_time = np.mean([a.time_spent_seconds for a in self.annotations 
                               if a.time_spent_seconds])
            estimated_remaining = avg_time * remaining / 60  # in minutes
        else:
            avg_time = 0
            estimated_remaining = 0
        
        return {
            'total_samples': total,
            'completed': completed,
            'remaining': remaining,
            'percentage': (completed / total * 100) if total > 0 else 0,
            'average_time_seconds': avg_time,
            'estimated_remaining_minutes': estimated_remaining,
            'session_duration_minutes': (datetime.now() - self.session_start).total_seconds() / 60
        }


def create_evaluation_app(interface: HumanEvaluationInterface) -> gr.Blocks:
    """
    Create the Gradio application for human evaluation.
    
    Args:
        interface: The evaluation interface instance
        
    Returns:
        Gradio Blocks application
    """
    
    with gr.Blocks(title="Pixelis Reasoning Quality Evaluation", theme=gr.themes.Soft()) as app:
        
        # Header
        gr.Markdown("""
        # 🔍 Pixelis Reasoning Quality Evaluation
        
        Welcome to the human evaluation interface for assessing reasoning trajectory quality.
        You will compare two model responses side-by-side and rate them on multiple dimensions.
        
        **Important**: This is a **blind evaluation** - you won't know which model produced which trajectory.
        Please evaluate based solely on the quality of reasoning you observe.
        """)
        
        # Session info
        with gr.Row():
            gr.Markdown(f"**Annotator ID**: {interface.annotator_id}")
            gr.Markdown(f"**Session ID**: {interface.session_id}")
        
        # Progress tracker
        with gr.Row():
            progress_info = gr.JSON(label="Progress", value=interface.calculate_progress())
        
        # Main evaluation interface
        with gr.Tabs():
            with gr.TabItem("📝 Evaluation"):
                
                # Question and image display
                with gr.Row():
                    with gr.Column(scale=2):
                        question_display = gr.Markdown("### Question: ")
                        image_display = gr.Image(label="Input Image", height=300)
                    
                    with gr.Column(scale=1):
                        ground_truth_display = gr.Textbox(
                            label="Ground Truth (if available)",
                            interactive=False
                        )
                        sample_info = gr.JSON(label="Sample Metadata")
                
                # Side-by-side trajectory comparison
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🅰️ Model A")
                        trajectory_a_display = gr.Markdown(
                            value="",
                            label="Trajectory A"
                        )
                        answer_a_display = gr.Textbox(
                            label="Final Answer",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 🅱️ Model B")
                        trajectory_b_display = gr.Markdown(
                            value="",
                            label="Trajectory B"
                        )
                        answer_b_display = gr.Textbox(
                            label="Final Answer",
                            interactive=False
                        )
                
                # Evaluation form
                gr.Markdown("---")
                gr.Markdown("## 📊 Evaluation Ratings")
                
                with gr.Row():
                    # Left model ratings
                    with gr.Column():
                        gr.Markdown("### Model A Ratings")
                        left_correct = gr.Checkbox(label="✅ Correct Answer")
                        left_coherence = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Coherence & Logicality",
                            info="1=Very Incoherent, 5=Very Coherent"
                        )
                        left_efficiency = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Efficiency & Intelligence",
                            info="1=Very Inefficient, 5=Very Efficient"
                        )
                        left_thoroughness = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Thoroughness",
                            info="1=Incomplete, 5=Very Thorough"
                        )
                    
                    # Right model ratings
                    with gr.Column():
                        gr.Markdown("### Model B Ratings")
                        right_correct = gr.Checkbox(label="✅ Correct Answer")
                        right_coherence = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Coherence & Logicality",
                            info="1=Very Incoherent, 5=Very Coherent"
                        )
                        right_efficiency = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Efficiency & Intelligence",
                            info="1=Very Inefficient, 5=Very Efficient"
                        )
                        right_thoroughness = gr.Slider(
                            1, 5, value=3, step=1,
                            label="Thoroughness",
                            info="1=Incomplete, 5=Very Thorough"
                        )
                
                # Overall preference
                gr.Markdown("### 🏆 Overall Preference")
                with gr.Row():
                    preference = gr.Radio(
                        choices=["Model A", "Model B", "Equal"],
                        label="Which model's reasoning do you prefer overall?",
                        value="Equal"
                    )
                    confidence = gr.Slider(
                        1, 5, value=3, step=1,
                        label="Confidence in Preference",
                        info="1=Not Confident, 5=Very Confident"
                    )
                
                # Comments
                comments = gr.Textbox(
                    label="Comments (Optional)",
                    placeholder="Any additional observations or notes about the trajectories...",
                    lines=3
                )
                
                # Action buttons
                with gr.Row():
                    skip_btn = gr.Button("⏭️ Skip", variant="secondary")
                    submit_btn = gr.Button("✅ Submit & Next", variant="primary")
                
                # Status message
                status_msg = gr.Markdown("")
            
            with gr.TabItem("📋 Guidelines"):
                gr.Markdown("""
                ## Evaluation Guidelines
                
                ### Rating Dimensions
                
                #### 1. Correctness (Binary)
                - Does the model arrive at the correct final answer?
                - This is a sanity check - reasoning quality is more important
                
                #### 2. Coherence & Logicality (1-5 Scale)
                - **5 (Very Coherent)**: Clear logical flow, each step follows naturally, no contradictions
                - **4 (Coherent)**: Generally logical with minor inconsistencies
                - **3 (Neutral)**: Some logical flow but with noticeable gaps
                - **2 (Incoherent)**: Poor logical flow, contradictions present
                - **1 (Very Incoherent)**: No clear logic, many contradictions
                
                #### 3. Efficiency & Intelligence (1-5 Scale)
                - **5 (Very Efficient)**: Direct approach, no wasted steps, shows clear understanding
                - **4 (Efficient)**: Good approach with minimal redundancy
                - **3 (Neutral)**: Reasonable approach but some inefficiencies
                - **2 (Inefficient)**: Many unnecessary steps, unclear strategy
                - **1 (Very Inefficient)**: Extremely redundant, no clear strategy
                
                #### 4. Thoroughness (1-5 Scale)
                - **5 (Very Thorough)**: Comprehensive exploration, considers alternatives
                - **4 (Thorough)**: Good coverage of relevant aspects
                - **3 (Neutral)**: Adequate exploration
                - **2 (Incomplete)**: Misses important aspects
                - **1 (Very Incomplete)**: Superficial or minimal exploration
                
                ### Important Notes
                - Focus on the reasoning process, not just the final answer
                - Consider each trajectory as a whole
                - Be consistent in your ratings across samples
                - Use the full scale (1-5) when rating
                """)
            
            with gr.TabItem("📈 Statistics"):
                stats_display = gr.DataFrame(
                    label="Session Statistics",
                    headers=["Metric", "Value"]
                )
                refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
        
        # Hidden state for tracking
        current_sample_state = gr.State(None)
        start_time_state = gr.State(None)
        left_model_state = gr.State(None)
        right_model_state = gr.State(None)
        
        # Event handlers
        def load_next_sample():
            """Load the next sample for evaluation."""
            sample = interface.get_current_sample()
            if sample is None:
                return {
                    question_display: "### ✅ All samples completed!",
                    image_display: None,
                    trajectory_a_display: "",
                    trajectory_b_display: "",
                    answer_a_display: "",
                    answer_b_display: "",
                    ground_truth_display: "",
                    sample_info: {},
                    current_sample_state: None,
                    start_time_state: None,
                    progress_info: interface.calculate_progress()
                }
            
            # Get blinded trajectories
            left_traj, right_traj, left_label, right_label = sample.get_blinded_pair()
            
            # Determine which model is on which side
            if left_label == "Model A":
                left_model = sample.model_a_name
                right_model = sample.model_b_name
                left_answer = sample.final_answer_a
                right_answer = sample.final_answer_b
            else:
                left_model = sample.model_b_name
                right_model = sample.model_a_name
                left_answer = sample.final_answer_b
                right_answer = sample.final_answer_a
            
            return {
                question_display: f"### Question: {sample.question_text}",
                image_display: sample.image_path if Path(sample.image_path).exists() else None,
                trajectory_a_display: interface.format_trajectory(left_traj),
                trajectory_b_display: interface.format_trajectory(right_traj),
                answer_a_display: left_answer,
                answer_b_display: right_answer,
                ground_truth_display: sample.ground_truth or "Not provided",
                sample_info: {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "has_ground_truth": sample.ground_truth is not None
                },
                current_sample_state: sample,
                start_time_state: datetime.now(),
                left_model_state: left_model,
                right_model_state: right_model,
                progress_info: interface.calculate_progress(),
                # Reset form
                left_correct: False,
                right_correct: False,
                left_coherence: 3,
                right_coherence: 3,
                left_efficiency: 3,
                right_efficiency: 3,
                left_thoroughness: 3,
                right_thoroughness: 3,
                preference: "Equal",
                confidence: 3,
                comments: ""
            }
        
        def submit_annotation(
            left_correct, left_coherence, left_efficiency, left_thoroughness,
            right_correct, right_coherence, right_efficiency, right_thoroughness,
            preference_value, confidence_value, comments_text,
            current_sample, start_time, left_model, right_model
        ):
            """Submit the current annotation and load the next sample."""
            
            if current_sample is None:
                return {
                    status_msg: "⚠️ No sample loaded",
                    progress_info: interface.calculate_progress()
                }
            
            # Calculate time spent
            time_spent = (datetime.now() - start_time).total_seconds() if start_time else 0
            
            # Map preference to left/right
            if preference_value == "Model A":
                pref = "left"
            elif preference_value == "Model B":
                pref = "right"
            else:
                pref = "equal"
            
            # Create annotation result
            annotation = AnnotationResult(
                annotation_id=str(uuid.uuid4()),
                sample_id=current_sample.sample_id,
                annotator_id=interface.annotator_id,
                timestamp=datetime.now(),
                left_model=left_model,
                right_model=right_model,
                left_correctness=left_correct,
                left_coherence=left_coherence,
                left_efficiency=left_efficiency,
                left_thoroughness=left_thoroughness,
                right_correctness=right_correct,
                right_coherence=right_coherence,
                right_efficiency=right_efficiency,
                right_thoroughness=right_thoroughness,
                preference=pref,
                confidence=confidence_value,
                comments=comments_text if comments_text else None,
                time_spent_seconds=time_spent
            )
            
            # Save annotation
            interface.save_annotation(annotation)
            
            # Move to next sample
            interface.current_index += 1
            
            # Load next sample
            next_results = load_next_sample()
            next_results[status_msg] = f"✅ Annotation saved! ({interface.current_index}/{len(interface.samples)} completed)"
            
            return next_results
        
        def skip_sample():
            """Skip the current sample."""
            interface.current_index += 1
            results = load_next_sample()
            results[status_msg] = f"⏭️ Sample skipped ({interface.current_index}/{len(interface.samples)})"
            return results
        
        def refresh_statistics():
            """Refresh session statistics."""
            stats = []
            
            if interface.annotations:
                # Calculate statistics
                left_coherence_scores = [a.left_coherence for a in interface.annotations]
                right_coherence_scores = [a.right_coherence for a in interface.annotations]
                left_efficiency_scores = [a.left_efficiency for a in interface.annotations]
                right_efficiency_scores = [a.right_efficiency for a in interface.annotations]
                
                stats.extend([
                    ["Total Annotations", len(interface.annotations)],
                    ["Avg. Time per Sample (sec)", f"{np.mean([a.time_spent_seconds for a in interface.annotations if a.time_spent_seconds]):.1f}"],
                    ["Avg. Coherence (Left)", f"{np.mean(left_coherence_scores):.2f}"],
                    ["Avg. Coherence (Right)", f"{np.mean(right_coherence_scores):.2f}"],
                    ["Avg. Efficiency (Left)", f"{np.mean(left_efficiency_scores):.2f}"],
                    ["Avg. Efficiency (Right)", f"{np.mean(right_efficiency_scores):.2f}"],
                    ["Left Preferred", sum(1 for a in interface.annotations if a.preference == "left")],
                    ["Right Preferred", sum(1 for a in interface.annotations if a.preference == "right")],
                    ["Equal Preference", sum(1 for a in interface.annotations if a.preference == "equal")],
                ])
            else:
                stats.append(["No annotations yet", ""])
            
            return pd.DataFrame(stats, columns=["Metric", "Value"])
        
        # Wire up event handlers
        submit_btn.click(
            fn=submit_annotation,
            inputs=[
                left_correct, left_coherence, left_efficiency, left_thoroughness,
                right_correct, right_coherence, right_efficiency, right_thoroughness,
                preference, confidence, comments,
                current_sample_state, start_time_state, left_model_state, right_model_state
            ],
            outputs=[
                question_display, image_display, trajectory_a_display, trajectory_b_display,
                answer_a_display, answer_b_display, ground_truth_display, sample_info,
                current_sample_state, start_time_state, left_model_state, right_model_state,
                progress_info, status_msg,
                left_correct, right_correct, left_coherence, right_coherence,
                left_efficiency, right_efficiency, left_thoroughness, right_thoroughness,
                preference, confidence, comments
            ]
        )
        
        skip_btn.click(
            fn=skip_sample,
            outputs=[
                question_display, image_display, trajectory_a_display, trajectory_b_display,
                answer_a_display, answer_b_display, ground_truth_display, sample_info,
                current_sample_state, start_time_state, progress_info, status_msg
            ]
        )
        
        refresh_stats_btn.click(
            fn=refresh_statistics,
            outputs=[stats_display]
        )
        
        # Load initial sample on startup
        app.load(
            fn=load_next_sample,
            outputs=[
                question_display, image_display, trajectory_a_display, trajectory_b_display,
                answer_a_display, answer_b_display, ground_truth_display, sample_info,
                current_sample_state, start_time_state, left_model_state, right_model_state,
                progress_info
            ]
        )
    
    return app


def main():
    """Main entry point for the human evaluation application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Human Evaluation Interface for Reasoning Quality")
    parser.add_argument(
        '--samples',
        type=str,
        required=True,
        help='Path to JSON file containing evaluation samples'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save annotation results'
    )
    parser.add_argument(
        '--annotator-id',
        type=str,
        help='Unique identifier for the annotator'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the interface on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7861,
        help='Port to run the interface on'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio link'
    )
    
    args = parser.parse_args()
    
    # Create interface
    interface = HumanEvaluationInterface(
        samples_path=Path(args.samples),
        output_dir=Path(args.output_dir),
        annotator_id=args.annotator_id
    )
    
    # Create and launch Gradio app
    app = create_evaluation_app(interface)
    
    logger.info(f"Launching Human Evaluation Interface on {args.host}:{args.port}")
    logger.info(f"Annotator ID: {interface.annotator_id}")
    logger.info(f"Output directory: {args.output_dir}")
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
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        raise


if __name__ == "__main__":
    main()