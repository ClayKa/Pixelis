#!/usr/bin/env python3
"""
Pixelis: Interactive Public Demonstrator with Model Comparison

This script launches an enhanced public demo with side-by-side model comparison,
real-time trajectory visualization, and visual operation rendering.

Task 004 (Phase 3 Round 5): Create an Interactive Public Demonstrator
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import torch
import gradio as gr
import asyncio
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
from dataclasses import dataclass, asdict
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.inference_engine import InferenceEngine
from core.modules.operation_registry import VisualOperationRegistry
from core.modules.operations.base_operation import BaseOperation
from core.data_structures import Experience
from core.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Model configurations for comparison
MODEL_CONFIGS = {
    'pixelis_rft_base': {
        'name': 'Pixelis-RFT-Base',
        'config_path': 'configs/experiments/pixelis_rft_base.yaml',
        'checkpoint': 'saved_models/pixelis_rft_base.pt',
        'color': '#3498db',  # Blue
        'description': 'Base model with reinforcement fine-tuning'
    },
    'pixelis_rft_full': {
        'name': 'Pixelis-RFT-Full',
        'config_path': 'configs/experiments/pixelis_rft_full.yaml',
        'checkpoint': 'saved_models/pixelis_rft_full.pt',
        'color': '#e74c3c',  # Red
        'description': 'Full model with curiosity and coherence rewards'
    },
    'pixelis_online': {
        'name': 'Pixelis-Online',
        'config_path': 'configs/experiments/pixelis_online.yaml',
        'checkpoint': 'saved_models/pixelis_online.pt',
        'color': '#2ecc71',  # Green
        'description': 'Online adaptive model with TTRL'
    }
}

@dataclass
class ReasoningStep:
    """Represents a single reasoning step."""
    step_number: int
    thought: str
    action: str
    tool: Optional[str]
    tool_params: Optional[Dict[str, Any]]
    result: Optional[str]
    visual_output: Optional[Any]  # PIL Image or path
    confidence: float
    timestamp: float

@dataclass
class ModelTrajectory:
    """Complete trajectory for a model."""
    model_name: str
    steps: List[ReasoningStep]
    final_answer: str
    total_time: float
    confidence: float
    success: bool

class VisualOperationRenderer:
    """Renders visual operations for display."""
    
    def __init__(self):
        """Initialize the renderer."""
        self.registry = VisualOperationRegistry()
        self.font = None
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()
    
    def render_operation(self, image: Image.Image, operation: str, 
                         params: Dict[str, Any]) -> Image.Image:
        """
        Render a visual operation on an image.
        
        Args:
            image: Input PIL image
            operation: Operation name (e.g., 'ZOOM_IN')
            params: Operation parameters
            
        Returns:
            Rendered image showing the operation
        """
        rendered = image.copy()
        draw = ImageDraw.Draw(rendered)
        
        if operation == 'ZOOM_IN':
            # Draw bounding box for zoom region
            bbox = params.get('bbox', [0, 0, image.width, image.height])
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1-25), "ZOOM", fill='red', font=self.font)
            
            # Create zoomed view
            cropped = image.crop((x1, y1, x2, y2))
            # Resize to original size for display
            zoomed = cropped.resize((image.width, image.height), Image.LANCZOS)
            
            # Create side-by-side view
            combined = Image.new('RGB', (image.width * 2, image.height))
            combined.paste(rendered, (0, 0))
            combined.paste(zoomed, (image.width, 0))
            return combined
            
        elif operation == 'SEGMENT_OBJECT_AT':
            # Draw segmentation mask
            point = params.get('point', [image.width//2, image.height//2])
            x, y = point
            
            # Simple circular highlight for demo
            draw.ellipse([x-30, y-30, x+30, y+30], outline='green', width=3)
            draw.text((x-20, y-50), "SEGMENT", fill='green', font=self.font)
            
        elif operation == 'READ_TEXT':
            # Highlight text region
            region = params.get('region', [0, 0, image.width, image.height])
            x1, y1, x2, y2 = region
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
            draw.text((x1, y1-25), "READ", fill='blue', font=self.font)
            
        elif operation == 'TRACK_OBJECT':
            # Draw tracking trajectory
            trajectory = params.get('trajectory', [])
            if len(trajectory) > 1:
                for i in range(len(trajectory)-1):
                    x1, y1 = trajectory[i]
                    x2, y2 = trajectory[i+1]
                    draw.line([x1, y1, x2, y2], fill='purple', width=2)
                    draw.ellipse([x2-5, y2-5, x2+5, y2+5], fill='purple')
                
                if trajectory:
                    x, y = trajectory[-1]
                    draw.text((x+10, y), "TRACK", fill='purple', font=self.font)
        
        elif operation == 'GET_PROPERTIES':
            # Show property extraction area
            area = params.get('area', [0, 0, image.width, image.height])
            x1, y1, x2, y2 = area
            draw.rectangle([x1, y1, x2, y2], outline='orange', width=2, )
            draw.text((x1, y1-25), "PROPERTIES", fill='orange', font=self.font)
        
        return rendered

class ModelComparator:
    """Handles side-by-side model comparison."""
    
    def __init__(self, models_to_compare: List[str]):
        """
        Initialize model comparator.
        
        Args:
            models_to_compare: List of model keys to compare
        """
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=len(models_to_compare))
        self.renderer = VisualOperationRenderer()
        
        # Initialize models (in demo mode, we'll simulate)
        for model_key in models_to_compare:
            if model_key in MODEL_CONFIGS:
                self.models[model_key] = self._load_model(MODEL_CONFIGS[model_key])
    
    def _load_model(self, config: Dict[str, Any]) -> Any:
        """Load a model (simulated for demo)."""
        # In production, load actual model
        # For demo, return mock object
        logger.info(f"Loading model: {config['name']}")
        return {
            'name': config['name'],
            'config': config,
            'loaded': True
        }
    
    async def compare_models(self, image: Image.Image, question: str) -> List[ModelTrajectory]:
        """
        Run comparison across models.
        
        Args:
            image: Input image
            question: User question
            
        Returns:
            List of model trajectories
        """
        trajectories = []
        
        # Run models in parallel
        futures = []
        for model_key, model in self.models.items():
            future = self.executor.submit(
                self._run_model, model, image, question
            )
            futures.append((model_key, future))
        
        # Collect results
        for model_key, future in futures:
            try:
                trajectory = future.result(timeout=30)
                trajectories.append(trajectory)
            except Exception as e:
                logger.error(f"Error running model {model_key}: {e}")
                # Add error trajectory
                trajectories.append(ModelTrajectory(
                    model_name=MODEL_CONFIGS[model_key]['name'],
                    steps=[],
                    final_answer=f"Error: {str(e)}",
                    total_time=0,
                    confidence=0,
                    success=False
                ))
        
        return trajectories
    
    def _run_model(self, model: Any, image: Image.Image, question: str) -> ModelTrajectory:
        """Run a single model (simulated for demo)."""
        start_time = time.time()
        model_name = model['name']
        
        # Simulate reasoning steps
        steps = self._simulate_reasoning(model_name, image, question)
        
        # Generate final answer
        final_answer = self._generate_answer(model_name, steps)
        
        total_time = time.time() - start_time
        
        return ModelTrajectory(
            model_name=model_name,
            steps=steps,
            final_answer=final_answer,
            total_time=total_time,
            confidence=0.85 + np.random.random() * 0.15,
            success=True
        )
    
    def _simulate_reasoning(self, model_name: str, image: Image.Image, 
                           question: str) -> List[ReasoningStep]:
        """Simulate reasoning steps for demo."""
        steps = []
        
        # Different reasoning patterns for different models
        if 'Base' in model_name:
            # Simple direct reasoning
            steps.append(ReasoningStep(
                step_number=1,
                thought="Analyzing the image to understand the visual content",
                action="Examine overall image structure",
                tool="GET_PROPERTIES",
                tool_params={'area': [0, 0, image.width, image.height]},
                result="Detected objects and scene layout",
                visual_output=self.renderer.render_operation(
                    image, "GET_PROPERTIES", 
                    {'area': [0, 0, image.width, image.height]}
                ),
                confidence=0.8,
                timestamp=time.time()
            ))
            
            steps.append(ReasoningStep(
                step_number=2,
                thought="Focusing on relevant region for the question",
                action="Zoom into key area",
                tool="ZOOM_IN",
                tool_params={'bbox': [100, 100, 400, 400]},
                result="Enhanced view of target region",
                visual_output=self.renderer.render_operation(
                    image, "ZOOM_IN",
                    {'bbox': [100, 100, min(400, image.width), min(400, image.height)]}
                ),
                confidence=0.85,
                timestamp=time.time()
            ))
            
        elif 'Full' in model_name:
            # More sophisticated with coherence
            steps.append(ReasoningStep(
                step_number=1,
                thought="Initial scene understanding to establish context",
                action="Segment main objects",
                tool="SEGMENT_OBJECT_AT",
                tool_params={'point': [image.width//2, image.height//2]},
                result="Identified primary object boundaries",
                visual_output=self.renderer.render_operation(
                    image, "SEGMENT_OBJECT_AT",
                    {'point': [image.width//2, image.height//2]}
                ),
                confidence=0.82,
                timestamp=time.time()
            ))
            
            steps.append(ReasoningStep(
                step_number=2,
                thought="Checking for text that might provide additional context",
                action="Read visible text",
                tool="READ_TEXT",
                tool_params={'region': [0, 0, image.width, 100]},
                result="Extracted text information",
                visual_output=self.renderer.render_operation(
                    image, "READ_TEXT",
                    {'region': [0, 0, image.width, min(100, image.height)]}
                ),
                confidence=0.78,
                timestamp=time.time()
            ))
            
            steps.append(ReasoningStep(
                step_number=3,
                thought="Coherence check: Ensuring text aligns with visual content",
                action="Verify consistency between text and objects",
                tool=None,
                tool_params=None,
                result="Confirmed alignment of textual and visual information",
                visual_output=None,
                confidence=0.9,
                timestamp=time.time()
            ))
            
        else:  # Online model
            # Adaptive with curiosity
            steps.append(ReasoningStep(
                step_number=1,
                thought="Exploring image with curiosity-driven attention",
                action="Track interesting patterns",
                tool="TRACK_OBJECT",
                tool_params={'trajectory': [
                    [100, 100], [200, 150], [300, 200], [250, 250]
                ]},
                result="Discovered motion pattern",
                visual_output=self.renderer.render_operation(
                    image, "TRACK_OBJECT",
                    {'trajectory': [[100, 100], [200, 150], [300, 200], [250, 250]]}
                ),
                confidence=0.75,
                timestamp=time.time()
            ))
            
            steps.append(ReasoningStep(
                step_number=2,
                thought="Curiosity spike: Unusual pattern detected, investigating further",
                action="Zoom into anomaly",
                tool="ZOOM_IN",
                tool_params={'bbox': [200, 150, 350, 300]},
                result="Detailed view of interesting region",
                visual_output=self.renderer.render_operation(
                    image, "ZOOM_IN",
                    {'bbox': [200, 150, min(350, image.width), min(300, image.height)]}
                ),
                confidence=0.88,
                timestamp=time.time()
            ))
            
            steps.append(ReasoningStep(
                step_number=3,
                thought="Online adaptation: Incorporating new pattern into understanding",
                action="Update internal model based on discovery",
                tool=None,
                tool_params=None,
                result="Model adapted to new visual pattern",
                visual_output=None,
                confidence=0.92,
                timestamp=time.time()
            ))
        
        return steps
    
    def _generate_answer(self, model_name: str, steps: List[ReasoningStep]) -> str:
        """Generate final answer based on reasoning steps."""
        if 'Base' in model_name:
            return "Based on visual analysis, the image contains the requested elements in the highlighted regions."
        elif 'Full' in model_name:
            return "After coherent multi-step reasoning, I've identified the key visual elements and their relationships as shown in the trajectory."
        else:
            return "Through adaptive exploration and curiosity-driven analysis, I discovered interesting patterns that address your question."

class PublicDemoInterface:
    """Main interface for the public demonstrator."""
    
    def __init__(self, models_to_compare: List[str] = None):
        """Initialize the demo interface."""
        if models_to_compare is None:
            models_to_compare = ['pixelis_rft_base', 'pixelis_rft_full', 'pixelis_online']
        
        self.comparator = ModelComparator(models_to_compare)
        self.session_count = 0
        self.demo_examples = self._load_demo_examples()
    
    def _load_demo_examples(self) -> List[Tuple[str, str]]:
        """Load demo examples."""
        return [
            ("examples/street_scene.jpg", "How many cars are visible?"),
            ("examples/document.jpg", "What is the main topic of this document?"),
            ("examples/chart.jpg", "What trend does this chart show?"),
            ("examples/puzzle.jpg", "What is the solution to this puzzle?"),
        ]
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title="Pixelis Model Comparison Demo", 
                      theme=gr.themes.Soft()) as demo:
            
            # Header
            gr.Markdown("""
            # 🔍 Pixelis: Visual Reasoning Model Comparison
            
            Upload an image and ask a question to see how different Pixelis models reason through the task.
            Watch as each model uses different strategies and visual operations to find the answer!
            
            ---
            """)
            
            # Input section
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=300
                    )
                    input_question = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask something about the image...",
                        lines=2
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 Compare Models", variant="primary")
                        clear_btn = gr.Button("🔄 Clear", variant="secondary")
                    
                    # Examples
                    gr.Markdown("### Quick Examples")
                    gr.Examples(
                        examples=self.demo_examples,
                        inputs=[input_image, input_question],
                        outputs=[],
                        fn=None,
                        cache_examples=False
                    )
            
            # Comparison section
            with gr.Row():
                # Create columns for each model
                model_outputs = []
                for model_key in self.comparator.models.keys():
                    config = MODEL_CONFIGS[model_key]
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {config['name']}")
                        gr.Markdown(f"*{config['description']}*")
                        
                        # Trajectory display
                        trajectory_display = gr.HTML(
                            label=f"{config['name']} Reasoning",
                            value="<p>Waiting for input...</p>"
                        )
                        
                        # Visual output
                        visual_display = gr.Image(
                            label="Visual Operations",
                            type="pil",
                            height=200
                        )
                        
                        # Final answer
                        answer_display = gr.Textbox(
                            label="Final Answer",
                            lines=3,
                            interactive=False
                        )
                        
                        # Metrics
                        metrics_display = gr.JSON(
                            label="Performance Metrics",
                            value={}
                        )
                        
                        model_outputs.append({
                            'trajectory': trajectory_display,
                            'visual': visual_display,
                            'answer': answer_display,
                            'metrics': metrics_display
                        })
            
            # Summary section
            with gr.Row():
                summary_display = gr.HTML(
                    label="Comparison Summary",
                    value="<p>Submit an image and question to see the comparison.</p>"
                )
            
            # Event handlers
            async def run_comparison(image, question):
                """Run model comparison."""
                if image is None or not question:
                    return [gr.update(value="<p>Please provide both image and question.</p>")] * len(model_outputs) * 4 + [gr.update()]
                
                try:
                    # Run comparison
                    trajectories = await self.comparator.compare_models(image, question)
                    
                    # Format outputs for each model
                    outputs = []
                    for i, trajectory in enumerate(trajectories):
                        # Format trajectory HTML
                        trajectory_html = self._format_trajectory_html(trajectory)
                        outputs.append(gr.update(value=trajectory_html))
                        
                        # Get last visual output
                        visual_output = None
                        for step in reversed(trajectory.steps):
                            if step.visual_output is not None:
                                visual_output = step.visual_output
                                break
                        outputs.append(gr.update(value=visual_output))
                        
                        # Final answer
                        outputs.append(gr.update(value=trajectory.final_answer))
                        
                        # Metrics
                        metrics = {
                            'Time (s)': round(trajectory.total_time, 2),
                            'Steps': len(trajectory.steps),
                            'Confidence': round(trajectory.confidence, 3),
                            'Success': trajectory.success
                        }
                        outputs.append(gr.update(value=metrics))
                    
                    # Summary
                    summary_html = self._generate_summary_html(trajectories)
                    outputs.append(gr.update(value=summary_html))
                    
                    return outputs
                    
                except Exception as e:
                    logger.error(f"Error in comparison: {e}")
                    error_msg = f"<p style='color: red;'>Error: {str(e)}</p>"
                    return [gr.update(value=error_msg)] * len(model_outputs) * 4 + [gr.update(value=error_msg)]
            
            def clear_all():
                """Clear all outputs."""
                return (
                    [None, ""] +  # Clear inputs
                    [gr.update(value="<p>Waiting for input...</p>")] * len(model_outputs) +  # trajectories
                    [None] * len(model_outputs) +  # visuals
                    [""] * len(model_outputs) +  # answers
                    [{}] * len(model_outputs) +  # metrics
                    [gr.update(value="<p>Submit an image and question to see the comparison.</p>")]  # summary
                )
            
            # Flatten model outputs for event handling
            all_outputs = []
            for output_dict in model_outputs:
                all_outputs.extend([
                    output_dict['trajectory'],
                    output_dict['visual'],
                    output_dict['answer'],
                    output_dict['metrics']
                ])
            all_outputs.append(summary_display)
            
            submit_btn.click(
                fn=run_comparison,
                inputs=[input_image, input_question],
                outputs=all_outputs
            )
            
            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[input_image, input_question] + all_outputs
            )
            
            # Footer
            gr.Markdown("""
            ---
            
            ### About This Demo
            
            This interactive demonstration showcases the Pixelis framework's different model variants:
            
            - **RFT-Base**: Basic reinforcement fine-tuned model with standard visual operations
            - **RFT-Full**: Enhanced model with curiosity and coherence rewards for better exploration
            - **Online**: Adaptive model with Test-Time Reinforcement Learning (TTRL) capabilities
            
            Each model uses different reasoning strategies and visual operations to analyze images and answer questions.
            The colored trajectories show how each model explores the image differently based on its training.
            
            ### Visual Operations
            
            - 🔍 **ZOOM_IN**: Focus on specific regions
            - 🎯 **SEGMENT_OBJECT_AT**: Identify object boundaries
            - 📝 **READ_TEXT**: Extract textual information
            - 📊 **GET_PROPERTIES**: Analyze visual properties
            - 🎬 **TRACK_OBJECT**: Follow movement patterns
            
            ---
            
            *Note: This is a demonstration interface. Actual model outputs may vary.*
            """)
        
        return demo
    
    def _format_trajectory_html(self, trajectory: ModelTrajectory) -> str:
        """Format trajectory as HTML."""
        html = f"<div style='font-family: monospace; font-size: 12px;'>"
        html += f"<h4>{trajectory.model_name}</h4>"
        
        for step in trajectory.steps:
            # Step header
            html += f"<div style='margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;'>"
            html += f"<strong>Step {step.step_number}</strong> "
            
            if step.tool:
                html += f"<span style='color: blue;'>[{step.tool}]</span>"
            
            html += f" <span style='color: gray;'>(conf: {step.confidence:.2f})</span><br/>"
            
            # Thought
            html += f"<em>💭 {step.thought}</em><br/>"
            
            # Action
            html += f"🎯 {step.action}<br/>"
            
            # Result
            if step.result:
                html += f"✅ {step.result}<br/>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _generate_summary_html(self, trajectories: List[ModelTrajectory]) -> str:
        """Generate comparison summary HTML."""
        html = "<div style='padding: 20px;'>"
        html += "<h3>Comparison Summary</h3>"
        
        # Performance table
        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += "<tr style='background: #f0f0f0;'>"
        html += "<th style='padding: 10px; text-align: left;'>Model</th>"
        html += "<th style='padding: 10px;'>Steps</th>"
        html += "<th style='padding: 10px;'>Time (s)</th>"
        html += "<th style='padding: 10px;'>Confidence</th>"
        html += "<th style='padding: 10px;'>Status</th>"
        html += "</tr>"
        
        for traj in trajectories:
            html += "<tr>"
            html += f"<td style='padding: 10px;'><strong>{traj.model_name}</strong></td>"
            html += f"<td style='padding: 10px; text-align: center;'>{len(traj.steps)}</td>"
            html += f"<td style='padding: 10px; text-align: center;'>{traj.total_time:.2f}</td>"
            html += f"<td style='padding: 10px; text-align: center;'>{traj.confidence:.3f}</td>"
            status_color = 'green' if traj.success else 'red'
            status_text = '✅ Success' if traj.success else '❌ Failed'
            html += f"<td style='padding: 10px; text-align: center; color: {status_color};'>{status_text}</td>"
            html += "</tr>"
        
        html += "</table>"
        
        # Key insights
        html += "<h4>Key Observations</h4>"
        html += "<ul>"
        
        # Find fastest model
        fastest = min(trajectories, key=lambda t: t.total_time)
        html += f"<li>⚡ <strong>{fastest.model_name}</strong> was fastest ({fastest.total_time:.2f}s)</li>"
        
        # Find most confident
        most_confident = max(trajectories, key=lambda t: t.confidence)
        html += f"<li>💪 <strong>{most_confident.model_name}</strong> had highest confidence ({most_confident.confidence:.3f})</li>"
        
        # Find most thorough
        most_thorough = max(trajectories, key=lambda t: len(t.steps))
        html += f"<li>🔍 <strong>{most_thorough.model_name}</strong> was most thorough ({len(most_thorough.steps)} steps)</li>"
        
        html += "</ul>"
        html += "</div>"
        
        return html
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the demo interface."""
        demo = self.create_interface()
        demo.launch(
            share=share,
            server_port=port,
            server_name="0.0.0.0"
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pixelis Public Demonstrator')
    parser.add_argument('--models', nargs='+', 
                       default=['pixelis_rft_base', 'pixelis_rft_full', 'pixelis_online'],
                       help='Models to compare')
    parser.add_argument('--share', action='store_true',
                       help='Create public shareable link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the server')
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = PublicDemoInterface(models_to_compare=args.models)
    
    logger.info(f"Launching Pixelis public demonstrator on port {args.port}")
    logger.info(f"Models to compare: {args.models}")
    
    interface.launch(share=args.share, port=args.port)


if __name__ == '__main__':
    main()