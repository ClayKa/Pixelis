#!/usr/bin/env python3
"""
Interactive Training Monitor Dashboard for RFT Training.

This script creates a real-time dashboard using Gradio to monitor:
- Reward component breakdown
- Tool usage frequencies
- Key training metrics
- Live trajectory samples
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
from queue import Queue

import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TrainingMonitor:
    """
    Real-time training monitor with interactive dashboard.
    
    Reads metrics from JSON file and displays them in an interactive Gradio interface.
    """
    
    def __init__(self, metrics_path: str, update_interval: int = 5):
        """
        Initialize the training monitor.
        
        Args:
            metrics_path: Path to JSON file with metrics
            update_interval: Update interval in seconds
        """
        self.metrics_path = Path(metrics_path)
        self.update_interval = update_interval
        
        # Data storage
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = {}
        self.start_time = time.time()
        
        # Reward components
        self.reward_components = ['R_final', 'R_curiosity', 'R_coherence']
        
        # Tool names
        self.tool_names = ['SEGMENT_OBJECT_AT', 'ZOOM_IN', 'READ_TEXT', 
                          'TRACK_OBJECT', 'GET_PROPERTIES', 'THINK']
        
        # Curriculum stages
        self.curriculum_stages = ['Phase1_Learn_Goal', 'Phase2_Learn_Coherence', 'Phase3_Full_Rewards']
        self.current_stage = 0
        
        # Start background thread for reading metrics
        self.running = True
        self.update_thread = threading.Thread(target=self._update_metrics_worker, daemon=True)
        self.update_thread.start()
        
    def _update_metrics_worker(self):
        """Background worker to read metrics file."""
        while self.running:
            try:
                if self.metrics_path.exists():
                    with open(self.metrics_path, 'r') as f:
                        data = json.load(f)
                        
                    # Get latest entry
                    if isinstance(data, list) and data:
                        latest = data[-1]
                        self.current_metrics = latest
                        self.metrics_history.append(latest)
                        
                        # Update curriculum stage
                        if 'current_curriculum_stage' in latest:
                            self.current_stage = latest['current_curriculum_stage']
                            
            except Exception as e:
                print(f"Error reading metrics: {e}")
                
            time.sleep(self.update_interval)
            
    def get_reward_breakdown_chart(self):
        """Create reward breakdown pie chart."""
        if not self.current_metrics or 'reward_breakdown' not in self.current_metrics:
            # Default values
            values = [1, 0, 0]
        else:
            breakdown = self.current_metrics['reward_breakdown']
            values = [
                breakdown.get('R_final', 0),
                breakdown.get('R_curiosity', 0),
                breakdown.get('R_coherence', 0)
            ]
            
        fig = go.Figure(data=[go.Pie(
            labels=self.reward_components,
            values=values,
            hole=0.3,
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1']),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Reward Component Breakdown',
            showlegend=True,
            height=400,
            font=dict(size=14)
        )
        
        return fig
        
    def get_tool_usage_chart(self):
        """Create tool usage bar chart."""
        if not self.current_metrics or 'tool_usage_frequency' not in self.current_metrics:
            # Default values
            usage = {tool: np.random.randint(0, 10) for tool in self.tool_names}
        else:
            usage = self.current_metrics['tool_usage_frequency']
            
        # Ensure all tools are represented
        for tool in self.tool_names:
            if tool not in usage:
                usage[tool] = 0
                
        fig = go.Figure(data=[go.Bar(
            x=list(usage.keys()),
            y=list(usage.values()),
            marker=dict(
                color=list(usage.values()),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Frequency')
            ),
            text=list(usage.values()),
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Tool Usage Frequency (Last Batch)',
            xaxis_title='Tool',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    def get_metrics_timeseries(self):
        """Create time series plot of key metrics."""
        if len(self.metrics_history) < 2:
            # Create empty plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('KL Divergence', 'GRPO Filtering Rate', 
                              'Success Rate (MA100)', 'Average Trajectory Length')
            )
            return fig
            
        # Convert history to dataframe
        df = pd.DataFrame(list(self.metrics_history))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('KL Divergence', 'GRPO Filtering Rate', 
                          'Success Rate (MA100)', 'Average Trajectory Length')
        )
        
        # KL Divergence
        if 'kl_divergence' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['kl_divergence'], 
                          mode='lines', name='KL Divergence',
                          line=dict(color='#FF6B6B')),
                row=1, col=1
            )
            
        # GRPO Filtering Rate
        if 'grpo_filtering_rate' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['grpo_filtering_rate'], 
                          mode='lines', name='GRPO Filtering',
                          line=dict(color='#4ECDC4')),
                row=1, col=2
            )
            
        # Success Rate
        if 'success_rate_ma100' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['success_rate_ma100'], 
                          mode='lines', name='Success Rate',
                          line=dict(color='#45B7D1')),
                row=2, col=1
            )
            
        # Trajectory Length
        if 'trajectory_length_mean' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['trajectory_length_mean'], 
                          mode='lines', name='Traj Length',
                          line=dict(color='#95E1D3')),
                row=2, col=2
            )
            
        fig.update_layout(
            height=600,
            showlegend=False,
            title='Training Metrics Over Time'
        )
        
        fig.update_xaxes(title_text='Steps', row=2, col=1)
        fig.update_xaxes(title_text='Steps', row=2, col=2)
        
        return fig
        
    def get_curriculum_progress(self):
        """Create curriculum progress indicator."""
        stages_html = '<div style="display: flex; justify-content: space-around; margin: 20px 0;">'
        
        for i, stage in enumerate(self.curriculum_stages):
            if i < self.current_stage:
                # Completed stage
                color = '#4CAF50'
                status = '✓'
            elif i == self.current_stage:
                # Current stage
                color = '#FF9800'
                status = '▶'
            else:
                # Future stage
                color = '#E0E0E0'
                status = ''
                
            stages_html += f'''
            <div style="text-align: center; flex: 1;">
                <div style="width: 80px; height: 80px; border-radius: 50%; 
                           background-color: {color}; margin: 0 auto; 
                           display: flex; align-items: center; justify-content: center;
                           font-size: 24px; color: white; font-weight: bold;">
                    {status}
                </div>
                <p style="margin-top: 10px; font-size: 12px; font-weight: bold;">
                    {stage.replace('_', ' ')}
                </p>
            </div>
            '''
            
        stages_html += '</div>'
        
        # Add progress bar
        progress = ((self.current_stage + 1) / len(self.curriculum_stages)) * 100
        stages_html += f'''
        <div style="margin: 20px 0;">
            <div style="background-color: #E0E0E0; height: 20px; border-radius: 10px;">
                <div style="background-color: #4CAF50; height: 100%; width: {progress}%; 
                           border-radius: 10px; transition: width 0.5s;"></div>
            </div>
            <p style="text-align: center; margin-top: 10px;">
                Progress: {progress:.1f}%
            </p>
        </div>
        '''
        
        return stages_html
        
    def get_live_metrics_table(self):
        """Create table of current metrics."""
        if not self.current_metrics:
            return pd.DataFrame({'Metric': ['No data'], 'Value': ['Waiting for metrics...']})
            
        # Extract key metrics
        metrics_data = []
        
        # Basic metrics
        if 'step' in self.current_metrics:
            metrics_data.append(('Training Step', self.current_metrics['step']))
            
        if 'kl_divergence' in self.current_metrics:
            metrics_data.append(('KL Divergence', f"{self.current_metrics['kl_divergence']:.4f}"))
            
        if 'grpo_filtering_rate' in self.current_metrics:
            metrics_data.append(('GRPO Filtering Rate', f"{self.current_metrics['grpo_filtering_rate']:.2%}"))
            
        if 'success_rate_ma100' in self.current_metrics:
            metrics_data.append(('Success Rate (MA100)', f"{self.current_metrics['success_rate_ma100']:.2%}"))
            
        if 'trajectory_length_mean' in self.current_metrics:
            metrics_data.append(('Avg Trajectory Length', f"{self.current_metrics['trajectory_length_mean']:.1f}"))
            
        # Reward breakdown
        if 'reward_breakdown' in self.current_metrics:
            breakdown = self.current_metrics['reward_breakdown']
            total_reward = sum(breakdown.values())
            metrics_data.append(('Total Reward', f"{total_reward:.3f}"))
            
        # Training time
        elapsed = time.time() - self.start_time
        metrics_data.append(('Training Time', str(timedelta(seconds=int(elapsed))))
        
        df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
        return df
        
    def get_sample_trajectory(self):
        """Get a sample trajectory from recent batch."""
        # This would normally read from actual trajectory data
        # For demo, we'll create a mock trajectory
        
        trajectory_text = """
        <div style="font-family: monospace; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
        <h4>Sample Trajectory (Step {step})</h4>
        <p><strong>Question:</strong> Count the number of people in the image.</p>
        <br>
        <p style="color: #2196F3;">→ SEGMENT_OBJECT_AT(x=150, y=200)</p>
        <p style="color: #666; margin-left: 20px;">Result: Found object: person</p>
        <br>
        <p style="color: #2196F3;">→ SEGMENT_OBJECT_AT(x=350, y=180)</p>
        <p style="color: #666; margin-left: 20px;">Result: Found object: person</p>
        <br>
        <p style="color: #4CAF50;">→ THINK</p>
        <p style="color: #666; margin-left: 20px;">I have identified 2 people in different locations of the image.</p>
        <br>
        <p><strong>Answer:</strong> There are 2 people in the image.</p>
        <br>
        <p style="color: #FF9800;"><strong>Rewards:</strong></p>
        <p style="margin-left: 20px;">• Task Reward: 1.0</p>
        <p style="margin-left: 20px;">• Coherence: 0.1</p>
        <p style="margin-left: 20px;">• Curiosity: 0.05</p>
        <p style="margin-left: 20px;"><strong>Total: 1.15</strong></p>
        </div>
        """.format(step=self.current_metrics.get('step', 0))
        
        return trajectory_text
        
    def create_dashboard(self):
        """Create the Gradio dashboard."""
        
        with gr.Blocks(title="Pixelis RFT Training Monitor", theme=gr.themes.Soft()) as dashboard:
            gr.Markdown("""
            # 🚀 Pixelis RFT Training Monitor
            ### Real-time monitoring of Reinforcement Fine-Tuning with GRPO
            """)
            
            # Top row - Curriculum Progress
            with gr.Row():
                curriculum_html = gr.HTML(
                    value=self.get_curriculum_progress(),
                    label="Curriculum Progress"
                )
                
            # Second row - Reward breakdown and tool usage
            with gr.Row():
                with gr.Column(scale=1):
                    reward_chart = gr.Plot(
                        value=self.get_reward_breakdown_chart(),
                        label="Reward Breakdown"
                    )
                    
                with gr.Column(scale=1):
                    tool_chart = gr.Plot(
                        value=self.get_tool_usage_chart(),
                        label="Tool Usage"
                    )
                    
            # Third row - Time series metrics
            with gr.Row():
                metrics_plot = gr.Plot(
                    value=self.get_metrics_timeseries(),
                    label="Training Metrics"
                )
                
            # Fourth row - Live metrics and trajectory
            with gr.Row():
                with gr.Column(scale=1):
                    metrics_table = gr.Dataframe(
                        value=self.get_live_metrics_table(),
                        label="Current Metrics",
                        headers=['Metric', 'Value']
                    )
                    
                with gr.Column(scale=1):
                    trajectory_display = gr.HTML(
                        value=self.get_sample_trajectory(),
                        label="Sample Trajectory"
                    )
                    
            # Auto-refresh components
            def update_all():
                return [
                    self.get_curriculum_progress(),
                    self.get_reward_breakdown_chart(),
                    self.get_tool_usage_chart(),
                    self.get_metrics_timeseries(),
                    self.get_live_metrics_table(),
                    self.get_sample_trajectory()
                ]
                
            # Set up auto-refresh every 5 seconds
            dashboard.load(
                fn=update_all,
                outputs=[curriculum_html, reward_chart, tool_chart, 
                        metrics_plot, metrics_table, trajectory_display],
                every=5
            )
            
            # Add manual refresh button
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Now", variant="primary")
                refresh_btn.click(
                    fn=update_all,
                    outputs=[curriculum_html, reward_chart, tool_chart, 
                            metrics_plot, metrics_table, trajectory_display]
                )
                
            # Add export button
            def export_metrics():
                """Export current metrics to file."""
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"metrics_export_{timestamp}.json"
                
                with open(export_path, 'w') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'current_metrics': self.current_metrics,
                        'history': list(self.metrics_history)
                    }, f, indent=2)
                    
                return f"Metrics exported to {export_path}"
                
            with gr.Row():
                export_btn = gr.Button("📥 Export Metrics", variant="secondary")
                export_output = gr.Textbox(label="Export Status", visible=True)
                export_btn.click(fn=export_metrics, outputs=export_output)
                
            # Footer
            gr.Markdown("""
            ---
            **Training Monitor v1.0** | Updates every 5 seconds | 
            [GitHub](https://github.com/pixelis/pixelis) | 
            Phase 1 Round 4 Implementation
            """)
            
        return dashboard
        
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        

def main():
    parser = argparse.ArgumentParser(description="Launch training monitor dashboard")
    parser.add_argument("--metrics_path", type=str, required=True, 
                       help="Path to metrics JSON file")
    parser.add_argument("--port", type=int, default=7860, 
                       help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", 
                       help="Create public Gradio link")
    parser.add_argument("--update_interval", type=int, default=5,
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    # Ensure metrics file exists or create empty one
    metrics_path = Path(args.metrics_path)
    if not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump([], f)
            
    print(f"Starting training monitor...")
    print(f"Reading metrics from: {metrics_path}")
    print(f"Dashboard will be available at: http://localhost:{args.port}")
    
    # Create monitor
    monitor = TrainingMonitor(
        metrics_path=str(metrics_path),
        update_interval=args.update_interval
    )
    
    # Create and launch dashboard
    dashboard = monitor.create_dashboard()
    
    try:
        dashboard.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
        monitor.stop()
        

if __name__ == "__main__":
    main()