#!/usr/bin/env python3
"""
Public Demonstrator Launch Script

Launches the Pixelis system in read-only mode for public demonstrations.
Enforces strict security and privacy policies to prevent data contamination
and protect user privacy.

Task 003 (Phase 2 Round 6): Enforce Read-Only Policy for Public Demonstrator
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import gradio as gr
import asyncio
from datetime import datetime
import hashlib
import time
from collections import deque

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.inference_engine import InferenceEngine
from core.modules.experience_buffer import ExperienceBuffer
from core.modules.voting import VotingModule
from core.modules.reward_shaping import RewardOrchestrator
from core.modules.privacy import PIIRedactor, PrivacyConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentFilter:
    """
    Filters inappropriate content from user inputs.
    """
    
    def __init__(self):
        """Initialize content filter with basic patterns."""
        # Note: In production, use a proper content moderation API
        self.blocked_patterns = []  # Add patterns as needed
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0
        }
    
    def is_appropriate(self, text: str) -> bool:
        """
        Check if text is appropriate for processing.
        
        Args:
            text: Input text to check
            
        Returns:
            True if appropriate, False otherwise
        """
        self.stats['total_requests'] += 1
        
        # Basic length check
        if len(text) > 5000:
            logger.warning("Input text too long")
            self.stats['blocked_requests'] += 1
            return False
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in text.lower():
                logger.warning(f"Blocked content detected: {pattern}")
                self.stats['blocked_requests'] += 1
                return False
        
        return True


class RateLimiter:
    """
    Rate limiting for API requests to prevent abuse.
    """
    
    def __init__(self, max_requests_per_minute: int = 10, max_requests_per_hour: int = 100):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute per user
            max_requests_per_hour: Maximum requests per hour per user
        """
        self.max_per_minute = max_requests_per_minute
        self.max_per_hour = max_requests_per_hour
        self.user_requests = {}  # user_id -> deque of timestamps
        self.global_requests = deque(maxlen=1000)
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if user is allowed to make a request.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        
        # Initialize user history if needed
        if user_id not in self.user_requests:
            self.user_requests[user_id] = deque()
        
        user_history = self.user_requests[user_id]
        
        # Remove old entries
        while user_history and user_history[0] < current_time - 3600:
            user_history.popleft()
        
        # Check per-minute limit
        recent_minute = [t for t in user_history if t > current_time - 60]
        if len(recent_minute) >= self.max_per_minute:
            logger.warning(f"Rate limit exceeded for user {user_id}: per-minute limit")
            return False
        
        # Check per-hour limit
        if len(user_history) >= self.max_per_hour:
            logger.warning(f"Rate limit exceeded for user {user_id}: per-hour limit")
            return False
        
        # Record request
        user_history.append(current_time)
        self.global_requests.append(current_time)
        
        return True


class PublicDemonstrator:
    """
    Public-facing demonstrator with strict security controls.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the public demonstrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.content_filter = ContentFilter()
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.config.get('rate_limit_per_minute', 10),
            max_requests_per_hour=self.config.get('rate_limit_per_hour', 100)
        )
        self.pii_redactor = PIIRedactor(PrivacyConfig(enable_pii_redaction=True))
        
        # Initialize the inference engine in READ-ONLY mode
        self.inference_engine = None
        self._initialize_inference_engine()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'blocked_requests': 0,
            'rate_limited_requests': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Public Demonstrator initialized in READ-ONLY mode")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for public demo."""
        return {
            # CRITICAL: Read-only mode for public demonstrator
            'read_only_mode': True,
            'disable_learning': True,
            'disable_updates': True,
            'disable_persistence': True,
            
            # Privacy settings
            'enable_pii_redaction': True,
            'enable_metadata_stripping': True,
            'log_privacy_stats': False,  # Don't log stats in public demo
            
            # Rate limiting
            'rate_limit_per_minute': 10,
            'rate_limit_per_hour': 100,
            
            # Resource limits
            'max_input_length': 1000,
            'max_image_size': 1024 * 1024 * 5,  # 5MB
            'inference_timeout': 30.0,
            
            # Security
            'enable_content_filtering': True,
            'enable_injection_detection': True,
            
            # Model settings
            'model_path': None,  # Use mock model for demo
            'use_mock_model': True,
            
            # Buffer settings (minimal for demo)
            'buffer_size': 100,
            'cold_start_threshold': 50,
            
            # Monitoring (disabled for public)
            'enable_wandb_logging': False,
            'enable_monitoring': False,
            
            # Demo settings
            'demo_title': "Pixelis Vision-Language Agent (Demo)",
            'demo_description': "Experience the Pixelis system in a safe, read-only demonstration mode.",
            'demo_examples': [
                ["What objects are in this image?", None],
                ["Describe the spatial relationships in the scene.", None],
                ["What text can you read in this image?", None]
            ]
        }
    
    def _initialize_inference_engine(self):
        """Initialize the inference engine with read-only configuration."""
        try:
            # Create mock model for demo
            from scripts.run_online_simulation import MockModel
            model = MockModel()
            
            # Initialize components
            experience_buffer = ExperienceBuffer(
                max_size=self.config.get('buffer_size', 100),
                embedding_dim=768
            )
            
            voting_module = VotingModule(config=self.config)
            reward_orchestrator = RewardOrchestrator(config=self.config)
            
            # Create inference engine with READ-ONLY configuration
            self.inference_engine = InferenceEngine(
                model=model,
                experience_buffer=experience_buffer,
                voting_module=voting_module,
                reward_orchestrator=reward_orchestrator,
                config=self.config
            )
            
            # Verify read-only mode is active
            if not self.inference_engine.read_only_mode:
                raise ValueError("CRITICAL: Inference engine not in read-only mode!")
            
            logger.info("Inference engine initialized in READ-ONLY mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def _generate_session_id(self, request: gr.Request) -> str:
        """
        Generate a session ID for rate limiting.
        
        Args:
            request: Gradio request object
            
        Returns:
            Hashed session identifier
        """
        # Use IP address for rate limiting (in production, use more sophisticated methods)
        client_ip = request.client.host if request and request.client else "unknown"
        session_id = hashlib.sha256(f"{client_ip}_{datetime.now().date()}".encode()).hexdigest()[:16]
        return session_id
    
    async def process_request(
        self,
        text_input: str,
        image_input: Optional[Any] = None,
        request: Optional[gr.Request] = None
    ) -> str:
        """
        Process a demonstration request.
        
        Args:
            text_input: User's text query
            image_input: Optional image input
            request: Gradio request object
            
        Returns:
            Model response or error message
        """
        self.stats['total_requests'] += 1
        
        try:
            # Generate session ID for rate limiting
            session_id = self._generate_session_id(request)
            
            # Check rate limiting
            if not self.rate_limiter.is_allowed(session_id):
                self.stats['rate_limited_requests'] += 1
                return "⚠️ Rate limit exceeded. Please wait a moment before trying again."
            
            # Validate input length
            if len(text_input) > self.config.get('max_input_length', 1000):
                self.stats['blocked_requests'] += 1
                return "❌ Input text is too long. Please keep it under 1000 characters."
            
            # Content filtering
            if self.config.get('enable_content_filtering', True):
                if not self.content_filter.is_appropriate(text_input):
                    self.stats['blocked_requests'] += 1
                    return "❌ Your input was blocked by the content filter. Please try a different query."
            
            # PII detection and warning
            pii_risk = self.pii_redactor.get_risk_assessment(text_input)
            if pii_risk['risk_level'] in ['high', 'medium']:
                # Redact PII before processing
                text_input, _ = self.pii_redactor.redact_text(text_input)
                logger.info(f"PII detected and redacted from input (risk: {pii_risk['risk_level']})")
            
            # Prepare input data
            input_data = {
                'question': text_input,
                'request_id': f"demo_{session_id}_{self.stats['total_requests']}"
            }
            
            # Handle image input if provided
            if image_input is not None:
                # In production, process the actual image
                # For demo, use mock image features
                input_data['image_features'] = torch.randn(1, 3, 224, 224)
            
            # Process through inference engine (READ-ONLY mode)
            result, confidence, metadata = await self.inference_engine.infer_and_adapt(input_data)
            
            # Verify read-only operation
            if not metadata.get('read_only', False):
                logger.error("CRITICAL: Read-only mode was bypassed!")
                return "❌ System error. Please contact support."
            
            # Format response
            response = self._format_response(result, confidence, metadata)
            
            self.stats['successful_requests'] += 1
            return response
            
        except asyncio.TimeoutError:
            self.stats['errors'] += 1
            return "⏱️ Request timed out. Please try again with a simpler query."
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.stats['errors'] += 1
            return "❌ An error occurred while processing your request. Please try again."
    
    def _format_response(self, result: Any, confidence: float, metadata: Dict[str, Any]) -> str:
        """
        Format the model response for display.
        
        Args:
            result: Model result
            confidence: Confidence score
            metadata: Response metadata
            
        Returns:
            Formatted response string
        """
        response_parts = []
        
        # Add main result
        if result:
            response_parts.append(f"**Response**: {result}")
        
        # Add confidence
        response_parts.append(f"**Confidence**: {confidence:.2%}")
        
        # Add demo notice
        response_parts.append("\n---")
        response_parts.append("*This is a demonstration running in read-only mode. No data is stored or used for training.*")
        
        return "\n\n".join(response_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get demonstration statistics."""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'uptime_hours': uptime / 3600,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'rate_limited_requests': self.stats['rate_limited_requests'],
            'errors': self.stats['errors'],
            'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1)
        }
    
    def create_gradio_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface for the demonstrator.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title=self.config['demo_title']) as interface:
            # Header
            gr.Markdown(f"# {self.config['demo_title']}")
            gr.Markdown(self.config['demo_description'])
            
            # Security notice
            gr.Markdown("""
            ### 🔒 Security & Privacy Notice
            - This demo runs in **read-only mode** - no learning or model updates occur
            - All inputs are automatically checked for personally identifiable information (PII)
            - No user data is stored persistently
            - Rate limiting is enforced to ensure fair access
            """)
            
            # Main interface
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a question about visual reasoning...",
                        lines=3,
                        max_lines=5
                    )
                    
                    image_input = gr.Image(
                        label="Upload Image (Optional)",
                        type="pil",
                        interactive=True
                    )
                    
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    output = gr.Markdown(label="Response")
                    
                    with gr.Accordion("Statistics", open=False):
                        stats_display = gr.JSON(label="Demo Statistics")
            
            # Examples
            if self.config.get('demo_examples'):
                gr.Examples(
                    examples=self.config['demo_examples'],
                    inputs=[text_input, image_input],
                    label="Example Queries"
                )
            
            # Event handlers
            submit_btn.click(
                fn=lambda *args: asyncio.run(self.process_request(*args)),
                inputs=[text_input, image_input],
                outputs=output
            )
            
            clear_btn.click(
                fn=lambda: ("", None, ""),
                inputs=[],
                outputs=[text_input, image_input, output]
            )
            
            # Auto-update statistics
            interface.load(
                fn=self.get_statistics,
                inputs=[],
                outputs=stats_display,
                every=30  # Update every 30 seconds
            )
            
            # Footer
            gr.Markdown("""
            ---
            © 2024 Pixelis Project | [Documentation](https://pixelis.ai/docs) | [GitHub](https://github.com/pixelis)
            """)
        
        return interface


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Launch Pixelis public demonstrator')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the demo on')
    parser.add_argument('--share', action='store_true',
                       help='Create a public shareable link')
    parser.add_argument('--auth', type=str, default=None,
                       help='Username:password for basic auth')
    
    args = parser.parse_args()
    
    # Security check: Ensure we're in read-only mode
    logger.info("=" * 60)
    logger.info("LAUNCHING PUBLIC DEMONSTRATOR IN READ-ONLY MODE")
    logger.info("No learning or model updates will occur")
    logger.info("All user data will be anonymized and not persisted")
    logger.info("=" * 60)
    
    # Initialize demonstrator
    demo = PublicDemonstrator(config_path=args.config)
    
    # Create Gradio interface
    interface = demo.create_gradio_interface()
    
    # Parse auth if provided
    auth = None
    if args.auth:
        parts = args.auth.split(':')
        if len(parts) == 2:
            auth = (parts[0], parts[1])
        else:
            logger.warning("Invalid auth format. Use username:password")
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        auth=auth,
        favicon_path=None,
        show_api=False,  # Disable API access for security
        max_threads=10,  # Limit concurrent processing
        quiet=False
    )


if __name__ == "__main__":
    main()