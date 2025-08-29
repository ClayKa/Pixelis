"""
Base Task Generator for CoTA (Chain-of-Thought-Action) data synthesis.

This module provides the abstract base class for all specialized task generators,
handling common functionality including API client initialization, prompt loading,
generation orchestration, checkpointing, and error handling.
"""

import json
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import openai
import os
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaseTaskGenerator(ABC):
    """
    Abstract base class for all specialized CoTA task generators.
    
    This class provides:
    - Prompt template loading and formatting
    - API client initialization and management
    - Generation loop with progress tracking
    - Checkpointing and recovery
    - Error handling and retry logic
    - Provenance tracking
    - Statistics collection
    """
    
    def __init__(
        self, 
        loaders: Dict[str, Any], 
        config: Dict[str, Any], 
        global_config: Dict[str, Any]
    ):
        """
        Initialize the base task generator.
        
        Args:
            loaders: Dictionary mapping loader names to initialized data loader objects
            config: Task-specific configuration from manifest
            global_config: Global configuration including API profiles
        """
        self.loaders = loaders
        self.config = config
        self.global_config = global_config
        self.task_name = self.config.get('name', 'unknown_task')
        self.generator_params = self.config.get('generator_params', {})
        
        # Initialize core components
        self.prompt_template = self._load_prompt_template()
        self.api_client = self._initialize_api_client()
        
        # Statistics tracking
        self.generation_stats = defaultdict(int)
        self.start_time = None
        
    def _load_prompt_template(self) -> str:
        """
        Load the prompt template from the configured file.
        
        Returns:
            The loaded prompt template string
        
        Raises:
            FileNotFoundError: If the prompt template file doesn't exist
        """
        prompt_path = Path(self.config.get('prompt_template', ''))
        if not prompt_path.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            prompt_path = project_root / prompt_path
            
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found for task '{self.task_name}': {prompt_path}"
            )
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        logger.info(f"Loaded prompt template from {prompt_path}")
        return template
    
    def _initialize_api_client(self) -> openai.OpenAI:
        """
        Initialize the OpenAI-compatible API client.
        
        Returns:
            Configured OpenAI client instance
        """
        # Get API configuration from global config
        api_profile = self.global_config.get('api_profiles', {}).get('generator_api', {})
        
        # Support both environment variables and config file
        api_key = api_profile.get('api_key') or os.getenv('OPENROUTER_API_KEY')
        base_url = api_profile.get('base_url', 'https://openrouter.ai/api/v1')
        
        if not api_key:
            logger.warning(
                "No API key found. Set OPENROUTER_API_KEY env var or configure in manifest. "
                "Using mock mode for development."
            )
            return None  # Will trigger mock mode in _call_llm_api
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        logger.info(f"Initialized API client for '{self.task_name}' with base URL: {base_url}")
        return client
    
    @abstractmethod
    def _build_context_placeholders(self) -> Dict[str, str]:
        """
        Build the context placeholders for the prompt template.
        
        This method must be implemented by each specialized generator to sample
        from data loaders and construct the task-specific context.
        
        Returns:
            Dictionary mapping placeholder names to their values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_context_placeholders()"
        )
    
    def _call_llm_api(self, prompt: str, attempt: int = 1) -> Dict[str, Any]:
        """
        Call the LLM API with retry logic and error handling.
        
        Args:
            prompt: The formatted prompt to send
            attempt: Current attempt number for retry tracking
            
        Returns:
            Parsed JSON response from the LLM
            
        Raises:
            Exception: After max retries are exhausted
        """
        max_retries = self.generator_params.get('max_retries', 3)
        retry_delay = self.generator_params.get('retry_delay', 2.0)
        
        # Mock mode for development/testing
        if self.api_client is None:
            return self._generate_mock_response()
        
        try:
            # Prepare API call parameters
            model = self.generator_params.get('model', 'meta-llama/llama-3.2-90b-vision-instruct')
            temperature = self.generator_params.get('temperature', 0.7)
            max_tokens = self.generator_params.get('max_tokens', 4096)
            
            # Make the API call
            response = self.api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates structured CoTA trajectories in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Extract and parse response
            response_content = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                parsed_response = json.loads(response_content)
                self.generation_stats['api_calls_successful'] += 1
                return parsed_response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response_content[:500]}...")
                raise
                
        except Exception as e:
            self.generation_stats['api_calls_failed'] += 1
            
            if attempt < max_retries:
                logger.warning(
                    f"API call failed (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay * attempt)  # Exponential backoff
                return self._call_llm_api(prompt, attempt + 1)
            else:
                logger.error(f"API call failed after {max_retries} attempts: {e}")
                raise
    
    def _generate_mock_response(self) -> Dict[str, Any]:
        """
        Generate a mock response for development/testing.
        
        Returns:
            Mock CoTA trajectory in the expected format
        """
        return {
            "task_id": f"mock_{self.task_name}_{random.randint(1000, 9999)}",
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "trajectory": [
                {
                    "step": 1,
                    "thought": "Mock thought process",
                    "action": "ZOOM_IN",
                    "parameters": {"coordinates": [100, 100, 200, 200]},
                    "result": "Mock result"
                }
            ],
            "final_answer": "Mock answer",
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "is_mock": True
            }
        }
    
    def _add_provenance(self, sample: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add provenance metadata to a generated sample.
        
        Args:
            sample: The generated CoTA sample
            context_data: Context information used for generation
            
        Returns:
            Sample with added provenance metadata
        """
        # Generate unique ID for this sample
        sample_id = hashlib.sha256(
            f"{self.task_name}_{datetime.now().isoformat()}_{random.random()}".encode()
        ).hexdigest()[:16]
        
        provenance = {
            "sample_id": sample_id,
            "task_name": self.task_name,
            "generator_class": self.__class__.__name__,
            "generated_at": datetime.now().isoformat(),
            "model": self.generator_params.get('model', 'unknown'),
            "temperature": self.generator_params.get('temperature', 0.7),
            "data_sources": list(context_data.get('source_datasets', [])),
            "generator_version": "1.0.0"
        }
        
        sample['provenance'] = provenance
        return sample
    
    def generate(self, num_samples: int, checkpoint_path: Path) -> List[Dict]:
        """
        Main generation method orchestrating the entire process.
        
        Args:
            num_samples: Number of samples to generate
            checkpoint_path: Path to save/load checkpoints
            
        Returns:
            List of generated CoTA samples
        """
        generated_samples = []
        self.start_time = time.time()
        
        # Ensure checkpoint directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load from checkpoint if exists
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    for line in f:
                        generated_samples.append(json.loads(line.strip()))
                logger.info(
                    f"Resumed from checkpoint: {len(generated_samples)}/{num_samples} "
                    f"samples for '{self.task_name}'"
                )
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
                generated_samples = []
        
        # Setup progress bar
        start_index = len(generated_samples)
        pbar = tqdm(
            range(start_index, num_samples),
            desc=f"Generating {self.task_name}",
            initial=start_index,
            total=num_samples
        )
        
        # Main generation loop
        checkpoint_interval = self.global_config.get('checkpoint_every_n_samples', 100)
        
        for i in pbar:
            try:
                # Build context for this sample
                context_placeholders = self._build_context_placeholders()
                
                # Format the prompt
                final_prompt = self.prompt_template.format(**context_placeholders)
                
                # Call LLM API
                cota_sample = self._call_llm_api(final_prompt)
                
                # Add provenance metadata
                cota_sample = self._add_provenance(cota_sample, context_placeholders)
                
                # Validate the generated sample
                if self._validate_sample(cota_sample):
                    generated_samples.append(cota_sample)
                    self.generation_stats['samples_generated'] += 1
                else:
                    logger.warning(f"Sample {i+1} failed validation, retrying...")
                    self.generation_stats['samples_invalid'] += 1
                    continue
                
                # Update progress bar with stats
                pbar.set_postfix({
                    'success_rate': f"{self.generation_stats['samples_generated']}/{i+1}",
                    'api_failures': self.generation_stats['api_calls_failed']
                })
                
                # Checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(generated_samples, checkpoint_path)
                    self._log_statistics()
                    
            except KeyboardInterrupt:
                logger.info("Generation interrupted by user. Saving checkpoint...")
                self._save_checkpoint(generated_samples, checkpoint_path)
                raise
                
            except Exception as e:
                logger.error(f"Failed to generate sample {i+1}: {e}")
                self.generation_stats['samples_failed'] += 1
                
                # Add a failed sample placeholder to maintain count
                failed_sample = {
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                    "task_name": self.task_name,
                    "sample_index": i
                }
                generated_samples.append(failed_sample)
        
        # Final save and statistics
        self._save_checkpoint(generated_samples, checkpoint_path)
        self._log_statistics()
        
        logger.info(
            f"Completed generation for '{self.task_name}': "
            f"{len(generated_samples)} samples generated"
        )
        
        return generated_samples
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a generated sample for structural integrity.
        
        Args:
            sample: The sample to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for required fields
        required_fields = ['trajectory', 'final_answer']
        for field in required_fields:
            if field not in sample:
                logger.debug(f"Sample missing required field: {field}")
                return False
        
        # Validate trajectory structure
        if not isinstance(sample.get('trajectory', None), list):
            logger.debug("Trajectory is not a list")
            return False
        
        if len(sample['trajectory']) == 0:
            logger.debug("Empty trajectory")
            return False
        
        # Validate each step in trajectory
        for step in sample['trajectory']:
            if not isinstance(step, dict):
                logger.debug(f"Invalid step format: {step}")
                return False
            
            # Check for minimum required step fields
            if 'action' not in step:
                logger.debug(f"Step missing action: {step}")
                return False
        
        return True
    
    def _save_checkpoint(self, samples: List[Dict], checkpoint_path: Path):
        """
        Save samples to checkpoint file.
        
        Args:
            samples: List of samples to save
            checkpoint_path: Path to save checkpoint
        """
        try:
            # Write to temporary file first for atomicity
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Atomic rename
            temp_path.replace(checkpoint_path)
            
            logger.debug(f"Saved checkpoint with {len(samples)} samples to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _log_statistics(self):
        """Log generation statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        stats_msg = (
            f"Generation Statistics for '{self.task_name}':\n"
            f"  - Samples generated: {self.generation_stats['samples_generated']}\n"
            f"  - Invalid samples: {self.generation_stats['samples_invalid']}\n"
            f"  - Failed samples: {self.generation_stats['samples_failed']}\n"
            f"  - Successful API calls: {self.generation_stats['api_calls_successful']}\n"
            f"  - Failed API calls: {self.generation_stats['api_calls_failed']}\n"
            f"  - Elapsed time: {elapsed_time:.2f} seconds\n"
            f"  - Generation rate: {self.generation_stats['samples_generated'] / max(elapsed_time, 1):.2f} samples/sec"
        )
        
        logger.info(stats_msg)