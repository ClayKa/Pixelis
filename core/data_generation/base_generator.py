"""
Base Task Generator for CoTA (Chain-of-Thought-Action) data synthesis.

This module provides the abstract base class for all specialized task generators,
handling common functionality including API client initialization, prompt loading,
generation orchestration, checkpointing, and error handling.
"""

import json
import time
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import os
import hashlib
from collections import defaultdict

try:
    import openai
except ImportError:  # pragma: no cover - exercised in minimal environments
    openai = None

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
        self.task_name = self.config.get('name') or self.config.get('task_name', 'unknown_task')
        self.generator_params = self.config.get('generator_params', {})
        
        # Configure validation strictness (default: strict for better data quality)
        self.validation_strictness = (
            self.config.get('generator_config', {}).get('validation_strictness')
            or self.config.get('validation_strictness')
            or 'strict'
        )
        # Options: 'strict', 'standard', 'lenient', 'ultra_lenient'
        logger.info(f"Validation strictness set to: {self.validation_strictness}")
        
        # Initialize core components
        self.prompt_template = self._load_prompt_template()
        self.api_client = self._initialize_api_client()
        
        # [REVISED] The base class ONLY loads the raw template text.
        # It does not parse any special blocks within it.
        # Subclasses are responsible for their own prompt-specific parsing.
        
        # Statistics tracking
        self.generation_stats = defaultdict(int)
        self.start_time = None
        
        # [NEW] Track used source samples to guarantee uniqueness
        self.used_source_sample_ids = set()
        
        # [NEW] Support for continuation mode (when called multiple times)
        self.continuation_mode = False
    
    def _load_prompt_template(self) -> str:
        """
        Load the prompt template from the configured file.
        
        Returns:
            The loaded prompt template string
        
        Raises:
            FileNotFoundError: If the prompt template file doesn't exist
        """
        prompt_template = self.config.get('prompt_template') or self.config.get('prompt_template_path')

        if not prompt_template:
            templates_dir = (
                self.global_config.get('data_generation', {}).get(
                    'prompt_templates_dir',
                    'core/data_generation/prompt_templates'
                )
            )
            stem = self.task_name[:-5] if self.task_name.endswith('_task') else self.task_name
            prompt_template = str(Path(templates_dir) / f"{stem}.md")

        prompt_path = Path(prompt_template)
        if not prompt_path.is_file():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            prompt_path = project_root / prompt_path
            
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt template not found for task '{self.task_name}': {prompt_path}"
            )
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        logger.info(f"Loaded prompt template from {prompt_path}")
        return template
    
    def _initialize_api_client(self) -> Optional[Any]:
        """
        Initializes the API client and verifies the API key presence.
        """
        if openai is None:
            logger.warning("OpenAI client package is not installed; using mock generation mode.")
            return None

        # Get API configuration from global config
        api_profile = self.global_config.get('api_profiles', {}).get('generator_api', {})
        api_key_env_var = api_profile.get('api_key_env_variable', 'OPENROUTER_API_KEY')
        
        # Get API key from environment
        api_key = os.getenv(api_key_env_var)
        
        # [DEBUG LOG 1] Verify that the API key was found in the environment.
        if not api_key:
            logger.error(f"CRITICAL: API key environment variable '{api_key_env_var}' is not set or is empty!")
            logger.warning("Using mock mode for development since no API key is available.")
            return None  # Will trigger mock mode in _call_llm_api
        else:
            logger.info(f"API Key '{api_key_env_var}' found and loaded successfully.")
        
        base_url = api_profile.get('api_base_url', 'https://openrouter.ai/api/v1')
        
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "Pixelis Project"},
        )
        
        logger.info(f"Initialized API client for '{self.task_name}' with base URL: {base_url}")
        return client
    
    @abstractmethod
    def _build_context_placeholders(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        [REVISED SIGNATURE]
        Build the context placeholders for the prompt template.
        
        Subclasses must now return a tuple:
        1. The dictionary of placeholders for the prompt.
        2. A dictionary of initial metadata for the sample.
        
        Returns:
            Tuple of (placeholders_dict, metadata_dict)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _build_context_placeholders()"
        )
    
    def _normalize_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Normalizes trajectory steps to handle various formats from different LLMs.
        This ensures consistent structure regardless of how the LLM formatted its response.
        
        Args:
            trajectory: List of trajectory steps in various possible formats
            
        Returns:
            Normalized trajectory with consistent structure
        """
        normalized_traj = []
        
        for step in trajectory:
            if not isinstance(step, dict):
                logger.debug(f"Skipping non-dict step: {type(step)}")
                continue
                
            new_step = {}
            
            # Normalize step type
            if 'type' in step:
                new_step['type'] = step['type'].lower() if isinstance(step['type'], str) else step['type']
            elif 'step' in step and isinstance(step['step'], str):
                new_step['type'] = step['step'].lower()
            elif 'step_type' in step:
                new_step['type'] = step['step_type'].lower()
            elif 'THOUGHT' in step:
                new_step['type'] = 'thought'
            elif 'ACTION' in step:
                new_step['type'] = 'action'
            elif 'action' in step:  # Direct action field
                new_step['type'] = 'action'
            elif 'thought' in step:  # Direct thought field
                new_step['type'] = 'thought'
            else:
                # Try to infer from content
                if any(key in step for key in ['name', 'operation', 'command']):
                    new_step['type'] = 'action'
                elif any(key in step for key in ['content', 'text', 'description']):
                    new_step['type'] = 'thought'
                else:
                    logger.debug(f"Cannot determine type for step: {step}")
                    continue  # Skip malformed step
            
            # Normalize content for thoughts
            if new_step['type'] == 'thought':
                if 'content' in step:
                    new_step['content'] = step['content']
                elif 'text' in step:
                    new_step['content'] = step['text']
                elif 'description' in step:
                    new_step['content'] = step['description']
                elif 'THOUGHT' in step:
                    new_step['content'] = step['THOUGHT']
                elif 'thought' in step:
                    new_step['content'] = step['thought']
                elif 'message' in step:
                    new_step['content'] = step['message']
                else:
                    # Try to extract any string value
                    for key, value in step.items():
                        if isinstance(value, str) and key not in ['type', 'step', 'step_type']:
                            new_step['content'] = value
                            break
            
            # Normalize action details
            if new_step['type'] == 'action':
                # Normalize action name
                if 'name' in step:
                    new_step['name'] = step['name']
                elif 'action' in step:
                    # Could be the action name or a nested dict
                    if isinstance(step['action'], str):
                        new_step['name'] = step['action']
                    elif isinstance(step['action'], dict) and 'name' in step['action']:
                        new_step['name'] = step['action']['name']
                        # Also extract parameters from nested structure
                        if 'parameters' in step['action']:
                            new_step['parameters'] = step['action']['parameters']
                elif 'ACTION' in step:
                    if isinstance(step['ACTION'], str):
                        new_step['name'] = step['ACTION']
                    elif isinstance(step['ACTION'], dict):
                        new_step['name'] = step['ACTION'].get('name', 'unknown')
                        new_step['parameters'] = step['ACTION'].get('parameters', {})
                elif 'operation' in step:
                    new_step['name'] = step['operation']
                elif 'command' in step:
                    new_step['name'] = step['command']
                
                # Normalize parameters
                if 'parameters' not in new_step:
                    if 'parameters' in step:
                        new_step['parameters'] = step['parameters']
                    elif 'params' in step:
                        new_step['parameters'] = step['params']
                    elif 'args' in step:
                        new_step['parameters'] = step['args']
                    elif 'arguments' in step:
                        new_step['parameters'] = step['arguments']
                    else:
                        # Extract any dict that looks like parameters
                        for key, value in step.items():
                            if isinstance(value, dict) and key not in ['type', 'action', 'ACTION']:
                                new_step['parameters'] = value
                                break
                        else:
                            new_step['parameters'] = {}
                
                # Add observation if present
                if 'observation' in step:
                    new_step['observation'] = step['observation']
                elif 'result' in step:
                    new_step['observation'] = step['result']
                elif 'output' in step:
                    new_step['observation'] = step['output']
                elif 'response' in step:
                    new_step['observation'] = step['response']
            
            # Only add the step if it has meaningful content
            if new_step.get('type') == 'thought' and 'content' in new_step:
                normalized_traj.append(new_step)
            elif new_step.get('type') == 'action' and 'name' in new_step:
                normalized_traj.append(new_step)
            else:
                logger.debug(f"Skipping incomplete step: {new_step}")
        
        return normalized_traj
    
    @abstractmethod
    def _validate_and_process_response(self, llm_response: Dict, context: Dict) -> Optional[Dict]:
        """
        [NEW ABSTRACT METHOD]
        Subclasses must implement this to validate the LLM's JSON output
        and process it into the final CoTA sample format.
        
        Args:
            llm_response: The raw JSON response from the LLM
            context: The context placeholders used for generation
            
        Returns:
            The final, validated CoTA sample dict, or None if validation fails.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _validate_and_process_response()"
        )
    
    def _call_llm_api(self, prompt: str, attempt: int = 1, context_placeholders: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calls the configured LLM API with extensive debugging and robust parsing.
        
        Args:
            prompt: The formatted prompt to send
            attempt: Current attempt number for retry tracking
            
        Returns:
            Parsed JSON response from the LLM
            
        Raises:
            Exception: After max retries are exhausted
        """
        logger.info("--- Preparing to call LLM API ---")
        raw_response_content = "Error: Response was not captured." # Default error message
        
        max_retries = self.generator_params.get('max_retries', 3)
        retry_delay = self.generator_params.get('retry_delay', 2.0)
        
        # Mock mode for development/testing
        if self.api_client is None:
            logger.debug("API client is None, using mock mode")
            return self._generate_mock_response(context_placeholders)
        
        try:
            # Get API configuration
            api_profile = self.global_config.get('api_profiles', {}).get('generator_api', {})
            model_to_use = api_profile.get('model', 'meta-llama/llama-3.3-8b-instruct:free')
            
            # [DEBUG LOG 2] Log the exact payload being sent to the API.
            payload = {
                "model": model_to_use,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": api_profile.get("temperature", 0.7),
                "max_tokens": api_profile.get("max_tokens", 2048)
            }
            logger.debug(f"API Request Payload:\n{json.dumps(payload, indent=2)}")
            
            # Make the actual API call
            response = self.api_client.chat.completions.create(**payload)
            
            # [DEBUG LOG 3] Log the full, successful Pydantic response object.
            # This shows us exactly what the server returned on success.
            logger.debug(f"Full API Response Object (Success):\n{response.model_dump_json(indent=2)}")
            
            raw_response_content = response.choices[0].message.content
            
        except Exception as e:
            # [DEBUG LOG 4 - CRITICAL] Log the specific error type and details.
            # This will capture AuthenticationError, RateLimitError, etc.
            logger.error(f"!!! API call to model '{model_to_use}' FAILED !!!")
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Details: {e}")
            
            # Retry logic
            if attempt < max_retries:
                logger.warning(
                    f"API call failed (attempt {attempt}/{max_retries}). "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay * attempt)  # Exponential backoff
                return self._call_llm_api(prompt, attempt + 1, context_placeholders)
            else:
                logger.error(f"API call failed after {max_retries} attempts")
                raise # Re-raise to mark the sample as failed
        
        # --- Robust JSON Parsing ---
        try:
            json_match = re.search(r'\{.*\}', raw_response_content, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON object found in the LLM response.")
            json_string = json_match.group(0)
            parsed_response = json.loads(json_string)
            self.generation_stats['api_calls_successful'] += 1
            return parsed_response
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from a successful API response. Error: {e}")
            # [DEBUG LOG 5] Log the raw text content that failed to parse.
            logger.error(f"Raw response content that caused parsing error:\n---\n{raw_response_content}\n---")
            raise
    
    def _generate_mock_response(self, context_placeholders: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a mock response for development/testing.
        
        Args:
            context_placeholders: Optional context with ground truth
            
        Returns:
            Mock CoTA trajectory in the expected format
        """
        # Generate a mock question based on the task
        questions = [
            "What detail can you see in the specified area?",
            "Is there something specific in this region?",
            "Can you identify what's in the zoomed area?",
            "What object is visible at this location?"
        ]
        question = random.choice(questions)
        
        # Use ground truth if available, otherwise use default mock answer
        final_answer = "A detail is visible in the specified area"
        if context_placeholders and 'ground_truth_answer' in context_placeholders:
            ground_truth = context_placeholders['ground_truth_answer']
            # Sometimes return exact match, sometimes return variation to test flexible validation
            variations = [
                ground_truth,  # Exact match
                f"Yes, {ground_truth.lower()}",  # Answer with prefix
                f"The answer is: {ground_truth}",  # Answer with explanation
                f"I can see {ground_truth} in the image",  # Contextual answer
            ]
            final_answer = random.choice(variations)
            logger.debug(f"Mock generator ground truth: '{ground_truth}' -> generated: '{final_answer}'")
        
        # Get difficulty from context or random
        difficulty = "Easy"
        if context_placeholders and 'difficulty' in context_placeholders:
            difficulty = context_placeholders['difficulty']
        
        # Get bbox from context or use default
        bbox = [100, 100, 200, 200]
        if context_placeholders and 'bbox' in context_placeholders:
            bbox_str = context_placeholders['bbox']
            # Parse bbox string if needed
            if isinstance(bbox_str, str) and bbox_str.startswith('['):
                try:
                    import ast
                    bbox = ast.literal_eval(bbox_str)
                except:
                    pass
        
        # Generate proper trajectory structure matching the prompt format
        trajectory = [
            {
                "type": "thought",
                "content": "I need to examine the specified area to find the detail. Let me zoom in to get a clearer view."
            },
            {
                "type": "action",
                "name": "ZOOM-IN",
                "parameters": {
                    "bbox": bbox
                }
            },
            {
                "type": "thought",
                "content": f"The zoom was successful. I can now see the detail clearly. The observation is: {final_answer}"
            }
        ]
        
        return {
            "question": question,
            "difficulty": difficulty,
            "trajectory": trajectory,
            "final_answer": final_answer,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "is_mock": True,
                "task_name": self.task_name
            }
        }
    
    def _finalize_metadata(self, cota_sample: Dict[str, Any], initial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines initial metadata from the subclass with global generator metadata.
        
        Args:
            cota_sample: The generated CoTA sample
            initial_metadata: Metadata from the specific task generator
            
        Returns:
            Sample with finalized metadata
        """
        # Start with the specific metadata from the generator
        final_metadata = initial_metadata.copy()
        
        # Add global metadata from the base class
        api_profile = self.global_config.get('api_profiles', {}).get('generator_api', {})
        
        final_metadata.update({
            'task_name': self.task_name,
            'llm_model_used': api_profile.get('model', 'unknown'),
            'temperature': api_profile.get('temperature', 0.7)
        })
        
        # Ensure the sample has a metadata key and update it
        if 'metadata' not in cota_sample:
            cota_sample['metadata'] = {}
        
        cota_sample['metadata'].update(final_metadata)
        
        return cota_sample
    
    
    def generate(self, num_samples: int) -> List[Dict]:
        """
        [REVISED - STATELESS]
        Generates a requested number of new samples without handling checkpoints.
        The method focuses purely on generation and validation, leaving state
        management to the caller.
        
        Args:
            num_samples: Number of samples to attempt generating
            
        Returns:
            List of valid CoTA samples (may be less than num_samples if some fail)
        """
        newly_generated_samples = []
        self.start_time = time.time()
        
        # Simple progress bar for generation attempts
        pbar = tqdm(
            range(num_samples),
            desc=f"Generating NEW '{self.task_name}'"
        )
        
        # Main generation loop - simple iteration
        for i in pbar:
            # Retry logic for validation failures
            max_retries = self.generator_params.get('max_validation_retries', 3)
            retry_count = 0
            sample_generated = False
            
            while retry_count < max_retries and not sample_generated:
                try:
                    # 1. Subclass builds the context
                    context_result = self._build_context_placeholders()
                    
                    # Handle both old (single dict) and new (tuple) return formats
                    if isinstance(context_result, tuple):
                        context_placeholders, initial_metadata = context_result
                    else:
                        # Backward compatibility for generators not yet updated
                        context_placeholders = context_result
                        initial_metadata = {}
                    
                    # 2. Base class formats the prompt
                    final_prompt = ""
                    try:
                        final_prompt = self.prompt_template.format(**context_placeholders)
                    except KeyError as ke:
                        # If a key is missing, log error and continue
                        logger.error(f"FATAL: Missing placeholder in prompt template for task '{self.task_name}'.")
                        logger.error(f"  --> The prompt template requires the key: {ke}")
                        logger.error(f"  --> The generator only provided these keys: {list(context_placeholders.keys())}")
                        self.generation_stats['samples_failed'] += 1
                        break  # Skip this sample entirely
                    
                    # 3. Base class calls the API
                    llm_response = self._call_llm_api(final_prompt, context_placeholders=context_placeholders)
                    
                    # 4. Subclass validates the response. This is the critical gate.
                    # It returns a valid sample dict, or None if validation fails.
                    cota_sample = self._validate_and_process_response(llm_response, context_placeholders)
                    
                    if cota_sample:
                        # SUCCESS CASE - Add valid sample to our list
                        cota_sample = self._finalize_metadata(cota_sample, initial_metadata)
                        newly_generated_samples.append(cota_sample)
                        self.generation_stats['samples_generated'] += 1
                        sample_generated = True
                        
                        pbar.set_postfix({
                            'valid': self.generation_stats['samples_generated'],
                            'invalid': self.generation_stats['samples_invalid'],
                            'failed': self.generation_stats['samples_failed'],
                            'retries': retry_count
                        })
                        
                        if retry_count > 0:
                            logger.info(f"Sample {i+1} passed validation after {retry_count} retries")
                        else:
                            logger.debug(f"Sample {i+1} passed validation")
                    else:
                        # VALIDATION FAILED CASE - Will retry
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.debug(f"Sample {i+1} failed validation, retrying ({retry_count}/{max_retries})")
                        else:
                            # Max retries exhausted
                            self.generation_stats['samples_invalid'] += 1
                            logger.warning(f"Sample {i+1} failed validation after {max_retries} attempts")
                    
                except KeyboardInterrupt:
                    logger.info("Generation interrupted by user.")
                    pbar.close()
                    raise
                    
                except Exception as e:
                    # HARD FAILURE CASE (e.g., API is down)
                    logger.error(f"Failed to generate sample {i+1}: {e}")
                    self.generation_stats['samples_failed'] += 1
                    break  # Stop retrying for this sample
        
        pbar.close()
        
        # Log final statistics
        self._log_statistics()
        
        logger.info(
            f"Completed generation batch for '{self.task_name}': "
            f"{len(newly_generated_samples)} valid samples generated out of {num_samples} attempts"
        )
        
        return newly_generated_samples
    
    
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
