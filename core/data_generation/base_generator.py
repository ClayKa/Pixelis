# core/data_generation/base_generator.py
"""
BaseGenerator Class
===================
Core logic for all data generation tasks including API connections,
request throttling, error handling, and prompt management.
"""

import os
import json
import time
import logging
import requests
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Base class encapsulating core logic for data generation.
    
    This class handles:
    - API connections and request management
    - Prompt loading and formatting
    - Error handling and retries
    - Request throttling
    """
    
    def __init__(self, config_path: str = "configs/data_generation_manifest.yaml"):
        """
        Initialize the BaseGenerator with configuration.
        
        Args:
            config_path: Path to the data generation manifest YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_configuration()
        self.api_config = self._setup_api_config()
        self.prompts = {}
        self.request_count = 0
        self.last_request_time = 0
        
        # Load all prompts into memory
        self._load_all_prompts()
        
        logger.info(f"BaseGenerator initialized with config from {self.config_path}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load the data generation manifest."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration with {len(config.get('tasks', {}))} tasks")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_api_config(self) -> Dict[str, Any]:
        """Setup API configuration from manifest or environment."""
        api_config = self.config.get('global_config', {}).get('api_config', {})
        
        # Get API key from environment variable
        api_key_env = api_config.get('api_key_env_variable', 'OPENROUTER_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            logger.warning(f"API key not found in environment variable: {api_key_env}")
        
        return {
            'api_key': api_key,
            'api_key_env': api_key_env,
            'model': api_config.get('model', 'qwen/qwen3-8b:free'),
            'base_url': api_config.get('base_url', 'https://openrouter.ai/api/v1/chat/completions'),
            'temperature': api_config.get('temperature', 0.7),
            'max_tokens': api_config.get('max_tokens', 2048),
            'retry_attempts': api_config.get('retry_attempts', 3),
            'retry_delay': api_config.get('retry_delay', 1),
            'rate_limit_delay': api_config.get('rate_limit_delay', 0.5)
        }
    
    def _load_all_prompts(self) -> None:
        """Load all prompt templates from the configuration."""
        tasks = self.config.get('tasks', {})
        
        for task_name, task_config in tasks.items():
            generator_params = task_config.get('generator_params', {})
            prompt_path = generator_params.get('prompt_template_path')
            
            if prompt_path:
                prompt_content = self._load_prompt_from_file(prompt_path)
                if prompt_content:
                    self.prompts[task_name] = prompt_content
                    logger.debug(f"Loaded prompt for task: {task_name}")
    
    def _load_prompt_from_file(self, prompt_path: str) -> Optional[str]:
        """Load a prompt template from a file."""
        path = Path(prompt_path)
        if not path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Loaded prompt from {prompt_path}: {len(content)} chars")
                return content
        except Exception as e:
            logger.error(f"Failed to load prompt from {prompt_path}: {e}")
            return None
    
    def generate(self, prompt_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Generate content using the specified prompt and variables.
        
        Args:
            prompt_name: Name of the prompt/task from the configuration
            variables: Dictionary of variables to format the prompt with
            
        Returns:
            API response text or None if generation failed
        """
        # Get the prompt template
        prompt_template = self.prompts.get(prompt_name)
        if not prompt_template:
            logger.error(f"Prompt not found for task: {prompt_name}")
            return None
        
        # Format the prompt with variables
        try:
            formatted_prompt = self._format_prompt(prompt_template, variables)
        except Exception as e:
            logger.error(f"Failed to format prompt: {e}")
            return None
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Make API call with retry logic
        for attempt in range(self.api_config['retry_attempts']):
            try:
                response = self._call_api(formatted_prompt)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.api_config['retry_attempts'] - 1:
                    time.sleep(self.api_config['retry_delay'] * (attempt + 1))
        
        logger.error(f"All API call attempts failed for prompt: {prompt_name}")
        return None
    
    def _format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Format a prompt template with provided variables.
        
        Args:
            template: Prompt template with placeholders
            variables: Variables to replace placeholders
            
        Returns:
            Formatted prompt string
        """
        formatted = template
        
        # Replace all variables in the template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))
        
        # Check for any remaining placeholders
        import re
        remaining = re.findall(r'\{[^}]+\}', formatted)
        if remaining:
            logger.warning(f"Unresolved placeholders in prompt: {remaining}")
        
        return formatted
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.api_config['rate_limit_delay']:
            sleep_time = self.api_config['rate_limit_delay'] - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """
        Make an API call with the formatted prompt.
        
        Args:
            prompt: Formatted prompt to send to the API
            
        Returns:
            API response text or None if the call failed
        """
        if not self.api_config['api_key']:
            logger.error("API key not configured")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pixelis-ai/pixelis",
            "X-Title": "Pixelis Data Generation"
        }
        
        data = {
            "model": self.api_config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that generates structured data for visual reasoning tasks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.api_config['temperature'],
            "max_tokens": self.api_config['max_tokens']
        }
        
        try:
            logger.debug(f"Making API request to {self.api_config['base_url']}")
            response = requests.post(
                self.api_config['base_url'],
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.debug(f"API call successful, received {len(content)} chars")
                return content
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited, retry after {retry_after}s")
                time.sleep(retry_after)
                return None
            
            logger.error(f"API error {response.status_code}: {response.text}")
            return None
            
        except requests.exceptions.Timeout:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            return None
    
    def get_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task configuration dictionary or None
        """
        return self.config.get('tasks', {}).get(task_name)
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks from configuration."""
        return list(self.config.get('tasks', {}).keys())
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            'request_count': self.request_count,
            'api_model': self.api_config['model'],
            'api_configured': bool(self.api_config['api_key'])
        }
    
    @abstractmethod
    def process_response(self, response: str, task_name: str) -> Any:
        """
        Process the API response for a specific task.
        
        This method should be overridden by specialized generators
        to handle task-specific response processing.
        
        Args:
            response: Raw API response text
            task_name: Name of the task being processed
            
        Returns:
            Processed response in task-specific format
        """
        pass