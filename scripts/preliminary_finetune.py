#!/usr/bin/env python3
"""
Preliminary Full Fine-Tuning Script for SVD Analysis
This script performs a brief full-parameter fine-tuning run on a small subset
of training data to produce a checkpoint for SVD analysis.
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class PreliminaryFinetuneConfig:
    """Configuration for preliminary fine-tuning"""
    
    # Model configuration
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B"
    model_type: str = "qwen2_vl"  # or "qwen3"
    
    # Data configuration
    data_path: str = None
    data_subset_ratio: float = 0.01  # Use 1% of data for preliminary tuning
    max_samples: int = 1000  # Maximum number of samples
    stratified_sampling: bool = True  # Maintain distribution of task types
    
    # Training configuration
    output_dir: str = "saved_models/preliminary_finetune"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    
    # WandB
    use_wandb: bool = False
    wandb_project: str = "pixelis-preliminary-finetune"
    wandb_run_name: str = None


class CoTADataset(Dataset):
    """Dataset for Chain-of-Thought-Action (CoTA) training data"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        processor=None,
        max_length: int = 2048,
        subset_ratio: float = 1.0,
        max_samples: int = None,
        stratified: bool = True
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Apply stratified sampling if requested
        if stratified and subset_ratio < 1.0:
            self.data = self._stratified_sample(self.data, subset_ratio, max_samples)
        elif subset_ratio < 1.0:
            # Random sampling
            n_samples = min(
                int(len(self.data) * subset_ratio),
                max_samples if max_samples else len(self.data)
            )
            indices = random.sample(range(len(self.data)), n_samples)
            self.data = [self.data[i] for i in indices]
            
        logger.info(f"Loaded {len(self.data)} samples for training")
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSON file"""
        
        if not os.path.exists(data_path):
            # Generate synthetic data for testing
            logger.warning(f"Data file not found at {data_path}. Generating synthetic data...")
            return self._generate_synthetic_data()
        
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Ensure data is a list
        if isinstance(data, dict):
            data = data.get('data', [])
            
        return data
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> List[Dict]:
        """Generate synthetic CoTA data for testing"""
        
        synthetic_data = []
        task_types = ['visual_reasoning', 'object_detection', 'text_reading', 'tracking']
        
        for i in range(n_samples):
            task_type = random.choice(task_types)
            
            # Create a synthetic sample
            sample = {
                'id': f'synthetic_{i}',
                'task_type': task_type,
                'image': None,  # Placeholder
                'question': f"Sample question {i} for {task_type}",
                'chain_of_thought': [
                    f"Step 1: Analyze the image",
                    f"Step 2: Apply {task_type} reasoning",
                    f"Step 3: Generate answer"
                ],
                'actions': [
                    {'action': 'SEGMENT_OBJECT_AT', 'params': {'x': 100, 'y': 200}},
                    {'action': 'GET_PROPERTIES', 'params': {'object_id': 1}}
                ],
                'answer': f"Answer for sample {i}"
            }
            synthetic_data.append(sample)
            
        return synthetic_data
    
    def _stratified_sample(
        self, 
        data: List[Dict], 
        ratio: float, 
        max_samples: int = None
    ) -> List[Dict]:
        """Perform stratified sampling to maintain task type distribution"""
        
        # Group by task type
        task_groups = {}
        for sample in data:
            task_type = sample.get('task_type', 'unknown')
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(sample)
        
        # Sample from each group proportionally
        sampled_data = []
        for task_type, samples in task_groups.items():
            n_samples = int(len(samples) * ratio)
            if n_samples > 0:
                sampled = random.sample(samples, min(n_samples, len(samples)))
                sampled_data.extend(sampled)
        
        # Apply max_samples limit if specified
        if max_samples and len(sampled_data) > max_samples:
            sampled_data = random.sample(sampled_data, max_samples)
            
        random.shuffle(sampled_data)
        return sampled_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Format input text
        input_text = self._format_input(sample)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # For language modeling, labels are the same as input_ids
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs
    
    def _format_input(self, sample: Dict) -> str:
        """Format sample into training text"""
        
        parts = []
        
        # Add question
        parts.append(f"Question: {sample.get('question', '')}")
        
        # Add chain of thought
        if 'chain_of_thought' in sample:
            parts.append("Reasoning:")
            for i, step in enumerate(sample['chain_of_thought'], 1):
                parts.append(f"  {i}. {step}")
        
        # Add actions
        if 'actions' in sample:
            parts.append("Actions:")
            for action in sample['actions']:
                action_str = f"  - {action['action']}"
                if 'params' in action:
                    param_str = ', '.join(f"{k}={v}" for k, v in action['params'].items())
                    action_str += f"({param_str})"
                parts.append(action_str)
        
        # Add answer
        parts.append(f"Answer: {sample.get('answer', '')}")
        
        return "\n".join(parts)


class PreliminaryFineTuner:
    """Handles preliminary fine-tuning for SVD analysis"""
    
    def __init__(self, config: PreliminaryFinetuneConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"preliminary_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config.__dict__
            )
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        if "qwen2.5-vl" in self.config.model_name_or_path.lower():
            # Load Qwen2.5-VL model
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.config.model_name_or_path)
            self.tokenizer = self.processor.tokenizer
            
        else:
            # Load standard Qwen model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
            self.processor = None
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B parameters")
    
    def prepare_dataset(self):
        """Prepare the training dataset"""
        
        # Create dataset
        self.train_dataset = CoTADataset(
            data_path=self.config.data_path or "data/cota_train.json",
            tokenizer=self.tokenizer,
            processor=self.processor,
            subset_ratio=self.config.data_subset_ratio,
            max_samples=self.config.max_samples,
            stratified=self.config.stratified_sampling
        )
        
        # Create validation dataset (10% of training data)
        val_size = max(1, int(len(self.train_dataset) * 0.1))
        val_indices = random.sample(range(len(self.train_dataset)), val_size)
        self.val_dataset = Subset(self.train_dataset, val_indices)
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def create_training_arguments(self):
        """Create training arguments"""
        
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name=self.config.wandb_run_name,
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            remove_unused_columns=False
        )
    
    def train(self):
        """Run the preliminary fine-tuning"""
        
        logger.info("Starting preliminary fine-tuning...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        self.prepare_dataset()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Save pretrained checkpoint (for SVD analysis)
        pretrained_path = Path(self.config.output_dir) / "pretrained"
        pretrained_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving pretrained model to {pretrained_path}")
        trainer.save_model(pretrained_path)
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        final_path = Path(self.config.output_dir) / "finetuned"
        final_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving finetuned model to {final_path}")
        trainer.save_model(final_path)
        
        # Save training metrics
        metrics_path = Path(self.config.output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info("Preliminary fine-tuning completed!")
        logger.info(f"Pretrained checkpoint: {pretrained_path}")
        logger.info(f"Finetuned checkpoint: {final_path}")
        
        return pretrained_path, final_path


def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preliminary fine-tuning for SVD analysis"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B",
        help="Model name or path"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models/preliminary_finetune",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=0.01,
        help="Fraction of data to use (0-1)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of training samples"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable WandB logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PreliminaryFinetuneConfig(
        model_name_or_path=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        data_subset_ratio=args.subset_ratio,
        max_samples=args.max_samples,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        seed=args.seed
    )
    
    # Run training
    tuner = PreliminaryFineTuner(config)
    pretrained_path, finetuned_path = tuner.train()
    
    print("\n" + "="*80)
    print("PRELIMINARY FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Pretrained checkpoint: {pretrained_path}")
    print(f"Finetuned checkpoint: {finetuned_path}")
    print("\nNext step: Run SVD analysis")
    print(f"python scripts/analyze_lora_ranks.py \\")
    print(f"  --pretrained {pretrained_path} \\")
    print(f"  --finetuned {finetuned_path}")
    print("="*80)


if __name__ == "__main__":
    main()