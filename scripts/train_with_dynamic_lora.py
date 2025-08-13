#!/usr/bin/env python3
"""
Example training script using dynamic LoRA configuration
Demonstrates how to use the SVD-determined ranks for efficient fine-tuning
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.models import create_model_with_dynamic_lora
from scripts.preliminary_finetune import CoTADataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function with dynamic LoRA"""
    
    parser = argparse.ArgumentParser(
        description="Train model with dynamic LoRA configuration"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B",
        help="Base model name or path"
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default="configs/lora_rank_config.json",
        help="Path to LoRA rank configuration from SVD analysis"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models/dynamic_lora_model",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 training"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    
    args = parser.parse_args()
    
    # Verify LoRA configuration exists
    if not Path(args.lora_config).exists():
        logger.error(f"LoRA configuration not found at {args.lora_config}")
        logger.error("Please run scripts/run_svd_analysis_workflow.sh first")
        return 1
    
    # Load and display LoRA configuration
    with open(args.lora_config, 'r') as f:
        lora_config = json.load(f)
    
    logger.info("="*80)
    logger.info("DYNAMIC LORA CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model: {lora_config.get('model_name', 'Unknown')}")
    logger.info(f"Timestamp: {lora_config.get('timestamp', 'Unknown')}")
    logger.info(f"Compression ratio: {lora_config.get('compression_ratio', 0):.2%}")
    logger.info(f"Layer ranks: {lora_config.get('layer_ranks', {})}")
    logger.info("="*80)
    
    # Create model with dynamic LoRA
    logger.info("Loading model with dynamic LoRA configuration...")
    
    peft_model, tokenizer_or_processor = create_model_with_dynamic_lora(
        model_name=args.model_name,
        rank_config_path=args.lora_config,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map="auto",
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )
    
    # Get tokenizer (handle both tokenizer and processor cases)
    if hasattr(tokenizer_or_processor, 'tokenizer'):
        tokenizer = tokenizer_or_processor.tokenizer
    else:
        tokenizer = tokenizer_or_processor
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info("Loading training dataset...")
    train_dataset = CoTADataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=tokenizer_or_processor if hasattr(tokenizer_or_processor, 'tokenizer') else None
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training with dynamic LoRA...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Model saved to: {args.output_dir}")
    print(f"LoRA configuration used: {args.lora_config}")
    print("\nTo load the trained model:")
    print("```python")
    print("from peft import PeftModel")
    print("from transformers import AutoModelForCausalLM")
    print(f"base_model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    print(f"model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")
    print("```")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())