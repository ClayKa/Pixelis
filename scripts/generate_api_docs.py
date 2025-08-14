#!/usr/bin/env python3
"""
Generate API documentation for Pixelis core modules.

This script automatically generates comprehensive API documentation
from docstrings in the codebase using Sphinx-style formatting.
"""

import os
import sys
import ast
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class APIItem:
    """Represents a documented API item."""
    name: str
    type: str  # 'class', 'function', 'method', 'property'
    module: str
    signature: Optional[str]
    docstring: Optional[str]
    members: List['APIItem'] = None
    
    def __post_init__(self):
        if self.members is None:
            self.members = []


class APIDocGenerator:
    """Generates API documentation from Python modules."""
    
    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_items: Dict[str, List[APIItem]] = {}
    
    def extract_module_api(self, module_path: str) -> List[APIItem]:
        """Extract API documentation from a Python module."""
        items = []
        
        # Convert file path to module name
        rel_path = Path(module_path).relative_to(project_root)
        module_name = str(rel_path.with_suffix('')).replace('/', '.')
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get all public members
            for name, obj in inspect.getmembers(module):
                if name.startswith('_'):
                    continue
                
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    items.append(self._document_class(name, obj, module_name))
                elif inspect.isfunction(obj) and obj.__module__ == module_name:
                    items.append(self._document_function(name, obj, module_name))
        
        except Exception as e:
            print(f"Warning: Could not process {module_name}: {e}")
        
        return items
    
    def _document_class(self, name: str, cls: type, module: str) -> APIItem:
        """Document a class and its members."""
        item = APIItem(
            name=name,
            type='class',
            module=module,
            signature=self._get_class_signature(cls),
            docstring=inspect.getdoc(cls),
            members=[]
        )
        
        # Document methods
        for method_name, method in inspect.getmembers(cls):
            if method_name.startswith('_') and not method_name.startswith('__'):
                continue  # Skip private methods
            
            if inspect.isfunction(method) or inspect.ismethod(method):
                method_item = APIItem(
                    name=method_name,
                    type='method',
                    module=module,
                    signature=self._get_signature(method),
                    docstring=inspect.getdoc(method)
                )
                item.members.append(method_item)
            elif isinstance(method, property):
                prop_item = APIItem(
                    name=method_name,
                    type='property',
                    module=module,
                    signature=None,
                    docstring=inspect.getdoc(method.fget) if method.fget else None
                )
                item.members.append(prop_item)
        
        return item
    
    def _document_function(self, name: str, func: callable, module: str) -> APIItem:
        """Document a standalone function."""
        return APIItem(
            name=name,
            type='function',
            module=module,
            signature=self._get_signature(func),
            docstring=inspect.getdoc(func)
        )
    
    def _get_signature(self, func: callable) -> str:
        """Get function signature as a string."""
        try:
            sig = inspect.signature(func)
            return str(sig)
        except:
            return "()"
    
    def _get_class_signature(self, cls: type) -> str:
        """Get class constructor signature."""
        try:
            init = cls.__init__
            if init is not object.__init__:
                sig = inspect.signature(init)
                # Remove 'self' parameter
                params = list(sig.parameters.values())[1:]
                new_sig = sig.replace(parameters=params)
                return str(new_sig)
        except:
            pass
        return "()"
    
    def scan_directory(self, directory: Path, pattern: str = "*.py"):
        """Scan directory for Python files and extract API documentation."""
        for py_file in directory.rglob(pattern):
            if '__pycache__' in str(py_file):
                continue
            if py_file.name == '__init__.py' and not py_file.read_text().strip():
                continue
            
            print(f"Processing {py_file}...")
            items = self.extract_module_api(str(py_file))
            
            if items:
                module_key = str(py_file.relative_to(project_root).parent)
                if module_key not in self.api_items:
                    self.api_items[module_key] = []
                self.api_items[module_key].extend(items)
    
    def generate_markdown(self):
        """Generate markdown documentation files."""
        # Generate main API index
        self._generate_index()
        
        # Generate module documentation
        for module_path, items in self.api_items.items():
            self._generate_module_doc(module_path, items)
    
    def _generate_index(self):
        """Generate main API index file."""
        index_path = self.output_dir / "index.md"
        
        content = ["# Pixelis API Reference\n\n"]
        content.append("## Core Modules\n\n")
        
        # Group by top-level module
        modules = {}
        for module_path in sorted(self.api_items.keys()):
            top_level = module_path.split('/')[0]
            if top_level not in modules:
                modules[top_level] = []
            modules[top_level].append(module_path)
        
        for top_level, paths in modules.items():
            content.append(f"### {top_level.title()}\n\n")
            for path in sorted(paths):
                link = path.replace('/', '_') + ".md"
                display_name = path.replace('/', '.')
                content.append(f"- [{display_name}]({link})\n")
            content.append("\n")
        
        index_path.write_text(''.join(content))
        print(f"Generated {index_path}")
    
    def _generate_module_doc(self, module_path: str, items: List[APIItem]):
        """Generate documentation for a single module."""
        output_file = self.output_dir / (module_path.replace('/', '_') + ".md")
        
        content = [f"# {module_path.replace('/', '.')}\n\n"]
        
        # Group items by type
        classes = [item for item in items if item.type == 'class']
        functions = [item for item in items if item.type == 'function']
        
        # Document classes
        if classes:
            content.append("## Classes\n\n")
            for cls in sorted(classes, key=lambda x: x.name):
                content.append(self._format_class(cls))
        
        # Document functions
        if functions:
            content.append("## Functions\n\n")
            for func in sorted(functions, key=lambda x: x.name):
                content.append(self._format_function(func))
        
        output_file.write_text(''.join(content))
        print(f"Generated {output_file}")
    
    def _format_class(self, item: APIItem) -> str:
        """Format class documentation."""
        lines = [f"### class `{item.name}`\n\n"]
        
        if item.signature and item.signature != "()":
            lines.append(f"```python\n{item.name}{item.signature}\n```\n\n")
        
        if item.docstring:
            lines.append(f"{item.docstring}\n\n")
        
        # Document methods
        methods = [m for m in item.members if m.type == 'method']
        properties = [m for m in item.members if m.type == 'property']
        
        if methods:
            lines.append("#### Methods\n\n")
            for method in sorted(methods, key=lambda x: x.name):
                lines.append(f"##### `{method.name}{method.signature}`\n\n")
                if method.docstring:
                    lines.append(f"{method.docstring}\n\n")
        
        if properties:
            lines.append("#### Properties\n\n")
            for prop in sorted(properties, key=lambda x: x.name):
                lines.append(f"##### `{prop.name}`\n\n")
                if prop.docstring:
                    lines.append(f"{prop.docstring}\n\n")
        
        lines.append("---\n\n")
        return ''.join(lines)
    
    def _format_function(self, item: APIItem) -> str:
        """Format function documentation."""
        lines = [f"### `{item.name}{item.signature}`\n\n"]
        
        if item.docstring:
            lines.append(f"{item.docstring}\n\n")
        
        lines.append("---\n\n")
        return ''.join(lines)


def generate_usage_examples():
    """Generate usage examples documentation."""
    examples_file = Path("docs/api/examples.md")
    
    content = """# Pixelis API Usage Examples

## Basic Inference

```python
from core.engine import InferenceEngine
from core.config_schema import InferenceConfig

# Initialize engine
config = InferenceConfig(
    model_path="outputs/models/pixelis-online",
    device="cuda",
    confidence_threshold=0.7
)
engine = InferenceEngine(config)

# Run inference
result = engine.infer(
    image="path/to/image.jpg",
    query="Count the number of cars"
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.trajectory}")
```

## Visual Operations

```python
from core.modules.operation_registry import VisualOperationRegistry
from PIL import Image

# Get registry instance
registry = VisualOperationRegistry()

# Load image
image = Image.open("path/to/image.jpg")

# Execute segmentation
result = registry.execute("SEGMENT_OBJECT_AT", {
    "image": image,
    "x": 100,
    "y": 200
})

# Extract text from region
text_result = registry.execute("READ_TEXT", {
    "image": result.segmented_region
})

print(f"Extracted text: {text_result.text}")
```

## Experience Buffer

```python
from core.modules.experience_buffer_enhanced import ExperienceBuffer
from core.data_structures import Experience

# Initialize buffer
buffer = ExperienceBuffer(
    max_size=100000,
    embedding_dim=768
)

# Add experience
experience = Experience(
    query="What color is the car?",
    trajectory=[...],  # List of actions
    answer="Red",
    confidence=0.85,
    embeddings=embeddings_tensor
)
buffer.add(experience)

# Search for similar experiences
neighbors = buffer.search_knn(
    query_embedding=query_emb,
    k=5
)

for neighbor in neighbors:
    print(f"Similar query: {neighbor.query}")
    print(f"Similarity: {neighbor.similarity}")
```

## Reward Calculation

```python
from core.modules.reward_shaping_enhanced import RewardOrchestrator
from core.data_structures import Trajectory

# Initialize orchestrator
orchestrator = RewardOrchestrator(
    task_weight=1.0,
    curiosity_weight=0.2,
    coherence_weight=0.3
)

# Calculate reward
trajectory = Trajectory(actions=[...])
reward_info = orchestrator.calculate_reward(
    trajectory=trajectory,
    state=current_state,
    is_correct=True
)

print(f"Total reward: {reward_info.total_reward}")
print(f"Task reward: {reward_info.task_reward}")
print(f"Curiosity reward: {reward_info.curiosity_reward}")
print(f"Coherence reward: {reward_info.coherence_reward}")
```

## Online Learning

```python
from core.engine.update_worker import UpdateWorker
from core.data_structures import UpdateTask

# Initialize update worker
worker = UpdateWorker(
    model_path="outputs/models/pixelis-online",
    learning_rate=1e-5,
    kl_weight=0.1
)

# Create update task
task = UpdateTask(
    experience=experience,
    reward=reward_tensor,
    learning_rate=1e-5
)

# Process update
worker.process_update(task)

# Get updated model
updated_model = worker.get_model()
```

## Training Script

```python
from scripts.train import main
from hydra import compose, initialize_config_dir

# Initialize Hydra
with initialize_config_dir(config_dir="configs"):
    cfg = compose(config_name="train_config")
    
    # Override parameters
    cfg.mode = "sft"
    cfg.training.batch_size = 4
    cfg.training.learning_rate = 5e-5
    
    # Run training
    main(cfg)
```

## Model Evaluation

```python
from scripts.evaluate import evaluate_model
import json

# Run evaluation
results = evaluate_model(
    model_path="outputs/models/pixelis-online",
    benchmark="mm-vet",
    output_dir="results/"
)

# Print results
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1_score']:.3f}")

# Save detailed results
with open("results/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Custom Visual Operation

```python
from core.modules.operations.base_operation import BaseOperation
from core.modules.operation_registry import VisualOperationRegistry

class CustomOperation(BaseOperation):
    \"\"\"Custom visual operation example.\"\"\"
    
    def __init__(self):
        super().__init__(
            name="CUSTOM_OP",
            description="Custom operation description"
        )
    
    def execute(self, image, params):
        # Implement custom logic
        result = process_image(image, params)
        return result
    
    def validate_params(self, params):
        required = ["param1", "param2"]
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")

# Register custom operation
registry = VisualOperationRegistry()
registry.register("CUSTOM_OP", CustomOperation())
```

## Configuration Management

```python
from hydra import initialize, compose
from omegaconf import OmegaConf

# Load configuration
with initialize(config_path="../configs"):
    cfg = compose(config_name="main")
    
    # Access nested configuration
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    
    # Override configuration
    cfg.training.batch_size = 8
    
    # Convert to dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)
```

## Monitoring and Logging

```python
import wandb
from core.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger("pixelis.training")

# Initialize WandB
wandb.init(
    project="pixelis",
    config=config_dict,
    tags=["training", "sft"]
)

# Log metrics
wandb.log({
    "loss": loss.item(),
    "accuracy": accuracy,
    "learning_rate": optimizer.param_groups[0]['lr']
})

# Log artifacts
wandb.log_artifact(
    "model_checkpoint.pt",
    type="model",
    name="pixelis-sft-checkpoint"
)

logger.info(f"Training completed. Final accuracy: {accuracy:.2%}")
```
"""
    
    examples_file.parent.mkdir(parents=True, exist_ok=True)
    examples_file.write_text(content)
    print(f"Generated {examples_file}")


def main():
    """Main entry point for API documentation generation."""
    parser = argparse.ArgumentParser(description="Generate API documentation for Pixelis")
    parser.add_argument(
        "--output-dir",
        default="docs/api",
        help="Output directory for documentation"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["core"],
        help="Modules to document"
    )
    args = parser.parse_args()
    
    # Initialize generator
    generator = APIDocGenerator(output_dir=args.output_dir)
    
    # Scan specified modules
    for module in args.modules:
        module_path = project_root / module
        if module_path.exists():
            print(f"\nScanning {module}...")
            generator.scan_directory(module_path)
        else:
            print(f"Warning: Module path {module_path} does not exist")
    
    # Generate documentation
    print("\nGenerating documentation...")
    generator.generate_markdown()
    
    # Generate usage examples
    print("\nGenerating usage examples...")
    generate_usage_examples()
    
    print(f"\nAPI documentation generated in {args.output_dir}/")
    print("View the main index at: docs/api/index.md")


if __name__ == "__main__":
    main()