# Pixelis API Usage Examples

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
    """Custom visual operation example."""
    
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
