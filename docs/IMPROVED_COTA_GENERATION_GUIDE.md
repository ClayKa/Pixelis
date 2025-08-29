# Improved CoTA Data Generation System - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Improvements](#core-improvements)
4. [Component Details](#component-details)
5. [Usage Guide](#usage-guide)
6. [Testing](#testing)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The improved CoTA (Chain-of-Thought-Action) data generation system is a comprehensive framework for generating high-quality training data for the Pixelis vision-language model. The system has been redesigned following **SOLID principles** and **KISS philosophy** to ensure maintainability, scalability, and reliability.

### Key Features
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **SOLID Principles**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **Comprehensive Testing**: Full test coverage with unit, integration, and performance tests
- **Flexible Templates**: Dynamic prompt template system with validation
- **Quality Assurance**: Multi-dimensional validation and scoring pipeline
- **Production Ready**: Error handling, logging, checkpointing, and monitoring

## Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                     CoTA Generation Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Data Loaders │───▶│  Generators  │───▶│  Validators  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                    │                    │          │
│         │                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Datasets   │    │   Templates  │    │   Scorers    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy
```
core/data_generation/
├── specialized_generator.py      # Main generator framework (SOLID)
├── data_loader_interface.py      # Unified data loader abstraction
├── prompt_templates.py            # Flexible template system
├── validation_and_scoring.py     # Quality assurance pipeline
└── trajectory_augmenter.py       # Data augmentation (in specialized_generator.py)
```

## Core Improvements

### 1. SOLID Principles Implementation

#### Single Responsibility Principle (SRP)
Each class has a single, well-defined responsibility:
- `SpecializedTaskGenerator`: Generates trajectories
- `DataLoaderInterface`: Loads data samples
- `PromptTemplate`: Manages prompt formatting
- `ValidationPipeline`: Validates trajectories
- `QualityScorer`: Scores trajectory quality

#### Open/Closed Principle (OCP)
The system is open for extension but closed for modification:
```python
# Easy to add new generators without modifying existing code
class NewTaskGenerator(SpecializedTaskGenerator):
    def generate_trajectory(self, sample_data, difficulty):
        # Custom implementation
        pass

# Register with factory
TaskGeneratorFactory.register("new_task", NewTaskGenerator)
```

#### Liskov Substitution Principle (LSP)
All derived classes can be substituted for their base classes:
```python
# Any DataLoaderInterface implementation works seamlessly
loader = DataLoaderFactory.create("coco", ...)  # COCODataLoader
loader = DataLoaderFactory.create("video", ...)  # VideoDataLoader
# Both work with the same interface
```

#### Interface Segregation Principle (ISP)
Interfaces are specific and focused:
```python
# Separate interfaces for different capabilities
class FilterableDataLoader(DataLoaderInterface):
    def filter_by_criteria(self, criteria): pass

class StreamableDataLoader(DataLoaderInterface):
    def stream_samples(self, batch_size): pass
```

#### Dependency Inversion Principle (DIP)
High-level modules depend on abstractions, not concretions:
```python
# Generator depends on interfaces, not specific implementations
def __init__(self, data_loaders: Dict[str, DataLoaderInterface], 
             llm_client: LLMClient):
    # Works with any implementation of the interfaces
```

### 2. KISS Philosophy

The system follows "Keep It Simple, Stupid" principles:
- **Clear abstractions**: Simple, intuitive interfaces
- **Minimal complexity**: Avoid over-engineering
- **Readable code**: Self-documenting with clear naming
- **Focused components**: Each module does one thing well

### 3. Key Improvements Over Original System

| Aspect | Original | Improved |
|--------|----------|----------|
| **Architecture** | Monolithic generators | Modular, pluggable components |
| **Data Loading** | Direct file access | Abstracted loader interface |
| **Templates** | Hardcoded strings | Dynamic template system |
| **Validation** | Basic checks | Multi-dimensional pipeline |
| **Testing** | Limited coverage | Comprehensive test suite |
| **Error Handling** | Basic try-catch | Graceful degradation & recovery |
| **Configuration** | Manual paths | Automated path discovery |

## Component Details

### 1. Specialized Generator Framework

**File**: `core/data_generation/specialized_generator.py`

**Key Classes**:
- `SpecializedTaskGenerator`: Abstract base for all generators
- `CoTATrajectory`: Data structure for trajectories
- `TrajectoryAugmenter`: Creates trap and correction samples
- `TaskGeneratorFactory`: Factory pattern for generator creation

**Features**:
- Automatic checkpointing every N samples
- Statistics tracking and reporting
- Quality scoring integration
- Configurable augmentation

**Usage Example**:
```python
from core.data_generation.specialized_generator import TaskGeneratorFactory

# Create generator
generator = TaskGeneratorFactory.create(
    task_type="geometric_reasoning",
    data_loaders={"coco": coco_loader},
    llm_client=llm_client,
    quality_scorer=scorer,
    output_dir=Path("outputs")
)

# Generate dataset
trajectories = generator.generate_dataset(
    target_count=1000,
    difficulty_distribution={
        DifficultyLevel.EASY: 0.3,
        DifficultyLevel.MEDIUM: 0.5,
        DifficultyLevel.HARD: 0.2
    },
    augmentation_config={
        "trap_probability": 0.2,
        "correction_probability": 0.2
    }
)
```

### 2. Data Loader Abstraction

**File**: `core/data_generation/data_loader_interface.py`

**Key Classes**:
- `DataLoaderInterface`: Protocol defining loader interface
- `BaseDataLoader`: Base implementation with caching
- `COCODataLoader`: COCO format loader with filtering
- `VideoDataLoader`: Video dataset loader with streaming
- `DocumentDataLoader`: Document understanding loader

**Features**:
- Unified interface for all dataset types
- Built-in caching with LRU eviction
- PyTorch Dataset integration
- Filtering and property-based queries

**Usage Example**:
```python
from core.data_generation.data_loader_interface import DataLoaderFactory

# Create loader from configuration
loader = DataLoaderFactory.create_from_config({
    'type': 'coco',
    'images_path': 'datasets/coco/images',
    'annotations_path': 'datasets/coco/annotations.json',
    'cache_size': 100
})

# Load sample
sample = loader.load_sample(0)

# Filter samples
indices = loader.filter_by_criteria({
    'min_objects': 3,
    'required_categories': ['person', 'car']
})
```

### 3. Prompt Template System

**File**: `core/data_generation/prompt_templates.py`

**Key Classes**:
- `PromptTemplate`: Abstract base for templates
- `ChainOfThoughtTemplate`: CoT reasoning templates
- `VisualOperationTemplate`: Operation-specific templates
- `TemplateManager`: Template loading and management
- `TemplateBuilder`: Fluent interface for custom templates

**Features**:
- Variable validation with regex patterns
- Conditional sections
- Default value support
- YAML-based configuration

**Usage Example**:
```python
from core.data_generation.prompt_templates import (
    ChainOfThoughtTemplate,
    TemplateBuilder,
    TemplateManager
)

# Use built-in template
template = ChainOfThoughtTemplate()
prompt = template.build(
    task_description="Identify objects",
    difficulty="medium",
    num_steps=5
)

# Create custom template
custom = (TemplateBuilder()
    .with_type(TemplateType.TASK_DESCRIPTION)
    .add_variable("task", "Task to perform")
    .add_section("header", "Perform: {task}")
    .build()
)
```

### 4. Validation and Quality Scoring

**File**: `core/data_generation/validation_and_scoring.py`

**Key Classes**:
- `ValidationPipeline`: Complete validation workflow
- `StructuralValidator`: Checks trajectory structure
- `LogicalValidator`: Validates logical consistency
- `DuplicateDetector`: Identifies duplicate trajectories
- `ComprehensiveQualityScorer`: Multi-dimensional scoring

**Quality Dimensions**:
- **Completeness**: All required fields present
- **Coherence**: Logical flow of reasoning
- **Correctness**: Answer quality indicators
- **Efficiency**: Optimal operation usage
- **Complexity**: Appropriate for difficulty
- **Consistency**: Internal consistency

**Usage Example**:
```python
from core.data_generation.validation_and_scoring import ValidationPipeline

# Create pipeline
pipeline = ValidationPipeline(
    level=ValidationLevel.STANDARD
)

# Validate trajectory
result = pipeline.validate_trajectory(trajectory_dict)

if result.is_valid:
    print(f"Quality score: {result.overall_score:.2f}")
else:
    for issue in result.issues:
        print(f"Error: {issue.message}")

# Generate report
report = pipeline.generate_report(Path("validation_report.txt"))
```

## Usage Guide

### 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Update configuration paths
python scripts/update_data_generation_config.py \
    --datasets-dir /path/to/datasets \
    --manifest configs/data_generation_manifest.yaml \
    --report dataset_report.txt
```

### 2. Generate CoTA Data

```bash
# Stage 1: Generate specialized datasets
python scripts/1_generate_specialized_datasets.py \
    --manifest configs/data_generation_manifest.yaml \
    --output-dir data_outputs/specialized \
    --verbose

# Stage 2: Fuse and validate
python scripts/2_fuse_and_validate_dataset.py \
    --fusion-manifest configs/data_fusion_manifest.yaml \
    --input-dir data_outputs/specialized \
    --output-dir data_outputs/final \
    --verbose
```

### 3. Custom Generator Implementation

```python
from core.data_generation.specialized_generator import SpecializedTaskGenerator

class CustomTaskGenerator(SpecializedTaskGenerator):
    def get_prompt_template(self, difficulty):
        # Return appropriate template
        return self.template_manager.get_template("custom_template")
    
    def generate_trajectory(self, sample_data, difficulty):
        # Generate trajectory logic
        prompt = self.get_prompt_template(difficulty).build(**sample_data)
        response = self.llm_client.generate(prompt)
        
        # Parse response into trajectory
        return self._parse_response(response, sample_data, difficulty)

# Register generator
TaskGeneratorFactory.register("custom_task", CustomTaskGenerator)
```

## Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/test_cota_generation_improved.py -v

# Run specific test class
pytest tests/test_cota_generation_improved.py::TestSpecializedGenerator -v

# Run with coverage
pytest tests/test_cota_generation_improved.py --cov=core.data_generation --cov-report=html
```

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Verify efficiency
4. **Error Handling Tests**: Test error recovery

## Best Practices

### 1. Generator Development
- Inherit from `SpecializedTaskGenerator`
- Implement required abstract methods
- Use quality scorer for validation
- Enable checkpointing for large datasets

### 2. Data Loader Implementation
- Inherit from `BaseDataLoader` for caching
- Implement filtering for large datasets
- Use streaming for memory efficiency
- Validate data on loading

### 3. Template Design
- Keep templates focused and reusable
- Use variables for dynamic content
- Add validation patterns for inputs
- Document required variables

### 4. Validation Strategy
- Start with lenient validation during development
- Move to standard/strict for production
- Monitor quality score distribution
- Set minimum quality thresholds

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Datasets
**Problem**: Datasets not found at expected paths
**Solution**: 
```bash
# Check dataset availability
python scripts/update_data_generation_config.py --report

# Use mock data for testing
python scripts/1_generate_specialized_datasets.py --use-mock
```

#### 2. Low Quality Scores
**Problem**: Generated trajectories have low quality scores
**Solution**:
- Review prompt templates
- Adjust difficulty distribution
- Increase temperature for diversity
- Check LLM response quality

#### 3. Memory Issues
**Problem**: Out of memory with large datasets
**Solution**:
- Reduce cache size in data loaders
- Use streaming for video datasets
- Enable checkpointing more frequently
- Process in smaller batches

#### 4. Validation Failures
**Problem**: Many trajectories failing validation
**Solution**:
- Start with `ValidationLevel.LENIENT`
- Review validation rules
- Check trajectory parsing logic
- Inspect failed trajectories for patterns

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via command line
python script.py --verbose --debug
```

### Performance Optimization

1. **Caching**: Adjust cache sizes based on available memory
2. **Parallel Processing**: Use multiprocessing for batch operations
3. **Checkpointing**: Save progress frequently for large datasets
4. **Profiling**: Use cProfile to identify bottlenecks

## Summary

The improved CoTA data generation system provides a robust, maintainable, and scalable solution for generating high-quality training data. By following SOLID principles and KISS philosophy, the system is:

- **Extensible**: Easy to add new generators and loaders
- **Maintainable**: Clean code with clear responsibilities
- **Reliable**: Comprehensive validation and error handling
- **Efficient**: Optimized with caching and streaming
- **Testable**: Full test coverage with mocking support

The modular architecture ensures that each component can be developed, tested, and deployed independently, while the unified interfaces guarantee seamless integration across the entire pipeline.