"""
Advanced Prompt Template System for CoTA Data Generation
Implements a flexible, maintainable prompt template system following KISS principle
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import yaml
import re

logger = logging.getLogger(__name__)


# ======================== TEMPLATE TYPES ========================

class TemplateType(Enum):
    """Types of prompt templates"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    VISUAL_OPERATION = "visual_operation"
    ERROR_CORRECTION = "error_correction"
    QUALITY_ASSESSMENT = "quality_assessment"
    TASK_DESCRIPTION = "task_description"


class OperationType(Enum):
    """Visual operation types"""
    ZOOM_IN = "ZOOM_IN"
    SELECT_FRAME = "SELECT_FRAME"
    SEGMENT_OBJECT_AT = "SEGMENT_OBJECT_AT"
    GET_PROPERTIES = "GET_PROPERTIES"
    READ_TEXT = "READ_TEXT"
    TRACK_OBJECT = "TRACK_OBJECT"


# ======================== TEMPLATE COMPONENTS ========================

@dataclass
class TemplateVariable:
    """Represents a variable in a template"""
    name: str
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation_regex: Optional[str] = None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value for this variable"""
        if value is None:
            if self.required and self.default_value is None:
                return False, f"Required variable '{self.name}' is missing"
            return True, None
        
        if self.validation_regex:
            if not re.match(self.validation_regex, str(value)):
                return False, f"Value '{value}' doesn't match pattern '{self.validation_regex}'"
        
        return True, None


@dataclass
class TemplateSection:
    """Represents a section of a template"""
    name: str
    content: str
    optional: bool = False
    condition: Optional[str] = None  # Simple condition expression
    
    def should_include(self, context: Dict[str, Any]) -> bool:
        """Check if this section should be included"""
        if not self.optional:
            return True
        
        if self.condition:
            # Simple evaluation (in production, use safe eval)
            try:
                # Very basic condition evaluation
                if "==" in self.condition:
                    var, val = self.condition.split("==")
                    var = var.strip()
                    val = val.strip().strip("'\"")
                    return str(context.get(var)) == val
                elif "!=" in self.condition:
                    var, val = self.condition.split("!=")
                    var = var.strip()
                    val = val.strip().strip("'\"")
                    return str(context.get(var)) != val
                elif ">" in self.condition:
                    var, val = self.condition.split(">")
                    return float(context.get(var.strip(), 0)) > float(val.strip())
                elif "<" in self.condition:
                    var, val = self.condition.split("<")
                    return float(context.get(var.strip(), 0)) < float(val.strip())
            except Exception as e:
                logger.warning(f"Failed to evaluate condition '{self.condition}': {e}")
                return False
        
        return True


# ======================== BASE TEMPLATE CLASS ========================

class PromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    def __init__(self, template_type: TemplateType):
        self.template_type = template_type
        self.variables: List[TemplateVariable] = []
        self.sections: List[TemplateSection] = []
    
    @abstractmethod
    def build(self, **kwargs) -> str:
        """Build the prompt from template"""
        pass
    
    def validate_inputs(self, **kwargs) -> Tuple[bool, List[str]]:
        """Validate all inputs against template requirements"""
        errors = []
        
        for var in self.variables:
            value = kwargs.get(var.name, var.default_value)
            is_valid, error = var.validate(value)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def get_required_variables(self) -> List[str]:
        """Get list of required variable names"""
        return [var.name for var in self.variables if var.required]
    
    def format_with_defaults(self, **kwargs) -> Dict[str, Any]:
        """Fill in missing values with defaults"""
        result = kwargs.copy()
        for var in self.variables:
            if var.name not in result and var.default_value is not None:
                result[var.name] = var.default_value
        return result


# ======================== CONCRETE TEMPLATE IMPLEMENTATIONS ========================

class ChainOfThoughtTemplate(PromptTemplate):
    """Template for generating chain-of-thought reasoning"""
    
    def __init__(self):
        super().__init__(TemplateType.CHAIN_OF_THOUGHT)
        
        # Define variables
        self.variables = [
            TemplateVariable("task_description", "The task to solve"),
            TemplateVariable("difficulty", "Task difficulty level"),
            TemplateVariable("constraints", "Any constraints or limitations", required=False),
            TemplateVariable("examples", "Example solutions", required=False),
            TemplateVariable("num_steps", "Expected number of reasoning steps", required=False, default_value=5)
        ]
        
        # Define sections
        self.sections = [
            TemplateSection("header", "You are an expert visual reasoning AI."),
            TemplateSection("task", "Task: {task_description}"),
            TemplateSection("difficulty", "Difficulty Level: {difficulty}"),
            TemplateSection(
                "constraints",
                "Constraints: {constraints}",
                optional=True,
                condition="constraints != None"
            ),
            TemplateSection(
                "examples",
                "Examples:\n{examples}",
                optional=True,
                condition="examples != None"
            ),
            TemplateSection(
                "instructions",
                """
Generate a detailed chain-of-thought reasoning process with exactly {num_steps} steps.
Each step should:
1. Clearly state what you're analyzing or considering
2. Explain your reasoning
3. Connect to the next step logically

Format each step as:
Step N: [Thought]
Reasoning: [Detailed explanation]
"""
            ),
            TemplateSection("output_format", "Provide your final answer after all reasoning steps.")
        ]
    
    def build(self, **kwargs) -> str:
        """Build the chain-of-thought prompt"""
        # Validate inputs
        is_valid, errors = self.validate_inputs(**kwargs)
        if not is_valid:
            raise ValueError(f"Invalid inputs: {errors}")
        
        # Fill defaults
        context = self.format_with_defaults(**kwargs)
        
        # Build prompt from sections
        prompt_parts = []
        for section in self.sections:
            if section.should_include(context):
                try:
                    content = section.content.format(**context)
                    prompt_parts.append(content)
                except KeyError as e:
                    logger.warning(f"Missing variable in section '{section.name}': {e}")
        
        return "\n\n".join(prompt_parts)


class VisualOperationTemplate(PromptTemplate):
    """Template for generating visual operation sequences"""
    
    def __init__(self, operation_type: OperationType):
        super().__init__(TemplateType.VISUAL_OPERATION)
        self.operation_type = operation_type
        
        # Define operation-specific variables
        self._define_operation_variables()
        
        # Common sections
        self.sections = [
            TemplateSection(
                "context",
                "Generate a sequence of {operation_type} operations to solve the following task:"
            ),
            TemplateSection("task", "Task: {task_description}"),
            TemplateSection("image_info", "Image Information:\n{image_metadata}"),
            TemplateSection(
                "operation_format",
                self._get_operation_format()
            ),
            TemplateSection(
                "requirements",
                "Requirements:\n- Use precise coordinates\n- Validate each operation\n- Handle edge cases"
            )
        ]
    
    def _define_operation_variables(self):
        """Define variables based on operation type"""
        base_vars = [
            TemplateVariable("task_description", "Task to solve"),
            TemplateVariable("image_metadata", "Image metadata"),
            TemplateVariable("operation_type", "Type of operation", default_value=self.operation_type.value)
        ]
        
        # Add operation-specific variables
        if self.operation_type == OperationType.ZOOM_IN:
            base_vars.extend([
                TemplateVariable("zoom_level", "Zoom level", required=False, default_value=2.0),
                TemplateVariable("target_region", "Target region description", required=False)
            ])
        elif self.operation_type == OperationType.SEGMENT_OBJECT_AT:
            base_vars.extend([
                TemplateVariable("object_description", "Object to segment"),
                TemplateVariable("coordinates", "Initial coordinates", required=False)
            ])
        elif self.operation_type == OperationType.TRACK_OBJECT:
            base_vars.extend([
                TemplateVariable("object_id", "Object to track"),
                TemplateVariable("start_frame", "Starting frame", default_value=0),
                TemplateVariable("end_frame", "Ending frame", required=False)
            ])
        
        self.variables = base_vars
    
    def _get_operation_format(self) -> str:
        """Get operation-specific format instructions"""
        formats = {
            OperationType.ZOOM_IN: """
Operation Format:
ZOOM_IN(x_center, y_center, zoom_factor)
- x_center, y_center: Center coordinates of zoom region
- zoom_factor: Magnification level (1.5 to 4.0)
""",
            OperationType.SEGMENT_OBJECT_AT: """
Operation Format:
SEGMENT_OBJECT_AT(x, y)
- x, y: Coordinates pointing to the object
Returns: Binary mask of the segmented object
""",
            OperationType.GET_PROPERTIES: """
Operation Format:
GET_PROPERTIES(mask)
- mask: Binary mask from segmentation
Returns: Object properties (area, perimeter, centroid, etc.)
""",
            OperationType.READ_TEXT: """
Operation Format:
READ_TEXT(x1, y1, x2, y2)
- x1, y1, x2, y2: Bounding box coordinates
Returns: Extracted text content
""",
            OperationType.SELECT_FRAME: """
Operation Format:
SELECT_FRAME(frame_number)
- frame_number: Index of frame to select
Returns: Selected frame image
""",
            OperationType.TRACK_OBJECT: """
Operation Format:
TRACK_OBJECT(object_id, start_frame, end_frame)
- object_id: Identifier of object to track
- start_frame, end_frame: Frame range
Returns: Object trajectory across frames
"""
        }
        return formats.get(self.operation_type, "Operation format not defined")
    
    def build(self, **kwargs) -> str:
        """Build the visual operation prompt"""
        is_valid, errors = self.validate_inputs(**kwargs)
        if not is_valid:
            raise ValueError(f"Invalid inputs: {errors}")
        
        context = self.format_with_defaults(**kwargs)
        
        prompt_parts = []
        for section in self.sections:
            if section.should_include(context):
                content = section.content.format(**context)
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)


class ErrorCorrectionTemplate(PromptTemplate):
    """Template for generating error correction trajectories"""
    
    def __init__(self):
        super().__init__(TemplateType.ERROR_CORRECTION)
        
        self.variables = [
            TemplateVariable("original_trajectory", "The original trajectory with error"),
            TemplateVariable("error_description", "Description of the error"),
            TemplateVariable("error_location", "Where the error occurs", required=False),
            TemplateVariable("correction_strategy", "How to correct", required=False)
        ]
        
        self.sections = [
            TemplateSection(
                "header",
                "Analyze the following trajectory and generate a corrected version:"
            ),
            TemplateSection(
                "original",
                "Original Trajectory:\n{original_trajectory}"
            ),
            TemplateSection(
                "error",
                "Identified Error: {error_description}"
            ),
            TemplateSection(
                "location",
                "Error Location: {error_location}",
                optional=True,
                condition="error_location != None"
            ),
            TemplateSection(
                "instructions",
                """
Generate a self-correcting trajectory that:
1. Proceeds normally until the error point
2. Recognizes the error with a reflection thought
3. Generates corrective reasoning
4. Continues with the corrected approach
5. Validates the final result

Include thoughts like:
- "Wait, this doesn't seem right..."
- "Let me reconsider this step..."
- "I need to correct my approach..."
"""
            )
        ]
    
    def build(self, **kwargs) -> str:
        """Build error correction prompt"""
        is_valid, errors = self.validate_inputs(**kwargs)
        if not is_valid:
            raise ValueError(f"Invalid inputs: {errors}")
        
        context = self.format_with_defaults(**kwargs)
        
        prompt_parts = []
        for section in self.sections:
            if section.should_include(context):
                content = section.content.format(**context)
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)


# ======================== TEMPLATE MANAGER ========================

class TemplateManager:
    """Manages and loads prompt templates"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path("configs/prompt_templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache loaded templates
        self._template_cache = {}
        
        # Register built-in templates
        self._register_builtin_templates()
    
    def _register_builtin_templates(self):
        """Register built-in template classes"""
        self._template_cache["chain_of_thought"] = ChainOfThoughtTemplate()
        self._template_cache["error_correction"] = ErrorCorrectionTemplate()
        
        # Register operation templates
        for op_type in OperationType:
            key = f"operation_{op_type.value.lower()}"
            self._template_cache[key] = VisualOperationTemplate(op_type)
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a template by name"""
        # Check cache
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Try to load from file
        template = self._load_from_file(template_name)
        if template:
            self._template_cache[template_name] = template
            return template
        
        raise ValueError(f"Template '{template_name}' not found")
    
    def _load_from_file(self, template_name: str) -> Optional[PromptTemplate]:
        """Load template from YAML file"""
        file_path = self.templates_dir / f"{template_name}.yaml"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return self._create_from_config(config)
        except Exception as e:
            logger.error(f"Failed to load template from {file_path}: {e}")
            return None
    
    def _create_from_config(self, config: Dict[str, Any]) -> PromptTemplate:
        """Create template from configuration"""
        # This would create a custom template from YAML config
        # For now, return a simple template
        class ConfiguredTemplate(PromptTemplate):
            def __init__(self, cfg):
                super().__init__(TemplateType.CHAIN_OF_THOUGHT)
                self.config = cfg
                
                # Parse variables
                for var_cfg in cfg.get('variables', []):
                    self.variables.append(TemplateVariable(**var_cfg))
                
                # Parse sections
                for sec_cfg in cfg.get('sections', []):
                    self.sections.append(TemplateSection(**sec_cfg))
            
            def build(self, **kwargs) -> str:
                is_valid, errors = self.validate_inputs(**kwargs)
                if not is_valid:
                    raise ValueError(f"Invalid inputs: {errors}")
                
                context = self.format_with_defaults(**kwargs)
                
                # Use template string from config
                template_str = self.config.get('template', '')
                return template_str.format(**context)
        
        return ConfiguredTemplate(config)
    
    def save_template(self, name: str, template: PromptTemplate):
        """Save template to file"""
        file_path = self.templates_dir / f"{name}.yaml"
        
        config = {
            'type': template.template_type.value,
            'variables': [
                {
                    'name': var.name,
                    'description': var.description,
                    'required': var.required,
                    'default_value': var.default_value,
                    'validation_regex': var.validation_regex
                }
                for var in template.variables
            ],
            'sections': [
                {
                    'name': sec.name,
                    'content': sec.content,
                    'optional': sec.optional,
                    'condition': sec.condition
                }
                for sec in template.sections
            ]
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Template saved to {file_path}")


# ======================== TEMPLATE BUILDERS ========================

class TemplateBuilder:
    """Fluent interface for building templates"""
    
    def __init__(self):
        self.template_type = TemplateType.CHAIN_OF_THOUGHT
        self.variables = []
        self.sections = []
    
    def with_type(self, template_type: TemplateType) -> 'TemplateBuilder':
        """Set template type"""
        self.template_type = template_type
        return self
    
    def add_variable(
        self,
        name: str,
        description: str,
        required: bool = True,
        default_value: Any = None,
        validation_regex: str = None
    ) -> 'TemplateBuilder':
        """Add a variable"""
        self.variables.append(TemplateVariable(
            name, description, required, default_value, validation_regex
        ))
        return self
    
    def add_section(
        self,
        name: str,
        content: str,
        optional: bool = False,
        condition: str = None
    ) -> 'TemplateBuilder':
        """Add a section"""
        self.sections.append(TemplateSection(
            name, content, optional, condition
        ))
        return self
    
    def build(self) -> PromptTemplate:
        """Build the template"""
        class CustomTemplate(PromptTemplate):
            def __init__(self, builder):
                super().__init__(builder.template_type)
                self.variables = builder.variables
                self.sections = builder.sections
            
            def build(self, **kwargs) -> str:
                is_valid, errors = self.validate_inputs(**kwargs)
                if not is_valid:
                    raise ValueError(f"Invalid inputs: {errors}")
                
                context = self.format_with_defaults(**kwargs)
                
                prompt_parts = []
                for section in self.sections:
                    if section.should_include(context):
                        content = section.content.format(**context)
                        prompt_parts.append(content)
                
                return "\n\n".join(prompt_parts)
        
        return CustomTemplate(self)


# ======================== USAGE EXAMPLES ========================

def create_example_templates():
    """Create example templates for different tasks"""
    
    # Example 1: Chain of Thought template
    cot_template = ChainOfThoughtTemplate()
    prompt = cot_template.build(
        task_description="Compare the sizes of objects in the image",
        difficulty="medium",
        constraints="Objects may be partially occluded",
        num_steps=4
    )
    print("Chain of Thought Prompt:")
    print(prompt)
    print("=" * 50)
    
    # Example 2: Visual Operation template
    seg_template = VisualOperationTemplate(OperationType.SEGMENT_OBJECT_AT)
    prompt = seg_template.build(
        task_description="Segment the cat in the image",
        image_metadata="Size: 640x480, Format: RGB",
        object_description="orange tabby cat"
    )
    print("Segmentation Operation Prompt:")
    print(prompt)
    print("=" * 50)
    
    # Example 3: Error Correction template
    error_template = ErrorCorrectionTemplate()
    prompt = error_template.build(
        original_trajectory="Step 1: Look at image\nStep 2: Segment at (100, 100)\nStep 3: Get answer",
        error_description="Incorrect coordinates for segmentation",
        error_location="Step 2"
    )
    print("Error Correction Prompt:")
    print(prompt)
    print("=" * 50)
    
    # Example 4: Custom template using builder
    custom_template = (TemplateBuilder()
        .with_type(TemplateType.TASK_DESCRIPTION)
        .add_variable("task_name", "Name of the task")
        .add_variable("num_examples", "Number of examples", default_value=3)
        .add_section("header", "Generate {num_examples} examples for: {task_name}")
        .add_section("format", "Use clear, detailed descriptions")
        .build()
    )
    
    prompt = custom_template.build(task_name="Object Counting")
    print("Custom Template Prompt:")
    print(prompt)
    print("=" * 50)
    
    # Example 5: Template Manager
    manager = TemplateManager()
    
    # Get built-in template
    template = manager.get_template("chain_of_thought")
    
    # Save custom template
    manager.save_template("custom_task", custom_template)
    
    print("Available templates:", list(manager._template_cache.keys()))


if __name__ == "__main__":
    print("Prompt Template System - Examples")
    print("=" * 50)
    create_example_templates()