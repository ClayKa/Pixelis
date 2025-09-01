"""
Geometric Comparison Task Generator for CoTA data synthesis.

This module generates Chain-of-Thought-Action trajectories for geometric reasoning
and property comparison tasks using SEGMENT_OBJECT_AT and GET_PROPERTIES operations.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)


class GeometricComparisonTaskGenerator(BaseTaskGenerator):
    """
    Generates CoTA samples for geometric reasoning and object property comparison.
    
    This generator creates tasks that require:
    - Segmenting objects at specific pixel coordinates
    - Extracting geometric properties (size, shape, orientation)
    - Comparing properties between multiple objects
    - Spatial relationship reasoning
    """
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initialize the geometric comparison task generator."""
        super().__init__(loaders, config, global_config)
        
        # Define difficulty-specific loaders mapping (from manifest)
        self.difficulty_loaders = {
            'easy': ['coco2017_train'],
            'medium': ['lvis_v1_train', 'sa1b_for_segmentation'],
            'hard': ['part_imagenet_train']
        }
        
        # Property types for comparison
        self.property_types = [
            'size', 'shape', 'orientation', 'texture', 
            'relative_position', 'aspect_ratio', 'symmetry'
        ]
        
        # Comparison operators
        self.comparison_ops = [
            'larger', 'smaller', 'equal', 'similar',
            'above', 'below', 'left_of', 'right_of',
            'more_complex', 'simpler', 'rounder', 'more_elongated'
        ]
        
        # [NEW] Store the styles in a structured way
        self.styles = [
            {
                'name': 'The Mathematician',
                'desc': 'Uses precise mathematical language and comparisons',
                'q': 'Calculate which object has greater area.',
                'a': 'Object A measures 15,234 pixels² while Object B measures 12,891 pixels². A > B.'
            },
            {
                'name': 'The Spatial Analyst',
                'desc': 'Focuses on spatial relationships and topology',
                'q': 'How are these objects spatially related?',
                'a': 'Object A is positioned 45° northeast of Object B with 23 pixels separation.'
            },
            {
                'name': 'The Comparative Judge',
                'desc': 'Makes judgments like a competition judge',
                'q': 'Which object wins in size?',
                'a': 'The winner is clearly the left object, dominating with 40% more area.'
            },
            {
                'name': 'The Geometry Teacher',
                'desc': 'Explains comparisons as educational lessons',
                'q': 'What geometric principles can we observe here?',
                'a': 'This demonstrates that circular objects have optimal area-to-perimeter ratios.'
            },
            {
                'name': 'The Architect',
                'desc': 'Analyzes structures and proportions professionally',
                'q': 'Analyze the structural proportions.',
                'a': 'The golden ratio is evident: Object A\'s dimensions are 1.618 times Object B\'s.'
            },
            {
                'name': 'The Minimalist Measurer',
                'desc': 'Provides only essential measurements',
                'q': 'Size comparison?',
                'a': 'A: 150x200. B: 100x180. A larger.'
            },
            {
                'name': 'The Visual Designer',
                'desc': 'Discusses visual balance and composition',
                'q': 'How do these elements balance visually?',
                'a': 'The larger element creates visual weight, offset by the smaller\'s position.'
            },
            {
                'name': 'The Data Scientist',
                'desc': 'Provides statistical analysis of properties',
                'q': 'What\'s the statistical relationship?',
                'a': 'Mean area difference: 25.3%, standard deviation: 4.2, confidence: 95%.'
            },
            {
                'name': 'The Inspector',
                'desc': 'Examines details methodically like quality control',
                'q': 'Does Object A meet the size criteria?',
                'a': 'Inspection complete: Object A exceeds minimum threshold by 15%.'
            },
            {
                'name': 'The Storyteller',
                'desc': 'Narrates the comparison as a story',
                'q': 'Tell me the tale of these two shapes.',
                'a': 'Once there were two shapes, and the larger one cast a shadow over its companion.'
            }
        ]
    
    def _build_context_placeholders(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build context placeholders for the geometric reasoning prompt.
        Returns tuple of (placeholders_dict, metadata_dict) as required by base class.
        """
        logger.debug(f"Building context for geometric comparison task")
        
        # Get a loader
        loader_name = None
        loader = None
        for name in ['coco2017_train', 'lvis_v1_train', 'sa1b_for_segmentation', 'part_imagenet_train']:
            if name in self.loaders:
                loader = self.loaders[name]
                loader_name = name
                break
        
        # [CRITICAL FIX] Sample unique source data
        max_attempts = 10
        unique_sample_found = False
        
        for attempt in range(max_attempts):
            # Create a unique ID for this potential sample
            sample_idx = random.randint(0, 10000)  # Using index as ID proxy
            unique_id = f"{loader_name}_{sample_idx}" if loader_name else f"mock_{sample_idx}"
            
            # Check if already used
            if unique_id not in self.used_source_sample_ids:
                self.used_source_sample_ids.add(unique_id)
                unique_sample_found = True
                logger.debug(f"Found unique sample: {unique_id} (attempt {attempt + 1})")
                break
            else:
                logger.debug(f"Sample {unique_id} already used, trying another...")
        
        if not unique_sample_found:
            logger.warning(f"Could not find unique sample after {max_attempts} attempts")
        
        # [NEW DIVERSITY LOGIC] Randomize task logic
        # Randomly choose the property to compare
        properties_to_compare = ['area', 'width', 'height', 'aspect_ratio', 'position']
        chosen_property = random.choice(properties_to_compare)
        
        # Randomly choose the comparison operator based on property
        if chosen_property == 'area':
            operators = ['larger', 'smaller', 'equal in size to']
        elif chosen_property in ['width', 'height']:
            operators = ['wider', 'narrower', 'taller', 'shorter']
        elif chosen_property == 'aspect_ratio':
            operators = ['more elongated', 'more square', 'similar aspect ratio to']
        else:  # position
            operators = ['above', 'below', 'left of', 'right of']
        
        chosen_operator = random.choice(operators)
        logger.debug(f"Task variation: Compare {chosen_property} using '{chosen_operator}'")
        
        # Generate objects for comparison with varied properties
        object_a = {
            "class_name": random.choice(["cat", "dog", "car", "tree", "building", "person"]),
            "point": [random.randint(100, 400), random.randint(100, 400)],
            "area": random.randint(5000, 20000)
        }
        
        object_b = {
            "class_name": random.choice(["cat", "dog", "car", "tree", "building", "person"]),
            "point": [random.randint(400, 700), random.randint(100, 400)],
            "area": random.randint(5000, 20000)
        }
        
        object_c = {
            "class_name": random.choice(["bird", "chair", "lamp", "bottle"]),
            "point": [random.randint(200, 600), random.randint(400, 600)],
            "area": random.randint(3000, 10000)
        }
        
        # Determine which object is larger
        if object_a["area"] > object_b["area"]:
            ground_truth_conclusion = f"The {object_a['class_name']} is larger than the {object_b['class_name']}."
        else:
            ground_truth_conclusion = f"The {object_b['class_name']} is larger than the {object_a['class_name']}."
        
        # [CRITICAL NEW LOGIC] Dynamic Style Forcing
        chosen_style = random.choice(self.styles)
        logger.debug(f"Selected style: {chosen_style['name']}")
        
        # Build the required placeholders
        # Format objects as JSON-like strings (these are values, not template variables)
        import json
        object_a_str = json.dumps(object_a)
        object_b_str = json.dumps(object_b)
        object_c_str = json.dumps(object_c)
        
        placeholders = {
            'source_dataset': loader_name.replace('_', ' ').title() if loader_name else "COCO",
            'task_goal': f"Compare the size of the {object_a['class_name']} and the {object_b['class_name']}.",
            'ground_truth_conclusion': ground_truth_conclusion,
            'object_A_details_json': object_a_str,
            'object_B_details_json': object_b_str,
            'object_C_details_json': object_c_str
        }
        
        # Create initial metadata
        metadata = {
            'task_type': 'geometric_comparison',
            'property_compared': chosen_property,
            'comparison_operator': chosen_operator,
            'style_used': chosen_style.get('name', 'Unknown'),
            'object_a_class': object_a['class_name'],
            'object_b_class': object_b['class_name']
        }
        
        logger.debug(f"Constructed placeholders: {list(placeholders.keys())}")
        return placeholders, metadata
    
    def _build_easy_context(self) -> Dict[str, str]:
        """Build context for easy difficulty level."""
        context = {}
        
        # Get COCO loader from manifest
        loader_name = 'coco2017_train'
        if loader_name not in self.loaders:
            logger.warning(f"Datasource '{loader_name}' not found for Easy difficulty. Check manifest.")
            return self._build_mock_easy_context()
        
        loader = self.loaders[loader_name]
        
        try:
            # Sample an image with multiple objects
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Extract relevant information
            if 'annotations' in sample and len(sample['annotations']) >= 2:
                # Select two objects for comparison
                obj1, obj2 = random.sample(sample['annotations'], 2)
                
                context['easy_source_dataset'] = 'COCO 2017'
                context['easy_image_description'] = self._describe_image(sample)
                context['easy_object_A_class'] = obj1.get('category_name', 'object')
                context['easy_object_A_bbox'] = str(obj1.get('bbox', [100, 100, 200, 200]))
                context['easy_object_B_class'] = obj2.get('category_name', 'object')
                context['easy_object_B_bbox'] = str(obj2.get('bbox', [300, 300, 400, 400]))
                context['easy_property_to_compare'] = random.choice(['size', 'position', 'shape'])
                context['easy_expected_operations'] = 'SEGMENT_OBJECT_AT, GET_PROPERTIES'
                
            else:
                # Fallback to mock if insufficient annotations
                return self._build_mock_easy_context()
                
        except Exception as e:
            logger.error(f"Error building easy context: {e}")
            return self._build_mock_easy_context()
        
        return context
    
    def _build_medium_context(self) -> Dict[str, str]:
        """Build context for medium difficulty level."""
        context = {}
        
        # Try medium dataset loaders from manifest (LVIS, SA1B)
        loader = None
        loader_name = None
        for name in ['lvis_v1_train', 'sa1b_for_segmentation']:
            if name in self.loaders:
                loader = self.loaders[name]
                loader_name = name
                break
        
        if not loader:
            logger.warning("Datasources 'lvis_v1_train' and 'sa1b_for_segmentation' not found for Medium difficulty. Check manifest.")
            return self._build_mock_medium_context()
        
        try:
            # Sample an image with fine-grained categories
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build medium difficulty context
            context['medium_source_dataset'] = loader_name.replace('_', ' ').upper()
            context['medium_image_description'] = self._describe_complex_scene(sample)
            context['medium_num_objects'] = str(random.randint(3, 5))
            context['medium_comparison_type'] = random.choice([
                'relative spatial arrangement',
                'shape complexity comparison',
                'size hierarchy determination'
            ])
            context['medium_spatial_constraint'] = random.choice([
                'within the left half of the image',
                'in the foreground',
                'arranged horizontally'
            ])
            context['medium_property_set'] = ', '.join(random.sample(self.property_types, 3))
            context['medium_expected_operations'] = 'Multiple SEGMENT_OBJECT_AT, GET_PROPERTIES, spatial analysis'
            
        except Exception as e:
            logger.error(f"Error building medium context: {e}")
            return self._build_mock_medium_context()
        
        return context
    
    def _build_hard_context(self) -> Dict[str, str]:
        """Build context for hard difficulty level."""
        context = {}
        
        # Try hard dataset loader from manifest (PartImageNet)
        loader_name = 'part_imagenet_train'
        loader = None
        if loader_name in self.loaders:
            loader = self.loaders[loader_name]
        
        if not loader:
            logger.warning("Datasource 'part_imagenet_train' not found for Hard difficulty. Check manifest.")
            return self._build_mock_hard_context()
        
        try:
            # Sample an image with part annotations
            sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
            
            # Build hard difficulty context with part-level reasoning
            context['hard_source_dataset'] = 'PartImageNet'
            context['hard_image_description'] = self._describe_part_level_image(sample)
            context['hard_main_object'] = sample.get('category_name', 'complex object')
            context['hard_num_parts'] = str(random.randint(4, 8))
            context['hard_reasoning_type'] = random.choice([
                'hierarchical part-whole relationships',
                'symmetry analysis across parts',
                'geometric pattern recognition',
                'structural integrity assessment'
            ])
            context['hard_comparison_criteria'] = random.choice([
                'bilateral symmetry of parts',
                'proportional relationships between components',
                'geometric regularity of arrangements',
                'fractal-like self-similarity'
            ])
            context['hard_analysis_depth'] = random.choice([
                'pixel-level precision',
                'sub-part decomposition',
                'multi-scale analysis'
            ])
            context['hard_expected_operations'] = 'Nested SEGMENT_OBJECT_AT, recursive GET_PROPERTIES, complex spatial reasoning'
            
        except Exception as e:
            logger.error(f"Error building hard context: {e}")
            return self._build_mock_hard_context()
        
        return context
    
    def _describe_image(self, sample: Dict) -> str:
        """Generate a natural language description of an image."""
        if 'caption' in sample:
            return sample['caption']
        
        # Generate description from annotations
        if 'annotations' in sample:
            num_objects = len(sample['annotations'])
            categories = list(set(ann.get('category_name', 'object') 
                                for ann in sample['annotations']))
            return f"An image containing {num_objects} objects including {', '.join(categories[:3])}"
        
        return "An image with multiple objects for comparison"
    
    def _describe_complex_scene(self, sample: Dict) -> str:
        """Generate description for medium difficulty scene."""
        base_desc = self._describe_image(sample)
        complexity_additions = [
            "with overlapping objects",
            "featuring varied scales and perspectives",
            "with complex spatial arrangements",
            "containing both foreground and background elements"
        ]
        return f"{base_desc} {random.choice(complexity_additions)}"
    
    def _describe_part_level_image(self, sample: Dict) -> str:
        """Generate description for hard difficulty with part-level details."""
        if 'part_annotations' in sample:
            num_parts = len(sample.get('part_annotations', []))
            return f"A detailed image of a {sample.get('category_name', 'complex object')} with {num_parts} annotated parts showing intricate geometric relationships"
        
        return f"A high-resolution image suitable for part-level geometric analysis"
    
    def _generate_task_description(self, difficulty: str) -> str:
        """Generate a task description based on difficulty."""
        descriptions = {
            'easy': "Compare basic geometric properties between two clearly visible objects",
            'medium': "Analyze spatial relationships and properties among multiple objects with specific constraints",
            'hard': "Perform complex part-level geometric reasoning with hierarchical analysis"
        }
        return descriptions.get(difficulty, descriptions['medium'])
    
    def _get_available_operations(self) -> str:
        """Get the list of available operations for this task type."""
        return """
        - SEGMENT_OBJECT_AT(x, y): Segment object at pixel coordinates
        - GET_PROPERTIES(object_id): Extract geometric properties of segmented object
        - COMPARE(property, object1, object2): Compare specific property between objects
        - SPATIAL_RELATION(object1, object2): Determine spatial relationship
        """
    
    def _get_output_format(self) -> str:
        """Get the expected output format specification."""
        return """
        {
            "trajectory": [
                {
                    "step": int,
                    "thought": "reasoning about what to do",
                    "action": "operation_name",
                    "parameters": {...},
                    "result": "operation output"
                }
            ],
            "final_answer": "conclusion based on geometric analysis",
            "extracted_properties": {
                "object1": {...},
                "object2": {...}
            },
            "comparison_result": "detailed comparison outcome"
        }
        """
    
    def _get_active_datasets(self) -> List[str]:
        """Get list of datasets being used."""
        active = []
        for loader_name in self.loaders.keys():
            if any(loader_name.startswith(prefix) for prefix in ['coco', 'lvis', 'ade20k', 'part']):
                active.append(loader_name)
        return active
    
    # Mock context builders for fallback
    def _build_mock_easy_context(self) -> Dict[str, str]:
        """Build mock context for easy level when loaders unavailable."""
        return {
            'easy_source_dataset': 'COCO 2017 (mock)',
            'easy_image_description': 'An image with a cat and a dog on a sofa',
            'easy_object_A_class': 'cat',
            'easy_object_A_bbox': '[120, 80, 220, 180]',
            'easy_object_B_class': 'dog',
            'easy_object_B_bbox': '[250, 100, 380, 240]',
            'easy_property_to_compare': 'size',
            'easy_expected_operations': 'SEGMENT_OBJECT_AT, GET_PROPERTIES'
        }
    
    def _build_mock_medium_context(self) -> Dict[str, str]:
        """Build mock context for medium level when loaders unavailable."""
        return {
            'medium_source_dataset': 'LVIS (mock)',
            'medium_image_description': 'A kitchen scene with multiple utensils and appliances',
            'medium_num_objects': '4',
            'medium_comparison_type': 'relative spatial arrangement',
            'medium_spatial_constraint': 'on the countertop',
            'medium_property_set': 'size, position, shape',
            'medium_expected_operations': 'Multiple SEGMENT_OBJECT_AT, GET_PROPERTIES'
        }
    
    def _build_mock_hard_context(self) -> Dict[str, str]:
        """Build mock context for hard level when loaders unavailable."""
        return {
            'hard_source_dataset': 'PartImageNet (mock)',
            'hard_image_description': 'A detailed view of a bicycle with visible components',
            'hard_main_object': 'bicycle',
            'hard_num_parts': '6',
            'hard_reasoning_type': 'hierarchical part-whole relationships',
            'hard_comparison_criteria': 'proportional relationships between components',
            'hard_analysis_depth': 'sub-part decomposition',
            'hard_expected_operations': 'Nested SEGMENT_OBJECT_AT, recursive GET_PROPERTIES'
        }
    
    def _validate_and_process_response(self, llm_response: Dict, context: Dict) -> Optional[Dict]:
        """
        Validates the LLM's response for geometric comparison tasks.
        Implements FLEXIBLE validation for trajectory structure.
        
        Args:
            llm_response: The raw JSON response from the LLM
            context: The context placeholders used for generation
            
        Returns:
            The validated CoTA sample dict, or None if validation fails
        """
        # 1. Basic structural validation
        if not isinstance(llm_response, dict):
            logger.warning(f"LLM response is not a dictionary: {type(llm_response)}")
            return None
        
        # [NEW] Normalize all keys to lowercase for robust checking
        try:
            normalized_response = {k.lower(): v for k, v in llm_response.items()}
        except AttributeError:
            logger.warning("Validation failed: LLM output was not a valid dictionary.")
            return None
        
        # Check for required fields using normalized keys
        if 'question' not in normalized_response:
            logger.warning(f"LLM response missing 'question'. Got keys: {list(normalized_response.keys())}")
            return None
        
        # Handle both 'final_answer' and 'finalanswer' cases
        if 'final_answer' not in normalized_response and 'finalanswer' not in normalized_response:
            logger.warning(f"LLM response missing 'final_answer'. Got keys: {list(normalized_response.keys())}")
            return None
        
        # Map normalized keys back to the original response structure
        llm_response['question'] = normalized_response.get('question', llm_response.get('question'))
        llm_response['final_answer'] = normalized_response.get('final_answer', normalized_response.get('finalanswer', llm_response.get('final_answer', llm_response.get('finalAnswer'))))
        
        # 2. FLEXIBLE Trajectory validation
        # Also check normalized keys for 'actions' or 'trajectory'
        trajectory = normalized_response.get('actions', normalized_response.get('trajectory', llm_response.get('actions', llm_response.get('trajectory', []))))
        
        if not isinstance(trajectory, list):
            logger.warning(f"Trajectory is not a list: {type(trajectory)}")
            return None
        
        # Normalize the trajectory to handle various formats
        normalized_trajectory = self._normalize_trajectory(trajectory)
        
        # [REVISED]: Check if the trajectory has at least the minimum required length
        min_trajectory_length = 3
        if len(normalized_trajectory) < min_trajectory_length:
            logger.warning(f"Validation failed: Trajectory must have at least {min_trajectory_length} steps. Got {len(normalized_trajectory)}")
            return None
        
        # [REVISED]: Check if at least one action exists anywhere in the trajectory
        action_exists = any(step.get('type') == 'action' for step in normalized_trajectory if isinstance(step, dict))
        
        if not action_exists:
            logger.warning("Validation failed: Trajectory must contain at least one 'action' step.")
            return None
        
        llm_response['trajectory'] = normalized_trajectory
        
        # For geometric tasks, we expect SEGMENT_OBJECT_AT or GET_PROPERTIES actions
        has_relevant_action = False
        for step in normalized_trajectory:
            if isinstance(step, dict):
                # Now check with normalized structure
                if step.get('type') == 'action':
                    action_name = step.get('name', '').upper().replace('-', '_')
                    # Check various naming conventions
                    if action_name in ['SEGMENT_OBJECT_AT', 'GET_PROPERTIES']:
                        has_relevant_action = True
                        
                        # Validate parameters for SEGMENT_OBJECT_AT action
                        if action_name == 'SEGMENT_OBJECT_AT':
                            parameters = step.get('parameters')
                            
                            # Check if 'parameters' field exists and is a dictionary
                            if not parameters or not isinstance(parameters, dict):
                                logger.warning(
                                    f"Validation failed: SEGMENT_OBJECT_AT action is missing a valid 'parameters' dictionary. Got: {parameters}"
                                )
                                return None
                            
                            # Check if 'point' or 'coordinates' key exists within 'parameters'
                            point = parameters.get('point') or parameters.get('coordinates')
                            if not point:
                                logger.warning(
                                    f"Validation failed: SEGMENT_OBJECT_AT parameters missing 'point' or 'coordinates'. Got: {parameters}"
                                )
                                return None
                            
                            # Validate point format (should be [x, y])
                            if not (isinstance(point, list) and len(point) == 2):
                                logger.warning(
                                    f"Validation failed: 'point' is not a list of 2 coordinates. Got: {point}"
                                )
                                return None
                        break
        
        if not has_relevant_action and len(normalized_trajectory) > 0:
            logger.debug("Geometric task should include segmentation or property actions")
            # Be lenient - just log, don't fail
        
        # 3. Answer validation (ultra-lenient)
        final_answer = llm_response.get('final_answer', '')
        validation_strictness = getattr(self, 'validation_strictness', 'ultra_lenient')
        
        if validation_strictness == 'ultra_lenient':
            # Accept any non-empty answer
            if len(str(final_answer).strip()) > 0:
                logger.debug(f"Ultra-lenient: Accepting answer '{final_answer[:50]}...'")
            else:
                logger.warning("Final answer is empty")
                return None
        
        # 4. Add difficulty if not present
        if 'difficulty' not in llm_response and 'difficulty' in context:
            llm_response['difficulty'] = context.get('difficulty', 'Medium')
        
        # 5. Log successful validation
        logger.info(f"✓ Geometric comparison sample validated successfully")
        
        return llm_response