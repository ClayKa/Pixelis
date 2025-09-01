"""
Centralized style definitions for dynamic style forcing in CoTA generation.

This module contains reusable style personas that can be imported by any task generator
to ensure diverse linguistic expression in generated samples.
"""

# Universal styles applicable to all task types
UNIVERSAL_STYLES = [
    {
        'name': 'The Direct Analyst',
        'desc': 'Provides straightforward, factual analysis',
        'q': 'What do you observe?',
        'a': 'The observation clearly shows the expected result.'
    },
    {
        'name': 'The Skeptical Investigator',
        'desc': 'Questions everything and demands evidence',
        'q': 'Is this really what it appears to be?',
        'a': 'Evidence confirms the initial hypothesis after thorough investigation.'
    },
    {
        'name': 'The Technical Expert',
        'desc': 'Uses precise technical terminology',
        'q': 'What are the technical specifications?',
        'a': 'Technical analysis reveals parameters within expected tolerances.'
    },
    {
        'name': 'The Storyteller',
        'desc': 'Narrates observations as a story',
        'q': 'What story unfolds here?',
        'a': 'The tale reveals itself through careful observation of details.'
    },
    {
        'name': 'The Scientist',
        'desc': 'Applies scientific method and formal language',
        'q': 'What empirical data can be extracted?',
        'a': 'Empirical observation yields quantifiable results with high confidence.'
    },
    {
        'name': 'The Detective',
        'desc': 'Investigates like solving a mystery',
        'q': 'What clues are present?',
        'a': 'The evidence points conclusively to the solution.'
    },
    {
        'name': 'The Minimalist',
        'desc': 'Uses the fewest words possible',
        'q': 'Result?',
        'a': 'Confirmed. Clear.'
    },
    {
        'name': 'The Teacher',
        'desc': 'Explains as educational lessons',
        'q': 'What can we learn from this?',
        'a': 'This teaches us an important principle about visual analysis.'
    },
    {
        'name': 'The Poet',
        'desc': 'Uses metaphorical and artistic language',
        'q': 'What beauty lies hidden within?',
        'a': 'Like dawn breaking, the truth emerges from shadows.'
    },
    {
        'name': 'The Engineer',
        'desc': 'Focuses on specifications and measurements',
        'q': 'What are the measured parameters?',
        'a': 'Measurements: X=125, Y=89, Accuracy=98.5%.'
    }
]

# Task-specific style adaptations
TEMPORAL_STYLES = [
    {
        'name': 'The Timeline Analyst',
        'desc': 'Focuses on temporal sequences and chronology',
        'q': 'When does the critical moment occur?',
        'a': 'The event manifests at T+4.3 seconds, frame 129.'
    },
    {
        'name': 'The Video Editor',
        'desc': 'Uses film and video editing terminology',
        'q': 'Where should we make the cut?',
        'a': 'Cut point identified at 00:03:45, transition frame optimal.'
    }
]

OCR_STYLES = [
    {
        'name': 'The Transcriptionist',
        'desc': 'Focuses on accurate text transcription',
        'q': 'What text is visible?',
        'a': 'Transcription: "IMPORTANT NOTICE" - all caps, serif font.'
    },
    {
        'name': 'The Document Analyst',
        'desc': 'Analyzes documents professionally',
        'q': 'What does the document contain?',
        'a': 'Document header indicates official correspondence, dated 2024.'
    }
]

TRACKING_STYLES = [
    {
        'name': 'The Motion Analyst',
        'desc': 'Specializes in movement and trajectory analysis',
        'q': 'How does the object move?',
        'a': 'Trajectory follows parabolic arc, velocity decreasing.'
    },
    {
        'name': 'The Surveillance Expert',
        'desc': 'Reports like security monitoring',
        'q': 'Track the subject.',
        'a': 'Subject tracked: entered at 0:15, exited at 0:47, path recorded.'
    }
]