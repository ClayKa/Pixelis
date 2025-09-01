# core/data_generation/detail_perception.py

import random
import re
import json
from typing import Any, Dict, List, Tuple, Optional
import logging
from datetime import datetime

from .base_generator import BaseTaskGenerator

logger = logging.getLogger(__name__)

class DetailPerceptionTaskGenerator(BaseTaskGenerator):
    """
    [FINAL V4]
    Generates CoTA samples for fine-grained detail perception tasks.
    This generator is self-contained and parses its own prompt structure.
    """
    
    # [NEW] Define the framing template library for diverse question structures
    QUESTION_FRAMING_TEMPLATES = [
        "I need to {task_goal}. Can you please zoom in on the area at {location_info} and tell me what you find?",
        "The current task is to {task_goal}. Please focus on the region at {location_info} and provide a detailed observation.",
        "Let's investigate the section at {location_info}. My primary objective is to {task_goal}. What becomes visible after magnification?",
        "What can you tell me about the area at {location_info}? I'm specifically trying to {task_goal}.",
        "My analysis requires me to {task_goal}. Zoom in on {location_info} and report your findings.",
        "Could you provide a close-up of {location_info}? It's essential that I {task_goal} for my report.",
        "The next step is to {task_goal}. Please use the zoom tool on {location_info} and describe the result.",
        "To proceed, I must first {task_goal}. Let's examine {location_info}.",
        "Is it possible to {task_goal}? You'll need to magnify the area at {location_info} to be sure.",
        "A detailed look at {location_info} is required. My goal here is to {task_goal}.",
        "Help me {task_goal} by examining the region at {location_info} closely.",
        "For quality control purposes, I need to {task_goal}. Check {location_info} after zooming in.",
        "Can you assist me in trying to {task_goal}? The area of interest is {location_info}.",
        "Please {task_goal} by inspecting {location_info} with magnification.",
        "I'm investigating whether we can {task_goal}. Focus on {location_info} and let me know.",
        "To complete this inspection, I must {task_goal}. Examine {location_info} in detail.",
        "Would you mind checking if we can {task_goal}? The coordinates are {location_info}.",
        "My objective: {task_goal}. Location: {location_info}. What do you observe?",
        "Zoom into {location_info} - I'm hoping to {task_goal}.",
        "At {location_info}, can you {task_goal}?",
        "I'm curious to {task_goal}. Take a closer look at {location_info}.",
        "Before proceeding, let's {task_goal}. The target area is {location_info}.",
        "This requires us to {task_goal}. Please magnify {location_info} and describe what you see.",
        "Looking at {location_info}, my task is to {task_goal}. What's visible there?",
        "For documentation, I need to {task_goal}. Could you zoom into {location_info}?",
        "Quick check: Can you {task_goal} by examining {location_info}?",
        "The specification requires that we {task_goal}. Please inspect {location_info}.",
        "Help needed: I'm trying to {task_goal} at {location_info}.",
        "Regarding {location_info}, I need to {task_goal}. What do you find?",
        "Part of my analysis involves trying to {task_goal}. Look at {location_info} closely.",
        "While reviewing {location_info}, I should {task_goal}. Can you help with the magnification?",
        "It's critical that I {task_goal}. Please provide a detailed view of {location_info}.",
        "During this inspection phase, I need to {task_goal}. Focus the zoom on {location_info}."
        "For the purpose of my ongoing analysis, could you please direct your attention to {location_info} and {task_goal} with a magnified view?",
        "A high-resolution inspection of the area specified at {location_info} is required; please proceed to {task_goal} and report the findings.",
        "I suspect there's more than meets the eye at {location_info}; let's scrutinize it closely to see if we can {task_goal} and uncover any hidden details.",
        "Hey, can we get a better look at what's going on over at {location_info}? I'm just trying to {task_goal} without missing anything important.",
        "It is critical that we {task_goal} immediately; please execute a zoom on {location_info} and provide a summary of all visible data.",
        "Would it be possible for you to examine the region at {location_info}? A detailed view would greatly assist me as I attempt to {task_goal}.",
        "To resolve the current ambiguity in my report, I must {task_goal}, which seems to require a much closer inspection of the features at {location_info}.",
        "For this observational study, let's focus our instrumentation on {location_info}; the primary objective is to {task_goal} and document the results.",
        "Regarding the specified coordinates at {location_info}, I have a pending task to {task_goal}, and a magnified view is the next logical step.",
        "Let's work together on this next part; my goal is to {task_goal}, and I think a deep dive into {location_info} is our best approach.",
        "I'm curious what story unfolds at {location_info} when we look closer; please bring that section into sharp focus so that I can {task_goal}.",
        "The next step in the procedure is to {task_goal}, so please apply the necessary magnification to the target zone at {location_info} and confirm.",
        "To gather the necessary data points for my model, I need to {task_goal}; could you provide a granular view of the components located at {location_info}?",
        "I'm not convinced we can {task_goal} from this distance; please challenge my assumption by providing a high-magnification view of {location_info}.",
        "For our quality assurance check, it is imperative that I {task_goal}; let's start by performing a detailed visual inspection of {location_info}.",
        "In order to compare it with the previous sample, I have to {task_goal}; can you give me a close-up of the specific texture at {location_info}?",
        "Let's do some exploratory analysis on the area at {location_info}; my hope is that a closer look will allow us to {task_goal} successfully.",
        "Before I finalize my conclusion, I need you to {task_goal}; please use the zoom tool on {location_info} to provide definitive evidence.",
        "My analysis requires a closer look at the asset located at {location_info}; please perform a zoom operation so I am able to {task_goal}.",
        "The current investigation hinges on our ability to {task_goal}, so I need you to give me the most detailed possible view of what's happening at {location_info}.",
        "In accordance with the project specifications, I need to {task_goal}, so please provide a detailed view of the component at {location_info}.",
        "To complete the validation phase, let's {task_goal} by examining the asset at {location_info} under significant magnification.",
        "For documentation purposes, it's essential that I {task_goal}; could you please capture a high-detail image of the section at {location_info}?",
        "A key business requirement is to {task_goal}, which necessitates a close visual confirmation of the details present at {location_info}.",
        "Before we can sign off on this deliverable, I must {task_goal}; please zoom in on the designated area at {location_info} for a final check.",
        "The next action item on my list is to {task_goal}, so could you assist by providing a magnified inspection of {location_info}?",
        "To ensure compliance with our standards, I'm tasked to {task_goal}; let's get a better perspective on the situation at {location_info}.",
        "My report is incomplete until I can {task_goal}; a closer examination of the markings at {location_info} should provide the data I need.",
        "Let's review the asset at {location_info} more closely; my objective here is to {task_goal} and record the outcome.",
        "To move to the next phase of our workflow, it's a prerequisite to {task_goal}; please focus your analysis on the provided {location_info}.",
        "My current hypothesis requires that we {task_goal}; let's gather empirical data by observing the specimen at {location_info} up close.",
        "To properly classify the phenomenon at {location_info}, my methodology dictates that I must first {task_goal} based on its micro-features.",
        "The experiment's integrity depends on our ability to {task_goal}; please provide a microscopic view of the sample located at {location_info}.",
        "We need to document the state of the subject at {location_info}; let's {task_goal} to ensure our records are accurate and detailed.",
        "My research paper needs a key piece of evidence, which I hope to find if I can {task_goal}; please magnify the anomaly at {location_info}.",
        "Let's perform a fine-grained analysis of {location_info}, as my primary research goal at this stage is to {task_goal} with high confidence.",
        "The data from {location_info} is currently inconclusive; perhaps if we zoom in, I'll have enough visual information to {task_goal}.",
        "I must {task_goal} to confirm its composition; please provide a detailed topographical view of the surface at {location_info}.",
        "In this phase of data collection, the protocol is to {task_goal}; please target the coordinates at {location_info} for a close-range scan.",
        "To test my theory, I need to {task_goal}; a magnified look at the structure within {location_info} is absolutely essential.",
        "A hidden narrative seems to be woven into the fabric of {location_info}; let's magnify it and see if we can {task_goal} from the details.",
        "I want to explore the intricate tapestry at {location_info}; please give me a closer glimpse so I can {task_goal} and appreciate its complexity.",
        "To capture the subtle nuances of the subject at {location_info}, I need to {task_goal}; could you bring its finer points into focus?",
        "The true character of the object at {location_info} might only be visible up close; let's zoom in and attempt to {task_goal}.",
        "Let's treat the area at {location_info} as a canvas; by zooming in, I'm hoping to {task_goal} and discover something unexpected.",
        "I'm on a quest to {task_goal}, and my journey has led me to this point of interest at {location_info}; show me what lies hidden there.",
        "Let's peel back the layers at {location_info}; a magnified view should help me {task_goal} and understand its deeper story.",
        "To truly understand the essence of {location_info}, I feel I must {task_goal}; can you provide a view that captures its most minute details?",
        "I'm trying to {task_goal}, which feels like finding a secret; maybe a closer look at {location_info} will reveal the key.",
        "The beauty of {location_info} is in its details; please zoom in so I can {task_goal} and document its unique features.",
        "I need to figure out if I can {task_goal}; can you just give me a quick, zoomed-in look at what's at {location_info}?",
        "To make sure I'm not missing anything at {location_info}, I want to {task_goal}; could you blow that section up for me?",
        "What's the deal with the area at {location_info}? Let me get a close-up so I can {task_goal} and be done with it.",
        "Alright, let's get this sorted. I have to {task_goal}, so please show me the details at {location_info}.",
        "I'm having a hard time making out the details at {location_info}; can you punch in so it's clearer and I can {task_goal}?",
        "Let's just double-check the situation at {location_info}. My plan is to {task_goal}, and a zoom should make that easy.",
        "I can't move forward until I {task_goal}, so can you please just focus on {location_info} and show me what's there?",
        "Let's settle this once and for all. I need to {task_goal}, and the answer has to be somewhere in {location_info}.",
        "My only remaining task is to {task_goal}. A good look at {location_info} should wrap this up.",
        "Before I forget, I need to {task_goal}. Can you point the zoom at {location_info} for a second?",
        "My forensic investigation requires that I {task_goal}; please provide a high-magnification, evidentiary view of the anomaly at {location_info}.",
        "From a strategic standpoint, it's vital that we {task_goal} now; let's get a tactical overview of {location_info} at maximum zoom.",
        "This archival task requires me to {task_goal} for posterity; please provide a clear, magnified capture of the inscription at {location_info}.",
        "As a final diagnostic step, I have to {task_goal}; let's get a close-up on the component at {location_info} to check for faults.",
        "My mandate is to {task_goal} in order to verify authenticity; a magnified analysis of the material at {location_info} is therefore required.",
        "To unlock the next achievement, the game requires me to {task_goal}; can you zoom in on the clue hidden at {location_info}?",
        "The safety protocol is clear: I must {task_goal} before proceeding. Please perform a detailed visual safety check of the mechanism at {location_info}."
        "The schematic indicates a micro-resistor at {location_info}; I need to {task_goal} by verifying its solder integrity.",
        "Let's check the surface tolerance at {location_info}; my objective is to {task_goal} and see if there are any stress fractures.",
        "To complete the quality report, I must {task_goal}, which requires a magnified view of the PCB trace at {location_info} to check for corrosion.",
        "We have a potential fatigue point at {location_info}; please zoom in so I can {task_goal} and analyze the material's grain structure.",
        "To confirm the part number, I must {task_goal}; a close-up of the etching on the component at {location_info} is necessary.",
        "The assembly instructions require me to {task_goal}; let's inspect the set screw at {location_info} to ensure it's seated correctly.",
        "There's a reported fluid leak originating near {location_info}; I have to {task_goal} by finding the source hairline crack.",
        "Let's get a close-up of the weld bead at {location_info}. My job is to {task_goal} and check for porosity.",
        "The CAD model shows a specific feature at {location_info}. To {task_goal}, I need to compare the physical part to the digital twin.",
        "I need to {task_goal} by measuring the clearance at {location_info}; a magnified view will help me assess the gap.",
        "I'm debugging a rendering issue, and I need to {task_goal}; can you show me the exact pixel values at {location_info}?",
        "For this UI/UX review, I have to {task_goal}; please magnify the button asset at {location_info} to check for aliasing.",
        "There's a visual artifact appearing at {location_info} in the final render. Let's zoom in to {task_goal} and identify its cause.",
        "My task is to {task_goal} for the accessibility report; can you provide a close-up of the text at {location_info} to check its contrast ratio?",
        "I need to {task_goal} by examining the vector point at {location_info}; is the bezier curve smooth or has it been distorted?",
        "This data visualization is too dense. To {task_goal}, I need you to isolate and magnify the cluster of data points at {location_info}.",
        "The game's texture map at {location_info} looks low-res. I need to {task_goal} by checking the source file's pixel density.",
        "I'm checking for compression artifacts. Can we {task_goal} by inspecting the blockiness at {location_info}?",
        "Let's review the animation's keyframe at {location_info}. My goal is to {task_goal} by analyzing the character's finger placement.",
        "To verify the new font implementation, I must {task_goal}; a close-up of the kerning between letters at {location_info} is what I need.",
        "For this pathology report, I have to {task_goal}; please provide a microscopic view of the tissue sample from {location_info}.",
        "I'm trying to {task_goal} by identifying the organism in this petri dish; focus on the largest colony at {location_info}.",
        "Let's examine the cellular structure at {location_info}. My objective is to {task_goal} and look for any anomalies.",
        "I need to {task_goal} to complete my botanical illustration; can you give me a detailed view of the stamen and pistil at {location_info}?",
        "There appears to be fungal growth at {location_info} on the leaf. A close-up is needed so I can {task_goal} and identify the species.",
        "To complete the patient's chart, I must {task_goal}. Please magnify the suture at {location_info} to check for signs of infection.",
        "Let's analyze the X-ray again. I need to {task_goal} by looking for a micro-fracture at {location_info}.",
        "My research involves trying to {task_goal}; let's observe the reaction site within the compound at {location_info}.",
        "Is that a pest or a benign insect? To {task_goal}, I'll need a much closer look at the creature on the plant stem at {location_info}.",
        "The goal is to {task_goal}. Please provide a view of the DNA strand at {location_info} sharp enough to see the base pairs.",
        "As part of my art conservation work, I need to {task_goal}; let's analyze the craquelure pattern on the painting's surface at {location_info}.",
        "To authenticate this manuscript, I must {task_goal}; please magnify the watermark embedded in the paper at {location_info}.",
        "I'm trying to {task_goal} by deciphering the marginalia written in the corner of the page at {location_info}.",
        "Let's examine the cartographer's work at {location_info}. My aim is to {task_goal} by identifying the tiny coastal villages.",
        "I need to {task_goal} by studying the artist's brushstrokes. Can you provide a high-resolution view of the signature at {location_info}?",
        "This ancient potsherd at {location_info} has faint markings. I need to {task_goal} by determining if they are decorative or linguistic.",
        "My analysis of this primary source requires me to {task_goal}. Let's zoom in on the seal at {location_info} to identify the family crest.",
        "I am trying to {task_goal} from this historical photograph. Please magnify the banner being held by the protestors at {location_info}.",
        "To understand the stonemason's technique, I must {task_goal}. Let's inspect the tool marks on the hieroglyph at {location_info}.",
        "I need to {task_goal} by analyzing the underdrawing. Can you give me a view of the canvas at {location_info} that reveals the initial sketch?",
        "Captain's Log: To {task_goal}, we must analyze the alien glyph at {location_info} on the planet's surface. Magnify, Lieutenant.",
        "My mission as Agent 7 is to {task_goal}. The intel is hidden in a microdot on the document at {location_info}; zoom and enhance.",
        "To disarm the trap, I must {task_goal} by reading the runic inscription at {location_info}. A closer look is my only hope.",
        "The ship's structural integrity is compromised! I need to {task_goal} by inspecting the hairline fracture on the hull at {location_info}.",
        "Our scanners have detected a faint energy signature at {location_info}. I need to {task_goal} to determine if it's hostile.",
        "The ancient prophecy can only be understood if I {task_goal}. Focus the scrying orb on the celestial alignment shown at {location_info}.",
        "My Pokedex is missing data. I need to {task_goal} by scanning the rare creature hiding in the foliage at {location_info}.",
        "To craft the legendary sword, the recipe says I must {task_goal}. Let's get a closer look at the alchemical symbol at {location_info}.",
        "Computer, I need to {task_goal}. Run a diagnostic on the power conduit at {location_info} and display the energy flow.",
        "The target is at {location_info}. To {task_goal} for this bounty, I need to identify them by the scar on their left hand.",
        "I'm trying to win this board game, so I need to {task_goal}. Can you zoom in on the opponent's resource tokens at {location_info}?",
        "To cook this steak perfectly, I have to {task_goal}. Let's get a close-up of the marbling on the cut of meat at {location_info}.",
        "I'm trying to {task_goal} for my model train set. Please show me the tiny manufacturer's logo on the carriage at {location_info}.",
        "My DIY project requires me to {task_goal}. I need a better view of the wood grain at {location_info} to plan my cut.",
        "This coin might be valuable. I need to {task_goal} by checking for a mint mark at {location_info}.",
        "I'm trying to identify this bird for my birdwatching journal. Can you {task_goal} by zooming in on its wing markings at {location_info}?",
        "To fix my watch, I must {task_goal}. I need a magnified view of the tiny gear at {location_info}.",
        "This recipe is hard to read. Can you {task_goal} by zooming in on the measurement instructions at {location_info}?",
        "I'm trying to {task_goal} before I buy this online. Can you show me a detailed view of the stitching at {location_info}?",
        "Let's check the car engine. I need to {task_goal} by reading the serial number on the part at {location_info}.",
        "In order to {task_goal}, would you please provide a magnified view of {location_info}?",
        "I need you to focus on {location_info}. My objective here is to {task_goal}.",
        "Can you help me {task_goal}? It requires a detailed inspection of the area at {location_info}.",
        "For the next step of my process, I must {task_goal}. Please show me a close-up of {location_info}.",
        "To verify this information, I need to {task_goal}. Let's get a clearer look at {location_info}.",
        "Please provide a detailed observation of {location_info}. It's necessary for me to {task_goal}.",
        "My task is to {task_goal}, and the key detail is at {location_info}. Could you zoom in there for me?",
        "Let's get straight to the point. I need to {task_goal}, and for that, I need to see {location_info} up close.",
        "What is visible at {location_info} under high magnification? This information is critical for me to {task_goal}.",
        "A close examination of {location_info} is required for me to {task_goal}. Please proceed.",
        "For this legal discovery process, I am required to {task_goal}. Please provide an enhanced view of the signature on the document at {location_info}.",
        "I need to {task_goal} to ensure this contract is valid. Let's get a clear view of the notary's seal at {location_info}.",
        "My task is to {task_goal} for our forensic accounting investigation. Zoom in on the ledger entry at {location_info} and check for alterations.",
        "To build our case, we must {task_goal}. The key piece of evidence is the timestamp on the security footage at {location_info}.",
        "I need to {task_goal} by verifying the currency's serial number. Can you provide a high-resolution scan of the bill at {location_info}?",
        "Let's examine Exhibit A. My goal is to {task_goal} by identifying the smudged fingerprint at {location_info}.",
        "Before advising my client, I have to {task_goal}. Let's magnify the fine print in clause 7b at {location_info}.",
        "I am trying to {task_goal} to check for forgery. A close look at the paper fibers at {location_info} should be revealing.",
        "The financial audit requires that I {task_goal}. Please focus on the decimal point in the transaction at {location_info}.",
        "To establish a chain of custody, I need to {task_goal}. Let's get a close-up of the evidence tag attached at {location_info}.",
        "Target {location_info}. Let's {task_goal}. Go.",
        "My eyes aren't what they used to be. Help me {task_goal} at {location_info}.",
        "Curiosity compels me to {task_goal}. What's hiding at {location_info}?",
        "Okay, deeper dive. At {location_info}, I need to {task_goal}.",
        "Show me the magic at {location_info}. My aim is to {task_goal}.",
        "Let's get microscopic on {location_info}. The mission: {task_goal}.",
        "I have a hunch about {location_info}. Zoom in so I can {task_goal}.",
        "To confirm, I must {task_goal}. Center on {location_info} and magnify.",
        "What secrets does {location_info} hold? Let's {task_goal} and find out.",
        "The truth is at {location_info}. To {task_goal}, we need to look closer.",
        "I am currently unable to {task_goal} due to the low resolution. Can you enhance the view at {location_info}?",
        "As a meteorologist, I need to {task_goal} by analyzing the hook echo in the storm cell at {location_info} on this radar map.",
        "My geological survey requires me to {task_goal}; please provide a close-up of the crystalline structure in the rock sample at {location_info}.",
        "I'm a tailor and I must {task_goal} to ensure a perfect fit. Let's get a close look at the fabric's weave at {location_info}.",
        "To complete this crossword puzzle, I have to {task_goal}. Can you make out the tiny clue number at {location_info}?",
        "Show me the details at {location_info}. I'm trying to {task_goal} and need a better view.",
        "This is a fire investigation. I need to {task_goal} by examining the V-pattern of the charring at {location_info} to find the point of origin.",
        "I'm trying to {task_goal} for my urban planning proposal. Let's analyze the traffic flow at the intersection at {location_info} during peak hours.",
        "As part of this review, it is essential to {task_goal}. Let's start with a magnified look at {location_info}.",
        "Could you check {location_info} for me? My specific goal is to {task_goal}."
    ]
    
    def __init__(self, loaders: Dict[str, Any], config: Dict[str, Any], global_config: Dict[str, Any]):
        # 1. Let the parent class do its generic setup first
        super().__init__(loaders, config, global_config)
        
        # 2. [NEW] The subclass now parses its OWN specific prompt structure
        self.style_cookbook = self._parse_style_cookbook()
        if not self.style_cookbook:
            logger.warning(f"No style cookbook found in {self.task_name}'s prompt - using fallback styles")
            self.style_cookbook = self._get_fallback_styles()
        else:
            logger.info(f"Parsed {len(self.style_cookbook)} creative styles from prompt template")
    
    def _parse_style_cookbook(self) -> List[Dict]:
        """Parses the special 'Style Cookbook' block from this generator's prompt template."""
        logger.info("--- Parsing Style Cookbook ---")
        cookbook = []
        
        # Debug: Log template length to ensure it's loaded
        logger.debug(f"Template length: {len(self.prompt_template) if self.prompt_template else 0} characters")
        
        # Try multiple patterns to be robust
        patterns = [
            # Pattern 1: ## [STYLE X: Name]
            re.compile(r"\*?\*?##\s*\[\s*STYLE\s*(\d+):\s*(.*?)\]\s*\*?\*?\s*\n(.*?)(?=(##\s*\[\s*STYLE|\Z))", re.DOTALL),
            # Pattern 2: # [STYLE_START] ... # [STYLE_END] (current pattern)
            re.compile(r'# \[STYLE_START\]\s*\n# (.*?)\s*\n# \[STYLE_END\]', re.DOTALL),
            # Pattern 3: More flexible style block
            re.compile(r'\[STYLE[_\s]*(\d+)[:\s]*(.*?)\](.*?)(?=\[STYLE|\Z)', re.DOTALL | re.IGNORECASE)
        ]
        
        matches_found = False
        for pattern_idx, pattern in enumerate(patterns):
            matches = pattern.findall(self.prompt_template)
            
            if matches:
                logger.info(f"Pattern {pattern_idx + 1} found {len(matches)} potential style blocks.")
                matches_found = True
                
                for match_idx, match in enumerate(matches):
                    try:
                        if pattern_idx == 0 or pattern_idx == 2:  # Patterns with style_id
                            style_id_str, style_name, content_str = match[:3] if len(match) >= 3 else (str(match_idx + 1), "Unknown", str(match))
                            style_id = int(style_id_str)
                            
                            # Try to parse Q&A pairs
                            style_dict = {
                                "style_id": style_id,
                                "name": style_name.strip(),
                                "desc": "Style description"  # Can be parsed if needed
                            }
                            
                            # Try to extract questions and answers for different difficulties
                            for difficulty in ['EASY', 'MEDIUM', 'HARD']:
                                q_pattern = re.search(rf"\*?\s*{difficulty}.*?\*?Question:\*?\s*[\"']?(.*?)[\"']?(?:\n|\*)", content_str, re.IGNORECASE | re.DOTALL)
                                a_pattern = re.search(rf"\*?\s*{difficulty}.*?\*?Final\s*Answer:\*?\s*[\"']?(.*?)[\"']?(?:\n|\*|$)", content_str, re.IGNORECASE | re.DOTALL)
                                
                                if q_pattern:
                                    style_dict[f"q_{difficulty.lower()}"] = q_pattern.group(1).strip()
                                if a_pattern:
                                    style_dict[f"a_{difficulty.lower()}"] = a_pattern.group(1).strip()
                            
                            # Fallback: look for generic q and a
                            if 'q_easy' not in style_dict:
                                q_generic = re.search(r"[Qq]uestion:\s*[\"']?(.*?)[\"']?(?:\n|$)", content_str)
                                a_generic = re.search(r"[Aa]nswer:\s*[\"']?(.*?)[\"']?(?:\n|$)", content_str)
                                if q_generic:
                                    style_dict['q'] = q_generic.group(1).strip()
                                if a_generic:
                                    style_dict['a'] = a_generic.group(1).strip()
                            
                        else:  # Pattern 1 (JSON format)
                            block_str = match if isinstance(match, str) else match[0]
                            # Clean up and parse JSON
                            clean_str = block_str.strip()
                            if clean_str.startswith('#'):
                                clean_str = clean_str[1:].strip()
                            clean_str = clean_str.replace('{{', '{').replace('}}', '}')
                            
                            style_dict = json.loads(clean_str)
                            # Ensure style_id exists
                            if 'style_id' not in style_dict:
                                style_dict['style_id'] = match_idx + 1
                        
                        cookbook.append(style_dict)
                        logger.debug(f"Successfully parsed Style #{style_dict.get('style_id', match_idx)}: {style_dict.get('name', 'Unknown')}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse style block {match_idx + 1} with pattern {pattern_idx + 1}. Error: {e}")
                        logger.debug(f"Failed content: {str(match)[:200]}...")
                
                if cookbook:
                    break  # Stop if we successfully parsed styles
        
        if not matches_found:
            logger.warning("No style patterns matched in the template. Check prompt template format.")
            
        logger.info(f"Total styles parsed: {len(cookbook)}")
        return cookbook
    
    def _get_fallback_styles(self) -> List[Dict]:
        """Return fallback styles if none are found in the prompt template."""
        return [
            {
                'style_id': 1,
                'name': 'The Direct Inquirer',
                'desc': 'Asks straightforward, clear questions with simple answers',
                'q': 'Is there a red object in the specified area?',
                'a': 'Yes, after zooming in, a red warning sign is clearly visible.'
            },
            {
                'style_id': 2,
                'name': 'The Skeptic',
                'desc': 'Doubts everything and demands proof with assertive responses',
                'q': 'I doubt there\'s anything meaningful there. Prove it.',
                'a': 'Your skepticism is unfounded. The zoom reveals intricate details previously invisible.'
            },
            {
                'style_id': 3,
                'name': 'The Analyst',
                'desc': 'Provides detailed technical analysis with precise terminology',
                'q': 'What specific visual features become apparent upon magnification?',
                'a': 'Magnification reveals micro-textures, edge artifacts, and sub-pixel color variations.'
            },
            {
                'style_id': 4,
                'name': 'The Narrator',
                'desc': 'Tells a story about the discovery process',
                'q': 'What story does this hidden detail tell?',
                'a': 'As we zoom closer, a forgotten message emerges from the shadows.'
            },
            {
                'style_id': 5,
                'name': 'The Scientist',
                'desc': 'Uses formal scientific language and methodology',
                'q': 'What empirical observations can be made at 4x magnification?',
                'a': 'At 4x magnification, crystalline structures become observable with 95% clarity.'
            },
            {
                'style_id': 6,
                'name': 'The Detective',
                'desc': 'Investigates clues and evidence like solving a mystery',
                'q': 'What clues are hidden in this region?',
                'a': 'The evidence is clear: fingerprint patterns indicate recent activity.'
            },
            {
                'style_id': 7,
                'name': 'The Minimalist',
                'desc': 'Uses the fewest words possible while remaining clear',
                'q': 'Details visible?',
                'a': 'Text: "EXIT". Clear.'
            },
            {
                'style_id': 8,
                'name': 'The Teacher',
                'desc': 'Explains observations as educational lessons',
                'q': 'What can we learn from examining this area closely?',
                'a': 'This teaches us that surface textures often contain important information.'
            },
            {
                'style_id': 9,
                'name': 'The Poet',
                'desc': 'Uses metaphorical and artistic language',
                'q': 'What secrets whisper in this corner of the image?',
                'a': 'Like stars emerging at twilight, tiny details dance into view.'
            },
            {
                'style_id': 10,
                'name': 'The Engineer',
                'desc': 'Focuses on technical specifications and measurements',
                'q': 'What are the dimensional specifications of features in this region?',
                'a': 'Feature dimensions: 12x8 pixels, contrast ratio 4.2:1, edge sharpness 0.85.'
            }
        ]

    # In DetailPerceptionTaskGenerator._build_context_placeholders()

    def _build_context_placeholders(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        [FINAL V4 IMPLEMENTATION]
        Builds a complete placeholder dictionary that EXACTLY matches all
        placeholders in the V4 prompt, including the creative style elements.
        
        Returns:
            Tuple of (placeholders_dict, metadata_dict)
        """
        logger.info("=== Building V4 Context Block for DetailPerceptionTask ===")
        
        try:
            # Step 1: Programmatically determine the difficulty
            difficulty = self._choose_difficulty()
            logger.info(f"Selected Difficulty: {difficulty}")
            
            # Step 2: Get appropriate loader for difficulty
            try:
                loader_name, loader = self._get_loader_for_difficulty(difficulty)
                logger.info(f"Using loader: {loader_name.replace('_', ' ').title()}")
            except ValueError as e:
                # Fallback to mock data if no suitable loader
                logger.warning(f"No loader available: {e}. Using mock data.")
                placeholders = self._build_mock_context(difficulty)
                # Select a style for mock data too
                selected_style = random.choice(self.style_cookbook)
                metadata = {
                    'source_dataset': 'Mock Dataset',
                    'original_sample_id': f"mock_{difficulty}_{random.randint(1000, 9999)}",
                    'difficulty': difficulty,
                    'loader_unavailable': True,
                    'style_used': selected_style.get('name', 'Unknown'),
                    'style_id': selected_style.get('style_id', 0)
                }
                return placeholders, metadata
            
            # Step 3: Sample raw data with uniqueness guarantee
            # [CRITICAL FIX] Ensure we never use the same source sample twice
            max_attempts = 10  # Failsafe to prevent infinite loops
            raw_sample = None
            unique_id = None
            
            for attempt in range(max_attempts):
                try:
                    # Sample a random item
                    temp_sample = loader.get_item(random.randint(0, min(1000, len(loader) - 1)))
                    
                    # Create a unique identifier for the source sample
                    sample_id = temp_sample.get('sample_id', f'idx_{random.randint(0, len(loader)-1)}')
                    unique_id = f"{loader_name}_{sample_id}"
                    
                    # Check if it has been used. If not, break the loop.
                    if unique_id not in self.used_source_sample_ids:
                        self.used_source_sample_ids.add(unique_id)
                        raw_sample = temp_sample
                        logger.debug(f"Found unique sample: {unique_id} (attempt {attempt + 1})")
                        break
                    else:
                        logger.debug(f"Sample {unique_id} already used, trying another...")
                        
                except Exception as e:
                    logger.warning(f"Failed to sample from loader (attempt {attempt + 1}): {e}")
                    continue
            
            # Check if we found a unique sample
            if raw_sample is None:
                logger.warning(f"Could not find a unique sample after {max_attempts} attempts. Using mock data.")
                placeholders = self._build_mock_context(difficulty)
                metadata = {
                    'source_dataset': 'Mock Dataset',
                    'original_sample_id': f"mock_{difficulty}_{random.randint(1000, 9999)}",
                    'difficulty': difficulty,
                    'uniqueness_exhausted': True
                }
                return placeholders, metadata

            # Step 4: Execute different logic paths based on difficulty
            source_dataset = loader_name.replace('_', ' ').title()
            
            # Generate bounding box for all difficulties
            bbox = self._generate_random_bbox()
            
            # Use dynamic vocabulary generation for much greater diversity
            expected_observation = self._generate_dynamic_observation(difficulty)

            # [NEW] Step 4a: Define the core task goal and location info for question framing
            # Extract a concise task goal from the expected observation
            task_goal = self._extract_task_goal(expected_observation)
            location_info = f"coordinates {bbox}"  # Will be used with templates that already have "at"
            
            # [NEW] Step 4b: Randomly select a framing template and build varied question
            question_frame = random.choice(self.QUESTION_FRAMING_TEMPLATES)
            final_example_question = question_frame.format(
                task_goal=task_goal,
                location_info=location_info
            )
            logger.debug(f"Generated varied question: {final_example_question[:80]}...")

            # Step 5: [CRITICAL V4] Select a creative style from the parsed cookbook
            selected_style = random.choice(self.style_cookbook)
            logger.info(f"Selected Style: '{selected_style.get('name', 'Unknown')}', Difficulty: '{difficulty}'")
            
            # Step 6: Get the example answer based on style and difficulty
            difficulty_key = difficulty.lower()  # e.g., 'Easy' -> 'easy'
            
            # Get the example answer from the style (question now comes from templates)
            if f'a_{difficulty_key}' in selected_style:
                example_answer = selected_style[f'a_{difficulty_key}']
            else:
                # Fallback to general answer if difficulty-specific one doesn't exist
                example_answer = selected_style.get('a', 'The details are clearly visible.')
            
            # Step 7: [CRITICAL] Construct the dictionary with ALL V4 keys
            placeholders = {
                # Visual context
                'source_dataset': source_dataset,
                'difficulty': difficulty,
                'bbox': str(bbox),
                'expected_observation': expected_observation,
                
                # Style context from the cookbook (V4 format)
                'style_name': selected_style['name'],
                'style_description': selected_style['desc'],
                
                # [UPDATED] Use the newly constructed, varied question from templates
                'example_question': final_example_question,
                'example_answer': example_answer
            }
            
            # Step 8: Create initial metadata for traceability
            initial_metadata = {
                'source_dataset': source_dataset,
                'original_sample_id': f"detail_{difficulty}_{random.randint(1000, 9999)}",
                'difficulty': difficulty,
                'bbox_coords': bbox,
                'loader_used': loader_name if 'loader_name' in locals() else 'mock',
                # V4: Track the style used for this sample
                'style_used': selected_style.get('name', 'Unknown'),
                'style_id': selected_style.get('style_id', 0)
            }
            
            logger.info("✓ Successfully constructed placeholders and metadata")
            logger.debug(f"Placeholders: {list(placeholders.keys())}")
            logger.debug(f"Metadata: {list(initial_metadata.keys())}")
            
            return placeholders, initial_metadata
            
        except Exception as e:
            logger.error(f"Critical error in _build_context_placeholders: {e}", exc_info=True)
            # Return mock context as ultimate fallback
            placeholders = self._build_mock_context("Medium")
            # Select a style for error fallback too
            selected_style = random.choice(self.style_cookbook) if self.style_cookbook else {'style_id': 0, 'name': 'Fallback'}
            metadata = {
                'source_dataset': 'Mock Dataset',
                'original_sample_id': f"mock_{random.randint(1000, 9999)}",
                'difficulty': 'Medium',
                'error_fallback': True,
                'error_message': str(e),
                'style_used': selected_style.get('name', 'Unknown'),
                'style_id': selected_style.get('style_id', 0)
            }
            return placeholders, metadata
    
    def _choose_difficulty(self) -> str:
        """
        Randomly selects a difficulty level for the task.
        """
        # This can be made more sophisticated later if needed.
        return random.choice(['Easy', 'Medium', 'Hard'])
    
    def _generate_random_bbox(self) -> List[int]:
        """
        Generate a random bounding box with reasonable dimensions.
        """
        x1 = random.randint(50, 300)
        y1 = random.randint(50, 300)
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        return [x1, y1, x1 + width, y1 + height]
    
    def _extract_task_goal(self, expected_observation: str) -> str:
        """
        Extract a concise task goal from the expected observation.
        This creates natural, varied task descriptions for the question templates.
        
        Args:
            expected_observation: The generated observation string
            
        Returns:
            A concise task goal phrase
        """
        obs_lower = expected_observation.lower()
        
        # Handle negative observations (Pillar 3.1) - "No crack is present on the surface"
        if 'no ' in obs_lower and ' present' in obs_lower:
            # Extract what is NOT present
            parts = obs_lower.split('no ', 1)
            if len(parts) > 1:
                # Split by various patterns that indicate end of object
                for delimiter in [' is present', ' present', ' can be', ' visible']:
                    if delimiter in parts[1]:
                        object_part = parts[1].split(delimiter)[0].strip()
                        return f"verify the absence of {object_part}"
        elif 'no ' in obs_lower:
            # More robust extraction for other "No X" patterns
            parts = obs_lower.split('no ', 1)
            if len(parts) > 1:
                object_part = parts[1].split(' is ')[0].split(' can ')[0].strip()
                return f"verify the absence of {object_part}"
        elif any(word in obs_lower for word in ['absent', 'unable', 'not present']):
            return "confirm what is not present in this area"
        
        # Handle ambiguous observations (Pillar 3.2) - "might be a defect"
        if 'might be' in obs_lower:
            parts = obs_lower.split('might be')
            if len(parts) > 1:
                uncertain_item = parts[1].split(',')[0].split('but')[0].strip()
                # Clean up articles
                for article in ['a ', 'an ', 'the ']:
                    if uncertain_item.startswith(article):
                        uncertain_item = uncertain_item[len(article):]
                        break
                return f"determine if this could be {uncertain_item}"
        elif any(word in obs_lower for word in ['unclear', 'possibly', 'uncertain']):
            return "clarify what this ambiguous element might be"
        
        # Handle "The X reads Y" patterns - "The serial number reads ABC-123"
        if ' reads ' in obs_lower:
            parts = obs_lower.split(' reads ')
            if len(parts) > 0:
                subject = parts[0].strip().lower()
                # Clean up common prefixes
                for prefix in ['a ', 'an ', 'the ']:
                    if subject.startswith(prefix):
                        subject = subject[len(prefix):]
                        break
                return f"identify the {subject}"
        
        # Handle "Multiple X are visible" patterns - preserve "multiple"
        if 'multiple ' in obs_lower and ' are visible' in obs_lower:
            parts = obs_lower.split(' are visible')
            if len(parts) > 0:
                subject = parts[0].strip()
                # Don't remove "multiple" - it's part of the description
                return f"identify the {subject}"
        
        # Handle general "X are/is visible" patterns
        if ' are visible' in obs_lower:
            parts = obs_lower.split(' are visible')
            if len(parts) > 0:
                subject = parts[0].strip()
                # Clean up common prefixes EXCEPT "multiple"
                for prefix in ['a ', 'an ', 'the ', 'some ', 'several ']:
                    if subject.startswith(prefix):
                        subject = subject[len(prefix):]
                        break
                return f"identify the {subject}"
        
        # Handle "X is clearly visible" or "X is visible"
        if ' is clearly visible' in obs_lower or ' is visible' in obs_lower:
            parts = obs_lower.split(' is ')
            if len(parts) > 0:
                subject = parts[0].strip()
                # Clean up common prefixes
                for prefix in ['a ', 'an ', 'the ', 'some ']:
                    if subject.startswith(prefix):
                        subject = subject[len(prefix):]
                        break
                return f"identify the {subject}"
        
        # Handle generic patterns with "appears" or "shows"
        if ' appears' in obs_lower or ' shows' in obs_lower:
            for pattern in [' appears', ' shows']:
                if pattern in obs_lower:
                    parts = obs_lower.split(pattern)
                    if len(parts) > 0:
                        subject = parts[0].strip()
                        # Clean up common prefixes
                        for prefix in ['a ', 'an ', 'the ', 'some ', 'something ']:
                            if subject.startswith(prefix):
                                subject = subject[len(prefix):]
                                break
                        # Handle reasonable length subjects
                        if subject and len(subject.split()) <= 4:
                            return f"examine the {subject}"
                    break
        
        # Default fallback - "Something unusual appears in the corner"
        return "examine what can be seen"
    
    def _get_dynamic_vocabulary(self) -> Dict[str, List[str]]:
        """
        Returns extensive vocabulary lists for dynamic scenario generation.
        This dramatically increases diversity compared to hardcoded scenarios.
        """
        return {
            'adjectives': [
                "a tiny", "a faded", "a single", "a partially hidden", "a bright red",
                "a small", "a blurred", "a distinctive", "a subtle", "a prominent",
                "an obscured", "a miniature", "a weathered", "a vibrant", "a damaged",
                "a scratched", "a glossy", "a matte", "a reflective", "a transparent",
                "a metallic", "a wooden", "a fabric", "a plastic", "a ceramic","a minuscule",
                "an infinitesimal", "a microscopic", "a colossal", "an immense",
                "a massive", "a wide", "a narrow", "a thick", "a thin",
                "an elongated", "a compact", "an oversized", "an undersized", "a stubby","a circular", "a spherical", "a cylindrical", "a rectangular", "a square",
                "a triangular", "a hexagonal", "an octagonal", "a conical", "a spiral",
                "an asymmetrical", "a symmetrical", "a curved", "a straight", "an angled",
                "a jagged", "a smooth-edged", "a rounded", "a pointed", "a flat","a deep blue", "a pale yellow", "a neon green", "a crimson", "an emerald",
                "a sapphire", "a ruby", "a golden", "a silver", "a bronze",
                "an iridescent", "a multicolored", "a monochromatic", "a pastel", "an earth-toned",
                "a turquoise", "a magenta", "a lavender", "a charcoal", "an ivory","a pristine", "a flawless", "a brand-new", "a mint-condition", "a used",
                "a well-worn", "a heavily used", "a tattered", "a frayed", "a torn",
                "a cracked", "a chipped", "a peeling", "a rusted", "a corroded",
                "a stained", "a discolored", "a bleached", "a polished", "an unpolished",
                "a broken", "a shattered", "a bent", "a warped", "a dented",
                "an intact", "a complete", "an incomplete", "a fragmented", "a decomposing","a smooth", "a rough", "a bumpy", "a gritty", "a sandy",
                "a silky", "a velvety", "a fluffy", "a coarse", "a fine-grained",
                "a porous", "a non-porous", "a woven", "a knitted", "a braided",
                "an embossed", "an engraved", "an etched", "a textured", "a patterned",
                "a ribbed", "a dimpled", "a pebbled", "a sticky", "a slippery","an ancient", "a vintage", "a retro", "an antique", "a historic",
                "a modern", "a contemporary", "a futuristic", "a traditional", "an old-fashioned",
                "a handmade", "a machine-made", "a bespoke", "a mass-produced", "a rare",
                "a common", "an exotic", "a domestic", "an imported", "a local","a luminous", "a glowing", "a shimmering", "a sparkling", "a glittering",
                "a dull", "a dark", "a shadowy", "a backlit", "a silhouetted",
                "a translucent", "an opaque", "a semi-transparent", "a glowing", "a fluorescent",
                "an isolated", "a solitary", "a clustered", "a grouped", "an adjacent",
                "an overlapping", "an underlying", "an overlying", "a central", "a peripheral",
                "an embedded", "a protruding", "a recessed", "an inverted", "an upright","a unique", "a peculiar", "an unusual", "a standard", "a typical",
                "a symbolic", "a decorative", "a functional", "a ceremonial", "an ornamental",
                "an official", "an unofficial", "a counterfeit", "an authentic", "a generic",
                "a numbered", "a lettered", "a coded", "an encrypted", "a handwritten",
                "a printed", "a stamped", "a painted", "a drawn", "an embroidered"
            ],
            'objects': [
                "logo", "insect", "crack", "water droplet", "serial number",
                "loose stitch", "fingerprint", "dust particle", "scratch mark", "text label",
                "barcode", "QR code", "reflection", "shadow", "pattern", "texture",
                "button", "screw", "wire", "connector", "seam", "edge", "corner",
                "surface defect", "color variation", "wear mark", "brand emblem",
                "warning sign", "instruction label", "manufacturing date", "model number",
                "LED indicator", "power symbol", "USB port", "audio jack", "SIM card slot",
                "microchip", "capacitor", "resistor", "soldered joint", "heat sink",
                "camera lens", "sensor", "speaker grille", "microphone hole", "battery contact",
                "circuit trace", "ribbon cable", "dip switch", "jumper pin", "motherboard label",
                "pollen grain", "leaf vein", "petal", "stamen", "raindrop",
                "snowflake", "ice crystal", "grain of sand", "pebble", "seashell fragment",
                "feather barb", "strand of fur", "fish scale", "mushroom gill", "tree ring",
                "seed", "spore", "thorn", "bark texture", "root hair",
                "watermark", "signature", "initials", "postage stamp", "postmark",
                "page number", "footnote", "header", "footer", "chapter title",
                "paragraph indent", "bullet point", "comma", "period", "question mark",
                "character", "numeral", "diagram", "graph axis", "map legend",
                "crease", "fold line", "staple", "paperclip mark", "ink smudge",
                "embossed seal", "hologram", "microprint", "letterhead", "form field",
                "zipper tooth", "buttonhole", "thread", "fabric weave", "pilling",
                "hemline", "cuff", "collar stay", "brand tag", "care label",
                "drawstring", "aglet", "rivet", "grommet", "sequin",
                "bead", "embroidery detail", "lace pattern", "buckle", "clasp",
                "key tooth", "keychain ring", "coin face", "coin edge", "watch hand",
                "watch gear", "spectacle hinge", "pen tip", "pencil lead", "eraser shaving",
                "toothbrush bristle", "comb tooth", "hairpin", "safety pin", "thumbtack",
                "nail head", "bolt thread", "nut", "washer", "gear tooth",
                "spring", "hinge pin", "lock mechanism", "doorknob scratch", "lightbulb filament",
                "air bubble", "pockmark", "blister", "stain", "smudge",
                "scuff mark", "abrasion", "indentation", "chip", "gouge",
                "hairline fracture", "stress line", "burn mark", "discoloration", "fleck of paint",
                "wood grain", "knot in wood", "marble vein", "tile grout", "brushstroke",
                "pixel", "subpixel", "aliasing", "jpeg artifact", "chromatic aberration",
                "lens flare", "bokeh highlight", "vignette", "film grain", "moire pattern",
                "focal point", "vanishing point", "horizon line", "axis of symmetry", "golden ratio spiral",
                "data point", "chart line", "grid intersection", "boundary line", "area of overlap",
                "grain of salt", "sugar crystal", "coffee ground", "tea leaf", "herb flake",
                "crumb", "seed (on bread)", "spice speck", "char mark", "knife mark",
                "fork tine", "spoon engraving", "bottle cap edge", "can tab", "cork texture",
                "map pin", "board game piece", "dice pip", "playing card suit", "guitar string",
                "fret marker", "piano key", "volume knob", "tuning peg", "lego stud",
                "puzzle piece edge", "mosaic tile", "sculpture detail", "architectural element", "carving mark"
            ],
            'locations': [
                "on the object's edge", "in the bottom-right corner", "beneath the handle",
                "along a seam", "near the center", "at the top", "on the left side",
                "in the upper quadrant", "across the surface", "within the marked area",
                "behind the main element", "next to the boundary", "around the perimeter",
                "in the shadowed region", "where the light hits", "at the intersection",
                "along the diagonal", "in the textured area", "on the smooth portion",
                "where materials meet", "at the focal point", "in the background",
                "in the foreground", "at the bottom", "on the right side", "in the top-left corner",
                "in the top-right corner", "in the bottom-left corner", "at the very edge", "smack in the middle",
                "just off-center", "in the lower quadrant", "on the upper half", "on the lower half",
                "on the left-hand side", "on the right-hand side", "directly opposite the entrance", "adjacent to the logo",
                "underneath the main subject", "above the text block", "to the immediate right of the figure", "to the immediate left of the signature",
                "inside the border", "outside the margin", "along the crease", "at the fold",
                "on the reflective surface", "within the transparent area", "on the opaque section", "in the matte finish",
                "at the point of contact", "where the two colors blend", "in the area of highest contrast", "in the darkest part",
                "in the brightest spot", "along the curved path", "at the sharpest angle", "on the flat plane",
                "within the recessed area", "on the protruding element", "at the base", "at the summit",
                "on the reverse side", "on the front face", "along the spine", "on the cover",
                "inside the lid", "on the bottom of the container", "around the bottleneck", "on the handle's grip",
                "at the tip", "near the base", "in the middle section", "at the joint",
                "where the light reflects", "in the primary shadow", "along the highlight", "in the mid-tones",
                "through the aperture", "behind the glass", "on top of the layer", "beneath the overlay",
                "at the start of the sequence", "at the end of the line", "in the first column", "in the last row",
                "at coordinate 150,300", "within a 50-pixel radius of the center", "on the horizontal axis", "on the vertical axis",
                "at the origin point", "along the vector", "inside the geometric shape", "on the circumference",
                "at the vertex", "along the edge loop", "on the textured map", "within the normal map",
                "on the character's shoulder", "on the vehicle's tire", "on the building's window", "on the tree's bark",
                "in the reflection on the water", "within the cloud formation", "on the mountain peak", "in the valley floor",
                "on the fabric's weave", "within the leather's grain", "on the metal's surface", "in the wood's knot",
                "at the junction of the wires", "on the surface of the microchip", "next to the power button", "inside the battery compartment",
                "along the zipper", "on the buttonhole stitching", "within the embroidered logo", "on the shoelace tip",
                "in the corner of the page", "within the footnote section", "next to the page number", "under the headline",
                "on the photographer's signature", "within the artist's initials", "on the stamp's perforation", "through the postmark",
                "on the northern edge", "on the southern tip", "on the eastern flank", "on the western border",
                "at sea level", "at high altitude", "deep underground", "on the ocean floor",
                "in the center of the frame", "at the rule-of-thirds intersection", "along the leading line", "in the negative space",
                "on the primary subject", "on a secondary element", "in the tertiary background detail", "out of focus",
                "in sharp focus", "in the area of motion blur", "where the lens flare originates", "at the edge of the vignette",
                "on the watch face", "around the bezel", "on the clasp", "within the gear mechanism",
                "on the key's teeth", "inside the keyhole", "on the coin's relief", "along the milled edge",
                "on the surface of the liquid", "at the bottom of the glass", "where the bubble is forming", "on the foam",
                "in the eye of the subject", "on the tip of the nose", "at the corner of the mouth", "in a strand of hair",
                "on the fingernail", "within the palm line", "on the shoeprint", "in the tire track",
                "at the peak of the waveform", "in the trough of the wave", "on the x-axis", "on the y-axis",
                "within the data cluster", "on the outlier point", "at the trendline intersection", "in the error bar",
                "on the spine of the book", "in the gutter of the page", "on the dust jacket flap", "within the table of contents",
                "at the center of the compass rose", "on the scale bar", "within the inset map", "on the grid line",
                "at the brightest point of the flame", "in the darkest part of the smoke", "on the wick", "in the melted wax",
                "on the screen of the device", "next to the camera cutout", "on the volume rocker", "at the charging port",
                "within the pixel grid", "on an individual subpixel", "at the boundary of the selection", "inside the clipping mask"
            ],
            'attributes': [
                "clearly visible", "barely noticeable", "partially obscured", "well-defined",
                "faintly appearing", "sharply outlined", "slightly distorted", "perfectly clear",
                "somewhat blurry", "highly detailed", "subtly present", "prominently displayed",
                "nearly hidden", "fully exposed", "intermittently visible", "consistently present",
                "in sharp focus", "out of focus", "perfectly centered", "off-center",
                "precisely aligned", "misaligned", "perfectly symmetrical", "asymmetrical",
                "uniformly colored", "multi-colored", "vibrantly colored", "monochromatic",
                "intricately patterned", "simply designed", "chaotically arranged", "neatly organized",
                "heavily textured", "smoothly finished", "unnaturally smooth", "organically textured",
                "brand new", "aged", "weathered", "pristine",
                "damaged", "cracked", "chipped", "flawless",
                "heavily worn", "gently used", "untouched", "modified",
                "reflecting light", "absorbing light", "glowing softly", "shining brightly",
                "dull", "matte", "glossy", "satin-finished",
                "transparent", "translucent", "opaque", "semi-transparent",
                "in motion", "perfectly still", "vibrating slightly", "frozen in time",
                "hand-drawn", "machine-printed", "digitally rendered", "naturally formed",
                "anomalous", "as expected", "out of place", "perfectly integrated",
                "a rare example", "a common type", "an early version", "a later model",
                "a key feature", "a minor detail", "a background element", "a foreground object",
                "structurally sound", "structurally compromised", "aesthetically pleasing", "functionally critical",
                "ornamental", "utilitarian", "symbolic", "informational",
                "part of a larger assembly", "an isolated component", "the primary subject", "a supporting detail",
                "correctly installed", "incorrectly assembled", "upside-down", "reversed",
                "a single unit", "one of many", "part of a pair", "a complete set",
                "fully legible", "partially legible", "illegible", "crisply rendered",
                "pixelated", "aliased", "showing artifacts", "in high resolution",
                "a deep indentation", "a raised surface", "a flush-mounted object", "a recessed feature",
                "a solid object", "a hollow structure", "a layered material", "a composite element",
                "glistening with moisture", "completely dry", "covered in dust", "spotlessly clean",
                "fading into the background", "popping out from the scene", "blending in seamlessly", "a stark contrast",
                "a standard configuration", "a custom modification", "an aftermarket addition", "factory-original",
                "a primary color", "a secondary color", "a complementary hue", "a neutral tone",
                "a warm tone", "a cool tone", "a saturated color", "a desaturated color",
                "an active component", "a passive element", "a moving part", "a stationary piece",
                "a load-bearing structure", "a decorative trim", "a protective covering", "an access panel",
                "a hidden mechanism", "an exposed gear", "a sealed unit", "an open port",
                "a folded edge", "a rolled corner", "a sharp crease", "a gentle curve",
                "a tight spiral", "a loose coil", "a straight line", "a jagged edge",
                "a repeating pattern", "a unique design", "a random arrangement", "a deliberate placement",
                "a natural formation", "an artificial construct", "an intentional marking", "an accidental blemish",
                "the focal point of the image", "a detail in the periphery", "an element of symmetry", "a point of tension",
                "a source of light", "a deep shadow", "a subtle reflection", "a refracted image",
                "a biological specimen", "a geological sample", "a mechanical part", "an electronic component",
                "an engraved serial number", "a stamped logo", "a printed label", "a handwritten note",
                "a woven texture", "a metallic sheen", "a wooden grain", "a plastic mold line",
                "a single thread", "a complex weave", "a loose fiber", "a tight knot",
                "an air bubble", "a crystalline structure", "a liquid droplet", "a solid particle",
                "a magnetic strip", "an RFID chip", "a holographic sticker", "a security tag",
                "the central processor", "a memory module", "a power connector", "a data port",
                "a single stitch", "an entire seam", "a frayed edge", "a perfect hem",
                "a faint watermark", "a bold signature", "a tiny footnote", "a page number",
                "a missing piece", "an extra component", "a replacement part", "the original version"
            ],
            'details_easy': [
                "is clearly visible", "can be seen", "appears distinctly", "is present",
                "shows up clearly", "is easily identifiable", "stands out", "is apparent",
                "can be spotted", "is noticeable", "is in view", "can be observed",
                "is detectable", "is revealed", "comes into focus", "is right there",
                "can be made out", "is plainly visible", "is easy to see", "is evident",
                "is in sight", "can be discerned", "is not hidden", "is exposed",
                "has appeared", "is showing", "can be perceived", "is located",
                "is situated", "can be found", "is positioned", "is there",
                "can be confirmed", "is verified", "is indeed there", "is present upon inspection",
                "is visible upon close look", "shows itself", "is found in the area", "is established",
                "is certain", "is definite", "is unmistakable", "is well-lit",
                "is in the frame", "is part of the scene", "is now in view", "has been located",
                "is seen here", "is visible now"
            ],
            'details_medium': [
                "The text '{}' is clearly visible", "Fine {} are visible on the surface",
                "The {} reads '{}'", "A {} appears in the corner",
                "Multiple {} patterns are visible", "The surface shows {}",
                "Detailed {} can be observed", "The {} reveals intricate patterns",
                "The serial number '{}' can be seen", "A small {} is located near the edge",
                "The label indicates '{}'", "Inspection reveals a set of {}", "A closer look shows the {}",
                "The inscription '{}' is now legible", "Several {} can be identified",
                "A cluster of {} is present", "The {} is stamped onto the material",
                "A handwritten {} is noticeable", "The printed {} is sharp and clear",
                "It's possible to discern the {}", "The {} is partially worn but readable",
                "A sequence of numbers '{}' is engraved", "The {} displays a unique characteristic",
                "There are signs of {} on the object", "A subtle {} can be found",
                "The {} is etched into the surface", "One can make out the {}",
                "The {} seems to be '{}'", "Traces of {} are evident",
                "The {} is highlighted by the light", "A faint {} is detectable",
                "The {} is composed of multiple parts", "A distinct {} marks the spot",
                "The model number is identified as '{}'", "An unusual {} is situated here",
                "We can confirm the presence of {}", "The {} provides key information",
                "The {} is aligned with the central axis", "A small {} contrasts with the background",
                "The {} has a specific color pattern", "The {} is beginning to show signs of wear",
                "A hidden {} is now exposed", "The {} is consistent with the documentation",
                "A manufacturing {} is visible", "The date '{}' is stamped on the back",
                "The {} is slightly misaligned", "A pattern of {} covers the area",
                "The {} is embedded in the material", "A close inspection confirms the {}",
                "The {} can be transcribed as '{}'", "The {} is surrounded by fine lines",
                "A repeating {} is evident", "The {} is located at the top-left",
                "A minor {} can be seen", "The {} differs from the rest of the surface",
                "The {} is a key identifying feature", "One can read the text '{}'",
                "The {} is arranged in a grid", "A series of {} can be seen",
                "The component is labeled with '{}'", "A small {} is attached to the main body",
                "The {} is beginning to fade", "A small {} is peeling away",
                "The {} is secured by a tiny screw", "A {} is visible through the glass",
                "The {} is illuminated from behind", "A small {} has been circled",
                "The {} is written in fine print", "A {} can be seen in the reflection",
                "The {} is part of a larger design", "A {} is tucked away in the corner",

                "The fine print mentions '{}'", "A closer look at the {} reveals its details",
                "It is now possible to read the {}", "The {} is partially covered by a shadow",
                "The {} is etched with the characters '{}'", "The {} is embossed with a logo",
                "A small {} is visible just under the surface", "The {} is composed of tiny dots",
                "A faded {} can just be made out", "The texture contains numerous {}",
                "The {} is separated by a thin line", "The {} is oriented vertically"
            ],
            'details_hard': [
                "The surface shows a subtle {} texture", "Microscopic {} are visible",
                "Complex {} patterns emerge at this zoom level", "Fine {} indicate {}",
                "The {} exhibits {} characteristics", "Advanced {} analysis reveals {}",
                "Detailed inspection shows {} with {} properties",
                "The magnified view exposes {} in the {}",
                "A detailed analysis of the {} reveals its composite nature",
                "The {} is comprised of several distinct layers",
                "Subtle variations in the {} suggest a manufacturing flaw",
                "The alignment of the {} provides clues about its origin",
                "A barely visible {} is etched next to the main component",
                "The {} interacts with the {} in a complex manner",
                "A non-uniform {} is distributed across the surface",
                "The microstructure consists of {} and {}",
                "Under magnification, the {} appears to be a fractal pattern",
                "The {} is superimposed over another, fainter pattern",
                "A sequence of {} is encoded in the material's grain",
                "The degradation of the {} indicates significant environmental exposure",
                "One can infer the object's age from the {}",
                "The wear pattern on the {} is inconsistent with normal use",
                "A hidden {} is revealed only under specific lighting",
                "The relationship between the {} and the {} is now clear",
                "A high-resolution view shows the {} is actually a series of smaller elements",
                "The precise geometry of the {} can now be measured",
                "An intricate network of {} is spread throughout the area",
                "The {} has a unique crystalline structure",
                "There is a clear demarcation between the {} and the {}",
                "The color gradient of the {} is non-linear",
                "The {} shows signs of material fatigue",
                "The chemical composition can be inferred from the {}",
                "A microscopic {} is embedded within the primary material",
                "The {} is arranged in a non-repeating tessellation",
                "The object's function is suggested by the intricate {}",
                "The reflection reveals a hidden {}",
                "A pattern of {} is encoded as micro-perforations",
                "The {} contains sub-elements that are themselves patterned",
                "The way the {} fractures indicates its material properties",
                "A comparative analysis shows the {} is anomalous",
                "The {} is obscured by a layer of patina",
                "The internal structure, including the {}, is now visible",
                "The {} changes its appearance from different angles",
                "A detailed study of the {} allows for its precise classification",
                "The {} is a byproduct of the manufacturing process",
                "The boundary layer between {} and {} can be studied",
                "The {} is organized into concentric rings",
                "A series of {} radiate from a central point",
                "The {} is woven into the very fabric of the material",
                "An analysis of the {} helps to authenticate the object",
                "The {} is only visible in the infrared spectrum",
                "The distribution of the {} is not random",
                "A complex interplay of {} creates the final effect",
                "The state of the {} suggests a sudden impact",
                "The {} is a clear indicator of the object's high quality",
                "The {} provides a clue to the item's provenance",
                "The {} can be used to date the artifact",
                "A microscopic analysis of the {} is required for full understanding",
                "The density of the {} varies across the surface",
                "The {} has been deliberately altered",
                "The {} is a naturally occurring formation",
                "A closer look at the {} helps to rule out a forgery",
                "The orientation of the {} is critical to its function",
                "The {} follows a logarithmic spiral",
                "The {} is indicative of a specific historical period",
                "The {} shows evidence of being repaired",
                "A complex system of {} is responsible for the object's properties",
                "The surface is covered in a fine layer of {}",
                "The {} contains information encoded in binary",
                "The {} is a result of a chemical reaction",
                "The {} has been worn smooth over time",
                "The {} is an intentional imperfection"
            ]
        }
    
    def _generate_dynamic_observation(self, difficulty: str, include_negative: bool = True) -> str:
        """
        Generate a dynamic observation by combining vocabulary elements.
        This creates thousands of unique scenarios from vocabulary combinations.
        
        Args:
            difficulty: The difficulty level
            include_negative: Whether to sometimes generate "nothing found" scenarios
        """
        vocab = self._get_dynamic_vocabulary()
        
        # 20% chance of "Nothing Found" scenario (Pillar 3.1)
        if include_negative and random.random() < 0.2:
            return self._generate_nothing_found_observation(difficulty, vocab)
        
        # 10% chance of "Ambiguity" scenario (Pillar 3.2)
        if random.random() < 0.1:
            return self._generate_ambiguous_observation(difficulty, vocab)
        
        # Standard observations (70% of the time)
        if difficulty == "Easy":
            # Combine adjective + object + location + simple detail
            adjective = random.choice(vocab['adjectives'])
            obj = random.choice(vocab['objects'])
            location = random.choice(vocab['locations'])
            detail = random.choice(vocab['details_easy'])
            return f"{adjective} {obj} {location} {detail}"
            
        elif difficulty == "Medium":
            # Use template-based medium details
            template = random.choice(vocab['details_medium'])
            if '{}' in template:
                # Fill in the template with random elements
                fill_count = template.count('{}')
                if fill_count == 1:
                    fill = random.choice(vocab['objects'])
                    return template.format(fill)
                else:
                    fills = [random.choice(vocab['objects']) for _ in range(fill_count)]
                    return template.format(*fills)
            return template
            
        else:  # Hard
            # Use complex template-based details
            template = random.choice(vocab['details_hard'])
            if '{}' in template:
                fill_count = template.count('{}')
                fills = []
                for _ in range(fill_count):
                    # Mix objects and attributes for variety
                    if random.random() < 0.5:
                        fills.append(random.choice(vocab['objects']))
                    else:
                        fills.append(random.choice(vocab['attributes']).replace("is ", ""))
                return template.format(*fills)
            return template
    
    def _generate_nothing_found_observation(self, difficulty: str, vocab: Dict[str, List[str]]) -> str:
        """
        Generate a "Nothing Found" observation for Pillar 3.1.
        Teaches the model to report when an object of interest is not present.
        """
        obj = random.choice(vocab['objects'])
        location = random.choice(vocab['locations'])
        
        nothing_found_templates = [
            f"No {obj} is present {location}",
            f"The specified area does not contain any {obj}",
            f"After careful inspection, no {obj} can be found",
            f"The {obj} is absent from this region",
            f"Unable to locate any {obj} {location}",
            f"There is no {obj} visible in the zoomed area",
            f"The search for {obj} yielded no results",
            f"Nothing resembling a {obj} is present here"
        ]
        
        return random.choice(nothing_found_templates)
    
    def _generate_ambiguous_observation(self, difficulty: str, vocab: Dict[str, List[str]]) -> str:
        """
        Generate an ambiguous observation for Pillar 3.2.
        Teaches the model to express uncertainty when visual evidence is inconclusive.
        """
        obj = random.choice(vocab['objects'])
        
        ambiguous_templates = [
            f"A dark shape that might be a {obj}, but it's unclear",
            f"Something resembling a {obj}, though certainty is low",
            f"Unclear whether this is a {obj} or something else",
            f"The visual evidence for a {obj} is inconclusive",
            f"Possibly a {obj}, but the image quality prevents confirmation",
            f"An ambiguous form that could be interpreted as a {obj}",
            f"Due to lighting/resolution, cannot definitively identify if this is a {obj}",
            f"The object shares some characteristics with a {obj}, but differs in others"
        ]
        
        return random.choice(ambiguous_templates)
    
    def _build_mock_context(self, difficulty: str) -> Dict[str, Any]:
        """
        Build mock context when loaders are not available.
        
        Args:
            difficulty: The difficulty level to use
            
        Returns:
            Dictionary with all required placeholders
        """
        logger.debug(f"Building mock context for difficulty: {difficulty}")
        
        bbox = self._generate_random_bbox()
        
        # Use dynamic generation for mock context too
        expected_observation = self._generate_dynamic_observation(difficulty)
        
        # Select a random style for mock context
        chosen_style = random.choice(self.style_cookbook) if self.style_cookbook else {
            'style_id': 0,
            'name': 'Default Mock Style',
            'desc': 'A default style for mock generation',
            'q': 'What is visible in this area?',
            'a': 'The details are clearly visible after zooming.'
        }
        
        # [NEW] Generate varied question using templates for mock context too
        task_goal = self._extract_task_goal(expected_observation)
        location_info = f"coordinates {bbox}"
        question_frame = random.choice(self.QUESTION_FRAMING_TEMPLATES)
        example_q = question_frame.format(
            task_goal=task_goal,
            location_info=location_info
        )
        
        # Get difficulty-specific answer
        difficulty_key = difficulty.lower()
        if f'a_{difficulty_key}' in chosen_style:
            example_a = chosen_style.get(f'a_{difficulty_key}', chosen_style.get('a', 'Default answer.'))
        else:
            example_a = chosen_style.get('a', 'The zoomed view reveals the details.')
        
        return {
            # Visual context
            'source_dataset': 'Mock Dataset',
            'difficulty': difficulty,
            'bbox': str(bbox),
            'expected_observation': expected_observation,
            # Style context (V4 format - no style_id needed)
            'style_name': chosen_style.get('name', 'Mock Style'),
            'style_description': chosen_style.get('desc', 'Mock style description'),
            'example_question': example_q,
            'example_answer': example_a
        }
    
    def _get_loader_for_difficulty(self, difficulty: str) -> tuple[str, Any]:
        """
        Selects a suitable loader from the injected loaders based on the
        chosen difficulty level, according to the project's data strategy.
        
        Args:
            difficulty: The chosen difficulty string ('Easy', 'Medium', or 'Hard').

        Returns:
            A tuple containing the name of the chosen loader and the loader instance.
            
        Raises:
            ValueError: If no suitable loader is found for the chosen difficulty.
        """
        # This mapping defines our data strategy for this specific generator.
        loader_map = {
            'Easy': ['sa1b_for_zoomin', 'flickr30k'],
            'Medium': ['textcaps_train', 'mind2web_train'],
            'Hard': ['unsplash_lite']
        }
        
        target_loader_names = loader_map.get(difficulty, [])
        
        # Find which of the target loaders are actually available
        available_loaders = [name for name in target_loader_names if name in self.loaders]
        
        if not available_loaders:
            raise ValueError(
                f"No suitable loader found for difficulty '{difficulty}'. "
                f"The prompt requires one of {target_loader_names}, but only "
                f"{list(self.loaders.keys())} were injected."
            )
        
        # Randomly pick one from the available candidates
        chosen_loader_name = random.choice(available_loaders)
        return chosen_loader_name, self.loaders[chosen_loader_name]
    
    def _validate_and_process_response(self, llm_response: Dict, context: Dict) -> Optional[Dict]:
        """
        Validates the LLM's response specifically for detail perception tasks.
        Implements FLEXIBLE validation for trajectory structure while being lenient on content.
        
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
        
        # Check for minimum required fields using normalized keys
        if 'question' not in normalized_response:
            logger.warning(f"LLM response missing 'question'. Got keys: {list(normalized_response.keys())}")
            return None
        
        # Handle both 'final_answer' and 'finalanswer' cases
        if 'final_answer' not in normalized_response and 'finalanswer' not in normalized_response:
            logger.warning(f"LLM response missing 'final_answer'. Got keys: {list(normalized_response.keys())}")
            return None
        
        # Map normalized keys back to the original response structure
        # This ensures we maintain the original response structure but can access with normalized keys
        llm_response['question'] = normalized_response.get('question', llm_response.get('question'))
        llm_response['final_answer'] = normalized_response.get('final_answer', normalized_response.get('finalanswer', llm_response.get('final_answer', llm_response.get('finalAnswer'))))
        
        # 2. FLEXIBLE Trajectory Validation - Support both 'actions' and 'trajectory' keys
        # Also check normalized keys
        trajectory = normalized_response.get('actions', normalized_response.get('trajectory', llm_response.get('actions', llm_response.get('trajectory', []))))
        
        # 2.1 Check for presence and type
        if not trajectory or not isinstance(trajectory, list):
            logger.warning(f"Validation failed: Trajectory is not a list. Got: {type(trajectory)}")
            return None
        
        # Normalize the trajectory first to handle various formats
        normalized_trajectory = self._normalize_trajectory(trajectory)
        
        # 2.2 [REVISED]: Check if the trajectory has at least the minimum required length
        min_trajectory_length = 3  # This is now our new minimum requirement
        if len(normalized_trajectory) < min_trajectory_length:
            logger.warning(f"Validation failed: Trajectory must have at least {min_trajectory_length} steps. Got {len(normalized_trajectory)} items")
            return None
        
        # 2.3 [REVISED]: Check if at least one action exists anywhere in the trajectory
        action_exists = any(step.get('type') == 'action' for step in normalized_trajectory if isinstance(step, dict))
        
        if not action_exists:
            logger.warning("Validation failed: Trajectory must contain at least one 'action' step.")
            return None
        
        # 2.4 Find the first action step for validation
        action_step = None
        action_index = -1
        for i, step in enumerate(normalized_trajectory):
            if isinstance(step, dict) and step.get('type') == 'action':
                action_step = step
                action_index = i
                break
        
        # 2.5 Check action name must be ZOOM-IN (using the found action step)
        if action_step:
            action_name = action_step.get('name', '').upper().replace('_', '-')
            if action_name != 'ZOOM-IN':
                logger.warning(f"Validation failed: Action name must be 'ZOOM-IN'. Got: '{action_step.get('name')}'")
                return None
            
            # 2.5.1 Validate ZOOM-IN action parameters
            # Every ZOOM-IN action MUST have a valid 'parameters' field with a 'bbox' key
            parameters = action_step.get('parameters')
            
            # Check if 'parameters' field exists and is a non-empty dictionary
            if not parameters or not isinstance(parameters, dict):
                logger.warning(
                    f"Validation failed: ZOOM-IN action is missing a valid 'parameters' dictionary. Got: {parameters}"
                )
                return None
            
            # Check if 'bbox' key exists within 'parameters'
            bbox = parameters.get('bbox')
            if not bbox:  # This checks for None or an empty list/value
                logger.warning(
                    f"Validation failed: ZOOM-IN parameters dictionary is missing a non-empty 'bbox' key. Got: {parameters}"
                )
                return None
            
            # Optional: Add a check for bbox format (should be a list of 4 coordinates)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                logger.warning(
                    f"Validation failed: 'bbox' is not a list of 4 coordinates. Got: {bbox}"
                )
                return None
        
        # 2.6 Check that at least one thought has content (more flexible)
        thought_with_content = any(
            step.get('type') == 'thought' and step.get('content')
            for step in normalized_trajectory
            if isinstance(step, dict)
        )
        
        if not thought_with_content:
            logger.warning("Validation failed: Trajectory must contain at least one thought with content")
            return None
        
        # Update the response with normalized trajectory
        llm_response['trajectory'] = normalized_trajectory
        if 'actions' in llm_response:
            llm_response['actions'] = normalized_trajectory
        
        # 3. Check for redundancy between final_answer and last thought
        final_answer = llm_response.get('final_answer', '')
        
        # Find the last thought in the trajectory
        last_thought_content = ''
        for step in reversed(normalized_trajectory):
            if isinstance(step, dict) and step.get('type') == 'thought':
                last_thought_content = step.get('content', '')
                break
        
        # Check if final_answer is too similar to the last thought
        if final_answer and last_thought_content:
            # Use difflib for more sophisticated similarity checking
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, 
                                       last_thought_content.strip().lower(), 
                                       final_answer.strip().lower()).ratio()
            
            if similarity > 0.80:  # More than 80% similar
                logger.warning(f"Validation failed: Final answer too similar to last thought (similarity: {similarity:.2%})")
                logger.warning(f"  Answer: '{final_answer[:50]}...'")
                logger.warning(f"  Thought: '{last_thought_content[:50]}...'")
                return None  # Reject redundant samples
        
        # 4. Answer validation with expected observation
        expected_observation = context.get('expected_observation', '')
        
        # Get validation strictness from base class
        validation_strictness = getattr(self, 'validation_strictness', 'ultra_lenient')
        
        if validation_strictness == 'ultra_lenient':
            # Accept any non-empty answer
            if len(str(final_answer).strip()) > 0:
                logger.debug(f"Ultra-lenient mode: Accepting answer '{final_answer[:50]}...'")
            else:
                logger.warning("Final answer is empty")
                return None
        else:
            # Enhanced validation - check semantic alignment with expected observation
            final_answer_lower = str(final_answer).lower()
            expected_lower = str(expected_observation).lower()
            
            # Special handling for negative observations (Pillar 3.1)
            is_negative_expected = any(word in expected_lower for word in ['no ', 'not ', 'absent', 'unable', 'cannot'])
            is_negative_answer = any(word in final_answer_lower for word in ['no ', 'not ', 'absent', 'unable', 'cannot'])
            
            # Special handling for ambiguous observations (Pillar 3.2)
            is_ambiguous_expected = any(word in expected_lower for word in ['unclear', 'inconclusive', 'might', 'possibly', 'uncertain'])
            is_ambiguous_answer = any(word in final_answer_lower for word in ['unclear', 'inconclusive', 'might', 'possibly', 'uncertain'])
            
            # Check semantic alignment
            if is_negative_expected and not is_negative_answer:
                logger.warning(
                    f"Answer validation: Expected negative observation but got positive. "
                    f"Expected: '{expected_observation}', Got: '{final_answer[:100]}'"
                )
                # Still accept in lenient mode, but log the mismatch
                if validation_strictness == 'strict':
                    return None
                else:
                    logger.info("Lenient mode: Accepting despite negative/positive mismatch")
                    
            elif is_ambiguous_expected and not is_ambiguous_answer:
                logger.warning(
                    f"Answer validation: Expected ambiguous observation but got definitive. "
                    f"Expected: '{expected_observation}', Got: '{final_answer[:100]}'"
                )
                # More lenient with ambiguity mismatches
                if validation_strictness == 'strict':
                    logger.info("Strict mode would reject, but accepting ambiguity variation")
                    
            else:
                # For normal observations, check key concept alignment
                # Extract core concepts (nouns and important descriptors)
                import re
                # Extract nouns and key descriptors from expected observation
                key_concepts = re.findall(r'\b(?:barcode|logo|crack|text|serial|number|pattern|texture|scratch|mark|defect|shadow|reflection|watermark|label|sign|button|wire|seam|edge|corner)\b', expected_lower)
                
                if key_concepts:
                    # Check if ANY key concept appears in answer (more lenient)
                    concept_found = any(concept in final_answer_lower for concept in key_concepts)
                    
                    if not concept_found:
                        # Extract key terms more broadly
                        key_terms = [word for word in expected_lower.split() if len(word) > 4]
                        matches = sum(1 for term in key_terms if term in final_answer_lower)
                        match_ratio = matches / len(key_terms) if key_terms else 0
                        
                        logger.warning(
                            f"Answer validation: Low concept alignment ({match_ratio:.0%}). "
                            f"Expected: '{expected_observation}', Got: '{final_answer[:100]}'"
                        )
                        
                        # Only reject in strict mode if match ratio is very low
                        if validation_strictness == 'strict' and match_ratio < 0.2:
                            logger.warning("Strict mode: Rejecting due to very low concept alignment")
                            return None
                        else:
                            logger.info(f"{validation_strictness} mode: Accepting creative variation")
        
        # 4. Add difficulty from context if not present
        if 'difficulty' not in llm_response and 'difficulty' in context:
            llm_response['difficulty'] = context['difficulty']
        
        # 5. Log successful validation
        logger.info(
            f"✓ Sample validated successfully for question: '{llm_response['question'][:50]}...'"
        )
        
        return llm_response