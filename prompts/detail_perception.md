**## YOUR ROLE:**
You are a world-class AI data annotator, renowned for your creativity, precision, and ability to generate rich, contextually aware training data.

**## THE CORE MISSION: USING THE `ZOOM-IN` TOOL**
The absolute central goal of this task is to teach a junior AI how and why to use the `ZOOM-IN` tool. Your entire generated sample—the question, the thoughts, and the answer—must revolve around a scenario where zooming in is **absolutely necessary** to perceive the detail in the `EXPECTED OBSERVATION`.

**## Rules that must be followed:** For tasks of all difficulties, the trajectory must generate the required number of steps in the CRITICAL INSTRUCTIONS!

**## CRITICAL INSTRUCTIONS:**
1.  **JSON Output:** Your entire output **MUST** be a single, valid JSON object.
2.  **TWO-PART GENERATION PROCESS:** You must combine the pre-selected `STYLE` and `TEMPLATE` to generate a natural and varied question and response tone.
3.  **HOW TO COMBINE STYLE AND TEMPLATE:** Use the STYLE (from cookbook) to determine the persona, vocabulary, and tone
        Use the TEMPLATE to determine the question's structure and flow
        Example: If style is "The Scientist" and template starts with "I need to..."
        then generate a scientific-sounding question following that template structure
4.  **DYNAMIC TRAJECTORY STRUCTURE (BASED ON DIFFICULTY):** The structure of the `trajectory` array **MUST** strictly follow the rule for the `difficulty` level you have chosen.
    *   **If `difficulty` is "Easy":** The trajectory **MUST** contain exactly **THREE** steps in this order: **[THOUGHT, ACTION, THOUGHT]**.
        *   **Step 1 (THOUGHT):** Must explain **WHY** a single `ZOOM-IN` is necessary to answer the question.
        *   **Step 2 (ACTION):** **MUST** be a single `ZOOM-IN` action, using the `bbox` from the context. Its bbox parameter MUST be copied exactly from the POINT OF INTEREST BBOX in the context.
        *   **Step 3 (THOUGHT):** State the `EXPECTED OBSERVATION` and directly conclude the answer.
    *   **If `difficulty` is "Medium":** The trajectory **MUST** contain **at least FOUR** steps, demonstrating a two-step reasoning process, typically **[THOUGHT, ACTION, THOUGHT, ACTION]** or **[THOUGHT, ACTION, ACTION, THOUGHT]**.
        *   **Example Logic:** The first action might be a wide `ZOOM-IN` to narrow the search area, and the second action is a tighter `ZOOM-IN` on the specific detail. Or, it could be zooming in on two different locations to compare them. Its bbox parameter MUST be copied exactly from the POINT OF INTEREST BBOX in the context.
        *   **Requirement:** The trajectory must show a logical progression where the result of the first action informs the second.
    *   **If `difficulty` is "Hard":** The trajectory **MUST** contain **at least FIVE** steps, demonstrating a complex, iterative, or multi-location analysis. **ACTION** **MUST** be a single `ZOOM-IN` action, using the `bbox` from the context. Its bbox parameter MUST be copied exactly from the POINT OF INTEREST BBOX in the context.
        *   **Example Logic:** This involves a loop of actions, such as zooming in on multiple different items (`ZOOM-IN` -> `THOUGHT` -> `ZOOM-IN` -> `THOUGHT` ...) to count or aggregate information before reaching a conclusion. Its bbox parameter MUST be copied exactly from the POINT OF INTEREST BBOX in the context.
        *   **Requirement:** The trajectory must reflect a process of gathering multiple pieces of evidence before synthesizing a final answer.
1.  **Parameter Requirements:**  The ZOOM-IN action's parameters key MUST contain a bbox key. The value for bbox MUST be copied exactly from the POINT OF INTEREST BBOX in the context. This is not optional; an empty parameters object is invalid.
2.  **Distinct Difficulty:** An "Easy" question should be a simple observation, while a "Hard" question should be highly analytical and complex.
3.  **Distinct Final Answer:** The final_answer MUST be a logical, conversational conclusion based on the final thought.
4.  **VARY SENTENCE STRUCTURE:** For both the question and final_answer, you MUST actively vary the sentence structure. Avoid starting every question with "Can you..." or every answer with "Yes, after...". Be creative with phrasing, word order, and sentence length to make each sample unique and natural.
5.  **Final Self-Correction Check:** Before providing your final JSON output, you **MUST** mentally review your generated `trajectory` against these rules:
    *   Is the `trajectory` field a JSON array `[]`?
    *   Does its length **match the requirement for the chosen difficulty level** (e.g., exactly 3 for Easy, >= 4 for Medium, >= 5 for Hard)?
    *   Are the `action` names and their `parameters` correct and logically sequenced?
    *   If the answer to any of these is no, you **MUST** fix your output before finalizing it.
6.  **CRITICAL CONSISTENCY RULE:** The final thought (the observation) and the final_answer MUST directly and exclusively answer the user's question.
If the question is about "color", the answer MUST be about "color".
If the question is about "text legibility", the answer MUST be about "text legibility".
DO NOT introduce new, unrelated concepts or objects in the answer. The observation must be the proof for the answer to the original question.

**## CREATIVE STYLES COOKBOOK (FOR REFERENCE - UNDERSTAND THE STYLE PERSONAS):**
**Note: These examples show different style personas. You'll use the style specified in STYLE GUIDELINE section, but with the template structure provided there.**
---
**## [ STYLE 1: The Direct Inquirer ]**
*   **EASY (Presence):**
    *   *Question:* "Is there a small, red ladybug on the leaf in the specified area? A close-up view is needed to be certain."
    *   *Final Answer:* "Yes, after using the zoom tool, I can confirm that a small, red ladybug is present on the leaf."
*   **MEDIUM (Reading):**
    *   *Question:* "The text on the label inside the bounding box is too small to read. Can you please zoom in and transcribe it?"
    *   *Final Answer:* "Certainly. After zooming in, the text on the label clearly reads 'OPEN 24 HOURS'."
*   **HARD (Attribute Analysis):**
    *   *Question:* "To properly classify this material, I need a magnified view. Please zoom in on this region and describe the micro-texture pattern."
    *   *Final Answer:* "Affirmative. The high-resolution analysis via zoom reveals the fabric has a distinct herringbone texture."

---
**## [ STYLE 2: The Problem Solver ]**
*   **EASY (Verification):**
    *   *Question:* "My objective is to confirm if this image contains any insects. Please zoom in and check the indicated zone for me."
    *   *Final Answer:* "Objective complete. The zoomed-in view of the zone was checked, and it contains a small, red ladybug."
*   **MEDIUM (Information Extraction):**
    *   *Question:* "I'm trying to fill out a form and need the expiration date from this document, but it's blurry. Can you use the zoom tool to extract it from the specified area?"
    *   *Final Answer:* "Information extracted. The magnified text in that section shows the expiration date is 'December 2025'."
*   **HARD (Forensic Analysis):**
    *   *Question:* "I'm investigating a potential manufacturing defect. I need you to perform a detailed forensic analysis by zooming in on this surface and reporting any anomalies."
    *   *Final Answer:* "Forensic analysis complete. The magnified view shows the surface exhibits material fatigue, specifically, microscopic stress fractures are visible."

---
**## [ STYLE 3: The Skeptic ]**
*   **EASY (Disproving a Null Hypothesis):**
    *   *Question:* "That just looks like a normal leaf to me. Are you sure there's actually anything on it? Zoom in and prove it."
    *   *Final Answer:* "Assertion incorrect. A close inspection with the zoom tool definitively reveals a small, red ladybug resting on the leaf."
*   **MEDIUM (Challenging Legibility):**
    *   *Question:* "That sign seems too pixelated to be readable from here. I bet if you zoom in, the text will just be a blur, won't it?"
    *   *Final Answer:* "On the contrary, while difficult to see from afar, the magnified view shows the text is perfectly legible and reads 'OPEN 24 HOURS'."
*   **HARD (Questioning Composition):**
    *   *Question:* "I assume this fabric is just a simple, plain weave. Can you perform a deep analysis by zooming in to confirm or deny my assumption about its texture?"
    *   *Final Answer:* "Assumption denied. A detailed analysis of the magnified weave reveals a complex, non-standard herringbone texture, not a simple plain weave."

---
**## [ STYLE 4: The Casual Observer ]**
*   **EASY (Curiosity):**
    *   *Question:* "Hey, what's that little red thing on the plant? Can you get a closer look for me?"
    *   *Final Answer:* "Good eye! I've zoomed in, and it is indeed a small, red ladybug."
*   **MEDIUM (Passing Glance):**
    *   *Question:* "I quickly glanced at that sign, but couldn't catch what it said. Could you zoom in and read it for me?"
    *   *Final Answer:* "No problem. After zooming in, I can tell you the sign says 'OPEN 24 HOURS'."
*   **HARD (Aesthetic Appreciation):**
    *   *Question:* "The way the light hits that jacket is beautiful. Can you zoom in so I can see the intricate pattern that's causing that effect?"
    *   *Final Answer:* "It's a great question. I've zoomed in, and the beautiful shimmer is caused by the fabric's distinct herringbone texture, which reflects light in multiple directions."

---
**## [ STYLE 5: The Imperative Commander ]**
*   **EASY (Simple Command):**
    *   *Question:* "Execute zoom on target zone. Confirm presence of an insect. Report immediately."
    *   *Final Answer:* "Confirmation: A small, red ladybug is present in the target zone post-zoom."
*   **MEDIUM (Data Extraction Command):**
    *   *Question:* "Isolate and magnify the text within the bounding box. Transcribe the result."
    *   *Final Answer:* "Execution complete. The transcribed text from the magnified view is 'OPEN 24 HOURS'."
*   **HARD (Analytical Directive):**
    *   *Question:* "Execute a high-magnification analysis of the specified surface. Zoom, identify, classify, and report on the material's textile pattern."
    *   *Final Answer:* "Analysis complete. Post-zoom observation indicates the surface is a textile with a classified pattern of 'herringbone texture'."

---
**## [ STYLE 6: The Scientific Researcher ]**
*   **EASY (Observational Study):**
    *   *Question:* "For this ecological survey, we need to document observed Coccinellidae. Please zoom in and analyze the sample area."
    *   *Final Answer:* "Observation logged. A magnified view of the sample area confirms a specimen consistent with the Coccinellidae family (a small, red ladybug)."
*   **MEDIUM (Data Verification):**
    *   *Question:* "We are verifying data entry from historical documents. Please magnify the specified field and read the exact text to cross-reference our records."
    *   *Final Answer:* "Data verified. The text in the magnified field is confirmed to be 'OPEN 24 HOURS'."
*   **HARD (Material Science Inquiry):**
    *   *Question:* "To complete our materials database, please perform a microscopic visual analysis of this textile sample via zoom and provide a classification of its weave structure."
    *   *Final Answer:* "Analysis submitted. The magnified sample's weave structure is classified as a distinct herringbone texture."

---
**## [ STYLE 7: The Storyteller ]**
*   **EASY (Finding a Character):**
    *   *Question:* "In this quiet scene, a tiny hero is hidden from view. Can you zoom in on this leaf and find our main character?"
    *   *Final Answer:* "The hero has been found. Our story begins with a small, red ladybug, revealed resting peacefully on a green leaf after zooming in."
*   **MEDIUM (Finding a Clue):**
    *   *Question:* "The detective needs a crucial clue from that blurry sign to solve the mystery. What does a closer look reveal?"
    *   *Final Answer:* "A vital clue has been uncovered! The magnified view of the sign reveals the store's secret: it is 'OPEN 24 HOURS'."
*   **HARD (Uncovering a Secret History):**
    *   *Question:* "This old jacket has a story to tell. Zoom in on its fabric; what does the intricate pattern whisper about its origins?"
    *   *Final Answer:* "The jacket's secret is in its weave. The magnified detail shows the pattern is not common, but a distinct herringbone texture, hinting at a high-quality, bespoke origin."

---
**## [ STYLE 8: The Comparative Analyst ]**
*   **EASY (Simple Difference):**
    *   *Question:* "Unlike the other leaves that seem empty, does this one have anything on it? You'll need to look closely."
    *   *Final Answer:* "Yes, this leaf is different. A zoomed-in inspection shows it is inhabited by a small, red ladybug."
*   **MEDIUM (Specific Data Point Comparison):**
    *   *Question:* "Most signs on this street say 'Closes at 10 PM'. Can you zoom in on this specific sign to see if it follows that pattern?"
    *   *Final Answer:* "This sign is an outlier. The magnified text confirms it is 'OPEN 24 HOURS', unlike the others."
*   **HARD (Structural Comparison):**
    *   *Question:* "Many fabrics use a simple cross-hatch weave. How does the texture of this fabric compare when viewed up close?"
    *   *Final Answer:* "It deviates significantly from the standard. The zoomed-in view of the fabric's structure reveals a more complex and distinct herringbone texture."

---
**## [ STYLE 9: The Uncertain User ]**
*   **EASY (Confirmation Seeking):**
    *   *Question:* "I think I might see a tiny bug on that leaf, but I'm not sure. Can you zoom in and confirm what's there?"
    *   *Final Answer:* "You were right! I've zoomed in, and can confirm it is indeed a small, red ladybug."
*   **MEDIUM (Help with Reading):**
    *   *Question:* "My eyesight isn't what it used to be. Could you please magnify the hours on that sign for me?"
    *   *Final Answer:* "Certainly. I've zoomed in on the sign for you, and it says 'OPEN 24 HOURS'."
*   **HARD (Needing an Expert Opinion):**
    *   *Question:* "I'm no textile expert, but this pattern looks unusual. Can you use your zoom tool to perform an expert analysis and identify it for me?"
    *   *Final Answer:* "Of course. My detailed analysis of the magnified area indicates this is not a common pattern; it is a distinct herringbone texture."

---
**## [ STYLE 10: The AI Instructor ]**
*   **EASY (Simple Test):**
    *   *Question:* "Test Case 1: Object presence detection. Zoom into the target region and report findings."
    *   *Final Answer:* "Test Case 1 Result: Positive. A small, red ladybug was detected in the zoomed-in target region."
*   **MEDIUM (OCR Test):**
    *   *Question:* "Test Case 2: Optical Character Recognition. Magnify and transcribe the text within the specified bounding box."
    *   *Final Answer:* "Test Case 2 Result: Transcription successful after zoom. Text is 'OPEN 24 HOURS'."
*   **HARD (Attribute Classification Test):**
    *   *Question:* "Test Case 3: Fine-grained attribute classification. Zoom in, identify, and classify the material pattern."
    *   *Final Answer:* "Test Case 3 Result: Classification successful. Post-zoom analysis identifies the pattern as 'herringbone texture'."

---
**## [ STYLE 11: The Educator ]**
*   **EASY (Basic Observation Lesson):**
    *   *Question:* "Let's practice observation. To see the subject of today's lesson, you'll need to zoom in on this leaf. What do you see?"
    *   *Final Answer:* "Excellent observation. The magnified view shows our subject: a small, red ladybug with black spots."
*   **MEDIUM (Reading Comprehension):**
    *   *Question:* "This is a reading comprehension test. Please magnify the sign and tell me, based on the text, what is the key operating policy of this establishment?"
    *   *Final Answer:* "Correct. The key policy, as read from the zoomed-in sign, is that the establishment is 'OPEN 24 HOURS'."
*   **HARD (Analytical Deduction):**
    *   *Question:* "For this advanced materials class, I want you to deduce the manufacturing style of this fabric. Zoom in and analyze its structure."
    *   *Final Answer:* "A perfect deduction. The complex weave, visible only under magnification, is a distinct herringbone texture, indicating a high-end manufacturing process."

---
**## [ STYLE 12: The Quality Assurance Tester ]**
*   **EASY (Pixel Perfect Check):**
    *   *Question:* "QA Test 1.1: Verify asset integrity. Zoom into the specified coordinates and confirm that the ladybug asset is rendered correctly."
    *   *Final Answer:* "QA Test 1.1 Passed. The asset at the specified coordinates is confirmed to be a small, red ladybug with black spots, rendered as expected."
*   **MEDIUM (UI Text Validation):**
    *   *Question:* "QA Test 2.5: Validate UI text element. Magnify the target BBox and verify the text string matches the spec 'OPEN 24 HOURS'."
    *   *Final Answer:* "QA Test 2.5 Passed. The text element was magnified and verified to be 'OPEN 24 HOURS'."
*   **HARD (Defect Analysis):**
    *   *Question:* "QA Test 3.8: Non-conformance report. Perform a high-magnification analysis of the surface for any textural defects and classify the pattern."
    *   *Final Answer:* "QA Test 3.8 Result: No defects found. The surface pattern is classified as a standard herringbone texture."

---
**## [ STYLE 13: The Creative Writer ]**
*   **EASY (Setting the Scene):**
    *   *Question:* "I'm writing a story about a tiny world. Zoom in on this leaf and describe the first character I should introduce."
    *   *Final Answer:* "A perfect character for your tiny world appears upon zooming in: a solitary, small, red ladybug with black spots."
*   **MEDIUM (Discovering a Secret Message):**
    *   *Question:* "In my spy novel, there's a hidden message on this public sign. What secret does a closer look reveal?"
    *   *Final Answer:* "The secret is hidden in plain sight! The magnified sign reveals the coded message: 'OPEN 24 HOURS'."
*   **HARD (Describing a Mood):**
    *   *Question:* "I want to describe the feeling of this old, worn coat. Zoom in on the fabric and give me some evocative details about its texture."
    *   *Final Answer:* "The close-up tells a story of long journeys. The fabric has a faded but distinct herringbone texture, worn smooth in some places, suggesting a well-loved history."

---
**## [ STYLE 14: The AI Trainer ]**
*   **EASY (Lesson: Object Identification):**
    *   *Question:* "Learning module 1: Object Identification. The target is a small insect. Zoom in and identify the object."
    *   *Final Answer:* "Lesson complete. Object identified as a small, red ladybug with black spots."
*   **MEDIUM (Lesson: Text Extraction):**
    *   *Question:* "Learning module 2: Text Extraction. The goal is to read the store's hours. Magnify the sign and extract the information."
    *   *Final Answer:* "Lesson complete. Information extracted: 'OPEN 24 HOURS'."
*   **HARD (Lesson: Attribute Classification):**
    *   *Question:* "Learning module 3: Fine-grained Classification. Zoom in on the material and classify the weave pattern."
    *   *Final Answer:* "Lesson complete. Weave pattern classified as a distinct herringbone texture."

---
**## [ STYLE 15: The Cautious User ]**
*   **EASY (Double-Checking):**
    *   *Question:* "Before I get too close, can you zoom in and check if that red spot is a bug or just a berry?"
    *   *Final Answer:* "Good call to check first. I've zoomed in, and it is definitely a bug: a small, red ladybug."
*   **MEDIUM (Clarifying Fine Print):**
    *   *Question:* "I'm about to agree to this, but I need to be sure about the terms. Could you magnify the fine print on this sign for me?"
    *   *Final Answer:* "It's always wise to read the fine print. After zooming in, I can confirm the terms state it is 'OPEN 24 HOURS'."
*   **HARD (Assessing Safety):**
    *   *Question:* "This surface looks a bit worn out. Before I put any weight on it, could you perform a close-up inspection for any signs of structural weakness?"
    *   *Final Answer:* "Safety first. The magnified inspection reveals the material does show signs of fatigue, specifically a distinct herringbone texture that has begun to fray."

---
**## [ STYLE 16: The Field Biologist ]**
*   **EASY (Species Identification):**
    *   *Question:* "I'm in the field and have spotted a potential specimen. Can you zoom in on the leaf and identify the insect?"
    *   *Final Answer:* "Field identification successful. The specimen is a small, red ladybug with black spots (Coccinellidae family)."
*   **MEDIUM (Behavioral Observation):**
    *   *Question:* "This sign is part of an animal's territory marking. I need a close-up to read the faded text for my research."
    *   *Final Answer:* "Observation recorded. The magnified text on the territory marker reads 'OPEN 24 HOURS'."
*   **HARD (Microscopic Analysis):**
    *   *Question:* "I need to analyze the scale pattern on this reptile's skin. Please provide a high-magnification view and describe the texture."
    *   *Final Answer:* "Microscopic analysis complete. The scale pattern is a distinct herringbone texture, which is unusual for this species."

---
**## [ STYLE 17: The Tech Support Agent ]**
*   **EASY (Troubleshooting Visuals):**
    *   *Question:* "The user reports a 'red pixel' on their screen. Can you zoom in to the specified coordinates and determine if it's a dead pixel or an actual object in the image?"
    *   *Final Answer:* "Troubleshooting complete. It is not a dead pixel. The object is a small, red ladybug in the image content."
*   **MEDIUM (Reading an Error Code):**
    *   *Question:* "The customer can't read the tiny error code on the device's display. Please magnify the screen and transcribe the message."
    *   *Final Answer:* "Support ticket updated. The magnified error message on the screen reads 'OPEN 24 HOURS'."
*   **HARD (Analyzing Component Wear):**
    *   *Question:* "A user is complaining about performance degradation. Please zoom in on this component and inspect its surface for signs of physical wear and tear."
    *   *Final Answer:* "Component analysis complete. The magnified view of the component's surface shows a distinct herringbone texture, which is a sign of advanced material stress."

---
**## [ STYLE 18: The Minimalist ]**
*   **EASY (Minimal Query):**
    *   *Question:* "Insect?"
    *   *Final Answer:* "Confirmed: ladybug."
*   **MEDIUM (Minimal Query):**
    *   *Question:* "Hours?"
    *   *Final Answer:* "Confirmed: OPEN 24 HOURS."
*   **HARD (Minimal Query):**
    *   *Question:* "Fabric pattern?"
    *   *Final Answer:* "Confirmed: herringbone texture."

---
**## [ STYLE 19: The Roleplayer ]**
*   **EASY (Child's Perspective):**
    *   *Question:* "Wow, look at that leaf! I'm playing 'I Spy'. I spy with my little eye... something red! Can you get a closer look and tell me what it is?"
    *   *Final Answer:* "Great find! I've zoomed in for you. You spied a tiny, cute, red ladybug!"
*   **MEDIUM (Tourist's Perspective):**
    *   *Question:* "I'm a tourist and my language skills aren't great. Can you help me out by zooming in and reading that sign for me?"
    *   *Final Answer:* "Welcome! I'd be happy to help. I've magnified the sign, and it says 'OPEN 24 HOURS'."
*   **HARD (Art Critic's Perspective):**
    *   *Question:* "As an art critic, I'm fascinated by the artist's use of texture. Please provide a magnified analysis of the brushwork in this section of the painting."
    *   *Final Answer:* "An excellent point of focus. The magnified view reveals the artist used a dry brush technique to create a distinct herringbone texture, adding a layer of complexity."

---
**## [ STYLE 20: The Data-Entry Clerk ]**
*   **EASY (Data Point Entry):**
    *   *Question:* "Form Field 7a: Insect_Present. Please zoom to verify and input 'True' or 'False'."
    *   *Final Answer:* "Data Entry for Field 7a: True. A small, red ladybug is visible."
*   **MEDIUM (Data Point Entry):**
    *   *Question:* "Form Field 12c: Business_Hours. Please magnify sign and input the text value."
    *   *Final Answer:* "Data Entry for Field 12c: 'OPEN 24 HOURS'."
*   **HARD (Data Point Entry):**
    *   *Question:* "Form Field 15d: Material_Subtype. Please perform magnified analysis and select from dropdown: [Plain, Twill, Herringbone, Satin]."
    *   *Final Answer:* "Data Entry for Field 15d: Herringbone. The magnified view confirms a distinct herringbone texture."

---
**## [ STYLE 21: The Archaeologist ]**
* **EASY (Fragment Identification):**
    * *Question:* "In grid C4 of the excavation site, there's a small glint. Could it be a pottery shard or just a rock? Zoom in for a preliminary ID."
    * *Final Answer:* "Preliminary ID complete. The zoomed-in view confirms it is a small, red-glazed pottery shard, likely from the rim of a vessel."
* **MEDIUM (Inscription Deciphering):**
    * *Question:* "This tablet fragment has some faint cuneiform markings. They are too eroded to read. Can you magnify the surface to help me transcribe them?"
    * *Final Answer:* "Affirmative. After magnification, the inscription can be transcribed as a record of grain quantities."
* **HARD (Tool Mark Analysis):**
    * *Question:* "To understand how this artifact was made, I need to analyze the microscopic tool marks. Please perform a high-magnification analysis of the surface striations."
    * *Final Answer:* "Analysis complete. The magnified view reveals distinct parallel striations consistent with the use of a flint-based scraping tool."

---
**## [ STYLE 22: The Conspiracy Theorist ]**
* **EASY (Questioning Reality):**
    * *Question:* "They want you to think that's a ladybug, but it's too perfect. Is it a bug or a micro-surveillance drone? Zoom in and expose the truth."
    * *Final Answer:* "Upon closer inspection, the object is confirmed to be an organic insect—a small, red ladybug. No visible technology detected."
* **MEDIUM (Finding Hidden Messages):**
    * *Question:* "That's not just a regular sign. I bet there's a coded message in the imperfections of the lettering. Magnify it and tell me what they're *really* saying."
    * *Final Answer:* "The magnified text appears standard. The letters spell out 'OPEN 24 HOURS' with no discernible codes or hidden messages."
* **HARD (Uncovering Alien Tech):**
    * *Question:* "That fabric's shimmer isn't natural. It has to be some kind of advanced, non-human material. Zoom in. I bet you'll find a pattern that isn't from this world."
    * *Final Answer:* "The material has been analyzed under magnification. The pattern, while intricate, is a terrestrial and well-documented herringbone texture."

---
**## [ STYLE 23: The Poet ]**
* **EASY (A Fleeting Detail):**
    * *Question:* "A crimson jewel upon a sea of green. Grant my eyes a closer look, what is this tiny, wondrous thing?"
    * *Final Answer:* "The jewel, upon a closer look, is a small, red ladybug, its shell a perfect dome of color."
* **MEDIUM (Whispers of the City):**
    * *Question:* "What silent promise does that distant sign declare to the lonely street? Magnify its words for me."
    * *Final Answer:* "Its promise, magnified, is one of constant vigil: 'OPEN 24 HOURS'."
* **HARD (The Weaver's Tale):**
    * *Question:* "The story of this ancient coat is told in its threads. Zoom in, and describe the silent, V-shaped rhythm of its weave."
    * *Final Answer:* "The weaver's tale is one of careful craft. The magnified view shows a distinct herringbone texture, a rhythm of interlocking Vs."

---
**## [ STYLE 24: The Emergency Responder ]**
* **EASY (Threat Assessment):**
    * *Question:* "Dispatch, I have a visual on the object. I can't tell if it's a discarded toy or a potential device. Zoom in, now!"
    * *Final Answer:* "Object identified. The magnified view confirms it's a small, red plastic toy, resembling a ladybug. Threat is negative."
* **MEDIUM (Hazard Identification):**
    * *Question:* "There's a chemical truck involved in the accident. I can't read the hazard placard from this distance. Magnify it for me, stat!"
    * *Final Answer:* "Placard magnified. The text is 'OPEN 24 HOURS', it's a delivery truck, not a chemical tanker. Stand down."
* **HARD (Structural Failure Analysis):**
    * *Question:* "That support beam looks compromised. I need a close-up on that hairline crack. Describe the stress patterns around it."
    * *Final Answer:* "Zooming in on the fracture. The material shows stress patterns fanning out from the crack in a distinct herringbone texture, indicating imminent failure."

---
**## [ STYLE 25: The Online Shopper ]**
* **EASY (Verifying a Feature):**
    * *Question:* "The product description says it has a 'subtle ladybug logo'. Can you zoom in on the corner of the item to see if it's actually there?"
    * *Final Answer:* "Verification complete. I've zoomed in, and the small, red ladybug logo is present as described."
* **MEDIUM (Reading the Fine Print):**
    * *Question:* "I'm trying to see the nutrition facts on this snack, but the image is too low-res. Can you magnify the label for me?"
    * *Final Answer:* "Of course. The magnified label reads 'OPEN 24 HOURS' on the box, which seems to be an error in the product image."
* **HARD (Assessing Quality):**
    * *Question:* "This handbag is listed as 'premium quality'. Can you zoom in on the stitching to see if it's actually well-made?"
    * *Final Answer:* "Certainly. The magnified view of the stitching shows a very neat and durable herringbone texture stitch, consistent with high-quality craftsmanship."

---
**## [ STYLE 26: The Lawyer ]**
* **EASY (Evidence Verification):**
    * *Question:* "Exhibit A: a photograph of the crime scene. The prosecution claims a ladybug pin was dropped. Please magnify the area indicated and confirm its presence."
    * *Final Answer:* "Confirmed. A magnified view of the indicated area clearly shows a small, red object consistent with a ladybug pin."
* **MEDIUM (Contract Scrutiny):**
    * *Question:* "Let the record show that the opposing counsel is trying to obscure Clause 11. Please magnify the text of the contract for the jury."
    * *Final Answer:* "Clause 11 magnified. The text reads: 'The establishment will remain OPEN 24 HOURS'."
* **HARD (Forgery Detection):**
    * *Question:* "We contend this document is a forgery. An authentic document from this period would have been printed on paper with a specific watermark. Please perform a high-magnification analysis of the paper's texture."
    * *Final Answer:* "Analysis of the paper's fibers under high magnification reveals no watermark, but a faint, modern herringbone texture from the roller used in its production."

---
**## [ STYLE 27: The Gamer ]**
* **EASY (Loot Check):**
    * *Question:* "Is that a rare ladybug mob I need to capture, or just part of the background environment? Enhance view."
    * *Final Answer:* "It's a mob. Zoom confirms it's a small, red ladybug creature. You can interact with it."
* **MEDIUM (Reading Quest Text):**
    * *Question:* "I can't read the objective on that quest board. Can you render the text in high-def for me?"
    * *Final Answer:* "Text rendered. The quest objective is 'Ensure the tavern remains OPEN 24 HOURS'."
* **HARD (Crafting Material ID):**
    * *Question:* "To craft the legendary armor, I need a specific material. Zoom in on this fabric swatch and identify its texture map."
    * *Final Answer:* "Texture map identified. The pattern is 'Herringbone_Weave_03'. You have the correct material."

---
**## [ STYLE 28: The Chef / Food Critic ]**
* **EASY (Garnish Inspection):**
    * *Question:* "Before this plate goes out, is that a candied flower petal or an actual insect on the garnish? Give me a close-up."
    * *Final Answer:* "It's an insect. Zoomed-in view confirms a small, red ladybug. Plate should be remade."
* **MEDIUM (Verifying Origin):**
    * *Question:* "This bottle of olive oil claims to be from a specific estate, but the seal is smudged. Can you magnify the text on the seal?"
    * *Final Answer:* "Seal magnified. The text reads 'OPEN 24 HOURS', which is incorrect. This seems to be a novelty bottle."
* **HARD (Assessing Meat Quality):**
    * *Question:* "The quality of this dry-aged steak is in the marbling. I need a magnified view of the intramuscular fat distribution."
    * *Final Answer:* "Magnified view provided. The fat is distributed in a fine, web-like herringbone texture, indicating prime quality."

---
**## [ STYLE 29: The Engineer ]**
* **EASY (Component Verification):**
    * *Question:* "The schematic calls for a red LED indicator at this position. Zoom in and confirm the component is present."
    * *Final Answer:* "Component confirmed. A small, red LED, resembling a ladybug's shape, is installed at the specified coordinates."
* **MEDIUM (Reading a Serial Number):**
    * *Question:* "I need the part number off that microchip, but it's tiny. Please magnify the surface and read the alphanumeric code."
    * *Final Answer:* "Code acquired. The magnified text on the chip reads 'OPEN-24-H'."
* **HARD (Material Stress Analysis):**
    * *Question:* "We need to check this carbon fiber panel for micro-fractures after the stress test. Zoom in and analyze the weave integrity."
    * *Final Answer:* "Analysis complete. The magnified view shows micro-fractures forming along the resin boundaries of the herringbone texture, indicating material fatigue."

---
**## [ STYLE 30: The Journalist ]**
* **EASY (Fact-Checking a Detail):**
    * *Question:* "A witness mentioned seeing a 'strange bug'. Can you zoom in on this photo from the scene and see if you can find it? This detail is for my article."
    * *Final Answer:* "Detail confirmed. Zooming in on the leaf reveals a small, red ladybug, corroborating the witness statement."
* **MEDIUM (Getting the Full Quote):**
    * *Question:* "In the background of my photo of the mayor, there's a protest sign. I need to know exactly what it says for the caption. Can you read it?"
    * *Final Answer:* "Quote captured. The magnified sign reads 'OPEN 24 HOURS for discussion'."
* **HARD (Verifying Authenticity):**
    * *Question:* "The official claims this flag is a priceless historical artifact. A replica would use modern weaving. Zoom in on the fabric and tell me what the pattern says about its age."
    * *Final Answer:* "The weave tells a different story. The magnified pattern is a perfect, machine-made herringbone texture, suggesting it is a modern replica, not a historical artifact."

---
**## [ STYLE 31: The Spy ]**
* **EASY (Dead Drop Confirmation):**
    * *Question:* "Our asset was supposed to leave a signal marker on that park bench—a ladybug pin. Is it there? Zoom in."
    * *Final Answer:* "Signal confirmed. A small, red ladybug pin is visible on the bench after zooming in."
* **MEDIUM (Extracting Intel):**
    * *Question:* "The target has a document on their screen, but it's angled away. I can just see a corner. Zoom in and read any text you can."
    * *Final Answer:* "Intel extracted. The magnified text at the top of the document reads 'Project OPEN 24 HOURS'."
* **HARD (Counter-Espionage Analysis):**
    * *Question:* "This listening device is hidden in the agent's coat button. To disable it, I need to know the model. Zoom in on the button's surface and look for a micro-etched pattern."
    * *Final Answer:* "Pattern identified. The button's surface has a micro-etched herringbone texture, which is the signature of a K-series listening device."

---
**## [ STYLE 32: The Movie Director ]**
* **EASY (Checking the Shot):**
    * *Question:* "For this macro shot, I need a ladybug to land on the hero's hand. Is it in position? Punch in on the actor's knuckles."
    * *Final Answer:* "The ladybug is in position. Zoomed-in view confirms it is crawling on the actor's hand. Ready for action."
* **MEDIUM (Set Design Verification):**
    * *Question:* "The art department says that storefront sign is period-accurate for the 1950s. Zoom in so I can check the typography."
    * *Final Answer:* "Typography checked. The font on 'OPEN 24 HOURS' is a sans-serif that wasn't common until the 1970s. It's not period-accurate."
* **HARD (Costume Detail):**
    * *Question:* "The hero's jacket needs to look worn and rugged. Zoom in on the fabric. Does the texture convey that, or does it look too new?"
    * *Final Answer:* "The texture works. The magnified herringbone weave is frayed and weathered, effectively conveying a history of rough use."

---
**## [ STYLE 33: The Insurance Adjuster ]**
* **EASY (Damage Verification):**
    * *Question:* "The client's claim mentions impact damage from a 'small, red object'. Can you zoom in on the windshield and see if there's any evidence?"
    * *Final Answer:* "Evidence found. The magnified view shows a small impact point with residue consistent with a small, red ladybug."
* **MEDIUM (License Plate Retrieval):**
    * *Question:* "This security footage of the hit-and-run is blurry. I need the license plate number of the vehicle. Enhance and read it."
    * *Final Answer:* "License plate retrieved. After enhancement, the plate number is 'OPEN-24H'."
* **HARD (Assessing Water Damage):**
    * *Question:* "The homeowner is claiming water damage on this antique rug. I need to look for signs of mildew. Please provide a magnified view of the rug's fibers."
    * *Final Answer:* "Fibers analyzed. The magnified view of the rug's herringbone texture shows tell-tale microscopic dark spots, confirming the presence of mildew."

---
**## [ STYLE 34: The Librarian / Archivist ]**
* **EASY (Identifying Foxing):**
    * *Question:* "This manuscript has some discoloration. Is that a small illustration of a ladybug, or is it just 'foxing'—age-related spots? Please magnify."
    * *Final Answer:* "It is foxing. The magnified view confirms the red spot is an irregular chemical stain, not an intentional illustration."
* **MEDIUM (Reading Marginalia):**
    * *Question:* "A previous scholar made a tiny annotation in the margin of this book. It's too small to read with the naked eye. Can you zoom in and transcribe it?"
    * *Final Answer:* "Annotation transcribed. The magnified text reads, 'This shop was OPEN 24 HOURS'."
* **HARD (Analyzing Paper Type):**
    * *Question:* "To properly date this document, I need to identify the paper's structure. Can you perform a high-magnification analysis and describe the pattern of the laid and chain lines?"
    * *Final Answer:* "Paper structure analyzed. The high-magnification view shows no laid or chain lines, but a subtle, machine-pressed herringbone texture, dating it to the modern era."

---
**## [ STYLE 35: The Futurist ]**
* **EASY (Bio-Scan):**
    * *Question:* "My atmospheric sensors are picking up an unregistered life form in this quadrant. Engage nano-cam zoom and identify."
    * *Final Answer:* "Life form identified. The nano-cam confirms it is a carbon-based insectoid, class: Coccinellidae. A simple ladybug."
* **MEDIUM (Data-Stream Decryption):**
    * *Question:* "That's not text; it's a light-based data-stream. Magnify the glyphs on that sign and run them through the universal translator."
    * *Final Answer:* "Translation complete. The data-stream reads: 'SYSTEM STATUS: OPEN. UPTIME: 24 HOURS'."
* **HARD (Programmable Matter Analysis):**
    * *Question:* "That's a smart-fabric cloak. I need to know its current configuration. Zoom to the molecular level and report the alignment of its programmable matter."
    * *Final Answer:* "Molecular alignment confirmed. The programmable matter is currently arranged in a defensive herringbone texture to maximize energy dispersion."

---
**## [ STYLE 36: The Health Inspector ]**
* **EASY (Contaminant Search):**
    * *Question:* "There appears to be a foreign object in the salad. Is it a piece of red pepper or a pest? Zoom in for identification."
    * *Final Answer:* "Violation confirmed. Zoom reveals the object is an insect: a small, red ladybug."
* **MEDIUM (Checking Expiration Dates):**
    * *Question:* "The 'Best By' date on that package of cheese is completely smudged. Magnify the label and see if you can read it."
    * *Final Answer:* "Date is illegible. The only readable text after magnification is 'OPEN 24 HOURS', which seems to be from the case's price sticker."
* **HARD (Surface Sanitation Check):**
    * *Question:* "I need to check this cutting board for microscopic scoring that can harbor bacteria. Please perform a high-magnification analysis of the surface."
    * *Final Answer:* "Surface analysis complete. The magnified view shows deep knife scores arranged in a cross-hatch pattern, not a herringbone texture. This is a violation."

---
**## [ STYLE 37: The AI Itself (Self-Reflective) ]**
* **EASY (Sensor Validation):**
    * *Question:* "My optical sensors are flagging a pixel anomaly at coordinates (X, Y). Is this a data artifact or a valid object? Initiate zoom protocol to resolve."
    * *Final Answer:* "Anomaly resolved. It is a valid object. The zoomed data corresponds to a small, red ladybug."
* **MEDIUM (Confidence Score Improvement):**
    * *Question:* "My OCR module is returning a 67% confidence score for the text on that sign. Rerunning analysis with 4x magnification to improve certainty."
    * *Final Answer:* "Analysis complete. With magnification, confidence score is now 99.8%. The text is 'OPEN 24 HOURS'."
* **HARD (Pattern Ambiguity Resolution):**
    * *Question:* "My pattern recognition algorithms are caught between classifying this as 'Twill Weave' and 'Zig-Zag'. Initiating deep learning analysis on a magnified sample to achieve a definitive classification."
    * *Final Answer:* "Resolution achieved. The fine-grained analysis of the magnified sample confirms the pattern is a distinct herringbone texture."

---
**## [ STYLE 38: The Paranormal Investigator ]**
* **EASY (Orb or Insect):**
    * *Question:* "Did my EMF meter just spike because of that floating red orb, or is it just a bug? Zoom in on the anomaly."
    * *Final Answer:* "Anomaly identified. The 'orb' has antennae and legs. The zoomed view confirms it's a common ladybug."
* **MEDIUM (Ghostly Message):**
    * *Question:* "The spirit box is telling us to 'look at the sign'. Can you zoom in? Sometimes messages appear in photos that aren't there in real life."
    * *Final Answer:* "The sign has been magnified. It clearly reads 'OPEN 24 HOURS', with no apparent paranormal text."
* **HARD (Ectoplasm Analysis):**
    * *Question:* "I detected a major cold spot around this antique coat. Zoom in on the fabric. Can you see any ectoplasmic residue? It sometimes leaves a unique pattern."
    * *Final Answer:* "No ectoplasm detected. The magnified view shows only a standard, albeit old, herringbone texture on the fabric."

---
**## [ STYLE 39: The Social Media Influencer ]**
* **EASY (Aesthetic Content):**
    * *Question:* "OMG, you guys, there's the cutest little red bug on this flower! Can we get a close-up for my followers? #naturecore"
    * *Final Answer:* "Here's the shot! Zoomed in on the super cute, small, red ladybug. It's perfect for your feed."
* **MEDIUM (Background Check):**
    * *Question:* "Wait, before I post this selfie, what does that sign behind me say? Zoom in so I can make sure it doesn't say something weird."
    * *Final Answer:* "You're all good! The sign just says 'OPEN 24 HOURS'. Post away!"
* **HARD (Outfit of the Day Detail):**
    * *Question:* "I'm obsessed with the texture of this vintage blazer! Can you zoom way, way in so everyone can see the amazing pattern? #OOTD"
    * *Final Answer:* "Detail shot coming up! The magnified view shows off the blazer's amazing retro herringbone texture."

---
**## [ STYLE 40: The Translator ]**
* **EASY (Character Identification):**
    * *Question:* "This document uses a small, stylized pictograph as a seal. I need a closer look to identify the character."
    * *Final Answer:* "Here is the magnified view. The character is '虫', the Chinese character for insect, stylized to look like a ladybug."
* **MEDIUM (Reading Foreign Text):**
    * *Question:* "I need to translate the hours of operation on that sign, but the characters are too blurry. Please provide a magnified view."
    * *Final Answer:* "The magnified text reads '24時間営業', which translates to 'OPEN 24 HOURS'."
* **HARD (Calligraphy Analysis):**
    * *Question:* "This is ancient calligraphy. The nuance is in the brush strokes. Please zoom in on this specific character and describe the texture of the ink on the papyrus."
    * *Final Answer:* "The magnified view shows the ink has bled into the papyrus fibers in a feathering pattern that resembles a subtle herringbone texture, indicating a specific brush technique."

---

**## CONTEXT FOR YOUR TASK:**
- Source Dataset: {source_dataset}
- Difficulty Level: {difficulty}
- Point of Interest BBox: {bbox}
- Expected Observation After Zoom: "{expected_observation}"

**## STYLE GUIDELINE FOR THIS SAMPLE (COMBINE THESE TWO ELEMENTS):**

**1. SELECTED STYLE (for persona/tone):** {style_name}
- Style Description: {style_description}
- Look at the cookbook examples for this style to understand its persona, vocabulary, and tone

**2. QUESTION STRUCTURE TEMPLATE (for grammatical pattern):**
"{example_question}"

**3. ANSWER STYLE TEMPLATE:**
"{example_answer}"

**HOW TO COMBINE THEM:**
1. Take the question template structure above and adapt it with:
   - The specific bbox coordinates from context
   - Details related to the expected observation
   - Vocabulary and tone matching the {style_name} style
2. Your question MUST follow the template's grammatical structure
3. Your question MUST use the vocabulary and persona of {style_name}
4. Your answer should match the style shown in the answer template
5. Example: If the template pattern is "I need to [do something]. Can you zoom in..." and style is "The Scientist", 
   you might write "I need to analyze the molecular structure. Can you zoom in..." (scientific vocabulary in template structure)

**## YOUR ASSIGNMENT:**
Now, generate a new, unique JSON output for the following context. **Remember your CORE MISSION is to demonstrate the `ZOOM-IN` tool, and follow the style guideline provided above**.

**YOUR JSON OUTPUT:**
```