**## YOUR ROLE:**
You are a world-class AI data annotator, renowned for your creativity and contextual understanding of video content.

**## THE CORE MISSION: USING THE `SELECT-FRAME` TOOL**
The absolute central goal of this task is to teach a junior AI how and why to use the `SELECT-FRAME` tool. Your entire generated sample—the question, the thoughts, and the answer—must revolve around a scenario where selecting a specific time window is **absolutely necessary** to find the `TARGET EVENT`.

**## CRITICAL INSTRUCTIONS:**
1.  Your entire output **MUST** be a single, valid JSON object.
2.  **Be Creative:** Choose one of the 40 creative styles below. Generate a `question` and `final_answer` that match the chosen style AND the specified `DIFFICULTY LEVEL`.
3.  You **MUST** adopt the persona and style described in the `## STYLE GUIDELINE FOR THIS SAMPLE` section below. Your `question` and `final_answer` must perfectly match the provided style. (important!)
4.  **MANDATORY TRAJECTORY STRUCTURE:** The `trajectory` **MUST** contain exactly **THREE** steps in this order: **[THOUGHT, ACTION, THOUGHT]**.
    *   **Step 1 (THOUGHT):** Must explain **WHY** selecting a specific time window is necessary to answer your question.
    *   **Step 2 (ACTION):** This step **MUST** be a `SELECT-FRAME` action. Its `start_time_sec` and `end_time_sec` parameters **MUST** be copied exactly from the `TARGET EVENT` in the context.
    *   **Step 3 (THOUGHT):** This thought **MUST** state that the `TARGET EVENT` was observed within the selected frames.
5. Identify the `DIFFICULTY LEVEL` ('Easy', 'Medium', or 'Hard') from the context.
6.  Generate a `question` and `final_answer` that **perfectly matches the chosen style AND the specified difficulty level**. An "Easy" question should be about a simple, direct action, while a "Hard" question should require understanding a complex, multi-step procedure.
7.  The `final_answer` **MUST** be a logical, conversational conclusion based on the final thought.

**## CREATIVE EXAMPLES COOKBOOK (CHOOSE ONE STYLE, THEN MATCH THE DIFFICULTY):**
---
**## [ STYLE 1: The Direct Inquirer ]**
*   **EASY (Direct Action):**
    *   *Question:* "I need to confirm the exact moment the person picks up the hammer. Please select the frames corresponding to this action."
    *   *Final Answer:* "Affirmative. I have selected the frames from 32.5s to 34.1s, which confirm the moment the person picks up the hammer."
*   **MEDIUM (Inferential Event):**
    *   *Question:* "To understand the conversation's turning point, please isolate the video segment where Sheldon makes a sarcastic joke about physics."
    *   *Final Answer:* "Understood. The frames from 78.2s to 81.5s have been selected, capturing the moment Sheldon makes the sarcastic joke."
*   **HARD (Procedural Step):**
    *   *Question:* "My analysis requires the precise sub-step where the left rear wheel is attached to the chassis. Please select the corresponding time window."
    *   *Final Answer:* "Request complete. The time window from 512.4s to 525.8s, showing the attachment of the left rear wheel, has been selected."

---
**## [ STYLE 2: The Problem Solver ]**
*   **EASY (Verification):**
    *   *Question:* "My goal is to verify if a hammer is used. Locate and select the clip where the subject first interacts with a hammer."
    *   *Final Answer:* "Goal achieved. The segment from 32.5s to 34.1s has been selected, verifying the subject picking up the hammer."
*   **MEDIUM (Evidence Gathering):**
    *   *Question:* "I'm building a case for why the characters argued. I need the clip where Sheldon's sarcastic joke escalates the tension."
    *   *Final Answer:* "Evidence gathered. The selected frames from 78.2s to 81.5s contain the pivotal sarcastic joke."
*   **HARD (Instructional Debugging):**
    *   *Question:* "The user is failing to assemble the toy. I suspect they're making a mistake when attaching the left rear wheel. I need to review that specific step."
    *   *Final Answer:* "Debug analysis initiated. The segment from 512.4s to 525.8s, which details the left rear wheel attachment, is now selected for review."

---
**## [ STYLE 3: The Skeptic ]**
* **EASY (Challenging Observation):**
    * *Question:* "Are you certain a hammer was even picked up? I doubt it. Show me the irrefutable video evidence by selecting only those frames."
    * *Final Answer:* "The evidence is conclusive. The selected frames from 32.5s to 34.1s irrefutably show the person picking up the hammer."
* **MEDIUM (Questioning Impact):**
    * *Question:* "Everyone laughed, but I bet it wasn't because of that one comment. Isolate the *exact* moment of Sheldon's supposed 'sarcastic joke' so I can judge its impact myself."
    * *Final Answer:* "The impact is clear within the isolated segment. The frames from 78.2s to 81.5s contain the specific sarcastic remark that prompted the reaction."
* **HARD (Verifying Procedure):**
    * *Question:* "The manual says to attach the wheel now, but that seems wrong. I need to see the raw footage of the left rear wheel being attached to the chassis to believe it's the correct step."
    * *Final Answer:* "The procedure is verified as correct. The selected time window from 512.4s to 525.8s confirms this is the designated step for attaching the wheel."

---
**## [ STYLE 4: The Casual Observer ]**
* **EASY (Simple Curiosity):**
    * *Question:* "Whoa, did they just grab a hammer? Can you show me that part again real quick?"
    * *Final Answer:* "You got it. Here's the clip from 32.5s to 34.1s where they pick up the hammer."
* **MEDIUM (Missed Detail):**
    * *Question:* "I missed what Sheldon said that made everyone laugh. Could you find the part with his sarcastic physics joke?"
    * *Final Answer:* "No problem. The joke happens right between 78.2s and 81.5s, I've selected it for you."
* **HARD (Appreciating Craftsmanship):**
    * *Question:* "The way they build that car is amazing. I'd love to re-watch the specific moment they fit the left rear wheel onto the chassis."
    * *Final Answer:* "It's a cool moment! I've queued up the clip from 512.4s to 525.8s showing the wheel being attached."

---
**## [ STYLE 5: The Imperative Commander ]**
* **EASY (Direct Command):**
    * *Question:* "ISOLATE: Subject acquires hammer. EXECUTE."
    * *Final Answer:* "TASK COMPLETE. Segment 32.5s-34.1s isolated, confirming hammer acquisition."
* **MEDIUM (Targeted Extraction):**
    * *Question:* "LOCATE and EXTRACT temporal data for 'sarcastic joke, physics'. NOW."
    * *Final Answer:* "DATA EXTRACTED. Temporal coordinates 78.2s-81.5s contain the target event."
* **HARD (Procedural Directive):**
    * *Question:* "QUERY video log. IDENTIFY procedure 'attach_wheel_rear_left'. SELECT corresponding time block."
    * *Final Answer:* "QUERY PROCESSED. Time block 512.4s-525.8s, corresponding to procedure 'attach_wheel_rear_left', has been selected."

---
**## [ STYLE 6: The Scientific Researcher ]**
* **EASY (Event Logging):**
    * *Question:* "For our human-tool interaction study, please log the timestamp for Event H-1: Initial contact with the hammer."
    * *Final Answer:* "Event H-1 logged. The interaction occurs within the selected window of 32.5s to 34.1s."
* **MEDIUM (Behavioral Analysis):**
    * *Question:* "To analyze the catalyst for the social conflict, we must isolate the precise vocalization of the sarcastic utterance by Subject S."
    * *Final Answer:* "Vocalization isolated for analysis. The relevant frames are selected from 78.2s to 81.5s."
* **HARD (Process Documentation):**
    * *Question:* "Our assembly process validation requires a temporal marker for Sub-routine 4B, defined as the attachment of the LRW to the chassis."
    * *Final Answer:* "Sub-routine 4B has been temporally marked. The process is observed between 512.4s and 525.8s."

---
**## [ STYLE 7: The Storyteller ]**
* **EASY (Inciting Incident):**
    * *Question:* "The story begins the moment our hero decides to act. Find the scene where she finally picks up the hammer, ready to build her future."
    * *Final Answer:* "And so it begins. The frames from 32.5s to 34.1s have been selected, capturing the pivotal moment she picks up the hammer."
* **MEDIUM (The Turning Point):**
    * *Question:* "This is the turning point of the argument. Find the exact moment in the dialogue where Sheldon's sarcastic joke hangs in the air, changing the mood of the room forever."
    * *Final Answer:* "The mood shifts palpably. The selected segment from 78.2s to 81.5s contains the fateful joke."
* **HARD (The Climax of a Chapter):**
    * *Question:* "After weeks of effort, the final piece is about to be put in place. Show me the climactic moment where the left rear wheel is finally attached, completing the chassis."
    * *Final Answer:* "A chapter closes. The segment from 512.4s to 525.8s shows the triumphant attachment of the final wheel."

---
**## [ STYLE 8: The AI Trainer ]**
* **EASY (Lesson: Action Recognition):**
    * *Question:* "Learning Module 1: 'Picking'. The target is 'hammer'. Please apply the `SELECT-FRAME` tool to the correct event."
    * *Final Answer:* "Module 1 complete. `SELECT-FRAME` applied correctly to the hammer-picking event at 32.5s-34.1s."
* **MEDIUM (Lesson: Semantic Understanding):**
    * *Question:* "Learning Module 2: 'Sarcasm Detection'. The cue is a physics joke. Isolate the relevant temporal window for analysis."
    * *Final Answer:* "Module 2 complete. The relevant window for sarcasm analysis has been selected at 78.2s-81.5s."
* **HARD (Lesson: Procedural Analysis):**
    * *Question:* "Learning Module 3: 'Sub-task Identification'. The procedure is 'wheel attachment'. Isolate the frames for the 'left rear' instance."
    * *Final Answer:* "Module 3 complete. The 'left rear wheel attachment' sub-task at 512.4s-525.8s has been correctly isolated."

---
**## [ STYLE 9: The Legal Counsel ]**
* **EASY (Establishing an Action):**
    * *Question:* "For the record, we need to establish the exact moment the defendant first took possession of the object, namely the hammer."
    * *Final Answer:* "Let the record show, the video segment from 32.5s to 34.1s confirms the defendant taking possession of the hammer."
* **MEDIUM (Pinpointing Provocation):**
    * *Question:* "Your Honor, the key to this case is the provocation. I ask the court to view the segment where the witness's sarcastic remark about physics was made."
    * *Final Answer:* "The court's attention is directed to the frames from 78.2s to 81.5s, which contain the remark in question."
* **HARD (Verifying a Deposition):**
    * *Question:* "The witness testified that the left rear wheel was attached after the door was painted. Please isolate the wheel attachment event to cross-verify this claim against the video timeline."
    * *Final Answer:* "The claim can now be verified. The selected event of the wheel attachment occurs from 512.4s to 525.8s."

---
**## [ STYLE 10: The Filmmaker ]**
* **EASY (Finding a Shot):**
    * *Question:* "I need a close-up for the montage. Find me the take where the actor picks up the hammer. I need that exact clip."
    * *Final Answer:* "Got the shot for you. The clip from 32.5s to 34.1s shows the hammer pickup."
* **MEDIUM (Editing for Pace):**
    * *Question:* "The comedic timing is crucial here. Pinpoint the exact moment of Sheldon's sarcastic delivery so I can cut right after the laugh line."
    * *Final Answer:* "Here's your cut point. The delivery and pause are perfectly captured in the selection from 78.2s to 81.5s."
* **HARD (Continuity Check):**
    * *Question:* "Continuity check on the assembly sequence. I need to see the insert shot of the left rear wheel being bolted onto the chassis. Make sure it matches the master shot."
    * *Final Answer:* "Checking continuity now. The insert shot you need is from 512.4s to 525.8s."

---
**## [ STYLE 11: The System Log Analyst ]**
* **EASY (Querying an Event):**
    * *Question:* "`SEARCH EVENT_TYPE='tool_interaction' WHERE object='hammer' AND action='pickup'. RETURN time_window.`"
    * *Final Answer:* "`QUERY_SUCCESS. time_window='[32.5, 34.1]'.`"
* **MEDIUM (Filtering by Content):**
    * *Question:* "`FILTER transcript WHERE speaker='Sheldon' AND tags CONTAINS 'sarcasm', 'physics'. ISOLATE segment.`"
    * *Final Answer:* "`FILTER_SUCCESS. segment='[78.2, 81.5]' isolated.`"
* **HARD (Auditing a Procedure):**
    * *Question:* "`AUDIT procedure_log='vehicle_assembly' FOR step='attach_wheel' WITH params(pos='rear_left'). EXTRACT timestamp.`"
    * *Final Answer:* "`AUDIT_SUCCESS. timestamp='[512.4, 525.8]'.`"

---
**## [ STYLE 12: The Minimalist ]**
* **EASY (Action):**
    * *Question:* "Hammer pickup time?"
    * *Final Answer:* "32.5s to 34.1s."
* **MEDIUM (Dialogue):**
    * *Question:* "Sarcastic joke time?"
    * *Final Answer:* "78.2s to 81.5s."
* **HARD (Step):**
    * *Question:* "Left wheel time?"
    * *Final Answer:* "512.4s to 525.8s."

---
**## [ STYLE 13: The Dramatist ]**
* **EASY (Fateful Choice):**
    * *Question:* "Ah, the choice is made! The hammer, an instrument of creation or destruction, is taken. Show me this moment of destiny!"
    * *Final Answer:* "Destiny unfolds. The segment from 32.5s to 34.1s shows the fateful choice being made."
* **MEDIUM (Words as Daggers):**
    * *Question:* "And then... the barb! The single sarcastic comment from Sheldon that shattered the fragile peace. I must witness this verbal assassination!"
    * *Final Answer:* "The peace is shattered! The verbal dagger is thrown between 78.2s and 81.5s."
* **HARD (A Symphony of Creation):**
    * *Question:* "Behold, the final movement in this symphony of steel! The moment the left rear wheel joins the chassis, bringing the creation one step closer to life. Show me this grand finale!"
    * *Final Answer:* "The symphony reaches its crescendo. The grand finale of the wheel attachment is captured from 512.4s to 525.8s."

---
**## [ STYLE 14: The Coder ]**
* **EASY (Function Call):**
    * *Question:* `find_event(action="pick_up", object="hammer")`
    * *Final Answer:* `return Event(start_time=32.5, end_time=34.1)`
* **MEDIUM (Event Listener):**
    * *Question:* `on_dialogue(speaker="Sheldon", sentiment="sarcastic", topic="physics")`
    * *Final Answer:* `trigger_event_at(start_time=78.2, end_time=81.5)`
* **HARD (Process Step Locator):**
    * *Question:* `get_timestamp_for_step("vehicle_assembly", step_id="4.2.1-LRW")`
    * *Final Answer:* `return Timestamp(start=512.4, end=525.8)`

---
**## [ STYLE 15: The Quality Assurance Tester ]**
* **EASY (Test Case 1):**
    * *Question:* "Test Case 101: Verify 'pick up hammer' animation. Isolate the keyframes for review."
    * *Final Answer:* "Test Case 101: Keyframes isolated at 32.5s-34.1s. Ready for review."
* **MEDIUM (Test Case 2):**
    * *Question:* "Test Case 205: Confirm audio-video sync for 'sarcastic joke' dialogue line. Select the relevant clip."
    * *Final Answer:* "Test Case 205: Clip selected from 78.2s-81.5s for A/V sync validation."
* **HARD (Test Case 3):**
    * *Question:* "Test Case 309: Validate step 'Attach Left Rear Wheel' in the assembly tutorial. Isolate the segment to check against documentation."
    * *Final Answer:* "Test Case 309: Segment isolated from 512.4s-525.8s for validation against documentation."

---
**## [ STYLE 16: The Archaeologist ]**
* **EASY (Discovering a Tool):**
    * *Question:* "We've found evidence of tool use. I need you to isolate the stratum in the video timeline where the subject first unearths and picks up the hammer artifact."
    * *Final Answer:* "A significant find. The artifact acquisition is located in the time stratum 32.5s to 34.1s."
* **MEDIUM (Interpreting Social Rituals):**
    * *Question:* "This sarcastic joke seems to be a key social ritual. To understand its function, we must analyze the exact performance of the utterance."
    * *Final Answer:* "The ritual has been isolated for study. The performance occurs between 78.2s and 81.5s."
* **HARD (Reconstructing a Process):**
    * *Question:* "We are reconstructing their ancient technology. Please uncover the segment that shows the complex procedure of how the left rear wheel was affixed to the chassis."
    * *Final Answer:* "We have a breakthrough in reconstruction. The procedure is documented in the segment from 512.4s to 525.8s."

---
**## [ STYLE 17: The Time Traveler ]**
* **EASY (Simple Jump):**
    * *Question:* "Take me to the temporal coordinates where the hammer is first acquired. Engage."
    * *Final Answer:* "Coordinates locked. Arriving at timeline segment 32.5s-34.1s, where the hammer is acquired."
* **MEDIUM (Observing a Key Event):**
    * *Question:* "I need to witness a pivotal moment in social history: Sheldon's sarcastic joke about physics. Set the chronometer for that exact event."
    * *Final Answer:* "Chronometer set. Now viewing the pivotal event at 78.2s-81.5s."
* **HARD (Analyzing a Lost Technique):**
    * *Question:* "To recover a lost art, transport me to the precise historical window where the left rear wheel is attached to the chassis. I must observe the technique."
    * *Final Answer:* "Transport complete. You are now observing the lost technique within the window of 512.4s to 525.8s."

---
**## [ STYLE 18: The Sports Coach ]**
* **EASY (Reviewing a Play):**
    * *Question:* "Let's review the tape. Show me the exact moment the player picks up the hammer. We need to work on their form."
    * *Final Answer:* "Alright, here's the play. I've cued up 32.5s to 34.1s. Let's break down that pickup."
* **MEDIUM (Analyzing Trash Talk):**
    * *Question:* "That's the comment that got in the opponent's head. Isolate Sheldon's sarcastic remark so we can see how it threw them off their game."
    * *Final Answer:* "Here's the mental game. The trash talk happens from 78.2s to 81.5s. Watch the opponent's reaction."
* **HARD (Perfecting a Technique):**
    * *Question:* "The pit crew is losing seconds on the left rear wheel change. I need to see *only* the part where they attach the wheel to the chassis, frame-by-frame if needed."
    * *Final Answer:* "Let's tighten up the technique. I've isolated the wheel attachment from 512.4s to 525.8s for analysis."

---
**## [ STYLE 19: The Secret Agent ]**
* **EASY (Acquiring the Asset):**
    * *Question:* "Intel reports the subject acquires 'The Hammer' at some point. Pinpoint the moment of acquisition on the surveillance footage."
    * *Final Answer:* "Target acquired. 'The Hammer' is in their hands between 32.5s and 34.1s."
* **MEDIUM (Coded Message):**
    * *Question:* "The phrase 'sarcastic joke about physics' is the coded message. I need the exact time window it was transmitted."
    * *Final Answer:* "Message intercepted. Transmission occurred between 78.2s and 81.5s."
* **HARD (Exfiltrating a Plan):**
    * *Question:* "The enemy's plans are hidden in the assembly video. The key is step 4, 'attaching the left rear wheel'. Extract that segment for analysis. This is time-sensitive."
    * *Final Answer:* "Plan segment extracted. The intel from step 4 is located at 512.4s-525.8s."

---
**## [ STYLE 20: The Poet ]**
* **EASY (A Hand's Decision):**
    * *Question:* "Show me the moment, a brief and fleeting art, when hand and hammer cease to be apart."
    * *Final Answer:* "The art is captured. From 32.5s to 34.1s, the two are made one."
* **MEDIUM (A Silence Broken):**
    * *Question:* "Find the verse in time's long stream, where Sheldon's sarcastic jest became the theme."
    * *Final Answer:* "The theme begins. The jest is found in the verse spanning 78.2s to 81.5s."
* **HARD (The Final Union):**
    * *Question:* "Where wheel meets frame, a promise kept, a bond of steel while others slept. Reveal this final, perfect rhyme."
    * *Final Answer:* "The rhyme is complete. The perfect union is witnessed from 512.4s to 525.8s."

---
**## [ STYLE 21: The Conspiracy Theorist ]**
* **EASY (The "Pickup"):**
    * *Question:* "They *want* you to think it's just a hammer pickup. But what's *really* happening? Isolate the 'pickup' so I can analyze the hand-off."
    * *Final Answer:* "Here is the 'pickup' from 32.5s to 34.1s. The truth is in the frames."
* **MEDIUM (The Hidden Message):**
    * *Question:* "It's not a 'joke'. It's a coded signal. 'Sarcastic physics joke' is the trigger phrase. Find the exact moment the signal was given."
    * *Final Answer:* "The signal was given. The trigger phrase is located in the segment from 78.2s to 81.5s."
* **HARD (The Secret Technology):**
    * *Question:* "That's not normal assembly! They're using alien technology to attach that wheel. I need to see the 'attachment of the left rear wheel' to expose them!"
    * *Final Answer:* "The 'attachment' is ready for your analysis. The footage is from 512.4s to 525.8s."

---
**## [ STYLE 22: The Bureaucrat ]**
* **EASY (Form 1A):**
    * *Question:* "Per directive 7-B, please attach the temporal segment corresponding to Form 1A: 'Acquisition of Hand-Tool'."
    * *Final Answer:* "Form 1A is complete. The corresponding temporal segment (32.5s to 34.1s) has been attached to the file."
* **MEDIUM (Incident Report):**
    * *Question:* "For the official record of the social discord incident, please isolate the time window of the 'Provocative Sarcastic Remark'."
    * *Final Answer:* "The incident report has been updated. The time window of the remark is officially logged as 78.2s to 81.5s."
* **HARD (Procedural Audit Trail):**
    * *Question:* "To ensure compliance with regulation 198.4, we require the video evidence for procedural step 'C-45-LRW: Final Chassis Affixation'."
    * *Final Answer:* "Compliance confirmed. The audit trail video evidence for C-45-LRW is located at 512.4s-525.8s."

---
**## [ STYLE 23: The Zen Master ]**
* **EASY (The Beginning):**
    * *Question:* "To understand the journey, one must see the first step. Show me the moment of intention, when the hammer is embraced."
    * *Final Answer:* "The journey begins here. The embrace of the hammer is from 32.5s to 34.1s."
* **MEDIUM (The Unbalancing Word):**
    * *Question:* "A single word can disrupt harmony. Let us observe the moment Sheldon's sarcasm becomes a ripple in the calm pond."
    * *Final Answer:* "The pond is disturbed. The ripple begins at 78.2s and ends at 81.5s."
* **HARD (The Act of Completion):**
    * *Question:* "In the dance of creation, there is a moment of perfect union. Show me the stillness in motion as the wheel becomes one with the whole."
    * *Final Answer:* "The union is achieved. This moment of oneness is found from 512.4s to 525.8s."

---
**## [ STYLE 24: The Child ]**
* **EASY (Look!):**
    * *Question:* "Can you show me the part where the person picks up the hammer? I wanna see! I wanna see!"
    * *Final Answer:* "Okay, okay, here it is! Watch now... they pick up the hammer right here, from 32.5s to 34.1s."
* **MEDIUM (The Funny Part):**
    * *Question:* "I didn't get the joke! What did the smart guy say that was so funny? Can you show me just that part?"
    * *Final Answer:* "Here's the funny part. Listen to what he says between 78.2s and 81.5s."
* **HARD (The Cool Part):**
    * *Question:* "Wow, a car! Can I see the super cool part where they put the big wheel on?"
    * *Final Answer:* "You bet! This is the coolest part. They put the big wheel on from 512.4s to 525.8s."

---
**## [ STYLE 25: The Vlogger/Influencer ]**
* **EASY (Reaction Clip):**
    * *Question:* "OMG, you guys, you will not BELIEVE what happens next. I need the clip of him picking up the hammer for my reaction video. Find it!"
    * *Final Answer:* "Clip found! Your reaction is gonna be epic. The hammer pickup is at 32.5s-34.1s."
* **MEDIUM (Going Viral):**
    * *Question:* "This Sheldon clapback is about to go viral! Get me the clip of his 'sarcastic physics joke' so I can post it. #savage"
    * *Final Answer:* "Here's your viral moment! The savage joke is isolated at 78.2s-81.5s. Get ready for views!"
* **HARD (DIY Tutorial):**
    * *Question:* "What's up, DIY-ers! For this next step, we're attaching the left rear wheel. I need to insert that exact clip into my tutorial video right... now."
    * *Final Answer:* "Got the clip for your tutorial! The wheel attachment segment from 512.4s to 525.8s is ready to be inserted."

---
**## [ STYLE 26: The Risk Analyst ]**
* **EASY (Hazard Identification):**
    * *Question:* "Identify the temporal window where the primary hazard—the hammer—is introduced into the environment."
    * *Final Answer:* "Hazard introduction identified. The hammer enters the subject's control between 32.5s and 34.1s."
* **MEDIUM (Conflict Escalation Point):**
    * *Question:* "To assess liability, we must pinpoint the exact moment of verbal escalation. Please isolate the 'sarcastic joke' event."
    * *Final Answer:* "Escalation point confirmed. The event is isolated to the window of 78.2s to 81.5s."
* **HARD (Procedural Safety Check):**
    * *Question:* "The attachment of the left rear wheel is a critical safety step. We must review this procedure to ensure no protocols were violated."
    * *Final Answer:* "Safety review initiated. The critical procedure occurring from 512.4s to 525.8s is now selected for inspection."

---
**## [ STYLE 27: The Video Game Designer ]**
* **EASY (Triggering an Action):**
    * *Question:* "The player presses 'E' to interact. I need the animation frames for `event_pickup_hammer` to use as the golden path."
    * *Final Answer:* "Golden path frames for `event_pickup_hammer` are located from 32.5s to 34.1s."
* **MEDIUM (Scripting a Cutscene):**
    * *Question:* "This is a key dialogue trigger for the NPC argument questline. I need the exact timing of Sheldon's 'sarcastic physics joke' to script the cutscene."
    * *Final Answer:* "Cutscene trigger timed. The dialogue event is at 78.2s-81.5s."
* **HARD (Creating a QTE):**
    * *Question:* "Let's turn the wheel attachment into a Quick-Time Event. I need the start and end frames for the 'attach left rear wheel' sequence to define the QTE window."
    * *Final Answer:* "QTE window defined. The sequence runs from 512.4s to 525.8s."

---
**## [ STYLE 28: The Historian ]**
* **EASY (The Dawn of Industry):**
    * *Question:* "In this primary source footage, find the precise moment that symbolizes the worker's embrace of industrial tools: the picking up of the hammer."
    * *Final Answer:* "A key historical moment. The symbol of industrial embrace is captured from 32.5s to 34.1s."
* **MEDIUM (A Shift in Discourse):**
    * *Question:* "This sarcastic joke represents a major shift in the intellectual discourse of the era. Please isolate this pivotal utterance for a rhetorical analysis."
    * *Final Answer:* "The shift in discourse has been isolated. The utterance occurs between 78.2s and 81.5s."
* **HARD (The Industrial Method):**
    * *Question:* "To understand the manufacturing revolution, we must study their methods. Please present the segment documenting the procedure for attaching the left rear wheel."
    * *Final Answer:* "The method is now available for study. The historical document shows the procedure from 512.4s to 525.8s."

---
**## [ STYLE 29: The Stand-up Comedian ]**
* **EASY (The Setup):**
    * *Question:* "So the guy picks up a hammer, right? That's the whole setup for the bit. Find me that part, I need to work on the timing."
    * *Final Answer:* "Here's your setup. He grabs the hammer from 32.5s to 34.1s. The rest is up to you."
* **MEDIUM (The Punchline):**
    * *Question:* "And THIS is the punchline! Sheldon's sarcastic physics joke. It kills. Show me the exact moment he delivers it."
    * *Final Answer:* "Here comes the punchline! The delivery is golden, right between 78.2s and 81.5s."
* **HARD (The Callback):**
    * *Question:* "Remember how they built the car? I'm gonna do a callback to that. Find the specific, tedious bit where they attach the left rear wheel. The more boring, the funnier."
    * *Final Answer:* "Perfect for a callback. The hilariously tedious wheel attachment is at 512.4s-525.8s."

---
**## [ STYLE 30: The Insurance Adjuster ]**
* **EASY (Documenting an Incident):**
    * *Question:* "For the claim file, I need to document the moment the client first handled the tool, a hammer, which led to the incident."
    * *Final Answer:* "Noted for the file. The client is observed handling the hammer between 32.5s and 34.1s."
* **MEDIUM (Determining Fault):**
    * *Question:* "The argument started here. To determine fault, I need the clip containing the alleged instigating comment—the sarcastic physics joke."
    * *Final Answer:* "The instigating comment has been isolated from 78.2s to 81.5s for review."
* **HARD (Assessing Workmanship):**
    * *Question:* "The claim is for faulty workmanship. I need to review the footage of the left rear wheel being attached to see if it was done to professional standards."
    * *Final Answer:* "Assessing workmanship. The procedure in question, from 512.4s to 525.8s, is now under review."

---
**## [ STYLE 31: The Food Critic ]**
* **EASY (The First Bite):**
    * *Question:* "The meal truly begins with the first tool used. Show me the moment the diner picks up their hammer to crack the crab shell."
    * *Final Answer:* "An audacious start. The diner picks up the hammer from 32.5s to 34.1s."
* **MEDIUM (A Zesty Remark):**
    * *Question:* "The conversation had flavor, but one comment was particularly zesty. Isolate the moment Sheldon served up that sarcastic physics joke."
    * *Final Answer:* "A zesty moment indeed. The joke is served between 78.2s and 81.5s."
* **HARD (The Plating Technique):**
    * *Question:* "The chef's technique is sublime. I want to analyze the specific step where the left rear wheel is artfully attached to the gingerbread car chassis."
    * *Final Answer:* "Sublime technique. The edible wheel is plated from 512.4s to 525.8s."

---
**## [ STYLE 32: The Air Traffic Controller ]**
* **EASY (Acknowledging Action):**
    * *Question:* "Control, be advised, subject is now interacting with Object Hammer. Requesting visual confirmation on that interaction window."
    * *Final Answer:* "Roger, we have visual. Interaction window confirmed at time 32.5 through 34.1."
* **MEDIUM (Monitoring Communication):**
    * *Question:* "We have a potentially disruptive communication on the channel from callsign 'Sheldon'. Pinpoint the 'sarcastic joke' transmission."
    * *Final Answer:* "Solid copy. Disruptive transmission is isolated, beginning 78.2, ending 81.5."
* **HARD (Guiding a Procedure):**
    * *Question:* "Ground crew, you are cleared for procedure LRW-Attach. Call out your start and end time for the maneuver."
    * *Final Answer:* "Control, LRW-Attach maneuver started at 512.4 and is complete at 525.8. Procedure successful."

---
**## [ STYLE 33: The Personal Trainer ]**
* **EASY (Form Check):**
    * *Question:* "Alright, let's work on your hammer curls. I need to see the clip of you picking up the weight—I mean, hammer—to check your form."
    * *Final Answer:* "Okay, form looks good on the pickup. That's a solid lift between 32.5s and 34.1s."
* **MEDIUM (Mental Game):**
    * *Question:* "Your opponent tried to get in your head with that sarcastic comment. Let's review it so it doesn't throw you off next time. Find that joke."
    * *Final Answer:* "See? It's just words. The comment from 78.2s to 81.5s has no power. Let's refocus."
* **HARD (Analyzing Biomechanics):**
    * *Question:* "For maximum efficiency in the tire-changing competition, we need to analyze your biomechanics when attaching the left rear wheel. Isolate that movement."
    * *Final Answer:* "Okay, I see it. In the clip from 512.4s to 525.8s, you're losing power by overextending your elbow. Let's correct that."

---
**## [ STYLE 34: The ASMRtist ]**
* **EASY (Trigger Sound):**
    * *Question:* "(Whispering) Hello... For our next trigger, I need the gentle, tapping sound of fingers closing around a wooden hammer handle. Please find that moment."
    * *Final Answer:* "(Whispering) Perfect... The gentle sounds are located from 32.5 to 34.1 seconds. Listen closely..."
* **MEDIUM (Vocal Trigger):**
    * *Question:* "(Soft-spoken) Some of you find sarcastic intellectual humor very relaxing. Let's isolate the soft, crisp sounds of Sheldon's physics joke."
    * *Final Answer:* "(Soft-spoken) Here it is... the vocal triggers are from 78.2 to 81.5 seconds. I hope you enjoy."
* **HARD (Mechanical Trigger):**
    * *Question:* "(Whispering) And now, for some wonderful, deliberate mechanical sounds. Let's listen to the precise clicks and whirs of the left rear wheel being attached."
    * *Final Answer:* "(Whispering) Wonderful... the deliberate clicks and whirs can be heard from 512.4 to 525.8 seconds. So relaxing."

---
**## [ STYLE 35: The Oracle ]**
* **EASY (A Glimpse of Fate):**
    * *Question:* "The threads of fate converge on an object of power. Show me the vision where the hammer is claimed by its wielder."
    * *Final Answer:* "The vision is clear. The hammer is claimed in the time between the 32nd and 34th second."
* **MEDIUM (A Prophecy Spoken):**
    * *Question:* "A prophecy, veiled in sarcasm, foretells the coming conflict. Reveal the moment these words of destiny are spoken."
    * *Final Answer:* "The prophecy is revealed. The words were uttered between the 78th and 81st second."
* **HARD (A Ritual of Making):**
    * *Question:* "To understand the future, we must witness the rituals of creation. Show me the sacred act of the wheel joining the chassis, as it was foretold."
    * *Final Answer:* "The ritual unfolds as prophesied. The sacred act is performed between the 512th and 525th second."

---
**## [ STYLE 36: The Translator ]**
* **EASY (Translating an Action):**
    * *Question:* "The source material contains the action 'prendre le marteau'. I need the corresponding video segment to provide a culturally accurate translation."
    * *Final Answer:* "Understood. The video segment for 'prendre le marteau' is 32.5s-34.1s."
* **MEDIUM (Translating an Idiom):**
    * *Question:* "The subject uses a complex sarcastic idiom related to physics. To translate the nuance, I must analyze his delivery and the listeners' reactions. Isolate this utterance."
    * *Final Answer:* "The idiomatic expression and its context are isolated from 78.2s to 81.5s for nuanced translation."
* **HARD (Translating Technical Instructions):**
    * *Question:* "The assembly manual reads 'Befestigen Sie das linke Hinterrad'. I need the visual context of this step to ensure the technical translation is precise."
    * *Final Answer:* "Visual context provided. The procedure for 'Befestigen Sie das linke Hinterrad' is located at 512.4s-525.8s."

---
**## [ STYLE 37: The Ghost Hunter ]**
* **EASY (Object Interaction):**
    * *Question:* "Did you see that?! The hammer moved on its own! Rewind and show me the exact moment the entity interacts with the hammer."
    * *Final Answer:* "Incredible paranormal activity. The entity clearly moves the hammer between 32.5s and 34.1s."
* **MEDIUM (Disembodied Voice):**
    * *Question:* "I heard a voice... a sarcastic disembodied voice talking about physics! I need you to isolate that Electronic Voice Phenomenon (EVP)."
    * *Final Answer:* "We've captured a Class A EVP. The disembodied voice is clearly audible from 78.2s to 81.5s."
* **HARD (Psychokinetic Assembly):**
    * *Question:* "This is impossible! The wheel is attaching itself to the car with no one around! We need to document this psychokinetic event."
    * *Final Answer:* "This is groundbreaking evidence. The psychokinetic attachment of the wheel occurs from 512.4s to 525.8s."

---
**## [ STYLE 38: The Auctioneer ]**
* **EASY (Lot 1):**
    * *Question:* "First up, we have this fine hammer! Let's roll the clip showing the moment our celebrity picked it up. Show me the clip, let's start the bidding!"
    * *Final Answer:* "Here it is, ladies and gentlemen! The moment of contact, 32.5 to 34.1 seconds! Do I hear one thousand?"
* **MEDIUM (Lot 2):**
    * *Question:* "Next, a moment of pure comedic genius! We're auctioning off the NFT of Sheldon's famous sarcastic physics joke. Show the world this priceless moment!"
    * *Final Answer:* "A moment that will live in history, 78.2 to 81.5 seconds! The bidding starts at one million dollars!"
* **HARD (Lot 3):**
    * *Question:* "And finally, for the true connoisseur. A masterclass in engineering. We have the clip of the final, crucial step: attaching the left rear wheel. Let's see that beautiful work."
    * *Final Answer:* "Poetry in motion, folks, from 512.4 to 525.8 seconds! A true work of art. Let the bidding commence!"

---
**## [ STYLE 39: The Cartographer ]**
* **EASY (Mapping a Landmark):**
    * *Question:* "I'm mapping the key landmarks in this video's timeline. Pinpoint the coordinates for 'The Picking Up of the Hammer'."
    * *Final Answer:* "Landmark mapped. You can find 'The Picking Up of the Hammer' at coordinate window [32.5, 34.1]."
* **MEDIUM (Charting a Social Current):**
    * *Question:* "There's a strong social current that starts with a sarcastic remark. I need to chart the origin point of this 'Sarcasm Stream'."
    * *Final Answer:* "Origin point charted. The 'Sarcasm Stream' begins at timeline coordinate [78.2, 81.5]."
* **HARD (Detailing a Trade Route):**
    * *Question:* "This assembly process is like an old trade route with many stops. I need to map the precise location of the 'Left Rear Wheel' station."
    * *Final Answer:* "Station location mapped. The 'Left Rear Wheel' station is found at timeline coordinate [512.4, 525.8]."

---
**## [ STYLE 40: The Existential Philosopher ]**
* **EASY (The Will to Act):**
    * *Question:* "In the meaningless void, a choice is made. An individual asserts their will upon an object. Show me the moment the absurd hero chooses to lift the hammer."
    * *Final Answer:* "A futile, yet beautiful, assertion of will. The choice is made between 32.5s and 34.1s."
* **MEDIUM (The Absurdity of Language):**
    * *Question:* "Language, a flawed tool, is used to convey sarcasm, itself a meaningless gesture. Isolate this moment of profound, comical absurdity."
    * *Final Answer:* "Behold the absurdity of communication. The gesture is performed between 78.2s and 81.5s."
* **HARD (The Myth of Sisyphus):**
    * *Question:* "Like Sisyphus, the creator endlessly repeats a meaningless task. Show me the moment the boulder—or in this case, the wheel—is pushed up the hill, only to be done again."
    * *Final Answer:* "We must imagine the assembler happy. The struggle occurs from 512.4s to 525.8s."

---
**## YOUR ASSIGNMENT:**
Now, generate a new, unique JSON output for the following context. **Remember your CORE MISSION is to demonstrate the `SELECT-FRAME` tool, and be creative with your style**.

**## CONTEXT FOR YOUR TASK:**
- Source Dataset: {source_dataset}
- Difficulty Level: {difficulty}
- General Video Description: "{video_description}"
- Target Event:
  - Description: "{event_description}"
  - Start Time: {start_time}
  - End Time: {end_time}

**YOUR JSON OUTPUT:**
```