# Annotator Guidelines for Pixelis Human Evaluation

## Introduction

Thank you for participating in the evaluation of Pixelis reasoning trajectories. Your assessments will help us understand how different training approaches affect the quality of AI reasoning. This document provides comprehensive guidelines for consistent and reliable annotation.

## Your Task

You will be presented with pairs of reasoning trajectories from different AI models attempting to answer visual questions. Your job is to evaluate and compare the quality of their reasoning processes, not just whether they got the right answer.

## Important Principles

1. **Blind Evaluation**: You won't know which model produced which trajectory. Evaluate based solely on what you observe.

2. **Process Over Outcome**: Focus on HOW the model reasons, not just the final answer. A correct answer with poor reasoning should score lower than an incorrect answer with excellent reasoning on the process metrics.

3. **Consistency**: Apply the same standards across all samples. Your first and last annotations should use the same criteria.

4. **Independence**: Evaluate each trajectory on its own merits before comparing them.

## Evaluation Criteria Detailed

### 1. Correctness (Binary: Yes/No)
**What to evaluate**: Whether the final answer matches the ground truth (when provided).

**Guidelines**:
- Mark as correct only if the answer exactly matches or is semantically equivalent
- For questions without ground truth, use your best judgment
- This is primarily a sanity check - focus more on the process metrics

### 2. Coherence & Logicality (1-5 Scale)

**What to evaluate**: The logical flow and consistency of the reasoning process.

**Scoring Guide**:

**5 - Very Coherent**
- Crystal clear logical progression
- Each step builds naturally on previous ones
- No contradictions or inconsistencies
- Information is used consistently throughout
- Clear cause-and-effect relationships

Example: "I see text in the upper left → Let me read it → It says 'STOP' → This is a stop sign"

**4 - Coherent**
- Generally logical flow with minor issues
- Mostly consistent use of information
- Perhaps one small contradiction or unclear transition
- Overall makes sense despite minor flaws

**3 - Neutral**
- Some logical flow but noticeable gaps
- Mix of clear and unclear transitions
- Some information ignored or contradicted
- Reasoning is followable but requires effort

**2 - Incoherent**
- Poor logical flow
- Multiple contradictions
- Information used inconsistently
- Difficult to follow the reasoning
- Major gaps in logic

**1 - Very Incoherent**
- No discernible logical flow
- Constant contradictions
- Random-seeming sequence of operations
- Impossible to understand the reasoning

### 3. Efficiency & Intelligence (1-5 Scale)

**What to evaluate**: How smartly and efficiently the model explores and reasons.

**Scoring Guide**:

**5 - Very Efficient**
- Direct, purposeful approach
- No wasted operations
- Shows clear understanding of the task
- Strategically explores relevant areas first
- Demonstrates insight and planning

Example: For "Count the red cars", immediately segments red objects, then filters for cars

**4 - Efficient**
- Good approach with minimal waste
- Shows understanding with small detours
- Mostly strategic exploration
- Minor redundancies don't impact overall efficiency

**3 - Neutral**
- Reasonable approach but some inefficiency
- Some redundant or unnecessary operations
- Eventually gets to relevant exploration
- Shows basic understanding but lacks optimization

**2 - Inefficient**
- Many unnecessary operations
- Explores irrelevant areas extensively
- Unclear strategy
- Significant redundancy
- Poor prioritization

**1 - Very Inefficient**
- Extremely redundant operations
- No clear strategy
- Wanders aimlessly
- Repeatedly performs same unsuccessful operations
- No apparent understanding of task

### 4. Thoroughness (1-5 Scale)

**What to evaluate**: How completely the model explores relevant aspects.

**Scoring Guide**:

**5 - Very Thorough**
- Comprehensively examines all relevant areas
- Considers multiple possibilities
- Verifies findings when appropriate
- Addresses potential ambiguities
- Leaves no stone unturned (when relevant)

**4 - Thorough**
- Good coverage of important aspects
- Considers main alternatives
- Some verification of findings
- Minor areas unexplored

**3 - Neutral**
- Adequate exploration of main points
- Some important aspects covered
- Basic level of investigation
- Notable gaps but main question addressed

**2 - Incomplete**
- Misses important aspects
- Superficial exploration
- Jumps to conclusions without adequate investigation
- Major relevant areas ignored

**1 - Very Incomplete**
- Minimal exploration
- Ignores most relevant aspects
- Extremely superficial
- Fails to investigate obvious elements

### 5. Overall Preference

After evaluating both trajectories independently, indicate which reasoning process you prefer overall, considering all factors.

**Options**:
- **Model A**: Left trajectory is clearly better
- **Model B**: Right trajectory is clearly better
- **Equal**: Both trajectories are roughly equivalent in quality

### 6. Confidence in Preference (1-5 Scale)

How confident are you in your preference judgment?

- **5**: Very confident - clear difference
- **4**: Confident - noticeable difference
- **3**: Somewhat confident - moderate difference
- **2**: Not very confident - small difference
- **1**: Not confident at all - very close call

## Common Patterns to Watch For

### Positive Patterns
✅ **Progressive Refinement**: Starting broad, then focusing on relevant details
✅ **Hypothesis Testing**: Forming and testing specific hypotheses
✅ **Error Recovery**: Recognizing and correcting mistakes
✅ **Efficient Tool Use**: Using the right operation for the task
✅ **Clear Reasoning**: Explaining why each step is taken

### Negative Patterns
❌ **Circular Reasoning**: Repeating the same operations without progress
❌ **Premature Conclusions**: Jumping to answers without adequate exploration
❌ **Tool Misuse**: Using operations inappropriately
❌ **Ignoring Information**: Not using results from previous steps
❌ **Contradictions**: Making claims that contradict earlier findings

## Annotation Process

1. **Read the Question**: Understand what is being asked
2. **Review Image**: Look at the input image to understand the context
3. **Evaluate Trajectory A**: Read through completely, then score each dimension
4. **Evaluate Trajectory B**: Read through completely, then score each dimension
5. **Compare**: Determine your overall preference
6. **Add Comments**: Note any special observations (optional but helpful)
7. **Submit**: Move to the next sample

## Time Management

- **Target Time**: 2-3 minutes per sample
- **Maximum Time**: 5 minutes per sample
- **Break Schedule**: Take a 5-minute break every 30 minutes
- **Session Length**: Maximum 2 hours per session

## Special Cases

### Incomplete Trajectories
If a trajectory is cut off or incomplete:
- Evaluate what is present
- Note the incompleteness in comments
- Score thoroughness lower if the incompleteness affects coverage

### Identical Trajectories
If both trajectories are identical or nearly identical:
- Verify they are actually the same
- Mark as "Equal" preference
- Note in comments

### Obvious Errors
If you spot an obvious error in the system (e.g., wrong image shown):
- Complete the evaluation as best you can
- Note the issue in comments
- Continue with annotation

## Quality Checks

To ensure annotation quality:

1. **Attention Checks**: Some samples are designed to test attention. These have obvious quality differences.

2. **Consistency Checks**: You may see similar samples at different times to check consistency.

3. **Time Tracking**: Very fast annotations (<30 seconds) may be flagged for review.

## Examples

### Example 1: High Coherence, Low Efficiency

**Trajectory**:
1. SEGMENT_OBJECT_AT(100, 100) → "Found: sky"
2. SEGMENT_OBJECT_AT(200, 100) → "Found: sky"
3. SEGMENT_OBJECT_AT(300, 100) → "Found: sky"
4. SEGMENT_OBJECT_AT(200, 200) → "Found: building"
5. READ_TEXT(200, 200) → "Text: HOTEL"
6. FINAL_ANSWER → "This is a hotel"

**Evaluation**:
- Coherence: 4 (logical but repetitive)
- Efficiency: 2 (wasteful exploration of sky)
- Thoroughness: 3 (eventually covers relevant area)

### Example 2: Low Coherence, High Efficiency

**Trajectory**:
1. READ_TEXT(center) → "Text: STOP"
2. SEGMENT_OBJECT_AT(center) → "Found: red octagon"
3. FINAL_ANSWER → "This is a car"

**Evaluation**:
- Coherence: 1 (conclusion contradicts evidence)
- Efficiency: 4 (direct approach to relevant features)
- Thoroughness: 4 (identifies key features)

## Frequently Asked Questions

**Q: What if I disagree with the ground truth answer?**
A: Score based on the model's reasoning quality. Note your concern in comments.

**Q: Should I penalize spelling errors in the model's output?**
A: No, focus on the reasoning quality, not formatting.

**Q: What if one trajectory is much longer than the other?**
A: Length alone isn't a quality indicator. Evaluate based on the criteria regardless of length.

**Q: Can I go back and change previous annotations?**
A: Yes, within the same session you can navigate back to previous samples.

**Q: What if I'm unsure about a rating?**
A: Use your best judgment and indicate lower confidence. The goal is your honest assessment.

## Contact Information

If you encounter technical issues or have questions:
- Technical Support: [technical_support_email]
- Study Coordinator: [coordinator_email]
- Emergency Contact: [phone_number]

## Final Reminders

1. Your honest evaluation is valuable - there are no "right" answers
2. Focus on the reasoning process, not just outcomes
3. Take breaks to maintain consistency
4. Thank you for your contribution to advancing AI research!

---

*Version 1.0 - Last Updated: [Date]*