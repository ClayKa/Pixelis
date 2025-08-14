# Human Evaluation Protocol for Pixelis Reasoning Quality

## Overview
This document defines the comprehensive protocol for human evaluation of the Pixelis vision-language agent's reasoning quality. Beyond simple correctness metrics, we assess the coherence, efficiency, and perceived intelligence of the model's reasoning trajectories.

## Evaluation Scope

### Primary Objectives
1. **Reasoning Coherence Assessment**: Evaluate whether the model's step-by-step reasoning is logical, consistent, and free from contradictions
2. **Efficiency & Intelligence Measurement**: Assess whether the model explores visual space efficiently and demonstrates intelligent decision-making
3. **Comparative Analysis**: Determine the contribution of different training components (curiosity rewards, coherence rewards, online learning) to perceived reasoning quality

### Models Under Evaluation
- **Baseline Models**:
  - `Pixel-Reasoner-Base`: Original Pixel-Reasoner without modifications
  - `Pixelis-SFT-Only`: Supervised fine-tuned model without RL
  
- **Ablation Models**:
  - `Pixelis-RFT-Base`: RL-trained with task reward only
  - `Pixelis-RFT-Coherence`: RL-trained with task + coherence rewards
  - `Pixelis-RFT-Curiosity`: RL-trained with task + curiosity rewards
  
- **Full Models**:
  - `Pixelis-RFT-Full`: RL-trained with all reward components
  - `Pixelis-Online`: Full model with TTRL online adaptation

## Research Hypotheses

### H1: Coherence Hypothesis
**Statement**: Models trained with trajectory coherence rewards (`R_coherence`) will produce reasoning trajectories rated as significantly more logical and coherent by human judges compared to models trained without this component.

**Specific Predictions**:
- `Pixelis-RFT-Coherence` > `Pixelis-RFT-Base` on coherence ratings (p < 0.05)
- `Pixelis-RFT-Full` > `Pixelis-RFT-Curiosity` on coherence ratings (p < 0.05)
- Effect size: Cohen's d > 0.5 (medium to large effect)

**Rationale**: The coherence reward explicitly penalizes repetitive actions and rewards smooth transitions between reasoning steps, which should manifest as more human-interpretable logical flow.

### H2: Efficiency/Intelligence Hypothesis
**Statement**: The online-adapted `Pixelis-Online` model will be perceived as more efficient and intelligent in its exploration strategy compared to purely offline-trained models.

**Specific Predictions**:
- `Pixelis-Online` > `Pixelis-RFT-Full` on efficiency ratings (p < 0.05)
- Reduced redundant visual operations (quantitative metric)
- Higher ratings on "demonstrates understanding" scale

**Rationale**: Online learning with k-NN retrieval allows the model to leverage past experiences, leading to more targeted and efficient visual exploration.

### H3: Curiosity-Driven Exploration Hypothesis
**Statement**: Models trained with curiosity rewards will show more diverse but purposeful exploration patterns, rated as more thorough without being wasteful.

**Specific Predictions**:
- `Pixelis-RFT-Curiosity` > `Pixelis-RFT-Base` on thoroughness ratings
- But NOT at the expense of efficiency (no significant decrease)
- Optimal balance in `Pixelis-RFT-Full` model

**Rationale**: Curiosity rewards encourage exploration of uncertain regions while the multi-component reward system maintains focus on the task.

## Evaluation Criteria

### 1. Correctness (Sanity Check)
- **Scale**: Binary (Correct/Incorrect)
- **Purpose**: Verify that reasoning quality improvements don't sacrifice accuracy
- **Weight**: Not included in primary analysis but used for filtering

### 2. Coherence & Logicality
- **Scale**: 5-point Likert (1=Very Incoherent to 5=Very Coherent)
- **Aspects Evaluated**:
  - Logical flow between steps
  - Absence of contradictions
  - Clear cause-effect relationships
  - Consistent use of information

### 3. Efficiency & Intelligence
- **Scale**: 5-point Likert (1=Very Inefficient to 5=Very Efficient)
- **Aspects Evaluated**:
  - Directness of approach
  - Avoidance of redundant operations
  - Strategic visual exploration
  - Apparent understanding of the task

### 4. Thoroughness (Secondary)
- **Scale**: 5-point Likert (1=Incomplete to 5=Very Thorough)
- **Aspects Evaluated**:
  - Coverage of relevant visual areas
  - Consideration of alternatives
  - Depth of analysis

## Evaluation Dimensions

### Trajectory-Level Assessment
Each complete reasoning trajectory is evaluated holistically, considering:
1. **Opening Strategy**: How well does the model begin its investigation?
2. **Middle Progression**: Is there clear progress toward the goal?
3. **Resolution Quality**: How effectively does it reach and justify its conclusion?

### Step-Level Patterns
Annotators also note specific patterns:
- **Positive Indicators**:
  - Progressive refinement of understanding
  - Efficient use of visual tools
  - Clear hypothesis formation and testing
  
- **Negative Indicators**:
  - Repetitive or circular reasoning
  - Unnecessary backtracking
  - Tool misuse or overuse
  - Logical inconsistencies

## Statistical Analysis Plan

### Primary Analyses
1. **Hypothesis Testing**:
   - Wilcoxon signed-rank test for paired comparisons
   - Mann-Whitney U test for independent samples
   - Bonferroni correction for multiple comparisons

2. **Effect Size Calculation**:
   - Cohen's d for continuous measures
   - Rank-biserial correlation for ordinal data

3. **Inter-Rater Reliability**:
   - Fleiss' Kappa for categorical agreement
   - Intraclass Correlation Coefficient (ICC) for ordinal ratings

### Secondary Analyses
1. **Factor Analysis**: Identify latent dimensions in human ratings
2. **Regression Analysis**: Predict overall quality from component ratings
3. **Cluster Analysis**: Identify common reasoning patterns

## Quality Control Measures

### Annotator Training
1. **Calibration Session**: 2-hour training with example trajectories
2. **Practice Round**: 20 practice annotations with feedback
3. **Qualification Test**: Must achieve κ > 0.6 agreement with expert annotations

### Data Quality Checks
1. **Attention Checks**: Include obvious cases to detect random responding
2. **Time Tracking**: Flag suspiciously fast annotations
3. **Consistency Monitoring**: Track intra-annotator agreement over time

### Disagreement Resolution
1. **Threshold**: Disagreements where σ > 1.5 on 5-point scale
2. **Resolution**: Expert adjudication by senior researchers
3. **Documentation**: Record resolution rationale for methodology transparency

## Ethical Considerations

### Annotator Welfare
- Maximum 2-hour annotation sessions with breaks
- Fair compensation at $25/hour minimum
- Right to withdraw without penalty

### Bias Mitigation
- Diverse annotator pool (different backgrounds, expertise levels)
- Randomized presentation order
- Balanced dataset composition

### Data Privacy
- No personally identifiable information in trajectories
- Secure storage of annotation data
- Anonymized reporting of results

## Success Metrics

### Minimum Viable Results
- At least one hypothesis confirmed with p < 0.05
- Inter-annotator agreement κ > 0.4 (fair agreement)
- Clear ranking of models on at least one dimension

### Target Outcomes
- All primary hypotheses confirmed
- κ > 0.6 (substantial agreement)
- Clear evidence of reasoning quality improvements
- Actionable insights for future development

## Timeline & Resources

### Phase Timeline
- Week 1: Interface development and testing
- Week 2: Annotator recruitment and training
- Week 3-4: Main annotation period
- Week 5: Analysis and reporting

### Resource Requirements
- 3-5 expert annotators
- 300 evaluation samples (100 questions × 3 annotations each)
- Estimated 150 person-hours of annotation
- Computational resources for trajectory generation

## Reporting Guidelines

### Publication Requirements
- Full disclosure of annotation protocol
- Complete inter-annotator agreement statistics
- Both successful and failed hypotheses reported
- Raw anonymized data made available

### Supplementary Materials
- Annotation interface screenshots
- Example trajectories with ratings
- Detailed statistical analyses
- Annotator guidelines document