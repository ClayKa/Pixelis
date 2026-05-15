
---

### **Action Plan: Enhancing SFT Data with "Minimal Perturbation" Augmentations**

**Objective:** To further increase the quality and challenge of our SFT dataset by introducing two new, highly sophisticated "Minimal Perturbation" augmentation strategies. These augmentations will create advanced trap samples that are nearly correct, forcing the model to develop a deeper, more robust understanding of both visual precision and logical soundness.

**Module to Modify:** `core/data_generation/trajectory_augmenter.py`
**Configuration File to Modify:** `configs/data_fusion_manifest.yaml`

---

#### **Part 1: [New Augmentation] Implement "Perceptual Near-Miss" Trap Samples**

**Goal:** To create trap samples where the visual action is subtly incorrect, teaching the model to pay close attention to precise geometric details.

**Core Idea:** Instead of a completely wrong action (e.g., zooming to a random corner), we will introduce a **slightly perturbed, almost correct** action.

**Implementation Specification (in `TrajectoryAugmenter`):**

1.  **Create a New Method:** `_augment_perceptual_near_miss(self, golden_sample: Dict) -> Dict:`
2.  **Logic:**
    a.  Duplicate the `golden_sample`.
    b.  Find the first `action` step in the trajectory (e.g., `ZOOM-IN` or `SEGMENT_OBJECT_AT`).
    c.  Get the correct bounding box (`bbox`) from its parameters.
    d.  **Apply Perturbation:** Randomly select **one** coordinate of the `bbox` (e.g., `x1`) and add a small, random delta to it (e.g., `x1_perturbed = x1 + (image_width * 0.05)`). This shifts the box slightly (by 5% of the image width).
    e.  Replace the `bbox` in the action's parameters with this new, slightly incorrect `perturbed_bbox`.
    f.  **Modify the `final_answer` and `provenance`**: Crucially, the `final_answer` for this new sample must now be marked as **incorrect**. You should also add a tag to its provenance, e.g., `sample['provenance']['trap_type'] = 'perceptual_near_miss'`.
    g.  Return the new, augmented trap sample.

**Example Outcome:**
*   **Original Golden Action:** `ZOOM-IN(bbox=[100, 100, 150, 150])`
*   **New Trap Action:** `ZOOM-IN(bbox=[105, 100, 150, 150])`
*   **Why it's powerful:** This teaches the model that being "close" is not "correct". It must be precise.

---
#### **Part 2: [New Augmentation] Implement "Logical Fallacy" Trap Samples**

**Goal:** To create trap samples where the visual perception is correct, but the textual reasoning contains a subtle logical flaw.

**Implementation Specification (in `TrajectoryAugmenter`):**

1.  **Create a New Method:** `_augment_logical_fallacy(self, golden_sample: Dict) -> Dict:`
2.  **This requires an LLM call.** This is a more advanced form of augmentation.
3.  **Logic:**
    a.  Duplicate the `golden_sample`.
    b.  Extract the `question` and the final `thought` from the golden trajectory. The final thought contains the correct reasoning (e.g., "The cat's area is 12800 and the dog's area is 15210. Therefore, the dog is larger.").
    c.  **Construct a "Fallacy Prompt"** and send it to the LLM:
        ```
        You are an expert in creating educational trap examples. I have a question and a correct reasoning step. Your task is to rewrite the reasoning to contain a subtle but definite logical fallacy, while still arriving at the INCORRECT final answer.

        Original Question: "Which animal is larger, the cat or the dog?"
        Correct Final Thought: "The cat's area is 12800 and the dog's area is 15210. Therefore, the dog is larger."
        
        Now, provide a new, flawed reasoning string.
        Example of a flawed reasoning: "I have the properties. The cat's area is 12800 and the dog's area is 15210. Since 12800 is a smaller number, it means the cat is a more compact and therefore larger animal."
        ```
    d.  Take the **flawed reasoning** returned by the LLM and replace the final `thought` in the new sample's trajectory with it.
    e.  Ensure the `final_answer` is now the **incorrect** one (e.g., "The cat is larger than the dog.").
    f.  Tag the sample: `sample['provenance']['trap_type'] = 'logical_fallacy'`.
    g.  Return the new trap sample.

**Why it's powerful:** This teaches the model to be a **critical thinker**. It must learn to not blindly trust the textual reasoning chain, but to validate it against the observed evidence.

---
#### **Part 3: Update the Augmentation Proportions**

**Goal:** To integrate these new, advanced trap samples into your final dataset composition without dramatically increasing the total number of trap samples.

**File to Modify:** `configs/data_fusion_manifest.yaml`

**Action:**
Subdivide the `trap_samples` proportion to include these new types.

**Revised `trajectory_augmentation` Configuration:**
```yaml
# In data_fusion_manifest.yaml

trajectory_augmentation:
  proportions:
    golden_positive: 0.6  # 60% are correct samples
    self_correction: 0.2  # 20% teach recovery from errors
    
    # [REVISED] The remaining 20% for traps is now subdivided
    trap_samples:
      total_proportion: 0.2
      # The proportions below are relative to the trap samples (sum to 1.0)
      sub_types:
        - name: "process_negative" # The original, more obvious traps
          proportion: 0.5 # 50% of traps are standard
        - name: "perceptual_near_miss"
          proportion: 0.25 # 25% are subtle perception traps
        - name: "logical_fallacy"
          proportion: 0.25 # 25% are subtle logic traps
```

**Your `TrajectoryAugmenter.process()` method will now:**
1.  Identify the number of samples to convert to traps (e.g., 20% of the input).
2.  Further divide that number according to the `sub_types` proportions.
3.  Call the correct augmentation method (`_augment_process_negative`, `_augment_perceptual_near_miss`, etc.) for each subset.

This plan provides a clear path to implementing a highly sophisticated data augmentation strategy. By introducing these "near-miss" and "logical fallacy" traps, you are creating a training curriculum that will forge an exceptionally robust and intelligent model.