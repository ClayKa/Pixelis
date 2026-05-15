### **Action Plan: Implementing a "Generate-Until-Valid" Loop**

**Objective:** To enhance the `BaseTaskGenerator`'s main generation loop to ensure it always produces the exact `target_sample_count` of **valid** CoTA samples. This will be achieved by implementing a retry mechanism that re-attempts generation for a single sample until it passes the subclass's validation logic.

**Guiding Principle:** We want to guarantee the final output quantity without sacrificing quality. The system should be resilient to sporadic LLM failures or non-compliant responses.

---
#### **Core Refactoring: Modify the `BaseTaskGenerator.generate()` Method**

The entire change will be centralized in the `generate()` method of your base class. The subclasses (`DetailPerceptionTaskGenerator`, etc.) will remain unchanged, as their `_validate_and_process_response` method already correctly returns `None` for invalid samples.

**File to Modify:** `core/data_generation/base_generator.py`

**Implementation Specification:**

The main `for` loop will be transformed into a `while` loop, which will continue until the list of **valid** generated samples reaches the target count.

**Revised `generate()` Method (Final Production Version):**

```python
# In core/data_generation/base_generator.py

class BaseTaskGenerator(ABC):
    
    # ... (__init__, _load_prompt_template, _call_llm_api, etc. remain the same) ...
    # ... (_build_context_placeholders and _validate_and_process_response are implemented in subclasses) ...

    def generate(self, num_samples: int, checkpoint_path: Path) -> List[Dict]:
        """
        [REVISED WITH GENERATE-UNTIL-VALID LOGIC]
        Main generation method that orchestrates the entire process, ensuring
        the final output contains exactly `num_samples` of valid items.
        """
        generated_samples = [] # This list will only store VALID samples
        self.start_time = time.time()
        
        # ... (Checkpoint loading logic remains the same, loading into generated_samples) ...

        start_index = len(generated_samples)
        if start_index >= num_samples:
            # ... (Logic to handle already-completed generation remains the same) ...
            return generated_samples[:num_samples]

        # --- START OF CRITICAL REFACTOR ---

        # Initialize progress bar for VALID samples
        pbar = tqdm(
            initial=start_index,
            total=num_samples,
            desc=f"Generating VALID '{self.task_name}'"
        )
        
        # Use a while loop to ensure we get the exact number of valid samples
        max_total_attempts = int(num_samples * 2.5) # Failsafe: allow 150% failure rate
        current_total_attempts = 0

        while len(generated_samples) < num_samples and current_total_attempts < max_total_attempts:
            current_total_attempts += 1
            
            try:
                # 1. Subclass builds the context
                context_placeholders, initial_metadata = self._build_context_placeholders()

                # 2. Base class formats the prompt
                final_prompt = self.prompt_template.format(**context_placeholders)
                
                # 3. Base class calls the API
                llm_response = self._call_llm_api(final_prompt)
                
                # 4. Subclass validates the response. This is the critical gate.
                # It returns a valid sample dict, or None if validation fails.
                cota_sample = self._validate_and_process_response(llm_response, context_placeholders)

                if cota_sample:
                    # --- SUCCESS CASE ---
                    # 5. Finalize metadata and add the VALID sample to our list
                    cota_sample = self._finalize_metadata(cota_sample, initial_metadata)
                    generated_samples.append(cota_sample)
                    
                    # 6. Update progress bar ONLY on success
                    pbar.update(1)
                    
                    self.generation_stats['samples_generated'] += 1
                    pbar.set_postfix({
                        'valid': self.generation_stats['samples_generated'],
                        'invalid': self.generation_stats['samples_invalid'],
                        'failed': self.generation_stats['samples_failed'],
                    })

                else:
                    # --- VALIDATION FAILED CASE ---
                    # The warning is logged inside the validation method.
                    # We just increment the stat and the loop will continue.
                    self.generation_stats['samples_invalid'] += 1
                    continue
                
                # Checkpointing logic (can be based on number of valid samples)
                if len(generated_samples) % self.global_config.get('checkpoint_every_n_samples', 100) == 0:
                    self._save_checkpoint(generated_samples, checkpoint_path)

            except Exception as e:
                # --- HARD FAILURE CASE (e.g., API is down) ---
                logger.error(f"A hard error occurred during generation attempt: {e}", exc_info=True)
                self.generation_stats['samples_failed'] += 1

        pbar.close()

        if current_total_attempts >= max_total_attempts:
            logger.error(f"FATAL: Reached max generation attempts ({max_total_attempts}) for task '{self.task_name}' "
                         f"but only produced {len(generated_samples)} valid samples. "
                         f"There is likely a persistent issue with the prompt or the API.")

        # --- END OF CRITICAL REFACTOR ---

        # ... (Final save and statistics logging) ...
        
        return generated_samples
```

---
