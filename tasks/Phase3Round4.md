**Round 4: Inference Acceleration & Optimization**

*   **Task 1: Profile and Identify Bottlenecks.**
    *   **Action:** Use `torch.profiler` to get a detailed breakdown. Your hypothesis should be that the main bottlenecks are: 1) The base model's forward pass, 2) The k-NN search, and 3) The dynamics model forward pass for the curiosity reward.
*   **Task 2: Apply Standard Optimizations.**
    *   **Action:**
        1.  **Compile:** Use `torch.compile()` on the main model and also on the `DynamicsModel` from the curiosity module.
        2.  **Quantize:** Use Post-Training Quantization (PTQ) to convert the main model's weights to INT8. Re-evaluate to ensure minimal performance drop.
        3.  **Implement Flash Attention:**
            *   **Action:** When loading your base model using the `transformers` library, explicitly enable Flash Attention 2 by passing the `attn_implementation="flash_attention_2"` argument.
            *   **Goal:** To accelerate the most computationally expensive part of the Transformer model with a state-of-the-art implementation.
*   **Task 3: Implement Service-Level Optimizations.**
    *   **Action:** To move from a single-inference script to a high-throughput service architecture, you will need to wrap your inference logic in a web server (e.g., using FastAPI). You can leverage existing high-performance serving frameworks like TGI (Text Generation Inference) or vLLM, which often have these features built-in.
    *   **Sub-Task 3.1: Implement Dynamic Batching.**
        *   **Action:** Configure the inference server to use dynamic batching, setting parameters like `max_batch_size` and `max_wait_time` to balance throughput and latency.
    *   **Sub-Task 3.2: Implement Multi-Level Caching.**
        *   **Action:** Implement an LRU cache. At a minimum, cache the final inference results. For advanced optimization, cache the results of expensive, deterministic modules like the k-NN search.
*   **Task 4: Implement Task-Specific Optimizations.**
    *   **Action:**
        1.  **Approximate k-NN:** For the temporal ensemble, experiment with using a smaller, approximate FAISS index or reducing `k` to speed up retrieval.
        2.  **Cache Reward Computations:** If a state is revisited, consider using a cached reward value instead of re-computing, especially for the coherence score.
*   **Task 5: (Optional) Export to a Dedicated Inference Engine.**
    *   **Action:** For peak performance, use a workflow like PyTorch -> ONNX -> TensorRT to build a highly optimized version of the entire inference engine for deployment.