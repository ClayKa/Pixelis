### **Project Proposal: Pixelis - An Evolving Agent for Pixel-Space Reasoning**

**1. Executive Summary**

This project introduces **Pixelis**, a novel vision-language agent designed to reason directly within the pixel space of images and videos. Current methodologies, such as Chain-of-Thought, confine reasoning to the textual domain, creating an information bottleneck and limiting interaction with rich visual data. Building upon the foundational concept of "pixel-space reasoning" from Pixel-Reasoner, which endows models with visual operations like `ZOOM-IN`, our work proposes a revolutionary training and deployment architecture. We achieve this by synergistically combining an enhanced **Reinforcement Fine-Tuning (RFT)** paradigm for robust offline training with a state-of-the-art **Test-Time Representation Learning (TTRL)** engine for continuous online evolution. The final `Pixelis-Online` agent will not only master a greatly expanded set of visual tools but will also possess the ability to learn, adapt, and improve its spatial reasoning capabilities throughout its deployment lifetime.

**2. Core Innovations & Technical Approach**

Our methodology is structured across two primary phases, designed to overcome the "learning trap" inherent in teaching agents new, complex skills.

**Phase 1: Offline Training via an Enhanced RFT Framework**

The initial phase focuses on forging a powerful and intelligent base agent. This process extends the core ideas of Reason-RFT through several key innovations:

*   **Expanded & Verifiable Toolset:** We move beyond basic operations by integrating a pluggable registry of advanced visual tools, including `SEGMENT_OBJECT_AT`, `GET_PROPERTIES`, `READ_TEXT`, and `TRACK_OBJECT`.
*   **"Explore & Focus" Reward Shaping:** At the heart of our RFT process is a sophisticated, dual-component reward system designed to cultivate an optimal reasoning policy:
    *   A **Curiosity-Driven Reward (`R_curiosity`)**, based on a forward dynamics model, incentivizes the agent to explore novel states and actively use its visual tools, directly counteracting policy stagnation.
    *   A **Trajectory Coherence Reward (`R_coherence`)**, based on semantic consistency, ensures that this exploration is logical and non-repetitive, guiding the agent to form structured, interpretable reasoning paths.
*   **Automated Curriculum Learning:** We employ a performance-driven curriculum that intelligently introduces these reward components only when the agent has demonstrated mastery over simpler objectives, ensuring a stable and efficient learning process powered by **GRPO (Generalized Reward Policy Optimization)**.

**Phase 2: Online Evolution via a State-of-the-Art TTRL Engine**

This phase transforms the powerful offline-trained agent into a continuously evolving entity. Our online engine represents a significant leap forward for TTRL:

*   **Asynchronous Architecture:** A decoupled, multi-process architecture ensures that low-latency inference is never blocked by the learning process.
*   **Intelligent Experience Buffer:** The agent is equipped with a long-term memory that uses a **hybrid k-NN index** (combining visual and intentional similarity) for retrieving relevant past experiences. A **time-boosted, value-aware priority** mechanism ensures that key knowledge is retained and "bad memories" are down-weighted over time.
*   **Gated, Conservative Updates:** The decision to learn is governed by a **confidence-gating mechanism** based on a temporal ensemble of the agent's current and past predictions. When an update is triggered, a **"three-tiered safety system"** (KL-penalty, gradient clipping, and EMA smoothing) ensures the update is stable and does not disrupt the model's existing knowledge base.

**3. Evaluation and Expected Outcomes**

We will conduct a comprehensive and rigorous evaluation to validate the superiority of the Pixelis framework.

*   **Ablation & Comparative Studies:** A series of carefully designed ablation studies will surgically isolate and quantify the contribution of our core innovations (Curiosity reward, Coherence reward, and the Online TTRL engine). We will directly compare against a faithfully reimplemented **`Pixel-Reasoner-Baseline`**.
*   **Custom Capabilities Benchmark:** To demonstrate the value of the new visual operations, we will create a new, challenging benchmark of spatial reasoning tasks that are impossible to solve without the enhanced toolset.
*   **Stress Testing:** The final `Pixelis-Online` agent will be subjected to extensive testing for continual learning, domain adaptation, efficiency, and robustness to noisy data.