# Pixelis Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Experience Buffer: High-Reliability Design Decisions](#experience-buffer-high-reliability-design-decisions)
4. [Training Pipeline](#training-pipeline)
5. [Online Learning Architecture](#online-learning-architecture)
6. [Data Flow](#data-flow)

## System Overview

Pixelis is a novel vision-language agent that operates directly in pixel space, combining offline reinforcement learning with online test-time reinforcement learning (TTRL) for continuous adaptation.

### Key Design Principles
- **Pixel-Space Reasoning**: Direct interaction with visual data at the pixel level
- **Dual Learning Paradigm**: Offline training for foundation + online adaptation for evolution
- **Production-Grade Reliability**: Enterprise-level consistency and fault tolerance
- **Modular Architecture**: Pluggable components for flexibility and scalability

## Core Components

### 1. Visual Operation Registry
- **Location**: `core/modules/operation_registry.py`
- **Purpose**: Central management of pixel-space operations
- **Design**: Singleton pattern with plugin-based operations
- **Operations**: `ZOOM_IN`, `SEGMENT_OBJECT_AT`, `GET_PROPERTIES`, `READ_TEXT`, `TRACK_OBJECT`

### 2. Reward System
- **Location**: `core/modules/reward_shaping_enhanced.py`
- **Components**:
  - Task Reward (R_task): Direct task performance
  - Curiosity Reward (R_curiosity): Exploration incentive
  - Coherence Reward (R_coherence): Logical consistency
- **Design**: Orchestrator pattern with normalized components

### 3. Data Structures
- **Location**: `core/data_structures.py`
- **Key Classes**:
  - `Experience`: Core unit of learning with trajectory and embeddings
  - `Trajectory`: Sequence of actions with rewards
  - `Action`: Individual reasoning or visual operation
  - `UpdateTask`: Asynchronous learning task

## Experience Buffer: High-Reliability Design Decisions

The Experience Buffer is designed as a production-grade component with enterprise-level reliability. Here are the key architectural decisions and their trade-offs:

### On Durability (WAL + Snapshots)

**Decision**: "We employ a Write-Ahead Log (WAL) and periodic snapshotting mechanism for data persistence."

**Rationale / Pro**: "This provides maximum crash consistency. No acknowledged write operation will ever be lost, even in the case of a sudden system failure."

**Trade-off / Con**: "The cost of this durability is a minor increase in write latency (due to requiring disk `fsync`) and a higher implementation complexity compared to a simple in-memory buffer."

### On Read Availability (Asynchronous Index Rebuilding)

**Decision**: "The FAISS index is rebuilt asynchronously in a background process, followed by an atomic swap."

**Rationale / Pro**: "This guarantees that read operations (i.e., k-NN searches) are never blocked and always operate with low latency on a fully consistent index."

**Trade-off / Con**: "The trade-off is that the index is not real-time. There is a configurable delay between when an experience is added and when it becomes searchable in the index."

### On Flexibility (Pluggable Backends)

**Decision**: "The persistence layer and FAISS backend are accessed through abstract adapter interfaces, allowing for configurable implementations (e.g., File vs. LMDB, CPU vs. GPU)."

**Rationale / Pro**: "This provides maximum long-term flexibility, allowing the system to be deployed on diverse hardware and to scale to higher throughputs by simply changing a configuration line."

**Trade-off / Con**: "The cost of this flexibility is an added layer of abstraction, which slightly increases the cognitive overhead for developers working on the buffer's core logic."

### Implementation Details

#### Persistence Architecture
```
Experience Buffer
├── Write Path
│   ├── WAL Write (experience_data.wal)
│   ├── Operation Log (index_operations.wal)
│   └── In-Memory Update (deque + dict)
├── Read Path
│   ├── k-NN Search (FAISS index)
│   ├── Priority Sampling
│   └── Direct Access (by ID)
└── Recovery Path
    ├── Load Snapshot
    ├── Replay Data WAL
    └── Replay Operations WAL
```

#### Consistency Model
- **Write Consistency**: Sequential WAL writes with fsync
- **Read Consistency**: Eventually consistent index with atomic swaps
- **Crash Recovery**: Full recovery from snapshot + WAL replay
- **Concurrency Control**: Process-safe locking for writes

#### Performance Characteristics
- **Write Latency**: ~1-5ms (WAL write + fsync)
- **Read Latency**: <1ms (in-memory index)
- **Recovery Time**: O(WAL_size) - typically <10s
- **Index Rebuild**: O(buffer_size) - asynchronous

## Training Pipeline

### Phase 1: Offline Training

#### SFT (Supervised Fine-Tuning)
- **Curriculum Learning**: Progressive difficulty stages
- **LoRA Configuration**: Data-driven rank selection via SVD
- **Data**: CoTA (Chain-of-Thought-Action) trajectories

#### RFT (Reinforcement Fine-Tuning)
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Reward Shaping**: Multi-component with curriculum
- **Trajectory Generation**: Online with reward calculation

### Phase 2: Online Evolution (TTRL)

#### Asynchronous Architecture
```
┌─────────────────┐     Requests      ┌─────────────────┐
│                 │ ────────────────► │                 │
│  Inference      │                   │    Update       │
│   Engine        │ ◄──────────────── │    Worker       │
│                 │     Updates        │                 │
└─────────────────┘                   └─────────────────┘
        │                                      │
        │                                      │
        ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│              Experience Buffer                       │
│  - Multi-factor priority                            │
│  - Hybrid k-NN retrieval                           │
│  - WAL persistence                                 │
└─────────────────────────────────────────────────────┘
```

#### Key Components
1. **Inference Engine** (`core/engine/inference_engine.py`)
   - Temporal ensemble voting
   - Confidence gating
   - Pseudo-label generation

2. **Update Worker** (`core/engine/update_worker.py`)
   - Conservative updates
   - KL divergence constraints
   - EMA model synchronization

3. **Experience Buffer** (`core/modules/experience_buffer_enhanced.py`)
   - Hybrid embeddings (visual + text)
   - Priority-based sampling
   - Value tracking

## Online Learning Architecture

### Safety Mechanisms

#### Three-Tiered Safety System
1. **Behavioral Guardrail**: KL-divergence penalty
2. **Magnitude Guardrail**: Gradient clipping
3. **Temporal Guardrail**: EMA smoothing

### Confidence Gating
- Minimum confidence threshold for updates
- Proportional learning rate adaptation
- Human-in-the-loop option for validation

### Cold Start Strategy
- Conservative mode during buffer warm-up
- Experience collection without updates
- Gradual transition to active learning

## Data Flow

### Training Data Flow
```
Raw Data → CoTA Synthesis → Quality Filtering → Curriculum Staging → Training
```

### Online Learning Data Flow
```
User Input → Inference → Voting → Experience Buffer → Update Queue → Model Update
                ↑                         ↓
                └──── k-NN Retrieval ─────┘
```

### Persistence Data Flow
```
Experience → WAL Write → In-Memory Update → Index Update → Snapshot
                ↓                                ↑
                └────── Recovery Path ──────────┘
```

## Deployment Considerations

### Hardware Requirements
- **GPU**: Recommended for FAISS index (fallback to CPU)
- **Memory**: ~16GB for buffer + model
- **Storage**: Fast SSD for WAL operations
- **CPU**: Multi-core for async processes

### Configuration Management
- **Hydra + OmegaConf**: Structured configuration
- **Environment Variables**: Override for deployment
- **Dynamic Parameters**: Runtime adaptation

### Monitoring and Observability
- **Metrics**: WandB integration
- **Logging**: Structured JSON logs
- **Alerting**: Automated threshold monitoring
- **Profiling**: Performance tracking

## Future Enhancements

### Planned Improvements
1. **Distributed Buffer**: Multi-node experience sharing
2. **Advanced Indexing**: Hierarchical navigable small worlds
3. **Active Learning**: Query synthesis for exploration
4. **Model Versioning**: A/B testing support

### Research Directions
1. **Meta-Learning**: Learning to learn from experiences
2. **Causal Reasoning**: Understanding action-effect relationships
3. **Multi-Modal Fusion**: Beyond visual-text to audio/video
4. **Federated Learning**: Privacy-preserving distributed training