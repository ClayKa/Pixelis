"""
Configuration Schema Module

Defines the strict structure, expected data types, and default values for all
configuration parameters using Python dataclasses. This serves as the single
source of truth for the project's entire configuration contract.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class TrainingMode(Enum):
    """Enumeration of training modes."""
    SFT = "sft"  # Supervised Fine-Tuning
    RFT = "rft"  # Reinforcement Fine-Tuning
    ONLINE = "online"  # Online learning


class VotingStrategy(Enum):
    """Enumeration of voting strategies."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    CONFIDENCE = "confidence"
    ENSEMBLE = "ensemble"


class RewardType(Enum):
    """Enumeration of reward types."""
    TASK = "task"
    CURIOSITY = "curiosity"
    COHERENCE = "coherence"
    COMBINED = "combined"


@dataclass
class ModelConfig:
    """Configuration for model architecture and loading."""
    
    # Model identification
    model_name: str = "Qwen/Qwen2.5-VL-7B"
    model_type: str = "qwen2_vl"
    
    # Model loading parameters
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"
    
    # Model architecture parameters
    max_length: int = 4096
    max_pixels: int = 4014080
    min_pixels: int = 401408
    image_resolution: int = 448
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 32  # Will be dynamically determined
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Flash attention
    use_flash_attention: bool = True
    
    # Gradient checkpointing
    gradient_checkpointing: bool = True
    
    # Model paths
    base_model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot enable both 8-bit and 4-bit loading")
        
        if self.max_pixels < self.min_pixels:
            raise ValueError("max_pixels must be >= min_pixels")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training mode
    mode: TrainingMode = TrainingMode.SFT
    
    # Basic training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"
    num_cycles: float = 0.5
    
    # Evaluation
    eval_steps: int = 500
    eval_batch_size: int = 8
    evaluation_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Checkpointing
    output_dir: str = "./outputs"
    save_strategy: str = "steps"
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    
    # Distributed training
    ddp_find_unused_parameters: bool = False
    fsdp: Optional[str] = None
    deepspeed: Optional[str] = None
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Reward weights
    task_reward_weight: float = 1.0
    curiosity_reward_weight: float = 0.3
    coherence_reward_weight: float = 0.2
    
    # Curiosity reward parameters
    curiosity_beta: float = 0.2  # Forward model weight
    curiosity_eta: float = 0.5   # Scaling factor
    intrinsic_reward_scale: float = 0.1
    
    # Coherence reward parameters
    coherence_threshold: float = 0.7
    repetition_penalty: float = 0.5
    trajectory_min_length: int = 2
    
    # Tool usage penalties
    tool_misuse_penalty: float = -0.1
    excessive_tool_use_threshold: int = 10
    excessive_tool_use_penalty: float = -0.2
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_clip_value: float = 10.0
    
    # Reward curriculum
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"step": 0, "weights": {"task": 1.0, "curiosity": 0.0, "coherence": 0.0}},
        {"step": 5000, "weights": {"task": 0.7, "curiosity": 0.2, "coherence": 0.1}},
        {"step": 10000, "weights": {"task": 0.5, "curiosity": 0.3, "coherence": 0.2}},
    ])
    
    def __post_init__(self):
        """Validate reward configuration."""
        total_weight = (
            self.task_reward_weight +
            self.curiosity_reward_weight +
            self.coherence_reward_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.task_reward_weight /= total_weight
            self.curiosity_reward_weight /= total_weight
            self.coherence_reward_weight /= total_weight


@dataclass
class OnlineConfig:
    """Configuration for online learning."""
    
    # Confidence gating
    confidence_threshold: float = 0.7
    min_confidence_for_update: float = 0.5
    
    # Learning rate adaptation
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-4
    lr_adaptation_strategy: str = "proportional"  # proportional, fixed, scheduled
    
    # KL divergence constraints
    kl_weight: float = 0.01
    max_kl_divergence: float = 0.05
    
    # EMA model
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_freq: int = 1
    
    # Experience buffer
    buffer_size: int = 10000
    k_neighbors: int = 5
    similarity_metric: str = "cosine"  # cosine, euclidean, manhattan
    
    # FAISS configuration
    faiss_backend: str = "gpu"  # gpu, cpu
    faiss_n_probes: int = 10  # For IVF index
    faiss_use_gpu_fallback: bool = True  # Fallback to CPU if GPU fails
    
    # Persistence configuration
    persistence_backend: str = "file"  # file, lmdb
    persistence_path: str = "./experience_buffer"
    enable_persistence: bool = True
    snapshot_interval: int = 100  # Operations before snapshot
    max_snapshots: int = 3  # Number of snapshots to keep
    
    # Hybrid embedding configuration
    visual_weight: float = 0.7  # Weight for visual embedding
    text_weight: float = 0.3  # Weight for text embedding
    
    # Voting
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED
    min_votes_required: int = 3
    
    # Update worker
    update_queue_size: int = 100
    update_batch_size: int = 1
    update_frequency: int = 1
    
    # Safety mechanisms
    gradient_clip_norm: float = 1.0
    max_updates_per_minute: int = 60
    
    # Human-in-the-Loop (HIL) configuration
    hil_mode_enabled: bool = False  # Enable HIL mode
    hil_review_percentage: float = 0.02  # Percentage of updates to review (2%)
    hil_interface_host: str = "127.0.0.1"  # HIL interface host
    hil_interface_port: int = 7860  # HIL interface port
    hil_auto_approve_timeout: Optional[int] = None  # Auto-approve after timeout (seconds)
    
    def __post_init__(self):
        """Validate online configuration."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("min_learning_rate must be <= max_learning_rate")
        
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        
        if self.visual_weight + self.text_weight != 1.0:
            # Normalize weights if they don't sum to 1
            total = self.visual_weight + self.text_weight
            self.visual_weight /= total
            self.text_weight /= total
        
        if self.faiss_backend not in ["gpu", "cpu"]:
            raise ValueError("faiss_backend must be 'gpu' or 'cpu'")
        
        if self.persistence_backend not in ["file", "lmdb"]:
            raise ValueError("persistence_backend must be 'file' or 'lmdb'")


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Data processing
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "difficulty"  # difficulty, length, random
    difficulty_bins: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    
    # Sampling
    sampling_strategy: str = "uniform"  # uniform, weighted, stratified
    sample_weights: Optional[Dict[str, float]] = None
    
    # Data format
    data_format: str = "json"  # json, jsonl, parquet, csv
    text_column: str = "text"
    label_column: str = "label"
    image_column: str = "image"
    
    def __post_init__(self):
        """Validate data configuration."""
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        
        if self.augmentation_prob < 0 or self.augmentation_prob > 1:
            raise ValueError("augmentation_prob must be between 0 and 1")


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Experiment identification
    experiment_name: str = "pixelis_experiment"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Tracking
    use_wandb: bool = True
    wandb_project: str = "pixelis"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online, offline, disabled
    
    # Reproducibility
    track_artifacts: bool = True
    save_code: bool = True
    save_config: bool = True
    save_environment: bool = True
    
    # Multi-seed runs
    num_seeds: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2024])
    
    # Ablation studies
    ablation_mode: bool = False
    ablation_components: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if self.num_seeds != len(self.seeds):
            # Generate seeds if mismatch
            import random
            random.seed(42)
            self.seeds = [random.randint(0, 10000) for _ in range(self.num_seeds)]


@dataclass
class SystemConfig:
    """Configuration for system-level settings."""
    
    # Hardware
    device: str = "cuda"
    num_gpus: int = 1
    gpu_ids: Optional[List[int]] = None
    
    # Memory management
    max_memory_mb: Optional[int] = None
    empty_cache_freq: int = 100
    
    # Parallelism
    num_workers: int = 4
    dataloader_num_workers: int = 4
    
    # Paths
    cache_dir: str = "./cache"
    temp_dir: str = "./tmp"
    log_dir: str = "./logs"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Debugging
    debug_mode: bool = False
    profile: bool = False
    detect_anomaly: bool = False
    
    def __post_init__(self):
        """Validate system configuration."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("Warning: CUDA not available, falling back to CPU")
        
        if self.num_gpus > torch.cuda.device_count():
            self.num_gpus = torch.cuda.device_count()
            print(f"Warning: Requested {self.num_gpus} GPUs but only {torch.cuda.device_count()} available")


@dataclass
class PixelisConfig:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def validate(self):
        """Validate the entire configuration."""
        # Trigger post_init validation for all sub-configs
        self.model.__post_init__()
        self.training.__post_init__()
        self.reward.__post_init__()
        self.online.__post_init__()
        self.data.__post_init__()
        self.experiment.__post_init__()
        self.system.__post_init__()
        
        # Cross-configuration validation
        if self.training.mode == TrainingMode.ONLINE and not self.online.use_ema:
            print("Warning: Online training without EMA model may be unstable")
        
        if self.training.batch_size * self.training.gradient_accumulation_steps > 64:
            print("Warning: Effective batch size > 64, may require lower learning rate")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "reward": self.reward.__dict__,
            "online": self.online.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__,
            "system": self.system.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PixelisConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "reward" in config_dict:
            config.reward = RewardConfig(**config_dict["reward"])
        if "online" in config_dict:
            config.online = OnlineConfig(**config_dict["online"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "experiment" in config_dict:
            config.experiment = ExperimentConfig(**config_dict["experiment"])
        if "system" in config_dict:
            config.system = SystemConfig(**config_dict["system"])
        
        return config