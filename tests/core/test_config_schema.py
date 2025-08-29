"""
Tests for config_schema.py module.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.config_schema import (
    TrainingMode,
    VotingStrategy,
    RewardType,
    ModelConfig,
    TrainingConfig,
    RewardConfig,
    CurriculumConfig,
    OnlineConfig,
    DataConfig,
    ExperimentConfig,
    SystemConfig,
    PixelisConfig
)


class TestEnums:
    """Test enum classes."""
    
    def test_training_mode_enum(self):
        """Test TrainingMode enum values."""
        assert TrainingMode.SFT.value == "sft"
        assert TrainingMode.RFT.value == "rft"
        assert TrainingMode.ONLINE.value == "online"
        assert len(TrainingMode) == 3
    
    def test_voting_strategy_enum(self):
        """Test VotingStrategy enum values."""
        assert VotingStrategy.MAJORITY.value == "majority"
        assert VotingStrategy.WEIGHTED.value == "weighted"
        assert VotingStrategy.CONFIDENCE.value == "confidence"
        assert VotingStrategy.ENSEMBLE.value == "ensemble"
        assert len(VotingStrategy) == 4
    
    def test_reward_type_enum(self):
        """Test RewardType enum values."""
        assert RewardType.TASK.value == "task"
        assert RewardType.CURIOSITY.value == "curiosity"
        assert RewardType.COHERENCE.value == "coherence"
        assert RewardType.COMBINED.value == "combined"
        assert len(RewardType) == 4


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test default values for ModelConfig."""
        config = ModelConfig()
        assert config.model_name == "Qwen/Qwen2.5-VL-7B"
        assert config.model_type == "qwen2_vl"
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.device_map == "auto"
        assert config.torch_dtype == "float16"
    
    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_name="custom/model",
            model_type="custom_type",
            load_in_8bit=True,
            device_map="cpu"
        )
        assert config.model_name == "custom/model"
        assert config.model_type == "custom_type"
        assert config.load_in_8bit is True
        assert config.device_map == "cpu"


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_training_config_defaults(self):
        """Test default values for TrainingConfig."""
        config = TrainingConfig()
        # Test basic attributes that should exist
        assert hasattr(config, '__dataclass_fields__')
    
    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(learning_rate=0.001, batch_size=16)
        # Verify the values were set
        assert config.learning_rate == 0.001
        assert config.batch_size == 16


class TestDataConfig:
    """Test DataConfig dataclass."""
    
    def test_data_config_initialization(self):
        """Test DataConfig initialization."""
        config = DataConfig()
        assert hasattr(config, '__dataclass_fields__')
        
    def test_data_config_with_values(self):
        """Test DataConfig with specific values."""
        config = DataConfig(train_data_path="test/path", max_seq_length=512)
        assert config.train_data_path == "test/path"
        assert config.max_seq_length == 512


class TestRewardConfig:
    """Test RewardConfig dataclass."""
    
    def test_reward_config_initialization(self):
        """Test RewardConfig initialization."""
        config = RewardConfig()
        assert hasattr(config, '__dataclass_fields__')
        
    def test_reward_config_with_values(self):
        """Test RewardConfig with specific values and normalization."""
        config = RewardConfig(task_reward_weight=0.8, curiosity_reward_weight=0.2, coherence_reward_weight=0.0)
        # The weights get normalized to sum to 1.0 in __post_init__
        assert config.task_reward_weight == 0.8  # 0.8/1.0 = 0.8
        assert config.curiosity_reward_weight == 0.2  # 0.2/1.0 = 0.2
        assert config.coherence_reward_weight == 0.0  # 0.0/1.0 = 0.0


class TestCurriculumConfig:
    """Test CurriculumConfig dataclass."""
    
    def test_curriculum_config_initialization(self):
        """Test CurriculumConfig initialization."""
        config = CurriculumConfig()
        assert hasattr(config, '__dataclass_fields__')


class TestOnlineConfig:
    """Test OnlineConfig dataclass."""
    
    def test_online_config_initialization(self):
        """Test OnlineConfig initialization."""
        config = OnlineConfig()
        assert hasattr(config, '__dataclass_fields__')


class TestSystemConfig:
    """Test SystemConfig dataclass."""
    
    def test_system_config_initialization(self):
        """Test SystemConfig initialization."""
        config = SystemConfig()
        assert hasattr(config, '__dataclass_fields__')


class TestPixelisConfig:
    """Test PixelisConfig dataclass."""
    
    def test_pixelis_config_initialization(self):
        """Test PixelisConfig initialization."""
        config = PixelisConfig()
        assert hasattr(config, '__dataclass_fields__')


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""
    
    def test_experiment_config_initialization(self):
        """Test ExperimentConfig initialization."""
        config = ExperimentConfig()
        assert hasattr(config, '__dataclass_fields__')
        
    def test_experiment_config_with_values(self):
        """Test ExperimentConfig with specific values."""
        config = ExperimentConfig(experiment_name="test_experiment", run_name="test_run")
        assert config.experiment_name == "test_experiment"
        assert config.run_name == "test_run"


class TestConfigIntegration:
    """Test configuration integration scenarios."""
    
    def test_all_configs_are_dataclasses(self):
        """Test that all config classes are proper dataclasses."""
        configs = [
            ModelConfig, TrainingConfig, RewardConfig, CurriculumConfig, 
            OnlineConfig, DataConfig, ExperimentConfig, SystemConfig, PixelisConfig
        ]
        
        for config_class in configs:
            # Check that they have the dataclass marker
            assert hasattr(config_class, '__dataclass_fields__')
            # Check that they can be instantiated
            instance = config_class()
            assert instance is not None
    
    def test_config_field_access(self):
        """Test that config fields can be accessed and modified."""
        model_config = ModelConfig()
        original_name = model_config.model_name
        
        # Modify a field
        model_config.model_name = "new_model"
        assert model_config.model_name == "new_model"
        assert model_config.model_name != original_name
    
    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = ModelConfig(model_name="test", load_in_8bit=True)
        config2 = ModelConfig(model_name="test", load_in_8bit=True)
        config3 = ModelConfig(model_name="different", load_in_8bit=True)
        
        assert config1 == config2
        assert config1 != config3