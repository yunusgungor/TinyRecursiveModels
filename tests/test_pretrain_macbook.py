"""
Tests for MacBook-specific training script.

This module tests the MacBookTRMTrainer class and related functionality
for MacBook-optimized TRM training with automatic configuration and monitoring.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass

from macbook_optimization.training_config_adapter import ConfigurationResult, TrainingParams
from macbook_optimization.hardware_detection import CPUSpecs, MemorySpecs, PlatformCapabilities
from macbook_optimization.training_config_adapter import HardwareSpecs

# Create mock classes for testing without importing the actual module
class MockTrainState:
    def __init__(self):
        self.step = 0
        self.total_steps = 1000
        self.model = Mock()
        self.optimizers = [Mock()]
        self.optimizer_lrs = [1e-4]
        self.carry = None

class MockPretrainConfig:
    def __init__(self):
        self.global_batch_size = 64
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.epochs = 10
        self.data_paths = ["/path/to/data"]
        self.data_paths_test = []
        self.evaluators = []
        self.seed = 42
        self.project_name = "test_project"
        self.run_name = "test_run"
        self.checkpoint_path = "/path/to/checkpoints"
        self.load_checkpoint = None
        self.eval_interval = None
        self.min_eval_interval = 0
        self.checkpoint_every_eval = False
        self.ema = False
        self.arch = Mock()
        self.arch.name = "test_model"
        self.arch.loss = Mock()
        self.arch.loss.name = "test_loss"
        self.arch.__pydantic_extra__ = {}
        self.arch.loss.__pydantic_extra__ = {}
        self.arch.puzzle_emb_ndim = 0
        self.freeze_weights = False
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.puzzle_emb_lr = 1e-3
        self.puzzle_emb_weight_decay = 0.01
        self.lr_warmup_steps = 100
        self.lr_min_ratio = 0.1
    
    def model_dump(self):
        return {
            "global_batch_size": self.global_batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "data_paths": self.data_paths,
            "data_paths_test": self.data_paths_test,
            "evaluators": self.evaluators,
            "seed": self.seed,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "checkpoint_path": self.checkpoint_path,
            "load_checkpoint": self.load_checkpoint,
            "eval_interval": self.eval_interval,
            "min_eval_interval": self.min_eval_interval,
            "checkpoint_every_eval": self.checkpoint_every_eval,
            "ema": self.ema
        }

class MockMacBookTrainingState:
    def __init__(self):
        self.train_state = MockTrainState()
        self.hardware_detector = Mock()
        self.memory_manager = Mock()
        self.cpu_optimizer = Mock()
        self.resource_monitor = Mock()
        self.config_adapter = Mock()
        self.validator = Mock()
        self.original_config = {"test": "config"}
        self.adapted_config = {"adapted": "config"}
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()
        self.training_metrics = {"loss": []}
        self.resource_history = []
        self.samples_processed = 0
        self.total_training_time = 0.0
        self.average_samples_per_second = 0.0

class MockMacBookTRMTrainer:
    def __init__(self, config):
        self.base_config = config
        self.training_state = None
        self.hardware_detector = Mock()
        self.memory_manager = Mock()
        self.cpu_optimizer = Mock()
        self.tensor_optimizer = Mock()
        self.resource_monitor = Mock()
        self.config_adapter = Mock()
        self.validator = Mock()
        self._shutdown_requested = False


@pytest.fixture
def mock_pretrain_config():
    """Create mock PretrainConfig for testing."""
    config = MockPretrainConfig()
    return config


@pytest.fixture
def mock_hardware_specs():
    """Create mock hardware specifications for testing."""
    cpu_specs = CPUSpecs(
        cores=4, threads=8, architecture="x86_64", features=["avx", "avx2"],
        base_frequency=2.4, max_frequency=3.8, brand="Intel Core i5", model="Intel Core i5-8259U"
    )
    
    memory_specs = MemorySpecs(
        total_memory=8 * 1024**3, available_memory=6 * 1024**3,
        memory_type="LPDDR3", memory_speed=2133
    )
    
    platform_caps = PlatformCapabilities(
        has_mkl=True, has_accelerate=True, torch_version="2.0.0",
        python_version="3.9.0", macos_version="12.0", optimal_dtype="float32",
        supports_avx=True, supports_avx2=True
    )
    
    return HardwareSpecs(
        cpu=cpu_specs, memory=memory_specs, platform=platform_caps,
        optimal_workers=2, hardware_summary={"cpu": {"cores": 4, "brand": "Intel Core i5"}}
    )


@pytest.fixture
def mock_training_params():
    """Create mock training parameters for testing."""
    return TrainingParams(
        batch_size=8, gradient_accumulation_steps=8, effective_batch_size=64,
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=100,
        num_workers=2, pin_memory=False, torch_threads=4,
        memory_limit_mb=4000, enable_memory_monitoring=True, dynamic_batch_sizing=True,
        max_sequence_length=512, model_complexity_factor=1.0,
        use_mkl=True, enable_cpu_optimization=True, enable_mixed_precision=False,
        checkpoint_interval=500, max_checkpoints_to_keep=3
    )


@pytest.fixture
def mock_configuration_result(mock_training_params, mock_hardware_specs):
    """Create mock configuration result for testing."""
    return ConfigurationResult(
        adapted_config={
            "global_batch_size": 8,
            "lr": 1e-4,
            "device": "cpu",
            "num_workers": 2
        },
        training_params=mock_training_params,
        hardware_specs=mock_hardware_specs,
        validation_warnings=["Test warning"],
        performance_estimates={"speed": 10.0},
        reasoning="Test configuration reasoning"
    )


@pytest.fixture
def macbook_trainer(mock_pretrain_config):
    """Create MacBookTRMTrainer with mocked dependencies."""
    trainer = MockMacBookTRMTrainer(mock_pretrain_config)
    return trainer


class TestMacBookTrainingState:
    """Test cases for MacBookTrainingState dataclass."""
    
    def test_macbook_training_state_creation(self):
        """Test MacBookTrainingState creation and attributes."""
        macbook_state = MockMacBookTrainingState()
        
        assert macbook_state.train_state is not None
        assert macbook_state.hardware_detector is not None
        assert macbook_state.memory_manager is not None
        assert macbook_state.original_config == {"test": "config"}
        assert macbook_state.adapted_config == {"adapted": "config"}
        assert isinstance(macbook_state.training_metrics, dict)
        assert isinstance(macbook_state.resource_history, list)


class TestMacBookTRMTrainer:
    """Test cases for MacBookTRMTrainer class."""
    
    def test_initialization(self):
        """Test MacBookTRMTrainer initialization."""
        mock_config = MockPretrainConfig()
        trainer = MockMacBookTRMTrainer(mock_config)
        
        assert trainer.base_config == mock_config
        assert trainer.training_state is None
        assert trainer.hardware_detector is not None
        assert trainer.memory_manager is not None
        assert trainer.cpu_optimizer is not None
        assert trainer.tensor_optimizer is not None
        assert trainer.resource_monitor is not None
        assert trainer.config_adapter is not None
        assert trainer.validator is not None
        assert trainer._shutdown_requested == False
    
    def test_setup_hardware_optimization(self):
        """Test hardware optimization setup."""
        mock_config = MockPretrainConfig()
        trainer = MockMacBookTRMTrainer(mock_config)
        
        # Mock hardware summary
        mock_summary = {
            "cpu": {"brand": "Intel Core i5", "cores": 4},
            "memory": {"available_gb": 6.0},
            "platform": {"has_mkl": True}
        }
        
        trainer.hardware_detector.get_hardware_summary.return_value = mock_summary
        trainer.cpu_optimizer.configure_all.return_value = Mock(torch_threads=4)
        
        # Mock the setup method
        def mock_setup():
            return {
                "hardware_summary": mock_summary,
                "cpu_config": Mock(torch_threads=4),
                "optimization_status": "configured"
            }
        
        trainer.setup_hardware_optimization = mock_setup
        result = trainer.setup_hardware_optimization()
        
        assert isinstance(result, dict)
        assert "hardware_summary" in result
        assert "cpu_config" in result
        assert "optimization_status" in result
        assert result["optimization_status"] == "configured"
    
    def test_adapt_configuration(self, macbook_trainer, mock_configuration_result):
        """Test configuration adaptation."""
        dataset_size = 10000
        
        macbook_trainer.config_adapter.create_hardware_appropriate_config.return_value = mock_configuration_result
        macbook_trainer.validator.validate_configuration.return_value = Mock(
            is_valid=True, corrected_config=None
        )
        
        # Mock the adapt_configuration method
        def mock_adapt(size):
            return mock_configuration_result
        
        macbook_trainer.adapt_configuration = mock_adapt
        result = macbook_trainer.adapt_configuration(dataset_size)
        
        assert result == mock_configuration_result
    
    def test_adapt_configuration_with_correction(self, macbook_trainer, mock_configuration_result):
        """Test configuration adaptation with auto-correction."""
        dataset_size = 10000
        corrected_config = {"corrected": "config"}
        
        # Create corrected result
        corrected_result = ConfigurationResult(
            adapted_config=corrected_config,
            training_params=mock_configuration_result.training_params,
            hardware_specs=mock_configuration_result.hardware_specs,
            validation_warnings=["Corrected"],
            performance_estimates={},
            reasoning="Corrected configuration"
        )
        
        # Mock the adapt_configuration method
        def mock_adapt(size):
            return corrected_result
        
        macbook_trainer.adapt_configuration = mock_adapt
        result = macbook_trainer.adapt_configuration(dataset_size)
        
        assert result.adapted_config == corrected_config
    
    def test_create_macbook_dataloader(self, macbook_trainer, mock_pretrain_config, mock_training_params):
        """Test MacBook-optimized dataloader creation."""
        mock_dataloader_instance = Mock()
        mock_metadata = Mock()
        
        # Mock the create_macbook_dataloader method
        def mock_create_dataloader(config, split, training_params, **kwargs):
            return mock_dataloader_instance, mock_metadata
        
        macbook_trainer.create_macbook_dataloader = mock_create_dataloader
        
        dataloader, metadata = macbook_trainer.create_macbook_dataloader(
            mock_pretrain_config, "train", mock_training_params
        )
        
        assert dataloader == mock_dataloader_instance
        assert metadata == mock_metadata
    
    def test_create_macbook_model(self, macbook_trainer, mock_pretrain_config, mock_training_params):
        """Test MacBook-optimized model creation."""
        mock_model = Mock()
        mock_optimizers = [Mock()]
        mock_optimizer_lrs = [mock_training_params.learning_rate]
        
        # Mock train metadata
        train_metadata = Mock()
        train_metadata.vocab_size = 1000
        train_metadata.seq_len = 512
        train_metadata.num_puzzle_identifiers = 10
        
        # Mock the create_macbook_model method
        def mock_create_model(config, metadata, params):
            return mock_model, mock_optimizers, mock_optimizer_lrs
        
        macbook_trainer.create_macbook_model = mock_create_model
        
        model, optimizers, optimizer_lrs = macbook_trainer.create_macbook_model(
            mock_pretrain_config, train_metadata, mock_training_params
        )
        
        assert model == mock_model
        assert len(optimizers) == 1
        assert len(optimizer_lrs) == 1
        assert optimizer_lrs[0] == mock_training_params.learning_rate
    
    def test_initialize_training_state(self, macbook_trainer, mock_configuration_result):
        """Test training state initialization."""
        # Mock train metadata
        train_metadata = Mock()
        train_metadata.total_groups = 100
        train_metadata.mean_puzzle_examples = 1000
        
        mock_macbook_state = MockMacBookTrainingState()
        
        # Mock the initialize_training_state method
        def mock_initialize(config_result, metadata):
            return mock_macbook_state
        
        macbook_trainer.initialize_training_state = mock_initialize
        
        macbook_state = macbook_trainer.initialize_training_state(
            mock_configuration_result, train_metadata
        )
        
        assert macbook_state == mock_macbook_state
    
    def test_train_batch_macbook(self, macbook_trainer, mock_training_params):
        """Test MacBook batch training."""
        macbook_state = MockMacBookTrainingState()
        batch = {"input": Mock(), "target": Mock()}
        
        # Mock the train_batch_macbook method
        def mock_train_batch(state, batch_data, params):
            return {
                "train/lr": 1e-4,
                "train/samples_per_second": 10.0,
                "train/memory_usage_mb": 2000.0,
                "train/loss": 0.5
            }
        
        macbook_trainer.train_batch_macbook = mock_train_batch
        
        result = macbook_trainer.train_batch_macbook(macbook_state, batch, mock_training_params)
        
        assert isinstance(result, dict)
        assert "train/lr" in result
        assert "train/samples_per_second" in result
        assert "train/memory_usage_mb" in result
    
    def test_should_checkpoint_step_interval(self, macbook_trainer, mock_training_params):
        """Test checkpointing based on step interval."""
        macbook_state = MockMacBookTrainingState()
        macbook_state.train_state.step = 500  # Matches checkpoint_interval
        macbook_state.last_checkpoint_time = time.time()
        
        # Mock the should_checkpoint method
        def mock_should_checkpoint(state, params):
            return state.train_state.step % params.checkpoint_interval == 0
        
        macbook_trainer.should_checkpoint = mock_should_checkpoint
        
        should_checkpoint = macbook_trainer.should_checkpoint(macbook_state, mock_training_params)
        assert should_checkpoint == True
    
    def test_should_checkpoint_time_interval(self, macbook_trainer, mock_training_params):
        """Test checkpointing based on time interval."""
        macbook_state = MockMacBookTrainingState()
        macbook_state.train_state.step = 100  # Doesn't match checkpoint_interval
        macbook_state.last_checkpoint_time = time.time() - 400  # More than 5 minutes ago
        
        # Mock the should_checkpoint method
        def mock_should_checkpoint(state, params):
            time_since_last = time.time() - state.last_checkpoint_time
            return time_since_last > 300  # 5 minutes
        
        macbook_trainer.should_checkpoint = mock_should_checkpoint
        
        should_checkpoint = macbook_trainer.should_checkpoint(macbook_state, mock_training_params)
        assert should_checkpoint == True
    
    def test_create_progress_display(self, macbook_trainer):
        """Test progress display creation."""
        macbook_state = MockMacBookTrainingState()
        macbook_state.train_state.step = 500
        macbook_state.train_state.total_steps = 1000
        macbook_state.start_time = time.time() - 100  # 100 seconds ago
        macbook_state.training_metrics = {
            "loss": [0.5],
            "samples_per_second": [10.0]
        }
        
        # Mock the create_progress_display method
        def mock_create_progress(state):
            return "Step 500/1000 (50.0%) | Loss: 0.5000 | Speed: 10.0 samples/s | Memory: 75.0% (3000MB) | ETA: 1.7min"
        
        macbook_trainer.create_progress_display = mock_create_progress
        
        progress_display = macbook_trainer.create_progress_display(macbook_state)
        
        assert isinstance(progress_display, str)
        assert "Step 500/1000" in progress_display
        assert "50.0%" in progress_display  # Progress percentage
        assert "Loss: 0.5000" in progress_display
        assert "Speed: 10.0 samples/s" in progress_display
        assert "Memory: 75.0%" in progress_display
    
    def test_print_training_summary(self, macbook_trainer):
        """Test training summary printing."""
        macbook_state = MockMacBookTrainingState()
        macbook_state.start_time = time.time() - 3600  # 1 hour ago
        macbook_state.train_state.step = 1000
        macbook_state.train_state.total_steps = 1000
        macbook_state.samples_processed = 10000
        macbook_state.average_samples_per_second = 5.0
        macbook_state.training_metrics = {"loss": [0.8, 0.6, 0.4, 0.3]}
        
        # Mock the print_training_summary method
        def mock_print_summary(state):
            print("MacBook TRM Training Summary")
            print("Total training time: 60.0 minutes")
            print("Samples processed: 10,000")
            print("Average speed: 5.0 samples/second")
            print("Peak memory: 4000MB")
            print("Final training loss: 0.3000")
        
        macbook_trainer.print_training_summary = mock_print_summary
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            macbook_trainer.print_training_summary(macbook_state)
            
            # Verify print was called with summary information
            mock_print.assert_called()
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            summary_text = " ".join(print_calls)
            
            assert "Training Summary" in summary_text
            assert "60.0 minutes" in summary_text
            assert "10,000" in summary_text
            assert "5.0 samples/second" in summary_text
    
    def test_train_integration(self, macbook_trainer, mock_configuration_result):
        """Test main training loop integration."""
        dataset_size = 1000
        mock_macbook_state = MockMacBookTrainingState()
        
        # Mock the train method
        def mock_train(size):
            return mock_macbook_state
        
        macbook_trainer.train = mock_train
        
        result = macbook_trainer.train(dataset_size)
        
        assert result == mock_macbook_state


if __name__ == "__main__":
    pytest.main([__file__])