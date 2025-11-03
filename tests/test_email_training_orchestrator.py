"""
Tests for Email Training Orchestrator

This module tests the EmailTrainingOrchestrator and related components for
complete email classification training pipeline with multi-phase training
and hyperparameter optimization.
"""

import pytest
import json
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from typing import Dict, List, Any

from macbook_optimization.email_training_orchestrator import (
    EmailTrainingOrchestrator, TrainingPhase, HyperparameterSearchSpace,
    TrainingResult, HyperparameterOptimizer
)
from macbook_optimization.email_training_config import EmailTrainingConfig
from macbook_optimization.multi_phase_training import (
    MultiPhaseTrainingStrategy, PhaseTransitionManager, AdaptiveLearningRateScheduler
)
from macbook_optimization.email_hyperparameter_optimization import (
    EmailHyperparameterOptimizer, BayesianOptimizer, HyperparameterSpace
)


@pytest.fixture
def sample_email_dataset():
    """Create sample email dataset for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create train directory
    train_dir = os.path.join(temp_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Sample email data
    emails = [
        {
            "id": f"email_{i:03d}",
            "subject": f"Test Subject {i}",
            "body": f"This is test email body {i} with sufficient content for testing purposes.",
            "sender": f"sender{i}@example.com",
            "recipient": "user@example.com",
            "category": ["newsletter", "work", "personal", "promotional"][i % 4],
            "language": "en"
        }
        for i in range(20)  # 20 sample emails
    ]
    
    # Write to JSONL file
    train_file = os.path.join(train_dir, "emails.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for email in emails:
            f.write(json.dumps(email) + '\n')
    
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hardware_detector():
    """Create mock hardware detector."""
    detector = Mock()
    detector.get_hardware_specs.return_value = Mock(
        cpu=Mock(cores=8, base_frequency=2.4),
        memory=Mock(total_memory=16*1024**3, available_memory=12*1024**3),
        platform=Mock(system="Darwin", has_mkl=True, supports_avx2=True)
    )
    return detector


@pytest.fixture
def mock_memory_manager():
    """Create mock memory manager."""
    manager = Mock()
    manager.monitor_memory_usage.return_value = Mock(
        used_mb=4000,
        available_mb=8000,
        percent_used=50.0
    )
    return manager


@pytest.fixture
def mock_progress_monitor():
    """Create mock progress monitor."""
    monitor = Mock()
    monitor.start_session.return_value = Mock()
    monitor.end_session.return_value = Mock(
        total_training_time=300.0,
        total_samples_processed=1000,
        peak_memory_usage_mb=5000.0
    )
    return monitor


@pytest.fixture
def mock_checkpoint_manager():
    """Create mock checkpoint manager."""
    manager = Mock()
    manager.get_latest_checkpoint_id.return_value = "checkpoint_001"
    manager._get_checkpoint_path.return_value = Path("/tmp/checkpoint_001.pt")
    return manager


@pytest.fixture
def sample_training_config():
    """Create sample training configuration."""
    return EmailTrainingConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        batch_size=4,
        learning_rate=1e-4,
        max_steps=100,
        target_accuracy=0.95
    )


class TestTrainingPhase:
    """Test TrainingPhase dataclass."""
    
    def test_training_phase_creation(self):
        """Test creating TrainingPhase."""
        phase = TrainingPhase(
            name="test_phase",
            description="Test phase description",
            steps=1000,
            learning_rate=1e-4,
            batch_size=8
        )
        
        assert phase.name == "test_phase"
        assert phase.description == "Test phase description"
        assert phase.steps == 1000
        assert phase.learning_rate == 1e-4
        assert phase.batch_size == 8
        
        # Check defaults
        assert phase.warmup_steps == 100
        assert phase.use_hierarchical_attention is True
        assert phase.data_filter == "all"
    
    def test_training_phase_with_overrides(self):
        """Test TrainingPhase with model config overrides."""
        overrides = {"hidden_size": 128, "num_layers": 1}
        
        phase = TrainingPhase(
            name="simple_phase",
            description="Simplified model phase",
            steps=500,
            learning_rate=2e-4,
            batch_size=4,
            model_config_overrides=overrides
        )
        
        assert phase.model_config_overrides == overrides
        assert phase.model_config_overrides["hidden_size"] == 128


class TestHyperparameterSearchSpace:
    """Test HyperparameterSearchSpace dataclass."""
    
    def test_default_search_space(self):
        """Test creating default search space."""
        space = HyperparameterSearchSpace()
        
        # Check defaults are set
        assert space.hidden_size == [256, 384, 512]
        assert space.num_layers == [2, 3]
        assert len(space.learning_rate) == 3
        assert len(space.batch_size) == 3
    
    def test_custom_search_space(self):
        """Test creating custom search space."""
        space = HyperparameterSearchSpace(
            hidden_size=[128, 256],
            learning_rate=[1e-5, 1e-4],
            batch_size=[2, 4]
        )
        
        assert space.hidden_size == [128, 256]
        assert space.learning_rate == [1e-5, 1e-4]
        assert space.batch_size == [2, 4]
        
        # Defaults should still be set for unspecified parameters
        assert space.num_layers == [2, 3]


class TestEmailTrainingOrchestrator:
    """Test EmailTrainingOrchestrator class."""
    
    def test_orchestrator_initialization(self):
        """Test EmailTrainingOrchestrator initialization."""
        orchestrator = EmailTrainingOrchestrator(
            output_dir="test_output",
            enable_monitoring=True,
            enable_checkpointing=True
        )
        
        assert orchestrator.output_dir.name == "test_output"
        assert orchestrator.hardware_detector is not None
        assert orchestrator.memory_manager is not None
        assert orchestrator.progress_monitor is not None
        assert orchestrator.checkpoint_manager is not None
        assert len(orchestrator.training_history) == 0
    
    def test_orchestrator_without_monitoring(self):
        """Test orchestrator without monitoring and checkpointing."""
        orchestrator = EmailTrainingOrchestrator(
            enable_monitoring=False,
            enable_checkpointing=False
        )
        
        assert orchestrator.progress_monitor is None
        assert orchestrator.checkpoint_manager is None
    
    @patch('macbook_optimization.email_training_orchestrator.HardwareDetector')
    @patch('macbook_optimization.email_training_orchestrator.EmailDatasetManager')
    def test_setup_training_environment_success(self, mock_dataset_manager, mock_hardware_detector, sample_email_dataset):
        """Test successful training environment setup."""
        # Mock hardware detector - not used directly since we use config_adapter
        
        # Mock dataset manager
        mock_manager_instance = Mock()
        mock_manager_instance.validate_email_dataset.return_value = {
            "valid": True,
            "total_emails": 20,
            "total_size_mb": 0.1,
            "category_distribution": {"work": 5, "personal": 5}
        }
        mock_manager_instance.create_email_dataloader.return_value = (Mock(), {"optimized_batch_size": 4})
        mock_dataset_manager.return_value = mock_manager_instance
        
        orchestrator = EmailTrainingOrchestrator()
        
        # Mock config adapter
        with patch.object(orchestrator.config_adapter, 'get_hardware_specs') as mock_hardware_specs:
            mock_hardware_specs.return_value = Mock(
                cpu=Mock(cores=8),
                memory=Mock(total_memory=16*1024**3, available_memory=12*1024**3),
                platform=Mock(macos_version="macOS 12.0")
            )
            
            with patch.object(orchestrator.config_adapter, 'create_email_hardware_config') as mock_config:
                mock_config.return_value = Mock(
                    adapted_config={"vocab_size": 1000, "batch_size": 4, "max_sequence_length": 256},
                    validation_warnings=["Test warning"],
                    performance_estimates={}
                )
                
                result = orchestrator.setup_training_environment(sample_email_dataset)
        
        assert result["success"] is True
        assert "hardware_specs" in result
        assert "dataset_info" in result
        assert "config_adaptation" in result
        assert len(result["warnings"]) >= 1
    
    def test_setup_training_environment_dataset_failure(self, sample_email_dataset):
        """Test training environment setup with dataset validation failure."""
        orchestrator = EmailTrainingOrchestrator()
        
        # Mock dataset manager to return invalid dataset
        with patch.object(orchestrator.dataset_manager, 'validate_email_dataset') as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Dataset not found"
            }
            
            result = orchestrator.setup_training_environment(sample_email_dataset)
        
        assert result["success"] is False
        assert len(result["errors"]) > 0
        assert "Dataset validation failed" in result["errors"][0]
    
    def test_create_single_phase_strategy(self, sample_training_config):
        """Test creating single-phase training strategy."""
        orchestrator = EmailTrainingOrchestrator()
        
        phases = orchestrator.create_training_phases(
            strategy="single",
            total_steps=1000,
            base_config=sample_training_config
        )
        
        assert len(phases) == 1
        assert phases[0].name == "main_training"
        assert phases[0].steps == 1000
        assert phases[0].learning_rate == sample_training_config.learning_rate
        assert phases[0].batch_size == sample_training_config.batch_size
    
    def test_create_multi_phase_strategy(self, sample_training_config):
        """Test creating multi-phase training strategy."""
        orchestrator = EmailTrainingOrchestrator()
        
        phases = orchestrator.create_training_phases(
            strategy="multi_phase",
            total_steps=1200,
            base_config=sample_training_config
        )
        
        assert len(phases) == 3
        assert phases[0].name == "warmup"
        assert phases[1].name == "main_training"
        assert phases[2].name == "fine_tuning"
        
        # Check step distribution
        total_steps = sum(phase.steps for phase in phases)
        assert total_steps == 1200
        
        # Check learning rate progression
        assert phases[0].learning_rate < phases[1].learning_rate
        assert phases[2].learning_rate < phases[1].learning_rate
    
    def test_create_progressive_strategy(self, sample_training_config):
        """Test creating progressive training strategy."""
        orchestrator = EmailTrainingOrchestrator()
        
        phases = orchestrator.create_training_phases(
            strategy="progressive",
            total_steps=1500,
            base_config=sample_training_config
        )
        
        assert len(phases) == 3
        assert phases[0].name == "simple_model"
        assert phases[1].name == "medium_model"
        assert phases[2].name == "full_model"
        
        # Check model complexity progression
        assert phases[0].model_config_overrides is not None
        assert phases[1].model_config_overrides is not None
        assert phases[2].model_config_overrides is None  # Full model
    
    def test_create_curriculum_strategy(self, sample_training_config):
        """Test creating curriculum learning strategy."""
        orchestrator = EmailTrainingOrchestrator()
        
        phases = orchestrator.create_training_phases(
            strategy="curriculum",
            total_steps=1000,
            base_config=sample_training_config
        )
        
        assert len(phases) == 3
        assert phases[0].name == "easy_emails"
        assert phases[1].name == "medium_emails"
        assert phases[2].name == "all_emails"
        
        # Check data filter progression
        assert phases[0].data_filter == "easy"
        assert phases[1].data_filter == "medium"
        assert phases[2].data_filter == "all"
    
    def test_invalid_strategy(self, sample_training_config):
        """Test creating phases with invalid strategy."""
        orchestrator = EmailTrainingOrchestrator()
        
        with pytest.raises(ValueError, match="Unknown training strategy"):
            orchestrator.create_training_phases(
                strategy="invalid_strategy",
                total_steps=1000,
                base_config=sample_training_config
            )
    
    @patch('macbook_optimization.email_training_orchestrator.MacBookEmailTRM')
    @patch('macbook_optimization.email_training_orchestrator.EmailTokenizer')
    @patch('macbook_optimization.email_training_orchestrator.EmailTrainingLoop')
    def test_execute_training_pipeline_success(self, mock_training_loop, mock_tokenizer, mock_model, sample_email_dataset):
        """Test successful training pipeline execution."""
        orchestrator = EmailTrainingOrchestrator()
        
        # Mock setup_training_environment
        with patch.object(orchestrator, 'setup_training_environment') as mock_setup:
            mock_setup.return_value = {
                "success": True,
                "warnings": [],
                "config_adaptation": {
                    "adapted_config": {
                        "vocab_size": 1000,
                        "hidden_size": 256,
                        "num_layers": 2,
                        "batch_size": 4,
                        "learning_rate": 1e-4,
                        "max_sequence_length": 256,
                        "num_email_categories": 10,
                        "use_hierarchical_attention": True,
                        "subject_attention_weight": 2.0
                    }
                }
            }
            
            # Mock dataset manager
            with patch.object(orchestrator.dataset_manager, 'load_email_dataset') as mock_load:
                mock_load.return_value = Mock()
                
                with patch.object(orchestrator.dataset_manager, 'create_email_dataloader') as mock_dataloader:
                    mock_dataloader.return_value = (Mock(), {"optimized_batch_size": 4})
                    
                    # Mock training loop
                    mock_loop_instance = Mock()
                    mock_loop_instance.train.return_value = {
                        "final_val_metrics": Mock(accuracy=0.92, loss=0.3),
                        "target_reached": False
                    }
                    mock_loop_instance.evaluate.return_value = Mock(
                        accuracy=0.92,
                        loss=0.3,
                        category_accuracies={"work": 0.9, "personal": 0.94}
                    )
                    mock_training_loop.return_value = mock_loop_instance
                    
                    # Mock torch.save to avoid pickle issues
                    with patch('macbook_optimization.email_training_orchestrator.torch') as mock_torch:
                        mock_torch.save = Mock()
                        
                        # Execute training
                        result = orchestrator.execute_training_pipeline(
                            dataset_path=sample_email_dataset,
                            strategy="single",
                            total_steps=100
                        )
        
        assert result.success is True
        assert result.final_accuracy == 0.92
        assert result.final_loss == 0.3
        assert len(result.phases_completed) > 0
        assert result.model_path is not None
    
    def test_execute_training_pipeline_setup_failure(self, sample_email_dataset):
        """Test training pipeline with setup failure."""
        orchestrator = EmailTrainingOrchestrator()
        
        # Mock failed setup
        with patch.object(orchestrator, 'setup_training_environment') as mock_setup:
            mock_setup.return_value = {
                "success": False,
                "errors": ["Setup failed"],
                "warnings": []
            }
            
            result = orchestrator.execute_training_pipeline(
                dataset_path=sample_email_dataset,
                total_steps=100
            )
        
        assert result.success is False
        assert len(result.errors) > 0
        assert "Setup failed" in result.errors
    
    def test_get_training_summary_empty(self):
        """Test getting training summary with no history."""
        orchestrator = EmailTrainingOrchestrator()
        
        summary = orchestrator.get_training_summary()
        
        assert "message" in summary
        assert summary["message"] == "No training runs completed"
    
    def test_get_training_summary_with_history(self):
        """Test getting training summary with training history."""
        orchestrator = EmailTrainingOrchestrator()
        
        # Add mock training results
        result1 = TrainingResult(
            success=True,
            training_id="test_001",
            start_time=datetime.now(),
            end_time=None,
            config=EmailTrainingConfig(),
            phases_completed=["main_training"],
            final_accuracy=0.92,
            best_accuracy=0.94,
            final_loss=None,
            best_loss=None,
            category_accuracies={},
            total_training_time=300.0,
            total_steps=1000,
            samples_processed=5000,
            peak_memory_usage_mb=4000.0,
            average_cpu_usage=60.0,
            model_path=None,
            checkpoint_path=None,
            errors=[],
            warnings=[]
        )
        
        result2 = TrainingResult(
            success=False,
            training_id="test_002",
            start_time=datetime.now(),
            end_time=None,
            config=EmailTrainingConfig(),
            phases_completed=[],
            final_accuracy=None,
            best_accuracy=None,
            final_loss=None,
            best_loss=None,
            category_accuracies={},
            total_training_time=100.0,
            total_steps=0,
            samples_processed=0,
            peak_memory_usage_mb=2000.0,
            average_cpu_usage=30.0,
            model_path=None,
            checkpoint_path=None,
            errors=["Training failed"],
            warnings=[]
        )
        
        orchestrator.training_history = [result1, result2]
        
        summary = orchestrator.get_training_summary()
        
        assert summary["total_runs"] == 2
        assert summary["successful_runs"] == 1
        assert summary["failed_runs"] == 1
        assert summary["best_accuracy"] == 0.94
        assert len(summary["recent_runs"]) == 2


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    def test_hyperparameter_optimizer_initialization(self):
        """Test HyperparameterOptimizer initialization."""
        orchestrator = Mock()
        search_space = HyperparameterSearchSpace()
        
        optimizer = HyperparameterOptimizer(
            orchestrator=orchestrator,
            search_space=search_space,
            optimization_strategy="bayesian"
        )
        
        assert optimizer.orchestrator == orchestrator
        assert optimizer.search_space == search_space
        assert optimizer.optimization_strategy == "bayesian"
        assert len(optimizer.optimization_history) == 0
        assert optimizer.best_config is None
    
    def test_optimize_hyperparameters_success(self):
        """Test successful hyperparameter optimization."""
        # Mock orchestrator
        orchestrator = Mock()
        
        # Mock the execute_training_pipeline method
        def mock_execute_training_pipeline(**kwargs):
            config = kwargs.get('config', EmailTrainingConfig())
            return TrainingResult(
                success=True,
                training_id="mock_trial",
                start_time=datetime.now(),
                end_time=datetime.now(),
                config=config,
                phases_completed=["main_training"],
                final_accuracy=0.85 + (config.learning_rate * 1000),  # Simulate LR effect
                best_accuracy=0.87 + (config.learning_rate * 1000),
                final_loss=0.5,
                best_loss=0.4,
                category_accuracies={},
                total_training_time=120.0,
                total_steps=1000,
                samples_processed=5000,
                peak_memory_usage_mb=3000.0,
                average_cpu_usage=60.0,
                model_path=None,
                checkpoint_path=None,
                errors=[],
                warnings=[]
            )
        
        orchestrator.execute_training_pipeline = mock_execute_training_pipeline
        
        optimizer = HyperparameterOptimizer(
            orchestrator=orchestrator,
            optimization_strategy="random"
        )
        
        # Create a temporary dataset path for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        result = optimizer.optimize_hyperparameters(
            dataset_path=temp_dir,
            num_trials=3,
            target_metric="accuracy"
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        assert result["success"] is True
        assert result["completed_trials"] == 3
        assert result["best_performance"] > 0
        assert len(result["optimization_history"]) == 3
        assert result["total_time"] > 0
    
    def test_optimize_hyperparameters_with_failures(self):
        """Test hyperparameter optimization with some failed trials."""
        orchestrator = Mock()
        
        # Mock training function that sometimes fails
        def mock_execute_training_pipeline(**kwargs):
            config = kwargs.get('config', EmailTrainingConfig())
            if config.batch_size == 2:  # Simulate failure for small batch size
                raise ValueError("Batch size too small")
            
            return TrainingResult(
                success=True,
                training_id="mock_trial",
                start_time=datetime.now(),
                end_time=datetime.now(),
                config=config,
                phases_completed=["main_training"],
                final_accuracy=0.8,
                best_accuracy=0.82,
                final_loss=0.4,
                best_loss=0.35,
                category_accuracies={},
                total_training_time=120.0,
                total_steps=1000,
                samples_processed=5000,
                peak_memory_usage_mb=3000.0,
                average_cpu_usage=60.0,
                model_path=None,
                checkpoint_path=None,
                errors=[],
                warnings=[]
            )
        
        orchestrator.execute_training_pipeline = mock_execute_training_pipeline
        
        optimizer = HyperparameterOptimizer(
            orchestrator=orchestrator,
            search_space=HyperparameterSearchSpace(batch_size=[2, 4, 8]),
            optimization_strategy="random"
        )
        
        # Create a temporary dataset path for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        result = optimizer.optimize_hyperparameters(
            dataset_path=temp_dir,
            num_trials=5,
            target_metric="accuracy"
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        # Should have some successful and some failed trials
        assert result["completed_trials"] < 5  # Some trials failed
        assert len(result["errors"]) > 0  # Should have error messages
        
        # But should still have some successful trials
        successful_trials = [t for t in result["optimization_history"] if t["success"]]
        assert len(successful_trials) > 0
    
    def test_get_optimization_summary_empty(self):
        """Test optimization summary with no history."""
        orchestrator = Mock()
        optimizer = HyperparameterOptimizer(orchestrator=orchestrator)
        
        summary = optimizer.get_optimization_summary()
        
        assert "message" in summary
        assert summary["message"] == "No optimization trials completed"
    
    def test_get_optimization_summary_with_history(self):
        """Test optimization summary with optimization history."""
        orchestrator = Mock()
        optimizer = HyperparameterOptimizer(orchestrator=orchestrator)
        
        # Add mock optimization history
        optimizer.optimization_history = [
            {
                "trial_id": 1,
                "parameters": {"learning_rate": 1e-4},
                "performance": 0.85,
                "success": True,
                "training_time": 100.0
            },
            {
                "trial_id": 2,
                "parameters": {"learning_rate": 2e-4},
                "performance": 0.87,
                "success": True,
                "training_time": 120.0
            }
        ]
        optimizer.best_performance = 0.87
        optimizer.best_config = EmailTrainingConfig(learning_rate=2e-4)
        
        summary = optimizer.get_optimization_summary()
        
        assert summary["total_trials"] == 2
        assert summary["successful_trials"] == 2
        assert summary["best_performance"] == 0.87
        assert summary["best_config"] is not None
        assert summary["optimization_strategy"] == "bayesian"


class TestMultiPhaseTrainingIntegration:
    """Integration tests for multi-phase training components."""
    
    def test_multi_phase_training_strategy_creation(self):
        """Test creating multi-phase training strategy."""
        from macbook_optimization.multi_phase_training import MultiPhaseTrainingStrategy
        
        phases = [
            {"name": "phase1", "steps": 500, "learning_rate": 1e-4, "batch_size": 4},
            {"name": "phase2", "steps": 1000, "learning_rate": 5e-5, "batch_size": 8}
        ]
        
        strategy = MultiPhaseTrainingStrategy(phases)
        
        assert len(strategy.phases) == 2
        assert strategy.current_phase_index == 0
        
        # Start strategy
        info = strategy.start_strategy()
        assert info["total_phases"] == 2
        assert info["total_target_steps"] == 1500
        
        # Get current phase
        phase_config = strategy.get_current_phase_config()
        assert phase_config["name"] == "phase1"
        assert phase_config["steps"] == 500
    
    def test_adaptive_learning_rate_scheduler(self):
        """Test adaptive learning rate scheduler."""
        from macbook_optimization.multi_phase_training import AdaptiveLearningRateScheduler
        
        # Skip this test if torch is not available
        try:
            import torch
            import torch.optim as optim
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Create a real optimizer for testing
        model = torch.nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        scheduler = AdaptiveLearningRateScheduler(
            optimizer=optimizer,
            phase_config={},
            total_steps=1000,
            warmup_steps=100,
            scheduler_type="cosine_with_warmup"
        )
        
        # Test warmup phase
        scheduler.last_epoch = 50  # Halfway through warmup
        lrs = scheduler.get_lr()
        assert len(lrs) == 1
        assert 0 < lrs[0] < 1e-4  # Should be between 0 and base LR
        
        # Test main phase
        scheduler.last_epoch = 500  # Halfway through main training
        lrs = scheduler.get_lr()
        assert len(lrs) == 1
        assert lrs[0] > 0  # Should be positive
        
        # Test performance factor update
        scheduler.update_performance_factor(0.8)
        assert len(scheduler.performance_history) == 1


class TestBayesianOptimizer:
    """Test BayesianOptimizer class."""
    
    def test_bayesian_optimizer_initialization(self):
        """Test BayesianOptimizer initialization."""
        from macbook_optimization.email_hyperparameter_optimization import HyperparameterSpace, BayesianOptimizer
        
        search_space = HyperparameterSpace()
        optimizer = BayesianOptimizer(search_space)
        
        assert optimizer.search_space == search_space
        assert optimizer.acquisition_function == "expected_improvement"
        assert optimizer.n_initial_points == 5
        assert len(optimizer.trials) == 0
    
    def test_suggest_parameters_initial(self):
        """Test parameter suggestion in initial phase."""
        from macbook_optimization.email_hyperparameter_optimization import HyperparameterSpace, BayesianOptimizer
        
        search_space = HyperparameterSpace()
        optimizer = BayesianOptimizer(search_space, n_initial_points=3)
        
        # Should use random sampling for initial points
        params1 = optimizer.suggest_parameters()
        params2 = optimizer.suggest_parameters()
        
        # Should return valid parameters
        assert "hidden_size" in params1
        assert "learning_rate" in params1
        assert "batch_size" in params1
        
        # Parameters should be different (with high probability)
        assert params1 != params2
    
    def test_suggest_parameters_bayesian(self):
        """Test parameter suggestion in Bayesian phase."""
        from macbook_optimization.email_hyperparameter_optimization import (
            HyperparameterSpace, BayesianOptimizer, OptimizationTrial
        )
        
        search_space = HyperparameterSpace()
        optimizer = BayesianOptimizer(search_space, n_initial_points=2)
        
        # Add some completed trials
        trial1 = OptimizationTrial(
            trial_id=1,
            parameters={"hidden_size": 256, "learning_rate": 1e-4, "batch_size": 4},
            objective_value=0.85,
            status="completed"
        )
        trial2 = OptimizationTrial(
            trial_id=2,
            parameters={"hidden_size": 512, "learning_rate": 2e-4, "batch_size": 8},
            objective_value=0.90,
            status="completed"
        )
        
        optimizer.trials = [trial1, trial2]
        
        # Should now use Bayesian optimization
        params = optimizer.suggest_parameters()
        
        assert "hidden_size" in params
        assert "learning_rate" in params
        assert "batch_size" in params
    
    def test_get_best_parameters(self):
        """Test getting best parameters."""
        from macbook_optimization.email_hyperparameter_optimization import (
            HyperparameterSpace, BayesianOptimizer, OptimizationTrial
        )
        
        search_space = HyperparameterSpace()
        optimizer = BayesianOptimizer(search_space)
        
        # No trials yet
        assert optimizer.get_best_parameters() is None
        
        # Add trials
        trial1 = OptimizationTrial(
            trial_id=1,
            parameters={"hidden_size": 256, "learning_rate": 1e-4},
            objective_value=0.85,
            status="completed"
        )
        trial2 = OptimizationTrial(
            trial_id=2,
            parameters={"hidden_size": 512, "learning_rate": 2e-4},
            objective_value=0.90,
            status="completed"
        )
        
        optimizer.trials = [trial1, trial2]
        
        best_params = optimizer.get_best_parameters()
        assert best_params == trial2.parameters  # Trial2 has higher objective


@pytest.mark.integration
class TestEmailTrainingOrchestratorIntegration:
    """Integration tests for complete training orchestrator workflow."""
    
    @patch('macbook_optimization.email_training_orchestrator.torch')
    def test_end_to_end_training_workflow(self, mock_torch, sample_email_dataset):
        """Test complete end-to-end training workflow."""
        # Mock torch operations
        mock_torch.save = Mock()
        
        orchestrator = EmailTrainingOrchestrator(
            output_dir="test_integration_output",
            enable_monitoring=False,  # Disable for simpler testing
            enable_checkpointing=False
        )
        
        # Mock all the heavy components
        with patch.object(orchestrator, 'setup_training_environment') as mock_setup:
            mock_setup.return_value = {
                "success": True,
                "warnings": [],
                "config_adaptation": {
                    "adapted_config": {
                        "vocab_size": 1000,
                        "hidden_size": 256,
                        "num_layers": 2,
                        "batch_size": 4,
                        "learning_rate": 1e-4,
                        "max_sequence_length": 256,
                        "num_email_categories": 10,
                        "use_hierarchical_attention": True,
                        "subject_attention_weight": 2.0,
                        "max_epochs": 1,
                        "target_accuracy": 0.95
                    }
                }
            }
            
            with patch('macbook_optimization.email_training_orchestrator.EmailTokenizer') as mock_tokenizer:
                with patch('macbook_optimization.email_training_orchestrator.MacBookEmailTRM') as mock_model:
                    with patch('macbook_optimization.email_training_orchestrator.EmailTrainingLoop') as mock_loop:
                        with patch.object(orchestrator.dataset_manager, 'load_email_dataset') as mock_load:
                            with patch.object(orchestrator.dataset_manager, 'create_email_dataloader') as mock_dataloader:
                                
                                # Setup mocks
                                mock_load.return_value = Mock()
                                mock_dataloader.return_value = (Mock(), {"optimized_batch_size": 4})
                                
                                mock_loop_instance = Mock()
                                mock_loop_instance.train.return_value = {
                                    "final_val_metrics": Mock(accuracy=0.96, loss=0.2),
                                    "target_reached": True
                                }
                                mock_loop_instance.evaluate.return_value = Mock(
                                    accuracy=0.96,
                                    loss=0.2,
                                    category_accuracies={"work": 0.95, "personal": 0.97}
                                )
                                mock_loop.return_value = mock_loop_instance
                                
                                # Execute training
                                result = orchestrator.execute_training_pipeline(
                                    dataset_path=sample_email_dataset,
                                    strategy="single",
                                    total_steps=50  # Small for testing
                                )
        
        # Verify results
        assert result.success is True
        assert result.final_accuracy == 0.96
        assert result.best_accuracy == 0.96
        assert len(result.phases_completed) == 1
        assert result.phases_completed[0] == "main_training"
        
        # Verify training history was updated
        assert len(orchestrator.training_history) == 1
        assert orchestrator.training_history[0] == result
    
    def test_hyperparameter_optimization_integration(self):
        """Test hyperparameter optimization integration."""
        orchestrator = Mock()
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            orchestrator=orchestrator,
            optimization_strategy="random"
        )
        
        # Create a temporary dataset path for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Mock the orchestrator's execute_training_pipeline method
        call_count = 0
        def mock_execute_training_pipeline(**kwargs):
            nonlocal call_count
            call_count += 1
            config = kwargs.get('config', EmailTrainingConfig())
            performance = 0.8 + (config.learning_rate * 1000) % 0.15
            return TrainingResult(
                success=True,
                training_id="mock_trial",
                start_time=datetime.now(),
                end_time=datetime.now(),
                config=config,
                phases_completed=["main_training"],
                final_accuracy=performance,
                best_accuracy=performance + 0.02,
                final_loss=1.0 - performance,
                best_loss=1.0 - performance - 0.01,
                category_accuracies={"work": performance, "personal": performance + 0.02},
                total_training_time=120.0,
                total_steps=1000,
                samples_processed=5000,
                peak_memory_usage_mb=3000.0,
                average_cpu_usage=60.0,
                model_path=None,
                checkpoint_path=None,
                errors=[],
                warnings=[]
            )
        
        orchestrator.execute_training_pipeline = mock_execute_training_pipeline
        
        # Run optimization
        result = optimizer.optimize_hyperparameters(
            dataset_path=temp_dir,
            num_trials=5,
            target_metric="accuracy"
        )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        # Verify optimization ran
        assert call_count == 5
        assert result["success"] is True
        assert result["completed_trials"] == 5
        assert result["best_performance"] > 0.8
        assert len(result["optimization_history"]) == 5
        
        # Verify best configuration was found
        assert result["best_config"] is not None
        assert result["best_performance"] > 0
    
    def test_multi_phase_strategy_integration(self):
        """Test multi-phase training strategy integration."""
        orchestrator = EmailTrainingOrchestrator()
        
        # Create phases for different strategies
        strategies = ["single", "multi_phase", "progressive", "curriculum"]
        
        for strategy in strategies:
            phases = orchestrator.create_training_phases(
                strategy=strategy,
                total_steps=1000,
                base_config=EmailTrainingConfig()
            )
            
            # Verify phases were created
            assert len(phases) > 0
            
            # Verify total steps (allow for rounding in integer division)
            total_steps = sum(phase.steps for phase in phases)
            assert abs(total_steps - 1000) <= len(phases)  # Allow for rounding errors
            
            # Verify phase names are unique
            phase_names = [phase.name for phase in phases]
            assert len(phase_names) == len(set(phase_names))
            
            # Verify all phases have required attributes
            for phase in phases:
                assert hasattr(phase, 'name')
                assert hasattr(phase, 'description')
                assert hasattr(phase, 'steps')
                assert hasattr(phase, 'learning_rate')
                assert hasattr(phase, 'batch_size')
                assert phase.steps > 0
                assert phase.learning_rate > 0
                assert phase.batch_size > 0


if __name__ == "__main__":
    pytest.main([__file__])