"""
Unit tests for gradient accumulation module.

Tests gradient accumulation correctness, memory-aware adjustments,
and training loop integration for MacBook optimization.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch, MagicMock

from macbook_optimization.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulator,
    AccumulationState,
    TrainingLoopIntegration,
)
from macbook_optimization.memory_management import MemoryManager


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestGradientAccumulationConfig:
    """Test GradientAccumulationConfig dataclass."""
    
    def test_gradient_accumulation_config_defaults(self):
        """Test GradientAccumulationConfig default values."""
        config = GradientAccumulationConfig()
        
        assert config.target_batch_size == 32
        assert config.max_micro_batch_size == 8
        assert config.min_micro_batch_size == 1
        assert config.gradient_clipping == 1.0
        assert config.scale_gradients is True
        assert config.sync_batch_norm is False
        assert config.accumulate_grad_batches is None
    
    def test_gradient_accumulation_config_custom_values(self):
        """Test GradientAccumulationConfig with custom values."""
        config = GradientAccumulationConfig(
            target_batch_size=64,
            max_micro_batch_size=16,
            gradient_clipping=0.5,
            scale_gradients=False
        )
        
        assert config.target_batch_size == 64
        assert config.max_micro_batch_size == 16
        assert config.gradient_clipping == 0.5
        assert config.scale_gradients is False


class TestAccumulationState:
    """Test AccumulationState dataclass."""
    
    def test_accumulation_state_defaults(self):
        """Test AccumulationState default values."""
        state = AccumulationState()
        
        assert state.current_step == 0
        assert state.accumulated_steps == 0
        assert state.effective_batch_size == 0
        assert state.micro_batch_size == 0
        assert state.accumulation_steps == 0
        assert state.total_samples_processed == 0
        assert state.gradient_scale == 1.0


class TestGradientAccumulator:
    """Test GradientAccumulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = GradientAccumulationConfig(
            target_batch_size=16,
            max_micro_batch_size=4,
            gradient_clipping=1.0
        )
        self.accumulator = GradientAccumulator(self.config)
        
        # Create test model and optimizer
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.accumulator.setup_optimizer(self.optimizer)
    
    def test_gradient_accumulator_initialization(self):
        """Test GradientAccumulator initialization."""
        assert self.accumulator.config == self.config
        assert self.accumulator.state.micro_batch_size == 4
        assert self.accumulator.state.accumulation_steps == 4  # 16 / 4
        assert self.accumulator.state.effective_batch_size == 16
        assert self.accumulator.state.gradient_scale == 0.25  # 1 / 4
    
    def test_gradient_accumulator_with_memory_manager(self):
        """Test GradientAccumulator with memory manager."""
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_memory_manager.current_batch_size = 2
        
        accumulator = GradientAccumulator(self.config, mock_memory_manager)
        
        # Should use memory manager's batch size as micro batch size
        assert accumulator.state.micro_batch_size == 2
        assert accumulator.state.accumulation_steps == 8  # 16 / 2
        assert accumulator.state.effective_batch_size == 16
        assert accumulator.state.gradient_scale == 0.125  # 1 / 8
    
    def test_should_accumulate(self):
        """Test should_accumulate logic."""
        # Initially should accumulate (step 0)
        assert self.accumulator.should_accumulate() is True
        
        # After 3 accumulated steps, should still accumulate
        self.accumulator.state.accumulated_steps = 3
        assert self.accumulator.should_accumulate() is True
        
        # After 3 accumulated steps (next would be 4th), should not accumulate
        assert self.accumulator.should_step() is True
    
    def test_should_step(self):
        """Test should_step logic."""
        # Initially should not step
        assert self.accumulator.should_step() is False
        
        # After 3 accumulated steps, should step
        self.accumulator.state.accumulated_steps = 3
        assert self.accumulator.should_step() is True
        
        # After 4 accumulated steps, should not step (reset cycle)
        self.accumulator.state.accumulated_steps = 4
        assert self.accumulator.should_step() is False
    
    def test_scale_loss(self):
        """Test loss scaling for gradient accumulation."""
        loss = torch.tensor(2.0)
        
        # With gradient scaling enabled
        scaled_loss = self.accumulator.scale_loss(loss)
        expected_scaled = loss * self.accumulator.state.gradient_scale
        assert torch.allclose(scaled_loss, expected_scaled)
        
        # With gradient scaling disabled
        self.accumulator.config.scale_gradients = False
        scaled_loss = self.accumulator.scale_loss(loss)
        assert torch.allclose(scaled_loss, loss)
    
    def test_accumulate_gradients(self):
        """Test gradient accumulation."""
        # Create dummy input and target
        x = torch.randn(4, 10)  # micro batch size 4
        target = torch.randn(4, 1)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        
        # Accumulate gradients
        info = self.accumulator.accumulate_gradients(self.model, loss)
        
        assert "scaled_loss" in info
        assert "original_loss" in info
        assert "accumulated_steps" in info
        assert "should_step" in info
        assert "effective_batch_size" in info
        assert "gradient_scale" in info
        
        assert info["accumulated_steps"] == 1
        assert info["should_step"] is False  # Should not step after 1 accumulation
        assert info["effective_batch_size"] == 16
        
        # Check that gradients exist
        for param in self.model.parameters():
            assert param.grad is not None
    
    def test_step_optimizer(self):
        """Test optimizer stepping."""
        # Accumulate gradients first
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)
        
        for i in range(4):  # Accumulate 4 times to trigger step
            output = self.model(x)
            loss = nn.MSELoss()(output, target)
            self.accumulator.accumulate_gradients(self.model, loss)
        
        # Now step should be triggered
        assert self.accumulator.should_step() is True
        
        step_info = self.accumulator.step_optimizer()
        
        assert step_info["stepped"] is True
        assert "grad_norm" in step_info
        assert step_info["optimizer_stepped"] is True
        assert self.accumulator.state.current_step == 1
        
        # Gradients should be zeroed after step
        for param in self.model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_step_optimizer_without_accumulation_complete(self):
        """Test optimizer stepping when accumulation is not complete."""
        # Only accumulate once (not enough for stepping)
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        self.accumulator.accumulate_gradients(self.model, loss)
        
        step_info = self.accumulator.step_optimizer()
        
        assert step_info["stepped"] is False
        assert step_info["reason"] == "accumulation_incomplete"
        assert self.accumulator.state.current_step == 0
    
    def test_gradient_clipping(self):
        """Test gradient clipping during optimizer step."""
        # Create model with large gradients
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)
        
        # Accumulate gradients 4 times
        for i in range(4):
            output = self.model(x)
            loss = nn.MSELoss()(output, target) * 100  # Large loss to create large gradients
            self.accumulator.accumulate_gradients(self.model, loss)
        
        step_info = self.accumulator.step_optimizer()
        
        assert step_info["stepped"] is True
        assert "grad_norm" in step_info
        # Gradient norm should be clipped to max 1.0
        assert step_info["grad_norm"] <= self.config.gradient_clipping + 1e-6  # Small tolerance
    
    def test_adjust_for_memory_pressure_high(self):
        """Test adjustment for high memory pressure."""
        original_micro_batch_size = self.accumulator.state.micro_batch_size
        original_accumulation_steps = self.accumulator.state.accumulation_steps
        
        self.accumulator.adjust_for_memory_pressure("high")
        
        # Should reduce micro batch size and increase accumulation steps
        assert self.accumulator.state.micro_batch_size < original_micro_batch_size
        assert self.accumulator.state.accumulation_steps > original_accumulation_steps
        # Effective batch size should remain approximately the same
        assert self.accumulator.state.effective_batch_size <= self.config.target_batch_size
    
    def test_adjust_for_memory_pressure_low(self):
        """Test adjustment for low memory pressure."""
        # Start with small micro batch size
        self.accumulator.state.micro_batch_size = 2
        self.accumulator.state.accumulation_steps = 8
        
        original_micro_batch_size = self.accumulator.state.micro_batch_size
        
        self.accumulator.adjust_for_memory_pressure("low")
        
        # Should increase micro batch size if possible
        assert self.accumulator.state.micro_batch_size >= original_micro_batch_size
        assert self.accumulator.state.micro_batch_size <= self.config.max_micro_batch_size
    
    def test_calculate_optimal_accumulation(self):
        """Test optimal accumulation parameter calculation."""
        target_batch_size = 32
        available_memory_mb = 2000.0
        model_params = 1_000_000
        
        params = self.accumulator.calculate_optimal_accumulation(
            target_batch_size, available_memory_mb, model_params
        )
        
        assert "micro_batch_size" in params
        assert "accumulation_steps" in params
        assert "effective_batch_size" in params
        assert "memory_utilization_mb" in params
        
        assert params["micro_batch_size"] >= 1
        assert params["accumulation_steps"] >= 1
        assert params["effective_batch_size"] == params["micro_batch_size"] * params["accumulation_steps"]
        assert params["memory_utilization_mb"] > 0
    
    def test_get_accumulation_info(self):
        """Test accumulation info retrieval."""
        info = self.accumulator.get_accumulation_info()
        
        assert "config" in info
        assert "state" in info
        assert "progress" in info
        
        # Check config info
        assert info["config"]["target_batch_size"] == self.config.target_batch_size
        assert info["config"]["gradient_clipping"] == self.config.gradient_clipping
        
        # Check state info
        assert info["state"]["micro_batch_size"] == self.accumulator.state.micro_batch_size
        assert info["state"]["accumulation_steps"] == self.accumulator.state.accumulation_steps
        
        # Check progress info
        assert "accumulation_progress" in info["progress"]
        assert "steps_until_optimizer_step" in info["progress"]
        assert "should_step_next" in info["progress"]
    
    def test_reset_accumulation(self):
        """Test accumulation state reset."""
        # Accumulate some gradients first
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        self.accumulator.accumulate_gradients(self.model, loss)
        
        assert self.accumulator.state.accumulated_steps == 1
        
        # Reset accumulation
        self.accumulator.reset_accumulation()
        
        assert self.accumulator.state.accumulated_steps == 0
        # Gradients should be zeroed
        for param in self.model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_update_target_batch_size(self):
        """Test target batch size update."""
        new_target = 64
        original_accumulation_steps = self.accumulator.state.accumulated_steps
        
        # Accumulate some gradients first
        x = torch.randn(4, 10)
        target = torch.randn(4, 1)
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        self.accumulator.accumulate_gradients(self.model, loss)
        
        self.accumulator.update_target_batch_size(new_target)
        
        assert self.accumulator.config.target_batch_size == new_target
        # Should recalculate parameters
        expected_accumulation_steps = new_target // self.accumulator.state.micro_batch_size
        assert self.accumulator.state.accumulation_steps == expected_accumulation_steps
        # Should reset accumulation state
        assert self.accumulator.state.accumulated_steps == 0


class TestTrainingLoopIntegration:
    """Test TrainingLoopIntegration class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = GradientAccumulationConfig(target_batch_size=8, max_micro_batch_size=2)
        self.accumulator = GradientAccumulator(config)
        self.integration = TrainingLoopIntegration(self.accumulator)
        
        # Create test model and optimizer
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.accumulator.setup_optimizer(self.optimizer)
    
    def test_training_step(self):
        """Test training step execution."""
        # Create dummy batch
        batch = (torch.randn(2, 10), torch.randn(2, 1))  # input, target
        
        def loss_fn(model, batch):
            x, target = batch
            output = model(x)
            return nn.MSELoss()(output, target)
        
        training_info = self.integration.training_step(self.model, batch, loss_fn)
        
        assert "loss" in training_info
        assert "scaled_loss" in training_info
        assert "accumulated_steps" in training_info
        assert "should_step" in training_info
        assert "stepped" in training_info
        
        assert training_info["loss"] > 0
        assert training_info["accumulated_steps"] == 1
        assert training_info["should_step"] is False  # First step shouldn't trigger optimizer
        assert training_info["stepped"] is False
    
    def test_training_step_with_optimizer_step(self):
        """Test training step that triggers optimizer step."""
        batch = (torch.randn(2, 10), torch.randn(2, 1))
        
        def loss_fn(model, batch):
            x, target = batch
            output = model(x)
            return nn.MSELoss()(output, target)
        
        # Execute multiple training steps to trigger optimizer step
        training_infos = []
        for i in range(4):  # 4 steps should trigger optimizer (8 target / 2 micro = 4)
            info = self.integration.training_step(self.model, batch, loss_fn)
            training_infos.append(info)
        
        # Last step should trigger optimizer
        last_info = training_infos[-1]
        assert last_info["stepped"] is True
        assert last_info["should_step"] is True
        assert "optimizer_stepped" in last_info
        assert last_info["optimizer_stepped"] is True
    
    def test_get_effective_learning_rate(self):
        """Test effective learning rate calculation."""
        lr = self.integration.get_effective_learning_rate()
        
        # Should return the base learning rate (no scheduler in this test)
        assert lr == 0.01
    
    def test_get_effective_learning_rate_with_scheduler(self):
        """Test effective learning rate with scheduler."""
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.accumulator.setup_optimizer(self.optimizer, scheduler)
        
        lr = self.integration.get_effective_learning_rate()
        assert lr == 0.01  # Initial learning rate
        
        # Step scheduler
        scheduler.step()
        lr = self.integration.get_effective_learning_rate()
        assert lr == 0.005  # 0.01 * 0.5
    
    def test_should_log_metrics(self):
        """Test metrics logging condition."""
        # Initially should not log (no optimizer step yet)
        assert self.integration.should_log_metrics() is False
        
        # After accumulation steps equal to accumulation_steps, should log
        self.accumulator.state.accumulated_steps = 4  # Equal to accumulation_steps
        assert self.integration.should_log_metrics() is True
        
        # After one more step, should not log
        self.accumulator.state.accumulated_steps = 5
        assert self.integration.should_log_metrics() is False
    
    def test_get_logging_info(self):
        """Test logging information retrieval."""
        logging_info = self.integration.get_logging_info()
        
        assert "effective_batch_size" in logging_info
        assert "micro_batch_size" in logging_info
        assert "accumulation_steps" in logging_info
        assert "gradient_scale" in logging_info
        assert "optimizer_steps" in logging_info
        assert "samples_processed" in logging_info
        assert "effective_learning_rate" in logging_info
        
        assert logging_info["effective_batch_size"] == 8
        assert logging_info["micro_batch_size"] == 2
        assert logging_info["accumulation_steps"] == 4
        assert logging_info["gradient_scale"] == 0.25
        assert logging_info["optimizer_steps"] == 0
        assert logging_info["effective_learning_rate"] == 0.01


class TestGradientAccumulationIntegration:
    """Integration tests for gradient accumulation system."""
    
    def test_full_training_loop_simulation(self):
        """Test complete training loop with gradient accumulation."""
        config = GradientAccumulationConfig(
            target_batch_size=16,
            max_micro_batch_size=4,
            gradient_clipping=1.0
        )
        accumulator = GradientAccumulator(config)
        integration = TrainingLoopIntegration(accumulator)
        
        # Create model and optimizer
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        accumulator.setup_optimizer(optimizer)
        
        # Simulate training loop
        total_steps = 8
        optimizer_steps = 0
        
        for step in range(total_steps):
            # Create batch
            batch = (torch.randn(4, 10), torch.randn(4, 1))
            
            def loss_fn(model, batch):
                x, target = batch
                output = model(x)
                return nn.MSELoss()(output, target)
            
            # Training step
            info = integration.training_step(model, batch, loss_fn)
            
            if info["stepped"]:
                optimizer_steps += 1
                
                # Log metrics when optimizer steps
                if integration.should_log_metrics():
                    logging_info = integration.get_logging_info()
                    assert logging_info["effective_batch_size"] == 16
                    assert logging_info["optimizer_steps"] == optimizer_steps
        
        # Should have 2 optimizer steps (8 micro batches / 4 accumulation = 2)
        assert optimizer_steps == 2
        assert accumulator.state.current_step == 2
    
    def test_memory_pressure_adaptation(self):
        """Test gradient accumulation adaptation to memory pressure."""
        config = GradientAccumulationConfig(
            target_batch_size=32,
            max_micro_batch_size=8
        )
        accumulator = GradientAccumulator(config)
        
        # Initial state
        assert accumulator.state.micro_batch_size == 8
        assert accumulator.state.accumulation_steps == 4
        
        # Simulate high memory pressure
        accumulator.adjust_for_memory_pressure("high")
        
        # Should reduce micro batch size and increase accumulation steps
        assert accumulator.state.micro_batch_size < 8
        assert accumulator.state.accumulation_steps > 4
        # Effective batch size should be maintained or close
        assert accumulator.state.effective_batch_size <= 32
        
        # Simulate low memory pressure
        accumulator.adjust_for_memory_pressure("low")
        
        # Should try to increase micro batch size back up
        assert accumulator.state.micro_batch_size <= 8  # Limited by max
    
    def test_gradient_accumulation_correctness(self):
        """Test that gradient accumulation produces correct gradients."""
        # Create two identical models
        model1 = SimpleModel()
        model2 = SimpleModel()
        
        # Copy weights to ensure they start identical
        model2.load_state_dict(model1.state_dict())
        
        # Create optimizers
        optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        
        # Setup gradient accumulation for model1
        config = GradientAccumulationConfig(
            target_batch_size=8,
            max_micro_batch_size=2,
            scale_gradients=True
        )
        accumulator = GradientAccumulator(config)
        accumulator.setup_optimizer(optimizer1)
        
        # Create data
        x_full = torch.randn(8, 10)  # Full batch
        target_full = torch.randn(8, 1)
        
        # Train model2 with full batch (reference)
        optimizer2.zero_grad()
        output2 = model2(x_full)
        loss2 = nn.MSELoss()(output2, target_full)
        loss2.backward()
        optimizer2.step()
        
        # Train model1 with gradient accumulation (4 micro batches of size 2)
        for i in range(4):
            start_idx = i * 2
            end_idx = start_idx + 2
            x_micro = x_full[start_idx:end_idx]
            target_micro = target_full[start_idx:end_idx]
            
            output1 = model1(x_micro)
            loss1 = nn.MSELoss()(output1, target_micro)
            
            accumulator.accumulate_gradients(model1, loss1)
            
            if accumulator.should_step():
                accumulator.step_optimizer()
        
        # Models should have very similar parameters after training
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), "Gradient accumulation should produce equivalent results"