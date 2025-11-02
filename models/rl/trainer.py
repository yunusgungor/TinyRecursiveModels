"""
RL Trainer for Gift Recommendation TRM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
from collections import deque

from .environment import GiftRecommendationEnvironment, UserProfile, EnvironmentState
from .rl_trm import RLEnhancedTRM


@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    
    # PPO parameters
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Experience collection
    experience_buffer_size: int = 10000
    min_experiences_for_update: int = 100
    
    # Evaluation
    eval_frequency: int = 50
    eval_episodes: int = 10
    
    # Logging
    log_frequency: int = 10
    save_frequency: int = 100
    checkpoint_dir: str = "checkpoints/rl_training"
    
    # Tool usage (if applicable)
    enable_tools: bool = False
    tool_usage_reward_weight: float = 0.1


@dataclass
class Experience:
    """Single experience tuple for RL training"""
    state: EnvironmentState
    action: Dict[str, Any]
    reward: float
    next_state: EnvironmentState
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    carry: Any
    available_gifts: List[Any]
    tool_calls: List[Any] = None
    
    def detach_tensors(self):
        """Detach all tensors from gradient graph to avoid PPO issues"""
        if isinstance(self.log_prob, torch.Tensor):
            self.log_prob = self.log_prob.detach()
        if isinstance(self.value, torch.Tensor):
            self.value = self.value.detach()
        
        # Detach carry tensors if it's a TRM carry object
        if hasattr(self.carry, '__dict__'):
            for key, value in self.carry.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(self.carry, key, value.detach())
        
        return self


class RLTrainer:
    """Trainer for RL-enhanced TRM models"""
    
    def __init__(self, model: RLEnhancedTRM, environment: GiftRecommendationEnvironment,
                 config: TrainingConfig):
        self.model = model
        self.env = environment
        self.config = config
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=config.experience_buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_step = 0
        self.episode_count = 0
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_model_path = None
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def collect_experience(self, num_episodes: int) -> List[Experience]:
        """Collect experiences by running episodes"""
        experiences = []
        
        for episode in range(num_episodes):
            episode_experiences = self._run_episode()
            experiences.extend(episode_experiences)
            
            # Log episode statistics
            episode_reward = sum(exp.reward for exp in episode_experiences)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(episode_experiences))
            self.episode_count += 1
            
            if self.episode_count % self.config.log_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_frequency:])
                avg_length = np.mean(self.episode_lengths[-self.config.log_frequency:])
                
                print(f"Episode {self.episode_count}: "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Length: {avg_length:.1f}")
                
                if wandb.run:
                    wandb.log({
                        "episode": self.episode_count,
                        "avg_reward": avg_reward,
                        "avg_episode_length": avg_length,
                        "total_experiences": len(self.experience_buffer)
                    })
        
        return experiences
    
    def _run_episode(self) -> List[Experience]:
        """Run a single episode and collect experiences"""
        # Generate random user profile for this episode
        user_profile = self._generate_random_user_profile()
        
        # Reset environment
        state = self.env.reset(user_profile)
        
        # Initialize model carry
        dummy_batch = {
            "inputs": torch.randint(0, self.model.config.vocab_size, (self.model.config.seq_len,)),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
        }
        carry = self.model.initial_carry(dummy_batch)
        
        experiences = []
        done = False
        step = 0
        
        while not done and step < self.config.max_steps_per_episode:
            # Get available gifts
            available_gifts = state.available_gifts
            
            # Forward pass through model
            # Import ToolEnhancedTRM at runtime to avoid circular import
            from models.tools.tool_enhanced_trm import ToolEnhancedTRM
            if isinstance(self.model, ToolEnhancedTRM) and self.config.enable_tools:
                new_carry, rl_output, tool_calls = self.model.forward_with_tools(
                    carry, state, available_gifts
                )
            else:
                rl_output = self.model.forward_rl(carry, state, available_gifts)
                new_carry = rl_output["carry"]
                tool_calls = []
            
            # Select action
            action = self.model.select_action(
                rl_output["action_probs"], 
                available_gifts,
                deterministic=False
            )
            
            # Take step in environment
            next_state, reward, done, info = self.env.step({
                "recommendations": action["recommendations"],
                "confidence_scores": action["confidence_scores"]
            })
            
            # Add tool usage reward if applicable
            if tool_calls and self.config.enable_tools:
                # Mock user feedback for tool reward calculation
                user_feedback = self._generate_mock_user_feedback(state.user_profile, action)
                tool_reward = self.model.compute_tool_usage_reward(
                    tool_calls, reward, user_feedback
                )
                reward += tool_reward
            
            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=action["log_probs"],
                value=rl_output["state_value"],
                carry=carry,
                available_gifts=available_gifts,
                tool_calls=tool_calls if tool_calls else None
            )
            
            # Detach tensors to prevent gradient graph issues in PPO
            experience.detach_tensors()
            
            experiences.append(experience)
            
            # Update for next step
            state = next_state
            carry = new_carry
            step += 1
        
        return experiences
    
    def _generate_random_user_profile(self) -> UserProfile:
        """Generate a random user profile for training"""
        ages = list(range(18, 80))
        hobbies_pool = ["gardening", "cooking", "reading", "sports", "music", "art", "technology", "travel"]
        relationships = ["mother", "father", "friend", "partner", "sibling", "colleague"]
        occasions = ["birthday", "christmas", "anniversary", "graduation", "wedding"]
        personality_traits = ["eco-conscious", "practical", "trendy", "traditional", "adventurous"]
        
        return UserProfile(
            age=np.random.choice(ages),
            hobbies=np.random.choice(hobbies_pool, size=np.random.randint(1, 4), replace=False).tolist(),
            relationship=np.random.choice(relationships),
            budget=float(np.random.uniform(20, 500)),
            occasion=np.random.choice(occasions),
            personality_traits=np.random.choice(personality_traits, size=np.random.randint(1, 3), replace=False).tolist()
        )
    
    def _generate_mock_user_feedback(self, user_profile: UserProfile, 
                                   action: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock user feedback for tool reward calculation"""
        # Simple heuristic-based feedback generation
        feedback = {
            "price_sensitive": user_profile.budget < 100,
            "quality_focused": "practical" in user_profile.personality_traits,
            "trendy": "trendy" in user_profile.personality_traits,
            "budget_conscious": user_profile.budget < 150
        }
        
        return feedback
    
    def update_policy(self, experiences: List[Experience]) -> Dict[str, float]:
        """Update policy using PPO algorithm"""
        if len(experiences) < self.config.min_experiences_for_update:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        
        # Convert experiences to tensors
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs_old = []
        values_old = []
        
        for exp in experiences:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
            log_probs_old.append(exp.log_prob)
            values_old.append(exp.value)
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        log_probs_old = torch.stack(log_probs_old)
        values_old = torch.stack(values_old).squeeze(-1)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, values_old, dones)
        advantages = returns - values_old
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.config.ppo_epochs):
            # Re-evaluate policy for all experiences
            current_log_probs = []
            current_values = []
            entropies = []
            
            for exp in experiences:
                # Forward pass - experiences are already detached
                from models.tools.tool_enhanced_trm import ToolEnhancedTRM
                
                if isinstance(self.model, ToolEnhancedTRM) and self.config.enable_tools:
                    _, rl_output, _ = self.model.forward_with_tools(
                        exp.carry, exp.state, exp.available_gifts
                    )
                else:
                    rl_output = self.model.forward_rl(
                        exp.carry, exp.state, exp.available_gifts
                    )
                
                current_values.append(rl_output["state_value"])
                
                # Compute log probability for taken action
                action_probs = rl_output["action_probs"]
                if exp.action["action_indices"]:
                    # Handle both 1D and 2D action_probs tensors
                    if action_probs.dim() == 1:
                        action_log_prob = torch.log(
                            action_probs[exp.action["action_indices"]] + 1e-8
                        ).sum()
                    else:
                        action_log_prob = torch.log(
                            action_probs[0, exp.action["action_indices"]] + 1e-8
                        ).sum()
                else:
                    action_log_prob = torch.tensor(0.0)
                
                current_log_probs.append(action_log_prob)
                
                # Entropy for exploration
                entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                entropies.append(entropy)
            
            current_log_probs = torch.stack(current_log_probs)
            current_values = torch.stack(current_values).squeeze(-1)
            entropies = torch.stack(entropies)
            
            # PPO loss computation
            ratio = torch.exp(current_log_probs - log_probs_old)
            
            # Policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1 - self.config.clip_ratio, 
                1 + self.config.clip_ratio
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(current_values, returns)
            
            # Entropy loss
            entropy_loss = -entropies.mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.config.value_loss_coef * value_loss + 
                self.config.entropy_coef * entropy_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropies.mean().item()
        
        self.training_step += 1
        
        # Average losses over epochs
        avg_policy_loss = total_policy_loss / self.config.ppo_epochs
        avg_value_loss = total_value_loss / self.config.ppo_epochs
        avg_entropy = total_entropy / self.config.ppo_epochs
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "advantages_mean": advantages.mean().item(),
            "returns_mean": returns.mean().item()
        }
    
    def _compute_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                        dones: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate current policy"""
        if num_episodes is None:
            num_episodes = self.config.eval_episodes
        
        self.model.eval()
        
        eval_rewards = []
        eval_lengths = []
        eval_tool_usage = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                user_profile = self._generate_random_user_profile()
                state = self.env.reset(user_profile)
                
                dummy_batch = {
                    "inputs": torch.randint(0, self.model.config.vocab_size, (self.model.config.seq_len,)),
                    "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
                }
                carry = self.model.initial_carry(dummy_batch)
                
                episode_reward = 0.0
                episode_length = 0
                episode_tool_calls = 0
                done = False
                
                while not done and episode_length < self.config.max_steps_per_episode:
                    available_gifts = state.available_gifts
                    
                    from models.tools.tool_enhanced_trm import ToolEnhancedTRM
                    if isinstance(self.model, ToolEnhancedTRM) and self.config.enable_tools:
                        new_carry, rl_output, tool_calls = self.model.forward_with_tools(
                            carry, state, available_gifts
                        )
                        episode_tool_calls += len(tool_calls)
                    else:
                        rl_output = self.model.forward_rl(carry, state, available_gifts)
                        new_carry = rl_output["carry"]
                    
                    action = self.model.select_action(
                        rl_output["action_probs"], 
                        available_gifts,
                        deterministic=True  # Deterministic for evaluation
                    )
                    
                    next_state, reward, done, info = self.env.step({
                        "recommendations": action["recommendations"],
                        "confidence_scores": action["confidence_scores"]
                    })
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                    carry = new_carry
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_tool_usage.append(episode_tool_calls)
        
        self.model.train()
        
        eval_stats = {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_tool_usage_mean": np.mean(eval_tool_usage) if eval_tool_usage else 0.0
        }
        
        return eval_stats
    
    def save_checkpoint(self, filepath: str = None):
        """Save training checkpoint"""
        if filepath is None:
            filepath = os.path.join(
                self.config.checkpoint_dir, 
                f"checkpoint_step_{self.training_step}.pt"
            )
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "config": self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        
        return filepath
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Try to load optimizer state, but reset if incompatible
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Could not load optimizer state ({e}). Resetting optimizer.")
            # Reset optimizer with current model parameters
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.episode_lengths = checkpoint["episode_lengths"]
        
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        print(f"Starting RL training for {self.config.num_episodes} episodes")
        
        # Initialize wandb if available
        if wandb.run is None:
            wandb.init(
                project="gift-recommendation-rl",
                config=self.config.__dict__
            )
        
        training_start_time = time.time()
        
        for episode_batch in range(0, self.config.num_episodes, self.config.batch_size):
            batch_size = min(self.config.batch_size, self.config.num_episodes - episode_batch)
            
            # Collect experiences
            experiences = self.collect_experience(batch_size)
            
            # Add to buffer
            self.experience_buffer.extend(experiences)
            
            # Update policy
            if len(self.experience_buffer) >= self.config.min_experiences_for_update:
                # Sample from buffer
                sample_size = min(len(self.experience_buffer), self.config.batch_size * 10)
                sampled_experiences = np.random.choice(
                    list(self.experience_buffer), 
                    size=sample_size, 
                    replace=False
                ).tolist()
                
                loss_stats = self.update_policy(sampled_experiences)
                
                if wandb.run:
                    wandb.log({
                        "training_step": self.training_step,
                        **loss_stats
                    })
            
            # Evaluation
            if self.episode_count % self.config.eval_frequency == 0:
                eval_stats = self.evaluate()
                
                print(f"Evaluation at episode {self.episode_count}:")
                for key, value in eval_stats.items():
                    print(f"  {key}: {value:.3f}")
                
                if wandb.run:
                    wandb.log({
                        "episode": self.episode_count,
                        **eval_stats
                    })
                
                # Save best model
                if eval_stats["eval_reward_mean"] > self.best_reward:
                    self.best_reward = eval_stats["eval_reward_mean"]
                    self.best_model_path = self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, "best_model.pt")
                    )
            
            # Save checkpoint
            if self.episode_count % self.config.save_frequency == 0:
                self.save_checkpoint()
        
        training_time = time.time() - training_start_time
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=50)
        
        training_summary = {
            "total_episodes": self.episode_count,
            "total_training_time": training_time,
            "final_eval_reward": final_eval["eval_reward_mean"],
            "best_reward": self.best_reward,
            "best_model_path": self.best_model_path
        }
        
        print("\nTraining completed!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final evaluation reward: {final_eval['eval_reward_mean']:.3f}")
        print(f"Best reward achieved: {self.best_reward:.3f}")
        
        return training_summary


if __name__ == "__main__":
    # Test the trainer
    from models.rl.rl_trm import RLEnhancedTRM
    from models.rl.environment import GiftRecommendationEnvironment
    
    # Create sample gift catalog
    import os
    os.makedirs("data", exist_ok=True)
    
    from models.rl.environment import create_sample_gift_catalog
    create_sample_gift_catalog("data/sample_gift_catalog.json")
    
    # Model config
    model_config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "hidden_size": 128,  # Smaller for testing
        "H_cycles": 2,
        "L_cycles": 2,
        "L_layers": 2,
        "num_heads": 4,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,
        "action_space_size": 10,
        "max_recommendations": 2
    }
    
    # Training config
    training_config = TrainingConfig(
        num_episodes=50,  # Small for testing
        batch_size=5,
        learning_rate=1e-3,
        eval_frequency=10,
        save_frequency=20,
        checkpoint_dir="test_checkpoints"
    )
    
    # Create components
    model = RLEnhancedTRM(model_config)
    env = GiftRecommendationEnvironment("data/sample_gift_catalog.json")
    trainer = RLTrainer(model, env, training_config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Run training
    summary = trainer.train()
    print("Training summary:", summary)