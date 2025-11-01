"""
RL-Enhanced TRM Model for Gift Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry
)
from .environment import EnvironmentState, GiftItem


class RLTRMConfig(TinyRecursiveReasoningModel_ACTV1Config):
    """Extended config for RL-enhanced TRM"""
    
    # RL-specific parameters
    action_space_size: int = 100  # Max number of gifts to choose from
    max_recommendations: int = 5  # Max recommendations per step
    value_head_hidden: int = 256
    policy_head_hidden: int = 256
    
    # Reward prediction
    reward_prediction: bool = True
    reward_head_hidden: int = 128
    
    # Action selection
    action_selection_method: str = "top_k"  # "top_k", "sampling", "epsilon_greedy"
    epsilon: float = 0.1  # For epsilon-greedy
    temperature: float = 1.0  # For sampling
    
    # Training parameters
    ppo_clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


class RLEnhancedTRM(TinyRecursiveReasoningModel_ACTV1):
    """TRM model enhanced with RL capabilities for gift recommendation"""
    
    def __init__(self, config_dict: dict):
        # Initialize base TRM
        super().__init__(config_dict)
        self.rl_config = RLTRMConfig(**config_dict)
        
        # RL-specific components
        self._init_rl_heads()
        self._init_state_encoders()
        
        # Action history for experience replay
        self.action_history = []
        self.reward_history = []
        
    def _init_rl_heads(self):
        """Initialize RL-specific neural network heads"""
        hidden_size = self.config.hidden_size
        
        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, self.rl_config.policy_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.rl_config.policy_head_hidden, self.rl_config.action_space_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head - estimates state value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, self.rl_config.value_head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.rl_config.value_head_hidden, 1)
        )
        
        # Reward prediction head (optional)
        if self.rl_config.reward_prediction:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_size, self.rl_config.reward_head_hidden),
                nn.ReLU(),
                nn.Linear(self.rl_config.reward_head_hidden, 1),
                nn.Tanh()  # Rewards typically in [-1, 1]
            )
        
        # Gift scoring head - scores individual gifts
        self.gift_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Concat user state + gift features
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Score between 0 and 1
        )
        
    def _init_state_encoders(self):
        """Initialize encoders for environment state"""
        # User profile encoder
        self.user_encoder = nn.Sequential(
            nn.Linear(26, self.config.hidden_size),  # 26-dim user features (5+8+7+6)
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        # Gift item encoder
        self.gift_encoder = nn.Sequential(
            nn.Linear(20, self.config.hidden_size),  # Assuming 20-dim gift features
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
    def encode_environment_state(self, env_state: EnvironmentState) -> torch.Tensor:
        """Encode environment state into model representation"""
        # Get state tensor from environment
        state_tensor = env_state.to_tensor(device=next(self.parameters()).device)
        
        # Encode user profile
        user_encoding = self.user_encoder(state_tensor.unsqueeze(0))
        
        return user_encoding
    
    def encode_gift_items(self, gift_items: List[GiftItem]) -> torch.Tensor:
        """Encode gift items into model representation"""
        device = next(self.parameters()).device
        
        if not gift_items:
            return torch.zeros(1, self.config.hidden_size, device=device)
        
        # Convert gift items to feature vectors
        gift_features = []
        for gift in gift_items:
            features = [
                gift.price / 1000.0,  # Normalize price
                gift.rating / 5.0,    # Normalize rating
                len(gift.tags),       # Number of tags
                gift.age_suitability[0] / 100.0,  # Min age normalized
                gift.age_suitability[1] / 100.0,  # Max age normalized
                len(gift.occasion_fit),  # Number of occasions
            ]
            
            # Add category encoding (simplified)
            categories = ['gardening', 'cooking', 'reading', 'sports', 'technology', 'art', 'other']
            category_encoding = [1.0 if cat == gift.category else 0.0 for cat in categories]
            
            # Add tag encoding (simplified)
            common_tags = ['organic', 'sustainable', 'educational', 'experience', 'health', 'smart', 'premium']
            tag_encoding = [1.0 if tag in gift.tags else 0.0 for tag in common_tags]
            
            # Combine all features
            all_features = features + category_encoding + tag_encoding
            
            # Pad or truncate to fixed size (20 features)
            all_features = all_features[:20] + [0.0] * max(0, 20 - len(all_features))
            gift_features.append(all_features)
        
        # Convert to tensor and encode
        gift_tensor = torch.tensor(gift_features, dtype=torch.float32, device=device)
        gift_encodings = self.gift_encoder(gift_tensor)
        
        return gift_encodings
    
    def forward_rl(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, 
                   env_state: EnvironmentState, 
                   available_gifts: List[GiftItem],
                   return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for RL training
        
        Args:
            carry: TRM carry state
            env_state: Current environment state
            available_gifts: List of available gift items
            return_all: Whether to return all intermediate values
            
        Returns:
            Dictionary containing RL outputs
        """
        # Encode environment state
        user_encoding = self.encode_environment_state(env_state)
        
        # Create batch for TRM forward pass
        # Use dummy tokens for TRM input
        seq_len = self.config.seq_len
        dummy_tokens = torch.randint(0, self.config.vocab_size, (seq_len,), device=user_encoding.device)
        
        batch = {
            "inputs": dummy_tokens,
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long, device=user_encoding.device)
        }
        
        # TRM forward pass
        # Convert ACT carry to inner carry
        inner_carry = carry.inner_carry if hasattr(carry, 'inner_carry') else carry
        new_carry, logits, (q_halt, q_continue) = self.inner(inner_carry, batch)
        
        # Get hidden state for RL heads
        # z_H shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        hidden_state = new_carry.z_H.mean(dim=1)  # Average over sequence
        
        # If batch_size > 1, take first batch element
        if hidden_state.dim() > 1 and hidden_state.size(0) > 1:
            hidden_state = hidden_state[0]  # Take first batch element
        
        # Policy output - action probabilities over available gifts
        action_logits = self.policy_head(hidden_state)
        
        # Limit to available gifts
        if len(available_gifts) < self.rl_config.action_space_size:
            # Mask unavailable actions
            mask = torch.zeros_like(action_logits)
            if action_logits.dim() == 1:
                mask[:len(available_gifts)] = 1.0
            else:
                mask[:, :len(available_gifts)] = 1.0
            action_logits = action_logits * mask + (1 - mask) * (-1e9)
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Value estimation
        state_value = self.value_head(hidden_state)
        
        # Reward prediction (if enabled)
        predicted_reward = None
        if self.rl_config.reward_prediction:
            predicted_reward = self.reward_head(hidden_state)
        
        # Gift scoring for available gifts
        gift_encodings = self.encode_gift_items(available_gifts)
        gift_scores = []
        
        for gift_encoding in gift_encodings:
            # Concatenate user state and gift encoding
            # hidden_state shape: [1, hidden_size], gift_encoding shape: [hidden_size]
            user_state_flat = hidden_state.squeeze(0)  # Remove batch dim: [hidden_size]
            
            combined = torch.cat([user_state_flat, gift_encoding], dim=0)  # [hidden_size * 2]
            score = self.gift_scorer(combined.unsqueeze(0))  # Add batch dimension for scorer
            gift_scores.append(score)
        
        gift_scores = torch.cat(gift_scores, dim=0) if gift_scores else torch.tensor([])
        
        # Prepare output
        output = {
            "carry": new_carry,
            "action_probs": action_probs,
            "state_value": state_value,
            "gift_scores": gift_scores,
            "halt_decision": torch.sigmoid(q_halt),
            "hidden_state": hidden_state
        }
        
        if predicted_reward is not None:
            output["predicted_reward"] = predicted_reward
            
        if return_all:
            output.update({
                "logits": logits,
                "q_halt": q_halt,
                "q_continue": q_continue,
                "user_encoding": user_encoding,
                "gift_encodings": gift_encodings
            })
        
        return output
    
    def select_action(self, action_probs: torch.Tensor, 
                     available_gifts: List[GiftItem],
                     deterministic: bool = False) -> Dict[str, Any]:
        """
        Select action based on policy output
        
        Args:
            action_probs: Action probabilities from policy head
            available_gifts: List of available gifts
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Dictionary containing selected action
        """
        device = action_probs.device
        num_available = len(available_gifts)
        
        if num_available == 0:
            return {
                "recommendations": [],
                "confidence_scores": [],
                "action_indices": [],
                "log_probs": torch.tensor(0.0, device=device)
            }
        
        # Limit probabilities to available gifts
        if action_probs.dim() == 1:
            available_probs = action_probs[:num_available]
        else:
            available_probs = action_probs[0, :num_available]  # Remove batch dim and limit
        
        if deterministic or self.rl_config.action_selection_method == "top_k":
            # Select top-k gifts
            k = min(self.rl_config.max_recommendations, num_available)
            top_k_values, top_k_indices = torch.topk(available_probs, k)
            
            selected_indices = top_k_indices.detach().cpu().numpy()
            confidence_scores = top_k_values.detach().cpu().numpy()
            log_probs = torch.log(top_k_values + 1e-8).sum()
            
        elif self.rl_config.action_selection_method == "sampling":
            # Sample from distribution
            k = min(self.rl_config.max_recommendations, num_available)
            
            # Apply temperature
            temp_probs = available_probs / self.rl_config.temperature
            temp_probs = F.softmax(temp_probs, dim=-1)
            
            # Sample without replacement
            selected_indices = []
            remaining_probs = temp_probs.clone()
            log_probs_list = []
            
            for _ in range(k):
                if remaining_probs.sum() < 1e-8:
                    break
                    
                # Sample one index
                idx = torch.multinomial(remaining_probs, 1).item()
                selected_indices.append(idx)
                log_probs_list.append(torch.log(remaining_probs[idx] + 1e-8))
                
                # Remove selected item from future sampling
                remaining_probs[idx] = 0.0
                remaining_probs = remaining_probs / (remaining_probs.sum() + 1e-8)
            
            selected_indices = np.array(selected_indices)
            confidence_scores = available_probs[selected_indices].cpu().numpy()
            log_probs = torch.stack(log_probs_list).sum() if log_probs_list else torch.tensor(0.0, device=device)
            
        elif self.rl_config.action_selection_method == "epsilon_greedy":
            # Epsilon-greedy selection
            if not deterministic and np.random.random() < self.rl_config.epsilon:
                # Random selection
                k = min(self.rl_config.max_recommendations, num_available)
                selected_indices = np.random.choice(num_available, k, replace=False)
                confidence_scores = available_probs[selected_indices].detach().cpu().numpy()
                log_probs = torch.log(available_probs[selected_indices] + 1e-8).sum()
            else:
                # Greedy selection (same as top_k)
                k = min(self.rl_config.max_recommendations, num_available)
                top_k_values, top_k_indices = torch.topk(available_probs, k)
                
                selected_indices = top_k_indices.detach().cpu().numpy()
                confidence_scores = top_k_values.detach().cpu().numpy()
                log_probs = torch.log(top_k_values + 1e-8).sum()
        
        # Get selected gifts
        selected_gifts = [available_gifts[i] for i in selected_indices]
        gift_ids = [gift.id for gift in selected_gifts]
        
        return {
            "recommendations": gift_ids,
            "confidence_scores": confidence_scores.tolist(),
            "action_indices": selected_indices.tolist(),
            "log_probs": log_probs,
            "selected_gifts": selected_gifts
        }
    
    def compute_rl_loss(self, experiences: List[Dict], gamma: float = 0.99) -> Dict[str, torch.Tensor]:
        """
        Compute RL loss (PPO-style) from collected experiences
        
        Args:
            experiences: List of experience dictionaries
            gamma: Discount factor
            
        Returns:
            Dictionary containing loss components
        """
        if not experiences:
            return {"total_loss": torch.tensor(0.0)}
        
        device = next(self.parameters()).device
        
        # Extract data from experiences
        states = []
        actions = []
        rewards = []
        log_probs_old = []
        values_old = []
        dones = []
        
        for exp in experiences:
            states.append(exp["state"])
            actions.append(exp["action"])
            rewards.append(exp["reward"])
            log_probs_old.append(exp["log_prob"])
            values_old.append(exp["value"])
            dones.append(exp["done"])
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        log_probs_old = torch.stack(log_probs_old)
        values_old = torch.stack(values_old).squeeze(-1)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, values_old, dones, gamma)
        advantages = returns - values_old
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Re-evaluate current policy
        current_values = []
        current_log_probs = []
        
        for i, exp in enumerate(experiences):
            # Forward pass with current policy
            rl_output = self.forward_rl(
                exp["carry"], 
                exp["env_state"], 
                exp["available_gifts"]
            )
            
            current_values.append(rl_output["state_value"])
            
            # Compute log prob for taken action
            action_probs = rl_output["action_probs"]
            action_indices = exp["action"]["action_indices"]
            
            if action_indices:
                action_log_probs = torch.log(action_probs[0, action_indices] + 1e-8).sum()
            else:
                action_log_probs = torch.tensor(0.0, device=device)
            
            current_log_probs.append(action_log_probs)
        
        current_values = torch.stack(current_values).squeeze(-1)
        current_log_probs = torch.stack(current_log_probs)
        
        # PPO loss computation
        ratio = torch.exp(current_log_probs - log_probs_old)
        
        # Policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.rl_config.ppo_clip_ratio, 
                           1 + self.rl_config.ppo_clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(current_values, returns)
        
        # Entropy loss (for exploration)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.rl_config.entropy_coef * entropy
        
        # Total loss
        total_loss = (policy_loss + 
                     self.rl_config.value_loss_coef * value_loss + 
                     entropy_loss)
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "advantages_mean": advantages.mean(),
            "returns_mean": returns.mean()
        }
    
    def _compute_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                        dones: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save_experience(self, experience: Dict):
        """Save experience for replay"""
        self.action_history.append(experience)
        
        # Limit history size
        if len(self.action_history) > 10000:
            self.action_history = self.action_history[-5000:]
    
    def get_experiences(self, batch_size: int = None) -> List[Dict]:
        """Get batch of experiences for training"""
        if batch_size is None or batch_size >= len(self.action_history):
            return self.action_history.copy()
        
        # Random sampling
        indices = np.random.choice(len(self.action_history), batch_size, replace=False)
        return [self.action_history[i] for i in indices]
    
    def clear_experiences(self):
        """Clear experience buffer"""
        self.action_history.clear()
        self.reward_history.clear()


if __name__ == "__main__":
    # Test RL-enhanced TRM
    config = {
        "batch_size": 1,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "hidden_size": 256,
        "H_cycles": 2,
        "L_cycles": 3,
        "L_layers": 2,
        "num_heads": 8,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "halt_max_steps": 5,
        "halt_exploration_prob": 0.1,
        "action_space_size": 50,
        "max_recommendations": 3
    }
    
    model = RLEnhancedTRM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    from .environment import UserProfile, EnvironmentState, GiftItem
    
    user = UserProfile(35, ["gardening"], "mother", 100.0, "birthday", ["eco-conscious"])
    gifts = [
        GiftItem("1", "Seeds", "gardening", 50.0, 4.5, ["organic"], "Seeds", (20, 60), ["birthday"]),
        GiftItem("2", "Book", "reading", 30.0, 4.0, ["educational"], "Book", (18, 80), ["birthday"])
    ]
    
    env_state = EnvironmentState(user, gifts, [], [], 0)
    
    with torch.no_grad():
        carry = model.initial_carry({"inputs": torch.randn(50), "puzzle_identifiers": torch.zeros(1, dtype=torch.long)})
        output = model.forward_rl(carry, env_state, gifts)
        action = model.select_action(output["action_probs"], gifts)
        
        print("Action probabilities shape:", output["action_probs"].shape)
        print("State value:", output["state_value"].item())
        print("Selected action:", action["recommendations"])
        print("Confidence scores:", action["confidence_scores"])