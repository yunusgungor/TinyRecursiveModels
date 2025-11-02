"""
Tool-Enhanced TRM Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import json
import numpy as np

from models.rl.rl_trm import RLEnhancedTRM, RLTRMConfig
from models.rl.environment import EnvironmentState, GiftItem
from .tool_registry import ToolRegistry, ToolCall
from .gift_tools import GiftRecommendationTools


class ToolEnhancedTRMConfig(RLTRMConfig):
    """Extended config for tool-enhanced TRM"""
    
    # Tool-specific parameters
    max_tool_calls_per_step: int = 3
    tool_call_threshold: float = 0.5  # Minimum confidence to call tool
    tool_result_encoding_dim: int = 128
    tool_selection_method: str = "confidence"  # "confidence", "random", "round_robin"
    
    # Tool integration
    tool_fusion_method: str = "concatenate"  # "concatenate", "attention", "gating"
    tool_attention_heads: int = 4
    
    # Training parameters
    tool_usage_reward_weight: float = 0.1
    tool_efficiency_penalty: float = 0.05


class ToolEnhancedTRM(RLEnhancedTRM):
    """TRM model enhanced with tool usage capabilities"""
    
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.tool_config = ToolEnhancedTRMConfig(**config_dict)
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._setup_tools()
        
        # Tool-related neural components
        self._init_tool_components()
        
        # Tool usage history
        self.tool_call_history = []
        
    def _setup_tools(self):
        """Setup and register all available tools"""
        gift_tools = GiftRecommendationTools()
        for tool in gift_tools.get_all_tools():
            self.tool_registry.register_tool(tool)
        
        print(f"Registered {len(self.tool_registry.list_tools())} tools")
    
    def _init_tool_components(self):
        """Initialize tool-related neural network components"""
        hidden_size = self.config.hidden_size
        num_tools = len(self.tool_registry.list_tools())
        
        # Tool selection head - decides which tool to use
        self.tool_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_tools + 1),  # +1 for "no tool"
            nn.Softmax(dim=-1)
        )
        
        # Tool parameter generator - generates parameters for tool calls
        self.tool_param_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.tool_config.tool_result_encoding_dim)
        )
        
        # Tool result encoder - encodes tool results back to hidden space
        self.tool_result_encoder = nn.Sequential(
            nn.Linear(self.tool_config.tool_result_encoding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Tool fusion mechanism
        if self.tool_config.tool_fusion_method == "attention":
            self.tool_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=self.tool_config.tool_attention_heads,
                batch_first=True
            )
        elif self.tool_config.tool_fusion_method == "gating":
            self.tool_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        # Tool usage predictor - predicts if tool usage will be beneficial
        self.tool_usage_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def decide_tool_usage(self, hidden_state: torch.Tensor, 
                         env_state: EnvironmentState) -> Tuple[str, float, Dict[str, Any]]:
        """
        Decide which tool to use based on current state
        
        Args:
            hidden_state: Current hidden state from TRM
            env_state: Current environment state
            
        Returns:
            Tuple of (tool_name, confidence, parameters)
        """
        # Get tool selection probabilities
        tool_probs = self.tool_selector(hidden_state.mean(dim=1))  # Average over sequence
        tool_names = self.tool_registry.list_tools() + ["no_tool"]
        
        if self.tool_config.tool_selection_method == "confidence":
            # Select tool with highest confidence
            max_prob, max_idx = torch.max(tool_probs, dim=-1)
            selected_tool = tool_names[max_idx.item()]
            confidence = max_prob.item()
            
        elif self.tool_config.tool_selection_method == "random":
            # Sample from distribution
            tool_idx = torch.multinomial(tool_probs, 1).item()
            selected_tool = tool_names[tool_idx]
            confidence = tool_probs[0, tool_idx].item()
            
        elif self.tool_config.tool_selection_method == "round_robin":
            # Simple round-robin selection (for testing)
            tool_idx = len(self.tool_call_history) % len(tool_names)
            selected_tool = tool_names[tool_idx]
            confidence = tool_probs[0, tool_idx].item()
        
        # Generate parameters if tool is selected
        parameters = {}
        if selected_tool != "no_tool" and confidence > self.tool_config.tool_call_threshold:
            parameters = self._generate_tool_parameters(
                hidden_state, selected_tool, env_state
            )
        else:
            selected_tool = "no_tool"
            confidence = 0.0
        
        return selected_tool, confidence, parameters
    
    def _generate_tool_parameters(self, hidden_state: torch.Tensor, 
                                 tool_name: str, env_state: EnvironmentState) -> Dict[str, Any]:
        """Generate parameters for a specific tool call"""
        # Generate parameter encoding
        param_encoding = self.tool_param_generator(hidden_state.mean(dim=1))
        
        # Convert encoding to tool-specific parameters
        user_profile = env_state.user_profile
        
        if tool_name == "price_comparison":
            # For price comparison, we need a product name
            # In a real implementation, this would be more sophisticated
            return {
                "product_name": self._extract_product_name_from_context(env_state),
                "max_sites": min(5, max(2, int(param_encoding[0].item() * 10) if param_encoding.dim() > 0 else 3)),
                "category": self._infer_category_from_hobbies(user_profile.hobbies)
            }
            
        elif tool_name == "inventory_check":
            return {
                "product_id": "default_product",
                "location": "TR"
            }
            
        elif tool_name == "review_analysis":
            return {
                "product_id": "default_product",
                "max_reviews": min(200, max(50, int(param_encoding[1].item() * 300) if param_encoding.dim() > 1 else int(param_encoding[0].item() * 300))),
                "language": "tr"
            }
            
        elif tool_name == "trend_analysis":
            return {
                "category": self._infer_category_from_hobbies(user_profile.hobbies),
                "time_period": "30d",
                "region": "TR"
            }
            
        elif tool_name == "budget_optimizer":
            return {
                "budget": user_profile.budget,
                "preferences": user_profile.hobbies,
                "occasion": user_profile.occasion
            }
        
        return {}
    
    def _extract_product_name_from_context(self, env_state: EnvironmentState) -> str:
        """Extract product name from environment context"""
        # Simple heuristic - in practice, this would be more sophisticated
        if env_state.current_recommendations:
            return env_state.current_recommendations[0].name
        
        # Fallback based on hobbies
        hobby_products = {
            "gardening": "garden tools",
            "cooking": "kitchen appliances", 
            "reading": "books",
            "sports": "fitness equipment",
            "technology": "gadgets"
        }
        
        for hobby in env_state.user_profile.hobbies:
            if hobby in hobby_products:
                return hobby_products[hobby]
        
        return "gift item"
    
    def _infer_category_from_hobbies(self, hobbies: List[str]) -> str:
        """Infer product category from user hobbies"""
        if not hobbies:
            return "general"
        
        # Simple mapping - could be more sophisticated
        category_mapping = {
            "gardening": "gardening",
            "cooking": "cooking",
            "reading": "books",
            "sports": "sports",
            "technology": "technology",
            "art": "art",
            "music": "music"
        }
        
        for hobby in hobbies:
            if hobby in category_mapping:
                return category_mapping[hobby]
        
        return hobbies[0]  # Fallback to first hobby
    
    def execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """Execute a tool call and return the result"""
        tool_call = ToolCall(tool_name=tool_name, parameters=parameters)
        result = self.tool_registry.call_tool(tool_call)
        
        # Add to history
        self.tool_call_history.append(result)
        
        return result
    
    def encode_tool_result(self, tool_result: Any) -> torch.Tensor:
        """
        Encode tool result into tensor representation
        
        Args:
            tool_result: Result from tool execution
            
        Returns:
            Tensor encoding of the result
        """
        device = next(self.parameters()).device
        
        if tool_result is None:
            return torch.zeros(self.tool_config.tool_result_encoding_dim, device=device)
        
        # Convert tool result to numerical features
        features = []
        
        if isinstance(tool_result, dict):
            # Extract numerical features from dictionary
            for key, value in tool_result.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, bool):
                    features.append(float(value))
                elif isinstance(value, str):
                    # Simple string encoding
                    features.append(float(hash(value) % 1000) / 1000.0)
                elif isinstance(value, list):
                    # List length and first few elements
                    features.append(float(len(value)))
                    for item in value[:3]:  # First 3 items
                        if isinstance(item, (int, float)):
                            features.append(float(item))
                        else:
                            features.append(float(hash(str(item)) % 100) / 100.0)
        
        elif isinstance(tool_result, (list, tuple)):
            features.append(float(len(tool_result)))
            for item in tool_result[:10]:  # First 10 items
                if isinstance(item, (int, float)):
                    features.append(float(item))
                else:
                    features.append(float(hash(str(item)) % 100) / 100.0)
        
        elif isinstance(tool_result, (int, float)):
            features.append(float(tool_result))
        
        else:
            # Fallback for other types
            features.append(float(hash(str(tool_result)) % 1000) / 1000.0)
        
        # Pad or truncate to fixed size
        target_size = self.tool_config.tool_result_encoding_dim
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        # Convert to tensor and encode
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        # Add batch dimension for encoder
        feature_tensor = feature_tensor.unsqueeze(0)  # [1, feature_dim]
        encoded_result = self.tool_result_encoder(feature_tensor)
        # Remove batch dimension
        encoded_result = encoded_result.squeeze(0)  # [hidden_size]
        
        return encoded_result
    
    def fuse_tool_results(self, hidden_state: torch.Tensor, 
                         tool_encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse tool results with hidden state
        
        Args:
            hidden_state: Current hidden state
            tool_encodings: List of encoded tool results
            
        Returns:
            Fused hidden state
        """
        if not tool_encodings:
            return hidden_state
        
        # Stack tool encodings
        tool_stack = torch.stack(tool_encodings, dim=0).unsqueeze(0)  # [1, num_tools, hidden_size]
        
        if self.tool_config.tool_fusion_method == "concatenate":
            # Robust tool fusion with proper dimension handling
            
            # Get tool summary - average across tools
            tool_summary = tool_stack.mean(dim=1)  # [1, tool_encoding_dim]
            
            # Store original hidden state shape
            original_shape = hidden_state.shape
            batch_size = hidden_state.size(0) if hidden_state.dim() > 1 else 1
            hidden_size = hidden_state.size(-1)
            
            # Ensure both tensors are 2D for processing
            if hidden_state.dim() == 1:
                hidden_state = hidden_state.unsqueeze(0)  # [1, hidden_size]
            if tool_summary.dim() == 1:
                tool_summary = tool_summary.unsqueeze(0)  # [1, tool_encoding_dim]
            
            # Ensure tool_summary matches hidden_state's last dimension
            if tool_summary.size(-1) != hidden_size:
                if not hasattr(self, 'tool_projection'):
                    self.tool_projection = nn.Linear(tool_summary.size(-1), hidden_size).to(hidden_state.device)
                tool_summary = self.tool_projection(tool_summary)
            
            # Ensure batch dimensions match
            if tool_summary.size(0) != hidden_state.size(0):
                tool_summary = tool_summary.expand(hidden_state.size(0), -1)
            
            # Concatenate along feature dimension
            combined = torch.cat([hidden_state, tool_summary], dim=-1)  # [batch, hidden_size * 2]
            
            # Project back to original hidden size
            if not hasattr(self, 'fusion_projection'):
                self.fusion_projection = nn.Linear(combined.size(-1), hidden_size).to(hidden_state.device)
            result = self.fusion_projection(combined)
            
            # Restore original shape
            if len(original_shape) == 1:
                result = result.squeeze(0)
            
            return result
            
        elif self.tool_config.tool_fusion_method == "attention":
            # Attention-based fusion
            fused_state, _ = self.tool_attention(
                hidden_state, tool_stack, tool_stack
            )
            return fused_state
            
        elif self.tool_config.tool_fusion_method == "gating":
            # Gating mechanism
            tool_summary = tool_stack.mean(dim=1)  # Average tool results
            gate_input = torch.cat([hidden_state, tool_summary], dim=-1)
            gate = self.tool_gate(gate_input)
            
            return hidden_state * gate + tool_summary * (1 - gate)
        
        else:
            # Default: simple addition
            return hidden_state + tool_stack.mean(dim=1)
    
    def forward_with_tools(self, carry, env_state: EnvironmentState, 
                          available_gifts: List[GiftItem],
                          max_tool_calls: Optional[int] = None) -> Tuple[Any, torch.Tensor, List[ToolCall]]:
        """
        Forward pass with tool usage
        
        Args:
            carry: TRM carry state
            env_state: Current environment state
            available_gifts: Available gift items
            max_tool_calls: Maximum number of tool calls (overrides config)
            
        Returns:
            Tuple of (new_carry, recommendations, tool_calls)
        """
        tool_calls = []
        tool_encodings = []
        
        max_calls = max_tool_calls or self.tool_config.max_tool_calls_per_step
        
        # Initial forward pass
        rl_output = self.forward_rl(carry, env_state, available_gifts)
        current_hidden = rl_output["hidden_state"]
        
        # Tool usage loop
        for step in range(max_calls):
            # Decide if we should use a tool
            tool_usage_prob = self.tool_usage_predictor(current_hidden).item()
            
            if tool_usage_prob < self.tool_config.tool_call_threshold:
                break
            
            # Select tool and generate parameters
            tool_name, confidence, parameters = self.decide_tool_usage(
                current_hidden.unsqueeze(1), env_state
            )
            
            if tool_name == "no_tool" or confidence < self.tool_config.tool_call_threshold:
                break
            
            # Execute tool call
            tool_call = self.execute_tool_call(tool_name, parameters)
            tool_calls.append(tool_call)
            
            if tool_call.success:
                # Encode tool result
                tool_encoding = self.encode_tool_result(tool_call.result)
                tool_encodings.append(tool_encoding)
                
                # Update hidden state with tool result
                current_hidden = self.fuse_tool_results(
                    current_hidden.unsqueeze(1), [tool_encoding]
                ).squeeze(1)
        
        # Final forward pass with tool-enhanced state
        if tool_encodings:
            # Update carry with tool-enhanced hidden state
            # Ensure dimensions match before updating carry
            if hasattr(carry, 'z_H'):
                target_shape = carry.z_H.shape
                
                if current_hidden.dim() == 1:
                    # Reshape to match target dimensions
                    if len(target_shape) == 3:  # [batch, seq, hidden]
                        current_hidden = current_hidden.unsqueeze(0).unsqueeze(0).expand(target_shape)
                    elif len(target_shape) == 2:  # [batch, hidden]
                        current_hidden = current_hidden.unsqueeze(0).expand(target_shape)
                elif current_hidden.dim() == 2 and len(target_shape) == 3:
                    # Add sequence dimension
                    current_hidden = current_hidden.unsqueeze(1).expand(target_shape)
                
                carry.z_H = current_hidden
            
            # Re-run RL forward with updated state
            final_output = self.forward_rl(carry, env_state, available_gifts)
        else:
            final_output = rl_output
        
        return final_output["carry"], final_output, tool_calls
    
    def compute_tool_usage_reward(self, tool_calls: List[ToolCall], 
                                 base_reward: float, user_feedback: Dict[str, Any]) -> float:
        """
        Compute additional reward based on tool usage effectiveness
        
        Args:
            tool_calls: List of tool calls made
            base_reward: Base reward from recommendation quality
            user_feedback: User feedback on recommendations
            
        Returns:
            Additional reward for tool usage
        """
        if not tool_calls:
            return 0.0
        
        tool_reward = 0.0
        
        for tool_call in tool_calls:
            if not tool_call.success:
                # Penalty for failed tool calls
                tool_reward -= 0.1
                continue
            
            # Reward based on tool type and user feedback
            if tool_call.tool_name == "price_comparison":
                if user_feedback.get("price_sensitive", False):
                    tool_reward += 0.2  # Good tool choice for price-sensitive user
                
            elif tool_call.tool_name == "review_analysis":
                if user_feedback.get("quality_focused", False):
                    tool_reward += 0.2  # Good for quality-focused user
                
            elif tool_call.tool_name == "trend_analysis":
                if user_feedback.get("trendy", False):
                    tool_reward += 0.15  # Good for trend-conscious user
                
            elif tool_call.tool_name == "budget_optimizer":
                if user_feedback.get("budget_conscious", False):
                    tool_reward += 0.25  # Very good for budget-conscious user
            
            # Efficiency penalty for too many tool calls
            if len(tool_calls) > 2:
                tool_reward -= self.tool_config.tool_efficiency_penalty
        
        return tool_reward * self.tool_config.tool_usage_reward_weight
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        if not self.tool_call_history:
            return {"total_calls": 0}
        
        # Count calls by tool
        tool_counts = {}
        success_counts = {}
        total_time = 0.0
        
        for call in self.tool_call_history:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
            if call.success:
                success_counts[call.tool_name] = success_counts.get(call.tool_name, 0) + 1
            total_time += call.execution_time
        
        # Calculate success rates
        success_rates = {
            tool: success_counts.get(tool, 0) / count
            for tool, count in tool_counts.items()
        }
        
        return {
            "total_calls": len(self.tool_call_history),
            "tool_counts": tool_counts,
            "success_rates": success_rates,
            "average_execution_time": total_time / len(self.tool_call_history),
            "most_used_tool": max(tool_counts.keys(), key=tool_counts.get) if tool_counts else None
        }
    
    def clear_tool_history(self):
        """Clear tool call history"""
        self.tool_call_history.clear()
        self.tool_registry.clear_history()


if __name__ == "__main__":
    # Test tool-enhanced TRM
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
        "max_recommendations": 3,
        "max_tool_calls_per_step": 2,
        "tool_call_threshold": 0.3
    }
    
    model = ToolEnhancedTRM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Available tools: {model.tool_registry.list_tools()}")
    
    # Test forward pass with tools
    from models.rl.environment import UserProfile, EnvironmentState, GiftItem
    
    user = UserProfile(35, ["gardening", "cooking"], "mother", 100.0, "birthday", ["eco-conscious"])
    gifts = [
        GiftItem("1", "Organic Seeds", "gardening", 50.0, 4.5, ["organic"], "Seeds", (20, 60), ["birthday"]),
        GiftItem("2", "Cookbook", "cooking", 30.0, 4.0, ["educational"], "Book", (18, 80), ["birthday"])
    ]
    
    env_state = EnvironmentState(user, gifts, [], [], 0)
    
    with torch.no_grad():
        carry = model.initial_carry({"inputs": torch.randn(50), "puzzle_identifiers": torch.zeros(1, dtype=torch.long)})
        new_carry, output, tool_calls = model.forward_with_tools(carry, env_state, gifts)
        
        print(f"\nMade {len(tool_calls)} tool calls:")
        for i, call in enumerate(tool_calls):
            print(f"  {i+1}. {call.tool_name}: {call.success}")
            if call.success and call.result:
                print(f"     Result keys: {list(call.result.keys()) if isinstance(call.result, dict) else type(call.result)}")
        
        # Get action from output
        action = model.select_action(output["action_probs"], gifts)
        print(f"\nFinal recommendations: {action['recommendations']}")
        print(f"Confidence scores: {action['confidence_scores']}")
        
        # Get tool usage stats
        stats = model.get_tool_usage_stats()
        print(f"\nTool usage stats: {stats}")