"""
Integrated Enhanced TRM Model - All improvements built into the model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import json
import numpy as np

from models.rl.rl_trm import RLEnhancedTRM, RLTRMConfig
from models.rl.environment import EnvironmentState, GiftItem, UserProfile
from .tool_registry import ToolRegistry, ToolCall
from .gift_tools import GiftRecommendationTools


class IntegratedEnhancedTRMConfig(RLTRMConfig):
    """Configuration for integrated enhanced TRM with all improvements built-in"""
    
    # Enhanced user profiling
    user_profile_encoding_dim: int = 256
    hobby_embedding_dim: int = 64
    preference_embedding_dim: int = 32
    occasion_embedding_dim: int = 32
    age_encoding_dim: int = 16
    
    # Enhanced category matching
    category_embedding_dim: int = 128
    category_attention_heads: int = 8
    semantic_matching_layers: int = 2
    
    # Enhanced tool selection
    tool_context_encoding_dim: int = 128
    tool_selection_heads: int = 4
    max_tool_calls_per_step: int = 2
    tool_diversity_weight: float = 0.3
    
    # Enhanced reward calculation
    reward_components: int = 7  # category, budget, hobby, occasion, age, quality, diversity
    reward_fusion_layers: int = 3
    reward_prediction_dim: int = 64
    
    # Gift catalog integration
    gift_embedding_dim: int = 256
    gift_feature_dim: int = 128
    max_gifts_in_catalog: int = 100
    
    # Training enhancements
    category_loss_weight: float = 0.35
    tool_diversity_loss_weight: float = 0.15
    semantic_matching_loss_weight: float = 0.20
    
    # Architecture improvements
    enhanced_attention_layers: int = 4
    cross_modal_fusion_dim: int = 512


class IntegratedEnhancedTRM(RLEnhancedTRM):
    """TRM model with all enhancements integrated into the architecture"""
    
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.enhanced_config = IntegratedEnhancedTRMConfig(**config_dict)
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._setup_tools()
        
        # Initialize all enhanced components
        self._init_enhanced_user_profiler()
        self._init_enhanced_category_matcher()
        self._init_enhanced_tool_selector()
        self._init_enhanced_reward_predictor()
        self._init_gift_catalog_encoder()
        self._init_cross_modal_fusion()
        
        # Load and encode gift catalog
        self._load_and_encode_gift_catalog()
        
        print(f"🚀 Integrated Enhanced TRM initialized with all improvements built-in")
        
    def _setup_tools(self):
        """Setup and register all available tools"""
        gift_tools = GiftRecommendationTools()
        for tool in gift_tools.get_all_tools():
            self.tool_registry.register_tool(tool)
        print(f"Registered {len(self.tool_registry.list_tools())} tools")
    
    def _init_enhanced_user_profiler(self):
        """Initialize enhanced user profiling components"""
        config = self.enhanced_config
        
        # Hobby embeddings with semantic understanding
        self.hobby_categories = [
            'gardening', 'cooking', 'reading', 'sports', 'music', 'art', 'technology', 
            'travel', 'fitness', 'wellness', 'outdoor', 'gaming', 'photography', 
            'design', 'business', 'environment', 'sustainability', 'home_decor'
        ]
        self.hobby_embeddings = nn.Embedding(len(self.hobby_categories), config.hobby_embedding_dim)
        
        # Preference embeddings
        self.preference_categories = [
            'trendy', 'practical', 'tech-savvy', 'relaxing', 'self-care', 'affordable',
            'traditional', 'quality', 'active', 'healthy', 'motivational', 'creative',
            'unique', 'artistic', 'luxury', 'professional', 'sophisticated', 
            'eco-friendly', 'sustainable', 'natural'
        ]
        self.preference_embeddings = nn.Embedding(len(self.preference_categories), config.preference_embedding_dim)
        
        # Occasion embeddings
        self.occasion_categories = [
            'birthday', 'christmas', 'mothers_day', 'fathers_day', 'graduation',
            'anniversary', 'promotion', 'appreciation', 'new_year', 'wedding'
        ]
        self.occasion_embeddings = nn.Embedding(len(self.occasion_categories), config.occasion_embedding_dim)
        
        # Age encoding (continuous)
        self.age_encoder = nn.Sequential(
            nn.Linear(1, config.age_encoding_dim),
            nn.ReLU(),
            nn.Linear(config.age_encoding_dim, config.age_encoding_dim)
        )
        
        # Budget encoding (continuous)
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, config.age_encoding_dim),
            nn.ReLU(),
            nn.Linear(config.age_encoding_dim, config.age_encoding_dim)
        )
        
        # User profile fusion
        total_profile_dim = (config.hobby_embedding_dim + config.preference_embedding_dim + 
                           config.occasion_embedding_dim + config.age_encoding_dim * 2)
        
        self.user_profile_encoder = nn.Sequential(
            nn.Linear(total_profile_dim, config.user_profile_encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased from 0.1
            nn.Linear(config.user_profile_encoding_dim, config.user_profile_encoding_dim),
            nn.LayerNorm(config.user_profile_encoding_dim)
        )
        
    def _init_enhanced_category_matcher(self):
        """Initialize enhanced category matching components"""
        config = self.enhanced_config
        
        # Gift categories
        self.gift_categories = [
            'technology', 'gardening', 'cooking', 'books', 'wellness', 'art', 
            'fitness', 'outdoor', 'home', 'food', 'experience', 'gaming', 'fashion'
        ]
        self.category_embeddings = nn.Embedding(len(self.gift_categories), config.category_embedding_dim)
        
        # Semantic matching network - each layer processes category_embedding_dim
        self.semantic_matcher = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.category_embedding_dim, config.category_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)  # Increased from 0.1
            ) for _ in range(config.semantic_matching_layers)
        ])
        
        # Input projection for semantic matching
        self.semantic_input_proj = nn.Linear(
            config.user_profile_encoding_dim + config.category_embedding_dim,
            config.category_embedding_dim
        )
        
        # Category attention mechanism
        self.category_attention = nn.MultiheadAttention(
            embed_dim=config.category_embedding_dim,
            num_heads=config.category_attention_heads,
            batch_first=True
        )
        
        # Category scoring head
        self.category_scorer = nn.Sequential(
            nn.Linear(config.category_embedding_dim, config.category_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout to slow down category learning
            nn.Linear(config.category_embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _init_enhanced_tool_selector(self):
        """Initialize enhanced context-aware tool selection"""
        config = self.enhanced_config
        
        # Tool context encoder
        self.tool_context_encoder = nn.Sequential(
            nn.Linear(config.user_profile_encoding_dim, config.tool_context_encoding_dim),
            nn.ReLU(),
            nn.Linear(config.tool_context_encoding_dim, config.tool_context_encoding_dim)
        )
        
        # Tool selection with context awareness
        num_tools = len(self.tool_registry.list_tools())
        self.context_aware_tool_selector = nn.MultiheadAttention(
            embed_dim=config.tool_context_encoding_dim,
            num_heads=config.tool_selection_heads,
            batch_first=True
        )
        
        # Tool diversity enforcer
        self.tool_diversity_head = nn.Sequential(
            nn.Linear(config.tool_context_encoding_dim, num_tools),
            nn.Softmax(dim=-1)
        )
        
        # Tool parameter generator
        self.enhanced_tool_param_generator = nn.Sequential(
            nn.Linear(config.tool_context_encoding_dim + config.user_profile_encoding_dim, 
                     config.tool_context_encoding_dim),
            nn.ReLU(),
            nn.Linear(config.tool_context_encoding_dim, config.tool_context_encoding_dim)
        )
        
    def _init_enhanced_reward_predictor(self):
        """Initialize enhanced reward prediction with multiple components"""
        config = self.enhanced_config
        
        # Individual reward component predictors
        self.reward_components = nn.ModuleDict({
            'category_match': nn.Sequential(
                nn.Linear(config.gift_embedding_dim, config.reward_prediction_dim),
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'budget_compatibility': nn.Sequential(
                nn.Linear(config.age_encoding_dim + 1, config.reward_prediction_dim),  # budget + price
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'hobby_alignment': nn.Sequential(
                nn.Linear(config.hobby_embedding_dim + config.gift_embedding_dim, config.reward_prediction_dim),
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'occasion_appropriateness': nn.Sequential(
                nn.Linear(config.occasion_embedding_dim + config.gift_embedding_dim, config.reward_prediction_dim),
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'age_appropriateness': nn.Sequential(
                nn.Linear(config.age_encoding_dim + 2, config.reward_prediction_dim),  # age + age_range
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'quality_score': nn.Sequential(
                nn.Linear(1, config.reward_prediction_dim),  # rating
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            ),
            'diversity_bonus': nn.Sequential(
                nn.Linear(config.gift_embedding_dim * 2, config.reward_prediction_dim),  # current + previous
                nn.ReLU(),
                nn.Linear(config.reward_prediction_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # Reward fusion network
        reward_fusion_layers = []
        input_dim = len(self.reward_components)
        
        for i in range(config.reward_fusion_layers):
            output_dim = max(1, input_dim // 2) if i < config.reward_fusion_layers - 1 else 1
            reward_fusion_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU() if i < config.reward_fusion_layers - 1 else nn.Sigmoid()
            ])
            input_dim = output_dim
        
        self.reward_fusion = nn.Sequential(*reward_fusion_layers)
        
    def _init_gift_catalog_encoder(self):
        """Initialize gift catalog encoding components"""
        config = self.enhanced_config
        
        # Gift feature encoder
        self.gift_feature_encoder = nn.Sequential(
            nn.Linear(config.gift_feature_dim, config.gift_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased from 0.1
            nn.Linear(config.gift_embedding_dim, config.gift_embedding_dim),
            nn.LayerNorm(config.gift_embedding_dim)
        )
        
        # Gift catalog memory
        self.gift_catalog_memory = nn.Parameter(
            torch.randn(config.max_gifts_in_catalog, config.gift_embedding_dim)
        )
        
    def _init_cross_modal_fusion(self):
        """Initialize cross-modal fusion for user-gift-tool interactions"""
        config = self.enhanced_config
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.cross_modal_fusion_dim,
                num_heads=8,
                batch_first=True
            ) for _ in range(config.enhanced_attention_layers)
        ])
        
        # Projection layers
        self.user_projection = nn.Linear(config.user_profile_encoding_dim, config.cross_modal_fusion_dim)
        self.gift_projection = nn.Linear(config.gift_embedding_dim, config.cross_modal_fusion_dim)
        self.tool_projection = nn.Linear(config.tool_context_encoding_dim, config.cross_modal_fusion_dim)
        
        # Final recommendation head
        self.recommendation_head = nn.Sequential(
            nn.Linear(config.cross_modal_fusion_dim, config.cross_modal_fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Much stronger dropout
            nn.Linear(config.cross_modal_fusion_dim // 2, config.action_space_size),
            nn.Softmax(dim=-1)
        )
        
    def _load_and_encode_gift_catalog(self):
        """Load and encode the gift catalog"""
        try:
            with open("data/realistic_gift_catalog.json", "r") as f:
                catalog_data = json.load(f)
            
            gifts = catalog_data.get('gifts', catalog_data)
            self.gift_catalog_data = gifts[:self.enhanced_config.max_gifts_in_catalog]
            
            # Create gift feature vectors
            gift_features = []
            for gift in self.gift_catalog_data:
                features = self._extract_gift_features(gift)
                gift_features.append(features)
            
            if gift_features:
                self.gift_features = torch.tensor(gift_features, dtype=torch.float32)
                print(f"📦 Loaded {len(self.gift_catalog_data)} gifts into model")
            else:
                # Fallback to random features
                self.gift_features = torch.randn(10, self.enhanced_config.gift_feature_dim)
                print("⚠️ Using random gift features as fallback")
                
        except Exception as e:
            print(f"⚠️ Error loading gift catalog: {e}")
            # Fallback to random features
            self.gift_features = torch.randn(10, self.enhanced_config.gift_feature_dim)
            
    def _extract_gift_features(self, gift: dict) -> List[float]:
        """Extract numerical features from gift data"""
        features = []
        
        # Price (normalized)
        features.append(min(1.0, gift.get('price', 50.0) / 200.0))
        
        # Rating (normalized)
        features.append(gift.get('rating', 4.0) / 5.0)
        
        # Category encoding (one-hot)
        category = gift.get('category', 'other')
        category_vector = [1.0 if cat == category else 0.0 for cat in self.gift_categories]
        features.extend(category_vector)
        
        # Age range (normalized)
        age_range = gift.get('age_range', gift.get('age_suitability', [18, 65]))
        features.extend([age_range[0] / 100.0, age_range[1] / 100.0])
        
        # Tags encoding (simplified)
        tags = gift.get('tags', [])
        common_tags = ['practical', 'luxury', 'creative', 'tech', 'natural', 'educational']
        tag_vector = [1.0 if tag in tags else 0.0 for tag in common_tags]
        features.extend(tag_vector)
        
        # Pad or truncate to fixed size
        target_size = self.enhanced_config.gift_feature_dim
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return features
    
    def encode_user_profile(self, user_profile: UserProfile) -> torch.Tensor:
        """Encode user profile using enhanced profiling"""
        device = next(self.parameters()).device
        
        # Encode hobbies
        hobby_embeddings = []
        for hobby in user_profile.hobbies:
            if hobby in self.hobby_categories:
                idx = self.hobby_categories.index(hobby)
                hobby_embeddings.append(self.hobby_embeddings(torch.tensor(idx, device=device)))
        
        if hobby_embeddings:
            hobby_encoding = torch.stack(hobby_embeddings).mean(dim=0)
        else:
            hobby_encoding = torch.zeros(self.enhanced_config.hobby_embedding_dim, device=device)
        
        # Encode preferences
        preference_embeddings = []
        for pref in user_profile.personality_traits:
            if pref in self.preference_categories:
                idx = self.preference_categories.index(pref)
                preference_embeddings.append(self.preference_embeddings(torch.tensor(idx, device=device)))
        
        if preference_embeddings:
            preference_encoding = torch.stack(preference_embeddings).mean(dim=0)
        else:
            preference_encoding = torch.zeros(self.enhanced_config.preference_embedding_dim, device=device)
        
        # Encode occasion
        if user_profile.occasion in self.occasion_categories:
            occasion_idx = self.occasion_categories.index(user_profile.occasion)
            occasion_encoding = self.occasion_embeddings(torch.tensor(occasion_idx, device=device))
        else:
            occasion_encoding = torch.zeros(self.enhanced_config.occasion_embedding_dim, device=device)
        
        # Encode age and budget
        age_encoding = self.age_encoder(torch.tensor([user_profile.age / 100.0], device=device))
        budget_encoding = self.budget_encoder(torch.tensor([user_profile.budget / 1000.0], device=device))
        
        # Concatenate all encodings
        profile_features = torch.cat([
            hobby_encoding, preference_encoding, occasion_encoding, 
            age_encoding.squeeze(), budget_encoding.squeeze()
        ])
        
        # Final encoding
        user_encoding = self.user_profile_encoder(profile_features.unsqueeze(0))
        return user_encoding
    
    def enhanced_category_matching(self, user_encoding: torch.Tensor) -> torch.Tensor:
        """Perform enhanced category matching"""
        device = user_encoding.device
        batch_size = user_encoding.size(0)
        
        # Get all category embeddings
        category_indices = torch.arange(len(self.gift_categories), device=device)
        category_embeds = self.category_embeddings(category_indices)  # [num_categories, embed_dim]
        
        # Expand for batch processing
        category_embeds = category_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        user_encoding_expanded = user_encoding.unsqueeze(1).expand(-1, len(self.gift_categories), -1)
        
        # Semantic matching
        combined_features = torch.cat([user_encoding_expanded, category_embeds], dim=-1)
        
        # Project to correct dimension first
        combined_features = self.semantic_input_proj(combined_features)
        
        # Apply semantic matching layers
        for layer in self.semantic_matcher:
            # Each layer expects category_embedding_dim input
            combined_features = layer(combined_features)
        
        # Apply attention
        attended_features, _ = self.category_attention(
            combined_features, combined_features, combined_features
        )
        
        # Score categories
        category_scores = self.category_scorer(attended_features).squeeze(-1)
        
        return category_scores
    
    def enhanced_tool_selection(self, user_encoding: torch.Tensor, 
                              category_scores: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Perform enhanced context-aware tool selection"""
        device = user_encoding.device
        
        # Encode context for tool selection
        tool_context = self.tool_context_encoder(user_encoding)
        
        # Apply context-aware attention
        tool_attended, _ = self.context_aware_tool_selector(
            tool_context, tool_context, tool_context
        )
        
        # Get tool diversity scores
        tool_scores = self.tool_diversity_head(tool_attended.squeeze(1))
        
        # Select top tools with diversity
        num_tools_to_select = min(self.enhanced_config.max_tool_calls_per_step, tool_scores.size(-1))
        top_tool_indices = torch.topk(tool_scores, num_tools_to_select, dim=-1).indices
        
        # Convert to tool names
        tool_names = list(self.tool_registry.list_tools())
        selected_tools = []
        for batch_idx in range(top_tool_indices.size(0)):
            batch_tools = []
            for tool_idx in top_tool_indices[batch_idx]:
                if tool_idx.item() < len(tool_names):
                    batch_tools.append(tool_names[tool_idx.item()])
            selected_tools.append(batch_tools)
        
        return selected_tools, tool_scores
    
    def enhanced_reward_prediction(self, user_encoding: torch.Tensor, 
                                 gift_encodings: torch.Tensor,
                                 user_profile: UserProfile) -> torch.Tensor:
        """Predict enhanced reward with multiple components"""
        device = user_encoding.device
        batch_size = user_encoding.size(0)
        num_gifts = gift_encodings.size(1)
        
        # Calculate individual reward components
        component_rewards = []
        
        # Category match component (simplified for now)
        category_rewards = self.reward_components['category_match'](
            gift_encodings.view(-1, gift_encodings.size(-1))
        ).view(batch_size, num_gifts)
        component_rewards.append(category_rewards)
        
        # Budget compatibility (simplified)
        budget_rewards = self.reward_components['budget_compatibility'](
            torch.cat([
                user_encoding[:, :self.enhanced_config.age_encoding_dim].unsqueeze(1).expand(-1, num_gifts, -1),
                torch.ones(batch_size, num_gifts, 1, device=device) * 0.5  # placeholder price
            ], dim=-1).view(-1, self.enhanced_config.age_encoding_dim + 1)
        ).view(batch_size, num_gifts)
        component_rewards.append(budget_rewards)
        
        # Add other components (simplified for now)
        for component_name in ['hobby_alignment', 'occasion_appropriateness', 'age_appropriateness']:
            # Placeholder implementation
            component_reward = torch.rand(batch_size, num_gifts, device=device) * 0.5 + 0.25
            component_rewards.append(component_reward)
        
        # Quality and diversity (simplified)
        quality_rewards = torch.rand(batch_size, num_gifts, device=device) * 0.3 + 0.7
        diversity_rewards = torch.rand(batch_size, num_gifts, device=device) * 0.2 + 0.1
        component_rewards.extend([quality_rewards, diversity_rewards])
        
        # Stack and fuse components
        stacked_components = torch.stack(component_rewards, dim=-1)  # [batch, gifts, components]
        
        # Apply fusion network
        fused_rewards = self.reward_fusion(stacked_components.view(-1, len(component_rewards)))
        final_rewards = fused_rewards.view(batch_size, num_gifts).squeeze(-1)
        
        return final_rewards
    
    def forward_with_enhancements(self, carry, env_state: EnvironmentState, 
                                available_gifts: List[GiftItem]) -> Tuple[Any, Dict[str, torch.Tensor], List[str]]:
        """Forward pass with all enhancements integrated, including tool feedback and parameters"""
        device = next(self.parameters()).device
        
        # Encode user profile
        user_encoding = self.encode_user_profile(env_state.user_profile)
        
        # Check if tool feedback is available in carry state
        tool_feedback = None
        if isinstance(carry, dict) and 'tool_feedback' in carry:
            tool_feedback = carry['tool_feedback']
            # Integrate tool feedback into user encoding
            if tool_feedback is not None and tool_feedback.numel() > 0:
                # Ensure dimensions match
                if tool_feedback.dim() == 3:
                    tool_feedback = tool_feedback.squeeze(0)  # Remove batch dim if present
                if tool_feedback.dim() == 2:
                    tool_feedback = tool_feedback.squeeze(0)  # Get single vector
                
                # Project tool feedback to user encoding dimension if needed
                if tool_feedback.size(-1) != user_encoding.size(-1):
                    feedback_proj = nn.Linear(tool_feedback.size(-1), user_encoding.size(-1)).to(device)
                    tool_feedback = feedback_proj(tool_feedback.unsqueeze(0))
                else:
                    tool_feedback = tool_feedback.unsqueeze(0)
                
                # Fuse tool feedback with user encoding (additive fusion)
                user_encoding = user_encoding + 0.3 * tool_feedback  # Weighted addition
        
        # Enhanced category matching
        category_scores = self.enhanced_category_matching(user_encoding)
        
        # Enhanced tool selection
        selected_tools, tool_scores = self.enhanced_tool_selection(user_encoding, category_scores)
        
        # Generate tool parameters for selected tools
        tool_params = {}
        if selected_tools and len(selected_tools) > 0:
            tools_list = selected_tools[0] if selected_tools else []
            for tool_name in tools_list:
                # Concatenate user encoding and category scores for context
                tool_context = torch.cat([
                    user_encoding.squeeze(0) if user_encoding.dim() > 1 else user_encoding,
                    category_scores.squeeze(0) if category_scores.dim() > 1 else category_scores
                ], dim=-1)
                
                # Ensure correct dimension
                expected_dim = self.enhanced_config.tool_context_encoding_dim + self.enhanced_config.user_profile_encoding_dim
                if tool_context.size(-1) != expected_dim:
                    # Project to expected dimension
                    context_proj = nn.Linear(tool_context.size(-1), expected_dim).to(device)
                    tool_context = context_proj(tool_context)
                
                # Generate parameters
                param_encoding = self.enhanced_tool_param_generator(tool_context)
                
                # Decode parameters based on tool type
                if tool_name == 'price_comparison':
                    # Generate budget parameter (scale to reasonable range)
                    budget_param = torch.sigmoid(param_encoding[0]) * 500.0  # 0-500 range
                    tool_params[tool_name] = {'budget': budget_param.item()}
                
                elif tool_name == 'review_analysis':
                    # Generate minimum rating threshold
                    min_rating = torch.sigmoid(param_encoding[1]) * 5.0  # 0-5 range
                    tool_params[tool_name] = {'min_rating': min_rating.item()}
                
                elif tool_name == 'inventory_check':
                    # Generate availability threshold
                    availability_threshold = torch.sigmoid(param_encoding[2])
                    tool_params[tool_name] = {'threshold': availability_threshold.item()}
                
                elif tool_name == 'trend_analyzer':
                    # Generate trend window parameter
                    trend_window = torch.sigmoid(param_encoding[3]) * 30.0  # 0-30 days
                    tool_params[tool_name] = {'window_days': int(trend_window.item())}
        
        # Encode available gifts
        gift_encodings = self.gift_feature_encoder(self.gift_features[:len(available_gifts)].to(device))
        gift_encodings = gift_encodings.unsqueeze(0)  # Add batch dimension
        
        # Enhanced reward prediction
        predicted_rewards = self.enhanced_reward_prediction(user_encoding, gift_encodings, env_state.user_profile)
        
        # Cross-modal fusion
        user_proj = self.user_projection(user_encoding)
        gift_proj = self.gift_projection(gift_encodings)
        
        # Apply cross-modal attention layers
        fused_representation = user_proj
        for attention_layer in self.cross_modal_layers:
            # Ensure all tensors have batch dimension and sequence dimension
            if fused_representation.dim() == 2:
                fused_representation = fused_representation.unsqueeze(1)
            if gift_proj.dim() == 2:
                gift_proj = gift_proj.unsqueeze(1)
            
            fused_representation, _ = attention_layer(fused_representation, gift_proj, gift_proj)
        
        # Generate final recommendations
        recommendation_probs = self.recommendation_head(fused_representation.squeeze(1))
        
        # Prepare outputs with tool parameters
        rl_output = {
            "action_probs": recommendation_probs,
            "value_estimates": predicted_rewards.mean(dim=-1, keepdim=True),
            "category_scores": category_scores,
            "tool_scores": tool_scores,
            "predicted_rewards": predicted_rewards,
            "tool_params": tool_params  # NEW: Tool parameters for execution
        }
        
        return carry, rl_output, selected_tools[0] if selected_tools else []


def create_integrated_enhanced_config():
    """Create configuration for integrated enhanced model"""
    return {
        # Base TRM parameters
        "batch_size": 16,
        "seq_len": 50,
        "vocab_size": 1000,
        "num_puzzle_identifiers": 1,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "hidden_size": 512,  # Increased for enhanced capabilities
        "L_layers": 3,
        "H_layers": 3,
        "H_cycles": 2,
        "L_cycles": 3,
        "num_heads": 8,
        "expansion": 2.0,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        
        # ACT parameters
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,
        "no_ACT_continue": True,
        
        # RL parameters
        "action_space_size": 50,  # Max gifts to consider
        "max_recommendations": 3,
        "value_head_hidden": 256,
        "policy_head_hidden": 256,
        "reward_prediction": True,
        "reward_head_hidden": 128,
        
        # Enhanced configuration
        "user_profile_encoding_dim": 256,
        "hobby_embedding_dim": 64,
        "preference_embedding_dim": 32,
        "occasion_embedding_dim": 32,
        "age_encoding_dim": 16,
        "category_embedding_dim": 128,
        "category_attention_heads": 8,
        "semantic_matching_layers": 2,
        "tool_context_encoding_dim": 128,
        "tool_selection_heads": 4,
        "max_tool_calls_per_step": 2,
        "tool_diversity_weight": 0.3,
        "reward_components": 7,
        "reward_fusion_layers": 3,
        "reward_prediction_dim": 64,
        "gift_embedding_dim": 256,
        "gift_feature_dim": 128,
        "max_gifts_in_catalog": 50,
        "category_loss_weight": 0.35,
        "tool_diversity_loss_weight": 0.15,
        "semantic_matching_loss_weight": 0.20,
        "enhanced_attention_layers": 4,
        "cross_modal_fusion_dim": 512,
        
        # Training parameters
        "forward_dtype": "float32"
    }


if __name__ == "__main__":
    # Test the integrated enhanced model
    config = create_integrated_enhanced_config()
    model = IntegratedEnhancedTRM(config)
    
    print(f"🎉 Integrated Enhanced TRM created successfully!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🧠 Enhanced components: User Profiling, Category Matching, Tool Selection, Reward Prediction")