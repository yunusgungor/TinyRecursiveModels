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
    tool_diversity_weight: float = 0.25  # BALANCED: Reduced from 0.4/0.55 to allow natural distribution
    
    # Tool-specific parameters (from tool_enhanced_trm)
    tool_call_threshold: float = 0.15  # BALANCED: Low enough to encourage usage, high enough to filter noise
    tool_result_encoding_dim: int = 128
    tool_selection_method: str = "confidence"  # "confidence", "random", "round_robin"
    tool_fusion_method: str = "concatenate"  # "concatenate", "attention", "gating"
    tool_attention_heads: int = 4
    tool_usage_reward_weight: float = 0.1
    tool_efficiency_penalty: float = 0.05
    
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
        
        # Load dynamic categories from dataset FIRST
        self._load_dynamic_categories()
        
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
        
    def _load_dynamic_categories(self):
        """Load categories dynamically from dataset files"""
        try:
            # Load gift catalog
            with open("data/gift_catalog.json", "r") as f:
                catalog_data = json.load(f)
            
            # Load user scenarios
            with open("data/user_scenarios.json", "r") as f:
                scenarios_data = json.load(f)
            
            # Extract unique gift categories
            gifts = catalog_data.get('gifts', catalog_data)
            self.gift_categories = sorted(list(set(
                gift.get('category', 'other') for gift in gifts
            )))
            
            # Extract unique tags as hobby categories
            all_tags = set()
            for gift in gifts:
                all_tags.update(gift.get('tags', []))
            self.hobby_categories = sorted(list(all_tags))
            
            # Extract unique occasions
            all_occasions = set()
            for gift in gifts:
                occasions = gift.get('occasions', [])
                if isinstance(occasions, list):
                    all_occasions.update(occasions)
            
            # Also get occasions from scenarios
            scenarios = scenarios_data.get('scenarios', [])
            for scenario in scenarios:
                occasion = scenario.get('profile', {}).get('occasion', '')
                if occasion:
                    all_occasions.add(occasion)
            
            self.occasion_categories = sorted(list(all_occasions))
            
            # Extract preferences from scenarios (hobbies field contains preferences/tags)
            all_preferences = set()
            for scenario in scenarios:
                profile = scenario.get('profile', {})
                all_preferences.update(profile.get('hobbies', []))
                all_preferences.update(profile.get('preferences', []))
            
            # Combine with tags for comprehensive preference list
            all_preferences.update(all_tags)
            self.preference_categories = sorted(list(all_preferences))
            
            print(f"📊 Loaded dynamic categories:")
            print(f"  - Gift categories: {len(self.gift_categories)}")
            print(f"  - Hobby categories: {len(self.hobby_categories)}")
            print(f"  - Occasion categories: {len(self.occasion_categories)}")
            print(f"  - Preference categories: {len(self.preference_categories)}")
            
        except Exception as e:
            print(f"⚠️ Error loading dynamic categories: {e}")
            print(f"⚠️ Falling back to default categories")
            # Fallback to minimal defaults
            self.gift_categories = ['technology', 'home', 'beauty', 'health', 'kitchen']
            self.hobby_categories = ['practical', 'modern', 'digital', 'stylish']
            self.occasion_categories = ['birthday', 'christmas', 'anniversary']
            self.preference_categories = ['practical', 'modern', 'stylish', 'digital']
    
    def _setup_tools(self):
        """Setup and register all available tools"""
        gift_tools = GiftRecommendationTools()
        for tool in gift_tools.get_all_tools():
            self.tool_registry.register_tool(tool)
        print(f"Registered {len(self.tool_registry.list_tools())} tools")
    
    def _init_enhanced_user_profiler(self):
        """Initialize enhanced user profiling components"""
        config = self.enhanced_config
        
        # Hobby embeddings with semantic understanding (categories loaded dynamically)
        self.hobby_embeddings = nn.Embedding(len(self.hobby_categories), config.hobby_embedding_dim)
        
        # Preference embeddings (categories loaded dynamically)
        self.preference_embeddings = nn.Embedding(len(self.preference_categories), config.preference_embedding_dim)
        
        # Occasion embeddings (categories loaded dynamically)
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
        
        # Gift categories (loaded dynamically)
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
        
        # Tool parameter generator context projection
        # Pre-create projection layer to ensure gradient flow
        self.tool_param_context_proj = nn.Linear(
            config.user_profile_encoding_dim + len(self.gift_categories),
            config.tool_context_encoding_dim + config.user_profile_encoding_dim
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
        
        # Tool usage predictor - predicts if tool usage will be beneficial
        self.tool_usage_predictor = nn.Sequential(
            nn.Linear(config.user_profile_encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Tool result encoder - encodes tool results back to hidden space
        self.tool_result_encoder_net = nn.Sequential(
            nn.Linear(config.tool_context_encoding_dim, config.user_profile_encoding_dim),
            nn.ReLU(),
            nn.Linear(config.user_profile_encoding_dim, config.user_profile_encoding_dim)
        )
        
        # Tool fusion projection layers
        self.tool_projection_layer = None  # Will be created dynamically
        # Fusion projection layer for checkpoint compatibility (512 -> 256)
        # Maps concatenated user+gift projections to user encoding dimension
        self.fusion_projection_layer = nn.Linear(512, 256)
        
        # Tool call history
        self.tool_call_history = []
        
    def _load_and_encode_gift_catalog(self):
        """Load and encode the gift catalog"""
        try:
            with open("data/gift_catalog.json", "r") as f:
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
        
        # Price (normalized) - safely parse
        try:
            price = float(gift.get('price', 50.0))
        except (ValueError, TypeError):
            price = 50.0
        features.append(min(1.0, price / 200.0))
        
        # Rating (normalized)
        features.append(gift.get('rating', 4.0) / 5.0)
        
        # Category encoding (one-hot)
        category = gift.get('category', 'other')
        category_vector = [1.0 if cat == category else 0.0 for cat in self.gift_categories]
        features.extend(category_vector)
        
        # Age range (normalized) - safely parse
        age_range = gift.get('age_range', gift.get('age_suitability', [18, 65]))
        if not isinstance(age_range, (list, tuple)) or len(age_range) != 2:
            age_range = [18, 65]  # Default if invalid
        try:
            age_min = float(age_range[0])
            age_max = float(age_range[1])
        except (ValueError, TypeError, IndexError):
            age_min, age_max = 18.0, 65.0
        features.extend([age_min / 100.0, age_max / 100.0])
        
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
        """Perform enhanced context-aware tool selection with improved selection strategy"""
        device = user_encoding.device
        
        # Encode context for tool selection
        tool_context = self.tool_context_encoder(user_encoding)
        
        # Apply context-aware attention
        tool_attended, _ = self.context_aware_tool_selector(
            tool_context, tool_context, tool_context
        )
        
        # Get tool diversity scores (softmax outputs sum to 1)
        tool_scores = self.tool_diversity_head(tool_attended.squeeze(1))
        
        # BALANCED STRATEGY: Smart Thresholding with Fallback
        # Addresses "Unnecessary Usage" vs "Tool Silence" trade-off
        tool_names = list(self.tool_registry.list_tools())
        selected_tools = []
        
        # Threshold for confident selection
        threshold = self.enhanced_config.tool_call_threshold  # 0.15
        
        for batch_idx in range(tool_scores.size(0)):
            batch_tools = []
            batch_scores = tool_scores[batch_idx]
            
            # 1. Primary Selection: Select all tools above threshold
            # Sort by score descending to prioritize best matches
            sorted_indices = torch.argsort(batch_scores, descending=True)
            
            for tool_idx in sorted_indices:
                score = batch_scores[tool_idx]
                if score > threshold and tool_idx < len(tool_names):
                    batch_tools.append(tool_names[tool_idx])
                    # Cap at max tools per step
                    if len(batch_tools) >= self.enhanced_config.max_tool_calls_per_step:
                        break
            
            # 2. Fallback Selection (Anti-Silence):
            # If no tools selected, pick the top one IF it has at least minimal relevance
            # This prevents "silence" but avoids forcing completely irrelevant tools
            if not batch_tools and len(tool_names) > 0:
                top_idx = sorted_indices[0]
                top_score = batch_scores[top_idx]
                
                # Minimal relevance check (very low bar, just to filter noise)
                if top_score > 0.05:
                    batch_tools.append(tool_names[top_idx])
            
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
                
                # Use registered projection layer for gradient flow
                tool_context = self.tool_param_context_proj(tool_context)
                
                # Generate parameters
                param_encoding = self.enhanced_tool_param_generator(tool_context)
                
                # Decode parameters based on tool type
                if tool_name == 'price_comparison':
                    # Generate budget parameter (scale to reasonable range)
                    budget_param = torch.sigmoid(param_encoding[0]) * 500.0  # 0-500 range
                    tool_params[tool_name] = {'budget': budget_param.item()}
                
                elif tool_name == 'review_analysis':
                    # ReviewAnalysisTool only accepts: product_id, max_reviews, language, gifts
                    # No min_rating parameter - gifts will be filtered by the tool itself
                    max_reviews = int(torch.sigmoid(param_encoding[1]) * 200.0)  # 0-200 range
                    tool_params[tool_name] = {'max_reviews': max(10, max_reviews)}
                
                elif tool_name == 'inventory_check':
                    # InventoryCheckTool accepts: gifts parameter only
                    tool_params[tool_name] = {}  # No additional parameters needed
                
                elif tool_name == 'trend_analyzer':
                    # TrendAnalysisTool accepts: category, time_period, region, gifts, user_age
                    # Generate time period (7d, 30d, 90d, 1y)
                    time_periods = ['7d', '30d', '90d', '1y']
                    period_idx = int(torch.sigmoid(param_encoding[3]) * len(time_periods))
                    period_idx = min(period_idx, len(time_periods) - 1)
                    tool_params[tool_name] = {'time_period': time_periods[period_idx]}
        
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
            return torch.zeros(self.enhanced_config.tool_context_encoding_dim, device=device)
        
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
        target_size = self.enhanced_config.tool_context_encoding_dim
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        # Convert to tensor and encode
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        feature_tensor = feature_tensor.unsqueeze(0)  # Add batch dimension
        encoded_result = self.tool_result_encoder_net(feature_tensor)
        encoded_result = encoded_result.squeeze(0)  # Remove batch dimension
        
        return encoded_result
    
    def fuse_tool_results(self, hidden_state: torch.Tensor, 
                         tool_encodings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse tool results with hidden state (robust dimension handling)
        
        Args:
            hidden_state: Current hidden state
            tool_encodings: List of encoded tool results
            
        Returns:
            Fused hidden state
        """
        if not tool_encodings:
            return hidden_state
        
        device = hidden_state.device
        
        # Stack tool encodings
        tool_stack = torch.stack(tool_encodings, dim=0).unsqueeze(0)  # [1, num_tools, encoding_dim]
        
        # Get tool summary - average across tools
        tool_summary = tool_stack.mean(dim=1)  # [1, encoding_dim]
        
        # Store original hidden state shape
        original_shape = hidden_state.shape
        hidden_size = hidden_state.size(-1)
        
        # Ensure both tensors are 2D for processing
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)  # [1, hidden_size]
        if tool_summary.dim() == 1:
            tool_summary = tool_summary.unsqueeze(0)  # [1, encoding_dim]
        
        # Ensure tool_summary matches hidden_state's last dimension
        if tool_summary.size(-1) != hidden_size:
            if self.tool_projection_layer is None:
                self.tool_projection_layer = nn.Linear(tool_summary.size(-1), hidden_size).to(device)
            tool_summary = self.tool_projection_layer(tool_summary)
        
        # Ensure batch dimensions match
        if tool_summary.size(0) != hidden_state.size(0):
            tool_summary = tool_summary.expand(hidden_state.size(0), -1)
        
        # Concatenate along feature dimension
        combined = torch.cat([hidden_state, tool_summary], dim=-1)  # [batch, hidden_size * 2]
        
        # Project back to original hidden size
        if self.fusion_projection_layer is None:
            self.fusion_projection_layer = nn.Linear(combined.size(-1), hidden_size).to(device)
        result = self.fusion_projection_layer(combined)
        
        # Restore original shape
        if len(original_shape) == 1:
            result = result.squeeze(0)
        
        return result
    
    def _extract_product_name_from_context(self, env_state: EnvironmentState) -> str:
        """Extract product name from environment context"""
        if env_state.current_recommendations:
            return env_state.current_recommendations[0].name
        
        # Fallback based on hobbies
        hobby_products = {
            "gardening": "garden tools",
            "cooking": "kitchen appliances", 
            "reading": "books",
            "sports": "fitness equipment",
            "technology": "gadgets",
            "fitness": "fitness equipment",
            "wellness": "wellness products",
            "art": "art supplies",
            "music": "musical instruments"
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
            "music": "music",
            "fitness": "fitness",
            "wellness": "wellness",
            "outdoor": "outdoor",
            "gaming": "gaming"
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
    
    def forward_with_tools(self, carry, env_state: EnvironmentState, 
                          available_gifts: List[GiftItem],
                          max_tool_calls: Optional[int] = None) -> Tuple[Any, Dict[str, torch.Tensor], List[ToolCall]]:
        """
        Forward pass with iterative tool usage
        
        Args:
            carry: TRM carry state
            env_state: Current environment state
            available_gifts: Available gift items
            max_tool_calls: Maximum number of tool calls
            
        Returns:
            Tuple of (new_carry, rl_output, tool_calls)
        """
        device = next(self.parameters()).device
        tool_calls = []
        tool_encodings = []
        used_tool_names = set()
        
        max_calls = max_tool_calls or self.enhanced_config.max_tool_calls_per_step
        
        # Initial encoding
        user_encoding = self.encode_user_profile(env_state.user_profile)
        
        # Tool usage loop
        for step in range(max_calls):
            # Decide if we should use a tool
            tool_usage_prob = self.tool_usage_predictor(user_encoding).item()
            
            if tool_usage_prob < 0.1:  # Threshold lowered to encourage exploration
                break
            
            # Enhanced tool selection
            category_scores = self.enhanced_category_matching(user_encoding)
            selected_tools, tool_scores = self.enhanced_tool_selection(user_encoding, category_scores)
            
            # EXPLORATION: Epsilon-greedy strategy during training
            # This forces the model to try new tools (like budget_optimizer) that might have low initial scores
            is_exploration = False
            if self.training and torch.rand(1).item() < 0.15:  # 15% exploration rate
                all_tools = list(self.tool_registry.list_tools())
                # Filter out already used tools
                available_candidates = [t for t in all_tools if t not in used_tool_names]
                if available_candidates:
                    # Pick random tool
                    import random
                    random_tool = random.choice(available_candidates)
                    # Override selection
                    selected_tools = [[random_tool]]
                    is_exploration = True
            
            if not selected_tools or not selected_tools[0]:
                break
            
            # Get first NEW tool (avoid duplicates in sequential calls)
            tool_name = None
            if selected_tools and selected_tools[0]:
                for candidate_tool in selected_tools[0]:
                    if candidate_tool not in used_tool_names:
                        tool_name = candidate_tool
                        break
            
            if not tool_name:
                break
            
            used_tool_names.add(tool_name)
            
            # Get tool parameters
            tool_params_dict = {}
            if tool_name:
                # Generate parameters
                tool_context = torch.cat([
                    user_encoding.squeeze(0) if user_encoding.dim() > 1 else user_encoding,
                    category_scores.squeeze(0) if category_scores.dim() > 1 else category_scores
                ], dim=-1)
                
                # Use registered projection layer for gradient flow
                tool_context = self.tool_param_context_proj(tool_context)
                
                param_encoding = self.enhanced_tool_param_generator(tool_context)
                
                # Decode parameters
                if tool_name == 'price_comparison':
                    budget_param = torch.sigmoid(param_encoding[0]) * 500.0
                    tool_params_dict = {'budget': budget_param.item()}
                elif tool_name == 'review_analysis':
                    max_reviews = int(torch.sigmoid(param_encoding[1]) * 200.0)
                    tool_params_dict = {'max_reviews': max(10, max_reviews)}
            
            # Execute tool
            try:
                tool_call = self.execute_tool_call(tool_name, tool_params_dict)
                tool_calls.append(tool_call)
                
                if tool_call.success and tool_call.result:
                    # Encode tool result
                    tool_encoding = self.encode_tool_result(tool_call.result)
                    tool_encodings.append(tool_encoding)
                    
                    # Update user encoding with tool result
                    user_encoding = self.fuse_tool_results(user_encoding, [tool_encoding])
            except Exception as e:
                print(f"⚠️ Tool execution failed: {e}")
                break
        
        # Final forward pass with tool-enhanced encoding
        carry, rl_output, selected_tools_final = self.forward_with_enhancements(
            carry, env_state, available_gifts
        )
        
        return carry, rl_output, tool_calls
    
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
                tool_reward -= 0.1
                continue
            
            # Reward based on tool type and user feedback
            if tool_call.tool_name == "price_comparison":
                if user_feedback.get("price_sensitive", False):
                    tool_reward += 0.2
            elif tool_call.tool_name == "review_analysis":
                if user_feedback.get("quality_focused", False):
                    tool_reward += 0.2
            elif tool_call.tool_name == "trend_analyzer":
                if user_feedback.get("trendy", False):
                    tool_reward += 0.15
            elif tool_call.tool_name == "inventory_check":
                if user_feedback.get("availability_important", False):
                    tool_reward += 0.15
            
            # Efficiency penalty for too many tool calls
            if len(tool_calls) > 2:
                tool_reward -= 0.05
        
        return tool_reward * 0.1  # Weight
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        if not self.tool_call_history:
            return {"total_calls": 0}
        
        tool_counts = {}
        success_counts = {}
        total_time = 0.0
        
        for call in self.tool_call_history:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
            if call.success:
                success_counts[call.tool_name] = success_counts.get(call.tool_name, 0) + 1
            total_time += call.execution_time
        
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