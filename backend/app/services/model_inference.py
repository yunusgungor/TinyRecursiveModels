"""Model inference service for TRM model"""

import torch
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from functools import lru_cache
import time

from app.models.schemas import UserProfile, GiftItem, GiftRecommendation
from app.core.config import settings
from app.core.exceptions import ModelInferenceError, ModelLoadError


logger = logging.getLogger(__name__)


# Performance monitoring constants (can be overridden by settings)
MAX_REASONING_TRACE_SIZE = settings.REASONING_MAX_TRACE_SIZE
REASONING_TIMEOUT_SECONDS = settings.REASONING_TIMEOUT_SECONDS


class ModelInferenceService:
    """Service for loading and running TRM model inference"""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize model inference service
        
        Args:
            checkpoint_path: Path to model checkpoint file
        """
        self.checkpoint_path = checkpoint_path or settings.MODEL_CHECKPOINT_PATH
        self.device = self._get_device()
        self.model = None
        self.model_loaded = False
        
        # Initialize reasoning service
        from app.services.reasoning_service import get_reasoning_service
        self.reasoning_service = get_reasoning_service()
        
        logger.info(f"ModelInferenceService initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """
        Determine device (GPU/CPU) for model inference
        
        Returns:
            torch.device: Selected device
        """
        # Check if CUDA is available and configured
        if settings.MODEL_DEVICE == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {device_name}")
            except (RuntimeError, AssertionError):
                # CUDA might be available but not properly initialized
                logger.info("Using GPU (CUDA)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        
        return device
    
    def load_model(self) -> None:
        """
        Load model from checkpoint file
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            checkpoint_path = Path(self.checkpoint_path)
            
            if not checkpoint_path.exists():
                raise ModelLoadError(
                    f"Checkpoint file not found: {self.checkpoint_path}"
                )
            
            logger.info(f"Loading model from {self.checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
            
            # Import model class
            from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM
            
            # Extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
            elif 'model_config' in checkpoint:
                config = checkpoint['model_config']
            else:
                # Use default config if not in checkpoint
                config = self._get_default_config()
                logger.warning("Config not found in checkpoint, using default")
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                raise ModelLoadError("Model state dict not found in checkpoint")
            
            # Initialize model with strict=False to handle missing/extra keys
            self.model = IntegratedEnhancedTRM(config, verbose=False)
            
            # Load state dict with strict=False to allow size mismatches
            # This will skip layers that don't match
            result = self.model.load_state_dict(state_dict, strict=False)
            
            # Handle result - it might be a tuple or None
            if result is not None and isinstance(result, tuple) and len(result) == 2:
                missing_keys, unexpected_keys = result
                if missing_keys:
                    logger.warning(f"Missing keys in state dict: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in state dict: {unexpected_keys[:5]}...")  # Show first 5
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
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
            "action_space_size": 100,
            "max_recommendations": 5,
        }
    
    def _encode_user_profile(self, profile: UserProfile) -> torch.Tensor:
        """
        Encode user profile to model input format
        
        Args:
            profile: User profile data
            
        Returns:
            torch.Tensor: Encoded user profile
        """
        if not self.model_loaded:
            raise ModelInferenceError("Model not loaded")
        
        try:
            # Import environment classes
            from models.rl.environment import UserProfile as ModelUserProfile
            
            # Convert API UserProfile to model UserProfile
            model_profile = ModelUserProfile(
                age=profile.age,
                hobbies=profile.hobbies,
                relationship=profile.relationship,
                budget=profile.budget,
                occasion=profile.occasion,
                personality_traits=profile.personality_traits
            )
            
            # Use model's encoding method
            with torch.no_grad():
                user_encoding = self.model.encode_user_profile(model_profile)
            
            return user_encoding
            
        except Exception as e:
            logger.error(f"Failed to encode user profile: {str(e)}")
            raise ModelInferenceError(f"Failed to encode user profile: {str(e)}")
    
    def _decode_model_output(
        self,
        model_output: Dict[str, torch.Tensor],
        available_gifts: List[GiftItem]
    ) -> List[GiftRecommendation]:
        """
        Decode model output to gift recommendations
        
        Args:
            model_output: Model output dictionary
            available_gifts: List of available gifts
            
        Returns:
            List[GiftRecommendation]: Decoded recommendations
        """
        try:
            recommendations = []
            
            # Extract action information
            action_result = model_output.get("action", {})
            selected_gifts = action_result.get("selected_gifts", [])
            confidence_scores = action_result.get("confidence_scores", [])
            
            # Create recommendations
            for rank, (gift, confidence) in enumerate(
                zip(selected_gifts, confidence_scores), start=1
            ):
                # Convert model GiftItem to API GiftItem
                api_gift = GiftItem(
                    id=gift.id,
                    name=gift.name,
                    category=gift.category,
                    price=gift.price,
                    rating=gift.rating,
                    image_url=f"https://cdn.trendyol.com/{gift.id}.jpg",
                    trendyol_url=f"https://www.trendyol.com/product/{gift.id}",
                    description=gift.description,
                    tags=gift.tags,
                    age_suitability=gift.age_suitability,
                    occasion_fit=gift.occasion_fit,
                    in_stock=True
                )
                
                # Create recommendation
                recommendation = GiftRecommendation(
                    gift=api_gift,
                    confidence_score=float(confidence),
                    reasoning=[
                        f"Category match: {gift.category}",
                        f"Price within budget: {gift.price}",
                        f"Rating: {gift.rating}/5.0"
                    ],
                    tool_insights=model_output.get("tool_results", {}),
                    rank=rank
                )
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to decode model output: {str(e)}")
            raise ModelInferenceError(f"Failed to decode model output: {str(e)}")
    
    async def generate_recommendations(
        self,
        user_profile: UserProfile,
        available_gifts: List[GiftItem],
        max_recommendations: int = 5,
        include_reasoning: bool = True,
        reasoning_level: str = "detailed",
        timeout: Optional[float] = None
    ) -> Tuple[List[GiftRecommendation], Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Generate gift recommendations using TRM model
        
        Args:
            user_profile: User profile data
            available_gifts: List of available gifts
            max_recommendations: Maximum number of recommendations
            include_reasoning: Whether to include reasoning trace
            reasoning_level: Level of reasoning detail ("basic", "detailed", "full")
            timeout: Timeout in seconds (default: from settings)
            
        Returns:
            Tuple of (recommendations, tool_results, reasoning_trace)
            
        Raises:
            ModelInferenceError: If inference fails
            asyncio.TimeoutError: If inference times out
        """
        if not self.model_loaded:
            raise ModelInferenceError("Model not loaded. Call load_model() first.")
        
        timeout = timeout or settings.MODEL_INFERENCE_TIMEOUT
        
        try:
            # Run inference with timeout
            result = await asyncio.wait_for(
                self._run_inference(
                    user_profile, 
                    available_gifts, 
                    max_recommendations,
                    include_reasoning,
                    reasoning_level
                ),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Model inference timed out after {timeout} seconds")
            raise ModelInferenceError(
                f"Model inference timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Model inference failed: {str(e)}")
            raise ModelInferenceError(f"Model inference failed: {str(e)}")
    
    async def _run_inference(
        self,
        user_profile: UserProfile,
        available_gifts: List[GiftItem],
        max_recommendations: int,
        include_reasoning: bool = True,
        reasoning_level: str = "detailed"
    ) -> Tuple[List[GiftRecommendation], Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Internal method to run model inference
        
        Args:
            user_profile: User profile data
            available_gifts: List of available gifts
            max_recommendations: Maximum number of recommendations
            include_reasoning: Whether to include reasoning trace
            reasoning_level: Level of reasoning detail
            
        Returns:
            Tuple of (recommendations, tool_results, reasoning_trace)
        """
        # Import required classes
        from models.rl.environment import (
            EnvironmentState,
            GiftItem as ModelGiftItem,
            UserProfile as ModelUserProfile
        )
        
        # Convert API models to internal models
        model_profile = ModelUserProfile(
            age=user_profile.age,
            hobbies=user_profile.hobbies,
            relationship=user_profile.relationship,
            budget=user_profile.budget,
            occasion=user_profile.occasion,
            personality_traits=user_profile.personality_traits
        )
        
        model_gifts = []
        for gift in available_gifts:
            model_gift = ModelGiftItem(
                id=gift.id,
                name=gift.name,
                category=gift.category,
                price=gift.price,
                rating=gift.rating,
                tags=gift.tags,
                description=gift.description,
                age_suitability=gift.age_suitability,
                occasion_fit=gift.occasion_fit
            )
            model_gifts.append(model_gift)
        
        # Create environment state
        env_state = EnvironmentState(
            user_profile=model_profile,
            available_gifts=model_gifts,
            current_recommendations=[],
            interaction_history=[],
            step_count=0
        )
        
        # Run model inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        model_output = await loop.run_in_executor(
            None,
            self._forward_pass,
            env_state,
            model_gifts,
            include_reasoning
        )
        
        # Decode output to recommendations
        recommendations = self._decode_model_output(model_output, available_gifts)
        
        # Limit to max recommendations
        recommendations = recommendations[:max_recommendations]
        
        # Extract tool results
        tool_results = model_output.get("tool_results", {})
        
        # Extract reasoning trace if requested and enabled
        reasoning_trace = None
        if include_reasoning and settings.REASONING_ENABLED:
            reasoning_start_time = time.time()
            
            try:
                # Extract reasoning trace with timeout protection
                reasoning_trace = await asyncio.wait_for(
                    self._extract_reasoning_trace_async(
                        model_output,
                        user_profile,
                        recommendations,
                        tool_results,
                        reasoning_level
                    ),
                    timeout=REASONING_TIMEOUT_SECONDS
                )
                
                # Enhance recommendations with dynamic reasoning
                for rec in recommendations:
                    try:
                        rec.reasoning = self.reasoning_service.generate_gift_reasoning(
                            rec.gift,
                            user_profile,
                            model_output.get("outputs", {}),
                            tool_results
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate reasoning for gift {rec.gift.id}: {str(e)}")
                        # Fall back to basic reasoning
                        rec.reasoning = [f"Recommended based on your profile"]
                
                # Monitor reasoning generation time
                reasoning_time = time.time() - reasoning_start_time
                logger.info(f"Reasoning generation completed in {reasoning_time:.3f}s")
                
                # Check if reasoning took too long (>10% of inference time)
                if reasoning_time > 0.5:  # Arbitrary threshold for warning
                    logger.warning(f"Reasoning generation took {reasoning_time:.3f}s - consider optimization")
                
            except asyncio.TimeoutError:
                logger.error(f"Reasoning generation timed out after {REASONING_TIMEOUT_SECONDS}s")
                # Fall back to basic reasoning
                reasoning_trace = self._create_basic_reasoning_trace(
                    user_profile, recommendations, tool_results
                )
                
                # Still try to add basic reasoning to recommendations
                for rec in recommendations:
                    rec.reasoning = [f"Recommended based on your profile"]
                    
            except Exception as e:
                logger.error(f"Failed to extract reasoning trace: {str(e)}", exc_info=True)
                # Fall back to basic reasoning on error
                reasoning_trace = self._create_basic_reasoning_trace(
                    user_profile, recommendations, tool_results
                )
                
                # Add basic reasoning to recommendations
                for rec in recommendations:
                    rec.reasoning = [f"Recommended based on your profile"]
        
        return recommendations, tool_results, reasoning_trace
    
    async def _extract_reasoning_trace_async(
        self,
        model_output: Dict[str, Any],
        user_profile: UserProfile,
        recommendations: List[GiftRecommendation],
        tool_results: Dict[str, Any],
        reasoning_level: str
    ) -> Dict[str, Any]:
        """
        Extract and format reasoning trace from model output (async version)
        
        Args:
            model_output: Model output dictionary
            user_profile: User profile
            recommendations: List of recommendations
            tool_results: Tool execution results
            reasoning_level: Level of reasoning detail
            
        Returns:
            Formatted reasoning trace dictionary
        """
        # Run synchronous extraction in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._extract_reasoning_trace,
            model_output,
            user_profile,
            recommendations,
            tool_results,
            reasoning_level
        )
    
    def _extract_reasoning_trace(
        self,
        model_output: Dict[str, Any],
        user_profile: UserProfile,
        recommendations: List[GiftRecommendation],
        tool_results: Dict[str, Any],
        reasoning_level: str
    ) -> Dict[str, Any]:
        """
        Extract and format reasoning trace from model output
        
        Args:
            model_output: Model output dictionary
            user_profile: User profile
            recommendations: List of recommendations
            tool_results: Tool execution results
            reasoning_level: Level of reasoning detail
            
        Returns:
            Formatted reasoning trace dictionary
        """
        from app.models.schemas import (
            ReasoningTrace,
            ToolSelectionReasoning,
            CategoryMatchingReasoning,
            AttentionWeights,
            ThinkingStep,
            ConfidenceExplanation
        )
        
        reasoning_trace = {}
        
        try:
            # Get raw reasoning trace from model if available
            raw_trace = model_output.get("reasoning_trace", {})
            
            # 1. Tool Selection Reasoning (detailed and full levels)
            if reasoning_level in ["detailed", "full"]:
                try:
                    tool_selection_raw = raw_trace.get("tool_selection", {})
                    if tool_selection_raw:
                        tool_selection_reasoning = self.reasoning_service.generate_tool_selection_reasoning(
                            tool_selection_raw,
                            user_profile
                        )
                        reasoning_trace["tool_selection"] = [
                            ToolSelectionReasoning(**data)
                            for data in tool_selection_reasoning.values()
                        ]
                    else:
                        reasoning_trace["tool_selection"] = []
                except Exception as e:
                    logger.warning(f"Failed to extract tool selection reasoning: {str(e)}")
                    reasoning_trace["tool_selection"] = []
            
            # 2. Category Matching Reasoning (detailed and full levels)
            if reasoning_level in ["detailed", "full"]:
                try:
                    category_matching_raw = raw_trace.get("category_matching", {})
                    if category_matching_raw:
                        category_reasoning = self.reasoning_service.generate_category_reasoning(
                            category_matching_raw,
                            user_profile
                        )
                        reasoning_trace["category_matching"] = [
                            CategoryMatchingReasoning(**data)
                            for data in category_reasoning.values()
                        ]
                    else:
                        reasoning_trace["category_matching"] = []
                except Exception as e:
                    logger.warning(f"Failed to extract category matching reasoning: {str(e)}")
                    reasoning_trace["category_matching"] = []
            
            # 3. Attention Weights (full level only)
            if reasoning_level == "full":
                try:
                    attention_weights_raw = raw_trace.get("attention_weights")
                    if attention_weights_raw:
                        reasoning_trace["attention_weights"] = AttentionWeights(**attention_weights_raw)
                    else:
                        reasoning_trace["attention_weights"] = None
                except Exception as e:
                    logger.warning(f"Failed to extract attention weights: {str(e)}")
                    reasoning_trace["attention_weights"] = None
            
            # 4. Thinking Steps (full level only)
            if reasoning_level == "full":
                try:
                    thinking_steps_raw = raw_trace.get("thinking_steps", [])
                    if thinking_steps_raw:
                        # Truncate thinking steps if too many
                        max_steps = 20
                        if len(thinking_steps_raw) > max_steps:
                            logger.warning(f"Truncating thinking steps from {len(thinking_steps_raw)} to {max_steps}")
                            thinking_steps_raw = thinking_steps_raw[:max_steps]
                        
                        reasoning_trace["thinking_steps"] = [
                            ThinkingStep(**step)
                            for step in thinking_steps_raw
                        ]
                    else:
                        reasoning_trace["thinking_steps"] = []
                except Exception as e:
                    logger.warning(f"Failed to extract thinking steps: {str(e)}")
                    reasoning_trace["thinking_steps"] = []
            
            # 5. Confidence Explanation (all levels)
            try:
                if recommendations:
                    first_rec = recommendations[0]
                    confidence_explanation = self.reasoning_service.explain_confidence_score(
                        first_rec.confidence_score,
                        first_rec.gift,
                        user_profile,
                        model_output.get("outputs", {})
                    )
                    reasoning_trace["confidence_explanation"] = ConfidenceExplanation(
                        **confidence_explanation
                    )
            except Exception as e:
                logger.warning(f"Failed to extract confidence explanation: {str(e)}")
                reasoning_trace["confidence_explanation"] = None
            
            # Create ReasoningTrace object and truncate if too large
            trace_obj = ReasoningTrace(**reasoning_trace)
            trace_dict = trace_obj.model_dump()
            
            # Check size and truncate if necessary
            trace_dict = self._truncate_reasoning_trace(trace_dict)
            
            return trace_dict
            
        except Exception as e:
            logger.error(f"Error extracting reasoning trace: {str(e)}", exc_info=True)
            # Return minimal reasoning trace on error
            return self._create_minimal_reasoning_trace()
    
    def _truncate_reasoning_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Truncate reasoning trace if it exceeds size limit
        
        Args:
            trace: Reasoning trace dictionary
            
        Returns:
            Truncated trace if necessary
        """
        try:
            import json
            trace_json = json.dumps(trace)
            trace_size = len(trace_json)
            
            if trace_size > MAX_REASONING_TRACE_SIZE:
                logger.warning(f"Reasoning trace size ({trace_size} chars) exceeds limit ({MAX_REASONING_TRACE_SIZE} chars), truncating")
                
                # Truncate thinking steps first (usually the largest component)
                if "thinking_steps" in trace and trace["thinking_steps"]:
                    original_steps = len(trace["thinking_steps"])
                    trace["thinking_steps"] = trace["thinking_steps"][:5]  # Keep only first 5 steps
                    logger.info(f"Truncated thinking steps from {original_steps} to 5")
                
                # Truncate tool selection if still too large
                trace_json = json.dumps(trace)
                if len(trace_json) > MAX_REASONING_TRACE_SIZE and "tool_selection" in trace:
                    original_tools = len(trace["tool_selection"])
                    trace["tool_selection"] = trace["tool_selection"][:3]  # Keep only first 3 tools
                    logger.info(f"Truncated tool selection from {original_tools} to 3")
                
                # Truncate category matching if still too large
                trace_json = json.dumps(trace)
                if len(trace_json) > MAX_REASONING_TRACE_SIZE and "category_matching" in trace:
                    original_categories = len(trace["category_matching"])
                    trace["category_matching"] = trace["category_matching"][:3]  # Keep only first 3 categories
                    logger.info(f"Truncated category matching from {original_categories} to 3")
                
                # Final check
                trace_json = json.dumps(trace)
                final_size = len(trace_json)
                logger.info(f"Final reasoning trace size: {final_size} chars")
                
        except Exception as e:
            logger.error(f"Error truncating reasoning trace: {str(e)}")
        
        return trace
    
    def _create_basic_reasoning_trace(
        self,
        user_profile: UserProfile,
        recommendations: List[GiftRecommendation],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a basic reasoning trace as fallback
        
        Args:
            user_profile: User profile
            recommendations: List of recommendations
            tool_results: Tool execution results
            
        Returns:
            Basic reasoning trace dictionary
        """
        try:
            from app.models.schemas import ConfidenceExplanation
            
            # Create minimal confidence explanation
            confidence_explanation = None
            if recommendations:
                confidence_explanation = ConfidenceExplanation(
                    score=recommendations[0].confidence_score,
                    level="medium",
                    factors={
                        "positive": ["Matched with your profile"],
                        "negative": []
                    }
                )
            
            return {
                "tool_selection": [],
                "category_matching": [],
                "attention_weights": None,
                "thinking_steps": [],
                "confidence_explanation": confidence_explanation.model_dump() if confidence_explanation else None
            }
        except Exception as e:
            logger.error(f"Error creating basic reasoning trace: {str(e)}")
            return self._create_minimal_reasoning_trace()
    
    def _create_minimal_reasoning_trace(self) -> Dict[str, Any]:
        """
        Create a minimal reasoning trace for error cases
        
        Returns:
            Minimal reasoning trace dictionary
        """
        return {
            "tool_selection": [],
            "category_matching": [],
            "attention_weights": None,
            "thinking_steps": [],
            "confidence_explanation": None
        }
    
    @lru_cache(maxsize=128)
    def _cache_profile_encoding(self, profile_hash: str) -> torch.Tensor:
        """
        Cache user profile encodings to avoid recomputation
        
        Args:
            profile_hash: Hash of user profile
            
        Returns:
            Cached encoding tensor
        """
        # This is a placeholder - actual encoding happens in _encode_user_profile
        # The cache is managed by lru_cache decorator
        return None
    
    def _forward_pass(
        self,
        env_state,
        available_gifts,
        capture_reasoning: bool = False
    ) -> Dict[str, Any]:
        """
        Perform forward pass through model with optimizations
        
        Args:
            env_state: Environment state
            available_gifts: List of available gifts
            capture_reasoning: Whether to capture reasoning trace
            
        Returns:
            Dict containing model outputs
        """
        with torch.no_grad():
            # Use torch.inference_mode for better performance
            with torch.inference_mode():
                # Initialize empty carry (no tool feedback yet)
                carry = {}
                
                # Check if model has forward_with_reasoning_trace method
                if capture_reasoning and hasattr(self.model, 'forward_with_reasoning_trace'):
                    # Forward pass with reasoning trace
                    new_carry, rl_output, selected_tools, reasoning_trace = self.model.forward_with_reasoning_trace(
                        carry,
                        env_state,
                        available_gifts,
                        execute_tools=True,
                        capture_reasoning=True,
                        max_thinking_steps=settings.REASONING_MAX_THINKING_STEPS
                    )
                else:
                    # Standard forward pass with enhancements
                    new_carry, rl_output, selected_tools = self.model.forward_with_enhancements(
                        carry,
                        env_state,
                        available_gifts,
                        execute_tools=True
                    )
                    reasoning_trace = {}
                
                # Select action using action probabilities from RL output
                action = self.model.select_action(
                    rl_output["action_probs"],
                    available_gifts,
                    deterministic=True
                )
                
                # Get confidence from value estimates
                confidence = rl_output.get("value_estimates", torch.tensor([0.5])).mean().item()
                
                return {
                    "action": action,
                    "outputs": rl_output,
                    "tool_results": rl_output.get("tool_results", {}),
                    "selected_tools": selected_tools,
                    "confidence": confidence,
                    "reasoning_trace": reasoning_trace
                }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_allocated": torch.cuda.memory_allocated(0),
                "cuda_memory_reserved": torch.cuda.memory_reserved(0),
            })
        
        return info


# Singleton instance
_model_service: Optional[ModelInferenceService] = None


def get_model_service() -> ModelInferenceService:
    """Get or create model inference service singleton"""
    global _model_service
    
    if _model_service is None:
        _model_service = ModelInferenceService()
    
    return _model_service
