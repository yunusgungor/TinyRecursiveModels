"""Model inference service for TRM model"""

import torch
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
from functools import lru_cache

from app.models.schemas import UserProfile, GiftItem, GiftRecommendation
from app.core.config import settings
from app.core.exceptions import ModelInferenceError, ModelLoadError


logger = logging.getLogger(__name__)


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
        timeout: Optional[float] = None
    ) -> Tuple[List[GiftRecommendation], Dict[str, Any]]:
        """
        Generate gift recommendations using TRM model
        
        Args:
            user_profile: User profile data
            available_gifts: List of available gifts
            max_recommendations: Maximum number of recommendations
            timeout: Timeout in seconds (default: from settings)
            
        Returns:
            Tuple of (recommendations, tool_results)
            
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
                self._run_inference(user_profile, available_gifts, max_recommendations),
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
        max_recommendations: int
    ) -> Tuple[List[GiftRecommendation], Dict[str, Any]]:
        """
        Internal method to run model inference
        
        Args:
            user_profile: User profile data
            available_gifts: List of available gifts
            max_recommendations: Maximum number of recommendations
            
        Returns:
            Tuple of (recommendations, tool_results)
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
            model_gifts
        )
        
        # Decode output to recommendations
        recommendations = self._decode_model_output(model_output, available_gifts)
        
        # Limit to max recommendations
        recommendations = recommendations[:max_recommendations]
        
        # Extract tool results
        tool_results = model_output.get("tool_results", {})
        
        return recommendations, tool_results
    
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
        available_gifts
    ) -> Dict[str, Any]:
        """
        Perform forward pass through model with optimizations
        
        Args:
            env_state: Environment state
            available_gifts: List of available gifts
            
        Returns:
            Dict containing model outputs
        """
        with torch.no_grad():
            # Use torch.inference_mode for better performance
            with torch.inference_mode():
                # Initialize empty carry (no tool feedback yet)
                carry = {}
                
                # Forward pass with enhancements - this returns RL output directly
                new_carry, rl_output, selected_tools = self.model.forward_with_enhancements(
                    carry,
                    env_state,
                    available_gifts,
                    execute_tools=True
                )
                
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
                    "confidence": confidence
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
