"""
Property-based tests for enhanced response schemas

Feature: model-reasoning-enhancement
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from pydantic import ValidationError
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.app.models.schemas import (
    ToolSelectionReasoning,
    CategoryMatchingReasoning,
    AttentionWeights,
    ThinkingStep,
    ConfidenceExplanation,
    ReasoningTrace,
    EnhancedGiftRecommendation,
    EnhancedRecommendationResponse,
    GiftItem,
    UserProfile
)


# Strategies for generating valid schema data

@st.composite
def tool_selection_reasoning_strategy(draw):
    """Generate valid ToolSelectionReasoning data"""
    return {
        "name": draw(st.sampled_from([
            "price_comparison", "review_analysis", "inventory_check", 
            "trend_analyzer", "budget_optimizer"
        ])),
        "selected": draw(st.booleans()),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "reason": draw(st.text(min_size=10, max_size=200)),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "priority": draw(st.integers(min_value=1, max_value=10)),
        "factors": {
            draw(st.text(min_size=3, max_size=20)): draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            for _ in range(draw(st.integers(min_value=0, max_value=5)))
        }
    }


@st.composite
def category_matching_reasoning_strategy(draw):
    """Generate valid CategoryMatchingReasoning data"""
    return {
        "category_name": draw(st.sampled_from([
            "Kitchen", "Technology", "Books", "Sports", "Fashion", "Home"
        ])),
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "reasons": [
            draw(st.text(min_size=10, max_size=100))
            for _ in range(draw(st.integers(min_value=1, max_value=5)))
        ],
        "feature_contributions": {
            draw(st.text(min_size=3, max_size=20)): draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
            for _ in range(draw(st.integers(min_value=1, max_value=5)))
        }
    }


@st.composite
def attention_weights_strategy(draw):
    """Generate valid AttentionWeights data with normalized weights"""
    # Generate user features with weights that sum to 1.0
    num_user_features = draw(st.integers(min_value=2, max_value=5))
    user_feature_names = ["hobbies", "budget", "age", "occasion", "preferences"][:num_user_features]
    
    # Generate random weights and normalize
    user_weights_raw = [draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                        for _ in range(num_user_features)]
    user_total = sum(user_weights_raw)
    user_features = {name: weight / user_total for name, weight in zip(user_feature_names, user_weights_raw)}
    
    # Generate gift features with weights that sum to 1.0
    num_gift_features = draw(st.integers(min_value=2, max_value=4))
    gift_feature_names = ["category", "price", "rating", "availability"][:num_gift_features]
    
    gift_weights_raw = [draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)) 
                        for _ in range(num_gift_features)]
    gift_total = sum(gift_weights_raw)
    gift_features = {name: weight / gift_total for name, weight in zip(gift_feature_names, gift_weights_raw)}
    
    return {
        "user_features": user_features,
        "gift_features": gift_features
    }


@st.composite
def thinking_step_strategy(draw):
    """Generate valid ThinkingStep data"""
    return {
        "step": draw(st.integers(min_value=1, max_value=20)),
        "action": draw(st.text(min_size=10, max_size=100)),
        "result": draw(st.text(min_size=10, max_size=200)),
        "insight": draw(st.text(min_size=10, max_size=200))
    }


@st.composite
def confidence_explanation_strategy(draw):
    """Generate valid ConfidenceExplanation data"""
    return {
        "score": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "level": draw(st.sampled_from(["high", "medium", "low"])),
        "factors": {
            "positive": [
                draw(st.text(min_size=10, max_size=100))
                for _ in range(draw(st.integers(min_value=0, max_value=5)))
            ],
            "negative": [
                draw(st.text(min_size=10, max_size=100))
                for _ in range(draw(st.integers(min_value=0, max_value=5)))
            ]
        }
    }


@st.composite
def reasoning_trace_strategy(draw):
    """Generate valid ReasoningTrace data"""
    return {
        "tool_selection": [
            draw(tool_selection_reasoning_strategy())
            for _ in range(draw(st.integers(min_value=0, max_value=5)))
        ],
        "category_matching": [
            draw(category_matching_reasoning_strategy())
            for _ in range(draw(st.integers(min_value=0, max_value=5)))
        ],
        "attention_weights": draw(attention_weights_strategy()) if draw(st.booleans()) else None,
        "thinking_steps": [
            draw(thinking_step_strategy())
            for _ in range(draw(st.integers(min_value=0, max_value=10)))
        ],
        "confidence_explanation": draw(confidence_explanation_strategy()) if draw(st.booleans()) else None
    }


class TestReasoningJSONSchemaCompliance:
    """
    Property 31: Reasoning JSON schema compliance
    Validates: Requirements 8.1
    """
    
    @given(data=tool_selection_reasoning_strategy())
    @settings(max_examples=100, deadline=None)
    def test_tool_selection_reasoning_schema_compliance(self, data):
        """
        Feature: model-reasoning-enhancement, Property 31: Reasoning JSON schema compliance
        
        For any API response with reasoning enabled, the reasoning information should 
        conform to the defined JSON schema.
        
        Validates: Requirements 8.1
        
        This test validates ToolSelectionReasoning schema compliance.
        """
        # Should successfully create model instance
        try:
            model = ToolSelectionReasoning(**data)
            
            # Verify all fields are present
            assert model.name == data["name"]
            assert model.selected == data["selected"]
            assert model.score == data["score"]
            assert model.reason == data["reason"]
            assert model.confidence == data["confidence"]
            assert model.priority == data["priority"]
            assert model.factors == data["factors"]
            
            # Verify model can be serialized to JSON
            json_data = model.model_dump()
            assert isinstance(json_data, dict)
            
            # Verify JSON can be deserialized back to model
            model_from_json = ToolSelectionReasoning(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    @given(data=category_matching_reasoning_strategy())
    @settings(max_examples=100, deadline=None)
    def test_category_matching_reasoning_schema_compliance(self, data):
        """
        Test CategoryMatchingReasoning schema compliance
        
        Validates: Requirements 8.1
        """
        try:
            model = CategoryMatchingReasoning(**data)
            
            # Verify all fields
            assert model.category_name == data["category_name"]
            assert model.score == data["score"]
            assert model.reasons == data["reasons"]
            assert model.feature_contributions == data["feature_contributions"]
            
            # Verify serialization
            json_data = model.model_dump()
            model_from_json = CategoryMatchingReasoning(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    @given(data=attention_weights_strategy())
    @settings(max_examples=100, deadline=None)
    def test_attention_weights_schema_compliance(self, data):
        """
        Test AttentionWeights schema compliance with normalization validation
        
        Validates: Requirements 8.1, 4.4
        """
        try:
            model = AttentionWeights(**data)
            
            # Verify all fields
            assert model.user_features == data["user_features"]
            assert model.gift_features == data["gift_features"]
            
            # Verify normalization (weights sum to 1.0)
            user_sum = sum(model.user_features.values())
            gift_sum = sum(model.gift_features.values())
            
            assert abs(user_sum - 1.0) < 0.01, \
                f"User features should sum to 1.0, got {user_sum}"
            assert abs(gift_sum - 1.0) < 0.01, \
                f"Gift features should sum to 1.0, got {gift_sum}"
            
            # Verify no negative weights
            for weight in model.user_features.values():
                assert weight >= 0.0, f"User feature weight should be non-negative, got {weight}"
            for weight in model.gift_features.values():
                assert weight >= 0.0, f"Gift feature weight should be non-negative, got {weight}"
            
            # Verify serialization
            json_data = model.model_dump()
            model_from_json = AttentionWeights(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    @given(data=thinking_step_strategy())
    @settings(max_examples=100, deadline=None)
    def test_thinking_step_schema_compliance(self, data):
        """
        Test ThinkingStep schema compliance
        
        Validates: Requirements 8.1, 5.2
        """
        try:
            model = ThinkingStep(**data)
            
            # Verify all fields
            assert model.step == data["step"]
            assert model.action == data["action"]
            assert model.result == data["result"]
            assert model.insight == data["insight"]
            
            # Verify step number is positive
            assert model.step > 0, f"Step number should be positive, got {model.step}"
            
            # Verify serialization
            json_data = model.model_dump()
            model_from_json = ThinkingStep(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    @given(data=confidence_explanation_strategy())
    @settings(max_examples=100, deadline=None)
    def test_confidence_explanation_schema_compliance(self, data):
        """
        Test ConfidenceExplanation schema compliance
        
        Validates: Requirements 8.1, 6.1
        """
        try:
            model = ConfidenceExplanation(**data)
            
            # Verify all fields
            assert model.score == data["score"]
            assert model.level == data["level"]
            assert model.factors == data["factors"]
            
            # Verify level is valid
            assert model.level in ["high", "medium", "low"], \
                f"Level should be high/medium/low, got {model.level}"
            
            # Verify score range
            assert 0.0 <= model.score <= 1.0, \
                f"Score should be in [0, 1], got {model.score}"
            
            # Verify serialization
            json_data = model.model_dump()
            model_from_json = ConfidenceExplanation(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    @given(data=reasoning_trace_strategy())
    @settings(max_examples=100, deadline=None)
    def test_reasoning_trace_schema_compliance(self, data):
        """
        Test ReasoningTrace schema compliance (complete reasoning structure)
        
        Validates: Requirements 8.1
        """
        try:
            model = ReasoningTrace(**data)
            
            # Verify all fields
            assert len(model.tool_selection) == len(data["tool_selection"])
            assert len(model.category_matching) == len(data["category_matching"])
            assert len(model.thinking_steps) == len(data["thinking_steps"])
            
            # Verify optional fields
            if data["attention_weights"] is not None:
                assert model.attention_weights is not None
            if data["confidence_explanation"] is not None:
                assert model.confidence_explanation is not None
            
            # Verify serialization
            json_data = model.model_dump()
            model_from_json = ReasoningTrace(**json_data)
            assert model_from_json == model
            
        except ValidationError as e:
            pytest.fail(f"Valid data failed validation: {e}")
    
    def test_reasoning_trace_with_empty_lists(self):
        """
        Test that ReasoningTrace accepts empty lists for optional fields
        """
        data = {
            "tool_selection": [],
            "category_matching": [],
            "attention_weights": None,
            "thinking_steps": [],
            "confidence_explanation": None
        }
        
        try:
            model = ReasoningTrace(**data)
            assert len(model.tool_selection) == 0
            assert len(model.category_matching) == 0
            assert len(model.thinking_steps) == 0
            assert model.attention_weights is None
            assert model.confidence_explanation is None
        except ValidationError as e:
            pytest.fail(f"Empty lists should be valid: {e}")
    
    def test_attention_weights_validation_rejects_invalid_sum(self):
        """
        Test that AttentionWeights validation rejects weights that don't sum to 1.0
        """
        # Weights that don't sum to 1.0
        invalid_data = {
            "user_features": {
                "hobbies": 0.5,
                "budget": 0.3,
                "age": 0.1
                # Sum = 0.9, not 1.0
            },
            "gift_features": {
                "category": 0.5,
                "price": 0.5
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AttentionWeights(**invalid_data)
        
        assert "sum to 1.0" in str(exc_info.value).lower()
    
    def test_attention_weights_validation_rejects_negative_weights(self):
        """
        Test that AttentionWeights validation rejects negative weights
        """
        invalid_data = {
            "user_features": {
                "hobbies": 1.2,
                "budget": -0.2  # Negative weight
            },
            "gift_features": {
                "category": 0.6,
                "price": 0.4
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AttentionWeights(**invalid_data)
        
        assert "negative" in str(exc_info.value).lower()
    
    def test_confidence_explanation_validation_rejects_invalid_level(self):
        """
        Test that ConfidenceExplanation validation rejects invalid level values
        """
        invalid_data = {
            "score": 0.75,
            "level": "very_high",  # Invalid level
            "factors": {
                "positive": ["Good match"],
                "negative": []
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ConfidenceExplanation(**invalid_data)
        
        # Should mention valid levels
        error_msg = str(exc_info.value).lower()
        assert "high" in error_msg or "medium" in error_msg or "low" in error_msg


class TestToolSelectionJSONSchemaCompliance:
    """
    Property 5: Tool selection JSON schema compliance
    Validates: Requirements 1.5, 8.2
    """
    
    @given(data=tool_selection_reasoning_strategy())
    @settings(max_examples=100, deadline=None)
    def test_tool_selection_json_schema_compliance_property(self, data):
        """
        Feature: model-reasoning-enhancement, Property 5: Tool selection JSON schema compliance
        
        For any API response containing tool selection reasoning, the response should 
        conform to the defined JSON schema with required fields (name, reason, confidence, priority).
        
        Validates: Requirements 1.5, 8.2
        """
        try:
            # Create model instance
            model = ToolSelectionReasoning(**data)
            
            # Verify all required fields are present
            required_fields = ["name", "selected", "score", "reason", "confidence", "priority"]
            for field in required_fields:
                assert hasattr(model, field), f"Required field '{field}' is missing"
                assert getattr(model, field) is not None, f"Required field '{field}' is None"
            
            # Verify field types
            assert isinstance(model.name, str), "name should be string"
            assert isinstance(model.selected, bool), "selected should be boolean"
            assert isinstance(model.score, (int, float)), "score should be numeric"
            assert isinstance(model.reason, str), "reason should be string"
            assert isinstance(model.confidence, (int, float)), "confidence should be numeric"
            assert isinstance(model.priority, int), "priority should be integer"
            assert isinstance(model.factors, dict), "factors should be dictionary"
            
            # Verify value constraints
            assert 0.0 <= model.score <= 1.0, f"score should be in [0, 1], got {model.score}"
            assert 0.0 <= model.confidence <= 1.0, f"confidence should be in [0, 1], got {model.confidence}"
            assert model.priority > 0, f"priority should be positive, got {model.priority}"
            assert len(model.reason) > 0, "reason should not be empty"
            
            # Verify factors are valid
            for factor_name, factor_value in model.factors.items():
                assert isinstance(factor_name, str), f"Factor name should be string, got {type(factor_name)}"
                assert isinstance(factor_value, (int, float)), f"Factor value should be numeric, got {type(factor_value)}"
                assert 0.0 <= factor_value <= 1.0, f"Factor value should be in [0, 1], got {factor_value}"
            
            # Verify JSON serialization produces correct structure
            json_data = model.model_dump()
            
            assert "name" in json_data, "JSON should contain 'name'"
            assert "selected" in json_data, "JSON should contain 'selected'"
            assert "score" in json_data, "JSON should contain 'score'"
            assert "reason" in json_data, "JSON should contain 'reason'"
            assert "confidence" in json_data, "JSON should contain 'confidence'"
            assert "priority" in json_data, "JSON should contain 'priority'"
            assert "factors" in json_data, "JSON should contain 'factors'"
            
            # Verify JSON can be deserialized back
            model_from_json = ToolSelectionReasoning(**json_data)
            assert model_from_json.name == model.name
            assert model_from_json.selected == model.selected
            assert model_from_json.score == model.score
            assert model_from_json.reason == model.reason
            assert model_from_json.confidence == model.confidence
            assert model_from_json.priority == model.priority
            assert model_from_json.factors == model.factors
            
        except ValidationError as e:
            pytest.fail(f"Valid tool selection data failed schema validation: {e}")
    
    def test_tool_selection_schema_rejects_missing_required_fields(self):
        """
        Test that schema validation rejects data missing required fields
        """
        # Missing 'reason' field
        invalid_data = {
            "name": "price_comparison",
            "selected": True,
            "score": 0.85,
            # "reason": missing
            "confidence": 0.85,
            "priority": 1,
            "factors": {}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ToolSelectionReasoning(**invalid_data)
        
        assert "reason" in str(exc_info.value).lower()
    
    def test_tool_selection_schema_rejects_invalid_score_range(self):
        """
        Test that schema validation rejects scores outside [0, 1] range
        """
        invalid_data = {
            "name": "price_comparison",
            "selected": True,
            "score": 1.5,  # Invalid: > 1.0
            "reason": "Test reason",
            "confidence": 0.85,
            "priority": 1,
            "factors": {}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ToolSelectionReasoning(**invalid_data)
        
        # Should mention the constraint
        assert "1" in str(exc_info.value) or "less" in str(exc_info.value).lower()
    
    def test_tool_selection_schema_rejects_invalid_priority(self):
        """
        Test that schema validation rejects non-positive priority values
        """
        invalid_data = {
            "name": "price_comparison",
            "selected": True,
            "score": 0.85,
            "reason": "Test reason",
            "confidence": 0.85,
            "priority": 0,  # Invalid: should be >= 1
            "factors": {}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ToolSelectionReasoning(**invalid_data)
        
        # Should mention the constraint
        assert "1" in str(exc_info.value) or "greater" in str(exc_info.value).lower()
    
    @given(
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        priority=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, deadline=None)
    def test_tool_selection_schema_with_varying_values(self, score, confidence, priority):
        """
        Test tool selection schema with various valid value combinations
        """
        data = {
            "name": "price_comparison",
            "selected": True,
            "score": score,
            "reason": "Test reason for property-based testing",
            "confidence": confidence,
            "priority": priority,
            "factors": {
                "budget_constraint": 0.9,
                "price_sensitivity": 0.8
            }
        }
        
        try:
            model = ToolSelectionReasoning(**data)
            
            # Verify values are preserved
            assert model.score == score
            assert model.confidence == confidence
            assert model.priority == priority
            
            # Verify JSON round-trip
            json_data = model.model_dump()
            model_from_json = ToolSelectionReasoning(**json_data)
            assert model_from_json.score == score
            assert model_from_json.confidence == confidence
            assert model_from_json.priority == priority
            
        except ValidationError as e:
            pytest.fail(f"Valid values (score={score}, confidence={confidence}, priority={priority}) failed validation: {e}")
