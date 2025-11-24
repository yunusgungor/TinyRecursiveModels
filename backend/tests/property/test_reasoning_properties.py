"""
Property-based tests for model reasoning enhancement

Feature: model-reasoning-enhancement
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import assume
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from models.tools.integrated_enhanced_trm import IntegratedEnhancedTRM, create_integrated_enhanced_config
from models.rl.environment import UserProfile


# Strategy for generating valid user profiles
@st.composite
def user_profile_strategy(draw):
    """Generate valid user profile for testing"""
    age = draw(st.integers(min_value=18, max_value=100))
    
    # Generate hobbies (1-5 non-empty strings)
    num_hobbies = draw(st.integers(min_value=1, max_value=5))
    hobbies = [
        draw(st.sampled_from([
            "cooking", "gardening", "reading", "sports", "technology",
            "fitness", "art", "music", "travel", "gaming"
        ])) for _ in range(num_hobbies)
    ]
    
    relationship = draw(st.sampled_from([
        "mother", "father", "friend", "partner", "sibling", "colleague"
    ]))
    
    budget = draw(st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    
    occasion = draw(st.sampled_from([
        "birthday", "christmas", "anniversary", "graduation", "wedding"
    ]))
    
    # Generate personality traits (0-3 strings)
    num_traits = draw(st.integers(min_value=0, max_value=3))
    personality_traits = [
        draw(st.sampled_from([
            "practical", "eco-friendly", "creative", "tech-savvy", "traditional"
        ])) for _ in range(num_traits)
    ]
    
    return UserProfile(
        age=age,
        hobbies=hobbies,
        relationship=relationship,
        budget=budget,
        occasion=occasion,
        personality_traits=personality_traits
    )


class TestAttentionWeightsNormalization:
    """
    Property 19: Attention weights normalization
    Validates: Requirements 4.4
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_weights_normalization_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 19: Attention weights normalization
        
        For any set of attention weights, the sum of all weights should equal 1.0 
        (±0.01 tolerance) and no weight should be negative.
        
        Validates: Requirements 4.4
        """
        with torch.no_grad():
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Generate category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Extract attention weights
            attention_weights = model.extract_attention_weights(
                user_encoding=user_encoding,
                gift_encodings=None,
                category_scores=category_scores
            )
            
            # Verify structure
            assert "user_features" in attention_weights, \
                "Attention weights should contain 'user_features'"
            assert "gift_features" in attention_weights, \
                "Attention weights should contain 'gift_features'"
            
            # Test user features normalization
            user_weights = attention_weights["user_features"]
            user_weight_sum = sum(user_weights.values())
            
            assert abs(user_weight_sum - 1.0) < 0.01, \
                f"User feature weights should sum to 1.0, got {user_weight_sum}"
            
            # Test no negative weights for user features
            for feature_name, weight in user_weights.items():
                assert weight >= 0.0, \
                    f"User feature weight for '{feature_name}' should be non-negative, got {weight}"
            
            # Test gift features normalization
            gift_weights = attention_weights["gift_features"]
            gift_weight_sum = sum(gift_weights.values())
            
            assert abs(gift_weight_sum - 1.0) < 0.01, \
                f"Gift feature weights should sum to 1.0, got {gift_weight_sum}"
            
            # Test no negative weights for gift features
            for feature_name, weight in gift_weights.items():
                assert weight >= 0.0, \
                    f"Gift feature weight for '{feature_name}' should be non-negative, got {weight}"
    
    def test_attention_weights_with_zero_encoding(self, model):
        """
        Test edge case: zero user encoding
        
        When user encoding is all zeros, weights should still be normalized
        """
        with torch.no_grad():
            device = next(model.parameters()).device
            
            # Create zero user encoding
            user_encoding = torch.zeros(1, model.enhanced_config.user_profile_encoding_dim, device=device)
            
            # Create zero category scores
            category_scores = torch.zeros(1, len(model.gift_categories), device=device)
            
            # Extract attention weights
            attention_weights = model.extract_attention_weights(
                user_encoding=user_encoding,
                gift_encodings=None,
                category_scores=category_scores
            )
            
            # Verify normalization even with zero inputs
            user_weight_sum = sum(attention_weights["user_features"].values())
            gift_weight_sum = sum(attention_weights["gift_features"].values())
            
            assert abs(user_weight_sum - 1.0) < 0.01, \
                f"User weights should sum to 1.0 even with zero encoding, got {user_weight_sum}"
            assert abs(gift_weight_sum - 1.0) < 0.01, \
                f"Gift weights should sum to 1.0 even with zero encoding, got {gift_weight_sum}"
    
    def test_attention_weights_with_extreme_values(self, model):
        """
        Test edge case: extreme user encoding values
        
        Weights should still be normalized even with very large or very small values
        """
        with torch.no_grad():
            device = next(model.parameters()).device
            
            # Create user encoding with extreme values
            user_encoding = torch.randn(1, model.enhanced_config.user_profile_encoding_dim, device=device) * 1000
            
            # Create category scores with extreme values
            category_scores = torch.randn(1, len(model.gift_categories), device=device) * 100
            
            # Extract attention weights
            attention_weights = model.extract_attention_weights(
                user_encoding=user_encoding,
                gift_encodings=None,
                category_scores=category_scores
            )
            
            # Verify normalization
            user_weight_sum = sum(attention_weights["user_features"].values())
            gift_weight_sum = sum(attention_weights["gift_features"].values())
            
            assert abs(user_weight_sum - 1.0) < 0.01, \
                f"User weights should sum to 1.0 with extreme values, got {user_weight_sum}"
            assert abs(gift_weight_sum - 1.0) < 0.01, \
                f"Gift weights should sum to 1.0 with extreme values, got {gift_weight_sum}"
            
            # Verify no negative weights
            for weight in attention_weights["user_features"].values():
                assert weight >= 0.0, f"User weight should be non-negative, got {weight}"
            for weight in attention_weights["gift_features"].values():
                assert weight >= 0.0, f"Gift weight should be non-negative, got {weight}"
    
    def test_attention_weights_consistency(self, model):
        """
        Test that attention weights are consistent across multiple calls with same input
        """
        with torch.no_grad():
            # Create a test profile
            profile = UserProfile(
                age=30,
                hobbies=["cooking", "reading"],
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Encode profile
            user_encoding = model.encode_user_profile(profile)
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Extract weights multiple times
            weights1 = model.extract_attention_weights(user_encoding, None, category_scores)
            weights2 = model.extract_attention_weights(user_encoding, None, category_scores)
            
            # Verify consistency
            for feature in weights1["user_features"]:
                assert abs(weights1["user_features"][feature] - weights2["user_features"][feature]) < 1e-6, \
                    f"User feature '{feature}' weights should be consistent"
            
            for feature in weights1["gift_features"]:
                assert abs(weights1["gift_features"][feature] - weights2["gift_features"][feature]) < 1e-6, \
                    f"Gift feature '{feature}' weights should be consistent"
    
    def test_attention_weights_feature_coverage(self, model):
        """
        Test that all expected features are present in attention weights
        """
        with torch.no_grad():
            # Create a test profile
            profile = UserProfile(
                age=25,
                hobbies=["technology"],
                relationship="partner",
                budget=1000.0,
                occasion="anniversary",
                personality_traits=["tech-savvy"]
            )
            
            # Encode profile
            user_encoding = model.encode_user_profile(profile)
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Extract weights
            weights = model.extract_attention_weights(user_encoding, None, category_scores)
            
            # Verify all expected user features are present
            expected_user_features = ["hobbies", "preferences", "occasion", "age", "budget"]
            for feature in expected_user_features:
                assert feature in weights["user_features"], \
                    f"User feature '{feature}' should be present in attention weights"
            
            # Verify all expected gift features are present
            expected_gift_features = ["category", "price", "rating"]
            for feature in expected_gift_features:
                assert feature in weights["gift_features"], \
                    f"Gift feature '{feature}' should be present in attention weights"
    
    @given(
        age=st.integers(min_value=18, max_value=100),
        budget=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_attention_weights_with_varying_demographics(self, age, budget, model):
        """
        Test that attention weights are properly normalized across different demographics
        """
        with torch.no_grad():
            profile = UserProfile(
                age=age,
                hobbies=["reading"],
                relationship="friend",
                budget=budget,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            user_encoding = model.encode_user_profile(profile)
            category_scores = model.enhanced_category_matching(user_encoding)
            
            weights = model.extract_attention_weights(user_encoding, None, category_scores)
            
            # Verify normalization
            user_sum = sum(weights["user_features"].values())
            gift_sum = sum(weights["gift_features"].values())
            
            assert abs(user_sum - 1.0) < 0.01, \
                f"User weights should sum to 1.0 for age={age}, budget={budget}, got {user_sum}"
            assert abs(gift_sum - 1.0) < 0.01, \
                f"Gift weights should sum to 1.0 for age={age}, budget={budget}, got {gift_sum}"



class TestThinkingStepsStructure:
    """
    Property 22: Thinking step structure
    Validates: Requirements 5.2
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_thinking_step_structure_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 22: Thinking step structure
        
        For any thinking step, the record should include step_number, action, 
        result, and insight fields.
        
        Validates: Requirements 5.2
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Create some dummy gifts
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="technology",
                    price=100.0 + i * 50,
                    rating=4.0,
                    tags=["practical", "modern"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            # Run forward pass to get model output
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=False
            )
            
            # Get thinking steps
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Verify that we have thinking steps
            assert len(thinking_steps) > 0, "Should generate at least one thinking step"
            
            # Verify structure of each step
            for step in thinking_steps:
                # Check all required fields are present
                assert "step" in step, "Thinking step should have 'step' field"
                assert "action" in step, "Thinking step should have 'action' field"
                assert "result" in step, "Thinking step should have 'result' field"
                assert "insight" in step, "Thinking step should have 'insight' field"
                
                # Check field types
                assert isinstance(step["step"], int), \
                    f"Step number should be int, got {type(step['step'])}"
                assert isinstance(step["action"], str), \
                    f"Action should be str, got {type(step['action'])}"
                assert isinstance(step["result"], str), \
                    f"Result should be str, got {type(step['result'])}"
                assert isinstance(step["insight"], str), \
                    f"Insight should be str, got {type(step['insight'])}"
                
                # Check that strings are non-empty
                assert len(step["action"]) > 0, "Action should not be empty"
                assert len(step["result"]) > 0, "Result should not be empty"
                assert len(step["insight"]) > 0, "Insight should not be empty"
                
                # Check step number is positive
                assert step["step"] > 0, f"Step number should be positive, got {step['step']}"
    
    def test_thinking_steps_major_steps_coverage(self, model):
        """
        Test that all major steps are recorded
        
        Major steps should include: encode, match, select, execute (if tools used), rank
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profile = UserProfile(
                age=30,
                hobbies=["cooking", "reading"],
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="kitchen",
                    price=100.0 + i * 50,
                    rating=4.5,
                    tags=["cooking", "practical"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            # Run with tool execution
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=True
            )
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Extract action names
            actions = [step["action"] for step in thinking_steps]
            
            # Check for major steps
            assert any("encode" in action.lower() or "profile" in action.lower() for action in actions), \
                "Should have user encoding step"
            assert any("match" in action.lower() or "categor" in action.lower() for action in actions), \
                "Should have category matching step"
            assert any("select" in action.lower() or "tool" in action.lower() for action in actions), \
                "Should have tool selection step"
            assert any("rank" in action.lower() or "gift" in action.lower() for action in actions), \
                "Should have gift ranking step"
    
    def test_thinking_steps_with_tool_execution(self, model):
        """
        Test that tool execution step is included when tools are executed
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profile = UserProfile(
                age=25,
                hobbies=["technology"],
                relationship="partner",
                budget=1000.0,
                occasion="anniversary",
                personality_traits=["tech-savvy"]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Tech Gift {i}",
                    category="technology",
                    price=200.0 + i * 100,
                    rating=4.8,
                    tags=["technology", "modern"],
                    age_suitability=(18, 65),
                    occasion_fit=["anniversary"],
                    description=f"Tech gift {i}"
                ) for i in range(5)
            ]
            
            # Run with tool execution enabled
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=True
            )
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # If tools were executed, there should be a tool execution step
            if model_output.get("executed_tools"):
                actions = [step["action"].lower() for step in thinking_steps]
                assert any("execute" in action or "tool" in action for action in actions), \
                    "Should have tool execution step when tools are executed"
    
    def test_thinking_steps_without_tool_execution(self, model):
        """
        Test that thinking steps work correctly when no tools are executed
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profile = UserProfile(
                age=30,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday",
                personality_traits=[]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Book {i}",
                    category="books",
                    price=20.0 + i * 5,
                    rating=4.0,
                    tags=["reading", "educational"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Book {i}"
                ) for i in range(5)
            ]
            
            # Run without tool execution
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=False
            )
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Should still have thinking steps even without tool execution
            assert len(thinking_steps) > 0, "Should have thinking steps even without tool execution"
            
            # Verify structure is still correct
            for step in thinking_steps:
                assert all(field in step for field in ["step", "action", "result", "insight"]), \
                    "All steps should have required fields"
    
    def test_thinking_steps_insight_quality(self, model):
        """
        Test that insights are meaningful and context-aware
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            # Test with a profile that has strong characteristics
            profile = UserProfile(
                age=22,
                hobbies=["gaming", "technology", "music"],
                relationship="friend",
                budget=2000.0,
                occasion="birthday",
                personality_traits=["tech-savvy", "creative"]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Gaming Gift {i}",
                    category="gaming",
                    price=500.0 + i * 100,
                    rating=4.7,
                    tags=["gaming", "technology"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Gaming gift {i}"
                ) for i in range(5)
            ]
            
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=True
            )
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Find the user encoding step
            encoding_step = next((s for s in thinking_steps if "encode" in s["action"].lower()), None)
            
            if encoding_step:
                insight = encoding_step["insight"].lower()
                # Insight should mention something about the user's characteristics
                # Check for hobby mentions or demographic info
                has_relevant_info = (
                    any(hobby in insight for hobby in ["gaming", "technology", "music"]) or
                    "young" in insight or
                    "premium" in insight or
                    "budget" in insight
                )
                assert has_relevant_info, \
                    f"User encoding insight should contain relevant profile information, got: {encoding_step['insight']}"


class TestChronologicalStepOrdering:
    """
    Property 26: Chronological step ordering
    Validates: Requirements 5.6
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chronological_step_ordering_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 26: Chronological step ordering
        
        For any API response containing thinking steps, the steps should be ordered 
        chronologically (step numbers in ascending order).
        
        Validates: Requirements 5.6
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Create some dummy gifts
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="technology",
                    price=100.0 + i * 50,
                    rating=4.0,
                    tags=["practical", "modern"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            # Run forward pass
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=True
            )
            
            # Get thinking steps
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Verify chronological ordering
            step_numbers = [step["step"] for step in thinking_steps]
            
            # Check that step numbers are in ascending order
            for i in range(len(step_numbers) - 1):
                assert step_numbers[i] < step_numbers[i + 1], \
                    f"Steps should be in chronological order: step {step_numbers[i]} should come before {step_numbers[i + 1]}"
            
            # Check that step numbers start from 1
            assert step_numbers[0] == 1, \
                f"First step should be numbered 1, got {step_numbers[0]}"
            
            # Check that step numbers are consecutive (no gaps)
            for i in range(len(step_numbers) - 1):
                assert step_numbers[i + 1] == step_numbers[i] + 1, \
                    f"Step numbers should be consecutive: {step_numbers[i]} should be followed by {step_numbers[i] + 1}, got {step_numbers[i + 1]}"
    
    def test_step_ordering_with_multiple_profiles(self, model):
        """
        Test that step ordering is consistent across different user profiles
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profiles = [
                UserProfile(
                    age=25,
                    hobbies=["cooking"],
                    relationship="mother",
                    budget=300.0,
                    occasion="birthday",
                    personality_traits=["practical"]
                ),
                UserProfile(
                    age=60,
                    hobbies=["gardening", "reading"],
                    relationship="father",
                    budget=1500.0,
                    occasion="christmas",
                    personality_traits=["traditional"]
                ),
                UserProfile(
                    age=35,
                    hobbies=["technology", "gaming"],
                    relationship="friend",
                    budget=800.0,
                    occasion="anniversary",
                    personality_traits=["tech-savvy", "creative"]
                )
            ]
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="general",
                    price=100.0 + i * 50,
                    rating=4.0,
                    tags=["practical"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday", "christmas"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            for profile in profiles:
                env_state = EnvironmentState(
                    user_profile=profile,
                available_gifts=[],
                    current_recommendations=[],
                    interaction_history=[],
                    step_count=0
                )
                
                carry = None
                carry, model_output, selected_tools = model.forward_with_enhancements(
                    carry, env_state, available_gifts, execute_tools=True
                )
                
                thinking_steps = model.get_thinking_steps(env_state, model_output)
                step_numbers = [step["step"] for step in thinking_steps]
                
                # Verify ordering for each profile
                assert step_numbers == sorted(step_numbers), \
                    f"Steps should be in ascending order for profile with age={profile.age}"
                assert step_numbers[0] == 1, \
                    f"First step should be 1 for profile with age={profile.age}"
    
    def test_step_ordering_stability(self, model):
        """
        Test that step ordering is stable across multiple calls with same input
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profile = UserProfile(
                age=30,
                hobbies=["reading", "cooking"],
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="books",
                    price=50.0 + i * 20,
                    rating=4.5,
                    tags=["reading", "educational"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            # Run multiple times
            step_sequences = []
            for _ in range(3):
                carry = None
                carry, model_output, selected_tools = model.forward_with_enhancements(
                    carry, env_state, available_gifts, execute_tools=True
                )
                
                thinking_steps = model.get_thinking_steps(env_state, model_output)
                step_numbers = [step["step"] for step in thinking_steps]
                step_sequences.append(step_numbers)
            
            # Verify all sequences are identical
            for i in range(len(step_sequences) - 1):
                assert step_sequences[i] == step_sequences[i + 1], \
                    "Step ordering should be stable across multiple calls with same input"
    
    def test_no_duplicate_step_numbers(self, model):
        """
        Test that there are no duplicate step numbers
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            profile = UserProfile(
                age=40,
                hobbies=["sports", "fitness"],
                relationship="partner",
                budget=1000.0,
                occasion="anniversary",
                personality_traits=["active"]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Fitness Gift {i}",
                    category="fitness",
                    price=150.0 + i * 75,
                    rating=4.6,
                    tags=["sports", "fitness", "health"],
                    age_suitability=(18, 65),
                    occasion_fit=["anniversary", "birthday"],
                    description=f"Fitness gift {i}"
                ) for i in range(5)
            ]
            
            carry = None
            carry, model_output, selected_tools = model.forward_with_enhancements(
                carry, env_state, available_gifts, execute_tools=True
            )
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            step_numbers = [step["step"] for step in thinking_steps]
            
            # Check for duplicates
            assert len(step_numbers) == len(set(step_numbers)), \
                f"Step numbers should be unique, found duplicates: {step_numbers}"
    
    def test_step_ordering_with_empty_model_output(self, model):
        """
        Test that step ordering works correctly even with minimal model output
        """
        from models.rl.environment import EnvironmentState
        
        with torch.no_grad():
            profile = UserProfile(
                age=30,
                hobbies=["reading"],
                relationship="friend",
                budget=200.0,
                occasion="birthday",
                personality_traits=[]
            )
            
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Create minimal model output
            model_output = {
                "action_probs": torch.randn(1, 10),
                "category_scores": torch.randn(1, 5)
            }
            
            thinking_steps = model.get_thinking_steps(env_state, model_output)
            
            # Should still have at least some steps
            assert len(thinking_steps) > 0, "Should generate steps even with minimal output"
            
            # Verify ordering
            step_numbers = [step["step"] for step in thinking_steps]
            assert step_numbers == sorted(step_numbers), \
                "Steps should be in ascending order even with minimal output"
            assert step_numbers[0] == 1, \
                "First step should be 1 even with minimal output"



class TestToolSelectionReasoningCompleteness:
    """
    Property 1: Tool selection reasoning completeness
    Validates: Requirements 1.1
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_tool_selection_reasoning_completeness_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 1: Tool selection reasoning completeness
        
        For any model inference that selects tools, the model should generate reasoning 
        that includes selection reason, confidence score, and priority for each selected tool.
        
        Validates: Requirements 1.1
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Create some dummy gifts
            available_gifts = [
                GiftItem(
                    id=f"gift_{i}",
                    name=f"Test Gift {i}",
                    category="technology",
                    price=100.0 + i * 50,
                    rating=4.0,
                    tags=["practical", "modern"],
                    age_suitability=(18, 65),
                    occasion_fit=["birthday"],
                    description=f"Test gift {i}"
                ) for i in range(5)
            ]
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get tool selection
            selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
            
            # Get tool selection reasoning
            tool_reasoning = model.explain_tool_selection(
                tool_scores=tool_scores,
                user_encoding=user_encoding,
                selected_tools=selected_tools[0] if selected_tools else []
            )
            
            # Verify reasoning exists for all tools
            tool_names = list(model.tool_registry.list_tools())
            assert len(tool_reasoning) > 0, "Should generate reasoning for at least one tool"
            
            # For each tool with reasoning, verify completeness
            for tool_name, reasoning in tool_reasoning.items():
                # Check all required fields are present
                assert "selected" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'selected' field"
                assert "score" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'score' field"
                assert "reason" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'reason' field"
                assert "confidence" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'confidence' field"
                assert "priority" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'priority' field"
                assert "factors" in reasoning, \
                    f"Tool '{tool_name}' reasoning should have 'factors' field"
                
                # Check field types
                assert isinstance(reasoning["selected"], bool), \
                    f"Tool '{tool_name}' selected should be bool, got {type(reasoning['selected'])}"
                assert isinstance(reasoning["score"], (int, float)), \
                    f"Tool '{tool_name}' score should be numeric, got {type(reasoning['score'])}"
                assert isinstance(reasoning["reason"], str), \
                    f"Tool '{tool_name}' reason should be str, got {type(reasoning['reason'])}"
                assert isinstance(reasoning["confidence"], (int, float)), \
                    f"Tool '{tool_name}' confidence should be numeric, got {type(reasoning['confidence'])}"
                assert isinstance(reasoning["priority"], int), \
                    f"Tool '{tool_name}' priority should be int, got {type(reasoning['priority'])}"
                assert isinstance(reasoning["factors"], dict), \
                    f"Tool '{tool_name}' factors should be dict, got {type(reasoning['factors'])}"
                
                # Check value ranges
                assert 0.0 <= reasoning["score"] <= 1.0, \
                    f"Tool '{tool_name}' score should be in [0, 1], got {reasoning['score']}"
                assert 0.0 <= reasoning["confidence"] <= 1.0, \
                    f"Tool '{tool_name}' confidence should be in [0, 1], got {reasoning['confidence']}"
                assert reasoning["priority"] > 0, \
                    f"Tool '{tool_name}' priority should be positive, got {reasoning['priority']}"
                
                # Check reason is non-empty
                assert len(reasoning["reason"]) > 0, \
                    f"Tool '{tool_name}' reason should not be empty"
                
                # Check factors are valid
                for factor_name, factor_value in reasoning["factors"].items():
                    assert isinstance(factor_value, (int, float)), \
                        f"Factor '{factor_name}' for tool '{tool_name}' should be numeric"
                    assert 0.0 <= factor_value <= 1.0, \
                        f"Factor '{factor_name}' for tool '{tool_name}' should be in [0, 1], got {factor_value}"
    
    def test_tool_selection_reasoning_for_selected_tools(self, model):
        """
        Test that selected tools have complete reasoning
        """
        from models.rl.environment import EnvironmentState, GiftItem
        
        with torch.no_grad():
            # Create a profile that should trigger tool selection
            profile = UserProfile(
                age=30,
                hobbies=["cooking", "technology"],
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical", "tech-savvy"]
            )
            
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get tool selection
            selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
            
            # Get tool selection reasoning
            tool_reasoning = model.explain_tool_selection(
                tool_scores=tool_scores,
                user_encoding=user_encoding,
                selected_tools=selected_tools[0] if selected_tools else []
            )
            
            # Verify that selected tools have reasoning marked as selected
            if selected_tools and len(selected_tools) > 0:
                for tool_name in selected_tools[0]:
                    assert tool_name in tool_reasoning, \
                        f"Selected tool '{tool_name}' should have reasoning"
                    assert tool_reasoning[tool_name]["selected"] is True, \
                        f"Selected tool '{tool_name}' should be marked as selected"
                    assert len(tool_reasoning[tool_name]["reason"]) > 0, \
                        f"Selected tool '{tool_name}' should have non-empty reason"


class TestLowConfidenceToolSelectionExplanation:
    """
    Property 4: Low confidence tool selection explanation
    Validates: Requirements 1.4
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_low_confidence_tool_selection_explanation_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 4: Low confidence tool selection explanation
        
        For any tool selection with confidence below 0.5, the reasoning should explain 
        why the confidence is low.
        
        Validates: Requirements 1.4
        """
        from models.rl.environment import EnvironmentState
        
        with torch.no_grad():
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get tool selection
            selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
            
            # Get tool selection reasoning
            tool_reasoning = model.explain_tool_selection(
                tool_scores=tool_scores,
                user_encoding=user_encoding,
                selected_tools=selected_tools[0] if selected_tools else []
            )
            
            # Find tools with low confidence (< 0.5)
            low_confidence_tools = [
                (tool_name, reasoning) 
                for tool_name, reasoning in tool_reasoning.items()
                if reasoning["confidence"] < 0.5
            ]
            
            # For each low confidence tool, verify explanation exists
            for tool_name, reasoning in low_confidence_tools:
                # Check that reason mentions low confidence or score
                reason_lower = reasoning["reason"].lower()
                
                # The reason should indicate why confidence is low
                # It should mention either "low confidence", "low score", or provide specific explanation
                has_low_confidence_explanation = (
                    "low confidence" in reason_lower or
                    "low score" in reason_lower or
                    "score:" in reason_lower or
                    "not selected" in reason_lower or
                    "lower priority" in reason_lower
                )
                
                assert has_low_confidence_explanation, \
                    f"Tool '{tool_name}' with confidence {reasoning['confidence']:.2f} should explain low confidence. " \
                    f"Reason: '{reasoning['reason']}'"
                
                # Verify reason is non-empty
                assert len(reasoning["reason"]) > 0, \
                    f"Tool '{tool_name}' with low confidence should have non-empty reason"
    
    def test_low_confidence_explanation_content(self, model):
        """
        Test that low confidence explanations provide meaningful information
        """
        from models.rl.environment import EnvironmentState
        
        with torch.no_grad():
            # Create a minimal profile that should result in low tool scores
            profile = UserProfile(
                age=25,
                hobbies=["reading"],
                relationship="friend",
                budget=5000.0,  # Very high budget - might not need price comparison
                occasion="birthday",
                personality_traits=[]
            )
            
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get tool selection
            selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
            
            # Get tool selection reasoning
            tool_reasoning = model.explain_tool_selection(
                tool_scores=tool_scores,
                user_encoding=user_encoding,
                selected_tools=selected_tools[0] if selected_tools else []
            )
            
            # Check that at least some tools have explanations
            assert len(tool_reasoning) > 0, "Should have reasoning for tools"
            
            # Verify that low confidence tools have explanations
            for tool_name, reasoning in tool_reasoning.items():
                if reasoning["confidence"] < 0.5:
                    # Reason should be informative (more than just a generic message)
                    assert len(reasoning["reason"]) > 10, \
                        f"Tool '{tool_name}' low confidence reason should be informative"
                    
                    # Should have factors explaining the low confidence
                    assert len(reasoning["factors"]) > 0, \
                        f"Tool '{tool_name}' should have factors explaining low confidence"
    
    @given(
        budget=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        num_hobbies=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_low_confidence_with_varying_profiles(self, budget, num_hobbies, model):
        """
        Test low confidence explanations across different user profiles
        """
        from models.rl.environment import EnvironmentState
        
        with torch.no_grad():
            # Generate hobbies
            hobby_options = ["cooking", "reading", "sports", "technology", "art"]
            hobbies = hobby_options[:num_hobbies] if num_hobbies > 0 else []
            
            profile = UserProfile(
                age=30,
                hobbies=hobbies,
                relationship="friend",
                budget=budget,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Create environment state
            env_state = EnvironmentState(
                user_profile=profile,
                available_gifts=[],
                current_recommendations=[],
                interaction_history=[],
                step_count=0
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get tool selection
            selected_tools, tool_scores = model.enhanced_tool_selection(user_encoding, category_scores)
            
            # Get tool selection reasoning
            tool_reasoning = model.explain_tool_selection(
                tool_scores=tool_scores,
                user_encoding=user_encoding,
                selected_tools=selected_tools[0] if selected_tools else []
            )
            
            # Verify all low confidence tools have explanations
            for tool_name, reasoning in tool_reasoning.items():
                if reasoning["confidence"] < 0.5:
                    assert len(reasoning["reason"]) > 0, \
                        f"Tool '{tool_name}' with confidence {reasoning['confidence']:.2f} " \
                        f"should have explanation (budget={budget}, hobbies={len(hobbies)})"



class TestCategoryScoreAndExplanationConsistency:
    """
    Property 11: Category score and explanation consistency
    Validates: Requirements 9.3
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_category_score_and_explanation_consistency_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 11: Category score and explanation consistency
        
        For any category reasoning, the explanations should be consistent with the numerical scores 
        (high scores should have positive explanations, low scores should have negative explanations).
        
        Validates: Requirements 9.3
        """
        with torch.no_grad():
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Verify reasoning exists
            assert len(category_reasoning) > 0, "Should generate reasoning for at least one category"
            
            # For each category, verify consistency between score and explanation
            for category_name, reasoning in category_reasoning.items():
                score = reasoning["score"]
                reasons = reasoning["reasons"]
                
                # Verify structure
                assert isinstance(score, (int, float)), \
                    f"Category '{category_name}' score should be numeric, got {type(score)}"
                assert isinstance(reasons, list), \
                    f"Category '{category_name}' reasons should be a list, got {type(reasons)}"
                assert len(reasons) > 0, \
                    f"Category '{category_name}' should have at least one reason"
                
                # Verify score range
                assert 0.0 <= score <= 1.0, \
                    f"Category '{category_name}' score should be in [0, 1], got {score}"
                
                # Check consistency based on score level
                reasons_text = " ".join(reasons).lower()
                
                if score > 0.7:
                    # High score - should have positive explanations
                    positive_indicators = [
                        "strong", "high", "excellent", "perfect", "great",
                        "suitable", "appropriate", "matches", "alignment",
                        "good", "relevant", "fit"
                    ]
                    
                    has_positive_explanation = any(
                        indicator in reasons_text for indicator in positive_indicators
                    )
                    
                    # Should NOT have predominantly negative language
                    negative_indicators = [
                        "weak", "low", "poor", "limited", "mismatch",
                        "inappropriate", "concerns"
                    ]
                    
                    negative_count = sum(1 for indicator in negative_indicators if indicator in reasons_text)
                    positive_count = sum(1 for indicator in positive_indicators if indicator in reasons_text)
                    
                    assert has_positive_explanation or positive_count > negative_count, \
                        f"Category '{category_name}' with high score {score:.2f} should have positive explanations. " \
                        f"Reasons: {reasons}"
                
                elif score < 0.3:
                    # Low score - should have negative/explanatory language
                    negative_indicators = [
                        "weak", "low", "poor", "limited", "mismatch",
                        "inappropriate", "concerns", "not", "less"
                    ]
                    
                    has_negative_explanation = any(
                        indicator in reasons_text for indicator in negative_indicators
                    )
                    
                    # Or should explain why it's low
                    explanatory_indicators = [
                        "score", "match", "relevance", "fit", "compatibility"
                    ]
                    
                    has_explanation = any(
                        indicator in reasons_text for indicator in explanatory_indicators
                    )
                    
                    assert has_negative_explanation or has_explanation, \
                        f"Category '{category_name}' with low score {score:.2f} should explain why score is low. " \
                        f"Reasons: {reasons}"
                
                else:
                    # Medium score (0.3-0.7) - should have balanced or moderate language
                    # Just verify it has some explanation
                    assert len(reasons_text) > 0, \
                        f"Category '{category_name}' with medium score {score:.2f} should have explanations"
    
    def test_category_explanation_with_extreme_scores(self, model):
        """
        Test that explanations are consistent with extreme scores (very high or very low)
        """
        with torch.no_grad():
            # Create a profile with strong characteristics
            profile = UserProfile(
                age=30,
                hobbies=["cooking", "baking"],
                relationship="mother",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical", "creative"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Find categories with extreme scores
            high_score_categories = [
                (name, reasoning) for name, reasoning in category_reasoning.items()
                if reasoning["score"] > 0.8
            ]
            
            low_score_categories = [
                (name, reasoning) for name, reasoning in category_reasoning.items()
                if reasoning["score"] < 0.2
            ]
            
            # Verify high score categories have positive language
            for category_name, reasoning in high_score_categories:
                reasons_text = " ".join(reasoning["reasons"]).lower()
                
                # Should contain positive indicators
                positive_words = ["strong", "high", "excellent", "perfect", "great", "suitable"]
                has_positive = any(word in reasons_text for word in positive_words)
                
                assert has_positive or "match" in reasons_text or "fit" in reasons_text, \
                    f"High score category '{category_name}' ({reasoning['score']:.2f}) should have positive language"
            
            # Verify low score categories have explanatory language
            for category_name, reasoning in low_score_categories:
                reasons_text = " ".join(reasoning["reasons"]).lower()
                
                # Should explain why it's low
                explanatory_words = ["weak", "low", "limited", "mismatch", "not"]
                has_explanation = any(word in reasons_text for word in explanatory_words)
                
                assert has_explanation or "score" in reasons_text, \
                    f"Low score category '{category_name}' ({reasoning['score']:.2f}) should explain low score"
    
    def test_category_feature_contributions_consistency(self, model):
        """
        Test that feature contributions are consistent with the overall score
        """
        with torch.no_grad():
            profile = UserProfile(
                age=25,
                hobbies=["technology", "gaming"],
                relationship="friend",
                budget=1000.0,
                occasion="birthday",
                personality_traits=["tech-savvy"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Verify feature contributions
            for category_name, reasoning in category_reasoning.items():
                score = reasoning["score"]
                factors = reasoning["feature_contributions"]
                
                # Verify factors structure
                assert isinstance(factors, dict), \
                    f"Category '{category_name}' feature_contributions should be dict"
                assert len(factors) > 0, \
                    f"Category '{category_name}' should have feature contributions"
                
                # Verify all factor values are in valid range
                for factor_name, factor_value in factors.items():
                    assert isinstance(factor_value, (int, float)), \
                        f"Factor '{factor_name}' for category '{category_name}' should be numeric"
                    assert 0.0 <= factor_value <= 1.0, \
                        f"Factor '{factor_name}' for category '{category_name}' should be in [0, 1], got {factor_value}"
    
    @given(
        age=st.integers(min_value=18, max_value=100),
        budget=st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_category_consistency_across_demographics(self, age, budget, model):
        """
        Test that category explanations are consistent across different demographics
        """
        with torch.no_grad():
            profile = UserProfile(
                age=age,
                hobbies=["reading", "cooking"],
                relationship="friend",
                budget=budget,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Verify consistency for each category
            for category_name, reasoning in category_reasoning.items():
                score = reasoning["score"]
                reasons = reasoning["reasons"]
                
                # Verify basic consistency
                assert len(reasons) > 0, \
                    f"Category '{category_name}' should have reasons (age={age}, budget={budget})"
                
                # Verify score-explanation consistency
                reasons_text = " ".join(reasons).lower()
                
                if score > 0.7:
                    # Should have some positive language
                    has_positive = any(
                        word in reasons_text 
                        for word in ["strong", "high", "good", "suitable", "match", "fit"]
                    )
                    assert has_positive, \
                        f"High score category '{category_name}' should have positive language " \
                        f"(age={age}, budget={budget}, score={score:.2f})"
                
                elif score < 0.3:
                    # Should have some explanatory language
                    has_explanation = any(
                        word in reasons_text 
                        for word in ["weak", "low", "limited", "not", "score"]
                    )
                    assert has_explanation, \
                        f"Low score category '{category_name}' should explain low score " \
                        f"(age={age}, budget={budget}, score={score:.2f})"


class TestTopCategoriesMinimumCount:
    """
    Property 9: Top categories minimum count
    Validates: Requirements 2.4
    """
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        config = create_integrated_enhanced_config()
        model = IntegratedEnhancedTRM(config, verbose=False)
        model.eval()
        return model
    
    @given(profile=user_profile_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_top_categories_minimum_count_property(self, profile: UserProfile, model):
        """
        Feature: model-reasoning-enhancement, Property 9: Top categories minimum count
        
        For any category matching operation, the model should return at least 
        the top 3 categories with their scores.
        
        Validates: Requirements 2.4
        """
        with torch.no_grad():
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Verify minimum count
            num_categories = len(category_reasoning)
            
            # Should have at least 3 categories (or all available if less than 3)
            min_expected = min(3, len(model.gift_categories))
            
            assert num_categories >= min_expected, \
                f"Should return at least {min_expected} categories, got {num_categories}"
            
            # Verify each category has a score
            for category_name, reasoning in category_reasoning.items():
                assert "score" in reasoning, \
                    f"Category '{category_name}' should have a score"
                assert isinstance(reasoning["score"], (int, float)), \
                    f"Category '{category_name}' score should be numeric"
                assert 0.0 <= reasoning["score"] <= 1.0, \
                    f"Category '{category_name}' score should be in [0, 1], got {reasoning['score']}"
    
    def test_top_categories_are_highest_scored(self, model):
        """
        Test that returned categories are indeed the top-scored ones
        """
        with torch.no_grad():
            profile = UserProfile(
                age=30,
                hobbies=["cooking", "reading", "technology"],
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical", "creative"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Extract scores from reasoning
            reasoning_scores = {
                name: reasoning["score"] 
                for name, reasoning in category_reasoning.items()
            }
            
            # Get all category scores
            category_scores_flat = category_scores.squeeze(0) if category_scores.dim() > 1 else category_scores
            all_scores = category_scores_flat.tolist()
            
            # Sort all scores in descending order
            sorted_all_scores = sorted(all_scores, reverse=True)
            
            # Get top 3 scores from all scores
            top_3_all_scores = sorted_all_scores[:min(3, len(sorted_all_scores))]
            
            # Get scores from reasoning
            reasoning_score_values = sorted(reasoning_scores.values(), reverse=True)
            
            # Verify that reasoning contains the top scores
            # Allow small floating point differences
            for i, expected_score in enumerate(top_3_all_scores):
                if i < len(reasoning_score_values):
                    actual_score = reasoning_score_values[i]
                    assert abs(actual_score - expected_score) < 0.01, \
                        f"Top {i+1} category score should be {expected_score:.3f}, got {actual_score:.3f}"
    
    def test_top_categories_with_minimal_profile(self, model):
        """
        Test that minimum count is maintained even with minimal user profile
        """
        with torch.no_grad():
            # Create minimal profile
            profile = UserProfile(
                age=25,
                hobbies=["reading"],
                relationship="friend",
                budget=100.0,
                occasion="birthday",
                personality_traits=[]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Should still have at least 3 categories
            min_expected = min(3, len(model.gift_categories))
            assert len(category_reasoning) >= min_expected, \
                f"Should return at least {min_expected} categories even with minimal profile, got {len(category_reasoning)}"
    
    def test_top_categories_with_rich_profile(self, model):
        """
        Test that minimum count is maintained with rich user profile
        """
        with torch.no_grad():
            # Create rich profile
            profile = UserProfile(
                age=35,
                hobbies=["cooking", "gardening", "reading", "technology", "fitness"],
                relationship="partner",
                budget=2000.0,
                occasion="anniversary",
                personality_traits=["practical", "creative", "tech-savvy", "eco-friendly"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Should have at least 3 categories
            min_expected = min(3, len(model.gift_categories))
            assert len(category_reasoning) >= min_expected, \
                f"Should return at least {min_expected} categories with rich profile, got {len(category_reasoning)}"
    
    @given(num_hobbies=st.integers(min_value=0, max_value=5))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_top_categories_count_with_varying_hobbies(self, num_hobbies, model):
        """
        Test that minimum count is maintained across different hobby counts
        """
        with torch.no_grad():
            # Generate hobbies
            hobby_options = ["cooking", "reading", "sports", "technology", "art", "music"]
            hobbies = hobby_options[:num_hobbies] if num_hobbies > 0 else []
            
            profile = UserProfile(
                age=30,
                hobbies=hobbies,
                relationship="friend",
                budget=500.0,
                occasion="birthday",
                personality_traits=["practical"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Should have at least 3 categories
            min_expected = min(3, len(model.gift_categories))
            assert len(category_reasoning) >= min_expected, \
                f"Should return at least {min_expected} categories with {num_hobbies} hobbies, got {len(category_reasoning)}"
    
    def test_category_ordering_by_score(self, model):
        """
        Test that categories are ordered by score (highest first)
        """
        with torch.no_grad():
            profile = UserProfile(
                age=28,
                hobbies=["technology", "gaming"],
                relationship="friend",
                budget=1500.0,
                occasion="birthday",
                personality_traits=["tech-savvy", "modern"]
            )
            
            # Encode user profile
            user_encoding = model.encode_user_profile(profile)
            
            # Get category scores
            category_scores = model.enhanced_category_matching(user_encoding)
            
            # Get category matching explanation
            category_reasoning = model.explain_category_matching(
                category_scores=category_scores,
                user_encoding=user_encoding
            )
            
            # Extract scores in order
            scores = [reasoning["score"] for reasoning in category_reasoning.values()]
            
            # Verify scores are in descending order (or equal)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], \
                    f"Categories should be ordered by score (descending): {scores[i]:.3f} should be >= {scores[i+1]:.3f}"
