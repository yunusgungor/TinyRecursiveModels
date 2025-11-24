"""
Property-based tests for ReasoningService

Feature: model-reasoning-enhancement
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import assume
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.app.services.reasoning_service import ReasoningService, get_reasoning_service
from backend.app.models.schemas import UserProfile, GiftItem


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


# Strategy for generating valid gift items
@st.composite
def gift_item_strategy(draw):
    """Generate valid gift item for testing"""
    gift_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    name = draw(st.sampled_from([
        "Premium Coffee Set", "Gardening Tool Kit", "Best Seller Book",
        "Fitness Tracker", "Smart Watch", "Cooking Utensils",
        "Art Supply Set", "Gaming Headset", "Travel Backpack"
    ]))
    
    category = draw(st.sampled_from([
        "Kitchen & Dining", "Garden & Outdoor", "Books", "Electronics",
        "Sports & Outdoor", "Art & Craft", "Gaming", "Luggage & Bags"
    ]))
    
    price = draw(st.floats(min_value=10.0, max_value=3000.0, allow_nan=False, allow_infinity=False))
    rating = draw(st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    
    tags = draw(st.lists(
        st.sampled_from(["practical", "modern", "eco-friendly", "premium", "tech", "creative"]),
        min_size=1,
        max_size=5
    ))
    
    age_min = draw(st.integers(min_value=18, max_value=50))
    age_max = draw(st.integers(min_value=age_min, max_value=100))
    
    occasion_fit = draw(st.lists(
        st.sampled_from(["birthday", "christmas", "anniversary", "graduation", "wedding"]),
        min_size=1,
        max_size=3
    ))
    
    return GiftItem(
        id=gift_id,
        name=name,
        category=category,
        price=price,
        rating=rating,
        image_url=f"https://cdn.trendyol.com/{gift_id}.jpg",
        trendyol_url=f"https://www.trendyol.com/product/{gift_id}",
        description=f"High-quality {name.lower()}",
        tags=tags,
        age_suitability=(age_min, age_max),
        occasion_fit=occasion_fit,
        in_stock=True
    )



class TestDynamicGiftReasoningGeneration:
    """
    Property 12: Dynamic gift reasoning generation
    Validates: Requirements 3.1, 3.6
    """
    
    @pytest.fixture(scope="class")
    def reasoning_service(self):
        """Create reasoning service instance for testing"""
        return ReasoningService()
    
    @given(
        profile=user_profile_strategy(),
        gift=gift_item_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_dynamic_gift_reasoning_generation_property(
        self,
        profile: UserProfile,
        gift: GiftItem,
        reasoning_service
    ):
        """
        Feature: model-reasoning-enhancement, Property 12: Dynamic gift reasoning generation
        
        For any gift recommendation, the backend should generate reasoning based on 
        user profile and model output, not static templates.
        
        Validates: Requirements 3.1, 3.6
        """
        # Generate reasoning
        model_output = {}
        tool_results = {}
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output=model_output,
            tool_results=tool_results
        )
        
        # Verify reasoning is generated
        assert isinstance(reasoning, list), "Reasoning should be a list"
        assert len(reasoning) > 0, "Should generate at least one reasoning statement"
        
        # Verify all reasoning items are strings
        for item in reasoning:
            assert isinstance(item, str), f"Reasoning item should be string, got {type(item)}"
            assert len(item) > 0, "Reasoning item should not be empty"
        
        # Verify reasoning is dynamic (contains profile-specific information)
        reasoning_text = " ".join(reasoning).lower()
        
        # Should contain at least one of: budget info, hobby info, age info, or occasion info
        has_budget_info = str(int(profile.budget)) in reasoning_text or "budget" in reasoning_text
        has_hobby_info = any(hobby.lower() in reasoning_text for hobby in profile.hobbies)
        has_age_info = str(profile.age) in reasoning_text or "age" in reasoning_text
        has_occasion_info = profile.occasion.lower() in reasoning_text
        
        has_dynamic_content = has_budget_info or has_hobby_info or has_age_info or has_occasion_info
        
        assert has_dynamic_content, \
            f"Reasoning should contain dynamic profile-specific information. " \
            f"Profile: age={profile.age}, budget={profile.budget}, hobbies={profile.hobbies}, occasion={profile.occasion}. " \
            f"Reasoning: {reasoning}"
    
    def test_reasoning_with_hobby_match(self, reasoning_service):
        """Test that hobby matching is included in reasoning"""
        profile = UserProfile(
            age=30,
            hobbies=["cooking", "reading"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        gift = GiftItem(
            id="test123",
            name="Premium Cookbook",
            category="Books",
            price=150.0,
            rating=4.5,
            image_url="https://cdn.trendyol.com/test123.jpg",
            trendyol_url="https://www.trendyol.com/product/test123",
            description="Comprehensive cooking guide",
            tags=["cooking", "books", "practical"],
            age_suitability=(18, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Should mention hobby match
        assert "hobby" in reasoning_text or "cooking" in reasoning_text or "reading" in reasoning_text, \
            f"Reasoning should mention hobby match. Got: {reasoning}"
    
    def test_reasoning_with_budget_optimization(self, reasoning_service):
        """Test that budget optimization is included in reasoning"""
        profile = UserProfile(
            age=25,
            hobbies=["technology"],
            relationship="friend",
            budget=1000.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        # Gift that uses 50% of budget (good value)
        gift = GiftItem(
            id="test456",
            name="Smart Watch",
            category="Electronics",
            price=500.0,
            rating=4.7,
            image_url="https://cdn.trendyol.com/test456.jpg",
            trendyol_url="https://www.trendyol.com/product/test456",
            description="Feature-rich smartwatch",
            tags=["technology", "smart", "modern"],
            age_suitability=(18, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Should mention budget
        assert "budget" in reasoning_text or "50" in reasoning_text or "value" in reasoning_text, \
            f"Reasoning should mention budget optimization. Got: {reasoning}"
    
    def test_reasoning_with_tool_insights(self, reasoning_service):
        """Test that tool insights are integrated into reasoning"""
        profile = UserProfile(
            age=35,
            hobbies=["gardening"],
            relationship="mother",
            budget=300.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        gift = GiftItem(
            id="test789",
            name="Garden Tool Set",
            category="Garden & Outdoor",
            price=200.0,
            rating=4.8,
            image_url="https://cdn.trendyol.com/test789.jpg",
            trendyol_url="https://www.trendyol.com/product/test789",
            description="Complete gardening toolkit",
            tags=["gardening", "outdoor", "practical"],
            age_suitability=(25, 70),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        # Provide tool results
        tool_results = {
            "review_analysis": {
                "average_rating": 4.8,
                "total_reviews": 150
            },
            "inventory_check": {
                "available": [gift]
            }
        }
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results=tool_results
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Should mention tool insights
        assert "rated" in reasoning_text or "stock" in reasoning_text or "4.8" in reasoning_text, \
            f"Reasoning should integrate tool insights. Got: {reasoning}"
    
    def test_reasoning_not_static_template(self, reasoning_service):
        """Test that reasoning is not just static templates"""
        # Generate reasoning for two different profiles with same gift
        profile1 = UserProfile(
            age=25,
            hobbies=["gaming"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=["tech-savvy"]
        )
        
        profile2 = UserProfile(
            age=60,
            hobbies=["reading"],
            relationship="mother",
            budget=200.0,
            occasion="christmas",
            personality_traits=["traditional"]
        )
        
        gift = GiftItem(
            id="test999",
            name="Universal Gift Card",
            category="Gift Cards",
            price=100.0,
            rating=4.0,
            image_url="https://cdn.trendyol.com/test999.jpg",
            trendyol_url="https://www.trendyol.com/product/test999",
            description="Universal gift card",
            tags=["gift", "universal"],
            age_suitability=(18, 100),
            occasion_fit=["birthday", "christmas"],
            in_stock=True
        )
        
        reasoning1 = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile1,
            model_output={},
            tool_results={}
        )
        
        reasoning2 = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile2,
            model_output={},
            tool_results={}
        )
        
        # Reasoning should be different for different profiles
        reasoning1_text = " ".join(reasoning1)
        reasoning2_text = " ".join(reasoning2)
        
        # They should not be identical (dynamic content should differ)
        assert reasoning1_text != reasoning2_text, \
            "Reasoning should be dynamic and differ for different profiles"
        
        # Each should contain profile-specific information
        assert "500" in reasoning1_text or "25" in reasoning1_text, \
            "Reasoning 1 should contain profile1-specific info"
        assert "200" in reasoning2_text or "60" in reasoning2_text, \
            "Reasoning 2 should contain profile2-specific info"



class TestHobbyMatchingExplanation:
    """
    Property 13: Hobby matching explanation
    Validates: Requirements 3.2
    """
    
    @pytest.fixture(scope="class")
    def reasoning_service(self):
        """Create reasoning service instance for testing"""
        return ReasoningService()
    
    @given(
        profile=user_profile_strategy(),
        gift=gift_item_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_hobby_matching_explanation_property(
        self,
        profile: UserProfile,
        gift: GiftItem,
        reasoning_service
    ):
        """
        Feature: model-reasoning-enhancement, Property 13: Hobby matching explanation
        
        For any gift recommendation where hobbies match gift tags, the reasoning 
        should identify which hobbies match and their match degree.
        
        Validates: Requirements 3.2
        """
        # Find matching hobbies
        matching_hobbies = reasoning_service._find_matching_hobbies(gift, profile.hobbies)
        
        # Generate reasoning
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # If there are matching hobbies, reasoning should mention them
        if matching_hobbies:
            # Should mention hobby or hobbies
            assert "hobby" in reasoning_text or "hobbies" in reasoning_text, \
                f"Reasoning should mention hobby match when hobbies match. " \
                f"Matching hobbies: {matching_hobbies}. Reasoning: {reasoning}"
            
            # Should mention at least one of the matching hobbies
            hobby_mentioned = any(hobby.lower() in reasoning_text for hobby in matching_hobbies)
            assert hobby_mentioned, \
                f"Reasoning should mention at least one matching hobby. " \
                f"Matching hobbies: {matching_hobbies}. Reasoning: {reasoning}"
    
    def test_single_hobby_match(self, reasoning_service):
        """Test explanation when single hobby matches"""
        profile = UserProfile(
            age=30,
            hobbies=["cooking", "reading", "sports"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        gift = GiftItem(
            id="test_cooking",
            name="Chef's Knife Set",
            category="Kitchen & Dining",
            price=250.0,
            rating=4.6,
            image_url="https://cdn.trendyol.com/test_cooking.jpg",
            trendyol_url="https://www.trendyol.com/product/test_cooking",
            description="Professional chef knives",
            tags=["cooking", "kitchen", "professional"],
            age_suitability=(25, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Should mention the matching hobby
        assert "cooking" in reasoning_text, \
            f"Should mention matching hobby 'cooking'. Got: {reasoning}"
        assert "hobby" in reasoning_text, \
            f"Should use word 'hobby' for single match. Got: {reasoning}"
    
    def test_multiple_hobby_match(self, reasoning_service):
        """Test explanation when multiple hobbies match"""
        profile = UserProfile(
            age=28,
            hobbies=["technology", "gaming", "music"],
            relationship="friend",
            budget=1500.0,
            occasion="birthday",
            personality_traits=["tech-savvy"]
        )
        
        gift = GiftItem(
            id="test_multi",
            name="Gaming Headset with Music Features",
            category="Gaming",
            price=800.0,
            rating=4.8,
            image_url="https://cdn.trendyol.com/test_multi.jpg",
            trendyol_url="https://www.trendyol.com/product/test_multi",
            description="High-end gaming and music headset",
            tags=["gaming", "music", "technology", "audio"],
            age_suitability=(18, 50),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        reasoning_text = " ".join(reasoning).lower()
        
        # Should mention hobbies (plural) or list multiple hobbies
        hobby_count = sum(1 for hobby in ["technology", "gaming", "music"] if hobby in reasoning_text)
        
        assert hobby_count >= 2 or "hobbies" in reasoning_text, \
            f"Should mention multiple matching hobbies. Got: {reasoning}"
    
    def test_no_hobby_match(self, reasoning_service):
        """Test that reasoning still works when no hobbies match"""
        profile = UserProfile(
            age=40,
            hobbies=["gardening", "reading"],
            relationship="partner",
            budget=600.0,
            occasion="anniversary",
            personality_traits=[]
        )
        
        gift = GiftItem(
            id="test_no_match",
            name="Gaming Console",
            category="Gaming",
            price=500.0,
            rating=4.5,
            image_url="https://cdn.trendyol.com/test_no_match.jpg",
            trendyol_url="https://www.trendyol.com/product/test_no_match",
            description="Latest gaming console",
            tags=["gaming", "entertainment", "technology"],
            age_suitability=(18, 65),
            occasion_fit=["birthday", "anniversary"],
            in_stock=True
        )
        
        reasoning = reasoning_service.generate_gift_reasoning(
            gift=gift,
            user_profile=profile,
            model_output={},
            tool_results={}
        )
        
        # Should still generate reasoning even without hobby match
        assert len(reasoning) > 0, "Should generate reasoning even without hobby match"
        
        # Should not claim hobby match when there isn't one
        reasoning_text = " ".join(reasoning).lower()
        if "hobby" in reasoning_text or "hobbies" in reasoning_text:
            # If it mentions hobbies, should not claim a match
            assert "gardening" not in reasoning_text and "reading" not in reasoning_text, \
                "Should not claim hobby match when hobbies don't match"



class TestConfidenceThresholdDifferentiation:
    """
    Property 30: Confidence threshold differentiation
    Validates: Requirements 6.5
    """
    
    @pytest.fixture(scope="class")
    def reasoning_service(self):
        """Create reasoning service instance for testing"""
        return ReasoningService()
    
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        profile=user_profile_strategy(),
        gift=gift_item_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_threshold_differentiation_property(
        self,
        confidence: float,
        profile: UserProfile,
        gift: GiftItem,
        reasoning_service
    ):
        """
        Feature: model-reasoning-enhancement, Property 30: Confidence threshold differentiation
        
        For any confidence score, the backend should generate different explanations 
        for high (>0.8), medium (0.5-0.8), and low (<0.5) confidence levels.
        
        Validates: Requirements 6.5
        """
        # Generate confidence explanation
        explanation = reasoning_service.explain_confidence_score(
            confidence=confidence,
            gift=gift,
            user_profile=profile,
            model_output={}
        )
        
        # Verify structure
        assert "score" in explanation, "Explanation should have 'score' field"
        assert "level" in explanation, "Explanation should have 'level' field"
        assert "factors" in explanation, "Explanation should have 'factors' field"
        
        # Verify score matches input
        assert abs(explanation["score"] - confidence) < 0.001, \
            f"Explanation score should match input confidence"
        
        # Verify level is correctly assigned based on confidence
        if confidence > 0.8:
            assert explanation["level"] == "high", \
                f"Confidence {confidence:.2f} should be classified as 'high', got '{explanation['level']}'"
        elif confidence > 0.5:
            assert explanation["level"] == "medium", \
                f"Confidence {confidence:.2f} should be classified as 'medium', got '{explanation['level']}'"
        else:
            assert explanation["level"] == "low", \
                f"Confidence {confidence:.2f} should be classified as 'low', got '{explanation['level']}'"
        
        # Verify factors structure
        assert "positive" in explanation["factors"], "Factors should have 'positive' list"
        assert "negative" in explanation["factors"], "Factors should have 'negative' list"
        assert isinstance(explanation["factors"]["positive"], list), \
            "Positive factors should be a list"
        assert isinstance(explanation["factors"]["negative"], list), \
            "Negative factors should be a list"
        
        # Verify that high confidence has more positive factors
        if confidence > 0.8:
            positive_count = len(explanation["factors"]["positive"])
            negative_count = len(explanation["factors"]["negative"])
            assert positive_count > 0, \
                f"High confidence ({confidence:.2f}) should have positive factors"
        
        # Verify that low confidence has negative factors or explanations
        if confidence < 0.5:
            negative_count = len(explanation["factors"]["negative"])
            positive_count = len(explanation["factors"]["positive"])
            # Low confidence should have some explanation (either negative factors or at least some factors)
            assert negative_count > 0 or positive_count > 0, \
                f"Low confidence ({confidence:.2f}) should have explanatory factors"
    
    def test_high_confidence_explanation(self, reasoning_service):
        """Test that high confidence generates appropriate explanation"""
        profile = UserProfile(
            age=30,
            hobbies=["cooking"],
            relationship="mother",
            budget=500.0,
            occasion="birthday",
            personality_traits=["practical"]
        )
        
        # Perfect match gift
        gift = GiftItem(
            id="test_high",
            name="Premium Cookware Set",
            category="Kitchen & Dining",
            price=400.0,  # 80% of budget
            rating=4.9,  # Excellent rating
            image_url="https://cdn.trendyol.com/test_high.jpg",
            trendyol_url="https://www.trendyol.com/product/test_high",
            description="Professional cookware",
            tags=["cooking", "kitchen", "practical"],
            age_suitability=(25, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        explanation = reasoning_service.explain_confidence_score(
            confidence=0.95,
            gift=gift,
            user_profile=profile,
            model_output={}
        )
        
        assert explanation["level"] == "high"
        assert len(explanation["factors"]["positive"]) > 0, \
            "High confidence should have positive factors"
        
        # Check for quality indicators in positive factors
        positive_text = " ".join(explanation["factors"]["positive"]).lower()
        has_quality_indicators = any(
            indicator in positive_text 
            for indicator in ["excellent", "strong", "good", "perfect", "appropriate", "match"]
        )
        assert has_quality_indicators, \
            f"High confidence positive factors should mention quality. Got: {explanation['factors']['positive']}"
    
    def test_medium_confidence_explanation(self, reasoning_service):
        """Test that medium confidence generates balanced explanation"""
        profile = UserProfile(
            age=35,
            hobbies=["reading", "technology"],
            relationship="friend",
            budget=300.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        # Moderate match gift
        gift = GiftItem(
            id="test_medium",
            name="E-Reader Device",
            category="Electronics",
            price=250.0,  # 83% of budget
            rating=3.8,  # Moderate rating
            image_url="https://cdn.trendyol.com/test_medium.jpg",
            trendyol_url="https://www.trendyol.com/product/test_medium",
            description="Digital reading device",
            tags=["reading", "technology", "digital"],
            age_suitability=(20, 70),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        explanation = reasoning_service.explain_confidence_score(
            confidence=0.65,
            gift=gift,
            user_profile=profile,
            model_output={}
        )
        
        assert explanation["level"] == "medium"
        # Medium confidence should have some factors (positive or negative or both)
        total_factors = len(explanation["factors"]["positive"]) + len(explanation["factors"]["negative"])
        assert total_factors > 0, "Medium confidence should have explanatory factors"
    
    def test_low_confidence_explanation(self, reasoning_service):
        """Test that low confidence generates appropriate explanation"""
        profile = UserProfile(
            age=25,
            hobbies=["sports", "fitness"],
            relationship="friend",
            budget=200.0,
            occasion="birthday",
            personality_traits=["active"]
        )
        
        # Poor match gift
        gift = GiftItem(
            id="test_low",
            name="Vintage Book Collection",
            category="Books",
            price=350.0,  # 175% of budget (over budget!)
            rating=2.5,  # Low rating
            image_url="https://cdn.trendyol.com/test_low.jpg",
            trendyol_url="https://www.trendyol.com/product/test_low",
            description="Old book collection",
            tags=["books", "vintage", "collectible"],
            age_suitability=(50, 80),  # Age mismatch
            occasion_fit=["christmas"],  # Occasion mismatch
            in_stock=True
        )
        
        explanation = reasoning_service.explain_confidence_score(
            confidence=0.25,
            gift=gift,
            user_profile=profile,
            model_output={}
        )
        
        assert explanation["level"] == "low"
        assert len(explanation["factors"]["negative"]) > 0, \
            "Low confidence should have negative factors explaining why"
        
        # Check for problem indicators in negative factors
        negative_text = " ".join(explanation["factors"]["negative"]).lower()
        has_problem_indicators = any(
            indicator in negative_text 
            for indicator in ["exceed", "mismatch", "lower", "weak", "limited", "no"]
        )
        assert has_problem_indicators, \
            f"Low confidence negative factors should explain problems. Got: {explanation['factors']['negative']}"
    
    def test_confidence_threshold_boundaries(self, reasoning_service):
        """Test confidence level assignment at threshold boundaries"""
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        gift = GiftItem(
            id="test_boundary",
            name="Test Gift",
            category="Books",
            price=250.0,
            rating=4.0,
            image_url="https://cdn.trendyol.com/test_boundary.jpg",
            trendyol_url="https://www.trendyol.com/product/test_boundary",
            description="Test gift",
            tags=["test"],
            age_suitability=(18, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        # Test boundary values
        test_cases = [
            (0.81, "high"),
            (0.80, "medium"),
            (0.51, "medium"),
            (0.50, "low"),
            (0.49, "low"),
        ]
        
        for confidence, expected_level in test_cases:
            explanation = reasoning_service.explain_confidence_score(
                confidence=confidence,
                gift=gift,
                user_profile=profile,
                model_output={}
            )
            
            assert explanation["level"] == expected_level, \
                f"Confidence {confidence} should be '{expected_level}', got '{explanation['level']}'"
    
    @given(
        confidence1=st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False),
        confidence2=st.floats(min_value=0.51, max_value=0.79, allow_nan=False, allow_infinity=False),
        confidence3=st.floats(min_value=0.81, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_different_levels_have_different_explanations(
        self,
        confidence1: float,
        confidence2: float,
        confidence3: float,
        reasoning_service
    ):
        """Test that different confidence levels produce different explanations"""
        profile = UserProfile(
            age=30,
            hobbies=["reading"],
            relationship="friend",
            budget=500.0,
            occasion="birthday",
            personality_traits=[]
        )
        
        gift = GiftItem(
            id="test_diff",
            name="Test Gift",
            category="Books",
            price=250.0,
            rating=4.0,
            image_url="https://cdn.trendyol.com/test_diff.jpg",
            trendyol_url="https://www.trendyol.com/product/test_diff",
            description="Test gift",
            tags=["test"],
            age_suitability=(18, 65),
            occasion_fit=["birthday"],
            in_stock=True
        )
        
        exp1 = reasoning_service.explain_confidence_score(confidence1, gift, profile, {})
        exp2 = reasoning_service.explain_confidence_score(confidence2, gift, profile, {})
        exp3 = reasoning_service.explain_confidence_score(confidence3, gift, profile, {})
        
        # Verify levels are different
        assert exp1["level"] == "low"
        assert exp2["level"] == "medium"
        assert exp3["level"] == "high"
        
        # Verify that the levels are actually different
        levels = {exp1["level"], exp2["level"], exp3["level"]}
        assert len(levels) == 3, "Should have three different confidence levels"
