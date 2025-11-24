# Design Document

## Overview

Bu doküman, Trendyol Gift Recommendation sisteminde modelin reasoning süreçlerinin (düşünme adımları, tool seçim mantığı, kategori eşleştirme açıklamaları, attention weights) yakalanması ve kullanıcıya sunulması için teknik tasarımı tanımlar.

### Current State

Mevcut sistemde:
- Model `IntegratedEnhancedTRM` eğitilmiş checkpoint'ten yükleniyor
- Tool execution otomatik gerçekleşiyor (price_comparison, review_analysis, inventory_check, trend_analyzer)
- Tool sonuçları `tool_results` dictionary'sinde dönüyor
- Basit statik reasoning strings oluşturuluyor (örn: "Category match: Kitchen")
- Confidence scores hesaplanıyor ancak açıklanmıyor

### Target State

Hedef sistemde:
- Model forward pass sırasında reasoning trace oluşturacak
- Tool selection reasoning (neden bu toollar seçildi) açıklanacak
- Category matching reasoning (kategoriler nasıl eşleştirildi) detaylandırılacak
- Attention weights (model neye odaklandı) çıkarılacak
- Step-by-step thinking process kaydedilecek
- Backend dinamik, context-aware reasoning üretecek
- API yapılandırılmış reasoning response dönecek
- Frontend reasoning'i görselleştirebilecek

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /api/recommendations?include_reasoning=true         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend Services Layer                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ModelInferenceService                               │   │
│  │  - generate_recommendations()                        │   │
│  │  - _run_inference()                                  │   │
│  │  - _decode_model_output()                            │   │
│  │  + extract_reasoning_trace()        [NEW]            │   │
│  │  + generate_dynamic_reasoning()     [NEW]            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ReasoningService                   [NEW]            │   │
│  │  - generate_tool_selection_reasoning()               │   │
│  │  - generate_category_reasoning()                     │   │
│  │  - generate_gift_reasoning()                         │   │
│  │  - explain_confidence_score()                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  IntegratedEnhancedTRM                               │   │
│  │  - forward_with_enhancements()                       │   │
│  │  + extract_attention_weights()      [NEW]            │   │
│  │  + get_thinking_steps()             [NEW]            │   │
│  │  + explain_tool_selection()         [NEW]            │   │
│  │  + explain_category_matching()      [NEW]            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Request
    │
    ▼
API Endpoint (include_reasoning=true)
    │
    ▼
ModelInferenceService.generate_recommendations()
    │
    ├─► Model.forward_with_enhancements()
    │       │
    │       ├─► encode_user_profile()
    │       │       └─► [Capture: user encoding summary]
    │       │
    │       ├─► enhanced_category_matching()
    │       │       └─► [Capture: category scores + reasoning]
    │       │
    │       ├─► enhanced_tool_selection()
    │       │       └─► [Capture: tool scores + selection reasoning]
    │       │
    │       ├─► execute_tools()
    │       │       └─► [Capture: tool execution results]
    │       │
    │       ├─► cross_modal_fusion()
    │       │       └─► [Capture: attention weights]
    │       │
    │       └─► recommendation_head()
    │               └─► [Capture: final recommendations]
    │
    ├─► ReasoningService.generate_reasoning()
    │       │
    │       ├─► generate_tool_selection_reasoning()
    │       ├─► generate_category_reasoning()
    │       ├─► generate_gift_reasoning()
    │       └─► explain_confidence_score()
    │
    └─► Format Response
            │
            ├─► recommendations: [...]
            ├─► tool_results: {...}
            └─► reasoning_trace: {
                    tool_selection: {...},
                    category_matching: {...},
                    gift_reasoning: [...],
                    attention_weights: {...},
                    thinking_steps: [...],
                    confidence_explanation: {...}
                }
```

## Components and Interfaces

### 1. Model Layer Enhancements

#### 1.1 IntegratedEnhancedTRM Extensions

**New Methods:**

```python
class IntegratedEnhancedTRM(RLEnhancedTRM):
    
    def extract_attention_weights(
        self,
        user_encoding: torch.Tensor,
        gift_encodings: torch.Tensor,
        category_scores: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract attention weights from model components
        
        Returns:
            {
                "user_features": {
                    "hobbies": 0.45,
                    "budget": 0.30,
                    "age": 0.15,
                    "occasion": 0.10
                },
                "gift_features": {
                    "category": 0.40,
                    "price": 0.35,
                    "rating": 0.25
                }
            }
        """
        pass
    
    def get_thinking_steps(
        self,
        env_state: EnvironmentState,
        model_output: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Generate step-by-step thinking process
        
        Returns:
            [
                {
                    "step": 1,
                    "action": "Encode user profile",
                    "result": "User encoding: [0.23, 0.45, ...]",
                    "insight": "Strong cooking interest detected"
                },
                ...
            ]
        """
        pass
    
    def explain_tool_selection(
        self,
        tool_scores: torch.Tensor,
        user_encoding: torch.Tensor,
        selected_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Explain why specific tools were selected
        
        Returns:
            {
                "price_comparison": {
                    "selected": True,
                    "score": 0.85,
                    "reason": "User has strict budget constraint (500 TL)",
                    "confidence": 0.85,
                    "priority": 1,
                    "factors": {
                        "budget_constraint": 0.9,
                        "price_sensitivity": 0.8
                    }
                },
                ...
            }
        """
        pass
    
    def explain_category_matching(
        self,
        category_scores: torch.Tensor,
        user_encoding: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Explain category matching process
        
        Returns:
            {
                "Kitchen": {
                    "score": 0.85,
                    "reasons": [
                        "User hobby: cooking (0.9 match)",
                        "Occasion: birthday (0.7 match)",
                        "Age appropriate: 35 years (0.8 match)"
                    ],
                    "feature_contributions": {
                        "hobby_match": 0.45,
                        "occasion_fit": 0.30,
                        "age_appropriateness": 0.25
                    }
                },
                ...
            }
        """
        pass
```

#### 1.2 Forward Pass Modifications

Mevcut `forward_with_enhancements()` metodunu genişletmek yerine, reasoning extraction için wrapper metod ekleyeceğiz:

```python
def forward_with_reasoning_trace(
    self,
    carry,
    env_state: EnvironmentState,
    available_gifts: List[GiftItem],
    execute_tools: bool = True,
    capture_reasoning: bool = True
) -> Tuple[Any, Dict[str, torch.Tensor], List[str], Dict[str, Any]]:
    """
    Forward pass with reasoning trace capture
    
    Returns:
        (carry, rl_output, selected_tools, reasoning_trace)
    """
    
    reasoning_trace = {}
    
    # Call existing forward_with_enhancements
    carry, rl_output, selected_tools = self.forward_with_enhancements(
        carry, env_state, available_gifts, execute_tools
    )
    
    if capture_reasoning:
        # Extract reasoning components
        reasoning_trace = {
            "tool_selection": self.explain_tool_selection(
                rl_output["tool_scores"],
                self.encode_user_profile(env_state.user_profile),
                selected_tools
            ),
            "category_matching": self.explain_category_matching(
                rl_output["category_scores"],
                self.encode_user_profile(env_state.user_profile)
            ),
            "attention_weights": self.extract_attention_weights(
                self.encode_user_profile(env_state.user_profile),
                None,  # Will be computed internally
                rl_output["category_scores"]
            ),
            "thinking_steps": self.get_thinking_steps(
                env_state,
                rl_output
            )
        }
    
    return carry, rl_output, selected_tools, reasoning_trace
```

### 2. Backend Service Layer

#### 2.1 ReasoningService (New Component)

```python
class ReasoningService:
    """Service for generating human-readable reasoning from model outputs"""
    
    def __init__(self):
        self.hobby_category_map = self._load_hobby_category_map()
        self.occasion_category_map = self._load_occasion_category_map()
    
    def generate_tool_selection_reasoning(
        self,
        tool_selection_trace: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Generate human-readable tool selection reasoning
        
        Args:
            tool_selection_trace: Raw tool selection data from model
            user_profile: User profile
            
        Returns:
            Structured reasoning for each tool
        """
        pass
    
    def generate_category_reasoning(
        self,
        category_trace: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """
        Generate human-readable category matching reasoning
        """
        pass
    
    def generate_gift_reasoning(
        self,
        gift: GiftItem,
        user_profile: UserProfile,
        model_output: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate dynamic, context-aware gift reasoning
        
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Hobby matching
        matching_hobbies = self._find_matching_hobbies(
            gift, user_profile.hobbies
        )
        if matching_hobbies:
            reasoning.append(
                f"Perfect match for your hobbies: {', '.join(matching_hobbies)}"
            )
        
        # Budget optimization
        budget_usage = (gift.price / user_profile.budget) * 100
        if budget_usage < 70:
            reasoning.append(
                f"Great value: Only {budget_usage:.0f}% of your budget"
            )
        elif budget_usage > 95:
            reasoning.append(
                f"Premium choice: Uses {budget_usage:.0f}% of budget"
            )
        
        # Tool insights integration
        if "review_analysis" in tool_results:
            avg_rating = tool_results["review_analysis"].get("average_rating", 0)
            if avg_rating >= 4.0:
                reasoning.append(f"Highly rated: {avg_rating}/5.0 stars")
        
        if "trend_analysis" in tool_results:
            trending = tool_results["trend_analysis"].get("trending", [])
            if gift in trending:
                reasoning.append("Currently trending in this category")
        
        # Age appropriateness
        age_min, age_max = gift.age_suitability
        if age_min <= user_profile.age <= age_max:
            reasoning.append(
                f"Age-appropriate for {user_profile.age} years old"
            )
        
        return reasoning
    
    def explain_confidence_score(
        self,
        confidence: float,
        gift: GiftItem,
        user_profile: UserProfile,
        model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain confidence score
        
        Returns:
            {
                "score": 0.85,
                "level": "high",
                "factors": {
                    "positive": [
                        "Strong category match (0.9)",
                        "Excellent reviews (4.5/5.0)"
                    ],
                    "negative": [
                        "Slightly above typical budget"
                    ]
                }
            }
        """
        pass
```

#### 2.2 ModelInferenceService Extensions

```python
class ModelInferenceService:
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        # ... existing code ...
        self.reasoning_service = ReasoningService()
    
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
        Generate recommendations with optional reasoning
        
        Args:
            include_reasoning: Whether to include reasoning trace
            reasoning_level: "basic", "detailed", or "full"
            
        Returns:
            (recommendations, tool_results, reasoning_trace)
        """
        pass
    
    def _extract_reasoning_trace(
        self,
        model_output: Dict[str, Any],
        user_profile: UserProfile,
        reasoning_level: str
    ) -> Dict[str, Any]:
        """
        Extract and format reasoning trace from model output
        """
        pass
```

## Data Models

### Reasoning Response Schema

```python
class ToolSelectionReasoning(BaseModel):
    """Tool selection reasoning for a single tool"""
    name: str
    selected: bool
    score: float
    reason: str
    confidence: float
    priority: int
    factors: Dict[str, float] = Field(default_factory=dict)

class CategoryMatchingReasoning(BaseModel):
    """Category matching reasoning"""
    category_name: str
    score: float
    reasons: List[str]
    feature_contributions: Dict[str, float]

class AttentionWeights(BaseModel):
    """Attention weights for features"""
    user_features: Dict[str, float]
    gift_features: Dict[str, float]

class ThinkingStep(BaseModel):
    """Single step in thinking process"""
    step: int
    action: str
    result: str
    insight: str

class ConfidenceExplanation(BaseModel):
    """Confidence score explanation"""
    score: float
    level: str  # "high", "medium", "low"
    factors: Dict[str, List[str]]  # "positive", "negative"

class ReasoningTrace(BaseModel):
    """Complete reasoning trace"""
    tool_selection: List[ToolSelectionReasoning]
    category_matching: List[CategoryMatchingReasoning]
    attention_weights: AttentionWeights
    thinking_steps: List[ThinkingStep]
    confidence_explanation: Optional[ConfidenceExplanation] = None

class EnhancedGiftRecommendation(GiftRecommendation):
    """Gift recommendation with enhanced reasoning"""
    reasoning: List[str]  # Dynamic reasoning
    reasoning_trace: Optional[ReasoningTrace] = None  # Detailed trace

class EnhancedRecommendationResponse(BaseModel):
    """Enhanced recommendation response"""
    recommendations: List[EnhancedGiftRecommendation]
    tool_results: Dict[str, Any]
    reasoning_trace: Optional[ReasoningTrace] = None
    inference_time: float
    cache_hit: bool
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Tool Selection Properties

**Property 1: Tool selection reasoning completeness**
*For any* model inference that selects tools, the model should generate reasoning that includes selection reason, confidence score, and priority for each selected tool.
**Validates: Requirements 1.1**

**Property 2: Tool selection factor explanation**
*For any* tool selection reasoning, the explanation should identify which user profile features (budget constraint, quality preference, etc.) influenced the tool selection.
**Validates: Requirements 1.2**

**Property 3: Tool priority ordering**
*For any* inference where multiple tools are selected, the reasoning should include priority ordering and factors affecting that ordering.
**Validates: Requirements 1.3**

**Property 4: Low confidence tool selection explanation**
*For any* tool selection with confidence below 0.5, the reasoning should explain why the confidence is low.
**Validates: Requirements 1.4**

**Property 5: Tool selection JSON schema compliance**
*For any* API response containing tool selection reasoning, the response should conform to the defined JSON schema with required fields (name, reason, confidence, priority).
**Validates: Requirements 1.5, 8.2**

### Category Matching Properties

**Property 6: Category scoring completeness**
*For any* category matching operation, the model should compute a score and contributing factors for each category.
**Validates: Requirements 2.1**

**Property 7: High score category explanation**
*For any* category with score above 0.7, the reasoning should explain which user features (hobby, age, occasion) strongly match that category.
**Validates: Requirements 2.2**

**Property 8: Low score category explanation**
*For any* category with score below 0.3, the reasoning should explain why the score is low (e.g., age mismatch, hobby mismatch).
**Validates: Requirements 2.3**

**Property 9: Top categories minimum count**
*For any* category matching operation, the model should return at least the top 3 categories with their scores.
**Validates: Requirements 2.4**

**Property 10: Category reasoning JSON schema compliance**
*For any* API response containing category reasoning, each category should include category_name, score, and reasons list.
**Validates: Requirements 2.5, 8.3**

**Property 11: Category score and explanation consistency**
*For any* category reasoning, the explanations should be consistent with the numerical scores (high scores should have positive explanations, low scores should have negative explanations).
**Validates: Requirements 9.3**

### Gift Reasoning Properties

**Property 12: Dynamic gift reasoning generation**
*For any* gift recommendation, the backend should generate reasoning based on user profile and model output, not static templates.
**Validates: Requirements 3.1, 3.6**

**Property 13: Hobby matching explanation**
*For any* gift recommendation where hobbies match gift tags, the reasoning should identify which hobbies match and their match degree.
**Validates: Requirements 3.2**

**Property 14: Budget optimization explanation**
*For any* gift recommendation, the reasoning should include the percentage of budget used and value assessment.
**Validates: Requirements 3.3**

**Property 15: Tool insights integration**
*For any* gift recommendation where tool results are available, the reasoning should integrate findings from each tool (rating, trend, availability).
**Validates: Requirements 3.4**

**Property 16: Age appropriateness explanation**
*For any* gift recommendation, the reasoning should explain whether the gift is age-appropriate for the user.
**Validates: Requirements 3.5**

### Attention Weights Properties

**Property 17: User features attention weights generation**
*For any* model inference, the model should compute attention weights for user features (hobbies, budget, age, occasion).
**Validates: Requirements 4.1, 4.2**

**Property 18: Gift features attention weights generation**
*For any* model inference, the model should compute attention weights for gift features (category, price, rating).
**Validates: Requirements 4.1, 4.3**

**Property 19: Attention weights normalization**
*For any* set of attention weights, the sum of all weights should equal 1.0 (±0.01 tolerance) and no weight should be negative.
**Validates: Requirements 4.4**

**Property 20: Attention weights visualization format**
*For any* API response containing attention weights, the weights should be in visualizable format (e.g., percentage values) with user_features and gift_features sections.
**Validates: Requirements 4.5, 8.4**

### Thinking Steps Properties

**Property 21: Major steps recording**
*For any* model inference, the model should create a thinking step record for each major step (encode, match, select, execute, rank).
**Validates: Requirements 5.1**

**Property 22: Thinking step structure**
*For any* thinking step, the record should include step_number, action, result, and insight fields.
**Validates: Requirements 5.2, 8.5**

**Property 23: User encoding insight**
*For any* thinking step that completes user encoding, the step should include an insight about the user profile (e.g., detected interests).
**Validates: Requirements 5.3**

**Property 24: Tool execution insight**
*For any* thinking step that completes tool execution, the step should include a summary of tool results.
**Validates: Requirements 5.4**

**Property 25: Gift ranking insight**
*For any* thinking step that completes gift ranking, the step should explain ranking criteria and top selections.
**Validates: Requirements 5.5**

**Property 26: Chronological step ordering**
*For any* API response containing thinking steps, the steps should be ordered chronologically (step numbers in ascending order).
**Validates: Requirements 5.6**

### Confidence Score Properties

**Property 27: Confidence score explanation**
*For any* confidence score generated by the model, the model should provide an explanation of how the score was calculated.
**Validates: Requirements 6.1**

**Property 28: High confidence factors**
*For any* confidence score above 0.8, the explanation should identify factors that increased the score (strong category match, high rating, etc.).
**Validates: Requirements 6.2**

**Property 29: Low confidence factors**
*For any* confidence score below 0.5, the explanation should identify factors that decreased the score (weak match, limited data, etc.).
**Validates: Requirements 6.3**

**Property 30: Confidence threshold differentiation**
*For any* confidence score, the backend should generate different explanations for high (>0.8), medium (0.5-0.8), and low (<0.5) confidence levels.
**Validates: Requirements 6.5**

### API Schema Properties

**Property 31: Reasoning JSON schema compliance**
*For any* API response with reasoning enabled, the reasoning information should conform to the defined JSON schema.
**Validates: Requirements 8.1**

### Optional Reasoning Properties

**Property 32: Reasoning level support**
*For any* API request with reasoning_level parameter, the backend should support basic, detailed, and full levels with appropriate content for each level.
**Validates: Requirements 10.5**

## Error Handling

### Model Layer Error Handling

1. **Attention Weights Extraction Failures**
   - If attention weights cannot be extracted, return zero weights with warning
   - Log extraction failure for debugging
   - Continue with inference without blocking

2. **Thinking Steps Recording Failures**
   - If a thinking step fails to record, log warning but continue
   - Ensure partial thinking steps are still returned
   - Never block inference due to reasoning capture failure

3. **Tool Selection Explanation Failures**
   - If tool selection reasoning cannot be generated, return basic reasoning
   - Include tool names and scores at minimum
   - Log failure for investigation

### Backend Layer Error Handling

1. **Reasoning Service Failures**
   - If ReasoningService fails, fall back to basic reasoning
   - Return recommendations without detailed reasoning trace
   - Log error with context for debugging

2. **JSON Serialization Failures**
   - If reasoning trace cannot be serialized, return error response
   - Include partial data if possible
   - Provide clear error message to client

3. **Performance Degradation**
   - If reasoning generation exceeds time budget, truncate or summarize
   - Return partial reasoning with indicator
   - Monitor and alert on performance issues

### API Layer Error Handling

1. **Invalid Reasoning Level**
   - If invalid reasoning_level provided, default to "basic"
   - Return warning in response
   - Document valid levels in error message

2. **Schema Validation Failures**
   - If response doesn't match schema, log validation error
   - Attempt to fix common issues (missing fields, wrong types)
   - Return error response if unfixable

## Testing Strategy

### Unit Testing

Unit tests will verify specific examples and edge cases:

1. **Model Layer Unit Tests**
   - Test attention weights extraction with known inputs
   - Test thinking steps generation for each major step
   - Test tool selection explanation with various tool combinations
   - Test category matching explanation with different score ranges
   - Edge case: Empty tool selection
   - Edge case: All categories with low scores
   - Edge case: Missing user profile fields

2. **Backend Layer Unit Tests**
   - Test ReasoningService with various model outputs
   - Test dynamic gift reasoning generation
   - Test confidence score explanation for different levels
   - Test JSON schema compliance
   - Edge case: Missing tool results
   - Edge case: Null or invalid model output
   - Edge case: Very large reasoning traces

3. **API Layer Unit Tests**
   - Test include_reasoning parameter handling
   - Test reasoning_level parameter handling
   - Test response schema validation
   - Edge case: Invalid parameter values
   - Edge case: Missing optional parameters

### Property-Based Testing

Property-based tests will verify universal properties across all inputs using **Hypothesis** (Python property-based testing library). Each test will run a minimum of 100 iterations.

1. **Tool Selection Properties (Properties 1-5)**
   - Generate random user profiles and available gifts
   - Run inference and verify tool selection reasoning completeness
   - Verify JSON schema compliance for all responses
   - **Feature: model-reasoning-enhancement, Property 1: Tool selection reasoning completeness**
   - **Feature: model-reasoning-enhancement, Property 5: Tool selection JSON schema compliance**

2. **Category Matching Properties (Properties 6-11)**
   - Generate random user profiles with various hobbies/ages/occasions
   - Verify category scoring completeness and consistency
   - Verify explanations match score levels (high/low)
   - **Feature: model-reasoning-enhancement, Property 11: Category score and explanation consistency**

3. **Gift Reasoning Properties (Properties 12-16)**
   - Generate random gift recommendations with various attributes
   - Verify reasoning is dynamic (not static templates)
   - Verify hobby matching, budget, and age explanations
   - **Feature: model-reasoning-enhancement, Property 12: Dynamic gift reasoning generation**

4. **Attention Weights Properties (Properties 17-20)**
   - Generate random model outputs with attention weights
   - Verify normalization (sum = 1.0, no negatives)
   - Verify JSON schema compliance
   - **Feature: model-reasoning-enhancement, Property 19: Attention weights normalization**

5. **Thinking Steps Properties (Properties 21-26)**
   - Generate random inference runs
   - Verify all major steps are recorded
   - Verify chronological ordering
   - **Feature: model-reasoning-enhancement, Property 26: Chronological step ordering**

6. **Confidence Score Properties (Properties 27-30)**
   - Generate random confidence scores across full range [0.0, 1.0]
   - Verify explanations exist for all scores
   - Verify different explanations for different threshold levels
   - **Feature: model-reasoning-enhancement, Property 30: Confidence threshold differentiation**

7. **API Schema Properties (Property 31)**
   - Generate random API requests with various parameters
   - Verify all responses conform to JSON schema
   - **Feature: model-reasoning-enhancement, Property 31: Reasoning JSON schema compliance**

### Integration Testing

Integration tests will verify end-to-end flows:

1. **Full Inference with Reasoning**
   - Test complete flow from API request to response
   - Verify all reasoning components are present
   - Verify performance within acceptable limits

2. **Reasoning Level Variations**
   - Test basic, detailed, and full reasoning levels
   - Verify appropriate content for each level
   - Verify performance scales appropriately

3. **Optional Reasoning**
   - Test include_reasoning=false (no reasoning)
   - Test include_reasoning=true (full reasoning)
   - Test default behavior (basic reasoning)

### Performance Testing

Performance tests will ensure reasoning doesn't significantly impact inference time:

1. **Inference Time Comparison**
   - Measure inference time without reasoning
   - Measure inference time with reasoning
   - Verify overhead is less than 10%

2. **Large Trace Handling**
   - Test with many tools selected
   - Test with many categories
   - Verify truncation/summarization works

3. **Concurrent Requests**
   - Test reasoning generation under load
   - Verify no memory leaks
   - Verify consistent performance


## Implementation Details

### Phase 1: Model Layer Enhancements

#### 1.1 Attention Weights Extraction

Mevcut model'de attention weights zaten hesaplanıyor ancak dışarı expose edilmiyor. İhtiyacımız olan:

```python
def extract_attention_weights(self, user_encoding, gift_encodings, category_scores):
    """Extract attention weights from model components"""
    
    # User features attention (from user_profile_encoder)
    # Approximate from encoding magnitudes
    user_features = {
        "hobbies": self._compute_feature_importance(user_encoding, "hobby"),
        "budget": self._compute_feature_importance(user_encoding, "budget"),
        "age": self._compute_feature_importance(user_encoding, "age"),
        "occasion": self._compute_feature_importance(user_encoding, "occasion")
    }
    
    # Normalize to sum to 1.0
    total = sum(user_features.values())
    user_features = {k: v/total for k, v in user_features.items()}
    
    # Gift features attention (from category_attention and gift_projection)
    gift_features = {
        "category": category_scores.std().item(),  # Variance as importance
        "price": 0.35,  # Approximate from model architecture
        "rating": 0.25   # Approximate from model architecture
    }
    
    # Normalize
    total = sum(gift_features.values())
    gift_features = {k: v/total for k, v in gift_features.items()}
    
    return {
        "user_features": user_features,
        "gift_features": gift_features
    }
```

#### 1.2 Thinking Steps Generation

```python
def get_thinking_steps(self, env_state, model_output):
    """Generate step-by-step thinking process"""
    
    steps = []
    
    # Step 1: User encoding
    user_encoding = self.encode_user_profile(env_state.user_profile)
    steps.append({
        "step": 1,
        "action": "Encode user profile",
        "result": f"User encoding shape: {user_encoding.shape}",
        "insight": self._summarize_user_profile(env_state.user_profile)
    })
    
    # Step 2: Category matching
    category_scores = model_output["category_scores"]
    top_categories = self._get_top_categories(category_scores, k=3)
    steps.append({
        "step": 2,
        "action": "Match categories",
        "result": f"Top categories: {', '.join(top_categories)}",
        "insight": f"Identified {len(top_categories)} relevant categories"
    })
    
    # Step 3: Tool selection
    selected_tools = model_output.get("executed_tools", [])
    steps.append({
        "step": 3,
        "action": "Select tools",
        "result": f"Selected tools: {', '.join(selected_tools)}",
        "insight": f"Chose {len(selected_tools)} tools for analysis"
    })
    
    # Step 4: Tool execution
    tool_results = model_output.get("tool_results", {})
    steps.append({
        "step": 4,
        "action": "Execute tools",
        "result": f"Executed {len(tool_results)} tools successfully",
        "insight": self._summarize_tool_results(tool_results)
    })
    
    # Step 5: Gift ranking
    action_probs = model_output["action_probs"]
    top_k = min(5, action_probs.size(-1))
    steps.append({
        "step": 5,
        "action": "Rank gifts",
        "result": f"Ranked top {top_k} gifts",
        "insight": "Selected gifts based on multi-criteria scoring"
    })
    
    return steps
```

#### 1.3 Tool Selection Explanation

```python
def explain_tool_selection(self, tool_scores, user_encoding, selected_tools):
    """Explain why specific tools were selected"""
    
    tool_names = list(self.tool_registry.list_tools())
    explanations = {}
    
    for idx, tool_name in enumerate(tool_names):
        if idx >= tool_scores.size(-1):
            break
            
        score = tool_scores[0, idx].item()
        selected = tool_name in selected_tools
        
        # Determine reason based on tool type and user profile
        reason = self._generate_tool_selection_reason(
            tool_name, score, user_encoding
        )
        
        explanations[tool_name] = {
            "selected": selected,
            "score": score,
            "reason": reason,
            "confidence": score,
            "priority": selected_tools.index(tool_name) + 1 if selected else 0
        }
    
    return explanations
```

### Phase 2: Backend Service Implementation

#### 2.1 ReasoningService Implementation

```python
class ReasoningService:
    """Service for generating human-readable reasoning"""
    
    def generate_gift_reasoning(self, gift, user_profile, model_output, tool_results):
        """Generate dynamic gift reasoning"""
        
        reasoning = []
        
        # 1. Hobby matching
        matching_hobbies = [
            hobby for hobby in user_profile.hobbies 
            if hobby.lower() in [tag.lower() for tag in gift.tags]
        ]
        if matching_hobbies:
            reasoning.append(
                f"Perfect match for your hobbies: {', '.join(matching_hobbies)}"
            )
        
        # 2. Budget optimization
        budget_usage = (gift.price / user_profile.budget) * 100
        if budget_usage < 70:
            reasoning.append(
                f"Great value: Only {budget_usage:.0f}% of your budget"
            )
        elif budget_usage > 95:
            reasoning.append(
                f"Premium choice: Uses {budget_usage:.0f}% of budget"
            )
        else:
            reasoning.append(
                f"Well-balanced: {budget_usage:.0f}% of your budget"
            )
        
        # 3. Tool insights
        if "review_analysis" in tool_results:
            avg_rating = tool_results["review_analysis"].get("average_rating", 0)
            if avg_rating >= 4.0:
                reasoning.append(f"Highly rated: {avg_rating}/5.0 stars")
        
        if "trend_analysis" in tool_results:
            trending = tool_results["trend_analysis"].get("trending", [])
            if any(g.id == gift.id for g in trending):
                reasoning.append("Currently trending in this category")
        
        if "inventory_check" in tool_results:
            available = tool_results["inventory_check"].get("available", [])
            if any(g.id == gift.id for g in available):
                reasoning.append("In stock and ready to ship")
        
        # 4. Age appropriateness
        age_min, age_max = gift.age_suitability
        if age_min <= user_profile.age <= age_max:
            reasoning.append(
                f"Age-appropriate for {user_profile.age} years old"
            )
        
        # 5. Occasion fit
        if user_profile.occasion in gift.occasion_fit:
            reasoning.append(
                f"Perfect for {user_profile.occasion}"
            )
        
        return reasoning
    
    def explain_confidence_score(self, confidence, gift, user_profile, model_output):
        """Explain confidence score"""
        
        level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        positive_factors = []
        negative_factors = []
        
        # Analyze factors
        category_scores = model_output.get("category_scores", torch.tensor([]))
        if category_scores.numel() > 0:
            max_category_score = category_scores.max().item()
            if max_category_score > 0.7:
                positive_factors.append(
                    f"Strong category match ({max_category_score:.2f})"
                )
            elif max_category_score < 0.3:
                negative_factors.append(
                    f"Weak category match ({max_category_score:.2f})"
                )
        
        # Budget analysis
        budget_usage = (gift.price / user_profile.budget) * 100
        if 50 <= budget_usage <= 90:
            positive_factors.append("Price within optimal budget range")
        elif budget_usage > 100:
            negative_factors.append("Price exceeds budget")
        
        # Rating analysis
        if gift.rating >= 4.0:
            positive_factors.append(f"Excellent reviews ({gift.rating}/5.0)")
        elif gift.rating < 3.0:
            negative_factors.append(f"Lower ratings ({gift.rating}/5.0)")
        
        return {
            "score": confidence,
            "level": level,
            "factors": {
                "positive": positive_factors,
                "negative": negative_factors
            }
        }
```

#### 2.2 ModelInferenceService Integration

```python
async def generate_recommendations(
    self,
    user_profile: UserProfile,
    available_gifts: List[GiftItem],
    max_recommendations: int = 5,
    include_reasoning: bool = True,
    reasoning_level: str = "detailed",
    timeout: Optional[float] = None
):
    """Generate recommendations with optional reasoning"""
    
    # Run inference
    recommendations, tool_results = await self._run_inference(
        user_profile, available_gifts, max_recommendations
    )
    
    reasoning_trace = None
    
    if include_reasoning:
        # Extract reasoning from model
        reasoning_trace = self._extract_reasoning_trace(
            recommendations,
            user_profile,
            tool_results,
            reasoning_level
        )
        
        # Enhance recommendations with dynamic reasoning
        for rec in recommendations:
            rec.reasoning = self.reasoning_service.generate_gift_reasoning(
                rec.gift,
                user_profile,
                {},  # model_output
                tool_results
            )
    
    return recommendations, tool_results, reasoning_trace
```

### Phase 3: API Layer Updates

#### 3.1 Enhanced Endpoint

```python
@router.post("/recommendations", response_model=EnhancedRecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    include_reasoning: bool = Query(default=True),
    reasoning_level: str = Query(default="detailed", regex="^(basic|detailed|full)$")
):
    """Get gift recommendations with optional reasoning"""
    
    model_service = get_model_service()
    
    recommendations, tool_results, reasoning_trace = await model_service.generate_recommendations(
        user_profile=request.user_profile,
        available_gifts=await get_available_gifts(),
        max_recommendations=request.max_recommendations,
        include_reasoning=include_reasoning,
        reasoning_level=reasoning_level
    )
    
    return EnhancedRecommendationResponse(
        recommendations=recommendations,
        tool_results=tool_results,
        reasoning_trace=reasoning_trace,
        inference_time=0.0,  # Will be measured
        cache_hit=False
    )
```

### Performance Considerations

1. **Lazy Reasoning Generation**
   - Only generate reasoning when requested
   - Cache reasoning for repeated requests
   - Use async processing for heavy computations

2. **Reasoning Level Optimization**
   - Basic: Only gift reasoning, no trace
   - Detailed: Gift reasoning + tool selection + category matching
   - Full: Everything including attention weights and thinking steps

3. **Memory Management**
   - Limit thinking steps to last N steps
   - Truncate large tool results
   - Compress attention weights to top K features

4. **Caching Strategy**
   - Cache reasoning templates
   - Cache user profile encodings
   - Cache category explanations

## Deployment Considerations

### Feature Flags

```python
# config.py
REASONING_ENABLED = os.getenv("REASONING_ENABLED", "true").lower() == "true"
REASONING_DEFAULT_LEVEL = os.getenv("REASONING_DEFAULT_LEVEL", "detailed")
REASONING_MAX_THINKING_STEPS = int(os.getenv("REASONING_MAX_THINKING_STEPS", "10"))
```

### Monitoring

1. **Metrics to Track**
   - Reasoning generation time
   - Reasoning trace size
   - Reasoning request percentage
   - Error rates by reasoning component

2. **Alerts**
   - Reasoning generation timeout
   - Schema validation failures
   - Performance degradation

### Backward Compatibility

- Old clients without `include_reasoning` parameter will get basic reasoning by default
- Response schema is backward compatible (reasoning fields are optional)
- Existing endpoints continue to work unchanged

## Security Considerations

1. **Information Disclosure**
   - Reasoning should not expose internal model architecture details
   - Tool parameters should be sanitized
   - User data should not leak in reasoning

2. **Input Validation**
   - Validate reasoning_level parameter
   - Sanitize user profile data
   - Limit reasoning trace size

3. **Rate Limiting**
   - Apply stricter rate limits for full reasoning requests
   - Monitor for abuse patterns
   - Implement request throttling

## Documentation Updates

### API Documentation

Update OpenAPI spec with:
- New query parameters (include_reasoning, reasoning_level)
- New response schemas (ReasoningTrace, ToolSelectionReasoning, etc.)
- Examples for each reasoning level

### User Guide

Add section explaining:
- How to request reasoning
- How to interpret reasoning trace
- Performance implications
- Best practices

### Developer Guide

Add section covering:
- How to extend reasoning components
- How to add new reasoning types
- Testing reasoning generation
- Debugging reasoning issues
