# Design Document

## Overview

Bu doküman, Trendyol Gift Recommendation sisteminde backend'den gelen model reasoning bilgilerinin frontend'de görselleştirilmesi için teknik tasarımı tanımlar.

### Current State

Mevcut frontend'de:
- Hediye önerileri basit kart formatında gösteriliyor
- Sadece hediye bilgileri (isim, fiyat, resim) görüntüleniyor
- Model reasoning bilgileri kullanıcıya sunulmuyor
- Backend'den gelen reasoning trace kullanılmıyor

### Target State

Hedef frontend'de:
- Hediye kartlarında dinamik reasoning açıklamaları gösterilecek
- Detaylı reasoning panel ile tool selection, category matching, attention weights görselleştirilecek
- Interactive chart'lar ile model davranışı anlaşılır hale gelecek
- Thinking steps timeline ile model süreci adım adım gösterilecek
- Confidence indicator ile güven skoru görselleştirilecek
- Responsive ve accessible tasarım sağlanacak
- Export ve karşılaştırma özellikleri eklenecek

## Architecture

### High-Level Component Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Pages Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  RecommendationsPage                                 │   │
│  │  - Fetches recommendations with reasoning            │   │
│  │  - Manages reasoning level state                     │   │
│  │  - Handles comparison mode                           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Components Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GiftRecommendationCard                              │   │
│  │  - Displays gift info                                │   │
│  │  - Shows basic reasoning                             │   │
│  │  - Confidence indicator                              │   │
│  │  - "Show Details" button                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ReasoningPanel (Detailed)                           │   │
│  │  - ToolSelectionCard                                 │   │
│  │  - CategoryMatchingChart                             │   │
│  │  - AttentionWeightsChart                             │   │
│  │  - ThinkingStepsTimeline                             │   │
│  │  - ConfidenceExplanationModal                        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hooks Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  useRecommendations()                                │   │
│  │  - Fetches recommendations from API                  │   │
│  │  - Manages reasoning level                           │   │
│  │  - Handles loading/error states                      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  useReasoningPanel()                                 │   │
│  │  - Manages panel open/close state                    │   │
│  │  - Handles filter selections                         │   │
│  │  - Manages chart type preferences                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  RecommendationService                               │   │
│  │  - fetchRecommendations()                            │   │
│  │  - exportReasoningAsJSON()                           │   │
│  │  - exportReasoningAsPDF()                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Core Components

#### 1.1 GiftRecommendationCard

```typescript
interface GiftRecommendationCardProps {
  gift: GiftItem;
  reasoning: string[];
  confidence: number;
  toolResults?: Record<string, any>;
  onShowDetails: () => void;
  isSelected?: boolean;
  onSelect?: () => void;
}

const GiftRecommendationCard: React.FC<GiftRecommendationCardProps> = ({
  gift,
  reasoning,
  confidence,
  toolResults,
  onShowDetails,
  isSelected,
  onSelect
}) => {
  // Render gift card with basic reasoning
  // Show confidence indicator
  // Display tool insights as icons
  // "Show Details" button
};
```

#### 1.2 ReasoningPanel

```typescript
interface ReasoningPanelProps {
  isOpen: boolean;
  onClose: () => void;
  reasoningTrace: ReasoningTrace;
  gift: GiftItem;
  userProfile: UserProfile;
  activeFilters: ReasoningFilter[];
  onFilterChange: (filters: ReasoningFilter[]) => void;
}

const ReasoningPanel: React.FC<ReasoningPanelProps> = ({
  isOpen,
  onClose,
  reasoningTrace,
  gift,
  userProfile,
  activeFilters,
  onFilterChange
}) => {
  // Render detailed reasoning panel
  // Show filtered sections based on activeFilters
  // Export button
};
```


#### 1.3 ToolSelectionCard

```typescript
interface ToolSelectionCardProps {
  toolSelection: ToolSelectionReasoning[];
}

const ToolSelectionCard: React.FC<ToolSelectionCardProps> = ({
  toolSelection
}) => {
  // Render tool selection list
  // Show selected tools with green checkmark
  // Show unselected tools in gray
  // Display confidence scores
  // Tooltips for reasons and factors
};
```

#### 1.4 CategoryMatchingChart

```typescript
interface CategoryMatchingChartProps {
  categories: CategoryMatchingReasoning[];
  onCategoryClick?: (category: CategoryMatchingReasoning) => void;
}

const CategoryMatchingChart: React.FC<CategoryMatchingChartProps> = ({
  categories,
  onCategoryClick
}) => {
  // Render horizontal bar chart
  // Green bars for high scores (>0.7)
  // Red bars for low scores (<0.3)
  // Yellow bars for medium scores
  // Click to show reasons
};
```

#### 1.5 AttentionWeightsChart

```typescript
interface AttentionWeightsChartProps {
  attentionWeights: AttentionWeights;
  chartType: 'bar' | 'radar';
  onChartTypeChange: (type: 'bar' | 'radar') => void;
}

const AttentionWeightsChart: React.FC<AttentionWeightsChartProps> = ({
  attentionWeights,
  chartType,
  onChartTypeChange
}) => {
  // Render bar chart or radar chart
  // Show user features and gift features
  // Display as percentages
  // Tooltips with full values
  // Chart type toggle button
};
```

#### 1.6 ThinkingStepsTimeline

```typescript
interface ThinkingStepsTimelineProps {
  steps: ThinkingStep[];
  onStepClick?: (step: ThinkingStep) => void;
}

const ThinkingStepsTimeline: React.FC<ThinkingStepsTimelineProps> = ({
  steps,
  onStepClick
}) => {
  // Render vertical timeline
  // Show steps in chronological order
  // Green checkmarks for completed steps
  // Click to expand step details
  // Scrollable if many steps
};
```

#### 1.7 ConfidenceIndicator

```typescript
interface ConfidenceIndicatorProps {
  confidence: number;
  onClick?: () => void;
}

const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  onClick
}) => {
  // Render confidence badge
  // Green for high (>0.8)
  // Yellow for medium (0.5-0.8)
  // Red for low (<0.5)
  // Click to show explanation modal
};
```

#### 1.8 ConfidenceExplanationModal

```typescript
interface ConfidenceExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: ConfidenceExplanation;
}

const ConfidenceExplanationModal: React.FC<ConfidenceExplanationModalProps> = ({
  isOpen,
  onClose,
  explanation
}) => {
  // Render modal with confidence explanation
  // Show positive factors (green)
  // Show negative factors (red)
  // Display confidence level and score
};
```

### 2. Custom Hooks

#### 2.1 useRecommendations

```typescript
interface UseRecommendationsOptions {
  userProfile: UserProfile;
  includeReasoning?: boolean;
  reasoningLevel?: 'basic' | 'detailed' | 'full';
}

interface UseRecommendationsReturn {
  recommendations: EnhancedGiftRecommendation[];
  toolResults: Record<string, any>;
  reasoningTrace: ReasoningTrace | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

const useRecommendations = (
  options: UseRecommendationsOptions
): UseRecommendationsReturn => {
  // Fetch recommendations from API
  // Handle loading and error states
  // Support reasoning levels
  // Cache results
};
```

#### 2.2 useReasoningPanel

```typescript
interface UseReasoningPanelReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  activeFilters: ReasoningFilter[];
  setFilters: (filters: ReasoningFilter[]) => void;
  chartType: 'bar' | 'radar';
  setChartType: (type: 'bar' | 'radar') => void;
}

const useReasoningPanel = (): UseReasoningPanelReturn => {
  // Manage panel state
  // Handle filter selections
  // Manage chart type preference
  // Persist preferences to localStorage
};
```

#### 2.3 useReasoningLevel

```typescript
interface UseReasoningLevelReturn {
  level: 'basic' | 'detailed' | 'full';
  setLevel: (level: 'basic' | 'detailed' | 'full') => void;
}

const useReasoningLevel = (): UseReasoningLevelReturn => {
  // Manage reasoning level preference
  // Persist to localStorage
  // Load on mount
};
```

### 3. Services

#### 3.1 RecommendationService

```typescript
class RecommendationService {
  async fetchRecommendations(
    userProfile: UserProfile,
    options: {
      includeReasoning?: boolean;
      reasoningLevel?: 'basic' | 'detailed' | 'full';
      maxRecommendations?: number;
    }
  ): Promise<EnhancedRecommendationResponse> {
    // Call backend API
    // Handle errors
    // Return typed response
  }

  exportReasoningAsJSON(
    reasoning: ReasoningTrace,
    gift: GiftItem
  ): void {
    // Convert to JSON
    // Trigger download
  }

  async exportReasoningAsPDF(
    reasoning: ReasoningTrace,
    gift: GiftItem
  ): Promise<void> {
    // Generate PDF with charts
    // Trigger download
  }

  copyReasoningLink(
    giftId: string,
    reasoningId: string
  ): Promise<void> {
    // Generate shareable link
    // Copy to clipboard
  }
}
```

## Data Models

### TypeScript Interfaces

```typescript
// Backend response types
interface ToolSelectionReasoning {
  name: string;
  selected: boolean;
  score: number;
  reason: string;
  confidence: number;
  priority: number;
  factors?: Record<string, number>;
}

interface CategoryMatchingReasoning {
  category_name: string;
  score: number;
  reasons: string[];
  feature_contributions: Record<string, number>;
}

interface AttentionWeights {
  user_features: Record<string, number>;
  gift_features: Record<string, number>;
}

interface ThinkingStep {
  step: number;
  action: string;
  result: string;
  insight: string;
}

interface ConfidenceExplanation {
  score: number;
  level: 'high' | 'medium' | 'low';
  factors: {
    positive: string[];
    negative: string[];
  };
}

interface ReasoningTrace {
  tool_selection: ToolSelectionReasoning[];
  category_matching: CategoryMatchingReasoning[];
  attention_weights: AttentionWeights;
  thinking_steps: ThinkingStep[];
  confidence_explanation?: ConfidenceExplanation;
}

interface EnhancedGiftRecommendation {
  gift: GiftItem;
  reasoning: string[];
  confidence: number;
  reasoning_trace?: ReasoningTrace;
}

interface EnhancedRecommendationResponse {
  recommendations: EnhancedGiftRecommendation[];
  tool_results: Record<string, any>;
  reasoning_trace?: ReasoningTrace;
  inference_time: number;
  cache_hit: boolean;
}

// UI state types
type ReasoningFilter = 
  | 'tool_selection' 
  | 'category_matching' 
  | 'attention_weights' 
  | 'thinking_steps';

interface ReasoningPanelState {
  isOpen: boolean;
  activeFilters: ReasoningFilter[];
  chartType: 'bar' | 'radar';
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Gift Card Rendering Properties

**Property 1: Reasoning display completeness**
*For any* gift recommendation with reasoning data, the gift card should display all reasoning strings on the card.
**Validates: Requirements 1.1**

**Property 2: Reasoning factor highlighting**
*For any* reasoning text containing hobby match, budget optimization, or age appropriateness, the frontend should highlight these factors separately.
**Validates: Requirements 1.2**

**Property 3: Tool insights icon rendering**
*For any* gift recommendation with tool insights (rating, trend, availability), the frontend should render corresponding icons.
**Validates: Requirements 1.3**

**Property 4: Expandable reasoning text**
*For any* reasoning text exceeding a threshold length, the frontend should provide an expandable "Show more" button.
**Validates: Requirements 1.4**

**Property 5: Detail panel opening**
*For any* click on reasoning area, the frontend should open the detailed reasoning panel.
**Validates: Requirements 1.5**

### Tool Selection Visualization Properties

**Property 6: Tool information completeness**
*For any* tool in tool selection reasoning, the frontend should display selection status, confidence score, and priority.
**Validates: Requirements 2.2**

**Property 7: Selected tool styling**
*For any* selected tool, the frontend should render it with green color and checkmark icon.
**Validates: Requirements 2.3**

**Property 8: Unselected tool styling**
*For any* unselected tool, the frontend should render it with gray color.
**Validates: Requirements 2.4**

**Property 9: Low confidence tooltip**
*For any* tool with confidence below 0.5, the frontend should display a tooltip explaining the low confidence reason.
**Validates: Requirements 2.5**

**Property 10: Tool hover tooltip**
*For any* tool on hover, the frontend should display a tooltip with selection reason and influencing factors.
**Validates: Requirements 2.6**

### Category Matching Visualization Properties

**Property 11: Minimum category count**
*For any* category matching display, the frontend should show at least the top 3 categories with their scores.
**Validates: Requirements 3.2**

**Property 12: High score category styling**
*For any* category with score above 0.7, the frontend should render it with a green progress bar.
**Validates: Requirements 3.3**

**Property 13: Low score category styling**
*For any* category with score below 0.3, the frontend should render it with a red progress bar.
**Validates: Requirements 3.4**

**Property 14: Category click expansion**
*For any* category click, the frontend should display matching reasons (hobby, age, occasion) as a list.
**Validates: Requirements 3.5**

**Property 15: Score percentage formatting**
*For any* category score display, the frontend should format scores as percentage values.
**Validates: Requirements 3.6**

### Attention Weights Visualization Properties

**Property 16: User features chart rendering**
*For any* user features attention weights, the frontend should render hobbies, budget, age, and occasion weights as a bar chart.
**Validates: Requirements 4.2**

**Property 17: Gift features chart rendering**
*For any* gift features attention weights, the frontend should render category, price, and rating weights as a bar chart.
**Validates: Requirements 4.3**

**Property 18: Weight percentage display**
*For any* attention weight, the frontend should display it as a percentage value.
**Validates: Requirements 4.4**

**Property 19: Weight hover tooltip**
*For any* bar hover, the frontend should display a tooltip with feature name and full value.
**Validates: Requirements 4.5**

**Property 20: Chart type switching**
*For any* chart type change request, the frontend should switch between bar chart and radar chart.
**Validates: Requirements 4.6**

### Thinking Steps Timeline Properties

**Property 21: Chronological step ordering**
*For any* thinking steps display, the frontend should render steps in chronological order on the timeline.
**Validates: Requirements 5.2**

**Property 22: Step information completeness**
*For any* thinking step, the frontend should display step number, action name, result, and insight.
**Validates: Requirements 5.3**

**Property 23: Completed step marking**
*For any* completed step, the frontend should mark it with a green checkmark.
**Validates: Requirements 5.4**

**Property 24: Step click expansion**
*For any* step click, the frontend should display step details in expanded format.
**Validates: Requirements 5.5**

**Property 25: Timeline scrollability**
*For any* timeline with many steps, the frontend should provide a scrollable area.
**Validates: Requirements 5.6**

### Confidence Indicator Properties

**Property 26: Confidence visual display**
*For any* gift recommendation, the frontend should display confidence score with a visual indicator.
**Validates: Requirements 6.1**

**Property 27: High confidence styling**
*For any* confidence score above 0.8, the frontend should display green color and "High Confidence" label.
**Validates: Requirements 6.2**

**Property 28: Medium confidence styling**
*For any* confidence score between 0.5 and 0.8, the frontend should display yellow color and "Medium Confidence" label.
**Validates: Requirements 6.3**

**Property 29: Low confidence styling**
*For any* confidence score below 0.5, the frontend should display red color and "Low Confidence" label.
**Validates: Requirements 6.4**

**Property 30: Confidence explanation modal**
*For any* confidence indicator click, the frontend should display confidence explanation (positive and negative factors) in a modal.
**Validates: Requirements 6.5**

**Property 31: Factor categorization**
*For any* confidence explanation display, the frontend should categorize factors as positive or negative.
**Validates: Requirements 6.6**

### Reasoning Level Management Properties

**Property 32: Panel toggle functionality**
*For any* "Show Detailed Analysis" button click, the frontend should open the detailed reasoning panel.
**Validates: Requirements 7.2**

**Property 33: Panel close button**
*For any* open detailed panel, the frontend should display a "Hide Detailed Analysis" button.
**Validates: Requirements 7.3**

**Property 34: Panel close functionality**
*For any* panel close action, the frontend should return to basic reasoning view.
**Validates: Requirements 7.4**

**Property 35: Reasoning level persistence (round-trip)**
*For any* reasoning level change, the frontend should save it to localStorage and load it on page refresh.
**Validates: Requirements 7.5, 7.6**

### Loading and Error States Properties

**Property 36: Loading state display**
*For any* reasoning data loading, the frontend should display a skeleton loader or spinner.
**Validates: Requirements 9.1**

**Property 37: Error message display**
*For any* API request failure, the frontend should display a user-friendly error message.
**Validates: Requirements 9.2**

**Property 38: Retry button display**
*For any* error state with retry capability, the frontend should display a "Retry" button.
**Validates: Requirements 9.4**

**Property 39: Request cancellation**
*For any* user action during reasoning loading, the frontend should cancel the loading request.
**Validates: Requirements 9.5**

### Responsive Design Properties

**Property 40: Mobile layout adaptation**
*For any* page view on mobile device, the frontend should adapt reasoning components to mobile layout.
**Validates: Requirements 10.1**

**Property 41: Vertical chart layout**
*For any* screen width below 768px, the frontend should display charts in vertical layout.
**Validates: Requirements 10.2**

**Property 42: Full-screen mobile modal**
*For any* detailed panel opening on mobile, the frontend should display it as a full-screen modal.
**Validates: Requirements 10.3**

**Property 43: Swipe gesture support**
*For any* touch gesture on mobile, the frontend should support swipe to close panel.
**Validates: Requirements 10.4**

**Property 44: Touch-friendly tooltips**
*For any* tooltip display on mobile, the frontend should use touch-friendly tooltips.
**Validates: Requirements 10.5**

### Filter Management Properties

**Property 45: Tool selection filter**
*For any* "Only Tool Selection" filter selection, the frontend should display only the tool selection section.
**Validates: Requirements 11.2**

**Property 46: Category matching filter**
*For any* "Only Category Matching" filter selection, the frontend should display only the category matching section.
**Validates: Requirements 11.3**

**Property 47: Attention weights filter**
*For any* "Only Attention Weights" filter selection, the frontend should display only the attention weights section.
**Validates: Requirements 11.4**

**Property 48: Show all filter**
*For any* "Show All" filter selection, the frontend should display all reasoning components.
**Validates: Requirements 11.5**

### Comparison Mode Properties

**Property 49: Compare button display**
*For any* multiple gift selection, the frontend should display a "Compare" button.
**Validates: Requirements 12.1**

**Property 50: Side-by-side comparison**
*For any* active comparison mode, the frontend should display selected gifts' reasoning side by side.
**Validates: Requirements 12.2**

**Property 51: Category score comparison chart**
*For any* category score comparison, the frontend should display scores in the same chart with different colors.
**Validates: Requirements 12.3**

**Property 52: Attention weights overlay**
*For any* attention weights comparison, the frontend should display weights as an overlay chart.
**Validates: Requirements 12.4**

**Property 53: Comparison mode exit**
*For any* comparison close action, the frontend should return to normal view.
**Validates: Requirements 12.5**

### Export Functionality Properties

**Property 54: JSON export**
*For any* JSON export selection, the frontend should download reasoning data in JSON format.
**Validates: Requirements 14.2**

**Property 55: PDF export**
*For any* PDF export selection, the frontend should download reasoning visualizations as PDF.
**Validates: Requirements 14.3**

**Property 56: Share link copy**
*For any* share selection, the frontend should copy reasoning link to clipboard.
**Validates: Requirements 14.4**

**Property 57: Export success message**
*For any* successful export, the frontend should display a success message.
**Validates: Requirements 14.5**

### Accessibility Properties

**Property 58: ARIA labels presence**
*For any* reasoning component, the frontend should include ARIA labels and roles.
**Validates: Requirements 15.1**

**Property 59: Chart alt text**
*For any* chart display, the frontend should provide alt text and descriptions.
**Validates: Requirements 15.2**

**Property 60: Keyboard navigation**
*For any* keyboard navigation usage, the frontend should provide access to all interactive elements.
**Validates: Requirements 15.3**

**Property 61: Screen reader compatibility**
*For any* screen reader usage, the frontend should read reasoning information meaningfully.
**Validates: Requirements 15.4**

**Property 62: Color-blind friendly design**
*For any* color usage, the frontend should also provide visual cues beyond color (patterns, icons).
**Validates: Requirements 15.5**


## Error Handling

### Component Error Boundaries

1. **Reasoning Panel Error Boundary**
   - Wrap ReasoningPanel in error boundary
   - Display fallback UI on component error
   - Log error details for debugging
   - Provide "Retry" button

2. **Chart Rendering Errors**
   - Handle invalid data gracefully
   - Display "Chart unavailable" message
   - Log data validation errors
   - Fallback to table view if possible

3. **API Request Errors**
   - Display user-friendly error messages
   - Differentiate between network errors and server errors
   - Provide retry mechanism
   - Cache last successful response

### Data Validation

1. **Reasoning Trace Validation**
   - Validate structure matches TypeScript interfaces
   - Handle missing optional fields
   - Provide default values for missing data
   - Log validation warnings

2. **Score Range Validation**
   - Ensure confidence scores are between 0 and 1
   - Validate attention weights sum to 1.0
   - Check category scores are non-negative
   - Clamp invalid values to valid range

## Testing Strategy

### Unit Testing

Unit tests will verify specific component behaviors using **Vitest** and **React Testing Library**:

1. **Component Rendering Tests**
   - Test GiftRecommendationCard renders with reasoning
   - Test ToolSelectionCard displays all tools correctly
   - Test CategoryMatchingChart renders bars with correct colors
   - Test AttentionWeightsChart switches between bar and radar
   - Test ThinkingStepsTimeline displays steps chronologically
   - Test ConfidenceIndicator shows correct color and label
   - Edge case: Empty reasoning array
   - Edge case: Missing tool results
   - Edge case: Invalid confidence score

2. **Interaction Tests**
   - Test "Show Details" button opens panel
   - Test category click expands reasons
   - Test tool hover shows tooltip
   - Test chart type toggle switches view
   - Test filter selection shows/hides sections
   - Edge case: Rapid button clicks
   - Edge case: Multiple simultaneous interactions

3. **Hook Tests**
   - Test useRecommendations fetches data correctly
   - Test useReasoningPanel manages state
   - Test useReasoningLevel persists to localStorage
   - Edge case: API timeout
   - Edge case: Invalid localStorage data

### Property-Based Testing

Property-based tests will verify universal properties using **fast-check** (JavaScript property-based testing library). Each test will run a minimum of 100 iterations.

1. **Rendering Properties (Properties 1-5)**
   - Generate random gift recommendations with reasoning
   - Verify all reasoning strings are displayed
   - Verify tool insights render as icons
   - **Feature: frontend-reasoning-visualization, Property 1: Reasoning display completeness**
   - **Feature: frontend-reasoning-visualization, Property 3: Tool insights icon rendering**

2. **Tool Selection Properties (Properties 6-10)**
   - Generate random tool selection data
   - Verify selected tools have green styling
   - Verify unselected tools have gray styling
   - Verify tooltips appear on hover
   - **Feature: frontend-reasoning-visualization, Property 7: Selected tool styling**
   - **Feature: frontend-reasoning-visualization, Property 8: Unselected tool styling**

3. **Category Matching Properties (Properties 11-15)**
   - Generate random category scores
   - Verify at least 3 categories are shown
   - Verify high scores have green bars
   - Verify low scores have red bars
   - **Feature: frontend-reasoning-visualization, Property 11: Minimum category count**
   - **Feature: frontend-reasoning-visualization, Property 12: High score category styling**

4. **Attention Weights Properties (Properties 16-20)**
   - Generate random attention weights
   - Verify all features are displayed
   - Verify weights are shown as percentages
   - Verify chart type switching works
   - **Feature: frontend-reasoning-visualization, Property 18: Weight percentage display**
   - **Feature: frontend-reasoning-visualization, Property 20: Chart type switching**

5. **Thinking Steps Properties (Properties 21-25)**
   - Generate random thinking steps
   - Verify chronological ordering
   - Verify all step information is displayed
   - Verify completed steps have checkmarks
   - **Feature: frontend-reasoning-visualization, Property 21: Chronological step ordering**
   - **Feature: frontend-reasoning-visualization, Property 22: Step information completeness**

6. **Confidence Properties (Properties 26-31)**
   - Generate random confidence scores across [0.0, 1.0]
   - Verify correct color for each range
   - Verify correct label for each range
   - Verify modal opens on click
   - **Feature: frontend-reasoning-visualization, Property 27: High confidence styling**
   - **Feature: frontend-reasoning-visualization, Property 28: Medium confidence styling**
   - **Feature: frontend-reasoning-visualization, Property 29: Low confidence styling**

7. **Persistence Properties (Property 35)**
   - Generate random reasoning levels
   - Save to localStorage and reload
   - Verify round-trip consistency
   - **Feature: frontend-reasoning-visualization, Property 35: Reasoning level persistence (round-trip)**

8. **Responsive Properties (Properties 40-44)**
   - Generate random viewport widths
   - Verify layout adapts correctly
   - Verify mobile-specific features work
   - **Feature: frontend-reasoning-visualization, Property 41: Vertical chart layout**
   - **Feature: frontend-reasoning-visualization, Property 42: Full-screen mobile modal**

9. **Filter Properties (Properties 45-48)**
   - Generate random filter selections
   - Verify correct sections are shown/hidden
   - **Feature: frontend-reasoning-visualization, Property 45: Tool selection filter**
   - **Feature: frontend-reasoning-visualization, Property 48: Show all filter**

10. **Accessibility Properties (Properties 58-62)**
    - Verify ARIA labels exist on all components
    - Verify keyboard navigation works
    - Verify color-blind friendly design
    - **Feature: frontend-reasoning-visualization, Property 58: ARIA labels presence**
    - **Feature: frontend-reasoning-visualization, Property 60: Keyboard navigation**

### Integration Testing

Integration tests will verify end-to-end flows using **Playwright**:

1. **Full Reasoning Flow**
   - Load recommendations page
   - Click "Show Details" on a gift
   - Verify all reasoning sections render
   - Test filter interactions
   - Test export functionality

2. **Comparison Mode Flow**
   - Select multiple gifts
   - Click "Compare" button
   - Verify side-by-side display
   - Test comparison charts
   - Exit comparison mode

3. **Mobile Responsive Flow**
   - Test on mobile viewport
   - Verify responsive layout
   - Test touch gestures
   - Verify full-screen modal

4. **Accessibility Flow**
   - Test keyboard-only navigation
   - Test screen reader compatibility
   - Verify focus management
   - Test color contrast

### Snapshot Testing

Snapshot tests will ensure UI consistency:

1. **Component Snapshots**
   - GiftRecommendationCard with various data
   - ReasoningPanel with all sections
   - ToolSelectionCard with different tool states
   - CategoryMatchingChart with various scores
   - AttentionWeightsChart in both modes
   - ThinkingStepsTimeline with different step counts

## Implementation Details

### Phase 1: Core Components

#### 1.1 GiftRecommendationCard Implementation

```typescript
const GiftRecommendationCard: React.FC<GiftRecommendationCardProps> = ({
  gift,
  reasoning,
  confidence,
  toolResults,
  onShowDetails,
  isSelected,
  onSelect
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const maxReasoningLength = 200;
  
  const shouldShowExpandButton = reasoning.join(' ').length > maxReasoningLength;
  
  return (
    <Card className={cn("gift-card", isSelected && "selected")}>
      <CardHeader>
        <img src={gift.image_url} alt={gift.name} />
        <h3>{gift.name}</h3>
        <ConfidenceIndicator 
          confidence={confidence}
          onClick={() => {/* Open explanation modal */}}
        />
      </CardHeader>
      
      <CardContent>
        <p className="price">{gift.price} TL</p>
        
        <div className="reasoning-section">
          {reasoning.slice(0, isExpanded ? undefined : 2).map((text, idx) => (
            <ReasoningItem key={idx} text={text} />
          ))}
          
          {shouldShowExpandButton && (
            <Button 
              variant="link" 
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? 'Daha az göster' : 'Daha fazla göster'}
            </Button>
          )}
        </div>
        
        {toolResults && (
          <div className="tool-insights">
            {toolResults.review_analysis && (
              <Tooltip content={`Rating: ${toolResults.review_analysis.average_rating}/5.0`}>
                <StarIcon className="text-yellow-500" />
              </Tooltip>
            )}
            {toolResults.trend_analysis?.trending && (
              <Tooltip content="Trending">
                <TrendingUpIcon className="text-green-500" />
              </Tooltip>
            )}
            {toolResults.inventory_check?.available && (
              <Tooltip content="In Stock">
                <CheckCircleIcon className="text-blue-500" />
              </Tooltip>
            )}
          </div>
        )}
      </CardContent>
      
      <CardFooter>
        <Button onClick={onShowDetails} variant="outline">
          Detaylı Analiz Göster
        </Button>
        {onSelect && (
          <Checkbox 
            checked={isSelected}
            onCheckedChange={onSelect}
            aria-label="Select for comparison"
          />
        )}
      </CardFooter>
    </Card>
  );
};
```

#### 1.2 Chart Library Selection

We'll use **Recharts** for data visualization:
- Bar charts for attention weights and category scores
- Radar charts for attention weights alternative view
- Responsive and accessible
- TypeScript support
- Customizable styling


#### 1.3 AttentionWeightsChart Implementation

```typescript
const AttentionWeightsChart: React.FC<AttentionWeightsChartProps> = ({
  attentionWeights,
  chartType,
  onChartTypeChange
}) => {
  const userFeaturesData = Object.entries(attentionWeights.user_features).map(
    ([name, value]) => ({
      name,
      value: value * 100, // Convert to percentage
      fullValue: value
    })
  );
  
  const giftFeaturesData = Object.entries(attentionWeights.gift_features).map(
    ([name, value]) => ({
      name,
      value: value * 100,
      fullValue: value
    })
  );
  
  return (
    <div className="attention-weights-chart">
      <div className="chart-header">
        <h3>Attention Weights</h3>
        <ToggleGroup type="single" value={chartType} onValueChange={onChartTypeChange}>
          <ToggleGroupItem value="bar" aria-label="Bar chart">
            <BarChartIcon />
          </ToggleGroupItem>
          <ToggleGroupItem value="radar" aria-label="Radar chart">
            <RadarChartIcon />
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      
      <div className="chart-section">
        <h4>User Features</h4>
        {chartType === 'bar' ? (
          <BarChart width={400} height={300} data={userFeaturesData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis label={{ value: 'Weight (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="custom-tooltip">
                      <p>{payload[0].payload.name}</p>
                      <p>{payload[0].payload.value.toFixed(1)}%</p>
                      <p>({payload[0].payload.fullValue.toFixed(3)})</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        ) : (
          <RadarChart width={400} height={300} data={userFeaturesData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="name" />
            <PolarRadiusAxis angle={90} domain={[0, 100]} />
            <Radar dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
            <Tooltip />
          </RadarChart>
        )}
      </div>
      
      <div className="chart-section">
        <h4>Gift Features</h4>
        {/* Similar chart for gift features */}
      </div>
    </div>
  );
};
```

#### 1.4 ThinkingStepsTimeline Implementation

```typescript
const ThinkingStepsTimeline: React.FC<ThinkingStepsTimelineProps> = ({
  steps,
  onStepClick
}) => {
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  
  return (
    <div className="thinking-steps-timeline" role="list">
      {steps.map((step) => (
        <div 
          key={step.step}
          className="timeline-item"
          role="listitem"
          onClick={() => {
            setExpandedStep(expandedStep === step.step ? null : step.step);
            onStepClick?.(step);
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              setExpandedStep(expandedStep === step.step ? null : step.step);
              onStepClick?.(step);
            }
          }}
          tabIndex={0}
          aria-expanded={expandedStep === step.step}
        >
          <div className="timeline-marker">
            <CheckCircleIcon className="text-green-500" />
            <span className="step-number">{step.step}</span>
          </div>
          
          <div className="timeline-content">
            <h4>{step.action}</h4>
            <p className="result">{step.result}</p>
            
            {expandedStep === step.step && (
              <div className="expanded-details">
                <p className="insight">
                  <strong>Insight:</strong> {step.insight}
                </p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
```

### Phase 2: State Management

#### 2.1 Reasoning Context

```typescript
interface ReasoningContextValue {
  reasoningLevel: 'basic' | 'detailed' | 'full';
  setReasoningLevel: (level: 'basic' | 'detailed' | 'full') => void;
  selectedGifts: string[];
  toggleGiftSelection: (giftId: string) => void;
  clearSelection: () => void;
  isComparisonMode: boolean;
  setComparisonMode: (enabled: boolean) => void;
}

const ReasoningContext = createContext<ReasoningContextValue | undefined>(undefined);

export const ReasoningProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [reasoningLevel, setReasoningLevel] = useLocalStorage<'basic' | 'detailed' | 'full'>(
    'reasoning-level',
    'detailed'
  );
  
  const [selectedGifts, setSelectedGifts] = useState<string[]>([]);
  const [isComparisonMode, setComparisonMode] = useState(false);
  
  const toggleGiftSelection = (giftId: string) => {
    setSelectedGifts(prev => 
      prev.includes(giftId) 
        ? prev.filter(id => id !== giftId)
        : [...prev, giftId]
    );
  };
  
  const clearSelection = () => {
    setSelectedGifts([]);
    setComparisonMode(false);
  };
  
  return (
    <ReasoningContext.Provider value={{
      reasoningLevel,
      setReasoningLevel,
      selectedGifts,
      toggleGiftSelection,
      clearSelection,
      isComparisonMode,
      setComparisonMode
    }}>
      {children}
    </ReasoningContext.Provider>
  );
};

export const useReasoningContext = () => {
  const context = useContext(ReasoningContext);
  if (!context) {
    throw new Error('useReasoningContext must be used within ReasoningProvider');
  }
  return context;
};
```

### Phase 3: API Integration

#### 3.1 API Client

```typescript
class RecommendationAPIClient {
  private baseURL: string;
  
  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }
  
  async fetchRecommendations(
    userProfile: UserProfile,
    options: {
      includeReasoning?: boolean;
      reasoningLevel?: 'basic' | 'detailed' | 'full';
      maxRecommendations?: number;
    } = {}
  ): Promise<EnhancedRecommendationResponse> {
    const params = new URLSearchParams({
      include_reasoning: String(options.includeReasoning ?? true),
      reasoning_level: options.reasoningLevel ?? 'detailed',
      max_recommendations: String(options.maxRecommendations ?? 5)
    });
    
    const response = await fetch(
      `${this.baseURL}/api/recommendations?${params}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_profile: userProfile })
      }
    );
    
    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }
    
    return response.json();
  }
}
```

### Phase 4: Responsive Design

#### 4.1 Breakpoints

```typescript
const breakpoints = {
  mobile: '(max-width: 767px)',
  tablet: '(min-width: 768px) and (max-width: 1023px)',
  desktop: '(min-width: 1024px)'
};

const useMediaQuery = (query: string): boolean => {
  const [matches, setMatches] = useState(false);
  
  useEffect(() => {
    const media = window.matchMedia(query);
    setMatches(media.matches);
    
    const listener = (e: MediaQueryListEvent) => setMatches(e.matches);
    media.addEventListener('change', listener);
    
    return () => media.removeEventListener('change', listener);
  }, [query]);
  
  return matches;
};

export const useIsMobile = () => useMediaQuery(breakpoints.mobile);
export const useIsTablet = () => useMediaQuery(breakpoints.tablet);
export const useIsDesktop = () => useMediaQuery(breakpoints.desktop);
```

#### 4.2 Mobile-Specific Components

```typescript
const MobileReasoningPanel: React.FC<ReasoningPanelProps> = (props) => {
  return (
    <Sheet open={props.isOpen} onOpenChange={(open) => !open && props.onClose()}>
      <SheetContent side="bottom" className="h-full">
        <SheetHeader>
          <SheetTitle>Detaylı Analiz</SheetTitle>
        </SheetHeader>
        
        <div className="mobile-reasoning-content">
          {/* Render reasoning sections in mobile-optimized layout */}
        </div>
      </SheetContent>
    </Sheet>
  );
};
```

### Phase 5: Accessibility

#### 5.1 ARIA Labels

```typescript
// Example ARIA implementation
<div 
  role="region" 
  aria-label="Gift reasoning information"
  aria-describedby="reasoning-description"
>
  <p id="reasoning-description" className="sr-only">
    This section contains detailed reasoning about why this gift was recommended
  </p>
  
  {/* Reasoning content */}
</div>
```

#### 5.2 Keyboard Navigation

```typescript
const useKeyboardNavigation = (
  items: any[],
  onSelect: (index: number) => void
) => {
  const [focusedIndex, setFocusedIndex] = useState(0);
  
  const handleKeyDown = (e: KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setFocusedIndex(prev => Math.min(prev + 1, items.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setFocusedIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
      case ' ':
        e.preventDefault();
        onSelect(focusedIndex);
        break;
    }
  };
  
  return { focusedIndex, handleKeyDown };
};
```

## Performance Considerations

### 1. Lazy Loading

```typescript
// Lazy load heavy components
const ReasoningPanel = lazy(() => import('./components/ReasoningPanel'));
const AttentionWeightsChart = lazy(() => import('./components/AttentionWeightsChart'));

// Use Suspense for loading states
<Suspense fallback={<Skeleton />}>
  <ReasoningPanel {...props} />
</Suspense>
```

### 2. Memoization

```typescript
// Memoize expensive computations
const processedReasoningData = useMemo(() => {
  return processReasoningTrace(reasoningTrace);
}, [reasoningTrace]);

// Memoize components
const MemoizedToolSelectionCard = memo(ToolSelectionCard);
```

### 3. Virtual Scrolling

```typescript
// Use virtual scrolling for long lists
import { useVirtualizer } from '@tanstack/react-virtual';

const ThinkingStepsTimeline: React.FC<Props> = ({ steps }) => {
  const parentRef = useRef<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: steps.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
  });
  
  return (
    <div ref={parentRef} style={{ height: '400px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map(virtualItem => (
          <div key={virtualItem.key} style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            transform: `translateY(${virtualItem.start}px)`
          }}>
            <TimelineItem step={steps[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
};
```

## Deployment Considerations

### Environment Variables

```typescript
// .env.example
VITE_API_BASE_URL=http://localhost:8000
VITE_ENABLE_REASONING=true
VITE_DEFAULT_REASONING_LEVEL=detailed
VITE_MAX_THINKING_STEPS=10
```

### Build Optimization

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom'],
          'charts': ['recharts'],
          'ui': ['@radix-ui/react-dialog', '@radix-ui/react-tooltip']
        }
      }
    }
  }
});
```

### Feature Flags

```typescript
const useFeatureFlag = (flag: string): boolean => {
  const flags = {
    enableReasoning: import.meta.env.VITE_ENABLE_REASONING === 'true',
    enableComparison: import.meta.env.VITE_ENABLE_COMPARISON === 'true',
    enableExport: import.meta.env.VITE_ENABLE_EXPORT === 'true'
  };
  
  return flags[flag as keyof typeof flags] ?? false;
};
```

## Security Considerations

### 1. XSS Prevention

```typescript
// Sanitize user-generated content
import DOMPurify from 'dompurify';

const SafeReasoningText: React.FC<{ text: string }> = ({ text }) => {
  const sanitized = DOMPurify.sanitize(text);
  return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;
};
```

### 2. API Security

```typescript
// Add CSRF token to requests
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  withCredentials: true,
  headers: {
    'X-CSRF-Token': getCsrfToken()
  }
});
```

## Documentation

### Component Documentation

Each component should have:
- JSDoc comments with usage examples
- Storybook stories for visual documentation
- TypeScript interfaces for props
- Accessibility notes

### Example:

```typescript
/**
 * Displays a gift recommendation card with reasoning information
 * 
 * @example
 * ```tsx
 * <GiftRecommendationCard
 *   gift={giftData}
 *   reasoning={["Perfect for cooking enthusiasts", "Within budget"]}
 *   confidence={0.85}
 *   onShowDetails={() => openPanel()}
 * />
 * ```
 * 
 * @accessibility
 * - Uses ARIA labels for screen readers
 * - Keyboard navigable with Tab and Enter
 * - Color-blind friendly with icons
 */
export const GiftRecommendationCard: React.FC<GiftRecommendationCardProps> = (props) => {
  // Implementation
};
```
