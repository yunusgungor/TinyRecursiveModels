/**
 * TypeScript interfaces for reasoning data models
 * These types match the backend reasoning trace structure
 */

// Tool Selection Reasoning
export interface ToolSelectionReasoning {
  name: string;
  selected: boolean;
  score: number;
  reason: string;
  confidence: number;
  priority: number;
  factors?: Record<string, number>;
}

// Category Matching Reasoning
export interface CategoryMatchingReasoning {
  category_name: string;
  score: number;
  reasons: string[];
  feature_contributions: Record<string, number>;
}

// Attention Weights
export interface AttentionWeights {
  user_features: Record<string, number>;
  gift_features: Record<string, number>;
}

// Thinking Steps
export interface ThinkingStep {
  step: number;
  action: string;
  result: string;
  insight: string;
}

// Confidence Explanation
export interface ConfidenceExplanation {
  score: number;
  level: 'high' | 'medium' | 'low';
  factors: {
    positive: string[];
    negative: string[];
  };
}

// Reasoning Trace (complete reasoning information)
export interface ReasoningTrace {
  tool_selection: ToolSelectionReasoning[];
  category_matching: CategoryMatchingReasoning[];
  attention_weights: AttentionWeights;
  thinking_steps: ThinkingStep[];
  confidence_explanation?: ConfidenceExplanation;
}

// Gift Item
export interface GiftItem {
  id: string;
  name: string;
  price: number;
  image_url?: string;
  category: string;
  rating?: number;
  availability?: boolean;
  description?: string;
}

// Enhanced Gift Recommendation
export interface EnhancedGiftRecommendation {
  gift: GiftItem;
  reasoning: string[];
  confidence: number;
  reasoning_trace?: ReasoningTrace;
}

// Enhanced Recommendation Response
export interface EnhancedRecommendationResponse {
  recommendations: EnhancedGiftRecommendation[];
  tool_results: Record<string, any>;
  reasoning_trace?: ReasoningTrace;
  inference_time: number;
  cache_hit: boolean;
}

// User Profile
export interface UserProfile {
  hobbies?: string[];
  age?: number;
  budget?: number;
  occasion?: string;
  gender?: string;
  relationship?: string;
}

// UI State Types
export type ReasoningFilter = 
  | 'tool_selection' 
  | 'category_matching' 
  | 'attention_weights' 
  | 'thinking_steps';

export interface ReasoningPanelState {
  isOpen: boolean;
  activeFilters: ReasoningFilter[];
  chartType: 'bar' | 'radar';
}

export type ReasoningLevel = 'basic' | 'detailed' | 'full';

// Chart Types
export type ChartType = 'bar' | 'radar';

// Export Options
export type ExportFormat = 'json' | 'pdf' | 'share';
