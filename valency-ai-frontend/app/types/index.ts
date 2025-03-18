// Valence-Arousal Types
export interface ValenceArousal {
  valence: number;
  arousal: number;
}

// Sentence Analysis Types
export interface SentenceAnalysis {
  sentence: string;
  influenceWeight: number;
  normalizedInfluence: number;
  valenceInfluence: number;
  arousalInfluence: number;
  predictedValence: number;
  predictedArousal: number;
}

// Overall Emotion Analysis Result
export interface EmotionAnalysisResult {
  overallEmotion: ValenceArousal;
  sentenceBreakdown: SentenceAnalysis[];
  timeAnalyzed?: string;
}

// Word-level Analysis for visualization
export interface WordAnalysis {
  word: string;
  valence: number;
  arousal: number;
  startIndex: number;
  endIndex: number;
}

// Enhanced Analysis Result with word-level details
export interface EnhancedAnalysisResult extends EmotionAnalysisResult {
  wordAnalysis: WordAnalysis[];
  recommendedImprovements: string[];
  emotionalArc: Array<{ position: number; valence: number; arousal: number }>;
  key_transitions: Array<{ 
    position: number; 
    description: string;
    severity: 'info' | 'warning' | 'critical' 
  }>;
}

// Pricing Tier
export interface PricingTier {
  name: string;
  price: number;
  features: string[];
  isPopular?: boolean;
  callToAction: string;
}

// Testimonial
export interface Testimonial {
  id: number;
  text: string;
  author: string;
  company: string;
  role: string;
  avatarUrl?: string;
}

// Industry Use Case
export interface IndustryUseCase {
  id: number;
  industry: string;
  title: string;
  description: string;
  benefits: string[];
  iconUrl: string;
}

// Blog Post
export interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  authorName: string;
  publishDate: string;
  readTime: string;
  imageUrl: string;
  slug: string;
}

// API Error
export interface ApiError {
  status: number;
  message: string;
  details?: string;
} 