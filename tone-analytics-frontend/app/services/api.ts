import axios, { AxiosError } from 'axios';
import { EmotionAnalysisResult, EnhancedAnalysisResult, ApiError, WordAnalysis } from '../types';

// Base API URL - would typically come from environment variables
// Updated to match backend routes which are at the root level, not under /api
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API client instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Demo mode flag - set to false by default to use the real backend
let demoMode = false;

export const enableDemoMode = () => {
  demoMode = true;
  console.log('API Service running in demo mode - using generated data');
};

export const disableDemoMode = () => {
  demoMode = false;
  console.log('API Service running in production mode - using neural network analysis');
};

// Check if the API is available
export const checkApiAvailability = async (): Promise<boolean> => {
  try {
    await apiClient.get('/');
    console.log('API server is available');
    disableDemoMode();
    return true;
  } catch (error) {
    console.warn('API server is unreachable:', error);
    enableDemoMode();
    return false;
  }
};

// Sample demo data for when backend is not available
const sampleAnalysisResult: EnhancedAnalysisResult = {
  overallEmotion: { valence: 0.72, arousal: 0.48 },
  sentenceBreakdown: [
    {
      sentence: "I'm really happy with how the interaction went.",
      influenceWeight: 0.35,
      normalizedInfluence: 0.45,
      valenceInfluence: 0.6,
      arousalInfluence: 0.2,
      predictedValence: 0.85,
      predictedArousal: 0.6,
    },
    {
      sentence: "There was a moment of confusion when discussing the return policy.",
      influenceWeight: 0.25,
      normalizedInfluence: 0.3,
      valenceInfluence: 0.3,
      arousalInfluence: 0.7,
      predictedValence: -0.3,
      predictedArousal: 0.65,
    },
    {
      sentence: "But the chatbot quickly clarified things, which was great.",
      influenceWeight: 0.4,
      normalizedInfluence: 0.25,
      valenceInfluence: 0.5,
      arousalInfluence: 0.4,
      predictedValence: 0.7,
      predictedArousal: 0.25,
    },
  ],
  wordAnalysis: [
    { word: "really", valence: 0.8, arousal: 0.6, startIndex: 4, endIndex: 10 },
    { word: "happy", valence: 0.9, arousal: 0.5, startIndex: 11, endIndex: 16 },
    { word: "confusion", valence: -0.4, arousal: 0.7, startIndex: 65, endIndex: 74 },
    { word: "quickly", valence: 0.5, arousal: 0.6, startIndex: 112, endIndex: 119 },
    { word: "great", valence: 0.85, arousal: 0.7, startIndex: 146, endIndex: 151 },
  ],
  emotionalArc: [
    { position: 0, valence: 0.85, arousal: 0.6 },
    { position: 1, valence: -0.3, arousal: 0.65 },
    { position: 2, valence: 0.7, arousal: 0.25 },
  ],
  key_transitions: [
    { position: 1, description: "Emotional drop when discussing the return policy", severity: 'warning' as 'warning' },
    { position: 2, description: "Recovery in positive sentiment after clarification", severity: 'info' as 'info' },
  ],
  recommendedImprovements: [
    "Consider providing clearer information about return policies upfront",
    "Maintain the friendly tone used in the beginning and end of the conversation",
    "The quick recovery after confusion is effective - replicate this pattern"
  ],
  timeAnalyzed: new Date().toISOString(),
};

// Generate a deterministic but varied sample result based on input text
// This helps make the demo mode more realistic by generating different results for different inputs
const generateDemoResult = (text: string): EnhancedAnalysisResult => {
  // Split the text into sentences (simplified)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  if (sentences.length === 0) {
    return sampleAnalysisResult;
  }
  
  // Generate a simple hash of the input text for deterministic randomness
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash) + text.charCodeAt(i);
    hash |= 0; // Convert to 32bit integer
  }
  
  // Seed the pseudo-random number generator
  const pseudoRandom = () => {
    hash = (hash * 9301 + 49297) % 233280;
    return hash / 233280;
  };
  
  // Check for positive/negative sentiment words (very simplified)
  const positiveWords = ['good', 'great', 'happy', 'excellent', 'love', 'appreciate', 'thanks', 'enjoyed', 'helpful'];
  const negativeWords = ['bad', 'terrible', 'sad', 'angry', 'frustrated', 'disappointed', 'annoyed', 'failed', 'issue', 'problem'];
  
  let positiveCount = 0;
  let negativeCount = 0;
  
  // Count positive and negative words
  const wordsInText = text.toLowerCase().split(/\s+/);
  wordsInText.forEach(word => {
    if (positiveWords.some(pos => word.includes(pos))) positiveCount++;
    if (negativeWords.some(neg => word.includes(neg))) negativeCount++;
  });
  
  // Calculate overall sentiment based on word counts
  let baseValence = 0.5;
  if (positiveCount > 0 || negativeCount > 0) {
    baseValence = 0.5 + (0.5 * (positiveCount - negativeCount) / (positiveCount + negativeCount));
  }
  
  // Add some randomness
  const overallValence = Math.max(-1, Math.min(1, baseValence + (pseudoRandom() * 0.3 - 0.15)));
  const overallArousal = 0.3 + pseudoRandom() * 0.4; // Most text is in the mid range for arousal
  
  // Generate sentence breakdown
  const sentenceBreakdown = sentences.map((sentence, i) => {
    // Analyze each sentence
    const words = sentence.toLowerCase().split(/\s+/);
    const posCount = words.filter(word => positiveWords.some(pos => word.includes(pos))).length;
    const negCount = words.filter(word => negativeWords.some(neg => word.includes(neg))).length;
    
    // Calculate sentence sentiment
    let sentValence = 0.5;
    if (posCount > 0 || negCount > 0) {
      sentValence = 0.5 + (0.5 * (posCount - negCount) / Math.max(1, posCount + negCount));
    }
    
    // Add some randomness
    const valence = Math.max(-1, Math.min(1, sentValence + (pseudoRandom() * 0.4 - 0.2)));
    const arousal = 0.2 + pseudoRandom() * 0.6;
    
    // Calculate influence factors
    const influenceWeight = 1 / sentences.length + (pseudoRandom() * 0.1);
    const normalizedInfluence = influenceWeight / sentences.length;
    
    return {
      sentence,
      influenceWeight,
      normalizedInfluence,
      valenceInfluence: influenceWeight * valence,
      arousalInfluence: influenceWeight * arousal,
      predictedValence: valence,
      predictedArousal: arousal,
    };
  });
  
  // Generate word analysis
  const wordAnalysis: WordAnalysis[] = [];
  let startIndex = 0;
  
  for (const sentence of sentences) {
    const words = sentence.split(/\s+/);
    const sentenceIndex = sentences.indexOf(sentence);
    const sentimentInfo = sentenceBreakdown[sentenceIndex];
    
    for (const word of words) {
      if (word.length > 3 && pseudoRandom() > 0.7) {
        const isPositive = positiveWords.some(pos => word.toLowerCase().includes(pos));
        const isNegative = negativeWords.some(neg => word.toLowerCase().includes(neg));
        
        let wordValence = sentimentInfo.predictedValence;
        if (isPositive) wordValence = Math.min(1, wordValence + 0.3);
        if (isNegative) wordValence = Math.max(-1, wordValence - 0.3);
        
        const wordIndex = text.indexOf(word, startIndex);
        if (wordIndex >= 0) {
          wordAnalysis.push({
            word,
            valence: wordValence,
            arousal: sentimentInfo.predictedArousal * (0.8 + pseudoRandom() * 0.4),
            startIndex: wordIndex,
            endIndex: wordIndex + word.length,
          });
        }
      }
      startIndex += word.length + 1; // +1 for space
    }
  }
  
  // Generate emotional arc
  const emotionalArc = sentenceBreakdown.map((sentence, index) => ({
    position: index,
    valence: sentence.predictedValence,
    arousal: sentence.predictedArousal,
  }));
  
  // Generate key transitions
  const key_transitions = [];
  for (let i = 1; i < sentenceBreakdown.length; i++) {
    const prev = sentenceBreakdown[i-1];
    const curr = sentenceBreakdown[i];
    
    const valenceDiff = curr.predictedValence - prev.predictedValence;
    
    if (Math.abs(valenceDiff) > 0.3) {
      key_transitions.push({
        position: i,
        description: valenceDiff > 0 
          ? "Significant positive shift in emotion" 
          : "Significant negative shift in emotion",
        severity: valenceDiff > 0 ? 'info' as 'info' : 'warning' as 'warning',
      });
    }
  }
  
  // Generate custom recommendations
  const recommendedImprovements = [];
  
  if (overallValence < 0.3) {
    recommendedImprovements.push("Consider using more positive language to improve overall tone");
  }
  
  if (key_transitions.some(t => t.severity === 'warning')) {
    recommendedImprovements.push("Address negative emotional transitions more smoothly");
  }
  
  if (overallArousal < 0.3) {
    recommendedImprovements.push("Use more engaging, emotionally resonant language");
  }
  
  recommendedImprovements.push("Maintain consistent tone throughout the conversation");
  
  if (recommendedImprovements.length < 3) {
    recommendedImprovements.push("Consider using more specific emotional language to better connect with users");
  }
  
  return {
    overallEmotion: { valence: overallValence, arousal: overallArousal },
    sentenceBreakdown,
    wordAnalysis,
    emotionalArc,
    key_transitions,
    recommendedImprovements,
    timeAnalyzed: new Date().toISOString(),
  };
};

// API functions
export const analyzeText = async (text: string): Promise<EnhancedAnalysisResult> => {
  try {
    if (demoMode) {
      // In demo mode, return a generated result after a simulated delay
      console.log('Using demo mode for text analysis');
      await new Promise(resolve => setTimeout(resolve, 1500));
      return generateDemoResult(text);
    }

    // In production mode, call the actual API
    console.log('Calling neural network API to analyze text...');
    
    const response = await apiClient.post<EnhancedAnalysisResult>('/analyze', { text });
    console.log('Neural network analysis complete');
    
    return response.data;
    
  } catch (error) {
    console.error('Error analyzing text:', error);
    
    // Try to handle the error gracefully
    const axiosError = error as AxiosError<{ message: string, detail: string }>;
    
    // If the server is unreachable, switch to demo mode
    if (axiosError.code === 'ERR_NETWORK' || axiosError.code === 'ECONNREFUSED') {
      console.warn('API server is unreachable. Switching to demo mode.');
      enableDemoMode();
      return generateDemoResult(text);
    }
    
    // Construct an API error
    const apiError: ApiError = {
      status: axiosError?.response?.status || 500,
      message: axiosError?.response?.data?.message || 'An error occurred while analyzing the text',
      details: axiosError?.response?.data?.detail || (error as Error).message,
    };
    
    throw apiError;
  }
}; 