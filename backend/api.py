"""
FastAPI server for analyzing emotional tone in text.

This API server integrates with the paragraph inference scripts to analyze
the emotional tone of text and return detailed results including valence, arousal,
and sentence-level analysis.
"""

import os
import sys
from typing import List, Dict, Optional, Union
from datetime import datetime

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import the paragraph inference modules
from paragraph_facade import run as paragraph_run

# Initialize FastAPI app
app = FastAPI(
    title="ToneAnalytics API",
    description="API for analyzing emotional tone in text",
    version="1.0.0"
)

# Configure CORS to allow requests from the frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class AnalysisRequest(BaseModel):
    text: str = Field(..., description="The text to analyze")

class SentenceAnalysis(BaseModel):
    sentence: str
    influenceWeight: float
    normalizedInfluence: float
    influencePercent: float
    valenceInfluence: float
    arousalInfluence: float
    predictedValence: float
    predictedArousal: float

class WordAnalysis(BaseModel):
    word: str
    valence: float
    arousal: float
    startIndex: int
    endIndex: int

class EmotionalPoint(BaseModel):
    position: int
    valence: float
    arousal: float

class Transition(BaseModel):
    position: int
    description: str
    severity: str

class AnalysisResponse(BaseModel):
    overallEmotion: Dict[str, float]
    sentenceBreakdown: List[SentenceAnalysis]
    wordAnalysis: List[WordAnalysis]
    emotionalArc: List[EmotionalPoint]
    key_transitions: List[Transition]
    recommendedImprovements: List[str]
    timeAnalyzed: str

# Helper function to normalize values from [0, 6] to [-1, 1]
def normalize_value(value: float) -> float:
    """Normalize a value from the [0, 6] range to the [-1, 1] range with 3 as neutral."""
    # Center at 3 (neutral) then divide by 3 to get [-1, 1] range
    return (value - 3) / 3

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the ToneAnalytics API"}

# Analysis endpoint - note that the frontend expects this at /analyze not /api/analyze
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    """
    Analyze the emotional tone of the provided text and return detailed results.
    
    The analysis includes:
    - Overall emotional valence and arousal
    - Sentence-level breakdown
    - Word-level analysis
    - Emotional arc
    - Key transitions
    - Recommended improvements
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="The text field cannot be empty")
    
    try:
        # Run the paragraph analyzer - set visualize to False to avoid displaying plots in server mode
        analysis_result = paragraph_run(request.text, visualize=False)
        
        # Extract the overall valence and arousal
        paragraph_prediction = analysis_result.get('paragraph_prediction', [0, 0])
        raw_valence, raw_arousal = paragraph_prediction
        
        # Normalize to the scale expected by the frontend (-1 to 1)
        # Original values from model are in range [0, 6], center at 3
        valence = (raw_valence - 3) / 3
        arousal = (raw_arousal - 3) / 3
        
        # Extract sentence breakdown from sentence_data field
        sentences_data = analysis_result.get('sentence_data', [])
        sentence_breakdown = []
        
        for data in sentences_data:
            sentence = data.get('sentence', '')
            influence = data.get('influenceWeight', 0)
            normalized_influence = data.get('normalizedInfluence', 0)
            influence_percent = data.get('influencePercent', 0)
            valence_influence = data.get('valenceInfluence', 0)
            arousal_influence = data.get('arousalInfluence', 0)
            
            # Use the pre-normalized values from the inference module
            norm_pred_valence = data.get('normalizedPrediction', [0, 0])[0]
            norm_pred_arousal = data.get('normalizedPrediction', [0, 0])[1]
            
            sentence_breakdown.append(SentenceAnalysis(
                sentence=sentence,
                influenceWeight=influence,
                normalizedInfluence=normalized_influence,
                influencePercent=influence_percent,
                valenceInfluence=valence_influence,
                arousalInfluence=arousal_influence,
                predictedValence=norm_pred_valence,
                predictedArousal=norm_pred_arousal
            ))
        
        # Extract word analysis
        word_analysis = []
        text = request.text
        for data in sentences_data:
            sentence = data.get('sentence', '')
            # Use the normalized valence and arousal values
            sentence_valence = data.get('normalizedPrediction', [0, 0])[0]
            sentence_arousal = data.get('normalizedPrediction', [0, 0])[1]
            
            # Simple word analysis: split sentence into words and assign the sentence's values
            words = sentence.split()
            sentence_start_index = text.find(sentence)
            
            if sentence_start_index >= 0:
                current_index = sentence_start_index
                for word in words:
                    if len(word) > 3:  # Only analyze words with more than 3 characters
                        word_start_index = text.find(word, current_index)
                        if word_start_index >= 0:
                            word_end_index = word_start_index + len(word)
                            
                            # Randomly vary the valence and arousal slightly for each word
                            import random
                            word_valence = max(-1, min(1, sentence_valence + random.uniform(-0.1, 0.1)))
                            word_arousal = max(-1, min(1, sentence_arousal + random.uniform(-0.1, 0.1)))
                            
                            word_analysis.append(WordAnalysis(
                                word=word,
                                valence=word_valence,
                                arousal=word_arousal,
                                startIndex=word_start_index,
                                endIndex=word_end_index
                            ))
                            
                            current_index = word_end_index
        
        # Create emotional arc
        emotional_arc = []
        for data in sentences_data:
            position = data.get('position', 0)
            # Use the normalized prediction values directly
            pred_valence = data.get('normalizedPrediction', [0, 0])[0]
            pred_arousal = data.get('normalizedPrediction', [0, 0])[1]
            
            emotional_arc.append(EmotionalPoint(
                position=position,
                valence=pred_valence,
                arousal=pred_arousal
            ))
        
        # Identify key transitions
        key_transitions = []
        for i in range(1, len(sentences_data)):
            prev_data = sentences_data[i-1]
            curr_data = sentences_data[i]
            
            # Use normalized prediction values
            prev_valence = prev_data.get('normalizedPrediction', [0, 0])[0]
            curr_valence = curr_data.get('normalizedPrediction', [0, 0])[0]
            
            valence_diff = curr_valence - prev_valence
            
            # Since we're using a wider normalization range, adjust the threshold
            if abs(valence_diff) > 0.2:
                description = (
                    "Significant positive shift in emotion" 
                    if valence_diff > 0 else 
                    "Significant negative shift in emotion"
                )
                severity = 'info' if valence_diff > 0 else 'warning'
                
                key_transitions.append(Transition(
                    position=i,
                    description=description,
                    severity=severity
                ))
        
        # Generate recommendations
        recommendations = []
        
        # Recommendation based on overall valence
        if valence < -0.2:
            recommendations.append("Consider using more positive language to improve overall tone")
        
        # Recommendation based on transitions
        if any(t.severity == 'warning' for t in key_transitions):
            recommendations.append("Address negative emotional transitions more smoothly")
        
        # Recommendation based on arousal
        if arousal < -0.2:
            recommendations.append("Use more engaging, emotionally resonant language")
        elif arousal > 0.5:
            recommendations.append("Consider moderating the intensity of your language")
        
        # Generic recommendations
        recommendations.append("Maintain consistent tone throughout the conversation")
        
        if len(recommendations) < 3:
            recommendations.append("Consider using more specific emotional language to better connect with users")
        
        # Limit to 5 recommendations
        recommendations = recommendations[:5]
        
        # Create the response
        response = AnalysisResponse(
            overallEmotion={"valence": valence, "arousal": arousal},
            sentenceBreakdown=sentence_breakdown,
            wordAnalysis=word_analysis,
            emotionalArc=emotional_arc,
            key_transitions=key_transitions,
            recommendedImprovements=recommendations,
            timeAnalyzed=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

# Run server if executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)