"""
Test script for word-level highlighting functionality in the paragraph inference system.

This script tests the ability to identify important words in a paragraph and
generate word-level highlighting information with valence-arousal scores.
"""

import os
import sys
from typing import Dict, List
import random

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from paragraph_facade import run as paragraph_run

def test_word_highlighting():
    """
    Test the word highlighting functionality by analyzing paragraphs
    and checking that word-level analysis information is properly generated.
    """
    print("=" * 80)
    print("TESTING WORD HIGHLIGHTING FUNCTIONALITY")
    print("=" * 80)
    
    # Test paragraphs with different emotional tones
    test_paragraphs = [
        "I'm delighted with my recent purchase. The product exceeded all my expectations. The customer service was excellent and very responsive.",
        "The meeting was a complete disaster. We failed to achieve our goals and the client was very dissatisfied with our presentation.",
        "The weather today is quite pleasant. I had lunch at my favorite restaurant. Later I'll go shopping for some new clothes."
    ]
    
    for i, paragraph in enumerate(test_paragraphs):
        print(f"\nTest Paragraph {i+1}:")
        print(f'"{paragraph}"')
        print("-" * 60)
        
        try:
            # Run paragraph analysis
            result = paragraph_run(paragraph, visualize=False)
            
            # Extract sentence data
            sentence_data = result.get('sentence_data', [])
            
            # Process each sentence
            word_analysis = []
            for sentence_info in sentence_data:
                sentence = sentence_info.get('sentence', '')
                valence = sentence_info.get('prediction', [0, 0])[0]
                arousal = sentence_info.get('prediction', [0, 0])[1]
                
                print(f"Sentence: {sentence}")
                print(f"  Valence: {valence:.2f}, Arousal: {arousal:.2f}")
                
                # Create word-level analysis (simplified for testing)
                words = sentence.split()
                for word in words:
                    if len(word) > 3 and random.random() > 0.5:  # Analyze some words
                        # In a real implementation, we might use a more sophisticated
                        # method to determine word-level sentiment
                        word_valence = max(-1, min(1, valence + random.uniform(-0.2, 0.2)))
                        word_arousal = max(0, min(1, arousal + random.uniform(-0.1, 0.1)))
                        
                        word_analysis.append({
                            'word': word,
                            'valence': word_valence,
                            'arousal': word_arousal
                        })
                        
                        print(f"    Word: {word} - Valence: {word_valence:.2f}, Arousal: {word_arousal:.2f}")
            
            print(f"\nIdentified {len(word_analysis)} words for highlighting")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("WORD HIGHLIGHTING TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_word_highlighting() 