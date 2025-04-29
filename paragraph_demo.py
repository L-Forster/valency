"""
Paragraph Tone Analysis Demo

This script demonstrates how to use the paragraph analysis
system to analyze the valence-arousal of paragraphs and
visualize how individual sentences contribute to the overall tone.
"""

import sys
import os
from paragraph_facade import run

def demo():
    """
    Run a demonstration of the paragraph analysis system
    """
    print("=" * 80)
    print("PARAGRAPH TONE ANALYSIS DEMO")
    print("=" * 80)
    print("\nThis demo shows how individual sentences contribute to the overall tone of a paragraph.")
    print("The model uses a transformer neural network with attention mechanism to analyze sentence contributions.")
    
    # Example 1: Positive paragraph
    print("\n\nEXAMPLE 1: POSITIVE PARAGRAPH")
    print("-" * 50)
    positive_paragraph = (
        "I'm delighted with my new job. The team is so welcoming and supportive. "
        "We celebrated a major project success last week. "
        "The office has a beautiful view of the park. "
        "I'm excited about the future opportunities here."
    )
    
    run(positive_paragraph)
    
    # Example 2: Negative paragraph
    print("\n\nEXAMPLE 2: NEGATIVE PARAGRAPH")
    print("-" * 50)
    negative_paragraph = (
        "The project failed miserably. Our team was disappointed with the results. "
        "The client rejected our proposal completely. "
        "We've been working overtime with no recognition. "
        "The deadline stress is becoming unbearable."
    )
    
    run(negative_paragraph)
    
    # Example 3: Mixed paragraph
    print("\n\nEXAMPLE 3: MIXED PARAGRAPH")
    print("-" * 50)
    mixed_paragraph = (
        "The weather was awful today. However, I had a wonderful lunch with my friend. "
        "My car broke down on the way home. "
        "Thankfully, a kind stranger stopped to help me. "
        "I missed my favorite show, but I got to catch up on my reading."
    )
    
    run(mixed_paragraph)
    
    # Example 4: From dataset
    print("\n\nEXAMPLE 4: PARAGRAPH FROM DATASET")
    print("-" * 50)
    print("Generating a paragraph from random sentences in the dataset...\n")
    
    run(from_dataset=True, num_sentences=5)
    
    print("\n" + "=" * 80)
    print("END OF DEMO")
    print("=" * 80)
    
    # Interactive mode
    try_interactive = input("\nWould you like to try your own paragraph? (y/n): ")
    if try_interactive.lower() in ['y', 'yes']:
        interactive_demo()

def interactive_demo():
    """
    Run an interactive demonstration where the user can input their own paragraphs
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter a paragraph to analyze (or type 'quit' to exit):")
    
    while True:
        user_input = input("\nYour paragraph: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Paragraph Tone Analyzer!")
            break
        
        if len(user_input.strip()) == 0:
            print("Please enter a paragraph or type 'quit' to exit.")
            continue
        
        run(user_input)

if __name__ == "__main__":
    # Check if models directory exists
    models_dir = "models/paragraph"
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory '{models_dir}' does not exist.")
        print("You may need to train the model first using train_paragraph_model.py")
        
        # Ask to continue anyway
        proceed = input("Continue with demo anyway? (y/n): ")
        if proceed.lower() not in ['y', 'yes']:
            print("Exiting. Please run train_paragraph_model.py first.")
            sys.exit(0)
    
    # Run the demo
    demo() 