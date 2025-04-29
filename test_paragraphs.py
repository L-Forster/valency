"""
Test the paragraph transformer model with different types of paragraphs
"""

from run_paragraph_inference import analyze_paragraph

# List of test paragraphs with different emotional tones
test_paragraphs = [
    # Positive paragraph
    """I'm absolutely thrilled with our new project. The team has been incredible in their support. 
    We've achieved some amazing results already. The client feedback has been overwhelmingly positive. 
    I'm looking forward to our continued success.""",
    
    # Negative paragraph
    """This has been a terrible experience overall. The product was faulty from the beginning. 
    Customer service was unhelpful and dismissive. The refund process was unnecessarily complicated. 
    I would not recommend this company to anyone.""",
    
    # Neutral/factual paragraph
    """The report contains five main sections. Each section addresses a different aspect of the project. 
    The data was collected over a six-month period. Statistical analysis was performed using standard methods. 
    The conclusion summarizes the key findings.""",
    
    # Mixed emotions paragraph
    """The conference was disappointing in terms of organization. However, the speakers were excellent and informative. 
    We had to deal with terrible accommodation conditions. Despite this, we made some valuable business connections. 
    Overall, it was a challenging but worthwhile experience.""",
    
    # High arousal paragraph
    """The roller coaster ride was exhilarating! My heart was racing as we plummeted down the steep drop. 
    Everyone was screaming and laughing hysterically. The twists and turns were incredibly intense. 
    I was shaking with excitement when it finally ended.""",
    
    # Low arousal paragraph
    """The meditation retreat was peaceful and calming. We spent hours in quiet contemplation each day. 
    The gentle sound of rain outside added to the tranquil atmosphere. The slow, deliberate movements of tai chi centered my thoughts. 
    I felt a deep sense of relaxation and serenity."""
]

def run_tests():
    """Run inference on all test paragraphs"""
    print("=" * 80)
    print("PARAGRAPH EMOTION ANALYSIS TESTS")
    print("=" * 80)
    
    for i, paragraph in enumerate(test_paragraphs):
        print(f"\n\nTEST {i+1}:")
        print("-" * 50)
        analyze_paragraph(paragraph)
        
        # Wait for user input before proceeding to the next test
        if i < len(test_paragraphs) - 1:
            input("\nPress Enter to continue to the next test...")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_tests() 