from paragraph_inference import inference, build_paragraph_from_dataset
from typing import Dict, Optional, Union

def analyze_paragraph(paragraph: str, visualize: bool = True) -> Dict:
    """
    Analyze the valence and arousal of a paragraph and show how each sentence contributes.
    
    Args:
        paragraph (str): The paragraph to analyze
        visualize (bool): Whether to generate a visualization of the results
        
    Returns:
        Dict: Analysis results containing paragraph prediction and sentence contributions
    """
    print(f"Analyzing paragraph: \n{paragraph}\n")
    return inference(paragraph, visualize=visualize)

def analyze_from_dataset(num_sentences: int = 5, dataset_path: str = "emobank.csv", 
                         visualize: bool = True) -> Dict:
    """
    Build a paragraph from dataset sentences and analyze it.
    
    Args:
        num_sentences (int): Number of sentences to include in the paragraph
        dataset_path (str): Path to the dataset CSV file
        visualize (bool): Whether to generate a visualization of the results
        
    Returns:
        Dict: Analysis results containing paragraph prediction and sentence contributions
    """
    paragraph = build_paragraph_from_dataset(dataset_path, num_sentences)
    return analyze_paragraph(paragraph, visualize)

def run(paragraph: Optional[str] = None, from_dataset: bool = False, 
        num_sentences: int = 5, visualize: bool = True) -> Dict:
    """
    Main entry point for paragraph analysis.
    
    Args:
        paragraph (str, optional): The paragraph to analyze. If None and from_dataset=True,
                                   a paragraph will be built from the dataset.
        from_dataset (bool): Whether to build a paragraph from the dataset
        num_sentences (int): Number of sentences if building from dataset
        visualize (bool): Whether to generate a visualization of the results
        
    Returns:
        Dict: Analysis results containing paragraph prediction and sentence contributions
    """
    if paragraph is None and from_dataset:
        return analyze_from_dataset(num_sentences, visualize=visualize)
    elif paragraph is not None:
        print("Analyzing paragraph...")
        return analyze_paragraph(paragraph, visualize=visualize)
    else:
        print("Error: Please provide a paragraph or set from_dataset=True")
        return {"error": "No paragraph provided"}

if __name__ == "__main__":
    # Example usage
    # Analyze a custom paragraph
    run("I love spending time with my family. The weather is beautiful today. We should go for a picnic in the park. The children will enjoy playing outside. It's important to enjoy these moments together.")
    
    # Generate a paragraph from the dataset and analyze it
    run(from_dataset=True, num_sentences=4) 