# """
# Run inference on the paragraph LSTM model with proper model architecture
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import nltk
# from paragraph_model import ParagraphLSTMModel
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict

# def analyze_paragraph(paragraph: str, visualize: bool = True) -> Dict:
#     """
#     Analyze a paragraph using the LSTM model
    
#     Args:
#         paragraph (str): The paragraph to analyze
#         visualize (bool): Whether to generate visualization
        
#     Returns:
#         Dict: Analysis results
#     """
#     # Create model with the correct architecture
#     model = ParagraphLSTMModel(input_size=384, hidden_size=32, num_layers=1, output_size=2)
    
#     # Load the trained model
#     try:
#         model.load_state_dict(torch.load('models/paragraph/paragraph_lstm_model.pth', 
#                                          map_location=torch.device('cpu')))
#         model.eval()
#         print("Model loaded successfully")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return {"error": str(e)}
    
#     # Load the sentence transformer model
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
#     # Split paragraph into sentences
#     sentences = nltk.sent_tokenize(paragraph)
    
#     if len(sentences) == 0:
#         return {"error": "No sentences found in the paragraph"}
    
#     # Get embeddings
#     embeddings = embedding_model.encode(sentences)
    
#     # Convert to tensor
#     embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
#     # Run inference
#     with torch.no_grad():
#         pred, attention = model(embeddings_tensor, return_attention=True)
        
#         # Convert attention to numpy
#         attention = attention.squeeze(2).squeeze(0).numpy()
    
#     # Calculate sentence contributions
#     sentence_contributions = attention  # Scale by number of sentences
    
#     # Print results
#     print(f"Paragraph: {paragraph}")
    
#     # Since we don't have the scalers, we'll just output the raw model prediction
#     print(f"Raw Prediction: {pred.numpy()[0]}")
    
#     print("\nSentence Contributions:")
#     for i, (sentence, contribution) in enumerate(zip(sentences, sentence_contributions)):
#         print(f"S{i+1} ({contribution:.2f}): {sentence}")
    
#     # Visualize if requested
#     if visualize:
#         visualize_analysis(sentences, sentence_contributions, pred.numpy()[0])
    
#     return {
#         "prediction": pred.numpy()[0].tolist(),
#         "sentences": sentences,
#         "sentence_contributions": sentence_contributions.tolist(),
#         "attention_weights": attention.tolist()
#     }

# def visualize_analysis(sentences: List[str], contributions: np.ndarray, prediction: np.ndarray):
#     """
#     Visualize the paragraph analysis with a line graph representation
#     """
#     # Create figure
#     fig = plt.figure(figsize=(12, 9))
#     gs = plt.GridSpec(2, 1, height_ratios=[1, 2])
    
#     # 1. Plot valence-arousal point in the top subplot with a nicer 2D visualization
#     ax1 = fig.add_subplot(gs[0])
    
#     # Create a background that makes it easier to see the emotional space
#     x = np.linspace(-1, 1, 100)
#     y = np.linspace(-1, 1, 100)
#     X, Y = np.meshgrid(x, y)
#     Z = np.exp(-(X**2 + Y**2)/0.5)  # Gaussian distribution for visual appeal
    
#     # Plot the background with a custom colormap
#     cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap (negative to positive)
#     im = ax1.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap=cmap, alpha=0.3)
    
#     # Normalize the prediction values to the [-1, 1] range for visualization
#     norm_val = max(min(prediction[0], 1), -1)  # Limit to [-1, 1]
#     norm_aro = max(min(prediction[1], 1), -1)  # Limit to [-1, 1]
    
#     # Plot the prediction point
#     ax1.scatter(norm_val, norm_aro, c='blue', s=150, marker='o', edgecolor='white', linewidth=2)
    
#     # Add a contour line around the current point to highlight it
#     circle = plt.Circle((norm_val, norm_aro), 0.1, fill=False, color='blue', linestyle='--', alpha=0.7)
#     ax1.add_patch(circle)
    
#     # Add quadrant labels
#     ax1.text(0.7, 0.7, 'Positive\nHigh Arousal', ha='center', fontsize=9, 
#             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
#     ax1.text(-0.7, 0.7, 'Negative\nHigh Arousal', ha='center', fontsize=9, 
#           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
#     ax1.text(0.7, -0.7, 'Positive\nLow Arousal', ha='center', fontsize=9, 
#           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
#     ax1.text(-0.7, -0.7, 'Negative\nLow Arousal', ha='center', fontsize=9, 
#           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
#     # Add axis lines and quadrant divisions
#     ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
#     ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
#     # Set limits, labels, and title
#     ax1.set_xlim(-1, 1)
#     ax1.set_ylim(-1, 1)
#     ax1.set_xlabel('Valence (Negative → Positive)')
#     ax1.set_ylabel('Arousal (Low → High)')
#     ax1.set_title('Paragraph Emotion Analysis', fontsize=12, fontweight='bold')
#     ax1.grid(True, linestyle='--', alpha=0.3)
    
#     # 2. Plot sentence contributions as a line graph
#     ax2 = fig.add_subplot(gs[1])
    
#     # Get the max contribution for scaling
#     max_contrib = max(contributions)
    
#     # Calculate optimum y-scaling to emphasize vertical differences
#     # This determines the height of the tallest peak in the line graph
#     peak_height = 0.8  # 80% of the plot height is the max peak
    
#     # Create x positions for sentences (evenly spaced)
#     x_positions = np.arange(len(sentences))
    
#     # Create a line graph with markers
#     ax2.plot(x_positions, contributions, marker='o', linestyle='-', linewidth=2.5, 
#            color='royalblue', markersize=10, markerfacecolor='white', markeredgewidth=2)
    
#     # Add horizontal gridlines for easier readability of values
#     ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
#     # Fill area under the line
#     ax2.fill_between(x_positions, contributions, alpha=0.2, color='royalblue')
    
#     # Add value annotations above each point
#     for i, contribution in enumerate(contributions):
#         ax2.annotate(f'{contribution:.2f}', 
#                   xy=(i, contribution), 
#                   xytext=(0, 10),  # 10 points vertical offset
#                   textcoords='offset points',
#                   ha='center', 
#                   va='bottom',
#                   fontsize=9,
#                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
#     # Add sentence labels below x-axis
#     for i, sentence in enumerate(sentences):
#         shortened = sentence if len(sentence) < 30 else sentence[:27] + '...'
#         ax2.annotate(f'S{i+1}: {shortened}', 
#                   xy=(i, -0.05 * max_contrib),  # Slightly below the x-axis
#                   xytext=(0, -15),  # 15 points vertical offset
#                   textcoords='offset points',
#                   ha='center', 
#                   va='top',
#                   fontsize=8,
#                   rotation=0,  # No rotation for better readability
#                   bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.7))
    
#     # Set axis properties
#     ax2.set_xticks(x_positions)
#     ax2.set_xticklabels([f'S{i+1}' for i in range(len(sentences))])
    
#     # Set y-axis limits to emphasize vertical differences
#     # Leave 20% of space at the top and 10% at the bottom
#     y_max = max_contrib * 1.2
#     y_min = 0 - (max_contrib * 0.1)  # Negative space for labels
#     ax2.set_ylim(y_min, y_max)
    
#     # Labels and title
#     ax2.set_xlabel('Sentences')
#     ax2.set_ylabel('Contribution Weight')
#     ax2.set_title('Sentence Contribution to Paragraph Emotion', fontsize=12, fontweight='bold')
    
#     # Remove top and right spines for cleaner look
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # Example usage
#     paragraph = "I'm delighted with my new job. The team is so welcoming and supportive. We celebrated a major project success last week. The office has a beautiful view of the park. I'm excited about the future opportunities here."
#     analyze_paragraph(paragraph)
    
#     # Try another example
#     paragraph2 = "This project has been a complete disaster. The team is frustrated and demotivated. The clients are unhappy with our progress. We've been working overtime without any appreciation. The stress levels are through the roof."
#     analyze_paragraph(paragraph2) 