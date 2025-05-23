# Emotion Analysis with transformers

This project analyzes the emotional tone (valence and arousal) of text using transformer neural networks. It can process both individual sentences and entire paragraphs, showing how different sentences contribute to the overall emotional tone.


## What it will look like once deployed:
![image](https://github.com/user-attachments/assets/393a1d34-a97d-45ea-a224-c0dc633a081d)

![image](https://github.com/user-attachments/assets/4abfcd63-e176-4bf9-b3e8-0e0aba42257e)

![image](https://github.com/user-attachments/assets/c84ff954-8332-42bb-861b-71280e89f5ea)



## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the paragraph model:
```
python train_paragraph_model.py
```

3. Run the demo:
```
python paragraph_demo.py
```

## Features

- **Sentence-level analysis**: Analyzes individual sentences using a pre-trained neural network
- **Paragraph-level analysis**: Uses transformers to model how emotions evolve through paragraphs
- **Attention mechanism**: Identifies which sentences have the most impact on the overall tone
- **Valence-Arousal model**: Measures emotions on two dimensions (positive/negative and calm/excited)
- **Visual output**: Generates visualizations showing sentence contributions to paragraph emotion

## How to Use

### Simple Interface

```python
from paragraph_facade import run

# Analyze a paragraph
run("I love spending time with my family. The weather is beautiful today. We should go for a picnic in the park.")

# Generate a paragraph from the dataset and analyze it
run(from_dataset=True, num_sentences=5)

# Analyze without visualization
run("This is a test paragraph.", visualize=False)
```

### Advanced Usage

```python
from paragraph_inference import inference
from paragraph_model import ParagraphProcessor

# Create a processor for direct access to model components
processor = ParagraphProcessor()

# Process a paragraph
embeddings, sentences, raw_embeddings = processor.process_paragraph("Your paragraph here.")

# Run inference with full control
results = inference("Your paragraph here.", visualize=True)
```

## Model Architecture

The system uses two models:
1. A sentence embedding model (Sentencetransformers) to convert text to vectors
2. A bidirectional transformers with attention to process sequences of sentences

The transformers model architecture includes:
- Input: Sequence of sentence embeddings (384-dimensional vectors)
- Bidirectional transformers layers (2 layers with 64 hidden units)
- Attention mechanism to weight sentence importance
- Output: Valence and arousal values for the paragraph

## Dataset

The models are trained on the EmoBank dataset, which provides valence and arousal annotations for text. The paragraph model is trained by creating synthetic paragraphs from sentences in the dataset, with the paragraph valence-arousal calculated as a weighted average of the sentence values.

## Files

- `paragraph_model.py`: Contains the transformers model architecture and preprocessing
- `paragraph_inference.py`: Handles inference and visualization
- `paragraph_facade.py`: Provides a simple interface for users
- `train_paragraph_model.py`: Script for training the paragraph model
- `paragraph_demo.py`: Demonstration script with examples
