from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import nltk
import math
from typing import List, Tuple, Dict

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    Taken from PyTorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Ensures batch_first=True compatibility.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Create positional encoding matrix pe of shape (max_len, 1, d_model)
        pe = torch.zeros(max_len, d_model) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape (1, max_len, d_model) for batch_first=True
        pe = pe.unsqueeze(0) 

        # Register pe as a buffer, so it's part of the model's state but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) is the sequence length
        # self.pe[:, :x.size(1)] selects pos encodings up to the seq length
        # These are added to the input embeddings x
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ParagraphTransformerModel(nn.Module):
    """
    Transformer model for processing paragraphs as sequences of sentence embeddings
    to predict valence and arousal values. Uses Transformer Encoder layers.
    """
    def __init__(self, input_size=384, d_model=384, nhead=8, num_encoder_layers=3, 
                 dim_feedforward=512, dropout=0.1, output_size=2, max_seq_len=100):
        """
        Args:
            input_size (int): Dimension of the input sentence embeddings.
            d_model (int): The number of expected features in the encoder/decoder inputs (must be divisible by nhead).
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model in nn.TransformerEncoderLayer.
            dropout (float): The dropout value.
            output_size (int): The number of output values (e.g., 2 for valence and arousal).
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        super(ParagraphTransformerModel, self).__init__()
        
        if d_model != input_size:
             # Optional: Linear layer to project input embeddings to d_model if they differ
            self.input_proj = nn.Linear(input_size, d_model)
        else:
            self.input_proj = nn.Identity() # No projection needed

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True  # Important: Input shape (batch, seq, feature)
        )
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Output layer
        # We'll pool the transformer output before the final layer. Mean pooling is common.
        self.fc = nn.Linear(d_model, output_size)
        
        self.d_model = d_model # Store d_model for potential use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            torch.Tensor: Output predictions for valence and arousal (batch_size, output_size)
        """
        # x shape: (batch_size, seq_length, input_size)
        
        # Project input if dimensions don't match d_model
        x = self.input_proj(x) # shape: (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x) # shape: (batch_size, seq_length, d_model)
        
        # Pass through Transformer Encoder
        # No mask needed here unless we want to ignore padding, which isn't handled in this basic example
        transformer_out = self.transformer_encoder(x) # shape: (batch_size, seq_length, d_model)
        
        # Pooling: Average the outputs across the sequence length dimension
        # Alternatives: Use output of a special [CLS] token, max pooling, etc.
        pooled_output = transformer_out.mean(dim=1) # shape: (batch_size, d_model)
        
        # Final prediction layer
        output = self.fc(pooled_output) # shape: (batch_size, output_size)
        
        return output


class ParagraphProcessor:
    """
    Handles processing of paragraphs for the Transformer model, including:
    - Breaking paragraphs into sentences
    - Generating sentence embeddings
    - Handling model inference
    """
    def __init__(self, model_dir='models/paragraph'):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # --- Helper function to find project root (same as before) ---
        def get_project_root():
            # Check if running in a standard script environment
            if '__file__' in globals():
                current_path = Path(__file__).resolve()
                # Traverse up to find a marker like .git or requirements.txt
                for parent in current_path.parents:
                    if (parent / ".git").exists() or (parent / "requirements.txt").exists():
                        return parent
                # Fallback if no marker found, assume script's parent dir
                return current_path.parent  
            else:
                # Fallback for interactive environments (like Jupyter, IPython)
                # Assumes the current working directory is relevant to the project
                return Path.cwd() 

        project_root = get_project_root()
        model_path = project_root / "models" / "paragraph" 
        print("Model Path: ", model_path)        
        self.model_dir = model_path
        
        # Ensure model directory exists 
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load scalers and model if available
        self.feature_scaler = self._load_feature_scaler()
        self.y_scaler = self._load_y_scaler()
        # --- Updated to load Transformer model ---
        self.transformer_model = self._load_transformer_model() 
    
    def _load_feature_scaler(self):
        """Load feature scaler from disk if it exists"""
        try:
            scaler_path = os.path.join(self.model_dir, 'paragraph_feature_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    print("Feature scaler loaded successfully")
                    return pickle.load(f)
            else:
                 print("No feature scaler found, creating new one.")
                 return StandardScaler() # Return new scaler if file doesn't exist
        except Exception as e:
            print(f"Error loading feature scaler: {e}. Creating new one.")
            return StandardScaler()
    
    def _load_y_scaler(self):
        """Load target scaler from disk if it exists"""
        try:
            scaler_path = os.path.join(self.model_dir, 'paragraph_y_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    print("Target scaler loaded successfully")
                    return pickle.load(f)
            else:
                print("No target scaler found, creating new one.")
                return StandardScaler() # Return new scaler if file doesn't exist
        except Exception as e:
            print(f"Error loading target scaler: {e}. Creating new one.")
            return StandardScaler()
    
    # --- Renamed and updated method ---
    def _load_transformer_model(self):
        """Load trained Transformer model from disk if it exists"""
        model_path = os.path.join(self.model_dir, 'paragraph_transformer_model.pth') # Updated filename
        
        # --- Instantiate the new Transformer model ---
        # Ensure parameters match the ones used during training
        # Using defaults here as an example
        model = ParagraphTransformerModel(
            input_size=384,       # Matches SentenceTransformer output
            d_model=384,          # Transformer internal dimension
            nhead=8,              # Number of attention heads (must divide d_model)
            num_encoder_layers=3, # Number of Transformer layers
            dim_feedforward=512, # Feedforward layer dimension
            dropout=0.1,
            output_size=2,        # Predicting valence and arousal
            max_seq_len=100       # Max expected sentences per paragraph (adjust if needed)
        )

        # Load saved parameters if available
        try:
            if os.path.exists(model_path):
                # Load state dict, ensuring compatibility with CPU if needed
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval() # Set model to evaluation mode
                print("Transformer model loaded successfully")
            else:
                print("No pre-trained Transformer model found at", model_path)
                # Depending on use case, might want to raise an error or return None
        except Exception as e:
            print(f"Error loading Transformer model: {e}")
            # Consider how to handle errors - return untrained model or raise?
        
        return model
    
    def split_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into individual sentences"""
        return nltk.sent_tokenize(paragraph)
    
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences"""
        embeddings = self.embedding_model.encode(sentences)
        
        # Apply feature scaling if scaler is trained (has mean_ attribute)
        if hasattr(self.feature_scaler, 'mean_') and self.feature_scaler.mean_ is not None:
            embeddings = self.feature_scaler.transform(embeddings)
            
        return embeddings
    
    def process_paragraph(self, paragraph: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Process a paragraph for prediction
        
        Returns:
            Tuple containing:
            - Scaled Sentence embeddings (seq_length, embedding_dim) - model input
            - List of sentences
            - Raw sentence embeddings before scaling (for analysis)
        """
        # Split paragraph into sentences
        sentences = self.split_into_sentences(paragraph)
        if not sentences: # Handle empty input
            return np.array([]).reshape(0, self.embedding_model.get_sentence_embedding_dimension()), [], np.array([]).reshape(0, self.embedding_model.get_sentence_embedding_dimension())

        # Get raw embeddings for analysis
        raw_embeddings = self.embedding_model.encode(sentences)
        
        # Get scaled embeddings for the model input
        scaled_embeddings = raw_embeddings # Default if scaler not fitted
        if hasattr(self.feature_scaler, 'mean_') and self.feature_scaler.mean_ is not None:
            scaled_embeddings = self.feature_scaler.transform(raw_embeddings)
        
        return scaled_embeddings, sentences, raw_embeddings
    
    # --- Updated method to save the correct model type ---
    def save_models_and_scalers(self, feature_scaler=None, y_scaler=None, transformer_model=None):
        """Save trained models and scalers to disk"""
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
            scaler_path = os.path.join(self.model_dir, 'paragraph_feature_scaler.pkl')
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(feature_scaler, f)
                print(f"Feature scaler saved to {scaler_path}")
            except Exception as e:
                print(f"Error saving feature scaler: {e}")

        if y_scaler is not None:
            self.y_scaler = y_scaler
            scaler_path = os.path.join(self.model_dir, 'paragraph_y_scaler.pkl')
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(y_scaler, f)
                print(f"Target scaler saved to {scaler_path}")
            except Exception as e:
                print(f"Error saving target scaler: {e}")

        # --- Save Transformer model ---
        if transformer_model is not None:
            self.transformer_model = transformer_model
            model_path = os.path.join(self.model_dir, 'paragraph_transformer_model.pth') # Updated filename
            try:
                torch.save(transformer_model.state_dict(), model_path)
                print(f"Transformer model saved to {model_path}")
            except Exception as e:
                print(f"Error saving Transformer model: {e}")
            
# Example Usage (Optional - demonstrates how to use the processor)
if __name__ == '__main__':
    processor = ParagraphProcessor() # Instantiate the processor (loads models/scalers if exist)

    test_paragraph = "This is the first sentence. This is the second sentence, which is a bit longer. Finally, a third sentence."

    # Process the paragraph to get embeddings etc.
    scaled_embeddings, sentences, raw_embeddings = processor.process_paragraph(test_paragraph)

    print("Sentences:", sentences)
    print("Scaled Embeddings Shape:", scaled_embeddings.shape) 
    # Expected: (num_sentences, embedding_dim) e.g., (3, 384)

    # --- Perform Inference ---
    if processor.transformer_model is not None and scaled_embeddings.size > 0:
        # Convert numpy array to torch tensor and add batch dimension
        input_tensor = torch.tensor(scaled_embeddings).unsqueeze(0).float() # Shape: (1, seq_len, input_size)
        
        # Ensure model is in evaluation mode
        processor.transformer_model.eval() 
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            predictions_scaled = processor.transformer_model(input_tensor) # Shape: (1, output_size)
        
        print("Raw Model Output (Scaled):", predictions_scaled.numpy())

        # Inverse transform predictions if y_scaler is fitted
        if hasattr(processor.y_scaler, 'scale_') and processor.y_scaler.scale_ is not None:
            final_predictions = processor.y_scaler.inverse_transform(predictions_scaled.numpy())
            print("Final Predictions (Valence, Arousal):", final_predictions)
        else:
            print("Final Predictions (y_scaler not fitted):", predictions_scaled.numpy())
    
    elif scaled_embeddings.size == 0:
        print("Paragraph was empty or resulted in no sentences.")
    else:
        print("Transformer model not loaded. Cannot perform inference.")

    # --- Example: Saving (if you had trained components) ---
    # Assuming you trained 'new_feature_scaler', 'new_y_scaler', 'trained_transformer_model'
    # processor.save_models_and_scalers(
    #     feature_scaler=new_feature_scaler, 
    #     y_scaler=new_y_scaler, 
    #     transformer_model=trained_transformer_model
    # )