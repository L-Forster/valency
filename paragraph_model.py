from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import nltk
from typing import List, Tuple, Dict

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ParagraphLSTMModel(nn.Module):
    """
    LSTM model for processing paragraphs as sequences of sentence embeddings
    to predict valence and arousal values.
    """
    def __init__(self, input_size=384, hidden_size=32, num_layers=1, output_size=2):
        super(ParagraphLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the LSTM model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Output predictions for valence and arousal
            torch.Tensor (optional): Attention weights for each sentence
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        
        # LSTM output: (batch_size, seq_length, hidden_size*2) for bidirectional
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention to get context vector
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        
        if return_attention:
            return output, attention_weights
        return output


class ParagraphProcessor:
    """
    Handles processing of paragraphs for the LSTM model, including:
    - Breaking paragraphs into sentences
    - Generating sentence embeddings
    - Handling model inference
    """
    def __init__(self, model_dir='models/paragraph'):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        def get_project_root():
            current_path = Path(__file__).resolve()
            for parent in current_path.parents:
                if (parent / ".git").exists() or (parent / "requirements.txt").exists():
                    return parent
            return current_path.parent  # Fallback

        project_root = get_project_root()
        model_path = project_root / "models" / "paragraph" 
        print("Model Path: ",model_path)        
        self.model_dir = model_path

        
        # Ensure model directory exists 
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load scalers and model if available
        self.feature_scaler = self._load_feature_scaler()
        self.y_scaler = self._load_y_scaler()
        self.lstm_model = self._load_lstm_model()
    
    def _load_feature_scaler(self):
        """Load feature scaler from disk if it exists"""
        try:
            scaler_path = os.path.join(self.model_dir, 'paragraph_feature_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading feature scaler: {e}")
        
        # Return a new scaler if none exists
        return StandardScaler()
    
    def _load_y_scaler(self):
        """Load target scaler from disk if it exists"""
        try:
            scaler_path = os.path.join(self.model_dir, 'paragraph_y_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading target scaler: {e}")
        
        # Return a new scaler if none exists
        return StandardScaler()
    
    def _load_lstm_model(self):
        """Load trained LSTM model from disk if it exists"""
        model_path = os.path.join(self.model_dir, 'paragraph_lstm_model.pth')
        from pathlib import Path
        # Create a new model
        model = ParagraphLSTMModel(input_size=384, hidden_size=32, num_layers=1, output_size=2)

        # Load saved parameters if available
        try:
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                print("LSTM model loaded successfully")
            else:
                print("No LSTM model found")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
        
        return model
    
    def split_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into individual sentences"""
        return nltk.sent_tokenize(paragraph)
    
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences"""
        embeddings = self.embedding_model.encode(sentences)
        
        # Apply feature scaling if scaler is trained
        if hasattr(self.feature_scaler, 'mean_'):
            embeddings = self.feature_scaler.transform(embeddings)
            
        return embeddings
    
    def process_paragraph(self, paragraph: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Process a paragraph for prediction
        
        Returns:
            Tuple containing:
            - Sentence embeddings (seq_length, embedding_dim)
            - List of sentences
            - Raw sentence embeddings before scaling (for analysis)
        """
        # Split paragraph into sentences
        sentences = self.split_into_sentences(paragraph)
        
        # Get raw embeddings for analysis
        raw_embeddings = self.embedding_model.encode(sentences)
        
        # Get scaled embeddings for the model
        embeddings = raw_embeddings
        if hasattr(self.feature_scaler, 'mean_'):
            embeddings = self.feature_scaler.transform(raw_embeddings)
        
        return embeddings, sentences, raw_embeddings
    
    def save_models_and_scalers(self, feature_scaler=None, y_scaler=None, lstm_model=None):
        """Save trained models and scalers to disk"""
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
            with open(os.path.join(self.model_dir, 'paragraph_feature_scaler.pkl'), 'wb') as f:
                pickle.dump(feature_scaler, f)
        
        if y_scaler is not None:
            self.y_scaler = y_scaler
            with open(os.path.join(self.model_dir, 'paragraph_y_scaler.pkl'), 'wb') as f:
                pickle.dump(y_scaler, f)
        
        if lstm_model is not None:
            self.lstm_model = lstm_model
            torch.save(lstm_model.state_dict(), 
                      os.path.join(self.model_dir, 'paragraph_lstm_model.pth'))
            
        print(f"Models and scalers saved to {self.model_dir}") 