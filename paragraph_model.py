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
from pathlib import Path 

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
    MODIFIED to return attention-based contribution weights.
    """
    def __init__(self, input_size=384, d_model=384, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1, output_size=2, max_seq_len=100):
        super(ParagraphTransformerModel, self).__init__()

        self.num_encoder_layers = num_encoder_layers # Store this

        if d_model != input_size:
            self.input_proj = nn.Linear(input_size, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )

        self.fc = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Transformer model.
        MODIFIED to return predictions and contribution weights.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Final predictions for valence and arousal (batch_size, output_size)
            - Contribution weights derived from last layer attention (batch_size, seq_length)
        """
        # x shape: (batch_size, seq_length, input_size)
        x = self.input_proj(x) # shape: (batch_size, seq_length, d_model)
        x = self.pos_encoder(x) # shape: (batch_size, seq_length, d_model)

        # --- Manual iteration through encoder layers to capture attention ---
        last_layer_attn_weights = None
        # src_mask and src_key_padding_mask are assumed None for this paragraph model
        src_mask = None
        src_key_padding_mask = None

        for i in range(self.num_encoder_layers):
            layer = self.transformer_encoder.layers[i]

            if i < self.num_encoder_layers - 1:
                # Standard forward for all but the last layer
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                # --- Manual forward for the last layer to get attention ---
                # 1. Self-Attention block
                # Note: Assuming no masks for simplicity in paragraph processing context
                #       If masks were used, they'd need to be passed here.
                attn_output, last_layer_attn_weights = layer.self_attn(
                    x, x, x,                    # query, key, value
                    attn_mask=src_mask,         # Use the same mask if provided
                    key_padding_mask=src_key_padding_mask, # Use the same mask if provided
                    need_weights=True,          # Request attention weights
                    average_attn_weights=False  # Get per-head weights first
                )
                x = x + layer.dropout1(attn_output)
                x = layer.norm1(x)

                # 2. Feedforward block
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = x + layer.dropout2(ff_output)
                x = layer.norm2(x)
                # --- End Manual forward for last layer ---

        # --- Calculate contribution weights from attention ---
        # last_layer_attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        if last_layer_attn_weights is not None:
            # Average across heads
            avg_attn_weights = last_layer_attn_weights.mean(dim=1) # Shape: (batch_size, seq_len, seq_len)
            
            # Debug raw attention values
            print(f"[DEBUG] Raw attention matrix:\n{avg_attn_weights[0]}")  # Print first batch element
            
            # Calculate contribution weights - try a different approach
            # Instead of averaging and applying softmax (which can make uniform),
            # use the diagonal of the attention matrix (self-attention scores)
            # as a more distinctive measure of contribution
            diag_attn = torch.diagonal(avg_attn_weights, dim1=1, dim2=2)  # Extract diagonal elements (self-attention)
            print(f"[DEBUG] Self-attention diagonal: {diag_attn[0]}")  # Print first batch
            
            # Use the self-attention scores directly without softmax normalization
            contribution_weights = diag_attn
            
            # Only normalize if the values are very different in scale
            # contribution_weights = torch.softmax(contribution_weights, dim=1)

            # --- Debugging shape ---
            # print(f"[DEBUG] Shape of avg_attn_weights: {avg_attn_weights.shape}") # Add for debugging

            # Check the number of dimensions before taking the mean
            if avg_attn_weights.ndim == 3:
                # Expected case: Average attention *received* by each token (dim 2)
                contribution_weights = avg_attn_weights.mean(dim=2) # Shape: (batch_size, seq_len)
            elif avg_attn_weights.ndim == 2:
                # Unexpected case: If it's already 2D, maybe just use it directly?
                # This depends on what it represents if it's 2D here.
                # Let's assume it might be (batch_size, seq_len) if S=1 caused squeezing?
                print("[DEBUG] Warning: avg_attn_weights was 2D, using directly as contribution_weights.")
                contribution_weights = avg_attn_weights
            else:
                # Fallback for other unexpected shapes
                print(f"[DEBUG] Warning: avg_attn_weights had unexpected ndim={avg_attn_weights.ndim}. Using uniform weights.")
                batch_size, seq_len = avg_attn_weights.shape[0], avg_attn_weights.shape[1] # Assuming at least 2D
                contribution_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len

            # Ensure weights sum to 1 (apply softmax regardless of how weights were derived)
            # Make sure contribution_weights is 2D before softmax dim=1
            if contribution_weights.ndim == 1:
                contribution_weights = contribution_weights.unsqueeze(0) # Add batch dim if missing

            if contribution_weights.ndim == 2:
                contribution_weights = torch.softmax(contribution_weights, dim=1) # Shape: (batch_size, seq_len)
            else:
                print(f"[DEBUG] Warning: Cannot apply softmax to contribution_weights with ndim={contribution_weights.ndim}")
                # Handle error or use unnormalized weights
            # --- MODIFICATION END ---

        else:
            # Fallback if attention wasn't calculated
            batch_size, seq_len, _ = x.shape
            contribution_weights = torch.ones(batch_size, seq_len, device=x.device) / seq_len
        # Pooling: Average the outputs across the sequence length dimension
        pooled_output = x.mean(dim=1) # shape: (batch_size, d_model)

        # Final prediction layer
        predictions = self.fc(pooled_output) # shape: (batch_size, output_size)

        return predictions, contribution_weights

# (Keep ParagraphProcessor class mostly as is, but update path logic)
class ParagraphProcessor:
    """
    Handles processing of paragraphs for the Transformer model, including:
    - Breaking paragraphs into sentences
    - Generating sentence embeddings
    - Handling model inference
    """
    def __init__(self, model_dir='models/paragraph'):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # --- Updated Helper function to find project root ---
        def get_project_root():
            current_path = Path(__file__).resolve()
            for parent in current_path.parents:
                # Look for a common project marker like .git or pyproject.toml or requirements.txt
                if (parent / ".git").exists() or \
                   (parent / "pyproject.toml").exists() or \
                   (parent / "requirements.txt").exists() or \
                   (parent / "README.md").exists(): # Added README as another potential marker
                    return parent
            # Fallback: Use the directory containing this script file
            return current_path.parent

        project_root = get_project_root()
        # Construct path relative to the project root
        # Assumes models are stored in <project_root>/models/paragraph
        model_path = project_root / model_dir # Use the argument for flexibility
        print(f"Attempting to load models from resolved path: {model_path}")
        self.model_dir = model_path

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Load scalers and model if available
        self.feature_scaler = self._load_feature_scaler()
        self.y_scaler = self._load_y_scaler()
        self.transformer_model = self._load_transformer_model()

    def _load_feature_scaler(self):
        """Load feature scaler from disk if it exists"""
        scaler_path = self.model_dir / 'paragraph_feature_scaler.pkl'
        try:
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                    print("Feature scaler loaded successfully")
                    # Check if it's fitted (optional but good practice)
                    if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                         print("Warning: Loaded feature scaler doesn't appear to be fitted.")
                    return scaler
            else:
                 print(f"No feature scaler found at {scaler_path}, creating new one.")
                 return StandardScaler()
        except Exception as e:
            print(f"Error loading feature scaler from {scaler_path}: {e}. Creating new one.")
            return StandardScaler()

    def _load_y_scaler(self):
        """Load target scaler from disk if it exists"""
        scaler_path = self.model_dir / 'paragraph_y_scaler.pkl'
        try:
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                    print("Target scaler loaded successfully")
                    if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
                         print("Warning: Loaded target scaler doesn't appear to be fitted.")
                    return scaler
            else:
                print(f"No target scaler found at {scaler_path}, creating new one.")
                return StandardScaler()
        except Exception as e:
            print(f"Error loading target scaler from {scaler_path}: {e}. Creating new one.")
            return StandardScaler()

    def _load_transformer_model(self):
        """Load trained Transformer model from disk if it exists"""
        model_path = self.model_dir / 'paragraph_transformer_model.pth'

        # --- Instantiate the Transformer model ---
        # Ensure parameters match the ones used during training
        # *** IMPORTANT: Use the *fixed* max_seq_len consistent with training and inference needs ***
        model = ParagraphTransformerModel(
            input_size=384,
            d_model=384,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            output_size=2,
            max_seq_len=100 # Use the fixed value (e.g., 100)
        )

        try:
            if model_path.exists():
                # Try loading state dict with strict=False initially if mismatches might occur
                # due to the forward method change (though unlikely to affect state dict keys)
                try:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    print("Transformer model loaded successfully (strict=True).")
                except RuntimeError as e:
                     print(f"RuntimeError loading state dict (strict=True): {e}")
                     print("Attempting to load with strict=False...")
                     # Load non-strictly if exact match fails (e.g., if buffer names changed slightly)
                     # Be cautious with this, ensure missing/unexpected keys are okay.
                     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
                     print("Transformer model loaded potentially with non-strict matching.")

                model.eval() # Set model to evaluation mode
            else:
                print(f"No pre-trained Transformer model found at {model_path}. Model is untrained.")
                # Return the untrained model instance
        except Exception as e:
            print(f"Error loading Transformer model state dict from {model_path}: {e}")
            # Return the untrained model instance

        return model

    def split_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into individual sentences"""
        # Added basic cleanup: remove extra whitespace before tokenizing
        paragraph = " ".join(paragraph.split())
        return nltk.sent_tokenize(paragraph)

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences"""
        if not sentences:
            return np.array([]).reshape(0, self.embedding_model.get_sentence_embedding_dimension())

        embeddings = self.embedding_model.encode(sentences)

        # Apply feature scaling if scaler is trained (has mean_ attribute)
        if hasattr(self.feature_scaler, 'mean_') and self.feature_scaler.mean_ is not None:
             # Check dimensions before transforming
            if embeddings.shape[1] == self.feature_scaler.n_features_in_:
                 embeddings = self.feature_scaler.transform(embeddings)
            else:
                 print(f"Warning: Embedding dimension ({embeddings.shape[1]}) doesn't match scaler expected features ({self.feature_scaler.n_features_in_}). Skipping scaling.")

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
        sentences = self.split_into_sentences(paragraph)
        if not sentences:
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            return np.array([]).reshape(0, embedding_dim), [], np.array([]).reshape(0, embedding_dim)

        # Get raw embeddings
        raw_embeddings = self.embedding_model.encode(sentences)

        # Get scaled embeddings for the model input
        scaled_embeddings = raw_embeddings.copy() # Start with raw
        if hasattr(self.feature_scaler, 'mean_') and self.feature_scaler.mean_ is not None:
            if raw_embeddings.shape[1] == self.feature_scaler.n_features_in_:
                 scaled_embeddings = self.feature_scaler.transform(raw_embeddings)
            else:
                 print(f"Warning: Embedding dimension ({raw_embeddings.shape[1]}) doesn't match scaler expected features ({self.feature_scaler.n_features_in_}). Using raw embeddings for model input.")


        return scaled_embeddings, sentences, raw_embeddings

    def save_models_and_scalers(self, feature_scaler=None, y_scaler=None, transformer_model=None):
        """Save trained models and scalers to disk"""
        os.makedirs(self.model_dir, exist_ok=True) # Ensure directory exists

        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
            scaler_path = self.model_dir / 'paragraph_feature_scaler.pkl'
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(feature_scaler, f)
                print(f"Feature scaler saved to {scaler_path}")
            except Exception as e:
                print(f"Error saving feature scaler: {e}")

        if y_scaler is not None:
            self.y_scaler = y_scaler
            scaler_path = self.model_dir / 'paragraph_y_scaler.pkl'
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(y_scaler, f)
                print(f"Target scaler saved to {scaler_path}")
            except Exception as e:
                print(f"Error saving target scaler: {e}")

        if transformer_model is not None:
            self.transformer_model = transformer_model
            model_path = self.model_dir / 'paragraph_transformer_model.pth'
            try:
                torch.save(transformer_model.state_dict(), model_path)
                print(f"Transformer model saved to {model_path}")
            except Exception as e:
                print(f"Error saving Transformer model: {e}")


