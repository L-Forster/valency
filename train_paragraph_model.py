import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import nltk
from tqdm import tqdm
import random
from typing import List, Tuple, Dict
from paragraph_model import ParagraphLSTMModel, ParagraphProcessor

# Download NLTK data for sentence tokenization if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ParagraphDatasetBuilder:
    """
    Class for building paragraph datasets from individual sentences
    """
    def __init__(self, dataset_path: str = "emobank.csv", min_sentences: int = 3, max_sentences: int = 8):
        """
        Initialize the dataset builder
        
        Args:
            dataset_path (str): Path to the dataset CSV file
            min_sentences (int): Minimum number of sentences in a paragraph
            max_sentences (int): Maximum number of sentences in a paragraph
        """
        self.dataset_path = dataset_path
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.processor = ParagraphProcessor()
        
        # Load and preprocess the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Filter out sentences with less than 3 words or more than 50 words
        self.df['word_count'] = self.df['text'].apply(lambda x: len(str(x).split()))
        self.valid_sentences = self.df[(self.df['word_count'] >= 3) & (self.df['word_count'] <= 50)]
        
        print(f"Loaded dataset with {len(self.df)} sentences")
        print(f"Valid sentences for paragraph building: {len(self.valid_sentences)}")
    
    def build_paragraph(self, sentences_df: pd.DataFrame) -> Tuple[List[str], List[float], List[float]]:
        """
        Build a paragraph from the provided sentences dataframe
        
        Args:
            sentences_df (pd.DataFrame): DataFrame containing sentences to use
        
        Returns:
            Tuple containing:
            - List of sentences in the paragraph
            - List of valence values for each sentence
            - List of arousal values for each sentence
        """
        # Extract sentences and their valence/arousal values
        sentences = sentences_df['text'].tolist()
        valence_values = sentences_df['V'].tolist()
        arousal_values = sentences_df['A'].tolist()
        
        return sentences, valence_values, arousal_values
    
    def create_dataset(self, paragraph_size: int = 5, max_paragraphs: int = 200) -> Dict:
        """
        Create a dataset of paragraphs with valence-arousal values
        using each sentence only once
        
        Args:
            paragraph_size (int): Number of sentences per paragraph
            max_paragraphs (int): Maximum number of paragraphs to generate
            
        Returns:
            Dict: Dataset dictionary containing:
            - X: List of sentence embedding sequences
            - y: Array of paragraph valence-arousal values
            - paragraphs: List of paragraph sentences
            - sentence_values: Valence-arousal values for each sentence
        """
        X = []  # Sentence embedding sequences
        y = []  # Paragraph valence-arousal values
        paragraphs = []  # Paragraph text (list of sentences)
        sentence_values = []  # Valence-arousal values for individual sentences
        
        # Shuffle the valid sentences
        valid_sentences_shuffled = self.valid_sentences.sample(frac=1).reset_index(drop=True)
        
        # Calculate how many paragraphs we can create with the available sentences
        num_available_paragraphs = len(valid_sentences_shuffled) // paragraph_size
        num_paragraphs = min(num_available_paragraphs, max_paragraphs)
        
        print(f"Creating dataset with {num_paragraphs} paragraphs (each using {paragraph_size} unique sentences)...")
        
        # Generate paragraphs by splitting the shuffled sentences
        for i in tqdm(range(num_paragraphs)):
            start_idx = i * paragraph_size
            end_idx = start_idx + paragraph_size
            
            # Get a batch of sentences for this paragraph
            paragraph_sentences = valid_sentences_shuffled.iloc[start_idx:end_idx]
            
            # Build a paragraph from these sentences
            sentences, valence_values, arousal_values = self.build_paragraph(paragraph_sentences)
            
            # Calculate paragraph valence-arousal as weighted average of sentence values
            # Longer sentences have more weight
            weights = [len(s.split()) for s in sentences]
            weights = np.array(weights) / sum(weights)
            
            paragraph_valence = np.average(valence_values, weights=weights)
            paragraph_arousal = np.average(arousal_values, weights=weights)
            
            # Get embeddings for each sentence
            embeddings = self.processor.embedding_model.encode(sentences)
            
            # Store data
            X.append(embeddings)
            y.append([paragraph_valence, paragraph_arousal])
            paragraphs.append(sentences)
            sentence_values.append(list(zip(valence_values, arousal_values)))
            
            # Create augmented paragraph with shuffled order
            if len(sentences) > 2:
                shuffled_sentences = sentences.copy()
                random.shuffle(shuffled_sentences)
                shuffled_embeddings = self.processor.embedding_model.encode(shuffled_sentences)
                
                X.append(shuffled_embeddings)
                y.append([paragraph_valence, paragraph_arousal])  # Same target values
                paragraphs.append(shuffled_sentences)
                sentence_values.append(list(zip(valence_values, arousal_values)))
        
        print(f"Created {len(X)} paragraphs using {len(X) * paragraph_size} unique sentences")
        
        return {
            "X": X,  # List of embedding sequences (variable length)
            "y": np.array(y),  # Array of paragraph valence-arousal values
            "paragraphs": paragraphs,  # List of paragraph sentences
            "sentence_values": sentence_values  # Valence-arousal values for each sentence
        }


def pad_sequences(sequences: List[np.ndarray], max_len: int = None) -> Tuple[np.ndarray, List[int]]:
    """
    Pad variable length sequences to max_len
    
    Args:
        sequences (List[np.ndarray]): List of embedding sequences
        max_len (int, optional): Maximum sequence length. If None, use the length of the longest sequence.
        
    Returns:
        Tuple containing:
        - padded_sequences: Numpy array of padded sequences
        - seq_lengths: List of original sequence lengths
    """
    # Get sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    
    # Get maximum sequence length if not provided
    if max_len is None:
        max_len = max(seq_lengths)
    
    # Get embedding dimension from first sequence
    embedding_dim = sequences[0].shape[1]
    
    # Initialize padded sequences array
    padded_sequences = np.zeros((len(sequences), max_len, embedding_dim))
    
    # Fill padded_sequences with data
    for i, (seq, seq_len) in enumerate(zip(sequences, seq_lengths)):
        padded_sequences[i, :seq_len] = seq
    
    return padded_sequences, seq_lengths


def custom_loss(pred, target):
    # Combine MSE with a regularization term
    mse_loss = nn.MSELoss()(pred, target)
    return mse_loss


def train_paragraph_model(dataset_dict: Dict, epochs: int = 50, batch_size: int = 16, 
                          learning_rate: float = 0.001, device: str = None) -> Tuple[ParagraphLSTMModel, Dict]:
    """
    Train the paragraph LSTM model
    
    Args:
        dataset_dict (Dict): Dataset dictionary from ParagraphDatasetBuilder.create_dataset
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        device (str, optional): Device to use for training ('cuda' or 'cpu').
                               If None, use CUDA if available.
    
    Returns:
        Tuple containing:
        - trained_model: Trained ParagraphLSTMModel
        - training_history: Dictionary with training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data from dataset_dict
    X_sequences = dataset_dict["X"]
    y = dataset_dict["y"]
    
    # Pad sequences
    X_padded, seq_lengths = pad_sequences(X_sequences)
    
    # Create feature scaler for standardizing input features
    feature_scaler = StandardScaler()
    
    # Reshape for feature scaling (from 3D to 2D)
    X_2d = X_padded.reshape(-1, X_padded.shape[2])
    X_2d_scaled = feature_scaler.fit_transform(X_2d)
    
    # Reshape back to 3D
    X_scaled = X_2d_scaled.reshape(X_padded.shape)
    
    # Create target scaler for standardizing output values
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    
    # Split into train, validation, and test sets
    X_train, X_test, y_train, y_test, train_lens, test_lens = train_test_split(
        X_scaled, y_scaled, seq_lengths, test_size=0.2, random_state=42
    )
    
    # Further split training data into training and validation
    X_train, X_val, y_train, y_val, train_lens, val_lens = train_test_split(
        X_train, y_train, train_lens, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create LSTM model
    input_size = X_train.shape[2]  # Embedding dimension
    hidden_size = 32  # Smaller hidden size
    num_layers = 1  # Fewer layers
    output_size = 2  # Valence and arousal
    
    model = ParagraphLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    
    # Define loss function and optimizer
    criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'batch_losses': []
    }
    
    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        
        for X_batch, y_batch in train_loader:
            # Move data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        # Calculate training loss
        train_loss = np.mean(batch_losses)
        history['batch_losses'].extend(batch_losses)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = X_val_tensor.to(device)
            y_val_tensor = y_val_tensor.to(device)
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test the model
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Convert predictions back to original scale for analysis
    test_pred = y_scaler.inverse_transform(test_outputs.cpu().numpy())
    test_true = y_scaler.inverse_transform(y_test_tensor.cpu().numpy())
    
    # Calculate mean absolute error for valence and arousal
    mae_valence = np.mean(np.abs(test_pred[:, 0] - test_true[:, 0]))
    mae_arousal = np.mean(np.abs(test_pred[:, 1] - test_true[:, 1]))
    
    print(f"Mean Absolute Error - Valence: {mae_valence:.4f}, Arousal: {mae_arousal:.4f}")
    
    # Save the model and scalers
    processor = ParagraphProcessor()
    processor.save_models_and_scalers(feature_scaler, y_scaler, model)
    
    # Also save the model directly to ensure it's saved correctly
    model_save_path = os.path.join('models/paragraph', 'paragraph_lstm_model.pth')
    os.makedirs('models/paragraph', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_true[:, 0], test_pred[:, 0], alpha=0.5, label='Valence')
    plt.scatter(test_true[:, 1], test_pred[:, 1], alpha=0.5, label='Arousal')
    plt.plot([1, 5], [1, 5], 'k--')  # Diagonal line
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, {
        'history': history,
        'test_loss': test_loss,
        'mae_valence': mae_valence,
        'mae_arousal': mae_arousal,
        'feature_scaler': feature_scaler,
        'y_scaler': y_scaler
    }


if __name__ == "__main__":
    # Create dataset builder
    dataset_builder = ParagraphDatasetBuilder()
    
    # Create dataset - using each sentence once (5 sentences per paragraph, max 200 paragraphs)
    dataset = dataset_builder.create_dataset(paragraph_size=5, max_paragraphs=200)
    
    # Train the model with fewer epochs since we have less data
    model, training_results = train_paragraph_model(
        dataset, epochs=50, batch_size=16, learning_rate=0.0005
    ) 