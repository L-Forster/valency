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

# --- Import the CORRECT model and processor ---
from paragraph_model import ParagraphTransformerModel, ParagraphProcessor 
# Note: Ensure paragraph_model.py contains the Transformer version and the 
# updated processor from the previous step.

# Download NLTK data for sentence tokenization if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ParagraphDatasetBuilder:
    """
    Class for building paragraph datasets from individual sentences.
    (No changes needed in this class itself, as it only prepares data)
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
        # Instantiate the processor (ensure it's the version aware of Transformers)
        self.processor = ParagraphProcessor() 
        
        # Load and preprocess the dataset
        try:
            self.df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}")
            print("Please ensure 'emobank.csv' is in the correct directory.")
            # Optionally download or provide instructions here
            # Example: download_emobank() 
            raise # Re-raise the error to stop execution if file is essential

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
        using each sentence only once, potentially with shuffling augmentation.
        
        Args:
            paragraph_size (int): Number of sentences per paragraph
            max_paragraphs (int): Maximum number of paragraphs to generate
            
        Returns:
            Dict: Dataset dictionary containing:
            - X: List of sentence embedding sequences (raw embeddings)
            - y: Array of paragraph valence-arousal values
            - paragraphs: List of paragraph sentences
            - sentence_values: Valence-arousal values for each sentence
        """
        X = []  # Sentence embedding sequences (will be scaled later)
        y = []  # Paragraph valence-arousal values
        paragraphs = []  # Paragraph text (list of sentences)
        sentence_values = []  # Valence-arousal values for individual sentences
        
        # Shuffle the valid sentences
        valid_sentences_shuffled = self.valid_sentences.sample(frac=1).reset_index(drop=True)
        
        # Calculate how many paragraphs we can create with the available sentences
        num_available_paragraphs = len(valid_sentences_shuffled) // paragraph_size
        
        if num_available_paragraphs == 0:
             print(f"Warning: Not enough valid sentences ({len(valid_sentences_shuffled)}) to create even one paragraph of size {paragraph_size}.")
             return {"X": [], "y": np.array([]), "paragraphs": [], "sentence_values": []}

        num_paragraphs = min(num_available_paragraphs, max_paragraphs)
        
        print(f"Creating dataset with {num_paragraphs} base paragraphs (each using {paragraph_size} unique sentences)...")
        
        # Generate paragraphs by splitting the shuffled sentences
        for i in tqdm(range(num_paragraphs)):
            start_idx = i * paragraph_size
            end_idx = start_idx + paragraph_size
            
            # Get a batch of sentences for this paragraph
            paragraph_sentences_df = valid_sentences_shuffled.iloc[start_idx:end_idx]
            
            # Build a paragraph from these sentences
            sentences, valence_values, arousal_values = self.build_paragraph(paragraph_sentences_df)
            
            # Calculate paragraph valence-arousal as weighted average of sentence values
            # Longer sentences have more weight
            words_per_sentence = [len(s.split()) for s in sentences]
            # Avoid division by zero if all sentences are empty (unlikely with filtering)
            total_words = sum(words_per_sentence)
            if total_words > 0:
                weights = np.array(words_per_sentence) / total_words
            else:
                weights = np.ones(len(sentences)) / len(sentences) # Equal weight if no words

            paragraph_valence = np.average(valence_values, weights=weights)
            paragraph_arousal = np.average(arousal_values, weights=weights)
            
            # Get embeddings for each sentence (using the processor's embedder)
            # Note: We store the raw embeddings here; scaling happens in the training function
            embeddings = self.processor.embedding_model.encode(sentences)
            
            # Store data for original order
            X.append(embeddings)
            y.append([paragraph_valence, paragraph_arousal])
            paragraphs.append(sentences)
            sentence_values.append(list(zip(valence_values, arousal_values)))
            
            # --- Data Augmentation: Create augmented paragraph with shuffled order ---
            if len(sentences) > 1: # Only shuffle if more than one sentence
                shuffled_indices = list(range(len(sentences)))
                random.shuffle(shuffled_indices)
                
                # Reorder sentences, embeddings, and corresponding sentence values
                shuffled_sentences = [sentences[j] for j in shuffled_indices]
                # Important: Use the *already computed* embeddings and reorder them
                shuffled_embeddings = embeddings[shuffled_indices] 
                shuffled_sentence_vals = [sentence_values[-1][j] for j in shuffled_indices] # Keep track if needed

                # Append shuffled version - Target value remains the same!
                X.append(shuffled_embeddings)
                y.append([paragraph_valence, paragraph_arousal]) 
                paragraphs.append(shuffled_sentences)
                # Store reordered sentence values if needed for analysis
                sentence_values.append(shuffled_sentence_vals) 
        
        print(f"Created {len(X)} total paragraphs (including augmented ones).")
        
        return {
            "X": X,  # List of embedding sequences (raw, variable length)
            "y": np.array(y),  # Array of paragraph valence-arousal values
            "paragraphs": paragraphs,  # List of paragraph sentences
            "sentence_values": sentence_values  # Valence-arousal values for each sentence
        }


def pad_sequences(sequences: List[np.ndarray], max_len: int = None, pad_value=0.0) -> Tuple[np.ndarray, List[int]]:
    """
    Pad variable length sequences to max_len.
    
    Args:
        sequences (List[np.ndarray]): List of embedding sequences.
        max_len (int, optional): Maximum sequence length. If None, use the length 
                                 of the longest sequence in the list.
        pad_value (float): Value used for padding. Default is 0.0.
        
    Returns:
        Tuple containing:
        - padded_sequences: Numpy array of padded sequences (batch, max_len, embedding_dim).
        - seq_lengths: List of original sequence lengths for each sequence in the batch.
    """
    if not sequences: # Handle empty input list
        return np.array([]).reshape(0, 0, 0), []

    # Get sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    
    # Determine maximum sequence length
    if max_len is None:
        max_len = max(seq_lengths) if seq_lengths else 0
    
    if max_len == 0: # Handle case where all sequences might be empty
       embedding_dim = sequences[0].shape[1] if sequences and sequences[0].ndim > 1 else 0
       return np.array([]).reshape(0, 0, embedding_dim), []


    # Get embedding dimension from the first non-empty sequence
    embedding_dim = 0
    for seq in sequences:
        if seq.ndim == 2 and seq.shape[1] > 0: # Check for valid embedding shape
            embedding_dim = seq.shape[1]
            break
            
    if embedding_dim == 0: # If no valid embeddings found
        print("Warning: Could not determine embedding dimension from sequences.")
        # Attempt to get from first sequence anyway, or handle as error
        try:
            embedding_dim = sequences[0].shape[1]
        except IndexError:
             return np.array([]).reshape(0, max_len, 0), seq_lengths


    # Initialize padded sequences array
    padded_sequences = np.full((len(sequences), max_len, embedding_dim), pad_value, dtype=np.float32)
    
    # Fill padded_sequences with data
    for i, seq in enumerate(sequences):
        length = min(seq_lengths[i], max_len) # Use the actual length or max_len, whichever is smaller
        if length > 0: # Only copy if there's data
             padded_sequences[i, :length] = seq[:length] # Handle sequences longer than max_len by truncating
            
    return padded_sequences, seq_lengths


def custom_loss(pred, target):
    """Simple Mean Squared Error Loss"""
    # Combine MSE with a regularization term
    mse_loss = nn.MSELoss()(pred, target)
    # Can add other terms here if needed, e.g., correlation loss, etc.
    return mse_loss

# --- Updated Function Signature for Return Type ---
def train_paragraph_model(dataset_dict: Dict, epochs: int = 50, batch_size: int = 16, 
                          learning_rate: float = 0.001, weight_decay: float = 1e-5,
                          transformer_params: Dict = None, # Pass Transformer params here
                          device: str = None) -> Tuple[ParagraphTransformerModel, Dict]:
    """
    Train the paragraph Transformer model.
    
    Args:
        dataset_dict (Dict): Dataset dictionary from ParagraphDatasetBuilder.create_dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 penalty) for optimizer.
        transformer_params (Dict, optional): Dictionary of parameters for the 
                                             ParagraphTransformerModel. If None, uses defaults.
        device (str, optional): Device for training ('cuda' or 'cpu'). Uses CUDA if available.
    
    Returns:
        Tuple containing:
        - trained_model: Trained ParagraphTransformerModel.
        - training_info: Dictionary with training history and scalers.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data from dataset_dict
    X_sequences = dataset_dict["X"] # List of numpy arrays (embeddings)
    y = dataset_dict["y"]           # Numpy array (targets)
    
    if not X_sequences or len(y) == 0:
        print("Error: Empty dataset provided to train_paragraph_model.")
        # Return None or raise an error as appropriate
        return None, {} 

    # --- Pad sequences ---
    # Pad using the function defined above. Padding value 0.0 is standard.
    # max_len=None calculates max length from the current dataset split.
    X_padded, seq_lengths = pad_sequences(X_sequences, max_len=None, pad_value=0.0)
    
    # Max sequence length determined after padding
    max_seq_len_actual = X_padded.shape[1]
    FIXED_MAX_SEQ_LEN = 100 
    print(f"Sequences padded to max length: {max_seq_len_actual}")

    if X_padded.size == 0:
        print("Error: Padded data is empty. Cannot proceed.")
        return None, {}

    # --- Feature Scaling (Input Embeddings) ---
    feature_scaler = StandardScaler()
    
    # Reshape for feature scaling (samples * seq_len, features)
    num_samples, seq_len, num_features = X_padded.shape
    # Avoid scaling padding: only fit on non-padded elements if significant padding exists
    # Simple approach: fit on all data (including potential padding zeros)
    X_2d = X_padded.reshape(-1, num_features) 
    
    # Fit the scaler ONLY on the training data later, transform all splits
    # We'll fit it after the train/test split

    # --- Target Scaling (Output Values) ---
    y_scaler = StandardScaler()
    # Fit the scaler ONLY on the training data later, transform all splits

    # --- Train/Validation/Test Split ---
    # Split indices first to keep sequences and lengths aligned
    indices = np.arange(len(X_padded))
    
    try:
        train_indices, test_indices, y_train_indices, y_test_indices = train_test_split(
            indices, indices, test_size=0.2, random_state=42, stratify=None # Stratify might be hard with continuous multi-output
        )
        train_indices, val_indices, y_train_indices, y_val_indices = train_test_split(
            train_indices, train_indices, test_size=0.2, random_state=42 # 0.2 of original 0.8 -> 0.16 validation
        )
    except ValueError as e:
         print(f"Error during train/test split: {e}")
         print(f"Dataset size: {len(indices)}. Ensure you have enough samples for splitting.")
         return None, {}


    # Apply indices to get data splits
    X_train_padded, X_val_padded, X_test_padded = X_padded[train_indices], X_padded[val_indices], X_padded[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    # Keep original sequence lengths aligned (optional, Transformer might not need them directly)
    # train_lens = [seq_lengths[i] for i in train_indices]
    # val_lens = [seq_lengths[i] for i in val_indices]
    # test_lens = [seq_lengths[i] for i in test_indices]
    
    # --- Fit Scalers on Training Data ONLY ---
    print("Fitting scalers on training data...")
    # Fit and transform X_train
    X_train_2d = X_train_padded.reshape(-1, num_features)
    X_train_scaled_2d = feature_scaler.fit_transform(X_train_2d)
    X_train_scaled = X_train_scaled_2d.reshape(X_train_padded.shape)
    
    # Transform X_val and X_test
    X_val_2d = X_val_padded.reshape(-1, num_features)
    X_val_scaled_2d = feature_scaler.transform(X_val_2d)
    X_val_scaled = X_val_scaled_2d.reshape(X_val_padded.shape)
    
    X_test_2d = X_test_padded.reshape(-1, num_features)
    X_test_scaled_2d = feature_scaler.transform(X_test_2d)
    X_test_scaled = X_test_scaled_2d.reshape(X_test_padded.shape)

    # Fit and transform y_train
    y_train_scaled = y_scaler.fit_transform(y_train)
    # Transform y_val and y_test
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)

    # --- Convert to PyTorch tensors ---
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # --- Instantiate the Transformer Model ---
    input_size = num_features # Embedding dimension from SentenceTransformer

    # Default parameters if none provided
    default_tf_params = {
        'input_size': input_size,
        'd_model': input_size,      # Often keep same as input for simplicity
        'nhead': 8,                 # Must divide d_model (384/8=48)
        'num_encoder_layers': 3,    # Number of Transformer blocks
        'dim_feedforward': 512,     # Hidden dim in FFN layers
        'dropout': 0.1,
        'output_size': 2,           # Valence and arousal
        'max_seq_len': FIXED_MAX_SEQ_LEN # Use actual max length from padded data
    }
    
    # Update defaults with any user-provided params
    
    # Update defaults with any user-provided params
    if transformer_params:
        final_tf_params = default_tf_params.copy()
        # Allow overriding if needed, but ensure 'max_seq_len' is consistent
        # If transformer_params contains 'max_seq_len', ensure it's also FIXED_MAX_SEQ_LEN
        # or handle the inconsistency explicitly. For simplicity, let's assume
        # the fixed value is desired unless explicitly overridden with the same value.
        if 'max_seq_len' in transformer_params and transformer_params['max_seq_len'] != FIXED_MAX_SEQ_LEN:
            print(f"Warning: transformer_params contains max_seq_len={transformer_params['max_seq_len']}, "
                  f"but training is set to use fixed max_seq_len={FIXED_MAX_SEQ_LEN}. Using the fixed value.")
        
        # Update other params provided by user
        params_to_update = {k: v for k, v in transformer_params.items() if k != 'max_seq_len'}
        final_tf_params.update(params_to_update)
        final_tf_params['max_seq_len'] = FIXED_MAX_SEQ_LEN # Ensure it's the fixed value

        # --- Add validation for parameters ---
        if final_tf_params['d_model'] % final_tf_params['nhead'] != 0:
             raise ValueError(f"d_model ({final_tf_params['d_model']}) must be divisible by nhead ({final_tf_params['nhead']})")
        print("Using Transformer parameters:", final_tf_params)
    else:
        final_tf_params = default_tf_params
        print("Using default Transformer parameters:", final_tf_params)


    model = ParagraphTransformerModel(**final_tf_params).to(device) # Use updated params




    # --- Define loss function and optimizer ---
    criterion = custom_loss # Using the defined MSE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # --- Training Setup ---
    history = {
        'train_loss': [],
        'val_loss': [],
        # 'batch_losses': [] # Optional: Track every batch loss
    }
    
    # Create DataLoader for efficient batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False # No need to shuffle validation
    )

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False 
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    patience_counter = 0
    best_model_state = None
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_train_losses = []
        
        # Training batch loop
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass - unpack the tuple
            predictions, _ = model(X_batch) # Unpack predictions and ignore weights here
            loss = criterion(predictions, y_batch) # Use only predictions for loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping if gradients explode
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        # Calculate average training loss for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # --- Validation ---
        model.eval() # Set model to evaluation mode
        epoch_val_losses = []
        with torch.no_grad(): # Disable gradient calculation for validation
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                # Unpack the tuple
                val_outputs, _ = model(X_val_batch) # Unpack predictions and ignore weights here
                val_loss = criterion(val_outputs, y_val_batch) # Use only predictions for loss
                epoch_val_losses.append(val_loss.item())
                
        avg_val_loss = np.mean(epoch_val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # --- Checkpointing and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict() # Save the best model state
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best validation loss: {best_val_loss:.4f}. Saving model state.")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Val loss did not improve ({avg_val_loss:.4f} vs best {best_val_loss:.4f}). Patience: {patience_counter}/{patience}")

        # Print progress (maybe less frequently)
        if (epoch + 1) % 1 == 0: # Print every epoch now with the improvement message
             print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} due to lack of validation improvement.")
            break
            
    print("--- Training Finished ---")

    # --- Load Best Model ---
    if best_model_state is not None:
        print("Loading best model state found during training.")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved (perhaps training stopped early or val loss never improved). Using last state.")

    # --- Final Evaluation on Test Set ---
    model.eval()
    test_losses = []
    all_test_outputs = []
    all_test_true = []
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device) # Scaled targets

            # Unpack the tuple
            test_outputs_batch, _ = model(X_test_batch) # Unpack predictions and ignore weights here
            test_loss = criterion(test_outputs_batch, y_test_batch) # Use only predictions for loss
            test_losses.append(test_loss.item())
            
            all_test_outputs.append(test_outputs_batch.cpu().numpy())
            all_test_true.append(y_test_batch.cpu().numpy())
            
    avg_test_loss = np.mean(test_losses)
    print(f"\n--- Final Test Results (using best model) ---")
    print(f"Test Loss (Scaled): {avg_test_loss:.4f}")
    
    # Concatenate results from all test batches
    test_pred_scaled = np.concatenate(all_test_outputs, axis=0)
    test_true_scaled = np.concatenate(all_test_true, axis=0)

    # --- Inverse Transform Predictions and True Values ---
    # Check if scaler was actually fitted (possible if training data was empty)
    if hasattr(y_scaler, 'scale_'):
        test_pred_orig = y_scaler.inverse_transform(test_pred_scaled)
        test_true_orig = y_scaler.inverse_transform(test_true_scaled) # Should be same as original y_test
    else:
        print("Warning: y_scaler was not fitted. Using scaled values for MAE calculation.")
        test_pred_orig = test_pred_scaled
        test_true_orig = test_true_scaled

    # Calculate Mean Absolute Error (MAE) on original scale
    mae_valence = np.mean(np.abs(test_pred_orig[:, 0] - test_true_orig[:, 0]))
    mae_arousal = np.mean(np.abs(test_pred_orig[:, 1] - test_true_orig[:, 1]))
    
    print(f"Mean Absolute Error - Valence: {mae_valence:.4f}")
    print(f"Mean Absolute Error - Arousal: {mae_arousal:.4f}")
    
    # --- Save the Final Model and Scalers ---
    processor = ParagraphProcessor() # Create a fresh processor instance to save
    # Use the correct keyword argument expected by the updated save method
    processor.save_models_and_scalers(
        feature_scaler=feature_scaler, 
        y_scaler=y_scaler, 
        transformer_model=model # Pass the trained model object
    )
    
    # Optional: Save the model directly again for redundancy/clarity
    model_save_path = os.path.join(processor.model_dir, 'paragraph_transformer_model.pth')
    try:
        torch.save(model.state_dict(), model_save_path)
        print(f"Redundant save: Transformer model state dict saved to {model_save_path}")
    except Exception as e:
        print(f"Error during redundant model save: {e}")

    # --- Plotting ---
    plt.figure(figsize=(14, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Scaled)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Scatter Plot (Predictions vs True)
    plt.subplot(1, 2, 2)
    # Use original scale values for interpretability
    plt.scatter(test_true_orig[:, 0], test_pred_orig[:, 0], alpha=0.5, label='Valence')
    plt.scatter(test_true_orig[:, 1], test_pred_orig[:, 1], alpha=0.5, label='Arousal', marker='x')
    # Determine plot limits based on data range + buffer
    min_val = min(test_true_orig.min(), test_pred_orig.min()) - 0.5
    max_val = max(test_true_orig.max(), test_pred_orig.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Fit') # Diagonal line
    plt.xlabel('True Values (Original Scale)')
    plt.ylabel('Predicted Values (Original Scale)')
    plt.title('Test Set: Predictions vs True Values')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes equal

    plt.tight_layout()
    # plt.show()
    
    # --- Return Trained Model and Info ---
    training_info = {
        'history': history,
        'test_loss_scaled': avg_test_loss,
        'mae_valence': mae_valence,
        'mae_arousal': mae_arousal,
        'feature_scaler': feature_scaler, # Return fitted scalers
        'y_scaler': y_scaler,
        'final_transformer_params': final_tf_params # Store params used
    }
    
    return model, training_info


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Ensure the dataset exists
    DATASET_FILE = "emobank.csv"
    if not os.path.exists(DATASET_FILE):
         print(f"Error: Dataset file '{DATASET_FILE}' not found in the current directory.")
         print("Please download emobank.csv or place it here.")
         # Exit or handle appropriately
         exit()

    # Create dataset builder
    dataset_builder = ParagraphDatasetBuilder(dataset_path=DATASET_FILE)
    
    # Create dataset 
    # Increase max_paragraphs for potentially better training with Transformers
    dataset = dataset_builder.create_dataset(paragraph_size=5, max_paragraphs=1000) 
    
    if not dataset or not dataset['X']:
        print("Dataset creation failed or resulted in an empty dataset. Exiting.")
        exit()

    # Define Transformer parameters (optional, can be None to use defaults)
    # Example: Smaller model for faster testing
    # tf_params = {
    #     'd_model': 128, 
    #     'nhead': 4, 
    #     'num_encoder_layers': 2,
    #     'dim_feedforward': 256
    # } 
    tf_params = None # Use defaults defined in train_paragraph_model

    # Train the Transformer model
    trained_transformer_model, training_results = train_paragraph_model(
        dataset, 
        epochs=60,              # Transformers might need more epochs or different LR schedule
        batch_size=32,          # Adjust batch size based on GPU memory
        learning_rate=0.0002,   # Often lower LR works better for Transformers
        weight_decay=1e-5,      # Regularization
        transformer_params=tf_params # Pass the custom params or None
    )

    if trained_transformer_model:
        print("\n--- Training Summary ---")
        print(f"Final Test Loss (Scaled): {training_results.get('test_loss_scaled', 'N/A'):.4f}")
        print(f"MAE Valence: {training_results.get('mae_valence', 'N/A'):.4f}")
        print(f"MAE Arousal: {training_results.get('mae_arousal', 'N/A'):.4f}")
        print("Training complete. Model and scalers saved.")
    else:
        print("Model training did not complete successfully.")