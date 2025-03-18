import torch
import pickle
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import os

# Load the pre-fitted scalers (saved during training).
def load_scalers():
    # Get the absolute path to the models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    # Load feature scaler
    feature_scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
    with open(feature_scaler_path, "rb") as f:
        feature_scaler = pickle.load(f)
        
    # Load target scaler
    y_scaler_path = os.path.join(models_dir, 'y_scaler.pkl')
    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)
        
    return feature_scaler, y_scaler

# Define the neural network model.
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(NeuralNetworkModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.network(x)


# Load the trained regression model.
def load_regression_model():
    # Get the absolute path to the model file
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'regression_dnn.pth')
    
    regression_model = NeuralNetworkModel(input_size=384, hidden_size=64, output_size=2)
    regression_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    regression_model.eval()
    return regression_model



# (Optional) If you need to transform new features using the same feature scaler,
# load it from disk.
def load_feature_scaler():
    # Get the absolute path to the feature scaler file
    feature_scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_scaler.pkl')
    
    with open(feature_scaler_path, "rb") as f:
        feature_scaler = pickle.load(f)
    return feature_scaler


# Initialize the sentence embedding model.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Embed a sentence.
def embed_sentence(sentence: str):
    return embedding_model.encode([sentence])


# Inference function using the loaded models and scalers.
def inference(sentence: str):
    # Get sentence embedding.
    embedding = embed_sentence(sentence)
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

    # Load the regression model and target scaler.
    regression_model = load_regression_model()
    feature_scaler, y_scaler =  load_scalers()


    embedding_norm = feature_scaler.transform(embedding)
    embedding_tensor = torch.tensor(embedding_norm, dtype=torch.float32)

    regression_model.eval()  # ensure model is in eval mode
    with torch.no_grad():
        y_pred_norm = regression_model(embedding_tensor)
        # Inverse transform using the pre-fitted scaler:
        y_pred = y_scaler.inverse_transform(y_pred_norm.numpy())
        print("valence, arousal:")
        print(sentence, "Prediction:", y_pred)


    print("Valence, Arousal:")
    print("Prediction:", y_pred)
    return y_pred


# if __name__ == "__main__":
    # sentence = input("Enter sentence: ")
    # inference(sentence)
