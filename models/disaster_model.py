import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def get_model():
    """Initializes and returns the Random Forest classifier configuration."""
    return RandomForestClassifier(n_estimators=100, random_state=42)

def save_model(model, model_path="models/disaster_risk_model.pkl"):
    """Saves the trained model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

def load_trained_model(model_path="models/disaster_risk_model.pkl"):
    """Loads a pre-trained model from disk for inference/routing."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train it first.")
    return joblib.load(model_path)