import joblib

def save_model(model, path):
    """Saves the trained model to the specified path."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Loads the trained model from the specified path."""
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
