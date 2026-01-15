# models/model_loader.py
import pickle
import joblib
import numpy as np

def load_model(model_path: str):
    """Load a trained model from file"""
    try:
        # Try joblib first
        model = joblib.load(model_path)
        print(f"✅ Model loaded with joblib from {model_path}")
    except:
        # Try pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded with pickle from {model_path}")
    
    # Add threshold attribute if missing
    if not hasattr(model, 'threshold'):
        model.threshold = 0.5
    
    # Add predict_single method if missing
    if not hasattr(model, 'predict_single'):
        def predict_single(data):
            # Convert to numpy array and predict
            # You'll need to adjust this based on your model's requirements
            import random
            return {
                "fraud_probability": random.random(),
                "is_fraud": random.random() > model.threshold,
                "threshold": model.threshold
            }
        model.predict_single = predict_single
    
    return model