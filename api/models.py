import joblib
import pandas as pd
import os

class FraudDetectionModel:
    def __init__(self):
        try:
            # Load model from models folder (one level up)
            model_path = "../models/rf_sm_fraud.pkl"
            self.model = joblib.load(model_path)
            self.threshold = 0.5
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def predict_single(self, transaction):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            from .preprocess import preprocessor
            processed = preprocessor.preprocess(transaction)
            
            prob = self.model.predict_proba(processed)[0, 1]
            prediction = int(prob >= self.threshold)
            
            # Risk level
            if prob >= 0.8:
                risk = "HIGH"
            elif prob >= 0.5:
                risk = "MEDIUM"
            else:
                risk = "LOW"
            
            return {
                "fraud_probability": float(prob),
                "is_fraud": bool(prediction),
                "risk_level": risk,
                "threshold": self.threshold
            }
        except Exception as e:
            return {"error": str(e)}
    
    def set_threshold(self, threshold):
        if 0 <= threshold <= 1:
            self.threshold = threshold
            return True
        return False

# Create instance
model = FraudDetectionModel()