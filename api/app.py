# app.py - Flask Fraud Detection API
"""
Run with: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).parent.parent  # Go up one level from api folder
MODEL_PATH = BASE_DIR / "models" / "rf_sm_fraud.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "models" / "feature_names.json"

# Log paths for debugging
logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Model Path: {MODEL_PATH}")
logger.info(f"Feature Names Path: {FEATURE_NAMES_PATH}")

# ========== FEATURE ENGINEERING ==========
class FeatureEngineer:
    """Feature engineering matching training pipeline"""
    
    def __init__(self, feature_names_path):
        try:
            with open(feature_names_path, 'r') as f:
                self.required_features = json.load(f)
            logger.info(f"✅ Loaded {len(self.required_features)} required features")
        except FileNotFoundError:
            logger.warning("⚠️ feature_names.json not found")
            self.required_features = None
        
        self.RARE_CATEGORY_THRESHOLD = 50
    
    def clean_and_transform(self, df):
        """Apply the same cleaning pipeline as training"""
        df_clean = df.copy()
        
        # 1. Drop unnecessary columns
        cols_to_drop = [
            "uuid", "lut_first_paid_date", "lut_last_paid_date",
            "ip", "isp", "latitude", "longitude", "to_bank"
        ]
        df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
        
        # 2. Convert datetime columns and extract features
        datetime_cols = ["date", "loginTime", "txn_timestamp"]
        
        for col in datetime_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                
                # Extract datetime components
                df_clean[col + "_year"] = df_clean[col].dt.year
                df_clean[col + "_month"] = df_clean[col].dt.month
                df_clean[col + "_day"] = df_clean[col].dt.day
                df_clean[col + "_hour"] = df_clean[col].dt.hour
                df_clean[col + "_minute"] = df_clean[col].dt.minute
                df_clean[col + "_second"] = df_clean[col].dt.second
                
                df_clean = df_clean.drop(columns=[col], errors='ignore')
        
        # 3. Fill missing values
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col] = df_clean[col].fillna("unknown")
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # 4. Encode categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Group rare categories
        for c in categorical_cols:
            vc = df_clean[c].value_counts(dropna=False)
            if len(vc) > 0:
                frequent = set(vc[vc >= self.RARE_CATEGORY_THRESHOLD].index)
                df_clean[c] = df_clean[c].where(df_clean[c].isin(frequent), other="rare_category")
        
        # One-hot encode
        df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        
        # 5. Convert to numeric
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    df_clean[col] = df_clean[col].astype('category').cat.codes
        
        df_clean = df_clean.fillna(0)
        
        return df_clean
    
    def prepare_features(self, transaction_dict):
        """Convert single transaction to features"""
        df = pd.DataFrame([transaction_dict])
        df_clean = self.clean_and_transform(df)
        
        # Align with training features
        if self.required_features:
            # Add missing features
            missing_features = set(self.required_features) - set(df_clean.columns)
            for feature in missing_features:
                df_clean[feature] = 0
            
            # Select required features in correct order
            df_clean = df_clean[self.required_features]
        
        return df_clean


# ========== MODEL CLASS ==========
class FraudModel:
    """Wrapper for fraud detection model"""
    
    def __init__(self, model_path, feature_engineer):
        self.threshold = 0.5
        self.feature_engineer = feature_engineer
        self.model = self._load_model(model_path)
        self.n_features = getattr(self.model, 'n_features_in_', None)
    
    def _load_model(self, model_path):
        """Load trained model"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"✅ Loaded model from {model_path}")
        return model
    
    def predict_single(self, transaction_dict):
        """Predict fraud for single transaction"""
        # Prepare features
        features = self.feature_engineer.prepare_features(transaction_dict)
        
        # Get prediction
        probability = float(self.model.predict_proba(features)[0, 1])
        is_fraud = probability > self.threshold
        
        # Determine risk level
        if probability > 0.8:
            risk_level = "Critical"
        elif probability > 0.5:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "transaction_id": transaction_dict.get("uuid", "unknown"),
            "fraud_probability": round(probability, 4),
            "is_fraud": bool(is_fraud),
            "risk_level": risk_level,
            "threshold": self.threshold
        }
    
    def predict_batch(self, transactions):
        """Predict fraud for multiple transactions"""
        results = []
        fraud_count = 0
        total_prob = 0.0
        
        for txn in transactions:
            result = self.predict_single(txn)
            results.append(result)
            if result['is_fraud']:
                fraud_count += 1
            total_prob += result['fraud_probability']
        
        total = len(results)
        
        return {
            "batch_id": f"batch_{int(datetime.now().timestamp())}",
            "predictions": results,
            "statistics": {
                "total_transactions": total,
                "fraud_count": fraud_count,
                "fraud_percentage": round((fraud_count / total * 100) if total > 0 else 0, 2),
                "avg_fraud_probability": round(total_prob / total if total > 0 else 0, 4)
            }
        }
    
    def set_threshold(self, threshold):
        """Update threshold"""
        if 0 <= threshold <= 1:
            self.threshold = threshold
            return True
        return False


# ========== INITIALIZE MODEL ==========
try:
    logger.info("=" * 80)
    logger.info("🚀 INITIALIZING FLASK FRAUD DETECTION API")
    logger.info("=" * 80)
    
    feature_engineer = FeatureEngineer(FEATURE_NAMES_PATH)
    fraud_model = FraudModel(MODEL_PATH, feature_engineer)
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"📊 Expected features: {fraud_model.n_features}")
    logger.info(f"🎯 Default threshold: {fraud_model.threshold}")
    logger.info("=" * 80)
    
except Exception as e:
    logger.error(f"❌ FAILED TO LOAD MODEL: {e}")
    fraud_model = None


# ========== API ROUTES ==========

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": fraud_model is not None
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy" if fraud_model else "unhealthy",
        "model_loaded": fraud_model is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not fraud_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    return jsonify({
        "model_type": "RandomForestClassifier",
        "training_method": "SMOTE-balanced",
        "threshold": fraud_model.threshold,
        "expected_features": fraud_model.n_features,
        "status": "ready"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for single transaction"""
    if not fraud_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Make prediction
        result = fraud_model.predict_single(data)
        
        logger.info(f"✅ Prediction: {result['transaction_id']} - "
                   f"Fraud: {result['is_fraud']}, Prob: {result['fraud_probability']}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict fraud for multiple transactions"""
    if not fraud_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({"error": "No transactions provided"}), 400
        
        transactions = data['transactions']
        threshold = data.get('threshold')
        
        # Update threshold if provided
        if threshold is not None:
            original_threshold = fraud_model.threshold
            fraud_model.set_threshold(threshold)
        
        # Make predictions
        result = fraud_model.predict_batch(transactions)
        
        # Restore original threshold
        if threshold is not None:
            fraud_model.set_threshold(original_threshold)
        
        logger.info(f"✅ Batch prediction: {len(transactions)} transactions, "
                   f"{result['statistics']['fraud_count']} frauds detected")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/threshold', methods=['POST'])
def update_threshold():
    """Update prediction threshold"""
    if not fraud_model:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        
        if threshold is None:
            return jsonify({"error": "Threshold value required"}), 400
        
        if not fraud_model.set_threshold(threshold):
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
        
        logger.info(f"🎯 Threshold updated to {threshold}")
        
        return jsonify({
            "message": "Threshold updated successfully",
            "new_threshold": fraud_model.threshold,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Threshold update error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/sample', methods=['GET'])
def sample():
    """Get sample transaction"""
    return jsonify({
        "uuid": "sample_001",
        "amount": 120000.0,
        "customer_age": 51.0,
        "browser": "Chrome_Some(66)",
        "channel": "channel_B",
        "date": "2024-01-15T10:30:00Z",
        "loginTime": "2024-01-15T10:29:00Z",
        "txn_timestamp": "2024-01-15T10:30:15Z",
        "os": "Windows_Some(10)",
        "paymentType": "F",
        "region": "ZA_GP"
    })


# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ========== RUN APP ==========

if __name__ == '__main__':
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING FLASK FRAUD DETECTION API")
    logger.info("=" * 80)
    logger.info("📍 Server: http://localhost:8000")
    logger.info("📋 Endpoints:")
    logger.info("   GET  /              - API info")
    logger.info("   GET  /health        - Health check")
    logger.info("   GET  /model/info    - Model details")
    logger.info("   POST /predict       - Single prediction")
    logger.info("   POST /predict/batch - Batch prediction")
    logger.info("   POST /threshold     - Update threshold")
    logger.info("   GET  /sample        - Sample transaction")
    logger.info("=" * 80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )