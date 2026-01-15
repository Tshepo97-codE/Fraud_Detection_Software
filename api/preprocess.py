import pandas as pd
import numpy as np
import json
import os

class FraudPreprocessor:
    def __init__(self):
        try:
            # Go up one level from api/ to find models/
            config_path = "../models/preprocessing_info.json"
            features_path = "../models/feature_names.json"
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            with open(features_path, 'r') as f:
                self.required_features = json.load(f)
            
            print(f"✅ Preprocessor loaded with {len(self.required_features)} features")
        except Exception as e:
            print(f"❌ Error loading preprocessor: {e}")
            self.config = {}
            self.required_features = []
    
    def preprocess(self, input_data):
        try:
            # Convert to DataFrame
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data.copy()
            
            # Handle date
            for col in self.config.get('datetime_columns', []):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df = df.drop(columns=[col])
            
            # Fill missing
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("unknown")
                else:
                    df[col] = df[col].fillna(0)
            
            # Get dummies
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            
            # Ensure all features
            for feature in self.required_features:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Keep only needed columns
            df = df[self.required_features]
            
            return df
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            raise ValueError(f"Preprocessing failed: {e}")

# Create instance
preprocessor = FraudPreprocessor()