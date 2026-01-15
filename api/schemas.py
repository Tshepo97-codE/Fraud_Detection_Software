
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class TransactionCreate(BaseModel):
    uuid: str
    amount: float
    customer_age: Optional[float] = None
    browser: Optional[str] = None
    channel: Optional[str] = None
    ip: Optional[str] = None
    date: Optional[str] = None
    loginTime: Optional[str] = None
    txn_timestamp: Optional[str] = None

class TransactionBatch(BaseModel):
    transactions: List[TransactionCreate]
    threshold: Optional[float] = Field(0.5, ge=0, le=1)

class PredictionResult(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float
    top_features: Dict[str, float]
    risk_level: str

class BatchPredictionResult(BaseModel):
    statistics: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    batch_id: str

class ModelInfo(BaseModel):
    model_type: str
    n_features: Any
    n_estimators: Any
    threshold: float
    feature_count: int
    timestamp: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0, le=1)
