import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

# Load Census Data
data_path = "data/census.csv"
df = pd.read_csv(data_path)

categorical_features = [
    "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
]

# Preprocess the data
X, y, encoder, lb = process_data(df, categorical_features, label="salary", training=True)

# Test if the model trains correctly
def test_train_model():
    model = train_model(X, y)
    print("test_train_model PASSED")
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    assert hasattr(model, "predict"), "Model does not have a predict function"
    

# Test inference function
def test_inference():
    model = train_model(X, y)
    preds = inference(model, X)
    print("test_inference PASSED")
    assert isinstance(preds, np.ndarray), "Inference did not return a numpy array"
    assert preds.shape[0] == X.shape[0], "Mismatch in prediction array shape"
    

# Test compute_model_metrics function
def test_compute_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    print("test_compute_metrics PASSED")
    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(fbeta, float), "F1-score is not a float"
    assert 0 <= precision <= 1, "Precision out of range"
    assert 0 <= recall <= 1, "Recall out of range"
    assert 0 <= fbeta <= 1, "F1-score out of range"
    


if __name__ == "__main__":
    pytest.main()
